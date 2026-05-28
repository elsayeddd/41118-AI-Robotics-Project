"""Run YOLO-guided cube sorting on a cluttered tabletop scene.

The robot captures one image, detects all cube instances with YOLO, optionally
uses a trained RL policy to choose the pick order, and moves every detected cube
to the target zone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from PIL import Image
from ultralytics import YOLO

from common_scene import (
    CUBE_Z,
    TARGET_ZONE,
    build_tabletop_scene,
    camera_params,
    capture_camera,
    load_camera_params,
    pixel_to_world,
    save_camera_params,
    setup_world,
)


SIM_HZ = 240
EE_LINK_INDEX = 11
FINGER_JOINTS = (9, 10)
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_YOLO_MODEL = Path("models") / "yolo_cube.pt"
DEFAULT_RL_POLICY = Path("models") / "sorting_policy.zip"
TARGET_ZONE_Y_MIN = 0.25
NEUTRAL_ARM_POSE = [0.0, -0.45, 0.0, -2.25, 0.0, 2.05, 0.8]
SAFE_TRAVEL_Z = 0.94
PRE_GRASP_Z = 0.82
GRASP_Z = 0.655
LIFT_Z = 0.90
PRE_PLACE_Z = 0.84
PLACE_Z = 0.675


def step(seconds: float, gui: bool):
    for _ in range(int(seconds * SIM_HZ)):
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / SIM_HZ)


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def set_gripper(robot_id: int, opening: float = 0.08, force: float = 150):
    finger = opening / 2.0
    for joint in FINGER_JOINTS:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=finger,
            force=force,
        )


def set_gripper_smooth(robot_id: int, opening: float, gui: bool, duration: float = 0.5, force: float = 150):
    start = np.array([p.getJointState(robot_id, joint)[0] for joint in FINGER_JOINTS], dtype=float)
    target = np.array([opening / 2.0, opening / 2.0], dtype=float)
    steps = max(1, int(duration * SIM_HZ))
    for i in range(steps):
        alpha = smoothstep((i + 1) / steps)
        values = start + (target - start) * alpha
        for joint, value in zip(FINGER_JOINTS, values):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(value),
                force=force,
            )
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / SIM_HZ)


def move_arm_joints(robot_id: int, target_pose, duration: float, gui: bool, force: float = 180):
    start = np.array([p.getJointState(robot_id, joint)[0] for joint in ARM_JOINTS], dtype=float)
    target = np.array(target_pose, dtype=float)
    steps = max(1, int(duration * SIM_HZ))
    for i in range(steps):
        alpha = smoothstep((i + 1) / steps)
        pose = start + (target - start) * alpha
        for joint, q in zip(ARM_JOINTS, pose):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(q),
                force=force,
                positionGain=0.05,
                velocityGain=0.8,
            )
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / SIM_HZ)


def park_arm(robot_id: int, gui: bool):
    move_arm_joints(robot_id, NEUTRAL_ARM_POSE, duration=0.9, gui=gui)
    set_gripper_smooth(robot_id, opening=0.08, gui=gui, duration=0.25, force=120)
    step(0.2, gui)


def move_ee_to(robot_id: int, pos, orn, duration: float, gui: bool):
    start_pos = np.array(p.getLinkState(robot_id, EE_LINK_INDEX)[4], dtype=float)
    end_pos = np.array(pos, dtype=float)
    steps = max(1, int(duration * SIM_HZ))
    for i in range(steps):
        alpha = smoothstep((i + 1) / steps)
        interp = start_pos + (end_pos - start_pos) * alpha
        joint_targets = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=EE_LINK_INDEX,
            targetPosition=interp.tolist(),
            targetOrientation=orn,
            maxNumIterations=80,
            residualThreshold=1e-4,
        )
        for joint_index, joint in enumerate(ARM_JOINTS):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_targets[joint_index],
                force=220,
                positionGain=0.045,
                velocityGain=0.75,
            )
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / SIM_HZ)


def capture_scene(width: int, height: int, gui: bool):
    cam_cfg = camera_params(width=width, height=height, fov=60.0)
    # The synthetic YOLO dataset is rendered with ER_TINY_RENDERER, so use the
    # same camera renderer at runtime to reduce sim-to-sim appearance shift.
    renderer = p.ER_TINY_RENDERER
    rgb, _, _ = capture_camera(cam_cfg, renderer=renderer)
    Image.fromarray(rgb, mode="RGB").save("scene_view.png")
    save_camera_params(cam_cfg, "camera_params.npz")
    step(0.1, gui)
    return cam_cfg


def detect_cubes(model_path: Path, image_path: Path, cam_cfg, conf: float):
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {model_path}. Run 07_train_yolo.py first or copy best.pt there."
        )

    model = YOLO(str(model_path))
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().numpy()]
        score = float(box.conf[0].detach().cpu().item())
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append(
            {
                "id": index,
                "confidence": score,
                "xyxy": [x1, y1, x2, y2],
                "center": [cx, cy],
                "world": pixel_to_world(cx, cy, cam_cfg, z=CUBE_Z),
                "sorted": False,
            }
        )
    return detections


def detect_cubes_with_retries(model_path: Path, image_path: Path, cam_cfg, conf: float):
    """Try progressively lower confidence thresholds before declaring a miss."""

    thresholds = [conf, min(conf, 0.15), min(conf, 0.10)]
    best = []
    for threshold in dict.fromkeys(thresholds):
        detections = detect_cubes(model_path, image_path, cam_cfg, threshold)
        if len(detections) > len(best):
            best = detections
        if len(detections) >= 3:
            return detections
    return best


def source_area_detections(detections):
    """Ignore cubes already in the target/sorting zone."""

    return [
        det
        for det in detections
        if det["world"][1] < TARGET_ZONE_Y_MIN
    ]


def draw_detections(image_path: Path, detections, output_path=Path("detection_result.png")):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        cx, cy = [int(round(v)) for v in det["center"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"cube {det['confidence']:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
    cv2.imwrite(str(output_path), image)


def place_targets(count: int):
    if count <= 0:
        return []
    spacing = 0.075
    offsets = (np.arange(count) - (count - 1) / 2.0) * spacing
    return [[float(TARGET_ZONE[0] + dx), float(TARGET_ZONE[1]), PLACE_Z] for dx in offsets]


def build_rl_observation(detections, chosen_ids):
    max_objects = 10
    features_per_object = 6
    obs = np.zeros((max_objects, features_per_object), dtype=np.float32)
    for i, det in enumerate(detections[:max_objects]):
        x, y, _ = det["world"]
        distance = float(np.linalg.norm(np.array([x - TARGET_ZONE[0], y - TARGET_ZONE[1]])))
        obs[i, 0] = np.interp(x, [0.30, 0.70], [-1.0, 1.0])
        obs[i, 1] = np.interp(y, [-0.25, 0.25], [-1.0, 1.0])
        obs[i, 2] = np.interp(det["confidence"], [0.0, 1.0], [-1.0, 1.0])
        obs[i, 3] = np.interp(distance, [0.0, 0.75], [1.0, -1.0])
        obs[i, 4] = 1.0
        obs[i, 5] = 1.0 if det["id"] in chosen_ids else -1.0
    stage_value = np.array([1.0], dtype=np.float32)
    return np.concatenate([obs.reshape(-1), stage_value]).astype(np.float32)


def order_detections_with_rl(detections, policy_path: Path):
    if not policy_path.exists():
        return sorted(detections, key=lambda det: det["world"][1], reverse=True)

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 not installed; using nearest-to-zone order.")
        return sorted(detections, key=lambda det: det["world"][1], reverse=True)

    policy = PPO.load(str(policy_path))
    remaining = {det["id"]: det for det in detections}
    chosen_ids = set()
    ordered = []
    while remaining:
        obs = build_rl_observation(detections, chosen_ids)
        action, _ = policy.predict(obs, deterministic=True)
        action = int(action)
        if action in remaining:
            chosen = remaining.pop(action)
        else:
            chosen = max(remaining.values(), key=lambda det: det["confidence"])
            remaining.pop(chosen["id"])
        chosen_ids.add(chosen["id"])
        ordered.append(chosen)
    return ordered


def pick_and_place_one(robot_id: int, detection, place, gui: bool):
    target = detection["world"]
    down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    safe_current = list(p.getLinkState(robot_id, EE_LINK_INDEX)[4])
    safe_current[2] = SAFE_TRAVEL_Z
    safe_grasp = [target[0], target[1], SAFE_TRAVEL_Z]
    pre_grasp = [target[0], target[1], PRE_GRASP_Z]
    grasp = [target[0], target[1], GRASP_Z]
    lift = [target[0], target[1], LIFT_Z]
    safe_place = [place[0], place[1], SAFE_TRAVEL_Z]
    pre_place = [place[0], place[1], PRE_PLACE_Z]
    place_down = [place[0], place[1], PLACE_Z]

    print(
        f"Picking cube_{detection['id']:02d} conf={detection['confidence']:.2f} "
        f"from ({target[0]:.3f}, {target[1]:.3f}) -> ({place[0]:.3f}, {place[1]:.3f})"
    )
    move_ee_to(robot_id, safe_current, down_orn, 0.7, gui)
    move_ee_to(robot_id, safe_grasp, down_orn, 1.0, gui)
    move_ee_to(robot_id, pre_grasp, down_orn, 0.8, gui)
    move_ee_to(robot_id, grasp, down_orn, 0.9, gui)
    set_gripper_smooth(robot_id, opening=0.00, gui=gui, duration=0.55, force=160)
    step(0.25, gui)
    move_ee_to(robot_id, lift, down_orn, 0.9, gui)
    move_ee_to(robot_id, safe_place, down_orn, 1.2, gui)
    move_ee_to(robot_id, pre_place, down_orn, 0.8, gui)
    move_ee_to(robot_id, place_down, down_orn, 0.75, gui)
    set_gripper_smooth(robot_id, opening=0.08, gui=gui, duration=0.45, force=120)
    step(0.45, gui)
    move_ee_to(robot_id, pre_place, down_orn, 0.7, gui)
    move_ee_to(robot_id, safe_place, down_orn, 0.8, gui)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--rl-policy", type=Path, default=DEFAULT_RL_POLICY)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--direct", action="store_true", help="Run without GUI.")
    return parser.parse_args()


def main():
    args = parse_args()
    gui = not args.direct
    setup_world(gui=gui)
    panda_id, cube_positions, _ = build_tabletop_scene(
        seed=args.seed,
        num_distractors=args.distractors,
        randomize_cube_colours=True,
        return_distractors=True,
    )
    step(1.0, gui)

    expected_cubes = len(cube_positions)
    targets = place_targets(expected_cubes)
    moved = 0
    missed_observations = 0

    while moved < expected_cubes and missed_observations < 2:
        # Park before every perception pass so the robot handles self-occlusion.
        park_arm(panda_id, gui)
        cam_cfg = capture_scene(args.width, args.height, gui)
        cam_cfg = load_camera_params("camera_params.npz")

        detections = detect_cubes_with_retries(args.model, Path("scene_view.png"), cam_cfg, args.conf)
        remaining = source_area_detections(detections)
        draw_detections(Path("scene_view.png"), remaining)
        print(
            f"Observation {moved + 1}: detected {len(detections)} candidates, "
            f"{len(remaining)} still outside target zone."
        )

        if not remaining:
            missed_observations += 1
            continue

        missed_observations = 0
        ordered = order_detections_with_rl(remaining, args.rl_policy)
        pick_and_place_one(panda_id, ordered[0], targets[moved], gui)
        moved += 1

    if moved < expected_cubes:
        print(f"Stopped after moving {moved}/{expected_cubes} cubes; no more source-area cubes were detected.")
    else:
        print(f"Moved all expected cubes: {moved}/{expected_cubes}.")

    print("Done. Cube sorting sequence complete.")
    step(2.0, gui)
    p.disconnect()


if __name__ == "__main__":
    main()
