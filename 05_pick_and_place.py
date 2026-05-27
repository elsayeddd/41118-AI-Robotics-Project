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


def step(seconds: float, gui: bool):
    for _ in range(int(seconds * SIM_HZ)):
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / SIM_HZ)


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


def park_arm(robot_id: int, gui: bool):
    for joint, q in enumerate([0, -0.4, 0, -2.2, 0, 2.0, 0.8]):
        p.resetJointState(robot_id, joint, q)
    set_gripper(robot_id, opening=0.08)
    step(0.5, gui)


def move_ee_to(robot_id: int, pos, orn, duration: float, gui: bool):
    joint_targets = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=EE_LINK_INDEX,
        targetPosition=pos,
        targetOrientation=orn,
        maxNumIterations=200,
        residualThreshold=1e-4,
    )
    for i, joint in enumerate(ARM_JOINTS):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_targets[i],
            force=220,
        )
    step(duration, gui)


def capture_scene(width: int, height: int, gui: bool):
    cam_cfg = camera_params(width=width, height=height, fov=60.0)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
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
    spacing = 0.055
    offsets = (np.arange(count) - (count - 1) / 2.0) * spacing
    return [[float(TARGET_ZONE[0] + dx), float(TARGET_ZONE[1]), 0.70] for dx in offsets]


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
    pre_grasp = [target[0], target[1], 0.79]
    grasp = [target[0], target[1], 0.65]
    lift = [target[0], target[1], 0.84]
    pre_place = [place[0], place[1], 0.84]
    place_down = [place[0], place[1], 0.70]

    print(
        f"Picking cube_{detection['id']:02d} conf={detection['confidence']:.2f} "
        f"from ({target[0]:.3f}, {target[1]:.3f}) -> ({place[0]:.3f}, {place[1]:.3f})"
    )
    move_ee_to(robot_id, pre_grasp, down_orn, 1.0, gui)
    move_ee_to(robot_id, grasp, down_orn, 1.0, gui)
    set_gripper(robot_id, opening=0.00, force=150)
    step(0.8, gui)
    move_ee_to(robot_id, lift, down_orn, 1.0, gui)
    move_ee_to(robot_id, pre_place, down_orn, 1.2, gui)
    move_ee_to(robot_id, place_down, down_orn, 1.0, gui)
    set_gripper(robot_id, opening=0.08, force=120)
    step(0.8, gui)
    move_ee_to(robot_id, pre_place, down_orn, 0.8, gui)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--rl-policy", type=Path, default=DEFAULT_RL_POLICY)
    parser.add_argument("--conf", type=float, default=0.35)
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
    panda_id, _, _ = build_tabletop_scene(
        seed=args.seed,
        num_distractors=args.distractors,
        randomize_cube_colours=True,
        return_distractors=True,
    )
    step(1.0, gui)

    # Park before perception so the robot handles the known self-occlusion problem.
    park_arm(panda_id, gui)
    cam_cfg = capture_scene(args.width, args.height, gui)
    cam_cfg = load_camera_params("camera_params.npz")

    detections = detect_cubes(args.model, Path("scene_view.png"), cam_cfg, args.conf)
    draw_detections(Path("scene_view.png"), detections)
    print(f"Detected {len(detections)} cube candidates.")
    if not detections:
        p.disconnect()
        raise RuntimeError("No cubes detected by YOLO.")

    ordered = order_detections_with_rl(detections, args.rl_policy)
    targets = place_targets(len(ordered))
    for detection, place in zip(ordered, targets):
        pick_and_place_one(panda_id, detection, place, gui)

    print("Done. All YOLO-detected cubes were moved to the sorting zone.")
    step(2.0, gui)
    p.disconnect()


if __name__ == "__main__":
    main()
