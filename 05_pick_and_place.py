"""
05_pick_and_place.py - AI-assisted multi-cube pick-and-place.

Pipeline:
1) Build scene
2) Capture camera image
3) Detect red, green, and blue cubes with the trained neural model
4) Pick and place all detected cubes into the target zone in one run
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from PIL import Image

from ai_perception import MODEL_PATH, load_camera_params, load_model, pixels_to_world, predict_pixels
from common_scene import TARGET_ZONE, build_tabletop_scene, camera_params, setup_world


GUI = True
SIM_HZ = 240
EE_LINK_INDEX = 11
FINGER_JOINTS = (9, 10)
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
MOVE_ORDER = ("red", "green", "blue")
PLACE_OFFSETS = {
    "red": [-0.055, 0.0, 0.0],
    "green": [0.0, 0.0, 0.0],
    "blue": [0.055, 0.0, 0.0],
}

X_BIAS = 0.0
Y_BIAS = 0.0


COLOUR_RANGES = {
    "red": [
        (np.array([0, 90, 60]), np.array([12, 255, 255])),
        (np.array([168, 90, 60]), np.array([179, 255, 255])),
    ],
    "green": [(np.array([35, 50, 20]), np.array([90, 255, 255]))],
    "blue": [(np.array([95, 80, 40]), np.array([135, 255, 255]))],
}
MARKER_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}


def step(seconds: float):
    steps = int(seconds * SIM_HZ)
    for _ in range(steps):
        p.stepSimulation()
        if GUI:
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


def move_ee_to(robot_id: int, pos, orn, duration: float = 1.0):
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
            force=200,
        )
    step(duration)


def park_arm(robot_id: int):
    for joint, q in enumerate([0, -0.4, 0, -2.2, 0, 2.0, 0.8]):
        p.resetJointState(robot_id, joint, q)
    set_gripper(robot_id, opening=0.08)
    step(0.5)


def capture_scene(width=640, height=480, fov=60.0):
    cam_cfg = camera_params(width=width, height=height, fov=fov)
    cam_target = cam_cfg["cam_target"]
    cam_eye = cam_cfg["cam_eye"]
    cam_up = cam_cfg["cam_up"]

    view = p.computeViewMatrix(cam_eye, cam_target, cam_up)
    proj = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=0.1,
        farVal=3.0,
    )

    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    arr = np.asarray(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    Image.fromarray(arr, mode="RGB").save("scene_view.png")
    np.savez(
        "camera_params.npz",
        cam_eye=cam_eye,
        cam_target=cam_target,
        cam_up=cam_up,
        fov=fov,
        width=width,
        height=height,
    )


def find_cube_hsv(img_hsv, colour_ranges, min_area=50, max_area=2500, min_fill=0.55):
    mask = None
    for low, high in colour_ranges:
        m = cv2.inRange(img_hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = float(w * h)
        if rect_area <= 0:
            continue
        fill = area / rect_area
        aspect = w / float(h)
        if fill < min_fill or aspect < 0.6 or aspect > 1.6:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        score = fill - abs(aspect - 1.0) * 0.2 - (area / max_area) * 0.05
        candidates.append((score, cx, cy))

    if not candidates:
        return None
    score, cx, cy = max(candidates, key=lambda item: item[0])
    return cx, cy, float(score)


def detect_with_hsv(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    detections = {}
    for name, ranges in COLOUR_RANGES.items():
        result = find_cube_hsv(img_hsv, ranges)
        if result is not None:
            detections[name] = result
    return detections


def draw_detections(img_bgr, detections, output_path="detection_result.png"):
    display = img_bgr.copy()
    for name, values in detections.items():
        cx, cy = values[:2]
        marker = MARKER_BGR[name]
        cv2.circle(display, (int(cx), int(cy)), 12, marker, 2)
        cv2.putText(display, name, (int(cx) + 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker, 2)
    cv2.imwrite(output_path, display)


def detect_world_coords():
    img_bgr = cv2.imread("scene_view.png")
    if img_bgr is None:
        raise RuntimeError("scene_view.png invalid/missing")

    fallback = detect_with_hsv(img_bgr)
    method = "hsv"
    detections = dict(fallback)

    if Path(MODEL_PATH).exists():
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        model = load_model(MODEL_PATH)
        neural = predict_pixels(model, rgb, threshold=0.45)
        detections = dict(neural)
        for name, value in fallback.items():
            detections.setdefault(name, value)
        method = "neural+fallback" if len(detections) != len(neural) else "neural"
    else:
        print(f"Neural model not found at {MODEL_PATH}; using HSV fallback.")

    draw_detections(img_bgr, detections)
    world = pixels_to_world(detections, load_camera_params("camera_params.npz"))
    return world, method


def pick_and_place_one(robot_id: int, colour: str, target, place):
    target = [float(target[0]) + X_BIAS, float(target[1]) + Y_BIAS, float(target[2])]
    down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    pre_grasp = [target[0], target[1], 0.78]
    grasp = [target[0], target[1], 0.65]
    lift = [target[0], target[1], 0.82]
    pre_place = [place[0], place[1], 0.82]
    place_down = [place[0], place[1], 0.70]

    print(f"Picking {colour} at {target} -> placing at {place}")
    move_ee_to(robot_id, pre_grasp, down_orn, duration=1.0)
    move_ee_to(robot_id, grasp, down_orn, duration=1.0)
    set_gripper(robot_id, opening=0.00, force=140)
    step(0.8)
    move_ee_to(robot_id, lift, down_orn, duration=1.0)
    move_ee_to(robot_id, pre_place, down_orn, duration=1.2)
    move_ee_to(robot_id, place_down, down_orn, duration=1.0)
    set_gripper(robot_id, opening=0.08, force=120)
    step(0.8)
    move_ee_to(robot_id, pre_place, down_orn, duration=0.8)


def main():
    print("RUNNING 05 multi-cube AI pick-and-place")
    setup_world(gui=GUI)
    panda_id, cube_positions = build_tabletop_scene(seed=0)
    step(1.0)
    park_arm(panda_id)

    capture_scene()
    detections, method = detect_world_coords()
    print(f"Detection method: {method}")
    print("Detected world coords:", detections)

    missing = [name for name in MOVE_ORDER if name not in detections]
    if missing:
        p.disconnect()
        raise RuntimeError(f"Missing cube detections: {missing}")

    for colour in MOVE_ORDER:
        place = TARGET_ZONE.astype(float) + np.array(PLACE_OFFSETS[colour], dtype=float)
        place = [float(place[0]), float(place[1]), 0.70]
        pick_and_place_one(panda_id, colour, detections[colour], place)

        cube_id = cube_positions[colour][0]
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        distance_to_zone = np.linalg.norm(np.array(cube_pos[:2]) - np.array(place[:2]))
        print(f"{colour} end-zone check: distance={distance_to_zone:.3f} m")

    print("Done. All detected cubes have been moved to the target zone.")
    step(2.0)
    p.disconnect()


if __name__ == "__main__":
    main()
