"""05_pick_and_place.py - Baseline pick/place with bias calibration and success checks."""

import time
import numpy as np
import pybullet as p

from common_scene import TARGET_ZONE, build_tabletop_scene, camera_params, setup_world
from PIL import Image
import cv2

GUI = False
TARGET_COLOUR = "blue"
EE_LINK_INDEX = 11
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
FINGER_JOINTS = [9, 10]
SIM_HZ = 240
X_BIAS = 0.0
Y_BIAS = 0.0


def step(seconds: float):
    for _ in range(int(seconds * SIM_HZ)):
        p.stepSimulation()
        if GUI:
            time.sleep(1.0 / SIM_HZ)


def set_gripper(robot_id, opening=0.08, force=120):
    finger_pos = opening / 2.0
    for j in FINGER_JOINTS:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=finger_pos, force=force)


def move_ee_to(robot_id, pos, orn, duration=1.0):
    q = p.calculateInverseKinematics(robot_id, EE_LINK_INDEX, pos, orn, maxNumIterations=200)
    for i, j in enumerate(ARM_JOINTS):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=q[i], force=200)
    step(duration)


def capture_scene():
    cfg = camera_params()
    view = p.computeViewMatrix(cfg["cam_eye"], cfg["cam_target"], cfg["cam_up"])
    proj = p.computeProjectionMatrixFOV(cfg["fov"], cfg["width"] / cfg["height"], 0.1, 3.0)
    _, _, rgb, _, _ = p.getCameraImage(cfg["width"], cfg["height"], view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    arr = np.asarray(rgb, dtype=np.uint8).reshape((cfg["height"], cfg["width"], 4))[:, :, :3]
    Image.fromarray(arr, mode="RGB").save("scene_view.png")
    np.savez("camera_params.npz", cam_eye=cfg["cam_eye"], cam_target=cfg["cam_target"], cam_up=cfg["cam_up"], fov=cfg["fov"], width=cfg["width"], height=cfg["height"])


def detect_world_coords():
    img_bgr = cv2.imread("scene_view.png")
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        "red": [(np.array([0, 90, 60]), np.array([12, 255, 255])), (np.array([168, 90, 60]), np.array([179, 255, 255]))],
        "green": [(np.array([35, 50, 20]), np.array([90, 255, 255]))],
        "blue": [(np.array([95, 80, 40]), np.array([135, 255, 255]))],
    }

    def find_cube(rr):
        mask = None
        for low, high in rr:
            m = cv2.inRange(img_hsv, low, high)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50 or area > 2500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            asp = w / float(h)
            fill = area / float(w * h)
            if not (0.6 <= asp <= 1.6) or fill < 0.55:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cand.append((fill, int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        if not cand:
            return None
        cand.sort(reverse=True)
        _, cx, cy = cand[0]
        return cx, cy

    cam = np.load("camera_params.npz")
    W, H, fov = int(cam["width"]), int(cam["height"]), float(cam["fov"])
    eye, target = cam["cam_eye"], cam["cam_target"]
    h = float(eye[2] - target[2])
    view_h = 2 * h * np.tan(np.deg2rad(fov) / 2)
    view_w = view_h * (W / H)

    out = {}
    for name, rr in ranges.items():
        px = find_cube(rr)
        if px is None:
            continue
        cx, cy = px
        wx = float(target[0]) - (cy - H / 2) * (view_h / H)
        wy = float(target[1]) + (cx - W / 2) * (view_w / W)
        out[name] = [wx, wy, float(target[2]) + 0.02]
    return out


setup_world(gui=GUI)
panda_id, cube_positions = build_tabletop_scene(seed=0)

for j, q in enumerate([0, -0.4, 0, -2.2, 0, 2.0, 0.8]):
    p.resetJointState(panda_id, j, q)
set_gripper(panda_id, opening=0.08)
step(1.0)

capture_scene()
detections = detect_world_coords()
print("Detected world coords:", detections)
if TARGET_COLOUR not in detections:
    raise RuntimeError(f"Target {TARGET_COLOUR} not detected")

target = detections[TARGET_COLOUR]
target[0] += X_BIAS
target[1] += Y_BIAS

down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
pre = [target[0], target[1], 0.78]
grasp = [target[0], target[1], 0.675]
lift = [target[0], target[1], 0.82]
place_pre = [float(TARGET_ZONE[0]), float(TARGET_ZONE[1]), 0.82]
place = [float(TARGET_ZONE[0]), float(TARGET_ZONE[1]), 0.70]

move_ee_to(panda_id, pre, down_orn, 1.0)
move_ee_to(panda_id, [target[0], target[1], pre[2]], down_orn, 0.6)
move_ee_to(panda_id, grasp, down_orn, 1.0)
set_gripper(panda_id, opening=0.0, force=140)
step(0.8)
move_ee_to(panda_id, lift, down_orn, 1.0)

cube_id = cube_positions[TARGET_COLOUR][0]
pos, _ = p.getBasePositionAndOrientation(cube_id)
print(f"Pick success check (z={pos[2]:.3f}):", pos[2] > 0.70)

move_ee_to(panda_id, place_pre, down_orn, 1.2)
move_ee_to(panda_id, place, down_orn, 1.0)
set_gripper(panda_id, opening=0.08)
step(0.8)
move_ee_to(panda_id, place_pre, down_orn, 1.0)

print("Done. Pick-and-place sequence complete.")
step(1.0)
p.disconnect()
