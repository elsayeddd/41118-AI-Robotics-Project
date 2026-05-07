"""
05_pick_and_place.py - Baseline IK pick-and-place using detected world coords.

Pipeline:
1) Build scene
2) Capture camera image + detect coloured cubes
3) Pick selected target colour
4) IK move: pre-grasp -> grasp -> close -> lift -> place -> open
"""

import time
import numpy as np
import pybullet as p
import cv2
from PIL import Image

from common_scene import build_tabletop_scene, setup_world, TARGET_ZONE

# Toggle GUI for visual tests
GUI = True

# Choose target cube colour: "red", "green", "blue"
TARGET_COLOUR = "green"

# Simulation/control constants
SIM_HZ = 240
EE_LINK_INDEX = 11
FINGER_JOINTS = (9, 10)
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]

# Optional calibration offsets (meters) if pickup is consistently shifted
X_BIAS = 0.0
Y_BIAS = 0.0


def step(seconds: float):
    """Step simulation for a duration."""
    steps = int(seconds * SIM_HZ)
    for _ in range(steps):
        p.stepSimulation()
        if GUI:
            time.sleep(1.0 / SIM_HZ)


def set_gripper(robot_id: int, opening: float = 0.08, force: float = 150):
    """
    Set gripper opening width (meters).
    opening is total width, each finger is half.
    """
    finger = opening / 2.0
    for j in FINGER_JOINTS:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=finger,
            force=force,
        )


def move_ee_to(robot_id: int, pos, orn, duration: float = 1.0):
    """Move end-effector via IK to target pose."""
    joint_targets = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=EE_LINK_INDEX,
        targetPosition=pos,
        targetOrientation=orn,
        maxNumIterations=200,
        residualThreshold=1e-4,
    )

    for i, j in enumerate(ARM_JOINTS):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_targets[i],
            force=200,
        )
    step(duration)


def capture_scene(width=640, height=480, fov=60.0):
    """Capture top-down image and save camera params."""
    cam_target = [0.5, 0.0, 0.625]
    cam_eye = [0.5, 0.0, 1.6]
    cam_up = [1, 0, 0]

    view = p.computeViewMatrix(cam_eye, cam_target, cam_up)
    proj = p.computeProjectionMatrixFOV(
        fov=fov, aspect=width / height, nearVal=0.1, farVal=3.0
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


def detect_world_coords():
    """Detect red/green/blue cubes and convert pixel coords to world coords."""
    img_bgr = cv2.imread("scene_view.png")
    if img_bgr is None:
        raise RuntimeError("scene_view.png invalid/missing")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Shadow-tolerant thresholds
    ranges = {
    "red": [
        (np.array([0, 90, 60]), np.array([12, 255, 255])),
        (np.array([168, 90, 60]), np.array([179, 255, 255])),
    ],
    "green": [(np.array([35, 50, 20]), np.array([90, 255, 255]))],
    "blue": [(np.array([95, 80, 40]), np.array([135, 255, 255]))],
}

    def find_cube(colour_ranges):
        mask = None
        for low, high in colour_ranges:
            m = cv2.inRange(img_hsv, low, high)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Prefer cube-like blob (reject giant target pad / noise)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > 3000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / float(h)
            fill = area / float(w * h)
            if not (0.5 <= aspect <= 1.8):
                continue
            if fill < 0.45:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            score = fill - abs(aspect - 1.0) * 0.2 - area * 1e-5
            candidates.append((score, cx, cy))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, cx, cy = candidates[0]
        return cx, cy

    cam = np.load("camera_params.npz")
    cam_eye = cam["cam_eye"]
    cam_target = cam["cam_target"]
    cam_up = cam["cam_up"]
    fov = float(cam["fov"])
    W = int(cam["width"])
    H = int(cam["height"])

    if not np.allclose(cam_up, [1, 0, 0]):
        raise ValueError("Unsupported camera orientation for this mapping.")

    h = float(cam_eye[2] - cam_target[2])
    view_h = 2 * h * np.tan(np.deg2rad(fov) / 2)
    view_w = view_h * (W / H)
    mpp_x = view_w / W
    mpp_y = view_h / H

    world = {}
    for name, rr in ranges.items():
        px = find_cube(rr)
        if px is None:
            continue

        cx, cy = px
        dx = cx - W / 2
        dy = cy - H / 2

        # Note the minus on Y to match observed image-to-world orientation
        wx = float(cam_target[0]) - dy * mpp_y
        wy = float(cam_target[1]) - dx * mpp_x
        wz = float(cam_target[2]) + 0.02

        world[name] = [wx, wy, wz]

    return world


def main():
    print("RUNNING CLEAN 05 v2")

    setup_world(gui=GUI)
    panda_id, cube_positions = build_tabletop_scene(seed=0)
    step(1.0)

    # Neutral arm pose
    for j, q in enumerate([0, -0.4, 0, -2.2, 0, 2.0, 0.8]):
        p.resetJointState(panda_id, j, q)
    set_gripper(panda_id, opening=0.08)
    step(0.5)

    # Perception
    capture_scene()
    detections = detect_world_coords()
    print("Detected world coords:", detections)

    if TARGET_COLOUR not in detections:
        p.disconnect()
        raise RuntimeError(f"Target colour '{TARGET_COLOUR}' not detected.")

    target = detections[TARGET_COLOUR]
    target[0] += X_BIAS
    target[1] += Y_BIAS

    place = [float(TARGET_ZONE[0]), float(TARGET_ZONE[1]), 0.70]

    # End-effector orientation: point down
    down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

    # Waypoints
    pre_grasp = [target[0], target[1], 0.78]
    grasp = [target[0], target[1], 0.65]
    lift = [target[0], target[1], 0.82]
    pre_place = [place[0], place[1], 0.82]
    place_down = [place[0], place[1], 0.70]

    print(f"Picking {TARGET_COLOUR} at {target}")

    move_ee_to(panda_id, pre_grasp, down_orn, duration=1.0)
    move_ee_to(panda_id, grasp, down_orn, duration=1.0)

    set_gripper(panda_id, opening=0.00, force=140)  # close
    step(0.8)

    move_ee_to(panda_id, lift, down_orn, duration=1.0)

    # Simple success signal
    cube_id = cube_positions[TARGET_COLOUR][0]
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    print(f"Pick success check (z={cube_pos[2]:.3f}):", cube_pos[2] > 0.70)

    move_ee_to(panda_id, pre_place, down_orn, duration=1.2)
    move_ee_to(panda_id, place_down, down_orn, duration=1.0)

    set_gripper(panda_id, opening=0.08, force=120)  # release
    step(0.8)

    move_ee_to(panda_id, pre_place, down_orn, duration=1.0)

    print("Done. Pick-and-place sequence complete.")
    step(2.0)
    p.disconnect()


if __name__ == "__main__":
    main()