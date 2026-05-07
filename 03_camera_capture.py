"""
03_camera_capture.py - Add an overhead camera to the scene and capture an image.

Goal: Render what the robot 'sees' from a camera mounted above the table.
This image is what the vision component will process in the next script.

Run with:
    python 03_camera_capture.py

Output:
    scene_view.png  - what an overhead camera sees
"""

import pybullet as p
import numpy as np
from PIL import Image
import time

from common_scene import build_tabletop_scene, camera_params, setup_world

GUI = True

setup_world(gui=GUI)
_, cube_positions = build_tabletop_scene(seed=0)

# Let physics settle
for _ in range(240):
    p.stepSimulation()

cam_cfg = camera_params(width=640, height=480, fov=60.0)
CAM_WIDTH, CAM_HEIGHT = cam_cfg["width"], cam_cfg["height"]
cam_target = cam_cfg["cam_target"]
cam_eye = cam_cfg["cam_eye"]
cam_up = cam_cfg["cam_up"]

view_matrix = p.computeViewMatrix(
    cameraEyePosition=cam_eye,
    cameraTargetPosition=cam_target,
    cameraUpVector=cam_up,
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=cam_cfg["fov"],
    aspect=CAM_WIDTH / CAM_HEIGHT,
    nearVal=0.1,
    farVal=3.0,
)

_, _, rgb, depth, seg = p.getCameraImage(
    width=CAM_WIDTH,
    height=CAM_HEIGHT,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_BULLET_HARDWARE_OPENGL,
)

#OLD FUNCTION - AS BACKUP
#rgb_array = np.array(rgb, dtype=np.uint8)
#if rgb_array.shape[-1] == 4:
#    rgb_array = rgb_array[:, :, :3]

#Image.fromarray(rgb_array).save("scene_view.png")

rgb_array = np.asarray(rgb, dtype=np.uint8)
rgb_array = rgb_array.reshape((CAM_HEIGHT, CAM_WIDTH, 4))  # enforce H x W x RGBA
rgb_array = rgb_array[:, :, :3]  # RGB only
Image.fromarray(rgb_array, mode="RGB").save("scene_view.png")

print(f"Saved scene_view.png  ({CAM_WIDTH}x{CAM_HEIGHT})")

np.savez(
    "camera_params.npz",
    cam_eye=cam_eye,
    cam_target=cam_target,
    fov=cam_cfg["fov"],
    width=CAM_WIDTH,
    height=CAM_HEIGHT,
    cam_up=cam_up,
)

print("\nGround-truth cube positions (for checking vision results):")
for name, (cube_id, expected_pos) in cube_positions.items():
    actual_pos, _ = p.getBasePositionAndOrientation(cube_id)
    print(f"  {name:5s}: world ({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f})")

if GUI:
    print("\nClosing GUI in 5 seconds...")
    for _ in range(5 * 240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

p.disconnect()
print("\nDone. Open scene_view.png to see what the camera captured.")
