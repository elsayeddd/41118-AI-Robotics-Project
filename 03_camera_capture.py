"""
03_camera_capture.py - Add an overhead camera to the scene and capture an image.

Goal: Render what the robot 'sees' from a camera mounted above the table.
This image is what the vision component will process in the next script.

Run with:
    python 03_camera_capture.py

Output:
    scene_view.png  - what an overhead camera sees
    
Also saves a copy with the GUI camera angle for reference:
    debug_view.png
"""

import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import time

GUI = True

if GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- Build the scene (same as 02_tabletop_scene.py) ---
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0], useFixedBase=True)
panda_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0.62],
    useFixedBase=True,
)


def spawn_cube(position, colour, size=0.04):
    half = size / 2
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=colour)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
    return p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


np.random.seed(0)
colours = {
    "red":   [1, 0, 0, 1],
    "green": [0, 0.7, 0, 1],
    "blue":  [0, 0, 1, 1],
}

cube_positions = {}
for name, colour in colours.items():
    x = np.random.uniform(0.35, 0.65)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.65
    cube_id = spawn_cube([x, y, z], colour)
    cube_positions[name] = (cube_id, [x, y, z])

# Target zone marker
target_pos = [0.5, 0.35, 0.625]
target_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.001], rgbaColor=[0.2, 1, 0.2, 0.5],
)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=target_pos)

# Let physics settle
for _ in range(240):
    p.stepSimulation()

# --- The camera setup. THIS IS THE NEW PART. ---
# We position a virtual camera looking straight down at the table.
# This is what the vision pipeline will process.

CAM_WIDTH, CAM_HEIGHT = 640, 480

# Camera mounted 1m above the centre of the table, looking down
cam_target = [0.5, 0, 0.625]      # point camera at centre of table
cam_eye = [0.5, 0, 1.6]            # 1m above the target
cam_up = [1, 0, 0]                 # 'up' direction in image (rotates view)

view_matrix = p.computeViewMatrix(
    cameraEyePosition=cam_eye,
    cameraTargetPosition=cam_target,
    cameraUpVector=cam_up,
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=CAM_WIDTH / CAM_HEIGHT,
    nearVal=0.1,
    farVal=3.0,
)

# Capture the image
_, _, rgb, depth, seg = p.getCameraImage(
    width=CAM_WIDTH,
    height=CAM_HEIGHT,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_BULLET_HARDWARE_OPENGL,
)

# Convert to numpy and save
rgb_array = np.array(rgb, dtype=np.uint8)
# Some PyBullet versions return RGBA, drop alpha if present
if rgb_array.shape[-1] == 4:
    rgb_array = rgb_array[:, :, :3]

Image.fromarray(rgb_array).save("scene_view.png")
print(f"Saved scene_view.png  ({CAM_WIDTH}x{CAM_HEIGHT})")
print("This is what the vision system will see.")

# Also save the camera-to-world info so the next script can convert
# pixel coordinates back into world coordinates for the arm to grasp
np.savez(
    "camera_params.npz",
    cam_eye=cam_eye,
    cam_target=cam_target,
    fov=60,
    width=CAM_WIDTH,
    height=CAM_HEIGHT,
)

# Print the ground-truth cube positions so we can sanity-check vision later
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
