"""
01_hello_panda.py - First sanity check.

Goal: Confirm PyBullet works and the Franka Panda arm loads.

Run with:
    python 01_hello_panda.py

Expected output:
    - A GUI window showing a Panda arm on a grey plane (if WSLg works), OR
    - A saved image 'hello_panda.png' if running headless.
    - Console prints info about the arm's 12 joints.

If the GUI window is black or doesn't appear, set GUI = False below to run
headless and save a screenshot instead.
"""

import pybullet as p
import pybullet_data
import time

# Toggle this if WSLg/GUI doesn't work on your setup
GUI = True

# Connect to physics engine
if GUI:
    physics_client = p.connect(p.GUI)
else:
    physics_client = p.connect(p.DIRECT)

# Tell PyBullet where to find its bundled URDFs (plane, panda, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load the ground and the arm
plane_id = p.loadURDF("plane.urdf")
panda_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)

# Print arm info so you know what you're working with
num_joints = p.getNumJoints(panda_id)
print(f"\nFranka Panda has {num_joints} joints:")
for i in range(num_joints):
    info = p.getJointInfo(panda_id, i)
    name = info[1].decode()
    joint_type = info[2]  # 0 = revolute, 4 = fixed
    print(f"  Joint {i:2d}: {name:30s} type={joint_type}")

# Set a nice camera angle
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=50,
    cameraPitch=-30,
    cameraTargetPosition=[0.3, 0, 0.3],
)

if GUI:
    print("\nGUI is open. Drag with mouse to rotate, scroll to zoom.")
    print("Closing in 15 seconds...")
    for _ in range(15 * 240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
else:
    # Headless: render a single frame and save it
    width, height = 960, 720
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.3, 0, 0.3],
        distance=1.5, yaw=50, pitch=-30, roll=0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=10,
    )
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj)
    from PIL import Image
    import numpy as np
    Image.fromarray(np.array(rgb, dtype=np.uint8)).save("hello_panda.png")
    print("\nSaved hello_panda.png")

p.disconnect()
print("\nDone.")
