"""
02_tabletop_scene.py - Build the scene we'll actually work with.

Goal: Arm + table + several coloured cubes. This is the world your
pick-and-place project lives in.

Run with:
    python 02_tabletop_scene.py
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

GUI = True

if GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Ground plane
p.loadURDF("plane.urdf")

# A table. PyBullet ships a table URDF.
table_id = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.5, 0, 0],  # in front of the arm
    useFixedBase=True,
)

# The arm. We raise it onto the table edge so it can reach the surface.
panda_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0.62],  # ~table height
    useFixedBase=True,
)

# Helper to spawn a cube of given colour at a given position.
# Colours are RGBA, values 0-1.
def spawn_cube(position, colour, size=0.04):
    half = size / 2
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=colour)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
    cube_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    return cube_id

# Spawn three cubes on the table at random positions within the arm's workspace.
# Table top is at z ~= 0.625, so cubes sit just above that.
np.random.seed(0)  # remove this line later for randomised scenes

colours = {
    "red":   [1, 0, 0, 1],
    "green": [0, 0.7, 0, 1],
    "blue":  [0, 0, 1, 1],
}

cube_ids = {}
for name, colour in colours.items():
    x = np.random.uniform(0.35, 0.65)
    y = np.random.uniform(-0.2, 0.2)
    z = 0.65
    cube_ids[name] = spawn_cube([x, y, z], colour)
    print(f"Spawned {name} cube at ({x:.2f}, {y:.2f})")

# Mark the target zone with a flat green pad on the table
target_pos = [0.5, 0.35, 0.625]
target_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.001],
    rgbaColor=[0.2, 1, 0.2, 0.5],
)
p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=target_visual,
    basePosition=target_pos,
)
print(f"Target zone at {target_pos}")

# Camera angle for viewing
p.resetDebugVisualizerCamera(
    cameraDistance=1.4,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.4, 0.1, 0.7],
)

# Let physics settle, then idle
print("\nScene loaded. Cubes will settle onto the table.")
print("Closing in 15 seconds...")
for _ in range(15 * 240):
    p.stepSimulation()
    if GUI:
        time.sleep(1.0 / 240.0)

p.disconnect()
