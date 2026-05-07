"""
02_tabletop_scene.py - Build the scene we'll actually work with.

Goal: Arm + table + several coloured cubes. This is the world your
pick-and-place project lives in.

Run with:
    python 02_tabletop_scene.py
"""

import pybullet as p
import time

from common_scene import TARGET_ZONE, build_tabletop_scene, setup_world

GUI = True

setup_world(gui=GUI)
_, cube_positions = build_tabletop_scene(seed=0)

for name, (_, expected_pos) in cube_positions.items():
    print(f"Spawned {name} cube at ({expected_pos[0]:.2f}, {expected_pos[1]:.2f})")

print(f"Target zone at {TARGET_ZONE.tolist()}")

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
