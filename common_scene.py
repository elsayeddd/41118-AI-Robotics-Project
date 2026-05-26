"""Shared scene/camera utilities for the Panda tabletop project."""

from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_data

TABLE_TOP_Z = 0.625
TABLE_CENTER = np.array([0.5, 0.0, TABLE_TOP_Z], dtype=float)
TARGET_ZONE = np.array([0.5, 0.35, TABLE_TOP_Z], dtype=float)

COLOURS = {
    "red": [1, 0, 0, 1],
    "green": [0, 0.7, 0, 1],
    "blue": [0, 0, 1, 1],
}
TARGET_PAD_COLOUR = [1.0, 1.0, 0.0, 0.5]  # yellow to avoid green-cube confusion


def setup_world(gui: bool = True) -> int:
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return mode


def spawn_cube(position, colour, size=0.04):
    half = size / 2
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=colour)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
    return p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def build_tabletop_scene(seed: int = 0):
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0], useFixedBase=True)
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.62], useFixedBase=True)

    rng = np.random.default_rng(seed)
    cube_positions = {}
    for name, colour in COLOURS.items():
        x = float(rng.uniform(0.35, 0.65))
        y = float(rng.uniform(-0.2, 0.2))
        cube_id = spawn_cube([x, y, 0.65], colour)
        cube_positions[name] = (cube_id, [x, y, 0.65])

    target_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.05, 0.05, 0.001],
        rgbaColor=TARGET_PAD_COLOUR,
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=TARGET_ZONE)
    return panda_id, cube_positions


def camera_params(width=640, height=480, fov=60.0):
    return {
        "width": width,
        "height": height,
        "fov": fov,
        "cam_target": TABLE_CENTER.tolist(),
        "cam_eye": [0.72, 0.0, 1.45],
        "cam_up": [1, 0, 0],
    }
