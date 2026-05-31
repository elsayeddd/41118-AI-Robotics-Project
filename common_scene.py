"""Shared scene, camera, and projection utilities for cube sorting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pybullet as p
import pybullet_data

TABLE_TOP_Z = 0.625
TABLE_CENTER = np.array([0.5, 0.0, TABLE_TOP_Z], dtype=float)
TARGET_ZONE = np.array([0.5, 0.35, TABLE_TOP_Z], dtype=float)
CUBE_SIZE = 0.04
CUBE_Z = TABLE_TOP_Z + CUBE_SIZE / 2 + 0.005

COLOURS = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "green": [0.0, 0.7, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
}

CUBE_COLOUR_PALETTE = (
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.7, 0.0, 1.0],
    [0.0, 0.15, 1.0, 1.0],
    [1.0, 0.55, 0.0, 1.0],
    [0.95, 0.1, 0.8, 1.0],
    [0.05, 0.85, 0.85, 1.0],
)

DISTRACTOR_COLOURS = (
    [0.9, 0.85, 0.2, 1.0],
    [0.2, 0.2, 0.2, 1.0],
    [0.8, 0.8, 0.8, 1.0],
    [0.45, 0.25, 0.1, 1.0],
    [0.55, 0.2, 0.65, 1.0],
)

TARGET_PAD_COLOUR = [1.0, 1.0, 0.0, 0.5]
WORKSPACE_BOUNDS = ((0.30, 0.70), (-0.25, 0.25))


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fov: float = 60.0
    cam_target: tuple[float, float, float] = (0.5, 0.0, TABLE_TOP_Z)
    cam_eye: tuple[float, float, float] = (0.72, 0.0, 1.45)
    cam_up: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def as_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "fov": self.fov,
            "cam_target": list(self.cam_target),
            "cam_eye": list(self.cam_eye),
            "cam_up": list(self.cam_up),
        }


def setup_world(gui: bool = True) -> int:
    mode = p.GUI if gui else p.DIRECT
    if p.isConnected():
        p.disconnect()
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return mode


def reset_world() -> None:
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)


def spawn_cube(position, colour, size=CUBE_SIZE, mass=0.05):
    half = size / 2
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=colour)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def spawn_distractor(position, rng: np.random.Generator, size_scale: float = 1.0):
    kind = str(rng.choice(["box", "cylinder", "sphere"]))
    colour = DISTRACTOR_COLOURS[int(rng.integers(0, len(DISTRACTOR_COLOURS)))]

    if kind == "box":
        half_extents = (rng.uniform(0.015, 0.055, size=3) * size_scale).tolist()
        half_extents[2] = max(half_extents[2], 0.012)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=colour)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        z = TABLE_TOP_Z + half_extents[2] + 0.002
    elif kind == "cylinder":
        radius = float(rng.uniform(0.015, 0.035) * size_scale)
        height = float(rng.uniform(0.025, 0.075) * size_scale)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=colour)
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        z = TABLE_TOP_Z + height / 2 + 0.002
    else:
        radius = float(rng.uniform(0.015, 0.035) * size_scale)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=colour)
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        z = TABLE_TOP_Z + radius + 0.002

    return p.createMultiBody(
        baseMass=0.04,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[position[0], position[1], z],
    )


def _sample_non_overlapping_xy(
    rng: np.random.Generator,
    occupied: list[np.ndarray],
    min_gap: float,
    bounds=WORKSPACE_BOUNDS,
) -> np.ndarray:
    x_bounds, y_bounds = bounds
    for _ in range(200):
        xy = np.array([rng.uniform(*x_bounds), rng.uniform(*y_bounds)], dtype=float)
        if all(np.linalg.norm(xy - other) >= min_gap for other in occupied):
            occupied.append(xy)
            return xy
    xy = np.array([rng.uniform(*x_bounds), rng.uniform(*y_bounds)], dtype=float)
    occupied.append(xy)
    return xy


def build_tabletop_scene(
    seed: int = 0,
    num_distractors: int = 0,
    cube_names: Iterable[str] = ("red", "green", "blue"),
    randomize_cube_colours: bool = False,
    return_distractors: bool = False,
):
    rng = np.random.default_rng(seed)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0], useFixedBase=True)
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.62], useFixedBase=True)

    occupied: list[np.ndarray] = []
    cube_positions = {}
    for name in cube_names:
        xy = _sample_non_overlapping_xy(rng, occupied, min_gap=0.075)
        if randomize_cube_colours:
            colour = CUBE_COLOUR_PALETTE[int(rng.integers(0, len(CUBE_COLOUR_PALETTE)))]
        else:
            colour = COLOURS.get(name, CUBE_COLOUR_PALETTE[0])
        cube_id = spawn_cube([float(xy[0]), float(xy[1]), CUBE_Z], colour)
        cube_positions[name] = (cube_id, [float(xy[0]), float(xy[1]), CUBE_Z])

    distractor_ids = []
    for _ in range(num_distractors):
        xy = _sample_non_overlapping_xy(rng, occupied, min_gap=0.055)
        distractor_ids.append(spawn_distractor([float(xy[0]), float(xy[1])], rng))

    target_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.08, 0.06, 0.001],
        rgbaColor=TARGET_PAD_COLOUR,
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=TARGET_ZONE)

    if return_distractors:
        return panda_id, cube_positions, distractor_ids
    return panda_id, cube_positions


def camera_params(width=640, height=480, fov=60.0):
    return CameraConfig(width=width, height=height, fov=fov).as_dict()


def jittered_camera_params(
    rng: np.random.Generator,
    width=640,
    height=480,
    fov=60.0,
    eye_jitter=0.0,
    target_jitter=0.0,
):
    base = CameraConfig(width=width, height=height, fov=fov)
    eye = np.array(base.cam_eye, dtype=float)
    target = np.array(base.cam_target, dtype=float)
    if eye_jitter:
        eye[:2] += rng.uniform(-eye_jitter, eye_jitter, size=2)
        eye[2] += rng.uniform(-eye_jitter * 0.6, eye_jitter * 0.6)
    if target_jitter:
        target[:2] += rng.uniform(-target_jitter, target_jitter, size=2)
    return {
        "width": width,
        "height": height,
        "fov": fov,
        "cam_target": target.tolist(),
        "cam_eye": eye.tolist(),
        "cam_up": list(base.cam_up),
    }


def compute_camera_matrices(cam_cfg):
    view = p.computeViewMatrix(
        cameraEyePosition=cam_cfg["cam_eye"],
        cameraTargetPosition=cam_cfg["cam_target"],
        cameraUpVector=cam_cfg["cam_up"],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam_cfg["fov"],
        aspect=cam_cfg["width"] / cam_cfg["height"],
        nearVal=0.1,
        farVal=3.0,
    )
    return view, proj


def capture_camera(cam_cfg, renderer=p.ER_BULLET_HARDWARE_OPENGL, shadow=True):
    view, proj = compute_camera_matrices(cam_cfg)
    _, _, rgb, depth, seg = p.getCameraImage(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
        shadow=shadow,
    )
    rgb_array = np.asarray(rgb, dtype=np.uint8).reshape((cam_cfg["height"], cam_cfg["width"], 4))[:, :, :3]
    depth_array = np.asarray(depth).reshape((cam_cfg["height"], cam_cfg["width"]))
    seg_array = np.asarray(seg).reshape((cam_cfg["height"], cam_cfg["width"]))
    return rgb_array, depth_array, seg_array


def save_camera_params(cam_cfg, path="camera_params.npz"):
    np.savez(
        path,
        cam_eye=cam_cfg["cam_eye"],
        cam_target=cam_cfg["cam_target"],
        cam_up=cam_cfg["cam_up"],
        fov=cam_cfg["fov"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
    )


def load_camera_params(path="camera_params.npz"):
    cam = np.load(path)
    return {
        "cam_eye": cam["cam_eye"].tolist(),
        "cam_target": cam["cam_target"].tolist(),
        "cam_up": cam["cam_up"].tolist() if "cam_up" in cam.files else [1.0, 0.0, 0.0],
        "fov": float(cam["fov"]),
        "width": int(cam["width"]),
        "height": int(cam["height"]),
    }


def pixel_to_world(cx: float, cy: float, cam_cfg, z: float = CUBE_Z):
    view, proj = compute_camera_matrices(cam_cfg)
    view_m = np.array(view, dtype=float).reshape(4, 4, order="F")
    proj_m = np.array(proj, dtype=float).reshape(4, 4, order="F")
    inv_view_proj = np.linalg.inv(proj_m @ view_m)

    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])
    x_ndc = (2.0 * cx / max(width - 1, 1)) - 1.0
    y_ndc = 1.0 - (2.0 * cy / max(height - 1, 1))

    near = inv_view_proj @ np.array([x_ndc, y_ndc, -1.0, 1.0])
    far = inv_view_proj @ np.array([x_ndc, y_ndc, 1.0, 1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    ray = far - near

    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray is parallel to the cube plane.")
    t = (z - near[2]) / ray[2]
    world = near + t * ray
    return [float(world[0]), float(world[1]), float(z)]


def segmentation_object_ids(seg_array):
    return np.asarray(seg_array, dtype=np.int64) & ((1 << 24) - 1)
