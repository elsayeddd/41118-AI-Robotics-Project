"""Neural cube perception utilities.

The model predicts one detection per known cube colour from an RGB camera
image. Each colour head outputs confidence plus normalized image coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import cv2
import numpy as np
import pybullet as p
import torch
from torch import nn


COLOUR_NAMES = ("red", "green", "blue")
MODEL_PATH = Path("models") / "cube_perception.pt"


class CubePerceptionNet(nn.Module):
    """Small CNN for simulated cube localisation."""

    def __init__(self, num_colours: int = len(COLOUR_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_colours * 3),
        )
        self.num_colours = num_colours

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x))
        return out.view(-1, self.num_colours, 3)


def load_model(path: str | Path = MODEL_PATH, device: str | torch.device | None = None) -> CubePerceptionNet:
    """Load a trained detector checkpoint."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(path), map_location=device)
    model = CubePerceptionNet(num_colours=len(COLOUR_NAMES)).to(device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_rgb(rgb: np.ndarray, image_size: Tuple[int, int] = (160, 120)) -> torch.Tensor:
    """Convert an RGB uint8 image to a normalized CHW tensor."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape H x W x 3.")
    resized = cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    return tensor


@torch.no_grad()
def predict_pixels(
    model: CubePerceptionNet,
    rgb: np.ndarray,
    threshold: float = 0.45,
    device: str | torch.device | None = None,
) -> Dict[str, Tuple[int, int, float]]:
    """Predict cube centers in pixel coordinates for all visible colours."""

    device = torch.device(device or next(model.parameters()).device)
    h, w = rgb.shape[:2]
    batch = preprocess_rgb(rgb).unsqueeze(0).to(device)
    raw = model(batch)[0]
    conf = torch.sigmoid(raw[:, 0])
    coords = torch.sigmoid(raw[:, 1:3])

    detections: Dict[str, Tuple[int, int, float]] = {}
    for i, name in enumerate(COLOUR_NAMES):
        score = float(conf[i].item())
        if score < threshold:
            continue
        x_norm = float(coords[i, 0].item())
        y_norm = float(coords[i, 1].item())
        cx = int(round(np.clip(x_norm, 0.0, 1.0) * (w - 1)))
        cy = int(round(np.clip(y_norm, 0.0, 1.0) * (h - 1)))
        detections[name] = (cx, cy, score)
    return detections


def pixel_to_world(
    cx: float,
    cy: float,
    cam: Mapping[str, np.ndarray | float | int],
    cube_height: float = 0.04,
) -> list[float]:
    """Convert image coordinates to world coordinates via ray-plane hit."""

    cam_eye = np.asarray(cam["cam_eye"], dtype=float)
    cam_target = np.asarray(cam["cam_target"], dtype=float)
    cam_up = np.asarray(cam["cam_up"], dtype=float)
    fov = float(cam["fov"])
    width = int(cam["width"])
    height = int(cam["height"])

    view = np.array(
        p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=cam_up.tolist(),
        ),
        dtype=float,
    ).reshape(4, 4, order="F")
    proj = np.array(
        p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width / height,
            nearVal=0.1,
            farVal=3.0,
        ),
        dtype=float,
    ).reshape(4, 4, order="F")
    inv_view_proj = np.linalg.inv(proj @ view)

    x_ndc = (2.0 * cx / max(width - 1, 1)) - 1.0
    y_ndc = 1.0 - (2.0 * cy / max(height - 1, 1))
    near = inv_view_proj @ np.array([x_ndc, y_ndc, -1.0, 1.0])
    far = inv_view_proj @ np.array([x_ndc, y_ndc, 1.0, 1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]

    ray = far - near
    plane_z = float(cam_target[2]) + cube_height / 2
    if abs(ray[2]) < 1e-8:
        raise ValueError("Camera ray is parallel to the cube-height plane.")
    t = (plane_z - near[2]) / ray[2]
    world = near + t * ray
    return [float(world[0]), float(world[1]), float(plane_z)]


def pixels_to_world(
    pixels: Mapping[str, Tuple[int, int, float] | Tuple[int, int]],
    cam: Mapping[str, np.ndarray | float | int],
) -> Dict[str, list[float]]:
    world = {}
    for name, values in pixels.items():
        cx, cy = values[:2]
        world[name] = pixel_to_world(float(cx), float(cy), cam)
    return world


def load_camera_params(path: str | Path = "camera_params.npz") -> Dict[str, np.ndarray | float | int]:
    cam = np.load(path)
    return {
        "cam_eye": cam["cam_eye"],
        "cam_target": cam["cam_target"],
        "cam_up": cam["cam_up"] if "cam_up" in cam.files else np.array([1.0, 0.0, 0.0]),
        "fov": float(cam["fov"]),
        "width": int(cam["width"]),
        "height": int(cam["height"]),
    }


def detections_to_tensor(
    labels: Mapping[str, Tuple[float, float]],
    width: int,
    height: int,
    colours: Iterable[str] = COLOUR_NAMES,
) -> torch.Tensor:
    """Build training target tensor shaped colours x (present, x_norm, y_norm)."""

    target = torch.zeros((len(tuple(colours)), 3), dtype=torch.float32)
    for i, name in enumerate(colours):
        if name not in labels:
            continue
        cx, cy = labels[name]
        target[i, 0] = 1.0
        target[i, 1] = float(cx) / max(width - 1, 1)
        target[i, 2] = float(cy) / max(height - 1, 1)
    return target
