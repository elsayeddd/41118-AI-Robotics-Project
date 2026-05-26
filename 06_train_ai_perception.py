"""Train the neural cube perception model with curriculum learning.

The training data is generated inside PyBullet, so no hand-labelled dataset is
needed. Curriculum stages gradually increase scene variation, and TensorBoard
logs show whether the model improves over time.

Run:
    python 06_train_ai_perception.py

TensorBoard:
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ai_perception import COLOUR_NAMES, MODEL_PATH, CubePerceptionNet, detections_to_tensor, preprocess_rgb
from common_scene import COLOURS, TABLE_CENTER, TARGET_PAD_COLOUR, TARGET_ZONE, spawn_cube


SIM_HZ = 240


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    epochs: int
    xy_span: Tuple[Tuple[float, float], Tuple[float, float]]
    camera_xy_jitter: float
    colour_noise: float
    image_noise: float
    min_cube_gap: float


CURRICULUM = (
    CurriculumStage(
        name="fixed_easy",
        epochs=4,
        xy_span=((0.42, 0.58), (-0.12, 0.12)),
        camera_xy_jitter=0.00,
        colour_noise=0.00,
        image_noise=0.00,
        min_cube_gap=0.09,
    ),
    CurriculumStage(
        name="random_positions",
        epochs=6,
        xy_span=((0.35, 0.65), (-0.20, 0.20)),
        camera_xy_jitter=0.00,
        colour_noise=0.04,
        image_noise=0.01,
        min_cube_gap=0.07,
    ),
    CurriculumStage(
        name="camera_jitter",
        epochs=6,
        xy_span=((0.33, 0.67), (-0.22, 0.22)),
        camera_xy_jitter=0.04,
        colour_noise=0.07,
        image_noise=0.02,
        min_cube_gap=0.06,
    ),
    CurriculumStage(
        name="full_variation",
        epochs=8,
        xy_span=((0.32, 0.68), (-0.24, 0.24)),
        camera_xy_jitter=0.07,
        colour_noise=0.10,
        image_noise=0.035,
        min_cube_gap=0.055,
    ),
)


def connect_direct() -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)


def reset_world() -> int:
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0], useFixedBase=True)
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.62], useFixedBase=True)
    for joint, q in enumerate([0, -0.4, 0, -2.2, 0, 2.0, 0.8]):
        p.resetJointState(panda_id, joint, q)
    target_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.05, 0.05, 0.001],
        rgbaColor=TARGET_PAD_COLOUR,
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=TARGET_ZONE)
    return panda_id


def jitter_colour(base_rgba, amount: float) -> list[float]:
    rgba = np.array(base_rgba, dtype=float)
    if amount > 0:
        rgba[:3] += np.random.uniform(-amount, amount, size=3)
    rgba[:3] = np.clip(rgba[:3], 0.0, 1.0)
    return rgba.tolist()


def sample_positions(stage: CurriculumStage) -> Dict[str, list[float]]:
    positions: Dict[str, list[float]] = {}
    x_range, y_range = stage.xy_span
    for name in COLOUR_NAMES:
        for _ in range(100):
            x = float(np.random.uniform(*x_range))
            y = float(np.random.uniform(*y_range))
            candidate = np.array([x, y])
            if all(np.linalg.norm(candidate - np.array(pos[:2])) >= stage.min_cube_gap for pos in positions.values()):
                positions[name] = [x, y, 0.65]
                break
        else:
            positions[name] = [float(np.random.uniform(*x_range)), float(np.random.uniform(*y_range)), 0.65]
    return positions


def camera_for_stage(stage: CurriculumStage, width: int, height: int, fov: float):
    jitter = stage.camera_xy_jitter
    cam_target = TABLE_CENTER.astype(float).copy()
    cam_eye = np.array([0.72, 0.0, 1.45], dtype=float)
    if jitter:
        cam_eye[:2] += np.random.uniform(-jitter, jitter, size=2)
        cam_target[:2] += np.random.uniform(-jitter * 0.35, jitter * 0.35, size=2)
    cam_up = np.array([1.0, 0.0, 0.0], dtype=float)
    view = p.computeViewMatrix(cam_eye.tolist(), cam_target.tolist(), cam_up.tolist())
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=width / height, nearVal=0.1, farVal=3.0)
    return cam_eye, cam_target, cam_up, view, proj


def render_sample(stage: CurriculumStage, width: int, height: int, fov: float):
    reset_world()
    cube_ids: Dict[str, int] = {}
    for name, position in sample_positions(stage).items():
        cube_ids[name] = spawn_cube(position, jitter_colour(COLOURS[name], stage.colour_noise))

    for _ in range(80):
        p.stepSimulation()

    _, _, _, view, proj = camera_for_stage(stage, width, height, fov)
    _, _, rgb, _, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )

    rgb_array = np.asarray(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    if stage.image_noise > 0:
        noise = np.random.normal(0, stage.image_noise * 255.0, size=rgb_array.shape)
        rgb_array = np.clip(rgb_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    seg_array = np.asarray(seg).reshape((height, width))
    object_ids = seg_array & ((1 << 24) - 1)
    labels: Dict[str, Tuple[float, float]] = {}
    for name, cube_id in cube_ids.items():
        ys, xs = np.where(object_ids == cube_id)
        if len(xs) == 0:
            continue
        labels[name] = (float(xs.mean()), float(ys.mean()))
    return rgb_array, labels


def make_batch(stage: CurriculumStage, batch_size: int, width: int, height: int, fov: float, device):
    images = []
    targets = []
    for _ in range(batch_size):
        rgb, labels = render_sample(stage, width, height, fov)
        images.append(preprocess_rgb(rgb))
        targets.append(detections_to_tensor(labels, width, height))
    return torch.stack(images).to(device), torch.stack(targets).to(device)


def detection_reward(pred: torch.Tensor, target: torch.Tensor, tolerance: float = 0.06) -> torch.Tensor:
    """Reward combines colour presence and normalized coordinate accuracy."""

    conf = torch.sigmoid(pred[..., 0])
    coords = torch.sigmoid(pred[..., 1:3])
    present = target[..., 0]
    true_coords = target[..., 1:3]
    visible = present > 0.5

    if not bool(visible.any()):
        return torch.tensor(0.0, device=pred.device)

    conf_ok = (conf[visible] > 0.5).float()
    error = torch.linalg.norm(coords[visible] - true_coords[visible], dim=-1)
    loc_reward = torch.clamp(1.0 - error / tolerance, min=0.0, max=1.0)
    return (0.35 * conf_ok + 0.65 * loc_reward).mean()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    connect_direct()

    model = CubePerceptionNet(num_colours=len(COLOUR_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    writer = SummaryWriter(log_dir=args.logdir)
    global_step = 0

    try:
        for stage_index, stage in enumerate(CURRICULUM):
            for epoch in range(stage.epochs):
                model.train()
                epoch_losses = []
                epoch_rewards = []

                for _ in range(args.batches_per_epoch):
                    images, target = make_batch(stage, args.batch_size, args.width, args.height, args.fov, device)
                    pred = model(images)

                    present = target[..., 0]
                    conf_loss = F.binary_cross_entropy_with_logits(pred[..., 0], present)
                    coord_loss = F.smooth_l1_loss(
                        torch.sigmoid(pred[..., 1:3]) * present.unsqueeze(-1),
                        target[..., 1:3] * present.unsqueeze(-1),
                    )
                    loss = conf_loss + 4.0 * coord_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    reward = detection_reward(pred.detach(), target)
                    writer.add_scalar("train/loss", float(loss.item()), global_step)
                    writer.add_scalar("train/reward", float(reward.item()), global_step)
                    writer.add_scalar("train/curriculum_stage", stage_index, global_step)
                    epoch_losses.append(float(loss.item()))
                    epoch_rewards.append(float(reward.item()))
                    global_step += 1

                model.eval()
                with torch.no_grad():
                    val_images, val_target = make_batch(stage, args.val_batch_size, args.width, args.height, args.fov, device)
                    val_pred = model(val_images)
                    val_reward = detection_reward(val_pred, val_target)

                writer.add_scalar("val/reward", float(val_reward.item()), global_step)
                writer.add_text("curriculum/stage", stage.name, global_step)
                print(
                    f"[{stage.name:16s}] epoch {epoch + 1:02d}/{stage.epochs:02d} "
                    f"loss={np.mean(epoch_losses):.4f} reward={np.mean(epoch_rewards):.3f} "
                    f"val_reward={float(val_reward.item()):.3f}"
                )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "colour_names": COLOUR_NAMES,
                "image_size": (160, 120),
                "curriculum": [stage.name for stage in CURRICULUM],
            },
            args.output,
        )
        print(f"\nSaved trained model to {args.output}")
    finally:
        writer.close()
        p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--logdir", type=str, default="runs/cube_perception")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
