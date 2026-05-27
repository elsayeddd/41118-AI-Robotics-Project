"""Generate a cluttered synthetic YOLO dataset for cube detection.

The dataset contains cube objects mixed with non-cube distractors. Labels are
generated from PyBullet segmentation masks, so every image is automatically
annotated in YOLO format.

Run:
    python 06_generate_yolo_dataset.py --samples 1200 --val-samples 240
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from common_scene import (
    build_tabletop_scene,
    capture_camera,
    jittered_camera_params,
    reset_world,
    segmentation_object_ids,
    setup_world,
)


@dataclass(frozen=True)
class DatasetStage:
    name: str
    fraction: float
    min_cubes: int
    max_cubes: int
    min_distractors: int
    max_distractors: int
    camera_jitter: float
    random_cube_colours: bool
    arm_occlusion: bool


CURRICULUM = (
    DatasetStage("easy_single_cube", 0.20, 1, 1, 0, 1, 0.00, False, False),
    DatasetStage("multi_cube", 0.25, 2, 3, 0, 2, 0.02, True, False),
    DatasetStage("light_clutter", 0.25, 2, 4, 2, 5, 0.04, True, False),
    DatasetStage("dense_clutter_occlusion", 0.30, 3, 5, 5, 10, 0.08, True, True),
)


def choose_stage(index: int, total: int) -> DatasetStage:
    progress = (index + 1) / max(total, 1)
    cumulative = 0.0
    for stage in CURRICULUM:
        cumulative += stage.fraction
        if progress <= cumulative:
            return stage
    return CURRICULUM[-1]


def set_arm_pose_for_image(panda_id: int, rng: np.random.Generator, occluding: bool):
    if occluding:
        pose = [
            float(rng.uniform(-0.35, 0.35)),
            float(rng.uniform(-0.9, -0.2)),
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(-2.5, -1.5)),
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(1.4, 2.4)),
            float(rng.uniform(0.2, 1.0)),
        ]
    else:
        pose = [0, -0.4, 0, -2.2, 0, 2.0, 0.8]
    for joint, q in enumerate(pose):
        p.resetJointState(panda_id, joint, q)


def visible_bbox_from_segmentation(object_ids, body_id: int, min_pixels: int = 12):
    ys, xs = np.where(object_ids == body_id)
    if len(xs) < min_pixels:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    x1 = np.clip(x1, 0, width - 1)
    x2 = np.clip(x2, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    y2 = np.clip(y2, 0, height - 1)
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = max(x2 - x1, 1.0) / width
    bh = max(y2 - y1, 1.0) / height
    return 0, xc, yc, bw, bh


def render_example(index: int, total: int, split: str, args, rng: np.random.Generator):
    stage = choose_stage(index, total)
    reset_world()

    num_cubes = int(rng.integers(stage.min_cubes, stage.max_cubes + 1))
    num_distractors = int(rng.integers(stage.min_distractors, stage.max_distractors + 1))
    cube_names = [f"cube_{i}" for i in range(num_cubes)]
    seed = int(rng.integers(0, 2_000_000_000))
    panda_id, cubes, _ = build_tabletop_scene(
        seed=seed,
        num_distractors=num_distractors,
        cube_names=cube_names,
        randomize_cube_colours=stage.random_cube_colours,
        return_distractors=True,
    )
    set_arm_pose_for_image(panda_id, rng, occluding=stage.arm_occlusion and rng.random() < 0.55)

    for _ in range(60):
        p.stepSimulation()

    cam_cfg = jittered_camera_params(
        rng,
        width=args.width,
        height=args.height,
        fov=float(rng.uniform(args.min_fov, args.max_fov)),
        eye_jitter=stage.camera_jitter,
        target_jitter=stage.camera_jitter * 0.35,
    )
    rgb, _, seg = capture_camera(cam_cfg, renderer=p.ER_TINY_RENDERER, shadow=True)

    if args.image_noise > 0:
        noise = rng.normal(0.0, args.image_noise * 255.0, size=rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    object_ids = segmentation_object_ids(seg)
    labels = []
    for cube_id, _ in cubes.values():
        bbox = visible_bbox_from_segmentation(object_ids, cube_id)
        if bbox is not None:
            labels.append(bbox_to_yolo(bbox, args.width, args.height))

    stem = f"{split}_{index:06d}"
    image_path = args.output / "images" / split / f"{stem}.jpg"
    label_path = args.output / "labels" / split / f"{stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    with label_path.open("w", encoding="utf-8") as handle:
        for class_id, xc, yc, bw, bh in labels:
            handle.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return stage.name, len(labels)


def write_data_yaml(output: Path):
    yaml_text = f"""path: {output.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: cube
"""
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets") / "cube_sorting")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=240)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-fov", type=float, default=52.0)
    parser.add_argument("--max-fov", type=float, default=68.0)
    parser.add_argument("--image-noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_split(split: str, count: int, args, rng: np.random.Generator):
    stage_counts = {}
    labelled = 0
    for i in range(count):
        stage, labels = render_example(i, count, split, args, rng)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        labelled += labels
        if (i + 1) % 50 == 0 or i + 1 == count:
            print(f"{split}: {i + 1}/{count} images, labels={labelled}")
    return stage_counts, labelled


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    setup_world(gui=False)
    try:
        train_counts, train_labels = generate_split("train", args.samples, args, rng)
        val_counts, val_labels = generate_split("val", args.val_samples, args, rng)
    finally:
        p.disconnect()

    write_data_yaml(args.output)
    print("\nDataset complete")
    print(f"  train labels: {train_labels}")
    print(f"  val labels:   {val_labels}")
    print(f"  data yaml:    {args.output / 'data.yaml'}")
    print(f"  curriculum train stages: {train_counts}")
    print(f"  curriculum val stages:   {val_counts}")


if __name__ == "__main__":
    main()
