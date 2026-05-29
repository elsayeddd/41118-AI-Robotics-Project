"""Evaluate YOLO cube detection across random cluttered scenes.

This gives report-ready metrics instead of relying on a single demo run:
- cube recall against PyBullet ground truth
- false positives on clutter/background
- mean localisation error for matched cube detections
- scene pass rate

Run:
    python 09_evaluate_system.py --scenes 50
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
from ultralytics import YOLO

from common_scene import (
    CUBE_Z,
    build_tabletop_scene,
    camera_params,
    capture_camera,
    pixel_to_world,
    save_camera_params,
    setup_world,
)


DEFAULT_MODEL = Path("models") / "yolo_cube.pt"
DEFAULT_OUTPUT = Path("evaluation") / "latest"


def settle(steps: int = 120, gui: bool = False):
    for _ in range(steps):
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / 240.0)


def configure_gui_view():
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.25,
        cameraYaw=42.0,
        cameraPitch=-42.0,
        cameraTargetPosition=[0.5, 0.0, 0.62],
    )


def park_arm(robot_id: int):
    for joint, q in enumerate([0.0, -0.45, 0.0, -2.25, 0.0, 2.05, 0.8]):
        p.resetJointState(robot_id, joint, q)


def detect(model: YOLO, image_path: Path, cam_cfg, conf: float):
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().numpy()]
        score = float(box.conf[0].detach().cpu().item())
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append(
            {
                "confidence": score,
                "xyxy": [x1, y1, x2, y2],
                "center": [cx, cy],
                "world": pixel_to_world(cx, cy, cam_cfg, z=CUBE_Z),
            }
        )
    return detections


def greedy_match(detections, cube_truth, distance_threshold: float):
    pairs = []
    for det_index, det in enumerate(detections):
        det_xy = np.array(det["world"][:2], dtype=float)
        for name, (_, pos) in cube_truth.items():
            true_xy = np.array(pos[:2], dtype=float)
            distance = float(np.linalg.norm(det_xy - true_xy))
            if distance <= distance_threshold:
                pairs.append((distance, det_index, name))

    pairs.sort(key=lambda item: item[0])
    used_detections = set()
    used_cubes = set()
    matches = []
    for distance, det_index, name in pairs:
        if det_index in used_detections or name in used_cubes:
            continue
        used_detections.add(det_index)
        used_cubes.add(name)
        matches.append((det_index, name, distance))
    return matches


def draw_scene(image_path: Path, detections, matches, output_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    matched_detection_ids = {det_index for det_index, _, _ in matches}
    for index, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        colour = (0, 220, 0) if index in matched_detection_ids else (0, 0, 255)
        label = "cube" if index in matched_detection_ids else "false+"
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image,
            f"{label} {det['confidence']:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def evaluate_scene(index: int, args, model: YOLO):
    seed = args.seed + index
    p.resetSimulation()
    panda_id, cube_truth, distractors = build_tabletop_scene(
        seed=seed,
        num_distractors=args.distractors,
        randomize_cube_colours=True,
        return_distractors=True,
    )
    park_arm(panda_id)
    if args.gui:
        configure_gui_view()
    settle(gui=args.gui)

    cam_cfg = camera_params(width=args.width, height=args.height, fov=60.0)
    rgb, _, _ = capture_camera(cam_cfg, renderer=p.ER_TINY_RENDERER)

    image_path = args.output / "images" / f"scene_{index:04d}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    save_camera_params(cam_cfg, args.output / "camera_params.npz")

    detections = detect(model, image_path, cam_cfg, conf=args.conf)
    matches = greedy_match(detections, cube_truth, args.match_distance)
    matched_cubes = {name for _, name, _ in matches}

    false_positives = max(0, len(detections) - len(matches))
    missed = len(cube_truth) - len(matched_cubes)
    mean_error = float(np.mean([distance for _, _, distance in matches])) if matches else None
    pass_scene = missed == 0 and false_positives <= args.max_false_positives

    if args.save_images:
        draw_scene(image_path, detections, matches, args.output / "annotated" / f"scene_{index:04d}.jpg")

    if args.gui and args.scene_delay > 0:
        time.sleep(args.scene_delay)

    return {
        "scene": index,
        "seed": seed,
        "cubes": len(cube_truth),
        "distractors": len(distractors),
        "detections": len(detections),
        "matched_cubes": len(matches),
        "missed_cubes": missed,
        "false_positives": false_positives,
        "mean_error_m": mean_error,
        "pass": pass_scene,
    }


def write_outputs(rows, args):
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_cubes = sum(row["cubes"] for row in rows)
    matched = sum(row["matched_cubes"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    pass_rate = sum(1 for row in rows if row["pass"]) / len(rows)
    errors = [row["mean_error_m"] for row in rows if row["mean_error_m"] is not None]
    summary = {
        "scenes": len(rows),
        "cube_recall": matched / total_cubes if total_cubes else 0.0,
        "false_positives_per_scene": false_positives / len(rows),
        "scene_pass_rate": pass_rate,
        "mean_localisation_error_m": float(np.mean(errors)) if errors else None,
        "match_distance_m": args.match_distance,
        "confidence_threshold": args.conf,
    }
    json_path = args.output / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation summary")
    print(f"  scenes:                 {summary['scenes']}")
    print(f"  cube recall:            {summary['cube_recall']:.3f}")
    print(f"  false positives/scene:  {summary['false_positives_per_scene']:.3f}")
    print(f"  scene pass rate:        {summary['scene_pass_rate']:.3f}")
    if summary["mean_localisation_error_m"] is not None:
        print(f"  mean localisation err:  {summary['mean_localisation_error_m']:.3f} m")
    print(f"  wrote:                  {csv_path}")
    print(f"  wrote:                  {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--match-distance", type=float, default=0.075)
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Show each evaluation scene in the PyBullet GUI.")
    parser.add_argument(
        "--scene-delay",
        type=float,
        default=0.75,
        help="Seconds to pause after each GUI scene so the detections can be inspected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"YOLO model not found at {args.model}.")

    setup_world(gui=args.gui)
    model = YOLO(str(args.model))
    rows = []
    try:
        for index in range(args.scenes):
            row = evaluate_scene(index, args, model)
            rows.append(row)
            print(
                f"scene {index + 1:03d}/{args.scenes}: "
                f"matched={row['matched_cubes']}/{row['cubes']} "
                f"false+={row['false_positives']} pass={row['pass']}"
            )
    finally:
        p.disconnect()

    write_outputs(rows, args)


if __name__ == "__main__":
    main()
