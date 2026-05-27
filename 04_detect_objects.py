"""Detect cube objects in the camera image with a fine-tuned YOLO model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from common_scene import CUBE_Z, load_camera_params, pixel_to_world


DEFAULT_MODEL = Path("models") / "yolo_cube.pt"


def run_yolo(model_path: Path, image_path: Path, conf: float):
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {model_path}. Train it with 07_train_yolo.py "
            "or copy runs/yolo_cube_sorting/.../weights/best.pt to models/yolo_cube.pt."
        )
    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    return results[0]


def detections_from_result(result):
    detections = []
    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].detach().cpu().numpy()
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        detections.append(
            {
                "id": index,
                "class_id": class_id,
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        )
    return detections


def draw_detections(image_bgr, detections, output_path: Path):
    display = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        cx, cy = [int(round(v)) for v in det["center"]]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"cube {det['confidence']:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
    cv2.imwrite(str(output_path), display)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("scene_view.png"))
    parser.add_argument("--camera", type=Path, default=Path("camera_params.npz"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("detection_result.png"))
    parser.add_argument("--conf", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"{args.image} not found. Run 03_camera_capture.py first.")

    result = run_yolo(args.model, args.image, args.conf)
    detections = detections_from_result(result)
    cam_cfg = load_camera_params(args.camera)

    print("\nYOLO cube detections:")
    if not detections:
        print("  no cubes detected")
    for det in detections:
        cx, cy = det["center"]
        world = pixel_to_world(cx, cy, cam_cfg, z=CUBE_Z)
        det["world"] = world
        print(
            f"  cube_{det['id']:02d}: conf={det['confidence']:.2f}, "
            f"pixel=({cx:.1f}, {cy:.1f}), world=({world[0]:.3f}, {world[1]:.3f}, {world[2]:.3f})"
        )

    draw_detections(image_bgr, detections, args.output)
    print(f"\nSaved annotated detections to {args.output}")


if __name__ == "__main__":
    main()
