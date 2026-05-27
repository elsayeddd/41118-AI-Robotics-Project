"""Fine-tune a pretrained YOLO detector for cluttered cube detection.

Run after generating the dataset:
    python 06_generate_yolo_dataset.py
    python 07_train_yolo.py

The default uses pretrained YOLO weights rather than training from scratch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets") / "cube_sorting" / "data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=str, default="runs/yolo_cube_sorting")
    parser.add_argument("--name", type=str, default="pretrained_yolo_cube")
    parser.add_argument("--copy-best-to", type=Path, default=Path("models") / "yolo_cube.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} not found. Run 06_generate_yolo_dataset.py first.")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "pretrained": True,
        "plots": True,
        "val": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"Fine-tuning pretrained YOLO model: {args.model}")
    results = model.train(**train_kwargs)
    metrics = model.val(data=str(args.data), imgsz=args.imgsz)

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_path = save_dir / "weights" / "best.pt"
    if best_path.exists():
        args.copy_best_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_best_to.write_bytes(best_path.read_bytes())
        print(f"Copied best model to {args.copy_best_to}")
    else:
        print(f"Training finished, but best weights were not found at {best_path}")

    print("\nValidation metrics summary:")
    print(metrics)
    print(f"\nOpen training plots and TensorBoard logs under: {save_dir}")


if __name__ == "__main__":
    main()
