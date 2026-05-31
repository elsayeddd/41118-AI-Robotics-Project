"""Run several random full-arm sorting demos in sequence.

This is the end-to-end stress/demo runner: each scene starts from a new random
layout, calls 05_pick_and_place.py, and records whether that scene completed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=Path("models") / "yolo_cube.pt")
    parser.add_argument("--rl-policy", type=Path, default=Path("models") / "sorting_policy.zip")
    parser.add_argument("--direct", action="store_true", help="Run without PyBullet GUI.")
    parser.add_argument("--scene-delay", type=float, default=1.0, help="Pause between scene runs.")
    return parser.parse_args()


def run_scene(args, index: int) -> int:
    seed = args.seed + index
    command = [
        sys.executable,
        "05_pick_and_place.py",
        "--seed",
        str(seed),
        "--distractors",
        str(args.distractors),
        "--conf",
        str(args.conf),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--model",
        str(args.model),
        "--rl-policy",
        str(args.rl_policy),
    ]
    if args.direct:
        command.append("--direct")

    print(f"\n=== Random sorting layout {index + 1}/{args.scenes} | seed={seed} ===")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main():
    args = parse_args()
    failures = 0
    for index in range(args.scenes):
        return_code = run_scene(args, index)
        if return_code != 0:
            failures += 1
            print(f"Scene {index + 1} failed with exit code {return_code}.")
        if index < args.scenes - 1 and args.scene_delay > 0:
            time.sleep(args.scene_delay)

    successes = args.scenes - failures
    print(f"\nBatch sorting demo complete: {successes}/{args.scenes} scene runs exited cleanly.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
