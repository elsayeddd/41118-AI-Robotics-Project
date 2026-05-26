"""
04_detect_objects.py - Find all coloured cubes in the camera image.

This script prefers the trained neural detector produced by
06_train_ai_perception.py. If the model is not available yet, it falls back to
the original HSV detector so the rest of the pipeline remains runnable.
"""

from __future__ import annotations

import cv2
import numpy as np

from ai_perception import MODEL_PATH, load_camera_params, load_model, pixels_to_world, predict_pixels


COLOUR_RANGES = {
    "red": [
        (np.array([0, 90, 60]), np.array([12, 255, 255])),
        (np.array([168, 90, 60]), np.array([179, 255, 255])),
    ],
    "green": [(np.array([35, 50, 20]), np.array([90, 255, 255]))],
    "blue": [(np.array([95, 80, 40]), np.array([135, 255, 255]))],
}

MARKER_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}


def find_cube_hsv(img_hsv, colour_ranges, min_area=50, max_area=2500, min_fill=0.55):
    """Return (cx, cy, score) of best cube-like blob, or None."""

    mask = None
    for low, high in colour_ranges:
        m = cv2.inRange(img_hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = float(w * h)
        if rect_area <= 0:
            continue

        fill = area / rect_area
        aspect = w / float(h)
        if fill < min_fill or aspect < 0.6 or aspect > 1.6:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        score = fill - abs(aspect - 1.0) * 0.2 - (area / max_area) * 0.05
        candidates.append((score, cx, cy))

    if not candidates:
        return None
    score, cx, cy = max(candidates, key=lambda item: item[0])
    return cx, cy, float(score)


def detect_with_hsv(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    detections = {}
    for name, ranges in COLOUR_RANGES.items():
        result = find_cube_hsv(img_hsv, ranges)
        if result is not None:
            detections[name] = result
    return detections


def detect_pixels(img_bgr):
    fallback = detect_with_hsv(img_bgr)
    if not MODEL_PATH.exists():
        print(f"Neural model not found at {MODEL_PATH}; using HSV fallback.")
        return fallback, "hsv"

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model = load_model(MODEL_PATH)
    neural = predict_pixels(model, rgb, threshold=0.45)

    # Keep the pipeline robust while the model is still being trained.
    merged = dict(neural)
    for name, detection in fallback.items():
        merged.setdefault(name, detection)
    return merged, "neural+fallback" if len(merged) != len(neural) else "neural"


def draw_detections(img_bgr, detections, output_path="detection_result.png"):
    display = img_bgr.copy()
    for name, values in detections.items():
        cx, cy = values[:2]
        colour = MARKER_BGR[name]
        cv2.circle(display, (int(cx), int(cy)), 12, colour, 2)
        cv2.putText(display, name, (int(cx) + 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.imwrite(output_path, display)


def main():
    img_bgr = cv2.imread("scene_view.png")
    if img_bgr is None:
        raise FileNotFoundError("scene_view.png not found. Run 03_camera_capture.py first.")

    detections, method = detect_pixels(img_bgr)
    draw_detections(img_bgr, detections)

    cam = load_camera_params("camera_params.npz")
    world = pixels_to_world(detections, cam)

    print(f"\nDetection method: {method}")
    print("World coordinates (estimated from pixel positions):")
    for name in ("red", "green", "blue"):
        if name not in world:
            print(f"  {name:5s}: not detected")
            continue
        x, y, z = world[name]
        score = detections[name][2] if len(detections[name]) > 2 else 1.0
        print(f"  {name:5s}: world ({x:.3f}, {y:.3f}, {z:.3f}), confidence={score:.2f}")


if __name__ == "__main__":
    main()
