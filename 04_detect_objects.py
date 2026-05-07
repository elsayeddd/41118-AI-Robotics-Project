"""
04_detect_objects.py - Find the coloured cubes in the camera image.
"""

import cv2
import numpy as np

img_bgr = cv2.imread("scene_view.png")
if img_bgr is None:
    raise FileNotFoundError("scene_view.png not found. Run 03_camera_capture.py first.")

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

COLOUR_RANGES = {
    "red":   [(np.array([0,   100, 80]),  np.array([10,  255, 255])),
              (np.array([170, 100, 80]),  np.array([179, 255, 255]))],
    "green": [(np.array([40,  80,  40]),  np.array([85,  255, 255]))],
    "blue":  [(np.array([100, 100, 60]),  np.array([130, 255, 255]))],
}


def find_cube(img_hsv, colour_ranges, min_area=50):
    mask = None
    for low, high in colour_ranges:
        m = cv2.inRange(img_hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area


detections = {}
display = img_bgr.copy()

for name, ranges in COLOUR_RANGES.items():
    result = find_cube(img_hsv, ranges)
    if result is None:
        continue
    cx, cy, area = result
    detections[name] = (cx, cy)

    bgr_marker = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}[name]
    cv2.circle(display, (cx, cy), 12, bgr_marker, 2)
    cv2.putText(display, name, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_marker, 2)

cv2.imwrite("detection_result.png", display)

cam = np.load("camera_params.npz")
cam_eye = cam["cam_eye"]
cam_target = cam["cam_target"]
cam_up = cam["cam_up"] if "cam_up" in cam.files else np.array([1.0, 0.0, 0.0])
fov = float(cam["fov"])
W = int(cam["width"])
H = int(cam["height"])

height_above_table = float(cam_eye[2] - cam_target[2])
fov_rad = np.deg2rad(fov)
view_height_m = 2 * height_above_table * np.tan(fov_rad / 2)
view_width_m = view_height_m * (W / H)
m_per_pixel_x = view_width_m / W
m_per_pixel_y = view_height_m / H

print("\nWorld coordinates (estimated from pixel positions):")
for name, (cx, cy) in detections.items():
    dx_px = cx - W / 2
    dy_px = cy - H / 2

    if not np.allclose(cam_up, [1, 0, 0]):
        raise ValueError("Unsupported camera orientation: update projection mapping.")

    world_x = float(cam_target[0]) - dy_px * m_per_pixel_y
    world_y = float(cam_target[1]) + dx_px * m_per_pixel_x
    world_z = float(cam_target[2])
    print(f"  {name:5s}: world ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")
