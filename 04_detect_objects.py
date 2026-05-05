"""
04_detect_objects.py - Find the coloured cubes in the camera image.

This is the FIRST AI-ish component, though it's actually classical CV
(no learning yet). HSV colour thresholding finds blobs of red/green/blue.
The output is each cube's pixel coordinates in the image.

Run with:
    python 04_detect_objects.py

Input:
    scene_view.png      (from 03_camera_capture.py)
    camera_params.npz   (from 03_camera_capture.py)

Output:
    detection_result.png  (the input image with detected cubes circled)
    Prints (pixel_x, pixel_y) for each colour found.

Why HSV not RGB?
    Brightness varies depending on lighting and camera angle.
    HSV separates colour (Hue) from brightness (Value), so a "red" cube
    is still detectable even if part of it is in shadow.
"""

import cv2
import numpy as np

# --- Load the captured camera image ---
img_bgr = cv2.imread("scene_view.png")
if img_bgr is None:
    raise FileNotFoundError(
        "scene_view.png not found. Run 03_camera_capture.py first."
    )

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# --- HSV ranges for each colour ---
# Hue is 0-179 in OpenCV (not 0-359). Saturation and Value are 0-255.
# Red wraps around 0/180, so it needs two ranges.
COLOUR_RANGES = {
    "red":   [(np.array([0,   100, 80]),  np.array([10,  255, 255])),
              (np.array([170, 100, 80]),  np.array([179, 255, 255]))],
    "green": [(np.array([40,  80,  40]),  np.array([85,  255, 255]))],
    "blue":  [(np.array([100, 100, 60]),  np.array([130, 255, 255]))],
}


def find_cube(img_hsv, colour_ranges, min_area=50):
    """Returns (cx, cy, area) of the largest blob matching the colour, or None."""
    mask = None
    for low, high in colour_ranges:
        m = cv2.inRange(img_hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    
    # Clean up speckle noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected blobs
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


# --- Detect each colour ---
detections = {}
display = img_bgr.copy()

print("Detection results:")
for name, ranges in COLOUR_RANGES.items():
    result = find_cube(img_hsv, ranges)
    if result is None:
        print(f"  {name:5s}: NOT FOUND")
        continue
    cx, cy, area = result
    detections[name] = (cx, cy)
    print(f"  {name:5s}: pixel ({cx}, {cy})  area={int(area)}")
    
    # Draw a circle on the output image
    bgr_marker = {
        "red":   (0, 0, 255),
        "green": (0, 255, 0),
        "blue":  (255, 0, 0),
    }[name]
    cv2.circle(display, (cx, cy), 12, bgr_marker, 2)
    cv2.putText(display, name, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, bgr_marker, 2)

cv2.imwrite("detection_result.png", display)
print("\nSaved detection_result.png with detections marked.")


# --- Convert pixel coordinates to world coordinates ---
# This is the bridge from "vision says cube is at pixel (320, 200)"
# to "the arm should reach to world position (0.5, 0.0, 0.65)"
#
# Because our camera looks straight down at a known table height, this is
# a simple linear mapping. For oblique cameras you'd need full intrinsics
# and depth — but straight-down keeps it tractable for v1.

cam = np.load("camera_params.npz")
cam_eye = cam["cam_eye"]
cam_target = cam["cam_target"]
fov = float(cam["fov"])
W = int(cam["width"])
H = int(cam["height"])

# Distance from camera to table surface
height_above_table = float(cam_eye[2] - cam_target[2])

# At given FOV and distance, work out how many metres each pixel represents
fov_rad = np.deg2rad(fov)
view_height_m = 2 * height_above_table * np.tan(fov_rad / 2)
view_width_m = view_height_m * (W / H)
m_per_pixel_x = view_width_m / W
m_per_pixel_y = view_height_m / H

print("\nWorld coordinates (estimated from pixel positions):")
for name, (cx, cy) in detections.items():
    # Pixel offset from image centre
    dx_px = cx - W / 2
    dy_px = cy - H / 2
    # Map to world. Camera up vector is [1,0,0] meaning image-up = world +X.
    # So image-y (down) = world -X, image-x (right) = world +Y.
    world_x = float(cam_target[0]) - dy_px * m_per_pixel_y
    world_y = float(cam_target[1]) + dx_px * m_per_pixel_x
    world_z = float(cam_target[2])  # cube sits on table
    print(f"  {name:5s}: world ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")
    
print("\nCompare these against the ground-truth positions printed by")
print("03_camera_capture.py to see how accurate the vision is.")
