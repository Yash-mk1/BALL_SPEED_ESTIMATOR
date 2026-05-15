import cv2
import numpy as np


VIDEO_PATH = "vids/ball_sample_video.mp4"
REAL_DISTANCE = 20.12  # Cricket pitch length in meters


# GLOBALS


calibration_points = []       
meters_per_pixel = None     
positions = []               
frame_count = 0


def mouse_callback(event, x, y, flags, param):
    """Records up to 2 mouse clicks for calibration."""
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 2:
        calibration_points.append((x, y))
        print(f"  Point {len(calibration_points)} selected: ({x}, {y})")


def calibrate(frame):
    """
    Shows the first frame and asks user to click 2 known real-world points.
    Returns meters_per_pixel ratio.
    """
    global calibration_points
    calibration_points = []

    clone = frame.copy()
    cv2.namedWindow("Calibrate - Click 2 points (e.g. both ends of pitch)")
    cv2.setMouseCallback("Calibrate - Click 2 points (e.g. both ends of pitch)", mouse_callback)

    print("\n[CALIBRATION] Click on 2 known real-world points (e.g. both stumps).")
    print("              Press ENTER when done.\n")

    while True:
        display = clone.copy()

        for i, pt in enumerate(calibration_points):
            cv2.circle(display, pt, 6, (0, 255, 255), -1)
            cv2.putText(display, f"P{i+1}", (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(calibration_points) == 2:
            cv2.line(display, calibration_points[0], calibration_points[1], (0, 255, 255), 2)

        cv2.imshow("Calibrate - Click 2 points (e.g. both ends of pitch)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(calibration_points) == 2:  # ENTER key
            break
        elif key == 27:  # ESC to quit
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Calibrate - Click 2 points (e.g. both ends of pitch)")

   
    p1, p2 = calibration_points
    pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    ratio = REAL_DISTANCE / pixel_dist

    print(f"  Pixel distance between points: {pixel_dist:.2f} px")
    print(f"  Meters per pixel: {ratio:.6f} m/px\n")
    return ratio




def is_circular(contour, min_circularity=0.6):
    """
    Returns True if the contour is roughly circular.
    Circularity = 4π × Area / Perimeter²
    Perfect circle → 1.0 | Irregular shape → much lower
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity >= min_circularity


def detect_ball(frame, bg_subtractor):
    """
    Detects the ball using background subtraction + circularity filter.
    Returns (cx, cy) if found, else None.
    """
    # Apply background subtractor to get foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Threshold: keep only high-confidence foreground pixels
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Remove noise with morphological opening
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter: must be large enough AND roughly circular
        if area > 30 and is_circular(cnt, min_circularity=0.6):
            if area > best_area:
                best_area = area
                best = cnt

    if best is not None:
        x, y, w, h = cv2.boundingRect(best)
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy), fg_mask

    return None, fg_mask


def calculate_speed(positions, fps, meters_per_pixel):
    """
    Calculates speed from consecutive position pairs.
    Returns median speed in km/h (robust to outliers/false detections).
    """
    if len(positions) < 2:
        return None, []

    speeds = []

    for i in range(1, len(positions)):
        f1, x1, y1 = positions[i - 1]
        f2, x2, y2 = positions[i]

        frame_diff = f2 - f1
        if frame_diff == 0:
            continue

        # Euclidean pixel distance between consecutive detections
        pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        real_dist = pixel_dist * meters_per_pixel

        time_diff = frame_diff / fps
        speed_mps = real_dist / time_diff
        speed_kmph = speed_mps * 3.6

        speeds.append(speed_kmph)

    if not speeds:
        return None, []

   
    median_speed = float(np.median(speeds))
    return median_speed, speeds




cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] FPS: {fps}")

# Read first frame for calibration
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] Cannot read video.")
    exit()

first_frame = cv2.resize(first_frame, (800, 600))

# --- Calibrate ---
meters_per_pixel = calibrate(first_frame)

# --- Background subtractor (MOG2) ---
# history=100: learns background from first 100 frames
# varThreshold=50: sensitivity (lower = more sensitive)
# detectShadows=False: we don't need shadow detection, keeps it clean
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=50,
    detectShadows=False
)


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Process video ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (800, 600))

    ball_pos, fg_mask = detect_ball(frame, bg_subtractor)

    if ball_pos:
        cx, cy = ball_pos
        positions.append((frame_count, cx, cy))
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(frame, "Ball", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show calibration points on every frame
    if len(calibration_points) == 2:
        cv2.line(frame, calibration_points[0], calibration_points[1], (0, 255, 255), 1)

    cv2.imshow("Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#Speed output 
print("\n--- RESULT ---")

if len(positions) >= 2:
    median_speed, all_speeds = calculate_speed(positions, fps, meters_per_pixel)

    total_frames = positions[-1][0] - positions[0][0]
    time_taken = total_frames / fps

    print(f"Ball detected in {len(positions)} frames")
    print(f"Time tracked: {time_taken:.3f} sec")
    print(f"Per-segment speeds (km/h): {[round(s, 2) for s in all_speeds]}")
    print(f"\n>>> Estimated Speed (median): {median_speed:.2f} km/h <<<")

else:
    print("Ball not detected properly.")
    print("Tips:")
    print("  - Ensure the ball is moving (background subtraction needs motion)")
    print("  - Try adjusting varThreshold in createBackgroundSubtractorMOG2")
    print("  - Check that the video file path is correct")
