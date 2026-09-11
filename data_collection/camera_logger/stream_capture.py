import cv2
import os
import time
import json
from datetime import datetime
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

# Metadata
SECTION_ID = "S01"
CHANNEL_ID = "C01"
CROP_NAME = "Palak"

# Hole Detection Thresholds
MIN_HOLE_AREA = 250        # Increased to ignore small background debris and floor tiles
MAX_HOLE_AREA = 8000
MIN_CIRCULARITY = 0.55     # Increased significantly to reject floor rectangles and lines, allowing only circles/ellipses
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 4.0
MIN_DARKNESS = 130 # Maximum pixel value for thresholding (dark objects)

# Target Warped Size (Landscape aspect ratio for a horizontal pipe)
WARP_WIDTH = 1500
WARP_HEIGHT = 250

# Crop Configuration
CROP_PADDING = 15

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
HOLE_MAP_FILE = os.path.join(CONFIG_DIR, "hole_map.json")
CAPTURE_DIR = os.path.join(BASE_DIR, "datasets", "captures")

# State Management
roi_points = []
calibrated_holes = []  # List of dicts with id, x, y, w, h
app_state = "IDLE"  # IDLE, SELECT_ROI, CALIBRATING

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def mouse_callback(event, x, y, flags, param):
    global roi_points, app_state
    if app_state == "SELECT_ROI" and event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            roi_points.append((x, y))
            print(f"[*] Point {len(roi_points)} selected at ({x}, {y})")
        if len(roi_points) == 4:
            app_state = "CALIBRATING"
            print("[*] 4 points selected. Press 'a' to auto-detect holes in the target region.")

def order_points(pts):
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # Grab the left-most and right-most points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # Sort the left-most coordinates according to their y-coordinates
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # Sort the right-most coordinates according to their y-coordinates
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")

def enhance_image(img):
    """Enhances clarity and contrast using CLAHE and unsharp masking."""
    # Convert to LAB space to equalize Lightness without messing up colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Unsharp mask for sharpness
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def warp_channel(frame, pts):
    rect = order_points(np.array(pts, dtype="float32"))
    dst = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (WARP_WIDTH, WARP_HEIGHT))
    return warped

def detect_holes_and_map(frame, pts):
    # 1. Get perspective matrix M to map coordinates later
    rect = order_points(np.array(pts, dtype="float32"))
    dst = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)

    # 2. Run EXACT old detection on the original full frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, MIN_DARKNESS, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    holes_warped = []
    poly_pts = np.array(pts, np.int32)
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            # hierarchy[0][i][3] != -1 means it's a hole inside another contour
            if parent_idx != -1:
                area = cv2.contourArea(cnt)
                if MIN_HOLE_AREA < area < MAX_HOLE_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    
                    if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * (area / (perimeter * perimeter))
                            if circularity > MIN_CIRCULARITY:
                                # 3. Check if hole center is inside the selected 4-point ROI
                                cx = x + w / 2.0
                                cy = y + h / 2.0
                                if cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0:
                                    # 4. Map the bounding box to the warped image coordinates
                                    pad = 10
                                    x1 = max(0, x - pad)
                                    y1 = max(0, y - pad)
                                    x2 = min(frame.shape[1], x + w + pad)
                                    y2 = min(frame.shape[0], y + h + pad)
                                    
                                    corners = np.array([
                                        [x1, y1], 
                                        [x2, y1], 
                                        [x2, y2], 
                                        [x1, y2]
                                    ], dtype="float32").reshape(-1, 1, 2)
                                    
                                    warped_corners = cv2.perspectiveTransform(corners, M)
                                    wx, wy, ww, wh = cv2.boundingRect(warped_corners)
                                    
                                    # FIX: Clamping to strictly avoid Invalid Dimensions
                                    wx = int(max(0, min(wx, WARP_WIDTH - 1)))
                                    wy = int(max(0, min(wy, WARP_HEIGHT - 1)))
                                    ww = int(min(ww, WARP_WIDTH - wx))
                                    wh = int(min(wh, WARP_HEIGHT - wy))
                                    
                                    # Only add if the resulting box is valid
                                    if ww > 5 and wh > 5:
                                        holes_warped.append((wx, wy, ww, wh))
                                        
    # 5. Sort top-to-bottom (roughly via 60px buckets), then left-to-right for proper sequential labeling
    holes_warped.sort(key=lambda r: (round(r[1] / 60) * 60, r[0]))
    
    return holes_warped

def load_config():
    global roi_points, calibrated_holes, app_state
    if os.path.exists(HOLE_MAP_FILE):
        try:
            with open(HOLE_MAP_FILE, 'r') as f:
                data = json.load(f)
                pts = data.get("roi_points", [])
                roi_points = [(p[0], p[1]) for p in pts]
                calibrated_holes = data.get("holes", [])
            if len(roi_points) == 4 and len(calibrated_holes) > 0:
                app_state = "IDLE"
                print(f"[*] Loaded config. ROI and {len(calibrated_holes)} holes found.")
            else:
                roi_points = []
                calibrated_holes = []
        except Exception as e:
            print(f"[*] Error loading config: {e}")
            roi_points = []
            calibrated_holes = []

def save_config():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {
        "section_id": SECTION_ID,
        "channel_id": CHANNEL_ID,
        "crop": CROP_NAME,
        "roi_points": roi_points,
        "holes": calibrated_holes
    }
    with open(HOLE_MAP_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[*] Calibration saved to {HOLE_MAP_FILE}")

def save_capture(frame, warped, timestamp):
    base_dir = os.path.join("datasets", "captures", timestamp)
    crops_dir = os.path.join(base_dir, "plant_crops")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    full_frame_path = os.path.join(base_dir, "full_frame.jpg")
    target_channel_path = os.path.join(base_dir, "target_channel.jpg")
    debug_overlay_path = os.path.join(base_dir, "debug_overlay.jpg")
    metadata_path = os.path.join(base_dir, "metadata.json")
    
    # Save full frame and target channel (clean)
    cv2.imwrite(full_frame_path, frame)
    cv2.imwrite(target_channel_path, warped)
    
    # Create and save debug overlay
    debug_img = warped.copy()
    for h in calibrated_holes:
        x, y, w, h_box = h['x'], h['y'], h['w'], h['h']
        cv2.rectangle(debug_img, (x, y), (x + w, y + h_box), (255, 0, 0), 2)
        cv2.putText(debug_img, h['id'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(debug_overlay_path, debug_img)
    
    plant_records = []
    expected_count = len(calibrated_holes)
    saved_count = 0
    failed_count = 0
    
    for h in calibrated_holes:
        p_id = h['id']
        x, y, w, h_box = h['x'], h['y'], h['w'], h['h']
        
        # Calculate square dimensions based on the larger side
        side = max(w, h_box) + 2 * CROP_PADDING
        half_side = side // 2
        
        center_x = x + w // 2
        center_y = y + h_box // 2
        
        x1 = max(0, center_x - half_side)
        y1 = max(0, center_y - half_side)
        x2 = min(WARP_WIDTH, center_x + half_side)
        y2 = min(WARP_HEIGHT, center_y + half_side)
        
        crop_path = os.path.join(crops_dir, f"{p_id}_{timestamp}.jpg")
        
        save_status = "failed"
        if x2 > x1 and y2 > y1:
            crop = warped[y1:y2, x1:x2]
            
            # Ensure crop is a perfect square (pad with black if it hits the image edge)
            h_crop, w_crop = crop.shape[:2]
            if h_crop != side or w_crop != side:
                square_crop = np.zeros((side, side, 3), dtype=np.uint8)
                x_off = (side - w_crop) // 2
                y_off = (side - h_crop) // 2
                square_crop[y_off:y_off+h_crop, x_off:x_off+w_crop] = crop
                crop = square_crop
                
            # Enhance the crop for visibility and clarity
            crop = enhance_image(crop)
            
            # Resize to standard dimensions for ML consistency (e.g. 224x224)
            final_size = 224
            crop = cv2.resize(crop, (final_size, final_size))
            
            success = cv2.imwrite(crop_path, crop)
            if success:
                saved_count += 1
                save_status = "saved"
            else:
                failed_count += 1
                print(f"[!] Error: Failed to imwrite {crop_path}")
        else:
            failed_count += 1
            print(f"[!] Error: Invalid dimensions for {p_id} crop")
            
        plant_records.append({
            "plant_id": p_id,
            "image_path": crop_path.replace("\\", "/"),
            "bbox": {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1)
            },
            "center_x": int(center_x),
            "center_y": int(center_y),
            "capture_timestamp": timestamp,
            "save_status": save_status
        })
        
    metadata = {
        "timestamp": timestamp,
        "section_id": SECTION_ID,
        "channel_id": CHANNEL_ID,
        "crop": CROP_NAME,
        "plant_count": expected_count,
        "plants": plant_records
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n[*] Capture Summary [{timestamp}]:")
    print(f"    Expected plant crops: {expected_count}")
    print(f"    Successfully saved: {saved_count}")
    print(f"    Failed: {failed_count}")
    if saved_count != expected_count:
        print("    WARNING: Not all calibrated plant crops were saved.")
    print(f"    Data saved in {base_dir}\n")

# =====================================================
# MAIN FUNCTION
# =====================================================

def main(camera_url):
    global roi_points, calibrated_holes, app_state
    
    # Do not auto-load config on startup per user request
    # load_config()

    print(f"Attempting to connect to: {camera_url}")
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {camera_url}")
        return

    print("Successfully connected to the camera stream.")
    print("---------------------------------------------")
    print("Controls:")
    print("  'r' : Select target channel ROI (click 4 corners: TL, TR, BR, BL)")
    print("  'a' : Auto-detect holes in the target channel")
    print("  'm' : Save current holes as permanently calibrated")
    print("  'c' : Clear ROI and detections")
    print("  's' : Capture clean dataset images")
    print("  'l' : Load last saved calibration")
    print("  'q' : Quit")
    print("---------------------------------------------")

    cv2.namedWindow('Hydroponic Live Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hydroponic Live Camera', 1280, 720)
    cv2.setMouseCallback('Hydroponic Live Camera', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from stream. Reconnecting in 2 seconds...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(camera_url)
            continue

        display_frame = frame.copy()
        
        # Draw ROI polygon
        if len(roi_points) > 0:
            pts = np.array(roi_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw lines and points
            if len(roi_points) == 4:
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            else:
                cv2.polylines(display_frame, [pts], False, (0, 255, 0), 2)
            for p in roi_points:
                cv2.circle(display_frame, p, 5, (0, 0, 255), -1)

        # Draw Overlay Text
        status_text = f"Target: {SECTION_ID}/{CHANNEL_ID} | Calibrated holes: {len(calibrated_holes)} | Capture status: {'READY' if app_state == 'IDLE' and len(calibrated_holes) > 0 else 'NOT READY'}"
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        warped_display = None
        if len(roi_points) == 4:
            # Show warped perspective if 4 points are selected
            warped = warp_channel(frame, roi_points)
            warped_display = warped.copy()
            
            # Draw calibrated holes
            for h in calibrated_holes:
                x, y, w, h_box = h['x'], h['y'], h['w'], h['h']
                cv2.rectangle(warped_display, (x, y), (x + w, y + h_box), (255, 0, 0), 2)
                cv2.putText(warped_display, h['id'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.namedWindow('Warped Target Channel', cv2.WINDOW_NORMAL)
            cv2.imshow('Warped Target Channel', warped_display)

        cv2.imshow('Hydroponic Live Camera', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            roi_points = []
            calibrated_holes = []
            app_state = "SELECT_ROI"
            print("[*] Click 4 points on the main window to select the target ROI.")
            if cv2.getWindowProperty('Warped Target Channel', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Warped Target Channel')

        elif key == ord('a'):
            if len(roi_points) == 4:
                print("[*] Auto-detecting holes...")
                warped = warp_channel(frame, roi_points)
                holes = detect_holes_and_map(frame, roi_points)
                
                calibrated_holes = []
                for i, (x, y, w, h) in enumerate(holes):
                    plant_id = f"{CHANNEL_ID}_P{i+1:02d}"
                    calibrated_holes.append({
                        "id": plant_id,
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    })
                app_state = "CALIBRATING"
                print(f"[*] Detected {len(holes)} holes. Press 'm' to save this calibration.")
            else:
                print("[!] Please select 4 ROI points first by pressing 'r'.")

        elif key == ord('m'):
            if app_state == "CALIBRATING" and len(roi_points) == 4 and len(calibrated_holes) > 0:
                save_config()
                app_state = "IDLE"
                print("[*] Calibration complete. Ready to capture.")
            else:
                print("[!] Cannot save. Please detect holes first.")

        elif key == ord('c'):
            roi_points = []
            calibrated_holes = []
            app_state = "IDLE"
            if cv2.getWindowProperty('Warped Target Channel', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Warped Target Channel')
            print("[*] Selection cleared.")

        elif key == ord('s'):
            if app_state == "IDLE" and len(roi_points) == 4 and len(calibrated_holes) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                warped = warp_channel(frame, roi_points)
                save_capture(frame, warped, timestamp)
            else:
                print("[!] Not ready to capture. Please select ROI (r) -> detect (a) -> save calibration (m).")

        elif key == ord('l'):
            load_config()

        elif key == ord('q'):
            print("Exiting camera stream...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Please replace the URL below with the actual URL/IP of your hosted server camera
    CAMERA_URL = "rtsp://admin:Techup%40132@192.168.100.35:554/cam/realmonitor?channel=1&subtype=0"
    main(CAMERA_URL)