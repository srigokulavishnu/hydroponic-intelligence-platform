import cv2
import json
import os

# =====================================================
# CONFIGURATION
# =====================================================
RTSP_URL = "rtsp://admin:Techup%40132@192.168.100.35:554/cam/realmonitor?channel=2&subtype=0"
ROI_FILE = "rois.json"

def main():
    print("Connecting to camera to capture a setup frame...")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Failed to connect to the camera.")
        return

    # Read a few frames to let the camera adjust (auto-exposure/focus)
    for _ in range(10):
        ret, frame = cap.read()
        
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame.")
        return

    print("Frame captured successfully.")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Click and drag to draw a box around a plant hole.")
    print("2. Press SPACE or ENTER to confirm that box.")
    print("3. Repeat for all plant holes.")
    print("4. Press ESC when you are finished selecting all ROIs.")
    print("5. Press 'c' to cancel the current selection.")
    print("--------------------------------------------------")

    # Select multiple ROIs
    # Returns a numpy array of [x, y, w, h]
    rois = cv2.selectROIs("Select ROIs (Areas of Interest)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if not len(rois):
        print("No ROIs were selected. Exiting.")
        return

    # Convert to list of dictionaries for JSON
    roi_data = []
    for i, (x, y, w, h) in enumerate(rois):
        roi_info = {
            "id": f"plant_hole_{i+1:03d}",
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h)
        }
        roi_data.append(roi_info)

    # Save to JSON file
    with open(ROI_FILE, "w") as f:
        json.dump(roi_data, f, indent=4)

    print(f"Successfully saved {len(roi_data)} ROIs to {ROI_FILE}")

    # Draw the saved ROIs on the frame and save an example image
    for roi in roi_data:
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, roi["id"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    preview_file = "roi_preview.jpg"
    cv2.imwrite(preview_file, frame)
    print(f"Saved a preview image with drawn ROIs to {preview_file}")

if __name__ == "__main__":
    main()