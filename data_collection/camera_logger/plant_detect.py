import cv2
import os
import time
from datetime import datetime
import json
import numpy as np
from ultralytics import YOLO

# =====================================================
# YOLO MODEL INITIALIZATION
# =====================================================

MODEL_PATH = "models/plant_detector.pt"
CONFIDENCE = 0.45

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# =====================================================
# RTSP CAMERA URL
# =====================================================

RTSP_URL="rtsp://admin:Techup%40132@192.168.100.35:554/cam/realmonitor?channel=2&subtype=0"

# Example:
# RTSP_URL = "rtsp://admin:password%40192.168.1.100:554/Streaming/Channels/101"

# =====================================================
# SAVE LOCATION
# =====================================================

SAVE_FOLDER = r"D:\Hydroponics Dataset\Original"
OUTPUT_ROOT = "data"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =====================================================
# CAPTURE INTERVAL
# 30 minutes = 1800 seconds
# =====================================================

CAPTURE_INTERVAL = 120

# =====================================================
# CONNECT CAMERA
# =====================================================

print("Connecting to CCTV Camera...")

cap = cv2.VideoCapture(RTSP_URL)

while not cap.isOpened():

    print("Unable to connect.")
    print("Retrying in 10 seconds...")

    time.sleep(10)

    cap = cv2.VideoCapture(RTSP_URL)

print("Camera Connected Successfully")

# =====================================================
# LAST CAPTURE TIME
# =====================================================

last_capture = 0

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    # ---------------------------------------------
    # Camera disconnected
    # ---------------------------------------------

    if not ret:

        print("Camera connection lost.")

        cap.release()

        time.sleep(5)

        cap = cv2.VideoCapture(RTSP_URL)

        continue

    # ---------------------------------------------
    # Current Time
    # ---------------------------------------------

    current_time = time.time()

    # ---------------------------------------------
    # Capture every interval
    # ---------------------------------------------

    if current_time - last_capture >= CAPTURE_INTERVAL:

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(
            SAVE_FOLDER,
            timestamp_str + ".jpg"
        )

        cv2.imwrite(filename, frame)

        print("Image Saved :", filename)

        # ---------------------------------------------
        # YOLO Detection & Cropping
        # ---------------------------------------------
        
        print("Running YOLO detection...")
        results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)
        result = results[0]

        crop_dir = os.path.join(OUTPUT_ROOT, "plant_crops", timestamp_compact)
        os.makedirs(crop_dir, exist_ok=True)

        plant_records = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            height, width = frame.shape[:2]

            for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box.astype(int)

                # Keep coordinates inside image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                # Ignore invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                plant_id = f"plant_{i + 1:03d}"
                crop_filename = f"{plant_id}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)

                cv2.imwrite(crop_path, crop)

                plant_records.append({
                    "plant_id": plant_id,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "crop_path": crop_path
                })

        # Save Metadata
        metadata = {
            "timestamp": timestamp_compact,
            "original_image": filename,
            "plant_count": len(plant_records),
            "plants": plant_records
        }

        metadata_path = os.path.join(crop_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Detection finished. Found {len(plant_records)} plants.")

        last_capture = current_time

    # ---------------------------------------------
    # Display Live Feed
    # ---------------------------------------------

    display = frame.copy()

    cv2.putText(
        display,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.imshow("CCTV Live", display)

    # ---------------------------------------------
    # Quit
    # ---------------------------------------------

    key = cv2.waitKey(1)

    if key == ord('q'):

        break

# =====================================================
# CLEANUP
# =====================================================

cap.release()

cv2.destroyAllWindows()

print("Program Closed")