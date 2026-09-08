import cv2
import os
import time
from datetime import datetime
import numpy as np
# from Plant_detector.plant_detect import PlantDetector 

def main(camera_url):
    """
    Connects to a live camera stream, displays it, and allows the user 
    to capture frames for a dataset by pressing 'c'.
    """
    print(f"Attempting to connect to: {camera_url}")
    # Initialize the video capture with the camera URL
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {camera_url}")
        return

    print("Successfully connected to the camera stream.")
    print("---------------------------------------------")
    print("Controls:")
    print("  Press 'r' to manually draw an ROI.")
    print("  Press 'a' to auto-detect empty holes (dark circles).")
    print("  Press 'c' to clear all ROIs.")
    print("  Press 's' to capture and save the image with blue boxes around the holes.")
    print("  Press 'q' to quit.")
    print("---------------------------------------------")

    # Directory to save the captured raw dataset images
    save_dir = "datasets/raw_captures"
    os.makedirs(save_dir, exist_ok=True)

    current_rois = []

    while True:
        # Read a frame from the live stream
        ret, frame = cap.read()

        if not ret:
            print("Warning: Could not read frame from stream. Reconnecting in 2 seconds...")
            time.sleep(2)
            # Attempt to reconnect
            cap.release()
            cap = cv2.VideoCapture(camera_url)
            continue

        display_frame = frame.copy()
        
        # Draw the ROIs if they exist (in Blue: 255, 0, 0)
        for i, (x, y, w, h) in enumerate(current_rois):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(display_frame, f"Hole {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the live feed in a window
        cv2.namedWindow('Hydroponic Live Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hydroponic Live Camera', 1280, 720)
        cv2.imshow('Hydroponic Live Camera', display_frame)

        # Wait for key press (1ms delay to allow OpenCV to update the window)
        key = cv2.waitKey(1) & 0xFF

        # Select ROI when 'r' is pressed
        if key == ord('r'):
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select ROI", 1280, 720)
            roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            if roi[2] > 0 and roi[3] > 0:
                current_rois.append(roi)
                print(f"[*] ROI added: {roi}")
                
        # Auto-detect when 'a' is pressed
        elif key == ord('a'):
            print("[*] Auto-detecting holes using hierarchy...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # The pipes are bright white, the holes and background are dark.
            # Threshold to get white pipes
            _, binary = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
            
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            current_rois = []
            if hierarchy is not None:
                for i, cnt in enumerate(contours):
                    # Check if the contour has a parent (meaning it's a hole inside a bright pipe)
                    parent_idx = hierarchy[0][i][3]
                    if parent_idx != -1:
                        area = cv2.contourArea(cnt)
                        # Holes are small ellipses. Filter out tiny noise and huge background areas
                        if 30 < area < 3000:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect_ratio = float(w) / h
                            
                            # Because of the camera angle, holes appear wide and short
                            if 1.0 <= aspect_ratio <= 4.5:
                                pad = 10
                                x1 = max(0, x - pad)
                                y1 = max(0, y - pad)
                                x2 = min(frame.shape[1], x + w + pad)
                                y2 = min(frame.shape[0], y + h + pad)
                                current_rois.append((x1, y1, x2 - x1, y2 - y1))
            
            # Sort the ROIs from top to bottom, left to right for consistent numbering
            # Group rows by rounding the y-coordinate to the nearest 40 pixels
            current_rois.sort(key=lambda r: (round(r[1] / 40) * 40, r[0]))
            
            print(f"[*] Auto-detected {len(current_rois)} holes.")
            
        # Clear ROIs when 'c' is pressed
        elif key == ord('c'):
            current_rois = []
            print("[*] All ROIs cleared. Will capture full frames.")

        # Capture frame when 's' is pressed
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not current_rois:
                # Save full frame
                image_filename = f"hole_capture_{timestamp}.jpg"
                image_path = os.path.join(save_dir, image_filename)
                cv2.imwrite(image_path, frame)
                print(f"[*] Full frame captured and saved to: {image_path}")
            else:
                # Create a copy of the frame to draw blue boxes on
                boxed_frame = frame.copy()
                for i, (x, y, w, h) in enumerate(current_rois):
                    cv2.rectangle(boxed_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(boxed_frame, f"Hole {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Save the full frame with blue boxes drawn
                image_filename = f"hole_capture_{timestamp}_boxed.jpg"
                image_path = os.path.join(save_dir, image_filename)
                cv2.imwrite(image_path, boxed_frame)
                print(f"[*] Captured frame with blue boxes saved to: {image_path}")
                
                # Save each ROI crop individually for training
                for i, (x, y, w, h) in enumerate(current_rois):
                    crop = frame[y:y+h, x:x+w]
                    crop_filename = f"hole_capture_{timestamp}_roi{i+1}.jpg"
                    crop_path = os.path.join(save_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
            
        # Exit when 'q' is pressed
        elif key == ord('q'):
            print("Exiting camera stream...")
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- CONFIGURE YOUR CAMERA URL HERE ---
    
    # Example 1: Local USB Webcam (usually 0 or 1)
    # CAMERA_URL = 0
    
    # Example 2: HTTP Stream from a Raspberry Pi (e.g. motion or mjpg-streamer)
    # CAMERA_URL = "http://192.168.1.100:8080/?action=stream"
    # CAMERA_URL = "http://192.168.1.100:8081"
    
    # Example 3: RTSP Stream from an IP Camera
    # CAMERA_URL = "rtsp://username:password@192.168.1.101:554/stream1"
    
    # Please replace the URL below with the actual URL/IP of your hosted server camera
    CAMERA_URL = "rtsp://admin:Techup%40132@192.168.100.35:554/cam/realmonitor?channel=1&subtype=0"
    main(CAMERA_URL)