import cv2
import os
import time
from datetime import datetime
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
    print("  Press 'c' to capture a frame and save it to the dataset.")
    print("  Press 'q' to quit.")
    print("---------------------------------------------")

    # Directory to save the captured raw dataset images
    save_dir = "datasets/raw_captures"
    os.makedirs(save_dir, exist_ok=True)

    # Optional: Initialize the detector if you want to process images immediately upon capture
    # detector = PlantDetector(model_path="models/plant_detector.pt")

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

        # Show the live feed in a window
        cv2.imshow('Hydroponic Live Camera', frame)

        # Wait for key press (1ms delay to allow OpenCV to update the window)
        key = cv2.waitKey(1) & 0xFF

        # Capture frame when 'c' is pressed
        if key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"plant_capture_{timestamp}.jpg"
            image_path = os.path.join(save_dir, image_filename)
            
            # Save the captured frame to disk
            cv2.imwrite(image_path, frame)
            print(f"[*] Frame captured and saved to: {image_path}")
            
            # If you want to automatically run your YOLO detection on the captured frame, 
            # uncomment the following lines:
            # print("[*] Running plant detection on captured frame...")
            # results_metadata = detector.detect_and_store(image_path, output_root="datasets/processed")
            # print(f"[*] Detection finished. Found {results_metadata['plant_count']} plants.")
            
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
    CAMERA_URL = "https://192.168.100.35/#/index/thepreview/" 
    
    main(CAMERA_URL)
