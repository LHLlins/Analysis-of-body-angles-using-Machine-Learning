"""
Main application for real-time pose classification from a video source.

This script uses the PoseEstimator and PoseClassifier classes from the 
`pipline.py` module to detect human poses in video frames and classify them 
into predefined categories (e.g., T-Pose, Warrior II, Tree Pose). 
It displays the processed video feed with landmarks, classified pose labels, 
and FPS.

Video Source:
- To use a webcam, set VIDEO_SOURCE = 0.
- To use a video file, set VIDEO_SOURCE = "path/to/your/video.avi".
"""
import cv2
import time
import numpy as np # Retained for frame manipulation if needed in future
from pipline import PoseEstimator, PoseClassifier

# --- Configuration Constants ---
VIDEO_SOURCE = 'videoFinal.avi'  # or 0 for webcam
# VIDEO_SOURCE = 0 # Example for webcam
WINDOW_NAME = 'Pose Classification'
TARGET_FRAME_HEIGHT = 640  # Target height for frame resizing
FPS_TEXT_COLOR = (203, 52, 247) # BGR color for FPS text
ESC_KEY = 27 # ASCII code for Escape key

def run_pose_classification():
    """
    Initializes video capture, processes frames for pose estimation and 
    classification, and displays the results.
    """
    # Instantiate the pose estimator and classifier
    pose_estimator = PoseEstimator(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    pose_classifier = PoseClassifier()

    # Initialize VideoCapture
    camera_video = cv2.VideoCapture(VIDEO_SOURCE)
    if not camera_video.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
        return

    prev_time = 0 # Used for FPS calculation

    # Main loop to process video frames
    while camera_video.isOpened():
        # Read a frame from the video source
        is_frame_read, frame = camera_video.read()

        # Check if frame was read successfully
        if not is_frame_read:
            print("Info: End of video file or stream.")
            break # Exit loop if no frame is read (e.g., end of video)

        # Flip the frame horizontally for a natural selfie-view display
        frame = cv2.flip(frame, 1)

        # Resize the frame while maintaining aspect ratio
        frame_height, frame_width, _ = frame.shape
        scale_ratio = TARGET_FRAME_HEIGHT / frame_height
        target_width = int(frame_width * scale_ratio)
        frame = cv2.resize(frame, (target_width, TARGET_FRAME_HEIGHT))
        
        # Start timer for FPS calculation
        current_time = time.time()

        # Perform pose landmark detection
        # landmarks_dict will contain 'pose', 'left_hand', 'right_hand' landmarks
        frame, landmarks_dict = pose_estimator.detectPose(frame, display=False)
        
        # Perform pose classification if landmarks are detected
        # The classifyPose method internally handles cases where specific landmarks might be None
        if landmarks_dict: 
            frame, _ = pose_classifier.classifyPose(landmarks_dict, frame, display=False)

        # Calculate FPS
        if prev_time > 0: # Avoid division by zero on the first frame
            fps = 1 / (current_time - prev_time)
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, FPS_TEXT_COLOR, 1)
        prev_time = current_time
        
        # Display the processed frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle key presses
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ESC_KEY:
            print("Info: 'ESC' key pressed. Exiting...")
            break # Exit loop if 'ESC' is pressed
    
    # Release resources
    camera_video.release()
    cv2.destroyAllWindows()
    print("Info: Video stream closed and resources released.")

if __name__ == "__main__":
    run_pose_classification()