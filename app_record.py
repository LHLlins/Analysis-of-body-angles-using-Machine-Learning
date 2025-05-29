"""
Records video from the default camera (webcam) and saves it to an AVI file.

This script captures video from the system's default camera (index 0),
displays the live feed in a window titled 'Recording...', and saves the
video to 'video_output.avi' using the XVID codec at 20 FPS.
The resolution of the output video matches the camera's default resolution.

Press the 'ESC' key to stop recording and close the application.
"""
import cv2

def record_video():
    """
    Main function to capture, display, and record video from the default camera.
    """
    # Initialize video capture from the default camera (index 0)
    video_capture_device = cv2.VideoCapture(0)

    # Check if the camera was opened successfully
    if not video_capture_device.isOpened():
        print("Error: Could not open camera.")
        return

    # Get frame width and height from the camera
    # These will be used for the VideoWriter to ensure matching resolution
    frame_width = int(video_capture_device.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture_device.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec (XVID is common for AVI) and create VideoWriter object
    # 20.0 is the frames per second (FPS).
    fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
    output_video_writer = cv2.VideoWriter(
        'video_output.avi',  # Output filename
        fourcc_codec,
        20.0,                # Frames per second
        (frame_width, frame_height) # Frame size
    )

    # Check if the VideoWriter was created successfully
    if not output_video_writer.isOpened():
        print("Error: Could not open video writer. Ensure XVID codec is available and "
              "check file permissions and path.")
        video_capture_device.release() # Release the camera if writer creation failed
        return

    print(f"Recording video at {frame_width}x{frame_height} resolution, 20 FPS...")
    print("Press 'ESC' to stop recording.")

    try:
        while video_capture_device.isOpened(): # Loop as long as the camera is accessible
            # Read a frame from the camera
            # is_frame_successfully_read is a boolean
            # current_video_frame is the captured video frame
            is_frame_successfully_read, current_video_frame = video_capture_device.read()

            if is_frame_successfully_read:
                # Display the resulting frame in a window
                cv2.imshow('Recording...', current_video_frame)

                # Write the frame to the output video file
                output_video_writer.write(current_video_frame)

                # Wait for a key press for 1 ms.
                # If 'ESC' key (ASCII 27) is pressed, break the loop.
                if cv2.waitKey(1) & 0xFF == 27: # 27 is the ASCII code for ESC
                    print("Recording stopped by user.")
                    break
            else:
                # If a frame was not read successfully, log an error and break
                print("Error: Could not read frame from camera. Stopping recording.")
                break
    finally:
        # Release all resources
        print("Releasing resources...")
        video_capture_device.release()
        output_video_writer.release()
        cv2.destroyAllWindows()
        print(f"Video saved as video_output.avi")

if __name__ == "__main__":
    record_video()