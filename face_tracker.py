import cv2
import time
import os
import numpy as np
from dotenv import load_dotenv
from head_pose_estimator import HeadPoseEstimator

"""
Simple Face Tracking Application

This program uses OpenCV to detect faces in a webcam stream and draw bounding boxes around them.
It's designed to be efficient and easy to understand with minimal dependencies.

Configuration parameters are at the top of the file for easy customization.
"""

# Load environment variables from .env file
load_dotenv()

# Configuration parameters - modify these to customize the application
CONFIG = {
    'camera_id': 0,                # Camera ID (usually 0 for built-in webcam)
    'frame_width': 640,            # Width of the camera frame
    'frame_height': 480,           # Height of the camera frame
    'box_color': (0, 255, 0),      # Color of face bounding box (BGR format)
    'box_thickness': 2,            # Thickness of bounding box lines
    'min_face_size': (30, 30),     # Minimum face size to detect
    'scale_factor': 1.1,           # How much the image size is reduced at each image scale
    'min_neighbors': 5,            # How many neighbors each candidate rectangle should have
    'show_fps': True,              # Whether to display FPS counter
    'flip_horizontal': True,       # Flip the camera horizontally (mirror mode)
    
    # Head pose estimation settings
    'enable_head_pose': True,      # Whether to enable head pose estimation
    'show_pose_axes': True,        # Whether to show pose axes
    'show_looking_status': True,   # Whether to show if face is looking at camera
    'landmark_model': 'models/shape_predictor_68_face_landmarks.dat',  # Path to dlib landmark model
    
    # Face count warning settings (loaded from .env file)
    'max_face_count': int(os.getenv('MAX_FACE_COUNT', 1)),  # Maximum number of faces before showing warning
    'face_count_warning_message': os.getenv('FACE_COUNT_WARNING_MESSAGE', 'Warning: Too many faces detected!'),
    'warning_text_color': (
        int(os.getenv('WARNING_TEXT_COLOR_B', 0)),
        int(os.getenv('WARNING_TEXT_COLOR_G', 0)),
        int(os.getenv('WARNING_TEXT_COLOR_R', 255))
    ),
}


def main():
    # Load the pre-trained face detector from OpenCV
    # This uses the Haar Cascade classifier which is efficient for real-time applications
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Check if the face cascade loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    # Initialize head pose estimator if enabled
    head_pose_estimator = None
    if CONFIG['enable_head_pose']:
        try:
            # Check if the landmark model file exists
            if not os.path.exists(CONFIG['landmark_model']):
                print(f"Warning: Landmark model file not found at {CONFIG['landmark_model']}")
                print("Run download_models.py to download the required model files.")
                CONFIG['enable_head_pose'] = False
            else:
                head_pose_estimator = HeadPoseEstimator(CONFIG['landmark_model'])
                print("Head pose estimation enabled.")
        except Exception as e:
            print(f"Error initializing head pose estimator: {e}")
            CONFIG['enable_head_pose'] = False
    
    # Initialize the webcam
    cap = cv2.VideoCapture(CONFIG['camera_id'])
    
    # Set the frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Face tracker started.")
    print("Controls:")
    print("  'q' - Quit the application")
    print("  'p' - Toggle head pose estimation")
    print("  'a' - Toggle pose axes display")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        
        # Flip the frame horizontally if configured
        if CONFIG['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        
        # Convert the frame to grayscale for face detection
        # This improves performance and is required for Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size']
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                CONFIG['box_color'], 
                CONFIG['box_thickness']
            )
            
            # Estimate head pose if enabled
            if CONFIG['enable_head_pose'] and head_pose_estimator is not None:
                # Pass the face rectangle to the head pose estimator
                face_rect = (x, y, w, h)
                
                # Get facial landmarks
                landmarks = head_pose_estimator.get_landmarks(gray, face_rect)
                
                if landmarks is not None:
                    # Estimate head pose
                    success, rotation_vector, translation_vector, euler_angles = \
                        head_pose_estimator.get_pose(gray, landmarks)
                    
                    if success:
                        # Determine if face is looking at camera
                        is_looking = head_pose_estimator.is_looking_at_camera(euler_angles)
                        
                        # Draw pose information if configured
                        if CONFIG['show_pose_axes'] or CONFIG['show_looking_status']:
                            head_pose_estimator.draw_pose_info(
                                frame, 
                                rotation_vector, 
                                translation_vector, 
                                euler_angles,
                                is_looking if CONFIG['show_looking_status'] else None
                            )
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on the frame if configured
        if CONFIG['show_fps']:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame, 
                fps_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
        
        # Display the number of faces detected
        face_count = len(faces)
        faces_text = f"Faces: {face_count}"
        cv2.putText(
            frame, 
            faces_text, 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # Display warning if face count exceeds the configured limit
        if face_count > CONFIG['max_face_count']:
            warning_text = CONFIG['face_count_warning_message']
            cv2.putText(
                frame,
                warning_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                CONFIG['warning_text_color'],
                2
            )
        
        # Display the resulting frame
        cv2.imshow('Face Tracker', frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Toggle head pose estimation
            CONFIG['enable_head_pose'] = not CONFIG['enable_head_pose']
            status = "enabled" if CONFIG['enable_head_pose'] else "disabled"
            print(f"Head pose estimation {status}")
        elif key == ord('a'):  # Toggle pose axes display
            CONFIG['show_pose_axes'] = not CONFIG['show_pose_axes']
            status = "enabled" if CONFIG['show_pose_axes'] else "disabled"
            print(f"Pose axes display {status}")
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Face tracker stopped.")


if __name__ == "__main__":
    main()