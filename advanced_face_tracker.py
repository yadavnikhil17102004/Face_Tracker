import cv2
import time
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from head_pose_estimator import HeadPoseEstimator

"""
Advanced Face Tracking Application

This program extends the basic face tracker with additional features:
- Eye detection within faces
- Screenshot capture functionality
- More detailed statistics
- Additional configuration options

Configuration parameters are at the top of the file for easy customization.
"""

# Load environment variables from .env file
load_dotenv()

# Configuration parameters - modify these to customize the application
CONFIG = {
    # Camera settings
    'camera_id': 0,                # Camera ID (usually 0 for built-in webcam)
    'frame_width': 640,            # Width of the camera frame
    'frame_height': 480,           # Height of the camera frame
    'flip_horizontal': True,       # Flip the camera horizontally (mirror mode)
    
    # Face detection settings
    'face_box_color': (0, 255, 0),  # Color of face bounding box (BGR format)
    'face_box_thickness': 2,        # Thickness of bounding box lines
    'min_face_size': (30, 30),      # Minimum face size to detect
    'scale_factor': 1.1,            # How much the image size is reduced at each image scale
    'min_neighbors': 5,              # How many neighbors each candidate rectangle should have
    
    # Eye detection settings
    'detect_eyes': True,           # Whether to detect eyes within faces
    'eye_box_color': (255, 0, 0),  # Color of eye bounding box (BGR format)
    'eye_box_thickness': 1,        # Thickness of eye bounding box lines
    
    # Head pose estimation settings
    'enable_head_pose': True,      # Whether to enable head pose estimation
    'show_pose_axes': True,        # Whether to show pose axes
    'show_looking_status': True,   # Whether to show if face is looking at camera
    'landmark_model': 'models/shape_predictor_68_face_landmarks.dat',  # Path to dlib landmark model
    
    # Display settings
    'show_fps': True,              # Whether to display FPS counter
    'show_help': True,             # Whether to display help text
    'text_color': (0, 0, 255),     # Color of text overlays (BGR format)
    'text_size': 0.7,              # Size of text overlays
    'text_thickness': 2,           # Thickness of text overlays
    
    # Screenshot settings
    'screenshot_dir': 'screenshots',  # Directory to save screenshots
    'screenshot_format': 'jpg',      # Format to save screenshots (jpg or png)
    
    # Face count warning settings (loaded from .env file)
    'max_face_count': int(os.getenv('MAX_FACE_COUNT', 1)),  # Maximum number of faces before showing warning
    'face_count_warning_message': os.getenv('FACE_COUNT_WARNING_MESSAGE', 'Warning: Too many faces detected!'),
    'warning_text_color': (
        int(os.getenv('WARNING_TEXT_COLOR_B', 0)),
        int(os.getenv('WARNING_TEXT_COLOR_G', 0)),
        int(os.getenv('WARNING_TEXT_COLOR_R', 255))
    ),
}


def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def take_screenshot(frame, directory=CONFIG['screenshot_dir'], format=CONFIG['screenshot_format']):
    """Save a screenshot of the current frame."""
    ensure_dir(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/face_tracker_{timestamp}.{format}"
    cv2.imwrite(filename, frame)
    return filename


def draw_text(frame, text, position, color=CONFIG['text_color'], 
              size=CONFIG['text_size'], thickness=CONFIG['text_thickness']):
    """Draw text on the frame with consistent formatting."""
    cv2.putText(
        frame, 
        text, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        size, 
        color, 
        thickness
    )


def main():
    # Load the pre-trained face and eye detectors from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Check if the cascades loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return
    
    if CONFIG['detect_eyes'] and eye_cascade.empty():
        print("Warning: Could not load eye cascade classifier, eye detection disabled")
        CONFIG['detect_eyes'] = False
    
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
    
    print("Advanced Face Tracker started.")
    print("Controls:")
    print("  'q' - Quit the application")
    print("  's' - Take a screenshot")
    print("  'h' - Toggle help text")
    print("  'e' - Toggle eye detection")
    print("  'p' - Toggle head pose estimation")
    print("  'a' - Toggle pose axes display")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Variables for tracking statistics
    total_faces_detected = 0
    max_faces_in_frame = 0
    
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG['scale_factor'],
            minNeighbors=CONFIG['min_neighbors'],
            minSize=CONFIG['min_face_size']
        )
        
        # Update statistics
        face_count = len(faces)
        total_faces_detected += face_count
        max_faces_in_frame = max(max_faces_in_frame, face_count)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                CONFIG['face_box_color'], 
                CONFIG['face_box_thickness']
            )
            
            # Detect eyes if configured
            if CONFIG['detect_eyes']:
                # Extract the region of interest (face area)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Draw rectangles around detected eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, 
                        (ex, ey), 
                        (ex + ew, ey + eh), 
                        CONFIG['eye_box_color'], 
                        CONFIG['eye_box_thickness']
                    )
            
            # Estimate head pose if enabled
            if CONFIG['enable_head_pose'] and head_pose_estimator is not None:
                try:
                    # Pass the face rectangle to the head pose estimator
                    face_rect = (x, y, w, h)
                    
                    # Get facial landmarks
                    landmarks = head_pose_estimator.get_landmarks(gray, face_rect)
                    
                    if landmarks is not None:
                        # Estimate head pose
                        success, rotation_vector, translation_vector, euler_angles = \
                            head_pose_estimator.get_pose(gray, landmarks)
                        
                        if success and rotation_vector is not None and translation_vector is not None:
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
                except Exception as e:
                    # Silently continue if pose estimation fails for this frame
                    pass
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display information on the frame
        y_position = 30  # Starting y position for text
        
        # Display FPS
        if CONFIG['show_fps']:
            draw_text(frame, f"FPS: {fps:.1f}", (10, y_position))
            y_position += 30
        
        # Display face count
        draw_text(frame, f"Faces: {face_count}", (10, y_position))
        y_position += 30
        
        # Display max faces detected
        draw_text(frame, f"Max Faces: {max_faces_in_frame}", (10, y_position))
        y_position += 30
        
        # Display warning if face count exceeds the configured limit
        if face_count > CONFIG['max_face_count']:
            draw_text(
                frame,
                CONFIG['face_count_warning_message'],
                (10, y_position),
                color=CONFIG['warning_text_color'],
                size=CONFIG['text_size'] * 1.2,  # Make warning slightly larger
                thickness=CONFIG['text_thickness']
            )
            y_position += 30
        
        # Display help text if configured
        if CONFIG['show_help']:
            help_y = CONFIG['frame_height'] - 120
            draw_text(frame, "Controls:", (10, help_y), size=0.5)
            draw_text(frame, "q: Quit  |  s: Screenshot", (10, help_y + 25), size=0.5)
            draw_text(frame, "h: Help  |  e: Eyes  |  p: Pose  |  a: Axes", (10, help_y + 50), size=0.5)
        
        # Display the resulting frame
        cv2.imshow('Advanced Face Tracker', frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Take screenshot
            filename = take_screenshot(frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('h'):  # Toggle help text
            CONFIG['show_help'] = not CONFIG['show_help']
        elif key == ord('e'):  # Toggle eye detection
            CONFIG['detect_eyes'] = not CONFIG['detect_eyes']
            status = "enabled" if CONFIG['detect_eyes'] else "disabled"
            print(f"Eye detection {status}")
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
    
    # Display final statistics
    print("\nFace Tracker Statistics:")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"Maximum faces in a single frame: {max_faces_in_frame}")
    print("Advanced Face Tracker stopped.")


if __name__ == "__main__":
    main()