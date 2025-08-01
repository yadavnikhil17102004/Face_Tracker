import cv2
import numpy as np
# Note: This version doesn't require dlib, using OpenCV only

"""
Head Pose Estimation Module

This module provides functionality to estimate the head pose (pitch, yaw, roll)
and determine if a face is looking at the camera.

It uses OpenCV for face detection and pose estimation.
"""

# 3D model points of a standard face
# These points correspond to specific facial landmarks in 3D space
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Threshold angles (in degrees) to determine if looking at camera
THRESHOLD_YAW = 20.0    # Left-right rotation
THRESHOLD_PITCH = 20.0  # Up-down rotation
THRESHOLD_ROLL = 20.0   # Tilt


class HeadPoseEstimator:
    def __init__(self, predictor_path=None):
        """
        Initialize the head pose estimator.
        
        Args:
            predictor_path: Not used in this OpenCV-only version.
        """
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Camera matrix (will be set based on image dimensions)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        # Initialize the face landmark detector
        # We're using a simpler approach with OpenCV's face detector only
        # This version doesn't use dlib's facial landmark detection
    
    def set_camera_matrix(self, image_size):
        """
        Set the camera matrix based on image dimensions.
        
        Args:
            image_size: Tuple of (width, height) of the image
        """
        focal_length = image_size[1]
        center = (image_size[1] / 2, image_size[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
    
    def get_landmarks(self, image, face_rect=None):
        """
        Estimate facial landmarks in the image using a simplified approach.
        
        Args:
            image: Input image (grayscale)
            face_rect: Optional rectangle of the face (x, y, w, h)
                      If None, face detection will be performed
        
        Returns:
            Estimated landmark points for pose estimation
            or None if no face detected
        """
        # If no face rectangle provided, detect faces
        if face_rect is None:
            faces = self.face_cascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            if len(faces) == 0:
                return None
            face_rect = faces[0]  # (x, y, w, h)
        
        # Extract face rectangle coordinates
        x, y, w, h = face_rect
        
        # Estimate facial landmarks based on face geometry
        # This is a simplified approach that doesn't require dlib
        landmarks = [
            (x + w // 2, y + h // 2 - h // 10),      # Nose tip (center of face, slightly above center)
            (x + w // 2, y + h - h // 8),             # Chin (bottom center)
            (x + w // 4, y + h // 3),                 # Left eye left corner
            (x + 3 * w // 4, y + h // 3),             # Right eye right corner
            (x + w // 3, y + 2 * h // 3),             # Left mouth corner
            (x + 2 * w // 3, y + 2 * h // 3)          # Right mouth corner
        ]
        
        return landmarks
    
    def get_pose(self, image, landmarks=None, face_rect=None):
        """
        Estimate head pose from facial landmarks.
        
        Args:
            image: Input image
            landmarks: Optional pre-detected landmarks
                      If None, landmarks will be detected
            face_rect: Optional face rectangle (required if landmarks is None)
        
        Returns:
            Tuple of (success, rotation_vector, translation_vector, euler_angles)
            where euler_angles is (pitch, yaw, roll) in degrees
            or (False, None, None, None) if estimation fails
        """
        # Set camera matrix if not already set
        if self.camera_matrix is None:
            self.set_camera_matrix(image.shape[:2][::-1])

        # Get landmarks if not provided
        if landmarks is None:
            landmarks = self.get_landmarks(image, face_rect)
            if landmarks is None:
                return False, None, None, None

        # Use the landmarks for pose estimation
        image_points = np.array(landmarks, dtype=np.float64)
        
        # Check if we have enough points for pose estimation
        if len(image_points) < 6:
            return False, None, None, None

        # Validate that all image points are finite and reasonable
        if not np.all(np.isfinite(image_points)):
            return False, None, None, None
            
        # Check if points are within image bounds
        h, w = image.shape[:2]
        if np.any(image_points < 0) or np.any(image_points[:, 0] >= w) or np.any(image_points[:, 1] >= h):
            return False, None, None, None

        try:
            # Solve for pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS.astype(np.float64), 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success or rotation_vector is None or translation_vector is None:
                return False, None, None, None
                
            # Validate the output vectors
            if (rotation_vector.size == 0 or translation_vector.size == 0 or
                not np.all(np.isfinite(rotation_vector)) or not np.all(np.isfinite(translation_vector))):
                return False, None, None, None

            # Convert rotation vector to rotation matrix and then to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)

            # Convert to degrees
            euler_angles = np.degrees(euler_angles)
            
            # Validate euler angles
            if not np.all(np.isfinite(euler_angles)):
                return False, None, None, None

            return True, rotation_vector, translation_vector, euler_angles
            
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return False, None, None, None
    
    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            R: Rotation matrix
        
        Returns:
            Tuple of (pitch, yaw, roll) in radians
        """
        # Check if the solution is singular (gimbal lock)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Pitch
            y = np.arctan2(-R[2, 0], sy)      # Yaw
            z = np.arctan2(R[1, 0], R[0, 0])  # Roll
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def is_looking_at_camera(self, euler_angles):
        """
        Determine if the face is looking at the camera based on pose angles.
        
        Args:
            euler_angles: Tuple of (pitch, yaw, roll) in degrees
        
        Returns:
            Boolean indicating if the face is looking at the camera
        """
        pitch, yaw, roll = euler_angles
        
        # Check if angles are within thresholds
        return (abs(yaw) < THRESHOLD_YAW and 
                abs(pitch) < THRESHOLD_PITCH and 
                abs(roll) < THRESHOLD_ROLL)
    
    def draw_pose_info(self, image, rotation_vector, translation_vector, euler_angles, is_looking=None):
        """
        Draw pose information and axes on the image.
        
        Args:
            image: Input image to draw on
            rotation_vector: Rotation vector from pose estimation
            translation_vector: Translation vector from pose estimation
            euler_angles: Tuple of (pitch, yaw, roll) in degrees
            is_looking: Boolean indicating if face is looking at camera (optional)
        
        Returns:
            Image with pose information drawn on it
        """
        # Check if rotation and translation vectors are valid
        if rotation_vector is None or translation_vector is None or euler_angles is None:
            return image
            
        # Validate input vectors
        if (not isinstance(rotation_vector, np.ndarray) or 
            not isinstance(translation_vector, np.ndarray) or
            rotation_vector.size == 0 or translation_vector.size == 0):
            return image
            
        # Make sure camera matrix is set
        if self.camera_matrix is None:
            self.set_camera_matrix(image.shape[:2][::-1])
            
        try:
            # Draw coordinate axes
            axis_length = 50
            axis_points = np.array([
                [0, 0, 0],            # Origin point
                [axis_length, 0, 0],  # X-axis end point (red)
                [0, axis_length, 0],  # Y-axis end point (green)
                [0, 0, axis_length]   # Z-axis end point (blue)
            ], dtype=np.float32)
            
            # Ensure rotation and translation vectors are the correct type
            rotation_vector = rotation_vector.astype(np.float32)
            translation_vector = translation_vector.astype(np.float32)
            
            # Project 3D points to image plane
            image_points, _ = cv2.projectPoints(
                axis_points, rotation_vector, translation_vector, 
                self.camera_matrix, self.dist_coeffs
            )
            
            # Origin point (nose tip)
            origin = tuple(map(int, image_points[0][0]))
            
            # Draw axes
            image = cv2.line(image, origin, tuple(map(int, image_points[1][0])), (0, 0, 255), 3)  # X-axis (red)
            image = cv2.line(image, origin, tuple(map(int, image_points[2][0])), (0, 255, 0), 3)  # Y-axis (green)
            image = cv2.line(image, origin, tuple(map(int, image_points[3][0])), (255, 0, 0), 3)  # Z-axis (blue)
            
            # Draw angle information
            pitch, yaw, roll = euler_angles
            cv2.putText(image, f"Pitch: {pitch:.1f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {yaw:.1f}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Roll: {roll:.1f}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw looking status if provided
            if is_looking is not None:
                status = "Looking at camera" if is_looking else "Looking away"
                color = (0, 255, 0) if is_looking else (0, 0, 255)  # Green if looking, red if not
                cv2.putText(image, status, (10, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
        except Exception as e:
            # If any error occurs during drawing, just return the original image
            print(f"Error drawing pose info: {e}")
            
        return image