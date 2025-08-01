import os

"""
Model Directory Creator Script

This script creates the models directory for the Face Tracker application.
We no longer need to download the dlib facial landmark predictor model.
"""

# Directory to save models
MODEL_DIR = 'models'


def create_models_directory():
    """
    Create the models directory if it doesn't exist.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created models directory at {MODEL_DIR}")
    else:
        print(f"Models directory already exists at {MODEL_DIR}")
    
    print("\nNote: This version of the head pose estimator uses OpenCV's built-in face detection")
    print("and doesn't require downloading additional model files.")


def main():
    """
    Create models directory.
    """
    print("Setting up directory for Face Tracker...")
    create_models_directory()
    print("Setup completed successfully.")


if __name__ == "__main__":
    main()