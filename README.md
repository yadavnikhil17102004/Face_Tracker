# Face Tracker

A lightweight and efficient face tracking application that uses your webcam to detect and track human faces in real-time. The program draws bounding boxes around detected faces and displays the number of faces detected along with the current FPS (Frames Per Second).

This repository includes two versions:
1. **Simple Face Tracker** - Basic version with face detection only
2. **Advanced Face Tracker** - Extended version with eye detection, screenshots, and more features

## Features

### Simple Face Tracker
- Real-time face detection using OpenCV
- FPS counter to monitor performance
- Face counter showing number of detected faces
- Configurable parameters for easy customization
- Mirror mode (horizontal flip) for intuitive interaction
- Face count warning when too many faces are detected (configurable via .env)

### Advanced Face Tracker
All features from the simple version, plus:
- Eye detection within detected faces
- Screenshot capture functionality (press 's')
- Toggle help display (press 'h')
- Toggle eye detection on/off (press 'e')
- Enhanced statistics tracking
- Organized screenshot storage
- Face count warning with customizable threshold and message

## Requirements

- Python 3.6 or higher
- OpenCV library
- python-dotenv (for configuration)
- A working webcam

## Installation

1. Clone or download this repository

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration
You can customize the face count warning by editing the `.env` file in the project root:

```
# Maximum number of faces before showing warning
MAX_FACE_COUNT=3

# Warning message to display when face count exceeds limit
FACE_COUNT_WARNING_MESSAGE="Warning: Too many faces detected!"

# Warning text color (BGR format)
WARNING_TEXT_COLOR_B=0
WARNING_TEXT_COLOR_G=0
WARNING_TEXT_COLOR_R=255
```

### Simple Face Tracker
Run the simple face tracker application:

```bash
python face_tracker.py
```

Or use the provided batch file (Windows):

```
run_face_tracker.bat
```

Press 'q' to quit the application.

### Advanced Face Tracker
Run the advanced face tracker application:

```bash
python advanced_face_tracker.py
```

Or use the provided batch file (Windows):

```
run_advanced_face_tracker.bat
```

**Controls:**
- Press 'q' to quit the application
- Press 's' to take a screenshot
- Press 'h' to toggle help text display
- Press 'e' to toggle eye detection on/off

## Configuration

### Simple Face Tracker
You can customize the simple face tracker by modifying the `CONFIG` dictionary in the `face_tracker.py` file:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `camera_id` | Camera device ID | 0 (built-in webcam) |
| `frame_width` | Width of camera frame | 640 |
| `frame_height` | Height of camera frame | 480 |
| `box_color` | Color of face bounding box (BGR format) | (0, 255, 0) (green) |
| `box_thickness` | Thickness of bounding box lines | 2 |
| `min_face_size` | Minimum face size to detect | (30, 30) |
| `scale_factor` | How much the image size is reduced at each scale | 1.1 |
| `min_neighbors` | How many neighbors each candidate rectangle should have | 5 |
| `show_fps` | Whether to display FPS counter | True |
| `flip_horizontal` | Flip the camera horizontally (mirror mode) | True |

### Advanced Face Tracker
The advanced face tracker has additional configuration options in the `CONFIG` dictionary in the `advanced_face_tracker.py` file:

#### Camera Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `camera_id` | Camera device ID | 0 (built-in webcam) |
| `frame_width` | Width of camera frame | 640 |
| `frame_height` | Height of camera frame | 480 |
| `flip_horizontal` | Flip the camera horizontally (mirror mode) | True |

#### Face Detection Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `face_box_color` | Color of face bounding box (BGR format) | (0, 255, 0) (green) |
| `face_box_thickness` | Thickness of bounding box lines | 2 |
| `min_face_size` | Minimum face size to detect | (30, 30) |
| `scale_factor` | How much the image size is reduced at each scale | 1.1 |
| `min_neighbors` | How many neighbors each candidate rectangle should have | 5 |

#### Eye Detection Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `detect_eyes` | Whether to detect eyes within faces | True |
| `eye_box_color` | Color of eye bounding box (BGR format) | (255, 0, 0) (blue) |
| `eye_box_thickness` | Thickness of eye bounding box lines | 1 |

#### Display Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `show_fps` | Whether to display FPS counter | True |
| `show_help` | Whether to display help text | True |
| `text_color` | Color of text overlays (BGR format) | (0, 0, 255) (red) |
| `text_size` | Size of text overlays | 0.7 |
| `text_thickness` | Thickness of text overlays | 2 |

#### Screenshot Settings
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `screenshot_dir` | Directory to save screenshots | 'screenshots' |
| `screenshot_format` | Format to save screenshots (jpg or png) | 'jpg' |

## How It Works

### Face Detection
Both versions of the application use the Haar Cascade classifier from OpenCV, which is a machine learning-based approach where a cascade function is trained from many positive and negative images. It's particularly efficient for face detection.

The basic process for both versions:
1. Captures video from your webcam
2. Converts each frame to grayscale
3. Applies the face detection algorithm
4. Draws rectangles around detected faces
5. Displays the processed frame with face counts and FPS

### Eye Detection (Advanced Version)
The advanced version also uses a Haar Cascade classifier specifically trained for eye detection. For each detected face:
1. A region of interest (ROI) is extracted from the face area
2. The eye detection algorithm is applied to this region
3. Rectangles are drawn around detected eyes

### Screenshots (Advanced Version)
The advanced version allows you to capture screenshots by pressing the 's' key:
1. Screenshots are saved to the 'screenshots' directory (created automatically)
2. Files are named with a timestamp (e.g., 'face_tracker_20230615_120530.jpg')
3. The format can be configured (JPG or PNG)

## Troubleshooting

- If the webcam doesn't open, check that your camera is connected and not being used by another application
- If face detection is not working well, try adjusting the `scale_factor` and `min_neighbors` parameters
- For better performance on slower computers, try reducing the `frame_width` and `frame_height`

## Further Extensions

Some ideas for further extending this application:

- Add face recognition to identify specific people
- Track faces across frames to assign consistent IDs
- Add emotion detection to recognize facial expressions
- Implement facial landmark detection (nose, mouth, etc.)
- Add age and gender estimation
- Create a recording feature to save video
- Implement motion detection to only track when movement occurs
- Add a GUI for adjusting settings without modifying code

## License

This project is open source and available for personal and educational use.