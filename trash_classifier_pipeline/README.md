# Real-Time Material Stream Identification System

This application provides real-time material classification using a live camera feed. It processes camera frames and classifies materials into one of seven categories: Glass, Paper, Cardboard, Plastic, Metal, Trash, or Unknown.

## Features

- **Real-time Processing**: Processes live camera frames in real-time
- **Dual Model Support**: Supports both SVM and k-NN classifiers
- **Confidence Thresholding**: Implements rejection mechanism for low-confidence predictions (Unknown class)
- **Visual Feedback**: Displays classification results with color-coded overlays
- **Performance Metrics**: Shows FPS for performance monitoring

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the application with default settings (SVM model, camera 0):
```bash
python realtime_camera_app.py
```

### Advanced Usage

Run with specific model and settings:
```bash
python realtime_camera_app.py --model svm --camera 0 --confidence 0.5
```

### Command Line Arguments

- `--pipeline-dir`: Directory containing trained models (default: `pipeline`)
- `--model`: Classifier model to use - `svm` or `knn` (default: `svm`)
- `--camera`: Camera device index (default: `0`)
- `--confidence`: Confidence threshold for predictions, 0.0-1.0 (default: `0.5`)

### Examples

Use k-NN classifier:
```bash
python realtime_camera_app.py --model knn
```

Use different camera:
```bash
python realtime_camera_app.py --camera 1
```

Adjust confidence threshold:
```bash
python realtime_camera_app.py --confidence 0.7
```

## Controls

While the application is running:
- **'q'**: Quit the application
- **'s'**: Save current frame with prediction to a file
- **'c'**: Change confidence threshold interactively

## Class Categories

The system classifies materials into the following categories:

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Glass | Items made of amorphous solid materials |
| 1 | Paper | Thin materials made from pressed cellulose pulp |
| 2 | Cardboard | Heavy-duty structured material |
| 3 | Plastic | Items made from high-molecular-weight organic compounds |
| 4 | Metal | Items made of elemental or compound metallic substances |
| 5 | Trash | Miscellaneous non-recyclable or contaminated waste |
| 6 | Unknown | Out-of-distribution items or low-confidence predictions |

## Model Architecture

The system uses:
- **Feature Extractor**: EfficientNetB3 CNN model for feature extraction
- **Classifier**: Either SVM (RBF kernel) or k-NN (distance-weighted) classifier
- **Preprocessing**: Image resizing to 300x300, normalization, and feature scaling

## Notes

- The application requires a webcam or camera device connected to your system
- For best results, ensure good lighting and clear view of the material
- The confidence threshold can be adjusted to balance between accuracy and rejection rate
- Lower confidence thresholds will classify more items as "Unknown"

