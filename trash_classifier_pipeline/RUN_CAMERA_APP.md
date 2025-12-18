# How to Run the Camera App

## Quick Start

1. Open PyCharm Terminal (View → Tool Windows → Terminal)
2. Navigate to the project directory:
   ```bash
   cd C:\ml\Material_Stream_Identification_System\trash_classifier_pipeline
   ```
3. Run the app:
   ```bash
   python realtime_camera_app.py
   ```

## Command Line Options

### Basic Usage
```bash
python realtime_camera_app.py
```

### Use Different Model
```bash
# Use SVM (default)
python realtime_camera_app.py --model svm

# Use k-NN
python realtime_camera_app.py --model knn
```

### Use Different Camera
```bash
# Default camera (0)
python realtime_camera_app.py --camera 0

# Second camera
python realtime_camera_app.py --camera 1
```

### Adjust Confidence Threshold
```bash
# Lower threshold (more predictions, less strict)
python realtime_camera_app.py --confidence 0.3

# Higher threshold (fewer predictions, more strict)
python realtime_camera_app.py --confidence 0.7
```

### Combine Options
```bash
python realtime_camera_app.py --model knn --camera 0 --confidence 0.6
```

## Controls While Running

- **'q'**: Quit the application
- **'s'**: Save current frame with prediction
- **'c'**: Change confidence threshold interactively

## Troubleshooting

If you get an error about the pipeline directory not found:
- Make sure you're in the `trash_classifier_pipeline` directory
- Or specify the full path: `python realtime_camera_app.py --pipeline-dir pipeline`



