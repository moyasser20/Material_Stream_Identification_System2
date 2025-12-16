
import cv2
import numpy as np
import argparse
from pathlib import Path
from model_loader import ModelLoader


class MaterialStreamIdentifier:
    """Real-time material classification using camera feed."""

    # FIXED: Class names must match your training data exactly
    CLASS_NAMES = {
        0: "Cardboard",
        1: "Glass",      # Fixed: was "glass"
        2: "Metal",      # Fixed: was "metal"
        3: "Paper",      # Fixed: was "paper"
        4: "Plastic",    # Fixed: was "plastic"
        5: "Trash",
        6: "Unknown"
    }

    # FIXED: Colors matched to correct classes
    CLASS_COLORS = {
        0: (139, 69, 19),      # Brown for Cardboard
        1: (0, 255, 255),      # Cyan for Glass
        2: (128, 128, 128),    # Gray for Metal
        3: (255, 255, 0),      # Yellow for Paper
        4: (255, 0, 255),      # Magenta for Plastic
        5: (0, 0, 255),        # Red for Trash
        6: (128, 128, 128)     # Gray for Unknown
    }

    def __init__(self, pipeline_dir="pipeline", model_type="svm", confidence_threshold=0.5):
        """
        Initialize the material stream identifier.

        Args:
            pipeline_dir: Directory containing trained models
            model_type: Type of classifier ("svm" or "knn")
            confidence_threshold: Minimum confidence to accept prediction
        """
        self.pipeline_dir = pipeline_dir
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        # Initialize model loader
        print("\n" + "="*60)
        print("🚀 Initializing Material Stream Identifier")
        print("="*60)

        try:
            self.model_loader = ModelLoader(pipeline_dir, model_type)
            self.model_loader.load_all()
        except Exception as e:
            print(f"\n❌ ERROR loading models: {e}")
            print("\n📋 Troubleshooting:")
            print("1. Make sure 'pipeline' folder exists in current directory")
            print("2. Check that all required files are present:")
            print(f"   - {pipeline_dir}/svm_model.pkl")
            print(f"   - {pipeline_dir}/knn_model.pkl")
            print(f"   - {pipeline_dir}/scaler.pkl")
            print(f"   - {pipeline_dir}/class_mapping.pkl")
            raise

        self.current_prediction = None
        self.current_confidence = 0.0
        self.frame_count = 0

        # Performance tracking
        self.prediction_times = []

    def draw_prediction(self, frame, prediction, confidence):
        """
        Draw prediction results on the frame with enhanced visuals.

        Args:
            frame: Input frame
            prediction: Predicted class ID
            confidence: Prediction confidence
        """
        class_name = self.CLASS_NAMES.get(prediction, "Unknown")
        color = self.CLASS_COLORS.get(prediction, (128, 128, 128))

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Create semi-transparent overlay for info box
        overlay = frame.copy()
        box_height = 140
        cv2.rectangle(overlay, (10, 10), (width - 10, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Prepare text
        text = f"Material: {class_name}"
        confidence_text = f"Confidence: {confidence:.1%}"
        model_text = f"Model: {self.model_type.upper()}"

        # Choose confidence color
        if confidence > 0.7:
            conf_color = (0, 255, 0)      # Green - High
        elif confidence > 0.5:
            conf_color = (0, 255, 255)    # Yellow - Medium
        else:
            conf_color = (0, 165, 255)    # Orange - Low

        # Draw prediction text with shadow effect for better visibility
        # Shadow
        cv2.putText(frame, text, (22, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
        # Main text in class color
        cv2.putText(frame, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        # White outline
        cv2.putText(frame, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence text with color based on level
        cv2.putText(frame, confidence_text, (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2, cv2.LINE_AA)

        # Model info
        cv2.putText(frame, model_text, (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw colored border based on predicted class
        border_thickness = 6
        cv2.rectangle(frame, (border_thickness, border_thickness),
                     (width - border_thickness, height - border_thickness),
                     color, border_thickness)

        return frame

    def process_frame(self, frame):
        """
        Process a single frame and return prediction.

        Args:
            frame: Input frame from camera (BGR format from OpenCV)

        Returns:
            Frame with prediction overlay
        """
        try:
            import time
            start_time = time.time()

            # Get prediction (model_loader handles BGR->RGB conversion)
            prediction, class_name, confidence = self.model_loader.predict(
                frame,
                return_probabilities=True,
                confidence_threshold=self.confidence_threshold
            )

            # Track processing time
            process_time = time.time() - start_time
            self.prediction_times.append(process_time)
            if len(self.prediction_times) > 30:
                self.prediction_times.pop(0)

            self.current_prediction = prediction
            self.current_confidence = confidence

            # Draw prediction on frame
            frame_with_prediction = self.draw_prediction(frame.copy(), prediction, confidence)

            # Add processing time info
            avg_time = np.mean(self.prediction_times) if self.prediction_times else 0
            time_text = f"Process: {avg_time*1000:.0f}ms"
            cv2.putText(frame_with_prediction, time_text,
                       (frame.shape[1] - 180, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return frame_with_prediction

        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            # Draw error message on frame
            cv2.putText(frame, f"ERROR: {str(e)[:50]}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

    def run(self, camera_index=0, window_name="Material Stream Identification"):
        """
        Run the real-time camera application.

        Args:
            camera_index: Camera device index (default: 0)
            window_name: Name of the display window
        """
        # Initialize camera
        print(f"\n📷 Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            print("\n📋 Troubleshooting:")
            print("- Check if camera is connected")
            print("- Try a different camera index: --camera 1")
            print("- Check camera permissions")
            print("- Close other apps using the camera")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "="*60)
        print("✅ Material Stream Identification System - READY")
        print("="*60)
        print("📋 Keyboard Controls:")
        print("  'q' - Quit application")
        print("  's' - Save current frame")
        print("  'c' - Change confidence threshold")
        print("  'i' - Show system information")
        print("  'm' - Switch model (SVM ↔ KNN)")
        print("="*60 + "\n")

        frame_count = 0
        fps_start_time = cv2.getTickCount()
        fps = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("⚠ Warning: Could not read frame from camera")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = cv2.getTickCount()
                    time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                    fps = 30 / time_diff
                    fps_start_time = fps_end_time

                # Display FPS
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(processed_frame, fps_text,
                           (processed_frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display keyboard shortcuts at bottom
                help_text = "Press: q=quit | s=save | i=info | c=confidence | m=model"
                cv2.putText(processed_frame, help_text,
                           (10, processed_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Show frame
                cv2.imshow(window_name, processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Quitting application...")
                    break

                elif key == ord('s'):
                    # Save current frame with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{self.CLASS_NAMES.get(self.current_prediction, 'Unknown')}_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📷 Saved: {filename}")

                elif key == ord('c'):
                    # Change confidence threshold
                    print(f"\n⚙️  Current confidence threshold: {self.confidence_threshold:.2f}")
                    try:
                        new_threshold = float(input("Enter new threshold (0.0-1.0): "))
                        if 0.0 <= new_threshold <= 1.0:
                            self.confidence_threshold = new_threshold
                            print(f"✓ Threshold updated to {new_threshold:.2f}")
                        else:
                            print("❌ Invalid! Must be between 0.0 and 1.0")
                    except ValueError:
                        print("❌ Invalid input")
                    except KeyboardInterrupt:
                        pass

                elif key == ord('i'):
                    # Show system info
                    print("\n" + "="*60)
                    print("📊 SYSTEM INFORMATION")
                    print("="*60)
                    print(f"Model Type: {self.model_type.upper()}")
                    print(f"Confidence Threshold: {self.confidence_threshold:.2f}")
                    print(f"Current Prediction: {self.CLASS_NAMES.get(self.current_prediction, 'Unknown')}")
                    print(f"Current Confidence: {self.current_confidence:.2%}")
                    print(f"Frame Count: {frame_count}")
                    print(f"FPS: {fps:.1f}")
                    if self.prediction_times:
                        print(f"Avg Processing Time: {np.mean(self.prediction_times)*1000:.1f}ms")
                    print(f"Camera Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    print("="*60 + "\n")

                elif key == ord('m'):
                    # Switch model
                    print("\n⚠️  Note: To switch models, restart with different --model argument")
                    print(f"Current: {self.model_type.upper()}")
                    print(f"To use KNN: python realtime_camera_app.py --model knn")
                    print(f"To use SVM: python realtime_camera_app.py --model svm")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Camera released. Application closed.")
            print(f"📊 Total frames processed: {frame_count}\n")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Real-Time Material Stream Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_camera_app.py
  python realtime_camera_app.py --model knn
  python realtime_camera_app.py --camera 1 --confidence 0.7
  python realtime_camera_app.py --model svm --confidence 0.6 --camera 0
        """
    )

    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default="pipeline",
        help="Directory containing trained models (default: pipeline)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "knn"],
        default="svm",
        help="Classifier model to use (default: svm)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.confidence <= 1.0:
        print("❌ Error: Confidence must be between 0.0 and 1.0")
        return

    # Check if pipeline directory exists
    pipeline_path = Path(args.pipeline_dir)
    if not pipeline_path.exists():
        print(f"❌ Error: Pipeline directory '{args.pipeline_dir}' not found!")
        print("\n📋 Setup Instructions:")
        print("1. Train your models in Kaggle")
        print("2. Save the pipeline using save_pipeline.py")
        print("3. Download the 'pipeline' folder")
        print("4. Place it in the same directory as this script")
        return

    # Initialize and run the application
    try:
        identifier = MaterialStreamIdentifier(
            pipeline_dir=args.pipeline_dir,
            model_type=args.model,
            confidence_threshold=args.confidence
        )

        identifier.run(camera_index=args.camera)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()