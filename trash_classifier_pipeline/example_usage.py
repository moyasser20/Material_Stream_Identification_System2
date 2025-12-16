"""
Example usage of the Material Stream Identification System
This script demonstrates how to use the model loader for single image prediction.
"""

import cv2
import argparse
from pathlib import Path
from model_loader import ModelLoader


def predict_image(image_path, pipeline_dir="pipeline", model_type="svm", confidence_threshold=0.5):
    """
    Predict the material class of a single image.
    
    Args:
        image_path: Path to the input image
        pipeline_dir: Directory containing trained models
        model_type: Type of classifier ("svm" or "knn")
        confidence_threshold: Confidence threshold for predictions
    """
    # Check if image exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return
    
    # Initialize model loader
    print(f"Loading models from '{pipeline_dir}'...")
    model_loader = ModelLoader(pipeline_dir, model_type)
    model_loader.load_all()
    
    # Make prediction
    print(f"\nProcessing image: {image_path.name}")
    prediction, class_name, confidence = model_loader.predict(
        image,
        return_probabilities=True,
        confidence_threshold=confidence_threshold
    )
    
    # Display results
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"Class ID: {prediction}")
    print(f"Class Name: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("="*50)
    
    # Display image with prediction
    display_image = image.copy()
    height, width = display_image.shape[:2]
    
    # Add text overlay
    text = f"{class_name} ({confidence:.1%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background
    cv2.rectangle(display_image, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(display_image, text, (15, 35), font, font_scale, (0, 255, 0), thickness)
    
    # Show image
    cv2.imshow("Prediction Result", display_image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predict material class from a single image"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
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
        help="Classifier model to use: svm or knn (default: svm)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    predict_image(
        args.image_path,
        pipeline_dir=args.pipeline_dir,
        model_type=args.model,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()

