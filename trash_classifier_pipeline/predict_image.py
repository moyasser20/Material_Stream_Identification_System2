
import cv2
import argparse
from pathlib import Path
from model_loader import ModelLoader


def predict_image(image_path, pipeline_dir="pipeline", model_type="svm", confidence_threshold=0.5):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Error: Image file '{image_path}' not found!")
        return

    print(f"📷 Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image from '{image_path}'")
        print("Make sure the file is a valid image (jpg, png, etc.)")
        return

    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    print(f"\n🔧 Loading models from '{pipeline_dir}'...")
    try:
        model_loader = ModelLoader(pipeline_dir, model_type)
        model_loader.load_all()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    print(f"\n🔍 Processing image: {image_path.name}")
    try:
        prediction, class_name, confidence = model_loader.predict(
            image,
            return_probabilities=True,
            confidence_threshold=confidence_threshold
        )

        print("\n" + "=" * 60)
        print("📊 PREDICTION RESULTS")
        print("=" * 60)
        print(f"Material Class: {class_name}")
        print(f"Class ID: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Model Used: {model_type.upper()}")
        print("=" * 60)

        display_image = image.copy()
        height, width = display_image.shape[:2]

        max_display_width = 1200
        if width > max_display_width:
            scale = max_display_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
            height, width = display_image.shape[:2]

        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)

        text = f"Material: {class_name}"
        confidence_text = f"Confidence: {confidence:.1%}"
        model_text = f"Model: {model_type.upper()}"

        if confidence > 0.7:
            text_color = (0, 255, 0)
        elif confidence > 0.5:
            text_color = (0, 255, 255)
        else:
            text_color = (0, 165, 255)

        cv2.putText(display_image, text, (22, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(display_image, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)

        cv2.putText(display_image, confidence_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_image, model_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        class_colors = {
            0: (139, 69, 19),
            1: (0, 255, 255),
            2: (128, 128, 128),
            3: (255, 255, 0),
            4: (255, 0, 255),
            5: (0, 0, 255),
            6: (128, 128, 128)
        }
        border_color = class_colors.get(prediction, (128, 128, 128))
        cv2.rectangle(display_image, (5, 5), (width - 5, height - 5), border_color, 5)

        window_name = "Material Classification Result"
        cv2.imshow(window_name, display_image)
        print("\n💡 Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        save_choice = input("\n💾 Save result image? (y/n): ").lower()
        if save_choice == 'y':
            output_filename = f"result_{class_name}_{image_path.stem}.jpg"
            cv2.imwrite(output_filename, display_image)
            print(f"✓ Saved to: {output_filename}")

        print("\n✅ Test completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Predict material class from a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_image.py photo.jpg
  python predict_image.py photo.jpg --model knn
  python predict_image.py photo.jpg --confidence 0.7
  python predict_image.py photo.jpg --model svm --confidence 0.6
        """
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
        help="Classifier model to use (default: svm)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("❌ Error: Confidence must be between 0.0 and 1.0")
        return

    pipeline_path = Path(args.pipeline_dir)
    if not pipeline_path.exists():
        print(f"❌ Error: Pipeline directory '{args.pipeline_dir}' not found!")
        print("\nMake sure you have:")
        print("1. Downloaded the 'pipeline' folder from Kaggle")
        print("2. Placed it in the same directory as this script")
        return

    predict_image(
        args.image_path,
        pipeline_dir=args.pipeline_dir,
        model_type=args.model,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()
