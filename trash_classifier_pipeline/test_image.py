
import cv2
import sys
from pathlib import Path
from model_loader import ModelLoader


def test_image(image_path):

    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image from '{image_path}'")
        print("Make sure the file is a valid image (jpg, png, etc.)")
        return False

    print(f"✓ Image loaded successfully! Size: {image.shape[1]}x{image.shape[0]}")

    print("\n🔧 Loading models...")
    try:
        model_loader = ModelLoader(pipeline_dir="pipeline", model_type="svm")
        model_loader.load_all()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nTroubleshooting:")
        print("1. Check that 'pipeline' folder exists in current directory")
        print("2. Verify all .pkl files are present in pipeline folder")
        return False

    print(f"\n🔍 Processing image: {image_path.name}")
    try:
        prediction, class_name, confidence = model_loader.predict(
            image,
            return_probabilities=True,
            confidence_threshold=0.5
        )

        print("\n" + "="*60)
        print("📊 PREDICTION RESULTS")
        print("="*60)
        print(f"Material Class: {class_name}")
        print(f"Class ID: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print("="*60)

        display_image = image.copy()
        height, width = display_image.shape[:2]

        max_width = 1000
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
            height, width = display_image.shape[:2]

        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, 10), (min(width - 10, 400), 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)

        text = f"Material: {class_name}"
        confidence_text = f"Confidence: {confidence:.1%}"

        if confidence > 0.7:
            color = (0, 255, 0)
        elif confidence > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.putText(display_image, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_image, confidence_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        window_name = "Material Classification Result"
        cv2.imshow(window_name, display_image)
        print("\n💡 Displaying result. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():

    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("Simple Image Test Script")
        print("="*60)
        print("\nUsage:")
        print("  python test_image.py <path_to_image>")
        print("\nExamples:")
        print("  python test_image.py test_photo.jpg")
        print("  python test_image.py /path/to/image.png")
        print("  python test_image.py \"C:\\Users\\YourName\\Desktop\\photo.jpg\"")
        print("\n" + "="*60)
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        print("\nTips:")
        print("- Check the file path is correct")
        print("- Use quotes if path contains spaces")
        print("- Make sure the file exists")
        sys.exit(1)

    print("\n" + "="*60)
    print("Material Classification - Image Test")
    print("="*60 + "\n")

    success = test_image(image_path)

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Test failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
