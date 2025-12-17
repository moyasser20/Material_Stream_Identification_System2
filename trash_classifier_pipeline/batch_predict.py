
import cv2
import argparse
import csv
from pathlib import Path
from datetime import datetime
from model_loader import ModelLoader
from tqdm import tqdm


def process_folder(input_folder, pipeline_dir="pipeline", model_type="svm",
                   confidence_threshold=0.5, save_images=True, output_folder="results"):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    if not image_files:
        print(f"❌ No images found in '{input_folder}'")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return

    print(f"✓ Found {len(image_files)} images to process")

    print(f"\n🔧 Loading models from '{pipeline_dir}'...")
    try:
        model_loader = ModelLoader(pipeline_dir, model_type)
        model_loader.load_all()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    print(f"\n🔍 Processing images...")
    results = []

    for image_file in tqdm(image_files, desc="Processing"):
        try:
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"⚠ Warning: Could not load {image_file.name}")
                continue

            prediction, class_name, confidence = model_loader.predict(
                image,
                return_probabilities=True,
                confidence_threshold=confidence_threshold
            )

            results.append({
                'filename': image_file.name,
                'class_id': prediction,
                'class_name': class_name,
                'confidence': confidence
            })

            if save_images:
                display_image = image.copy()
                height, width = display_image.shape[:2]

                max_width = 1200
                if width > max_width:
                    scale = max_width / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))
                    height, width = display_image.shape[:2]

                overlay = display_image.copy()
                cv2.rectangle(overlay, (10, 10), (min(width - 10, 450), 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)

                text = f"Material: {class_name}"
                conf_text = f"Confidence: {confidence:.1%}"

                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                cv2.putText(display_image, text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(display_image, conf_text, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                class_colors = {
                    0: (139, 69, 19), 1: (0, 255, 255), 2: (128, 128, 128),
                    3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 0, 255), 6: (128, 128, 128)
                }
                border_color = class_colors.get(prediction, (128, 128, 128))
                cv2.rectangle(display_image, (5, 5), (width - 5, height - 5), border_color, 5)

                output_filename = output_path / f"result_{image_file.stem}.jpg"
                cv2.imwrite(str(output_filename), display_image)

        except Exception as e:
            print(f"⚠ Error processing {image_file.name}: {e}")
            continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_path / f"predictions_{timestamp}.csv"

    print(f"\n💾 Saving results to CSV...")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'class_id', 'class_name', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    print(f"Output folder: {output_path.absolute()}")
    print(f"CSV file: {csv_filename.name}")

    class_counts = {}
    for result in results:
        class_name = result['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nClass Distribution:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"\nAverage Confidence: {avg_confidence:.2%}")
    print("=" * 60)

    print(f"\n✅ Batch processing completed successfully!")
    print(f"📁 Check results in: {output_path.absolute()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images for material classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_predict.py test_images/
  python batch_predict.py test_images/ --model knn
  python batch_predict.py test_images/ --output my_results
  python batch_predict.py test_images/ --no-save-images
        """
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing images to process"
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
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output folder for results (default: results)"
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save annotated images (only CSV)"
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("❌ Error: Confidence must be between 0.0 and 1.0")
        return

    pipeline_path = Path(args.pipeline_dir)
    if not pipeline_path.exists():
        print(f"❌ Error: Pipeline directory '{args.pipeline_dir}' not found!")
        return

    print("\n" + "=" * 60)
    print("Material Classification - Batch Processing")
    print("=" * 60 + "\n")

    process_folder(
        input_folder=args.input_folder,
        pipeline_dir=args.pipeline_dir,
        model_type=args.model,
        confidence_threshold=args.confidence,
        save_images=not args.no_save_images,
        output_folder=args.output
    )


if __name__ == "__main__":
    main()
