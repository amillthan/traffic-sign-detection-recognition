from pathlib import Path
from src.pipeline import detect_and_classify



# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FOLDER = PROJECT_ROOT / "data" / "external_test"

image_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    [
        path
        for path in TEST_FOLDER.iterdir()
        if path.suffix.lower() in image_extensions
    ],
    key=lambda p: p.name.lower()
)


print("=" * 70)
print("EXTERNAL END-TO-END PIPELINE TEST")
print("=" * 70)
print(f"Total test images: {len(image_paths)}")


total_detections = 0
images_with_detections = 0


for image_path in image_paths:

    print("\n" + "=" * 70)
    print(f"Image: {image_path.name}")
    print("=" * 70)

    try:
        results = detect_and_classify(
            image_path,
            detection_confidence=0.25
        )

        number_of_detections = len(results)

        print(f"Traffic signs detected: {number_of_detections}")

        if number_of_detections == 0:
            print("No traffic sign detected.")
            continue

        images_with_detections += 1
        total_detections += number_of_detections

        for i, result in enumerate(results, start=1):

            print(f"\n--- Detection {i} ---")

            print(
                f"Bounding box: "
                f"{result['bounding_box']}"
            )

            print(
                f"Detection confidence: "
                f"{result['detection_confidence']:.2f}%"
            )

            print(
                f"CNN classification: "
                f"{result['sign_name']}"
            )

            print(
                f"CNN confidence: "
                f"{result['classification_confidence']:.2f}%"
            )

            print("\nTop 3 CNN predictions:")

            for prediction in result["top_3"]:
                print(
                    f"  {prediction['sign_name']} - "
                    f"{prediction['confidence']:.2f}%"
                )

    except Exception as error:
        print(f"ERROR: {error}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total external images: {len(image_paths)}")
print(f"Images with detections: {images_with_detections}")
print(
    f"Images without detections: "
    f"{len(image_paths) - images_with_detections}"
)
print(f"Total detected boxes: {total_detections}")
