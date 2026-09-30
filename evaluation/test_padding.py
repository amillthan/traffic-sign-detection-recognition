from pathlib import Path
from src.pipeline import detect_and_classify


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FOLDER = PROJECT_ROOT / "data" / "external_test"

# Focus on the three difficult images
test_images = [
    "test 2.jpg",   # 80 km/h
    "test 4.jpg",   # right curve
    "test 5.jpg",   # uneven/bumpy road
]

padding_values = [0.0, 0.15, 0.25, 0.35]

for image_name in test_images:

    image_path = TEST_FOLDER / image_name

    print("\n" + "=" * 70)
    print(f"IMAGE: {image_name}")
    print("=" * 70)

    for padding in padding_values:

        results = detect_and_classify(
            image_path,
            detection_confidence=0.25,
            padding_ratio=padding
        )

        print(f"\nPadding: {int(padding * 100)}%")

        if not results:
            print("No detection")
            continue

        for i, result in enumerate(results, start=1):

            print(
                f"Detection {i}: "
                f"{result['sign_name']} "
                f"({result['classification_confidence']:.2f}%)"
            )
            
