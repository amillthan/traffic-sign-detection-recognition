from pathlib import Path
from src.pipeline import detect_and_classify



# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FOLDER = PROJECT_ROOT / "data" / "external_test"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "external_evaluation" / "crops"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    [
        path
        for path in TEST_FOLDER.iterdir()
        if path.suffix.lower() in image_extensions
    ],
    key=lambda p: p.name.lower()
)


print("Saving detected traffic-sign crops...\n")

total_crops = 0


for image_path in image_paths:

    results = detect_and_classify(
        image_path,
        detection_confidence=0.25
    )

    print(
        f"{image_path.name}: "
        f"{len(results)} detection(s)"
    )

    for i, result in enumerate(results, start=1):

        crop = result["crop"]

        output_name = (
            f"{image_path.stem}_"
            f"detection_{i}_"
            f"{result['sign_name'].replace(' ', '_').replace('/', '_')}.jpg"
        )

        output_path = OUTPUT_FOLDER / output_name

        crop.save(output_path)

        print(
            f"  Saved: {output_name}"
        )

        total_crops += 1


print("\nFinished.")
print(f"Total crops saved: {total_crops}")
print(f"Output folder: {OUTPUT_FOLDER}")
