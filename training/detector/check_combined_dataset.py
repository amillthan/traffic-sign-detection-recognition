from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "combined_dataset"

print("=" * 60)
print("COMBINED DATASET INTEGRITY CHECK")
print("=" * 60)

total_images = 0
total_labels = 0
total_boxes = 0
invalid_labels = 0

for split in ["train", "val"]:

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    images = list(image_dir.iterdir())
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Label files:", len(labels))

    total_images += len(images)
    total_labels += len(labels)

    for image_path in images:

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print("Missing label:", image_path.name)
            invalid_labels += 1
            continue

        with open(label_path, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                # Empty file = valid negative image
                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print("Invalid YOLO line:", label_path.name, line)
                    invalid_labels += 1
                    continue

                class_id, x, y, w, h = map(float, parts)

                # We only have one class: traffic_sign = 0
                if class_id != 0:
                    print("Invalid class:", label_path.name, class_id)
                    invalid_labels += 1

                # YOLO coordinates should be normalized
                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    print(
                        "Invalid coordinates:",
                        label_path.name,
                        x, y, w, h
                    )
                    invalid_labels += 1

                total_boxes += 1


print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Total images:", total_images)
print("Total label files:", total_labels)
print("Total bounding boxes:", total_boxes)
print("Problems found:", invalid_labels)

if invalid_labels == 0:
    print("\nDataset integrity check PASSED.")
else:
    print("\nDataset integrity check FAILED.")