from pathlib import Path
from collections import Counter
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MTSD_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "mtsd"
    / "raw"
)

IMAGE_DIR = (
    MTSD_RAW_DIR
    / "images_train_0"
    / "images"
)

ANNOTATION_DIR = (
    MTSD_RAW_DIR
    / "mtsd_v2_fully_annotated"
    / "annotations"
)

image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]

print("MTSD images found:", len(images))

label_counts = Counter()

images_with_annotations = 0
images_without_annotations = 0
total_objects = 0

for image_path in images:

    annotation_path = ANNOTATION_DIR / f"{image_path.stem}.json"

    if not annotation_path.exists():
        images_without_annotations += 1
        continue

    images_with_annotations += 1

    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    objects = annotation.get("objects", [])

    total_objects += len(objects)

    for obj in objects:
        label = obj.get("label")

        if label:
            label_counts[label] += 1


print("\n" + "=" * 60)
print("MTSD TRAIN.0 ANALYSIS")
print("=" * 60)

print("Images:", len(images))
print("Images with annotations:", images_with_annotations)
print("Images without annotations:", images_without_annotations)
print("Total annotated objects:", total_objects)
print("Unique labels:", len(label_counts))

print("\n" + "=" * 60)
print("ALL MTSD LABELS")
print("=" * 60)

for label, count in label_counts.most_common():
    print(f"{count:6}  {label}")