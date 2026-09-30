from pathlib import Path
import random
import shutil

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Original GTSDB YOLO dataset
GTSDB_DIR = (
    PROJECT_ROOT
    / "data"
    / "gtsdb"
    / "yolo_dataset"
)

# Selected MTSD dataset
MTSD_DIR = (
    PROJECT_ROOT
    / "data"
    / "mtsd"
    / "selected_training"
)

# New combined dataset
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "combined_dataset"
)

OUTPUT_TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
OUTPUT_VAL_IMAGES = OUTPUT_DIR / "images" / "val"

OUTPUT_TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
OUTPUT_VAL_LABELS = OUTPUT_DIR / "labels" / "val"


# ---------------------------------------------------------
# Make sure output folders exist
# ---------------------------------------------------------

for folder in [
    OUTPUT_TRAIN_IMAGES,
    OUTPUT_VAL_IMAGES,
    OUTPUT_TRAIN_LABELS,
    OUTPUT_VAL_LABELS
]:
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Clear previous combined dataset
# ---------------------------------------------------------

for folder in [
    OUTPUT_TRAIN_IMAGES,
    OUTPUT_VAL_IMAGES,
    OUTPUT_TRAIN_LABELS,
    OUTPUT_VAL_LABELS
]:
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()


# ---------------------------------------------------------
# Collect GTSDB images
# ---------------------------------------------------------

gtsdb_images = []

for split in ["train", "val"]:

    image_dir = GTSDB_DIR / "images" / split
    label_dir = GTSDB_DIR / "labels" / split

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() != ".png":
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        gtsdb_images.append({
            "source": "gtsdb",
            "image": image_path,
            "label": label_path
        })


# ---------------------------------------------------------
# Collect selected MTSD images
# ---------------------------------------------------------

mtsd_image_dir = MTSD_DIR / "images"
mtsd_label_dir = MTSD_DIR / "labels"

mtsd_images = []

for image_path in mtsd_image_dir.iterdir():

    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    label_path = mtsd_label_dir / f"{image_path.stem}.txt"

    mtsd_images.append({
        "source": "mtsd",
        "image": image_path,
        "label": label_path
    })


# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

all_images = gtsdb_images + mtsd_images

print("=" * 60)
print("SOURCE DATASETS")
print("=" * 60)

print("GTSDB images:", len(gtsdb_images))
print("MTSD images:", len(mtsd_images))
print("Total images:", len(all_images))


# ---------------------------------------------------------
# Shuffle reproducibly
# ---------------------------------------------------------

random.seed(42)
random.shuffle(all_images)


# ---------------------------------------------------------
# 80 / 20 split
# ---------------------------------------------------------

split_index = int(len(all_images) * 0.8)

train_items = all_images[:split_index]
val_items = all_images[split_index:]


# ---------------------------------------------------------
# Copy function
# ---------------------------------------------------------

def copy_item(item, output_image_dir, output_label_dir):

    source = item["source"]
    image_path = item["image"]
    label_path = item["label"]

    # Prefix filenames to guarantee there are no collisions
    new_stem = f"{source}_{image_path.stem}"

    destination_image = (
        output_image_dir /
        f"{new_stem}{image_path.suffix.lower()}"
    )

    destination_label = (
        output_label_dir /
        f"{new_stem}.txt"
    )

    shutil.copy2(image_path, destination_image)

    # Positive image
    if label_path.exists():
        shutil.copy2(label_path, destination_label)

    # Negative image: create an empty YOLO label file
    else:
        destination_label.touch()


# ---------------------------------------------------------
# Copy training data
# ---------------------------------------------------------

for item in train_items:
    copy_item(
        item,
        OUTPUT_TRAIN_IMAGES,
        OUTPUT_TRAIN_LABELS
    )


# ---------------------------------------------------------
# Copy validation data
# ---------------------------------------------------------

for item in val_items:
    copy_item(
        item,
        OUTPUT_VAL_IMAGES,
        OUTPUT_VAL_LABELS
    )


# ---------------------------------------------------------
# Count positive / negative images
# ---------------------------------------------------------

def count_positive_negative(items):

    positive = 0
    negative = 0

    for item in items:

        label_path = item["label"]

        if label_path.exists() and label_path.stat().st_size > 0:
            positive += 1
        else:
            negative += 1

    return positive, negative


train_positive, train_negative = count_positive_negative(train_items)
val_positive, val_negative = count_positive_negative(val_items)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COMBINED DATASET CREATED")
print("=" * 60)

print("Training images:", len(train_items))
print("  Positive:", train_positive)
print("  Negative:", train_negative)

print("\nValidation images:", len(val_items))
print("  Positive:", val_positive)
print("  Negative:", val_negative)

print("\nTotal:", len(train_items) + len(val_items))

print("\nDataset location:")
print(OUTPUT_DIR)