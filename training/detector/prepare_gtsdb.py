from pathlib import Path
from PIL import Image
import random
import shutil

# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "gtsdb" / "FullIJCNN2013"
GT_FILE = SOURCE_DIR / "gt.txt"

YOLO_DIR = PROJECT_ROOT / "data" / "gtsdb" / "yolo_dataset"

print("Source directory:", SOURCE_DIR)
print("Ground truth file:", GT_FILE)

print("\nChecking files...")

print("Source directory exists:", SOURCE_DIR.exists())
print("gt.txt exists:", GT_FILE.exists())


# -----------------------------
# Read ground-truth annotations
# -----------------------------

annotations = []

with open(GT_FILE, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        parts = line.split(";")

        image_name = parts[0]
        x_min = int(parts[1])
        y_min = int(parts[2])
        x_max = int(parts[3])
        y_max = int(parts[4])
        original_class_id = int(parts[5])

        annotations.append({
            "image": image_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "class_id": original_class_id
        })


# -----------------------------
# Dataset information
# -----------------------------

image_files = list(SOURCE_DIR.glob("*.ppm"))

annotated_images = set(
    annotation["image"]
    for annotation in annotations
)

print("\nDataset information:")
print("Total road images:", len(image_files))
print("Total traffic sign annotations:", len(annotations))
print("Images containing traffic signs:", len(annotated_images))
print("Images without annotated traffic signs:",
      len(image_files) - len(annotated_images))


# Show some annotations
print("\nFirst 5 annotations:")

for annotation in annotations[:5]:
    print(annotation)



# -----------------------------
# Convert bounding box to YOLO format
# -----------------------------

def convert_to_yolo(x_min, y_min, x_max, y_max, image_width, image_height):

    box_width = x_max - x_min
    box_height = y_max - y_min

    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    # Normalize values between 0 and 1
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


# -----------------------------
# Test conversion using first annotation
# -----------------------------

test_annotation = annotations[0]

test_image_path = SOURCE_DIR / test_annotation["image"]

with Image.open(test_image_path) as image:
    image_width, image_height = image.size


yolo_box = convert_to_yolo(
    test_annotation["x_min"],
    test_annotation["y_min"],
    test_annotation["x_max"],
    test_annotation["y_max"],
    image_width,
    image_height
)


print("\nYOLO conversion test:")
print("Image:", test_annotation["image"])
print("Image size:", image_width, "x", image_height)

print(
    "Original bounding box:",
    test_annotation["x_min"],
    test_annotation["y_min"],
    test_annotation["x_max"],
    test_annotation["y_max"]
)

print("YOLO bounding box:", yolo_box)


# -----------------------------
# Create YOLO dataset folders
# -----------------------------

train_images_dir = YOLO_DIR / "images" / "train"
val_images_dir = YOLO_DIR / "images" / "val"

train_labels_dir = YOLO_DIR / "labels" / "train"
val_labels_dir = YOLO_DIR / "labels" / "val"

directories = [
    train_images_dir,
    val_images_dir,
    train_labels_dir,
    val_labels_dir
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)

print("\nYOLO dataset folders created successfully.")

print("Train images:", train_images_dir)
print("Validation images:", val_images_dir)
print("Train labels:", train_labels_dir)
print("Validation labels:", val_labels_dir)


# -----------------------------
# Split images into train / validation
# -----------------------------

random.seed(42)

all_images = sorted(SOURCE_DIR.glob("*.ppm"))

# Shuffle images
random.shuffle(all_images)

split_index = int(len(all_images) * 0.8)

train_images = all_images[:split_index]
val_images = all_images[split_index:]

print("\nDataset split:")
print("Total images:", len(all_images))
print("Training images:", len(train_images))
print("Validation images:", len(val_images))

# -----------------------------
# Check positive / negative distribution
# -----------------------------

def count_positive_negative(image_list, annotated_image_names):

    positive = 0
    negative = 0

    for image_path in image_list:

        if image_path.name in annotated_image_names:
            positive += 1
        else:
            negative += 1

    return positive, negative


train_positive, train_negative = count_positive_negative(
    train_images,
    annotated_images
)

val_positive, val_negative = count_positive_negative(
    val_images,
    annotated_images
)


print("\nPositive / Negative distribution:")

print(
    f"Training   -> Positive: {train_positive}, "
    f"Negative: {train_negative}"
)

print(
    f"Validation -> Positive: {val_positive}, "
    f"Negative: {val_negative}"
)


# -----------------------------
# Group annotations by image
# -----------------------------

annotations_by_image = {}

for annotation in annotations:

    image_name = annotation["image"]

    if image_name not in annotations_by_image:
        annotations_by_image[image_name] = []

    annotations_by_image[image_name].append(annotation)


# -----------------------------
# Process one dataset split
# -----------------------------

def process_split(image_list, output_images_dir, output_labels_dir):

    for image_path in image_list:

        # Convert PPM image to PNG for YOLO
        destination_image = output_images_dir / (
            image_path.stem + ".png"
        )

        with Image.open(image_path) as image:

            image_width, image_height = image.size

            image.convert("RGB").save(
                destination_image,
                format="PNG"
            )

        # YOLO label filename
        label_path = output_labels_dir / (
            image_path.stem + ".txt"
        )

        # Get annotations for this image
        # Negative images return an empty list
        image_annotations = annotations_by_image.get(
            image_path.name,
            []
        )

        yolo_lines = []

        for annotation in image_annotations:

            x_center, y_center, box_width, box_height = convert_to_yolo(
                annotation["x_min"],
                annotation["y_min"],
                annotation["x_max"],
                annotation["y_max"],
                image_width,
                image_height
            )

            # Detector has only ONE class:
            # 0 = traffic_sign
            yolo_line = (
                f"0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            yolo_lines.append(yolo_line)

        # Create label file.
        # Negative images intentionally get an empty label file.
        with open(label_path, "w") as label_file:

            if yolo_lines:
                label_file.write("\n".join(yolo_lines))


# -----------------------------
# Generate train dataset
# -----------------------------

print("\nGenerating training dataset...")

process_split(
    train_images,
    train_images_dir,
    train_labels_dir
)


# -----------------------------
# Generate validation dataset
# -----------------------------

print("Generating validation dataset...")

process_split(
    val_images,
    val_images_dir,
    val_labels_dir
)

print("\nYOLO dataset generated successfully!")