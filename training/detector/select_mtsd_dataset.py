from collections import Counter
from pathlib import Path
import json
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MTSD_DIR = (
    PROJECT_ROOT
    / "data"
    / "mtsd"
    / "raw"
)

IMAGE_DIR = (
    MTSD_DIR
    / "images_train_0"
    / "images"
)

ANNOTATION_DIR = (
    MTSD_DIR
    / "mtsd_v2_fully_annotated"
    / "annotations"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "mtsd"
    / "selected_training"
)

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"


# MTSD labels that clearly correspond to signs
# supported by our GTSRB classifier.
SUPPORTED_PATTERNS = [

    # Speed limits
    "regulatory--maximum-speed-limit-20--",
    "regulatory--maximum-speed-limit-30--",
    "regulatory--maximum-speed-limit-50--",
    "regulatory--maximum-speed-limit-60--",
    "regulatory--maximum-speed-limit-70--",
    "regulatory--maximum-speed-limit-80--",
    "regulatory--maximum-speed-limit-100--",
    "regulatory--maximum-speed-limit-120--",

    # Priority / regulatory
    "regulatory--yield--",
    "regulatory--stop--",
    "regulatory--priority-road--",
    "regulatory--no-entry--",
    "regulatory--keep-right--",
    "regulatory--roundabout--",

    # Overtaking / vehicle restrictions
    "regulatory--no-overtaking--",
    "regulatory--no-overtaking-by-heavy-goods-vehicles--",

    # Warning signs
    "warning--curve-left--",
    "warning--curve-right--",
    "warning--roadworks--",
    "warning--slippery-road-surface--",
    "warning--children--",
    "warning--wild-animals--",
    "warning--road-narrows--",
    "warning--traffic-signals--",
    "warning--pedestrians-crossing--",
    "warning--bicycles-crossing--",
    "warning--other-danger--",
    "warning--uneven-road--",
    "warning--two-way-traffic--",
]


def is_supported_label(label):
    """
    Return True if an MTSD label matches one of the
    traffic-sign types supported by our GTSRB classifier.
    """
    return any(pattern in label for pattern in SUPPORTED_PATTERNS)


def get_sign_category(label):
    """
    Remove the MTSD graphical variant suffix.

    Example:
    warning--curve-left--g2
    becomes
    warning--curve-left
    """
    parts = label.split("--")

    if parts[-1].startswith("g"):
        return "--".join(parts[:-1])

    return label


# ---------------------------------------------------------
# Analyse available MTSD images
# ---------------------------------------------------------

image_extensions = {".jpg", ".jpeg", ".png"}

image_files = [
    path for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]

print("MTSD images available:", len(image_files))


usable_images = []
supported_object_count = 0


for image_path in image_files:

    annotation_path = ANNOTATION_DIR / f"{image_path.stem}.json"

    if not annotation_path.exists():
        continue

    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    supported_objects = []

    for obj in annotation.get("objects", []):

        label = obj.get("label", "")

        if is_supported_label(label):
            supported_objects.append(obj)

    # Keep the image only if it contains at least
    # one sign relevant to our classifier.
    if supported_objects:

        usable_images.append({
            "image_path": image_path,
            "annotation_path": annotation_path,
            "width": annotation["width"],
            "height": annotation["height"],
            "objects": supported_objects
        })

        supported_object_count += len(supported_objects)


supported_label_counts = Counter()

for item in usable_images:
    for obj in item["objects"]:
        supported_label_counts[obj["label"]] += 1


print("\n" + "=" * 60)
print("MTSD SUPPORTED-SIGN ANALYSIS")
print("=" * 60)

print("Total MTSD images:", len(image_files))
print("Usable images:", len(usable_images))
print("Supported sign objects:", supported_object_count)


print("\n" + "=" * 60)
print("SUPPORTED LABEL DISTRIBUTION")
print("=" * 60)

for label, count in supported_label_counts.most_common():
    print(f"{count:5}  {label}")



category_counts = Counter()

for item in usable_images:
    for obj in item["objects"]:
        category = get_sign_category(obj["label"])
        category_counts[category] += 1


print("\n" + "=" * 60)
print("GROUPED SIGN CATEGORY DISTRIBUTION")
print("=" * 60)

for category, count in category_counts.most_common():
    print(f"{count:5}  {category}")

print("\nTotal supported categories:", len(category_counts))


# ---------------------------------------------------------
# Select a balanced subset of MTSD images
# ---------------------------------------------------------

TARGET_IMAGES = 800
random.seed(42)

# Shuffle so that selection is reproducible but not dependent
# on the original file ordering.
shuffled_images = usable_images.copy()
random.shuffle(shuffled_images)

selected_images = []
selected_ids = set()
selected_category_counts = Counter()

# First pass:
# Try to collect up to this many images for every category.
PER_CATEGORY_TARGET = 30

for category in category_counts:

    category_candidates = []

    for item in shuffled_images:

        categories_in_image = {
            get_sign_category(obj["label"])
            for obj in item["objects"]
        }

        if category in categories_in_image:
            category_candidates.append(item)

    for item in category_candidates:

        image_id = item["image_path"].stem

        if image_id in selected_ids:
            continue

        if selected_category_counts[category] >= PER_CATEGORY_TARGET:
            break

        selected_images.append(item)
        selected_ids.add(image_id)

        # One image may contain several supported sign categories.
        categories_in_image = {
            get_sign_category(obj["label"])
            for obj in item["objects"]
        }

        for image_category in categories_in_image:
            selected_category_counts[image_category] += 1


# Second pass:
# Fill the remaining positions until we reach 800 images.
for item in shuffled_images:

    if len(selected_images) >= TARGET_IMAGES:
        break

    image_id = item["image_path"].stem

    if image_id in selected_ids:
        continue

    selected_images.append(item)
    selected_ids.add(image_id)

    categories_in_image = {
        get_sign_category(obj["label"])
        for obj in item["objects"]
    }

    for category in categories_in_image:
        selected_category_counts[category] += 1


print("\n" + "=" * 60)
print("BALANCED SUBSET SELECTION")
print("=" * 60)

print("Target images:", TARGET_IMAGES)
print("Selected images:", len(selected_images))

print("\nCategory coverage in selected images:")

for category, count in sorted(
    selected_category_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{count:5}  {category}")


# ---------------------------------------------------------
# Create the selected MTSD YOLO dataset
# ---------------------------------------------------------

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

print("\nCreating selected MTSD YOLO dataset...")


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert MTSD xmin/ymin/xmax/ymax coordinates
    into normalized YOLO coordinates.
    """

    xmin = float(bbox["xmin"])
    ymin = float(bbox["ymin"])
    xmax = float(bbox["xmax"])
    ymax = float(bbox["ymax"])

    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


total_labels_written = 0

for item in selected_images:

    source_image = item["image_path"]

    destination_image = OUTPUT_IMAGE_DIR / source_image.name

    # Copy the original MTSD image.
    shutil.copy2(source_image, destination_image)

    image_width = item["width"]
    image_height = item["height"]

    label_lines = []

    for obj in item["objects"]:

        bbox = obj["bbox"]

        x_center, y_center, box_width, box_height = (
            convert_bbox_to_yolo(
                bbox,
                image_width,
                image_height
            )
        )

        # One-class YOLO detector:
        # class 0 = traffic_sign
        label_lines.append(
            f"0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    label_path = OUTPUT_LABEL_DIR / f"{source_image.stem}.txt"

    with open(label_path, "w", encoding="utf-8") as file:
        file.write("\n".join(label_lines))

    total_labels_written += len(label_lines)


print("\n" + "=" * 60)
print("MTSD YOLO DATASET CREATED")
print("=" * 60)

print("Images copied:", len(selected_images))
print("YOLO label files:", len(selected_images))
print("Bounding boxes written:", total_labels_written)

print("\nImages:")
print(OUTPUT_IMAGE_DIR)

print("\nLabels:")
print(OUTPUT_LABEL_DIR)