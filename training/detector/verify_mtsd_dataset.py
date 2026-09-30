from pathlib import Path
from PIL import Image, ImageDraw
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SELECTED_MTSD_DIR = (
    PROJECT_ROOT
    / "data"
    / "mtsd"
    / "selected_training"
)

IMAGE_DIR = (
    SELECTED_MTSD_DIR
    / "images"
)

LABEL_DIR = (
    SELECTED_MTSD_DIR
    / "labels"
)

OUTPUT_DIR = (
    SELECTED_MTSD_DIR
    / "verification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get all images
image_files = list(IMAGE_DIR.glob("*.jpg"))

# Select 10 reproducible random images
random.seed(42)
sample_images = random.sample(image_files, min(10, len(image_files)))

for image_path in sample_images:

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            class_id, x_center, y_center, box_width, box_height = map(
                float, line.strip().split()
            )

            # Convert normalized YOLO coordinates back to pixels
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            xmin = x_center - box_width / 2
            ymin = y_center - box_height / 2
            xmax = x_center + box_width / 2
            ymax = y_center + box_height / 2

            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline="red",
                width=5
            )

    output_path = OUTPUT_DIR / image_path.name
    image.save(output_path)

    print("Verified:", image_path.name)

print("\nVerification images saved to:")
print(OUTPUT_DIR)