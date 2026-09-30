from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "gtsdb"
    / "yolo_dataset"
    / "images"
)

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "gtsdb"
    / "yolo_dataset"
    / "labels"
)

def show_yolo_image(split, image_name):

    image_path = IMAGE_DIR / split / image_name
    label_path = LABEL_DIR / split / (
        Path(image_name).stem + ".txt"
    )

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    with open(label_path, "r") as file:

        lines = file.readlines()

    print("Image:", image_name)
    print("Image size:", width, "x", height)
    print("Number of detected labels:", len(lines))

    for line in lines:

        parts = line.strip().split()

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert YOLO normalized values
        # back into pixel coordinates

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2

        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline="red",
            width=4
        )

        draw.text(
            (x_min, y_min - 15),
            f"traffic_sign ({class_id})",
            fill="red"
        )

    plt.figure(figsize=(14, 8))

    plt.imshow(image)

    plt.axis("off")

    plt.title(
        f"{image_name} - {len(lines)} traffic sign(s)"
    )

    plt.show()


show_yolo_image(
    "train",
    "00000.png"
)