from pathlib import Path

from ultralytics import YOLO


# ==================================================
# Project Root
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# Paths
# ==================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "detector"
    / "best.pt"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "external_evaluation"
    / "detector_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Check Paths
# ==================================================

print("Detector model:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

print("External test folder:", TEST_DIR)
print("Test folder exists:", TEST_DIR.exists())


# ==================================================
# Find Test Images
# ==================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}

test_images = sorted(
    [
        image
        for image in TEST_DIR.iterdir()
        if image.suffix.lower() in image_extensions
    ]
)

print(
    "\nExternal test images:",
    len(test_images)
)


# ==================================================
# Load Detector
# ==================================================

print(
    "\nLoading trained traffic-sign detector..."
)

model = YOLO(
    str(MODEL_PATH)
)

print(
    "Detector loaded successfully!"
)


# ==================================================
# Test Images
# ==================================================

total_images = len(test_images)

images_with_detections = 0
images_without_detections = 0
total_detections = 0


print("\nStarting external testing...")
print("=" * 60)


for image_path in test_images:

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device="cpu",
        verbose=False
    )

    result = results[0]

    number_of_detections = len(
        result.boxes
    )

    print(
        f"\nImage: {image_path.name}"
    )

    print(
        f"Traffic signs detected: "
        f"{number_of_detections}"
    )


    # ----------------------------------------------
    # No Detection
    # ----------------------------------------------

    if number_of_detections == 0:

        images_without_detections += 1

        print(
            "Result: NO DETECTION"
        )


    # ----------------------------------------------
    # Detections Found
    # ----------------------------------------------

    else:

        images_with_detections += 1

        total_detections += (
            number_of_detections
        )

        for i, box in enumerate(
            result.boxes,
            start=1
        ):

            confidence = (
                float(box.conf[0]) * 100
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            print(
                f"Detection {i}: "
                f"{confidence:.2f}% confidence"
            )

            print(
                "Bounding box: "
                f"({x1:.0f}, {y1:.0f}) -> "
                f"({x2:.0f}, {y2:.0f})"
            )


    # ----------------------------------------------
    # Save Annotated Image
    # ----------------------------------------------

    result.save(
        filename=str(
            OUTPUT_DIR
            / image_path.name
        )
    )


# ==================================================
# Final Summary
# ==================================================

print("\n" + "=" * 60)

print(
    "EXTERNAL TEST SUMMARY"
)

print("=" * 60)

print(
    f"Total images: {total_images}"
)

print(
    f"Images with detections: "
    f"{images_with_detections}"
)

print(
    f"Images without detections: "
    f"{images_without_detections}"
)

print(
    f"Total detected objects: "
    f"{total_detections}"
)

print(
    "\nAnnotated images saved to:"
)

print(OUTPUT_DIR)

print(
    "\nExternal testing completed!"
)