from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from src.classifier import predict_traffic_sign


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETECTOR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "detector"
    / "best.pt"
)


# ==================================================
# Load YOLO Detector
# ==================================================

print("Loading traffic-sign detector...")

detector = YOLO(str(DETECTOR_MODEL_PATH))

print("Traffic-sign detector loaded successfully!")


# ==================================================
# Detect and Classify
# ==================================================

def detect_and_classify(
    image,
    detection_confidence=0.25,
    padding_ratio=0.15
):

    # --------------------------------------------------
    # Open Image
    # --------------------------------------------------

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")


    # --------------------------------------------------
    # YOLO Detection
    # --------------------------------------------------

    detection_results = detector.predict(
        source=image,
        imgsz=640,
        conf=detection_confidence,
        iou=0.45,
        device="cpu",
        verbose=False
    )

    result = detection_results[0]

    final_results = []


    # --------------------------------------------------
    # Process Each Detection
    # --------------------------------------------------

    for box in result.boxes:

        detection_confidence_score = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # --------------------------------------------------
        # Add 15% Padding Around YOLO Crop
        # --------------------------------------------------

        box_width = x2 - x1
        box_height = y2 - y1

        padding_x = int(
            box_width * padding_ratio
        )

        padding_y = int(
            box_height * padding_ratio
        )

        image_width, image_height = image.size

        crop_x1 = max(
            0,
            x1 - padding_x
        )

        crop_y1 = max(
            0,
            y1 - padding_y
        )

        crop_x2 = min(
            image_width,
            x2 + padding_x
        )

        crop_y2 = min(
            image_height,
            y2 + padding_y
        )


        # --------------------------------------------------
        # Crop Traffic Sign
        # --------------------------------------------------

        cropped_sign = image.crop(
            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2
            )
        )


        # --------------------------------------------------
        # CNN Classification
        # --------------------------------------------------

        classifications = predict_traffic_sign(
            cropped_sign
        )

        best_classification = classifications[0]


        # --------------------------------------------------
        # Store Result
        # --------------------------------------------------

        final_results.append({

            "bounding_box":
                (x1, y1, x2, y2),

            "detection_confidence":
                detection_confidence_score * 100,

            "class_id":
                best_classification["class_id"],

            "sign_name":
                best_classification["sign_name"],

            "classification_confidence":
                best_classification["confidence"],

            "top_3":
                classifications,

            "crop":
                cropped_sign
        })


    return final_results