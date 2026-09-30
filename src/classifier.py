from pathlib import Path
import json

import numpy as np
from PIL import Image
from keras.models import load_model


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier"
    / "traffic_sign_cnn.keras"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "models"
    / "classifier"
    / "class_names.json"
)


# ==================================================
# Load CNN Model
# ==================================================

print("Loading CNN traffic-sign classifier...")

model = load_model(MODEL_PATH)

print("CNN classifier loaded successfully!")


# ==================================================
# Load Class Names
# ==================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)


# ==================================================
# Image Preprocessing
# ==================================================

def preprocess_image(image):

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")

    # Resize to CNN input size
    image = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Reproduce the exact preprocessing used during training
    grayscale = np.sum(
        image_array / 3,
        axis=2,
        keepdims=True
    )

    normalized = (
        grayscale - 128
    ) / 128

    # Add batch dimension
    processed = np.expand_dims(
        normalized,
        axis=0
    )

    return processed


# ==================================================
# Traffic Sign Classification
# ==================================================

def predict_traffic_sign(image):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Get Top 3 predictions
    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "class_id": int(index),

            "sign_name":
                class_names[index],

            "confidence":
                float(
                    predictions[index] * 100
                )
        })

    return results