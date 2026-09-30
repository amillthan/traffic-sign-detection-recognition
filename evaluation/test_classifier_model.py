from pathlib import Path
import json
import numpy as np
from keras.models import load_model


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load saved CNN model
model = load_model(PROJECT_ROOT / "models" / "classifier" / "traffic_sign_cnn.keras")

# Load traffic sign class names
with open(PROJECT_ROOT / "models" / "classifier" / "class_names.json", "r") as file:
    class_names = json.load(file)

print("Model loaded successfully!")
print("Number of classes:", len(class_names))
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
