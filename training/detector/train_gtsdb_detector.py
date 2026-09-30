from pathlib import Path
from ultralytics import YOLO
import yaml

# -----------------------------
# Project paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_YAML = BASE_DIR / "traffic_sign_gtsdb.yaml"
DATASET_DIR = PROJECT_ROOT / "data" / "gtsdb" / "yolo_dataset"

MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "detector"

with open(DATA_YAML, "r", encoding="utf-8") as file:
    data_config = yaml.safe_load(file)

data_config["path"] = str(DATASET_DIR)

# -----------------------------
# Check paths
# -----------------------------

print("Dataset YAML:", DATA_YAML)
print("Dataset exists:", DATA_YAML.exists())

print("Pretrained model:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

print("Training results:", RESULTS_DIR)

# -----------------------------
# Load pretrained YOLO26n
# -----------------------------

print("\nLoading pretrained YOLO26n...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Model task:", model.task)

# -----------------------------
# Training configuration
# -----------------------------

print("\nReady for traffic-sign detector training.")


# -----------------------------
# Train the detector
# -----------------------------

results = model.train(
    data=data_config,

    # Input image size
    imgsz=640,

    # Training
    epochs=50,
    batch=4,

    # Stop if validation does not improve
    patience=10,

    # CPU training
    device="cpu",
    workers=0,

    # Save results inside results/detector
    project=str(RESULTS_DIR),
    name="traffic_sign_yolo26n",

    # Keep existing run instead of creating confusing names
    exist_ok=True,

    # Save trained model
    save=True,

    # Show training information
    verbose=True
)

print("\nTraining completed!")