import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_YAML = BASE_DIR / "traffic_sign_gtsdb.yaml"
MODEL_PATH = PROJECT_DIR / "yolo26n.pt"

print("Dataset YAML:", DATA_YAML)
print("Dataset YAML exists:", DATA_YAML.exists())

print("Model:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")

print("\nStarting dataset validation...")

results = model.val(
    data=str(DATA_YAML),
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0
)

print("\nDataset validation completed!")