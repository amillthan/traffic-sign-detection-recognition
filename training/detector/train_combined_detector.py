from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_YAML = BASE_DIR / "traffic_sign_combined.yaml"
MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "detector"

DATASET_DIR = PROJECT_ROOT / "data" / "combined_dataset"

with open(DATA_YAML, "r", encoding="utf-8") as file:
    data_config = yaml.safe_load(file)

data_config["path"] = str(DATASET_DIR)

print("=" * 60)
print("TRAFFIC SIGN DETECTOR - COMBINED DATASET")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )
else:
    print("CUDA GPU is not available.")
    print("Training will use CPU.")

print("\nDataset:", DATA_YAML)
print("Starting model:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

results = model.train(
    data=data_config,

    epochs=50,
    imgsz=640,

    # MX550 has limited VRAM.
    # Start conservatively with one image per batch.
    batch=4,

    device="cpu",
    workers=0,

    patience=10,

    project=str(RESULTS_DIR),
    name="traffic_sign_combined_cpu",

    exist_ok=False
)