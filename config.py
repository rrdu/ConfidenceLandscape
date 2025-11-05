#config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_ROOT = BASE_DIR / "data" / "images"

#############################################################
#Models to use
MODELS = [
    "ResNet18",
    "Mobilenet_V3_Large",
    "EfficientNet_B0"
]
#############################################################
#Image classes
IMAGE_CLASSES = [
    "Airplane",
    "Cat",
    "Dalmatian",
    "Jaguar",
    "School Bus"
]

CLASS_DIR = {
    "Airplane":   "airplane",
    "Cat":        "cat",
    "Dalmatian":  "dalmatian",
    "Jaguar":     "jaguar",       
    "School Bus": "school_bus",
}
#############################################################
#Perturbation metrics
PERTURB_AXES = {
    "Rotation":  {"min": -40, "max": 40, "steps": 33, "label": "Rotation (°)"},
    "Brightness":{"min": 0.5, "max": 1.5, "steps": 33, "label": "Brightness"},
    "Noise":     {"min": 0.0, "max": 0.20, "steps": 33, "label": "Noise σ"},
}