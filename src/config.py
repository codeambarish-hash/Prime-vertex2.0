"""
Central Configuration Module for Vision-Based Defect Detection
==============================================================
Defines filesystem paths, taxonomy classes, image resolution standards,
and environmental augmentation parameters for industrial manufacturing inspection.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core Directory Hierarchy
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Defect Taxonomy and Class Mappings
# -----------------------------------------------------------------------------
# 0 = Normal/Pristine baseline, 1-6 = Specific Manufacturing Anomaly Classes
DEFECT_CLASSES: List[str] = [
    "normal",
    "crack",
    "scratch",
    "dent",
    "stain",
    "discoloration",
    "dimensional_irregularity",
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(DEFECT_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(DEFECT_CLASSES)}
NUM_CLASSES: int = len(DEFECT_CLASSES)

# Color palettes for bounding box & segmentation mask visualization (RGB format)
DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "normal": (34, 197, 94),                   # Green
    "crack": (239, 68, 68),                    # Red
    "scratch": (249, 115, 22),                 # Orange
    "dent": (234, 179, 8),                     # Amber / Yellow
    "stain": (168, 85, 247),                   # Purple
    "discoloration": (59, 130, 246),           # Blue
    "dimensional_irregularity": (236, 72, 153), # Magenta / Pink
}

# -----------------------------------------------------------------------------
# Image and Processing Specifications
# -----------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
IMAGE_CHANNELS: int = 3
DEFAULT_SEED: int = 42

# Dataset Split Ratios
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# -----------------------------------------------------------------------------
# MVTec Anomaly Detection (AD) Specifications
# -----------------------------------------------------------------------------
# MVTec AD is the industry standard benchmark containing 15 categories (5 textures, 10 objects)
# Each category contains pristine 'good' training images and defective test images with pixel masks.
MVTEC_CATEGORIES: List[str] = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Official MVTec download mirror base URL
MVTEC_BASE_URL: str = "https://www.mydrive.ch/shares/38536/3830184030e49fe7474e85d91d0f122c/download"
