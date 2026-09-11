"""
Inspectra AI — TensorFlow / Keras Pipeline Configuration
========================================================
Centralized configuration parameters for Keras model training, evaluation,
data pipeline preparation, and inference serving.
Reads dynamically from `training_config.yaml` with robust fallback defaults.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Base project directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_KERAS_DIR: Path = PROJECT_ROOT / "models" / "keras"
OUTPUTS_KERAS_DIR: Path = PROJECT_ROOT / "outputs" / "keras"
TRAINING_CONFIG_YAML_PATH: Path = PROJECT_ROOT / "training_config.yaml"

# Ensure directories exist
MODELS_KERAS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_KERAS_DIR.mkdir(parents=True, exist_ok=True)


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Robust, lightweight YAML parser using pure Python standard library.
    Handles nested sections, primitives (int, float, bool, str), and lists.
    Used when PyYAML is not installed.
    """
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    result: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for line in text.splitlines():
        raw_line = line.rstrip()
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())

        # Top-level key
        if indent == 0 and ":" in stripped:
            parts = stripped.split(":", 1)
            key = parts[0].strip()
            val = parts[1].split("#")[0].strip()
            current_section = key
            current_subsection = None
            if val:
                result[key] = _cast_yaml_value(val)
            else:
                result[key] = {}
        # 1st level indent (nested dict or list item)
        elif indent == 2 and current_section:
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip('"').strip("'")
                if not isinstance(result[current_section], list):
                    result[current_section] = []
                result[current_section].append(val)
            elif ":" in stripped:
                parts = stripped.split(":", 1)
                sub_key = parts[0].strip()
                sub_val = parts[1].split("#")[0].strip()
                current_subsection = sub_key
                if not isinstance(result[current_section], dict):
                    result[current_section] = {}
                if sub_val:
                    result[current_section][sub_key] = _cast_yaml_value(sub_val)
                else:
                    result[current_section][sub_key] = {}
        # 2nd level indent (deep nested dict e.g. callbacks.early_stopping.patience)
        elif indent == 4 and current_section and current_subsection:
            if ":" in stripped:
                parts = stripped.split(":", 1)
                deep_key = parts[0].strip()
                deep_val = parts[1].split("#")[0].strip()
                if isinstance(result[current_section].get(current_subsection), dict):
                    result[current_section][current_subsection][deep_key] = _cast_yaml_value(deep_val)

    return result


def _cast_yaml_value(val_str: str) -> Any:
    """Helper to convert YAML string literal to Python typed value."""
    cleaned = val_str.strip().strip('"').strip("'")
    if cleaned.lower() == "true":
        return True
    if cleaned.lower() == "false":
        return False
    if cleaned.lower() in ("null", "none"):
        return None
    # List syntax [a, b, c]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        items = cleaned[1:-1].split(",")
        return [_cast_yaml_value(item.strip()) for item in items if item.strip()]
    # Numbers
    try:
        if "." in cleaned or "e" in cleaned.lower():
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def load_training_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration dictionary from training_config.yaml."""
    target_path = config_path or TRAINING_CONFIG_YAML_PATH
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            return parse_simple_yaml(content)
        except Exception as e:
            print(f"[Config] Warning: Could not parse {target_path}: {e}")
    return {}


# Load YAML dictionary
_CFG = load_training_config()

# -----------------------------------------------------------------------------
# Defect Taxonomy & Display Labels
# -----------------------------------------------------------------------------
DEFECT_CLASSES: List[str] = _CFG.get("classes") or [
    "Normal",
    "Crack",
    "Scratch",
    "Dent",
    "Stain",
    "Discoloration",
    "Dimensional Irregularity"
]

NUM_CLASSES: int = len(DEFECT_CLASSES)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(DEFECT_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(DEFECT_CLASSES)}

# -----------------------------------------------------------------------------
# Model & Architecture Configuration
# -----------------------------------------------------------------------------
_model_cfg = _CFG.get("model", {})
MODEL_NAME: str = _model_cfg.get("name", "EfficientNetB0")

# Handle image_size as integer (e.g., 224) or list/tuple (e.g., [224, 224])
_raw_img_size = _CFG.get("image_size") or _model_cfg.get("image_size", 224)
if isinstance(_raw_img_size, (list, tuple)) and len(_raw_img_size) == 2:
    IMAGE_SIZE: Tuple[int, int] = (int(_raw_img_size[0]), int(_raw_img_size[1]))
elif isinstance(_raw_img_size, (int, float)):
    IMAGE_SIZE: Tuple[int, int] = (int(_raw_img_size), int(_raw_img_size))
else:
    IMAGE_SIZE: Tuple[int, int] = (224, 224)

INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
DROPOUT_RATE: float = float(_model_cfg.get("dropout_rate", 0.30))

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
_training_cfg = _CFG.get("training", {})
BATCH_SIZE: int = int(_CFG.get("batch_size") or _training_cfg.get("batch_size", 32))
EPOCHS: int = int(_CFG.get("epochs") or _training_cfg.get("epochs", 50))
FINE_TUNE_EPOCHS: int = int(_training_cfg.get("fine_tune_epochs", 15))
INITIAL_LEARNING_RATE: float = float(
    _CFG.get("learning_rate") or _training_cfg.get("initial_learning_rate", 1e-3)
)
FINE_TUNE_LEARNING_RATE: float = float(_training_cfg.get("fine_tune_learning_rate", 1e-4))

# -----------------------------------------------------------------------------
# Data Splits & Paths
# -----------------------------------------------------------------------------
_dataset_cfg = _CFG.get("dataset", {})
VAL_SPLIT: float = float(_CFG.get("val_split") or _dataset_cfg.get("val_split", 0.15))
TEST_SPLIT: float = float(_CFG.get("test_split") or _dataset_cfg.get("test_split", 0.15))
TRAIN_SPLIT: float = float(
    _dataset_cfg.get("train_split", max(0.0, 1.0 - (VAL_SPLIT + TEST_SPLIT)))
)
RANDOM_SEED: int = int(_dataset_cfg.get("random_seed", 42))

# -----------------------------------------------------------------------------
# Confidence & Quality Gate Thresholds
# -----------------------------------------------------------------------------
_inference_cfg = _CFG.get("inference", {})
KERAS_CONFIDENCE_THRESHOLD: float = float(
    _CFG.get("confidence_threshold") or _inference_cfg.get("confidence_threshold", 0.60)
)
UNCERTAIN_LABEL: str = _inference_cfg.get("uncertain_label", "Uncertain prediction")

# -----------------------------------------------------------------------------
# File Artifact Paths
# -----------------------------------------------------------------------------
_paths_cfg = _CFG.get("paths", {})
SAVED_MODEL_PATH: Path = PROJECT_ROOT / _paths_cfg.get("saved_model", "models/keras/defect_classifier.keras")
COMBINED_SAVED_MODEL_PATH: Path = PROJECT_ROOT / "models" / "keras" / "defect_classifier_combined.keras"
CLASS_NAMES_PATH: Path = PROJECT_ROOT / _paths_cfg.get("class_names_json", "models/keras/class_names.json")
COMBINED_CLASS_NAMES_PATH: Path = PROJECT_ROOT / "models" / "keras" / "class_names_combined.json"
MODEL_CONFIG_PATH: Path = PROJECT_ROOT / _paths_cfg.get("model_config_json", "models/keras/model_config.json")
COMBINED_MODEL_CONFIG_PATH: Path = PROJECT_ROOT / "models" / "keras" / "model_config_combined.json"
TRAINING_HISTORY_PATH: Path = PROJECT_ROOT / _paths_cfg.get("training_history_json", "models/keras/training_history.json")
DATASET_VERSION_PATH: Path = PROJECT_ROOT / "models" / "keras" / "dataset_version.json"

# Evaluation Output Paths
CLASSIFICATION_REPORT_PATH: Path = PROJECT_ROOT / _paths_cfg.get("classification_report_json", "outputs/keras/classification_report.json")
CONFUSION_MATRIX_PATH: Path = PROJECT_ROOT / _paths_cfg.get("confusion_matrix_png", "outputs/keras/confusion_matrix.png")
COMBINED_CONFUSION_MATRIX_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "confusion_matrix_combined.png"
TRAINING_HISTORY_PLOT_PATH: Path = PROJECT_ROOT / _paths_cfg.get("training_history_png", "outputs/keras/training_history.png")
EVALUATION_METRICS_PATH: Path = PROJECT_ROOT / _paths_cfg.get("evaluation_json", "outputs/keras/evaluation.json")
BASELINE_METRICS_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "baseline_metrics.json"
COMBINED_METRICS_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "combined_metrics.json"
COMPARISON_REPORT_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "comparison_report.json"
DATASET_MERGE_REPORT_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "dataset_merge_report.json"
CLASS_DISTRIBUTION_PATH: Path = PROJECT_ROOT / "outputs" / "keras" / "class_distribution.json"
