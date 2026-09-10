import React, { useState } from 'react';
import { Code2, Copy, Check, FileText, Folder, FileCode } from 'lucide-react';

interface FileDefinition {
  path: string;
  name: string;
  type: 'python' | 'text' | 'markdown';
  description: string;
  code: string;
}

const PROJECT_FILES: FileDefinition[] = [
  {
    path: 'requirements.txt',
    name: 'requirements.txt',
    type: 'text',
    description: 'All required deep learning and computer vision dependencies with consistent PyTorch stack',
    code: `# Core Deep Learning & Computer Vision (Consistent PyTorch Stack)
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-image>=0.21.0

# Robustness Augmentations (Lighting, Orientation, Backgrounds)
albumentations>=1.3.1
scipy>=1.11.0
pillow>=10.0.0

# Visualization, Anomaly Heatmaps & Reporting
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0

# Dataset acquisition & utilities
requests>=2.31.0`
  },
  {
    path: 'src/config.py',
    name: 'config.py',
    type: 'python',
    description: 'Central project paths, 7-class defect taxonomy, color maps, and MVTec categories',
    code: `from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Defect Taxonomy (0 = Normal, 1-6 = Manufacturing Defect Classes)
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

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15`
  },
  {
    path: 'src/preprocessing.py',
    name: 'preprocessing.py',
    type: 'python',
    description: 'Inspectra AI Preprocessing: Aspect-ratio letterbox, LAB CLAHE, Bilateral Denoising, Albumentations & Matplotlib visualization',
    code: `"""
Inspectra AI — AI Powered Visual Inspection
Module: src/preprocessing.py
"""
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt

def resize_and_pad(image, target_size=(256, 256), pad_color=(0, 0, 0), mask=None):
    """Preserves aspect ratio, resizes isotropically, and symmetrically letterboxes."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded, None, {"scale": scale, "pad_top": pad_top, "pad_left": pad_left}

def normalize_lighting_clahe(image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Normalizes lighting by applying CLAHE strictly on L-channel in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

def denoise_bilateral(image, d=7, sigma_color=50.0, sigma_space=50.0):
    """Edge-preserving filter: smooths grain while keeping hairline cracks and scratches sharp."""
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def normalize_background(image, bg_fill_color=(15, 15, 20)):
    """Segment product foreground and standardize conveyor belt to uniform neutral tone."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, fg_mask = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological closing to fill cavities
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bg_canvas = np.full_like(image, bg_fill_color)
    return np.where(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB) == 255, image, bg_canvas), fg_mask

def visualize_preprocessing_batch(sample_images, sample_titles=None, save_path="outputs/preprocessing_comparison.png"):
    """Renders side-by-side comparative inspection report using Matplotlib."""
    fig, axes = plt.subplots(len(sample_images), 5, figsize=(16, 3.2 * len(sample_images)))
    # [Raw Input | Background Norm | LAB CLAHE | Bilateral Denoised | Letterbox 256x256]
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()`
  },
  {
    path: 'src/synthetic_generator.py',
    name: 'synthetic_generator.py',
    type: 'python',
    description: 'OpenCV procedural generation for substrates, 6 defect types, ground truth masks & lighting',
    code: `import math, random
import numpy as np
import cv2

class SyntheticSurfaceGenerator:
    """Generates brushed metal, cast iron, machined parts, and ceramic substrates."""
    @staticmethod
    def generate_brushed_metal(width: int, height: int) -> np.ndarray:
        base_val = np.random.randint(160, 200)
        noise = np.random.normal(0, 18, (height, width)).astype(np.float32)
        kernel = np.ones((1, 45), dtype=np.float32) / 45
        brushed = cv2.filter2D(noise, -1, kernel)
        return cv2.cvtColor(np.clip(base_val + brushed, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

class SyntheticDefectInjector:
    """Injects cracks, scratches, dents, stains, discoloration & dimensional irregularities with pixel masks."""
    @staticmethod
    def inject_crack(image: np.ndarray):
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # Random walk fractal crack with shadow furrow & specular edge
        # Returns (sample_image, binary_mask, info_dict)
        ...`
  },
  {
    path: 'src/prepare_dataset.py',
    name: 'prepare_dataset.py',
    type: 'python',
    description: 'Dual-mode dataset preparation (Synthetic or MVTec AD) with stats printing & splitting',
    code: `import argparse, json
from src.synthetic_generator import generate_synthetic_sample
from src.mvtec_downloader import download_and_extract_mvtec_category

def compute_and_print_dataset_stats(records):
    """Outputs tabular class counts, split counts, and mask coverage area metrics."""
    ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "mvtec"], default="synthetic")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()
    ...`
  },
  {
    path: 'src/model.py',
    name: 'model.py',
    type: 'python',
    description: 'PyTorch Multi-Task Architecture: Binary Anomaly + 7-Class Classifier + U-Net Localization Head',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class DefectInspectionNet(nn.Module):
    """Multi-Task Joint Architecture:
    - Shared Multi-Scale Convolutional Feature Encoder
    - Binary Anomaly Detection Head (Sigmoid logit)
    - Multi-Class Defect Categorization Head (Softmax logits for 7 classes)
    - Pixel-Level Defect Localization Decoder (U-Net skip connections producing [B, 1, H, W] mask)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super().__init__()
        ...`
  },
  {
    path: 'src/anomaly_detection.py',
    name: 'anomaly_detection.py',
    type: 'python',
    description: 'Convolutional Autoencoder (trained only on normal), MSE/SSIM threshold calibration & Transfer Learning backup',
    code: `import math, torch, cv2, numpy as np
import torch.nn as nn
from src.preprocessing import InspectraPreprocessor

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder trained EXCLUSIVELY on normal defect-free workpieces."""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = AutoencoderConvBlock(in_channels, 32)
        self.enc2 = AutoencoderConvBlock(32, 64)
        self.enc3 = AutoencoderConvBlock(64, 128)
        self.enc4 = AutoencoderConvBlock(128, 256)
        self.enc5 = AutoencoderConvBlock(256, 512)
        self.bottleneck_enc = nn.Conv2d(512, latent_dim, 1)
        self.bottleneck_dec = nn.Conv2d(latent_dim, 512, 1)
        # Decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.dec5 = AutoencoderDeconvBlock(512, 256)
        self.dec4 = AutoencoderDeconvBlock(256, 128)
        self.dec3 = AutoencoderDeconvBlock(128, 64)
        self.dec2 = AutoencoderDeconvBlock(64, 32)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.bottleneck_enc(self.enc5(self.enc4(self.enc3(self.enc2(self.enc1(x))))))
        return self.dec1(self.dec2(self.dec3(self.dec4(self.dec5(self.bottleneck_dec(z))))))

class AnomalyDetector:
    """Production Anomaly Detector: autoencoder or transfer_learning (ResNet/EfficientNet)."""
    def __init__(self, method='autoencoder', error_metric='hybrid', threshold_k=2.8):
        self.method = method
        self.model = ConvAutoencoder() if method == 'autoencoder' else PretrainedBinaryClassifier()
        self.threshold = 0.05
    
    def predict(self, image):
        # Returns (label: 'defective'/'normal', anomaly_score, confidence, residual_map)
        ...`
  },
  {
    path: 'src/defect_classifier.py',
    name: 'defect_classifier.py',
    type: 'python',
    description: 'Multi-class CNN transfer learning (EfficientNetB0/MobileNetV2), Focal Loss, Albumentations, and Confusion Matrix',
    code: `import torch, torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import albumentations as A

class DefectClassifier(nn.Module):
    """Multi-Class Transfer Learning CNN (EfficientNetB0 / MobileNetV2) for 7 defect classes."""
    def __init__(self, backbone_name="efficientnet_b0", num_classes=7, pretrained=True):
        super().__init__()
        # Backbone feature extractor + Custom Industrial MLP Head
        self.features = ... # EfficientNet-B0 or MobileNetV2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes) # crack, scratch, dent, stain, discoloration, dim_irreg, normal
        )

    def freeze_backbone(self): ...
    def unfreeze_backbone(self): ...

class MultiClassFocalLoss(nn.Module):
    """Focal loss FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) for severe class imbalance."""
    ...

def fit_defect_classifier(model, train_loader, val_loader, epochs_frozen=3, epochs_fine_tune=5):
    # Early stopping + ReduceLROnPlateau / CosineAnnealingLR
    ...

def plot_confusion_matrix_and_metrics(cm, report, save_path="outputs/defect_classifier_evaluation.png"):
    # 3-panel dashboard: Confusion Matrix Heatmap, Precision/Recall/F1 bars, Loss/F1 curves
    ...`
  },
  {
    path: 'src/inspect.py',
    name: 'inspect.py',
    type: 'python',
    description: 'Inference pipeline: extracts bounding boxes [x, y, w, h] and renders 4-panel report',
    code: `import cv2
import numpy as np

def overlay_defect_localization(image, mask, predicted_class, confidence, bbox=None):
    """Draws colored heatmap overlay, contour boundaries, and CAD-style corner notches."""
    ...

def create_inspection_quad_report(raw_image, mask, predicted_class, confidence, bbox=None):
    """Builds a 2x2 grid: [Raw Sensor | Anomaly Heatmap | Binary Mask | Composite HUD Overlay]"""
    ...`
  },
  {
    path: 'src/localization.py',
    name: 'localization.py',
    type: 'python',
    description: 'Defect localization: Grad-CAM / Grad-CAM++, DefectUNet segmentation, OpenCV contour extraction & mm calibration',
    code: `import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

class GradCAM:
    """Computes Grad-CAM and Grad-CAM++ saliency heatmaps on CNN feature extractors."""
    def __init__(self, model, target_layer=None, method="gradcam_plus_plus"):
        ...
    def generate_heatmap(self, input_tensor, target_class=None):
        ...

class DefectUNet(nn.Module):
    """Lightweight 4-level U-Net segmentation network for pixel-level defect masks."""
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        ...

class LocalizationPostProcessor:
    """Post-processes heatmaps/masks with Otsu/percentile binarization, cv2.findContours, and mm calibration."""
    def __init__(self, pixel_to_mm=0.1, min_area_px=15.0):
        ...

def overlay_localization(image, heatmap, binary_mask, defect_class, confidence, regions, pixel_to_mm=0.1):
    """Overlays bounding boxes, contour boundaries, physical dimension tags, and HUD banner."""
    ...

def create_localization_report(original_image, heatmap, binary_mask, overlay_image, defect_class, confidence, defect_regions, save_path=None, pixel_to_mm=0.1):
    """Generates 4-panel industrial inspection scorecard: Raw | Grad-CAM | Mask | Calibrated HUD."""
    ...`
  },
  {
    path: 'src/refinement.py',
    name: 'refinement.py',
    type: 'python',
    description: 'False-positive & missed-detection reduction: Dual confidence gating, morphological filtering, TTA ensemble voting, and borderline manual review logging',
    code: `import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from src.config import DEFECT_CLASSES, OUTPUTS_DIR

class DecisionStatus(str, Enum):
    CONFIRMED_DEFECT = "CONFIRMED_DEFECT"
    CLEAN_PASS = "CLEAN_PASS"
    SUPPRESSED_FALSE_POSITIVE = "SUPPRESSED_FALSE_POSITIVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"

class DualConfidenceGate:
    """Only flags a defect if BOTH Anomaly Detector AND Classifier agree above thresholds."""
    def evaluate(self, anomaly_score, predicted_class, class_confidence):
        ...

class MorphologicalDefectFilter:
    """Applies opening/closing and purges noise blobs under calibrated min_area_px."""
    def filter_mask(self, mask, min_area_px=25.0):
        ...

class TestTimeAugmentationEngine:
    """Multi-view TTA (H-Flip, V-Flip, Rotations) with inverse coordinate alignment and voting."""
    def evaluate_tta_ensemble(self, image, inference_fn, transforms=None):
        ...

class ManualReviewManager:
    """Logs borderline confidence, model conflict, and split TTA samples to manual_review_queue.json."""
    def check_borderline_criteria(self, confidence, dual_agreed, tta_vote_ratio, defect_area_px):
        ...
    def log_record(self, record):
        ...

class InspectionRefiner:
    """Unified refinement pipeline orchestrating dual-gating, morphology, TTA, and review ledger."""
    def refine_inspection(self, image, anomaly_score, predicted_class, confidence, raw_mask=None):
        ...`
  },
  {
    path: 'src/evaluate.py',
    name: 'evaluate.py',
    type: 'python',
    description: 'Comprehensive evaluation suite: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Mask/Box IoU, and ASCII report generation',
    code: `"""
Evaluation Module for Inspectra AI — AI Powered Visual Inspection
=================================================================
Calculates and plots:
  - Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC
  - Per-class Precision/Recall/F1, Macro F1, Weighted F1
  - KxK Confusion Matrix across 7 industrial defect categories
  - Spatial Mask IoU, Bounding Box IoU, Mean IoU, IoU@0.50 & IoU@0.70
  - Plots ROC Curve, Precision-Recall Curve, Confusion Matrix
  - Generates outputs/evaluation_report.txt and outputs/evaluation_metrics.json
"""

from src.config import DEFECT_CLASSES, OUTPUTS_DIR

def compute_binary_metrics(y_true, y_scores, threshold=0.50):
    """Computes Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC."""
    ...

def compute_multiclass_metrics(y_true, y_pred, class_names=None):
    """Computes Confusion Matrix, Per-class Precision/Recall/F1, Macro & Weighted F1."""
    ...

def compute_mask_iou(pred_mask, gt_mask):
    """Computes Intersection over Union (IoU) and Dice coefficient for segmentation masks."""
    ...

def compute_box_iou(box1, box2):
    """Calculates 2D IoU between two bounding boxes."""
    ...

def compute_localization_metrics(sample_mask_pairs, sample_box_pairs=(), y_true_binary=None):
    """Calculates test set Mean Mask IoU, Defective-only mIoU, Box IoU, and IoU@0.50/0.70."""
    ...

def evaluate_inspection_pipeline(data_source=None, output_dir=OUTPUTS_DIR, decision_threshold=0.50, save_plots=True):
    """Master evaluator executing binary, multiclass, and localization evaluation, plotting curves and saving report."""
    ...`
  },
  {
    path: 'src/pipeline.py',
    name: 'pipeline.py',
    type: 'python',
    description: 'End-to-End Orchestrator: run_inspection(image_path), visualize_inspection_result(), and run_batch_inspection(folder)',
    code: `"""
Inspectra AI — End-to-End Industrial Visual Inspection Pipeline
================================================================
Unified orchestration module integrating all 6 stages of the inspection lifecycle:
  1. Image Ingestion & Standardized Preprocessing (CLAHE, Bilateral Denoising, Letterbox)
  2. Unsupervised Anomaly Detection & Surface Reconstruction Scoring
  3. Multi-Class Deep Defect Classification (7 Industrial Taxonomy Classes)
  4. Spatial Saliency & Bounding Box Localization (Grad-CAM / Contour Analysis)
  5. False-Positive Refinement (Dual-Confidence Gate, Morphological Pruning, TTA Consensus)
  6. Industrial HUD Visualizer & Batch Audit Logging (Annotated PNGs + Summary CSV)
"""

from src.config import DEFECT_CLASSES, OUTPUTS_DIR, DEFAULT_IMAGE_SIZE
from src.refinement import InspectionRefiner, RefinementConfig

def run_inspection(image_path, anomaly_threshold=0.50, pixel_to_mm=0.10):
    """
    Executes end-to-end industrial inspection on a single product image:
    Loads -> Preprocesses -> Anomaly Detection -> Classification & Localization -> Refinement.
    Returns structured result dict:
      {is_defective, defect_type, confidence, bounding_boxes, defect_area, anomaly_score, ...}
    """
    # 1. Load Image
    w, h, rgb_buf = load_inspection_image(image_path)
    # 2. Preprocess (Letterbox + LAB CLAHE + Bilateral Smoothing)
    tw, th, prep_buf, meta = preprocess_image(w, h, rgb_buf, target_size=DEFAULT_IMAGE_SIZE)
    # 3. Anomaly Detection
    anomaly_score, candidate_defective, residual_grid = detect_surface_anomaly(tw, th, prep_buf)
    # 4. Classification & Localization (if defective)
    if candidate_defective or anomaly_score >= anomaly_threshold:
        defect_type, confidence = classify_defect_category(anomaly_score, residual_grid=residual_grid)
        bboxes, area_px, area_mm2 = localize_defect_regions(tw, th, defect_type, anomaly_score, residual_grid=residual_grid)
    else:
        defect_type, confidence = "normal", 1.0 - anomaly_score
        bboxes, area_px, area_mm2 = [], 0.0, 0.0
    # 5. Dual-Confidence Gate & False-Positive Refinement
    refiner = InspectionRefiner(RefinementConfig())
    # Returns structured inspection result
    return {
        "is_defective": is_defective,
        "defect_type": defect_type,
        "confidence": confidence,
        "bounding_boxes": bboxes,
        "defect_area": area_mm2,
        "anomaly_score": anomaly_score,
        "inference_time_ms": latency_ms
    }

def visualize_inspection_result(image_or_path, result, save_path=None, output_dir=OUTPUTS_DIR):
    """Draws color-coded bounding boxes, CAD corner brackets, and telemetry HUD on image, saves to outputs/."""
    ...

def run_batch_inspection(folder_path, output_dir=OUTPUTS_DIR, csv_filename="inspection_summary.csv"):
    """Runs pipeline over a folder of test images, saves annotated PNGs + summary CSV (filename, is_defective, defect_type, confidence, area)."""
    ...`
  },
  {
    path: 'src/app.py',
    name: 'app.py',
    type: 'python',
    description: 'Lightweight Flask Web Application: POST /analyze, GET /report/<batch_id>, and single-page demo studio',
    code: `from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from src.pipeline import run_inspection, visualize_inspection_result

app = Flask(__name__, template_folder="../templates", static_folder="../static")
UPLOAD_DIR = Path("temp/uploads")
REPORT_DIR = Path("outputs/reports")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Accepts 1 or more images, executes run_inspection(), returns base64 annotated visualizations
    uploaded_files = request.files.getlist("files")
    results = []
    for file_obj in uploaded_files:
        save_path = save_and_validate(file_obj)
        res = run_inspection(save_path)
        ann_path = visualize_inspection_result(save_path, res)
        # Base64 encode for direct frontend rendering without extra requests
        b64_ann = base64.b64encode(open(ann_path, "rb").read()).decode("utf-8")
        results.append({
            "filename": file_obj.filename,
            "is_defective": res["is_defective"],
            "defect_type": res["defect_type"],
            "confidence": res["confidence"],
            "bounding_boxes": res["bounding_boxes"],
            "defect_area": res["defect_area"],
            "anomaly_score": res["anomaly_score"],
            "annotated_image_base64": b64_ann
        })
    return jsonify({"batch_id": batch_id, "summary": summary, "results": results})

@app.route("/report/<batch_id>", methods=["GET"])
def download_report(batch_id):
    # Returns downloadable CSV of batch inspection results
    return send_file(csv_path, mimetype="text/csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)`
  },
  {
    path: 'templates/index.html',
    name: 'index.html',
    type: 'markdown',
    description: 'Single-page web interface: drag-and-drop upload, analyze button, telemetry HUD, and batch results table',
    code: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Inspectra AI — Vision-Based Defect Inspection Studio</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <!-- Drag-and-drop / click-to-upload area -->
  <div class="dropzone" id="dropzone">
    <input type="file" id="fileInput" name="files" multiple accept=".jpg,.jpeg,.png,.bmp">
    <h3>Drag & drop inspection images here or click to browse</h3>
  </div>

  <!-- Loading spinner -->
  <div id="loadingCard" class="spinner-container hidden">
    <div class="spinner"></div>
    <p>Running multi-stage defect detection pipeline...</p>
  </div>

  <!-- Single inspection result HUD -->
  <div id="singleResultPanel" class="hidden">
    <img id="annotatedImage" alt="Defect HUD Overlay">
    <div class="metric-card">Status: <span id="singleStatusBadge"></span></div>
    <div class="metric-card">Defect: <span id="singleCategoryPill"></span></div>
    <div class="metric-card">Confidence: <span id="singleConfidenceVal"></span></div>
    <div class="metric-card">Anomaly Score: <span id="singleAnomalyScoreVal"></span></div>
    <div class="metric-card">Area: <span id="singleAreaVal"></span></div>
  </div>

  <!-- Batch results table & CSV download -->
  <div id="batchResultPanel" class="hidden">
    <a id="downloadCsvBtn" href="#">Download CSV Report</a>
    <table id="resultsTable">...</table>
  </div>
  <script src="/static/js/app.js"></script>
</body>
</html>`
  },
  {
    path: 'main.py',
    name: 'main.py',
    type: 'python',
    description: 'CLI entrypoint supporting --input <image_or_folder>, batch processing, final run summary, and --evaluate',
    code: `import argparse
from src.pipeline import run_inspection, run_batch_inspection, visualize_inspection_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspectra AI — End-to-End Visual Inspection")
    parser.add_argument("--input", "-i", type=str, default="data/test_samples", help="Path to image or folder")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation suite")
    args = parser.parse_args()

    if args.evaluate:
        # Run evaluation suite
        ...
    elif Path(args.input).is_file():
        # Single specimen inspection
        res = run_inspection(args.input)
        visualize_inspection_result(args.input, res)
        # Print run summary
    else:
        # Batch folder inspection & summary CSV
        batch = run_batch_inspection(args.input)
        # Print final run summary (processed, defects found, breakdown, avg latency)`
  }
];

export const CodeExplorerTab: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileDefinition>(PROJECT_FILES[0]);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Sidebar: File List (4 cols) */}
        <div className="lg:col-span-4 space-y-2 bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <div className="flex items-center gap-2 px-2 pb-2 border-b border-slate-100">
            <Folder className="w-4 h-4 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              Project Structure
            </h3>
          </div>

          <div className="space-y-1">
            {PROJECT_FILES.map((file) => (
              <button
                key={file.path}
                onClick={() => setSelectedFile(file)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-mono transition-colors flex items-center justify-between ${
                  selectedFile.path === file.path
                    ? 'bg-indigo-50 text-indigo-900 font-bold border border-indigo-200'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <FileCode className={`w-3.5 h-3.5 shrink-0 ${
                    selectedFile.path === file.path ? 'text-indigo-600' : 'text-slate-400'
                  }`} />
                  <span className="truncate">{file.path}</span>
                </div>
                <span className="text-[10px] text-slate-400 uppercase font-sans">
                  {file.type}
                </span>
              </button>
            ))}
          </div>

          <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-500">
            <p className="font-semibold text-slate-700">Project Directory Layout:</p>
            <ul className="list-disc pl-4 space-y-0.5 mt-1 text-slate-500">
              <li><code>data/raw/</code> & <code>data/processed/</code></li>
              <li><code>src/</code> core modules</li>
              <li><code>models/</code> & <code>outputs/</code></li>
              <li><code>notebooks/</code> Jupyter exploration</li>
            </ul>
          </div>
        </div>

        {/* Right Area: Code Viewer (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden flex flex-col">
          {/* Header Bar */}
          <div className="bg-slate-900 text-slate-200 px-4 py-3 flex items-center justify-between border-b border-slate-800">
            <div>
              <div className="font-mono text-xs font-bold text-emerald-400">
                {selectedFile.path}
              </div>
              <div className="text-[11px] text-slate-400">
                {selectedFile.description}
              </div>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 text-xs font-mono transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          {/* Code Body */}
          <pre className="p-4 bg-slate-950 text-slate-100 font-mono text-xs overflow-x-auto leading-relaxed max-h-[520px]">
            <code>{selectedFile.code}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
