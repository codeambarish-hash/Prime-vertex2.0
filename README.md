# Vision-Based Defect Detection for Manufacturing Quality Inspection

A deep learning and computer vision pipeline for automated manufacturing quality assurance. The system addresses three primary manufacturing requirements:
1. **Anomaly Detection**: Fast binary screening (*Defective* vs. *Normal*).
2. **Defect Categorization**: Multi-class classification across 6 industrial defect classes (*crack, scratch, dent, stain, discoloration, dimensional irregularity*) plus *normal*.
3. **Defect Localization**: Pixel-level segmentation mask generation with anomaly heatmaps and bounding box coordinate extraction `[x, y, w, h]`.
4. **Environmental Robustness**: Engineered to operate reliably under variable factory illumination (glare, shadows, non-uniform gradients), part orientations, and conveyor backgrounds.

---

## 1. Project Directory Structure

```text
vision-based-defect-detection/
├── data/
│   ├── raw/                 # Unprocessed datasets (e.g. MVTec AD downloads)
│   └── processed/           # Standardized images & pixel masks (train/val/test splits)
│       ├── images/          # RGB product images (train/, val/, test/)
│       ├── masks/           # Pixel-level ground truth masks (train/, val/, test/)
│       ├── annotations.json # Structured JSON catalog with bbox & metadata
│       └── dataset_stats.json # Automated dataset statistics report
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Central configuration (paths, classes, parameters)
│   ├── synthetic_generator.py # OpenCV procedural engine for surfaces & defects
│   ├── mvtec_downloader.py  # MVTec AD benchmark downloader & parser
│   ├── prepare_dataset.py   # Dataset preparation, splitting, & stats generation
│   ├── dataset.py           # PyTorch Dataset with Albumentations augmentations
│   ├── model.py             # Multi-Task Network (Detection + Classification + UNet Localization)
│   └── inspect.py           # Inference visualizer & 4-panel report generator
├── models/                  # Checkpoints, saved weights, and export artifacts
├── outputs/                 # Inspection reports, confusion matrices, and heatmaps
├── notebooks/
│   └── 01_dataset_exploration.ipynb # Interactive exploration and visualization
├── requirements.txt         # Pinned python dependencies
├── main.py                  # Primary CLI pipeline entrypoint
└── README.md
```

---

## 2. Requirements & Installation

Install dependencies using Python 3.9+ with pip:

```bash
pip install -r requirements.txt
```

Core libraries include:
- **Computer Vision**: `opencv-python>=4.8.0`, `scikit-image>=0.21.0`, `pillow>=10.0.0`
- **Deep Learning**: `torch>=2.0.0`, `torchvision>=0.15.0`
- **Data & Statistics**: `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `scipy>=1.11.0`
- **Augmentation & Robustness**: `albumentations>=1.3.1`
- **Visualization**: `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `tqdm>=4.66.0`

---

## 3. Dataset Preparation & Statistical Summary

You can choose between procedural OpenCV synthetic generation or the official MVTec AD benchmark.

### Option A: OpenCV Synthetic Generation (Default & Offline-Ready)
Generates realistic manufacturing substrates (*brushed metal, cast iron, machined parts, ceramic tiles*) with physically simulated defects, ground truth masks, and environmental perturbations (lighting gradients, spotlights, rotations):

```bash
python main.py --mode synthetic --num-samples 50 --img-size 256
```
*(Generates 50 samples per class = 350 samples total across the 7 classes).*

### Option B: MVTec AD Industrial Benchmark
Downloads and standardizes an MVTec category (e.g. `metal_nut`, `tile`, `bottle`, `capsule`):

```bash
python main.py --mode mvtec --mvtec-category metal_nut --img-size 256
```
*Note: If network access or mirror issues occur, the script automatically falls back to the OpenCV synthetic generator to guarantee a runnable pipeline.*

### Tabular Statistics Output
Once loaded or generated, the script outputs statistical telemetry:
- Total image count and resolution
- Train / Validation / Test stratified split distribution (70% / 15% / 15%)
- Per-class counts and percentage distribution
- Normal vs. Defective balance
- Ground truth mask coverage metrics (min, mean, max defect surface percentage)

---

## 4. Multi-Task Deep Learning Architecture

The model in `src/model.py` (`DefectInspectionNet`) uses a unified multi-task architecture:
- **Shared Convolutional Encoder**: Multi-scale feature extraction with residual blocks.
- **Binary Anomaly Head**: Rapid Pass/Fail gating via global average pooling and sigmoid activation.
- **Multi-Class Classifier Head**: Categorizes defect type into *crack, scratch, dent, stain, discoloration, dimensional irregularity*.
- **Localization Decoder Head**: U-Net skip-connected decoder producing a continuous anomaly probability heatmap `[B, 1, H, W]`.
- **Bounding Box Extraction**: Automatically thresholds the predicted heatmap and computes OpenCV connected components to yield `[x, y, w, h]` bounding boxes with area metrics.
