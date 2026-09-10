"""
Dataset Preparation and Standardization Pipeline
================================================
Orchestrates raw dataset acquisition and preparation:
  1. Synthetic Dataset generation via OpenCV (procedural surfaces, realistic defect injection, ground truth masks)
  2. MVTec AD dataset download, extraction, and standardization
  3. Stratified Train / Validation / Test dataset splitting (70 / 15 / 15)
  4. Pixel-level mask alignment and bounding box metadata serialization
  5. Statistical summary calculation and tabular reporting

Usage Examples:
  python src/prepare_dataset.py --mode synthetic --num-samples 350 --img-size 256
  python src/prepare_dataset.py --mode mvtec --mvtec-category metal_nut
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DEFECT_CLASSES,
    CLASS_TO_IDX,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.synthetic_generator import generate_synthetic_sample
from src.mvtec_downloader import (
    download_and_extract_mvtec_category,
    get_mvtec_taxonomy_class,
    MVTEC_CATEGORIES,
)


def setup_processed_directories(base_dir: Path) -> Dict[str, Path]:
    """Sets up standardized train/val/test directory tree for images and masks."""
    splits = ["train", "val", "test"]
    dirs = {}
    for split in splits:
        img_dir = base_dir / "images" / split
        mask_dir = base_dir / "masks" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        dirs[f"img_{split}"] = img_dir
        dirs[f"mask_{split}"] = mask_dir
    return dirs


def prepare_synthetic_dataset(
    output_dir: Path,
    num_samples_per_class: int = 50,
    img_size: Tuple[int, int] = (256, 256),
    seed: int = DEFAULT_SEED
) -> List[Dict[str, Any]]:
    """
    Generates balanced synthetic dataset across all 7 defect classes (normal + 6 defect types).
    Returns list of metadata records.
    """
    random.seed(seed)
    np.random.seed(seed)
    w, h = img_size
    dir_map = setup_processed_directories(output_dir)
    metadata_records: List[Dict[str, Any]] = []

    total_samples = num_samples_per_class * len(DEFECT_CLASSES)
    print(f"\n[SYNTHETIC GENERATOR] Generating {total_samples} samples across {len(DEFECT_CLASSES)} classes...")
    print(f"Classes: {', '.join(DEFECT_CLASSES)}")
    print(f"Target Resolution: {w}x{h} px\n")

    sample_counter = 0

    for defect_type in DEFECT_CLASSES:
        print(f" -> Generating '{defect_type}' ({num_samples_per_class} samples)...")
        for i in range(num_samples_per_class):
            sample_counter += 1
            # Stratified split allocation
            rand_val = random.random()
            if rand_val < TRAIN_RATIO:
                split = "train"
            elif rand_val < TRAIN_RATIO + VAL_RATIO:
                split = "val"
            else:
                split = "test"

            # Generate sample image, mask, and defect annotation
            image, mask, info = generate_synthetic_sample(width=w, height=h, defect_type=defect_type)

            sample_id = f"syn_{defect_type}_{i:04d}"
            img_filename = f"{sample_id}.png"
            mask_filename = f"{sample_id}_mask.png"

            img_path = dir_map[f"img_{split}"] / img_filename
            mask_path = dir_map[f"mask_{split}"] / mask_filename

            # Save image and ground truth mask
            cv2.imwrite(str(img_path), image)
            cv2.imwrite(str(mask_path), mask)

            record = {
                "sample_id": sample_id,
                "split": split,
                "class_name": defect_type,
                "class_idx": CLASS_TO_IDX[defect_type],
                "is_defective": info["is_defective"],
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "width": w,
                "height": h,
                "channels": 3,
                "bbox": info.get("bbox", [0, 0, 0, 0]),
                "defect_pixel_count": int(np.sum(mask > 0)),
                "defect_coverage_pct": round(float(np.sum(mask > 0) / (w * h) * 100), 2)
            }
            metadata_records.append(record)

    return metadata_records


def prepare_mvtec_dataset(
    category: str,
    output_dir: Path,
    img_size: Tuple[int, int] = (256, 256)
) -> List[Dict[str, Any]]:
    """
    Downloads/loads MVTec category, standardizes to train/val/test format with aligned masks.
    """
    w, h = img_size
    dir_map = setup_processed_directories(output_dir)
    metadata_records: List[Dict[str, Any]] = []

    # Attempt download or access raw directory
    category_dir = download_and_extract_mvtec_category(category)

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Missing MVTec 'train/good' folder in: {category_dir}")

    print(f"\n[MVTEC PREPARATION] Processing category: '{category}'...")

    # 1. Process Normal training images
    good_files = sorted(list(train_good_dir.glob("*.png")))
    random.shuffle(good_files)
    
    # Split normal images into train (80%) and val (20%)
    num_train = int(len(good_files) * 0.8)
    
    for idx, fpath in enumerate(good_files):
        split = "train" if idx < num_train else "val"
        sample_id = f"mvtec_{category}_good_{idx:04d}"
        
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img_resized = cv2.resize(img, (w, h))
        mask_blank = np.zeros((h, w), dtype=np.uint8)

        img_out = dir_map[f"img_{split}"] / f"{sample_id}.png"
        mask_out = dir_map[f"mask_{split}"] / f"{sample_id}_mask.png"

        cv2.imwrite(str(img_out), img_resized)
        cv2.imwrite(str(mask_out), mask_blank)

        metadata_records.append({
            "sample_id": sample_id,
            "split": split,
            "class_name": "normal",
            "class_idx": CLASS_TO_IDX["normal"],
            "is_defective": False,
            "image_path": str(img_out.relative_to(PROJECT_ROOT)),
            "mask_path": str(mask_out.relative_to(PROJECT_ROOT)),
            "width": w,
            "height": h,
            "channels": 3,
            "bbox": [0, 0, 0, 0],
            "defect_pixel_count": 0,
            "defect_coverage_pct": 0.0
        })

    # 2. Process Defective test samples + normal test samples
    for defect_folder in sorted(test_dir.iterdir()):
        if not defect_folder.is_dir():
            continue
        
        raw_defect_name = defect_folder.name
        is_good = (raw_defect_name == "good")
        project_class = "normal" if is_good else get_mvtec_taxonomy_class(raw_defect_name)
        
        test_images = sorted(list(defect_folder.glob("*.png")))
        for idx, fpath in enumerate(test_images):
            sample_id = f"mvtec_{category}_{raw_defect_name}_{idx:04d}"
            split = "test"

            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img_resized = cv2.resize(img, (w, h))

            # Locate ground truth mask
            mask_resized = np.zeros((h, w), dtype=np.uint8)
            if not is_good and gt_dir.exists():
                gt_defect_dir = gt_dir / raw_defect_name
                gt_mask_path = gt_defect_dir / f"{fpath.stem}_mask.png"
                if gt_mask_path.exists():
                    mask_raw = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_raw is not None:
                        mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_resized = (mask_resized > 127).astype(np.uint8) * 255

            img_out = dir_map[f"img_{split}"] / f"{sample_id}.png"
            mask_out = dir_map[f"mask_{split}"] / f"{sample_id}_mask.png"

            cv2.imwrite(str(img_out), img_resized)
            cv2.imwrite(str(mask_out), mask_resized)

            defect_pixels = int(np.sum(mask_resized > 0))
            bbox = [0, 0, 0, 0]
            if defect_pixels > 0:
                bx, by, bw, bh = cv2.boundingRect(mask_resized)
                bbox = [bx, by, bw, bh]

            metadata_records.append({
                "sample_id": sample_id,
                "split": split,
                "class_name": project_class,
                "class_idx": CLASS_TO_IDX[project_class],
                "is_defective": not is_good,
                "image_path": str(img_out.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_out.relative_to(PROJECT_ROOT)),
                "width": w,
                "height": h,
                "channels": 3,
                "bbox": bbox,
                "defect_pixel_count": defect_pixels,
                "defect_coverage_pct": round(float(defect_pixels / (w * h) * 100), 2)
            })

    return metadata_records


def compute_and_print_dataset_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics and outputs an industrial tabular overview to the console.
    """
    total_images = len(records)
    if total_images == 0:
        print("[WARNING] No dataset records found to analyze.")
        return {}

    # Split breakdown
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    # Class breakdown
    class_counts: Dict[str, int] = {cls: 0 for cls in DEFECT_CLASSES}
    defective_count = 0
    normal_count = 0

    defect_coverages: List[float] = []
    pixel_counts: List[int] = []

    widths = set()
    heights = set()
    channels = set()

    for r in records:
        cname = r["class_name"]
        class_counts[cname] = class_counts.get(cname, 0) + 1
        if r["is_defective"]:
            defective_count += 1
            defect_coverages.append(r["defect_coverage_pct"])
            pixel_counts.append(r["defect_pixel_count"])
        else:
            normal_count += 1

        widths.add(r["width"])
        heights.add(r["height"])
        channels.add(r["channels"])

    mean_coverage = float(np.mean(defect_coverages)) if defect_coverages else 0.0
    max_coverage = float(np.max(defect_coverages)) if defect_coverages else 0.0
    min_coverage = float(np.min(defect_coverages)) if defect_coverages else 0.0

    stats = {
        "total_images": total_images,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "normal_vs_defective": {
            "normal": normal_count,
            "normal_pct": round((normal_count / total_images) * 100, 1),
            "defective": defective_count,
            "defective_pct": round((defective_count / total_images) * 100, 1),
        },
        "image_dimensions": {
            "widths": list(widths),
            "heights": list(heights),
            "channels": list(channels),
            "resolution": f"{list(widths)[0]}x{list(heights)[0]}x{list(channels)[0]}"
        },
        "defect_localization_metrics": {
            "mean_coverage_pct": round(mean_coverage, 2),
            "min_coverage_pct": round(min_coverage, 2),
            "max_coverage_pct": round(max_coverage, 2),
            "total_masks_with_defects": len(defect_coverages)
        }
    }

    # -------------------------------------------------------------------------
    # Pretty Tabular Console Print
    # -------------------------------------------------------------------------
    line = "=" * 68
    sub_line = "-" * 68
    print(f"\n{line}")
    print(" VISION-BASED DEFECT DETECTION — DATASET SUMMARY STATISTICS")
    print(f"{line}")
    print(f" Total Dataset Images:  {total_images}")
    print(f" Image Resolution:      {stats['image_dimensions']['resolution']}")
    print(f" Normal Samples:        {normal_count} ({stats['normal_vs_defective']['normal_pct']}%)")
    print(f" Defective Samples:     {defective_count} ({stats['normal_vs_defective']['defective_pct']}%)")
    print(sub_line)
    print(f" {'SPLIT':<12} | {'COUNT':<10} | {'RATIO':<10}")
    print(sub_line)
    for sp, cnt in split_counts.items():
        pct = round((cnt / total_images) * 100, 1)
        print(f" {sp.upper():<12} | {cnt:<10} | {pct:>5.1f} %")

    print(sub_line)
    print(f" {'DEFECT CLASS':<28} | {'COUNT':<10} | {'PERCENTAGE':<10}")
    print(sub_line)
    for cname in DEFECT_CLASSES:
        cnt = class_counts.get(cname, 0)
        pct = round((cnt / total_images) * 100, 1)
        tag = "[NORMAL]" if cname == "normal" else "[ANOMALY]"
        print(f" {cname:<24} {tag:<4} | {cnt:<10} | {pct:>5.1f} %")

    print(sub_line)
    print(" DEFECT LOCALIZATION (GROUND TRUTH MASKS):")
    print(f"  • Mean Defect Coverage:   {mean_coverage:.2f} % of total image area")
    print(f"  • Min Defect Coverage:    {min_coverage:.2f} %")
    print(f"  • Max Defect Coverage:    {max_coverage:.2f} %")
    print(f"  • Masks with Annotations: {len(defect_coverages)}")
    print(f"{line}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Based Defect Detection: Dataset Preparation & Ingestion Engine"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "mvtec"],
        default="synthetic",
        help="Dataset source mode: 'synthetic' (OpenCV procedural engine) or 'mvtec' (MVTec AD benchmark)"
    )
    parser.add_argument(
        "--mvtec-category",
        type=str,
        default="metal_nut",
        choices=MVTEC_CATEGORIES,
        help="MVTec category to acquire when mode is 'mvtec'"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of synthetic samples per defect class (total = 7 * num_samples)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Standardized image dimension (height = width = img_size)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED_DIR),
        help="Target directory for processed dataset"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    res = (args.img_size, args.img_size)

    records: List[Dict[str, Any]] = []

    if args.mode == "mvtec":
        try:
            records = prepare_mvtec_dataset(category=args.mvtec_category, output_dir=out_dir, img_size=res)
        except Exception as e:
            print(f"\n[ALERT] MVTec AD ingestion encountered an error: {e}")
            print("[FALLBACK] Automatically switching to OpenCV synthetic generation to guarantee runnable dataset...\n")
            records = prepare_synthetic_dataset(output_dir=out_dir, num_samples_per_class=args.num_samples, img_size=res)
    else:
        records = prepare_synthetic_dataset(output_dir=out_dir, num_samples_per_class=args.num_samples, img_size=res)

    # Compute and print comprehensive summary statistics
    stats = compute_and_print_dataset_stats(records)

    # Save annotations and stats JSON in data/processed
    annotations_file = out_dir / "annotations.json"
    stats_file = out_dir / "dataset_stats.json"

    with open(annotations_file, "w") as f:
        json.dump(records, f, indent=2)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[SAVED] Annotations catalog -> {annotations_file}")
    print(f"[SAVED] Dataset stats record -> {stats_file}\n")


if __name__ == "__main__":
    main()
