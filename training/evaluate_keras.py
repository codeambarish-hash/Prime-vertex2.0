#!/usr/bin/env python3
"""
Inspectra AI — Standalone TensorFlow / Keras Evaluation Suite
============================================================
Evaluates the trained Keras defect classifier model against holdout test specimens.
Computes:
  - Overall accuracy, weighted precision, recall, and F1-score
  - Per-class classification report across all 7 defect classes
  - Confusion matrix visualization
  - Critical manufacturing error metrics:
      * False Negatives (Defective product escaped detection as Normal)
      * False Positives (Normal product falsely rejected as Defective)

Usage:
  python training/evaluate_keras.py [--model_path models/keras/defect_classifier_combined.keras] [--data_dir data/unified_keras_dataset]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keras_config import (
    DEFECT_CLASSES,
    NUM_CLASSES,
    SAVED_MODEL_PATH,
    COMBINED_SAVED_MODEL_PATH,
    OUTPUTS_KERAS_DIR,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    COMBINED_CONFUSION_MATRIX_PATH,
    EVALUATION_METRICS_PATH,
    COMBINED_METRICS_PATH,
)
from training.train_keras import (
    collect_dataset_samples,
    render_confusion_matrix_pure,
)
from src.keras_model import run_keras_inference


def evaluate_model(
    model_path: Optional[Path] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs evaluation on holdout test specimens with detailed metrics."""
    if model_path is None:
        model_path = COMBINED_SAVED_MODEL_PATH if COMBINED_SAVED_MODEL_PATH.exists() else SAVED_MODEL_PATH

    if data_dir is None:
        if (PROJECT_ROOT / "data" / "unified_keras_dataset").exists():
            data_dir = PROJECT_ROOT / "data" / "unified_keras_dataset"
        elif (PROJECT_ROOT / "data" / "keras_dataset").exists():
            data_dir = PROJECT_ROOT / "data" / "keras_dataset"
        else:
            data_dir = PROJECT_ROOT / "data"

    print("=" * 76)
    print("   INSPECTRA AI — TENSORFLOW / KERAS INDEPENDENT EVALUATION")
    print("=" * 76)
    print(f"Target Model:   {model_path}")
    print(f"Data Directory: {data_dir}")
    print("-" * 76)

    if not model_path.exists():
        print(f"[Warning] Model file not found at {model_path}.")
        print("Please train the model first by running: python training/train_keras.py")

    print("[1/4] Collecting evaluation specimens...")
    samples = collect_dataset_samples(data_dir)
    if not samples:
        print(f"No samples found in {data_dir}.")
        return {"status": "error", "message": "No samples found"}

    print(f"Cataloged {len(samples)} specimens across 7 taxonomy classes.")

    print("[2/4] Executing model inference across evaluation specimens...")
    clean_classes = ["normal", "crack", "scratch", "dent", "stain", "discoloration", "dimensional_irregularity"]
    cm = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    y_true: List[int] = []
    y_pred: List[int] = []

    for s in samples:
        true_idx = s["label_idx"]
        y_true.append(true_idx)

        # Run inference
        res = run_keras_inference(s["path"])
        pred_cls_str = res.get("class", "normal").lower().replace(" ", "_")

        pred_idx = 0
        if pred_cls_str in clean_classes:
            pred_idx = clean_classes.index(pred_cls_str)
        else:
            # check in title case
            raw = res.get("raw_class", "Normal")
            if raw in DEFECT_CLASSES:
                pred_idx = DEFECT_CLASSES.index(raw)

        y_pred.append(pred_idx)
        cm[true_idx][pred_idx] += 1

    print("[3/4] Calculating precision, recall, F1, and critical escape metrics...")
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = round(correct / max(1, total), 4)

    # Per-class metrics
    per_class_report: Dict[str, Dict[str, Any]] = {}
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_supports = []

    for i, cls_name in enumerate(clean_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(NUM_CLASSES) if r != i)
        fn = sum(cm[i][c] for c in range(NUM_CLASSES) if c != i)
        support = sum(cm[i])

        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0

        per_class_report[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1-score": round(f1, 4),
            "support": support
        }
        if support > 0:
            class_precisions.append(prec * support)
            class_recalls.append(rec * support)
            class_f1s.append(f1 * support)
            class_supports.append(support)

    total_supp = sum(class_supports) if class_supports else 1
    weighted_prec = round(sum(class_precisions) / total_supp, 4)
    weighted_rec = round(sum(class_recalls) / total_supp, 4)
    weighted_f1 = round(sum(class_f1s) / total_supp, 4)

    normal_idx = clean_classes.index("normal")
    false_negatives = sum(cm[r][normal_idx] for r in range(NUM_CLASSES) if r != normal_idx)
    false_positives = sum(cm[normal_idx][c] for c in range(NUM_CLASSES) if c != normal_idx)

    eval_result = {
        "model_path": str(model_path),
        "dataset_evaluated": str(data_dir),
        "total_test_samples": total,
        "accuracy": acc,
        "precision": weighted_prec,
        "recall": weighted_rec,
        "f1_score": weighted_f1,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "per_class_report": per_class_report
    }

    print("\n" + "=" * 76)
    print("   EVALUATION SCORECARD")
    print("=" * 76)
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Weighted Precision:      {weighted_prec * 100:.2f}%")
    print(f"Weighted Recall:         {weighted_rec * 100:.2f}%")
    print(f"Weighted F1 Score:       {weighted_f1 * 100:.2f}%")
    print("-" * 76)
    print(f"False Negatives (Defective marked Normal - High Risk): {false_negatives}")
    print(f"False Positives (Normal marked Defective - Scrap Cost): {false_positives}")
    print("=" * 76)
    print(f"\nPer-Class Breakdown:")
    print(f"{'Class':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 68)
    for cls_name, met in per_class_report.items():
        print(f"{cls_name:<28} {met['precision']:<10.4f} {met['recall']:<10.4f} {met['f1-score']:<10.4f} {met['support']:<8d}")
    print("=" * 76)

    # Save to outputs/keras
    print("\n[4/4] Saving evaluation reports and visual matrix...")
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(eval_result, f, indent=2)
    with open(COMBINED_METRICS_PATH, "w") as f:
        json.dump(eval_result, f, indent=2)
    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        json.dump(per_class_report, f, indent=2)

    render_confusion_matrix_pure(cm, clean_classes, COMBINED_CONFUSION_MATRIX_PATH)
    render_confusion_matrix_pure(cm, clean_classes, CONFUSION_MATRIX_PATH)

    print(f"[✓] Evaluation artifacts saved to {OUTPUTS_KERAS_DIR}")
    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Keras defect detection model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    model_p = Path(args.model_path) if args.model_path else None
    data_p = Path(args.data_dir) if args.data_dir else None

    evaluate_model(
        model_path=model_p,
        data_dir=data_p
    )


if __name__ == "__main__":
    main()
