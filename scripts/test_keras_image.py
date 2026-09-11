#!/usr/bin/env python3
"""
Inspectra AI — Single Image TensorFlow / Keras Inference Test Utility
====================================================================
Accepts an image path, executes the Keras inference pipeline, and prints
the defect classification, confidence, severity, and probabilities.

Usage:
  python scripts/test_keras_image.py <image_path> [--threshold 0.60]
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keras_model import run_keras_inference


def main():
    parser = argparse.ArgumentParser(description="Test TensorFlow/Keras inference on a single image")
    parser.add_argument("image_path", type=str, help="Path to image file (.png, .jpg, .bmp)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold (default: 0.60)")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[Error] Image not found: {img_path}")
        sys.exit(1)

    print("=" * 70)
    print("  INSPECTRA AI — TENSORFLOW / KERAS INFERENCE TEST")
    print("=" * 70)
    print(f"Target Image:        {img_path}")
    print(f"Threshold Applied:   {args.threshold}")
    print("-" * 70)

    result = run_keras_inference(img_path, confidence_threshold=args.threshold)

    print(f"Predicted Class:     {result.get('class')}")
    print(f"Raw Class:           {result.get('raw_class')}")
    print(f"Confidence Score:    {result.get('confidence', 0.0) * 100:.2f}%")
    print(f"Defect Severity:     {result.get('severity')}")
    print(f"Is Defective:        {result.get('is_defective')}")
    print(f"Uncertain Flag:      {result.get('uncertain')}")
    print(f"Processing Time:     {result.get('processing_time_ms', 0.0)} ms")
    print("\nClass Probabilities:")
    for cls_name, prob in result.get("probabilities", {}).items():
        bar = "█" * int(prob * 30)
        print(f"  • {cls_name:25s}: {prob*100:5.1f}%  {bar}")
    print("=" * 70)


if __name__ == "__main__":
    main()
