"""
Industrial Inspection Engine: Inference, Localization & Report Generation
========================================================================
Runs manufacturing inference on raw product images:
  1. Detects Defect vs. Normal state
  2. Classifies defect category (crack, scratch, dent, stain, discoloration, dimensional_irregularity)
  3. Computes pixel localization mask and extracts precise bounding boxes
  4. Generates an inspection report card image saved to outputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from src.config import (
    DEFECT_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEFECT_COLORS,
    OUTPUTS_DIR,
    DEFAULT_IMAGE_SIZE,
)


def overlay_defect_localization(
    image: np.ndarray,
    mask: np.ndarray,
    predicted_class: str,
    confidence: float,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays colored pixel heatmap and bounding box annotation on inspected product image.
    """
    h, w = image.shape[:2]
    annotated = image.copy()
    color_rgb = DEFECT_COLORS.get(predicted_class, (239, 68, 68))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # OpenCV uses BGR

    if predicted_class != "normal" and mask is not None and np.sum(mask) > 0:
        # 1. Color overlay over defect region
        colored_mask = np.zeros_like(image)
        mask_bool = mask > 127
        colored_mask[mask_bool] = color_bgr
        annotated = cv2.addWeighted(annotated, 1.0, colored_mask, alpha, 0)

        # 2. Draw contour boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color_bgr, 2)

        # 3. Draw localized bounding box
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            bx, by, bw, bh = bbox
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color_bgr, 2)
            
            # Corner accents for high-precision industrial CAD look
            line_len = min(12, bw // 4, bh // 4)
            cv2.line(annotated, (bx, by), (bx + line_len, by), color_bgr, 3)
            cv2.line(annotated, (bx, by), (bx, by + line_len), color_bgr, 3)
            cv2.line(annotated, (bx + bw, by), (bx + bw - line_len, by), color_bgr, 3)
            cv2.line(annotated, (bx + bw, by), (bx + bw, by + line_len), color_bgr, 3)

    # 4. Add industrial HUD telemetry header
    hud_bg = np.zeros((48, w, 3), dtype=np.uint8)
    status_text = "STATUS: DEFECT DETECTED" if predicted_class != "normal" else "STATUS: PASS (PRISTINE)"
    status_color = (0, 0, 255) if predicted_class != "normal" else (0, 200, 0)
    
    cv2.putText(annotated, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    meta_text = f"TYPE: {predicted_class.upper()} | CONF: {confidence * 100:.1f}%"
    cv2.putText(annotated, meta_text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    return annotated


def create_inspection_quad_report(
    raw_image: np.ndarray,
    mask: np.ndarray,
    predicted_class: str,
    confidence: float,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Creates a 4-panel industrial inspection summary visualizer:
      [Top-Left] Raw Input Image
      [Top-Right] Anomaly Heatmap (Inferno / Jet Colormap)
      [Bottom-Left] Binary Threshold Mask
      [Bottom-Right] Bounding Box & HUD Overlay
    """
    h, w = raw_image.shape[:2]

    # Panel 1: Raw
    p1 = raw_image.copy()
    cv2.putText(p1, "1. RAW SENSOR CAPTURE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Panel 2: Heatmap
    norm_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p2 = cv2.applyColorMap(norm_mask, cv2.COLORMAP_INFERNO)
    cv2.putText(p2, "2. ANOMALY DENSITY HEATMAP", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Panel 3: Binary Segmentation
    p3 = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(p3, "3. LOCALIZATION MASK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Panel 4: Composite Inspection Overlay
    p4 = overlay_defect_localization(raw_image, mask, predicted_class, confidence, bbox)
    cv2.putText(p4, "4. LOCALIZED DEFECT + HUD", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Assemble 2x2 grid
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    report_quad = np.vstack([top_row, bottom_row])

    return report_quad
