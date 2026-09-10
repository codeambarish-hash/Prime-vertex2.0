"""
Multi-Task Deep Learning Architecture for Industrial Defect Inspection
======================================================================
Joint model fulfilling all 3 manufacturing inspection requirements simultaneously:
  1. Anomaly Detection: Binary classification (Normal vs. Defective)
  2. Defect Categorization: Multi-class classification across 7 classes
  3. Defect Localization: Pixel-level segmentation decoder producing anomaly heatmaps and bounding boxes
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES, IDX_TO_CLASS


class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm + LeakyReLU residual building block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DefectInspectionNet(nn.Module):
    """
    Unified Multi-Task Inspection Network:
    Encoder-Decoder structure with dual classification heads and pixel segmentation head.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # ---------------------------------------------------------------------
        # Shared Feature Encoder (Multi-Scale Feature Hierarchy)
        # ---------------------------------------------------------------------
        self.enc1 = ConvBlock(in_channels, 32)      # -> [B, 32, H, W]
        self.pool1 = nn.MaxPool2d(2, 2)            # -> [B, 32, H/2, W/2]

        self.enc2 = ConvBlock(32, 64)              # -> [B, 64, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2, 2)            # -> [B, 64, H/4, W/4]

        self.enc3 = ConvBlock(64, 128)             # -> [B, 128, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2, 2)            # -> [B, 128, H/8, W/8]

        self.enc4 = ConvBlock(128, 256)            # -> [B, 256, H/8, W/8]
        self.pool4 = nn.MaxPool2d(2, 2)            # -> [B, 256, H/16, W/16]

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)      # -> [B, 512, H/16, W/16]

        # ---------------------------------------------------------------------
        # Head 1 & 2: Global Classification & Anomaly Detection
        # ---------------------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)            # 7 classes (normal + 6 defects)
        )
        self.binary_detector_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)                       # 1 output: logit for defective vs normal
        )

        # ---------------------------------------------------------------------
        # Head 3: Defect Localization Decoder (U-Net style with skip connections)
        # ---------------------------------------------------------------------
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Final 1x1 conv to produce pixel localization anomaly probability
        self.final_mask_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder forward pass
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Global features for classification
        feat = self.gap(b).flatten(1)
        class_logits = self.classifier_fc(feat)
        binary_logit = self.binary_detector_fc(feat).squeeze(1)

        # Decoder forward pass with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask_logits = self.final_mask_conv(d1)

        return {
            "class_logits": class_logits,                  # Shape: [B, num_classes]
            "is_defective_logit": binary_logit,            # Shape: [B]
            "mask_logits": mask_logits,                    # Shape: [B, 1, H, W]
            "mask_prob": torch.sigmoid(mask_logits),       # Pixel anomaly heatmap [0, 1]
            "is_defective_prob": torch.sigmoid(binary_logit),
            "class_probs": F.softmax(class_logits, dim=-1)
        }


class MultiTaskDefectLoss(nn.Module):
    """
    Balanced multi-objective loss combining:
      1. Binary Cross-Entropy for anomaly detection
      2. Categorical Cross-Entropy for defect classification
      3. Combination of Dice Loss + BCE for pixel localization mask
    """
    def __init__(self, weight_binary: float = 1.0, weight_class: float = 1.0, weight_mask: float = 2.0):
        super().__init__()
        self.w_bin = weight_binary
        self.w_cls = weight_class
        self.w_mask = weight_mask
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred_probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Computes continuous Dice loss for segmentation masks."""
        pred_flat = pred_probs.view(-1)
        target_flat = targets.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1.0 - dice

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 1. Binary anomaly loss
        loss_bin = self.bce_loss(predictions["is_defective_logit"], targets["is_defective"])

        # 2. Multi-class defect classification loss
        loss_cls = self.ce_loss(predictions["class_logits"], targets["class_label"])

        # 3. Mask localization loss (BCE + Dice)
        mask_pred_prob = predictions["mask_prob"]
        loss_mask_bce = F.binary_cross_entropy(mask_pred_prob, targets["mask"])
        loss_mask_dice = self.dice_loss(mask_pred_prob, targets["mask"])
        loss_mask = loss_mask_bce + loss_mask_dice

        # Total combined loss
        total_loss = self.w_bin * loss_bin + self.w_cls * loss_cls + self.w_mask * loss_mask

        return {
            "total_loss": total_loss,
            "loss_binary": loss_bin,
            "loss_class": loss_cls,
            "loss_mask": loss_mask
        }


def extract_bounding_boxes_from_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 15
) -> List[Tuple[int, int, int, int]]:
    """
    Extracts bounding boxes [x, y, w, h] from continuous probability heatmap.
    Uses connected components and contour thresholding.
    """
    import cv2
    binary_map = (mask >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((int(x), int(y), int(w), int(h)))

    return boxes
