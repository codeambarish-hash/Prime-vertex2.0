"""
PyTorch Dataset and Albumentations Augmentation Pipeline
========================================================
Implements PyTorch Dataset abstraction for manufacturing inspection with
industrial-grade data augmentations designed to handle:
  - Lighting variations (intensity changes, shadows, CLAHE, gamma)
  - Orientation shifts (arbitrary rotation, flip, affine shear)
  - Sensor noise and blur (Gaussian noise, motion blur, sensor defocus)
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional Albumentations for production computer vision pipelines
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from src.config import PROJECT_ROOT, DEFAULT_IMAGE_SIZE, NUM_CLASSES


def get_industrial_transforms(
    split: str = "train",
    img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Any:
    """
    Constructs robust Albumentations pipeline tailored for manufacturing inspection.
    Includes illumination simulation, perspective distortion, and optical surface noise.
    """
    w, h = img_size
    if not HAS_ALBUMENTATIONS:
        return None

    if split == "train":
        return A.Compose([
            A.Resize(h, w),
            # Geometric & Orientation Robustness
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.10, rotate_limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),
            
            # Illumination & Lighting Robustness
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.8),
            
            # Optical Surface & Sensor Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Deterministic evaluation transforms
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class DefectDataset(Dataset):
    """
    PyTorch Dataset yielding:
      - Transformed Image Tensor [3, H, W]
      - Binary Ground Truth Segmentation Mask [1, H, W] (for localization)
      - Binary Defect Label (0=normal, 1=defective)
      - Multi-Class Categorical Label (0 to 6)
      - Bounding Box [x, y, w, h] and metadata record
    """
    def __init__(
        self,
        annotations: List[Dict[str, Any]],
        split: str = "train",
        img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        transform: Optional[Any] = None
    ):
        self.split = split
        self.img_size = img_size
        self.records = [r for r in annotations if r.get("split") == split]
        self.transform = transform or get_industrial_transforms(split=split, img_size=img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = PROJECT_ROOT / rec["image_path"]
        mask_path = PROJECT_ROOT / rec["mask_path"]

        # Load RGB image and grayscale ground truth mask
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(self.img_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        # Ensure normalized binary mask (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Apply augmentations
        if self.transform is not None and HAS_ALBUMENTATIONS:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0) # [1, H, W]
        else:
            # Fallback pure PyTorch / NumPy conversion
            image_resized = cv2.resize(image, self.img_size)
            mask_resized = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            image_norm = image_resized.astype(np.float32) / 255.0
            image_norm = (image_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.tensor(image_norm, dtype=torch.float32).permute(2, 0, 1)
            mask_tensor = torch.tensor(mask_resized, dtype=torch.float32).unsqueeze(0)

        is_defective = 1 if rec["is_defective"] else 0
        class_idx = rec["class_idx"]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "is_defective": torch.tensor(is_defective, dtype=torch.float32),
            "class_label": torch.tensor(class_idx, dtype=torch.long),
            "sample_id": rec["sample_id"],
            "bbox": torch.tensor(rec.get("bbox", [0, 0, 0, 0]), dtype=torch.float32),
            "class_name": rec["class_name"]
        }


def create_dataloaders(
    annotations_path: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory helper to build train, val, and test DataLoaders."""
    with open(annotations_path, "r") as f:
        records = json.load(f)

    train_ds = DefectDataset(records, split="train", img_size=img_size)
    val_ds = DefectDataset(records, split="val", img_size=img_size)
    test_ds = DefectDataset(records, split="test", img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
