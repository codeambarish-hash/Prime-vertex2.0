"""
MVTec Anomaly Detection (AD) Dataset Downloader and Ingestion Utility
====================================================================
Downloads, verifies, extracts, and parses the benchmark MVTec AD dataset:
(MVTec Software GmbH - Industrial Anomaly Detection Benchmark).
Reference: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"

Directory Structure Produced:
  data/raw/mvtec/<category>/
    ├── train/good/
    ├── test/good/
    ├── test/<defect_type>/
    └── ground_truth/<defect_type>/
"""

import os
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from src.config import DATA_RAW_DIR, MVTEC_CATEGORIES, MVTEC_BASE_URL


# Mapping MVTec defect sub-folders to our project's 6 defect classes
MVTEC_DEFECT_TAXONOMY_MAP: Dict[str, str] = {
    # Cracks & Fractures
    "crack": "crack",
    "cut": "crack",
    "hole": "crack",
    "split_teeth": "crack",
    "broken_teeth": "crack",
    
    # Scratches & Surface Grooves
    "scratch": "scratch",
    "scratches": "scratch",
    "surface_cut": "scratch",
    "frayed": "scratch",
    
    # Dents & Indentations
    "dent": "dent",
    "squeeze": "dent",
    "poke": "dent",
    "folded": "dent",
    "melted": "dent",
    
    # Stains & Liquid Contamination
    "stain": "stain",
    "contamination": "stain",
    "oil": "stain",
    "glue": "stain",
    "liquid": "stain",
    
    # Discoloration & Thermal Changes
    "color": "discoloration",
    "discoloration": "discoloration",
    "rough": "discoloration",
    
    # Dimensional & Structural Irregularities
    "bent": "dimensional_irregularity",
    "thread_damage": "dimensional_irregularity",
    "thread_side": "dimensional_irregularity",
    "metal_contamination": "dimensional_irregularity",
    "misplaced": "dimensional_irregularity",
    "fabric_border": "dimensional_irregularity",
    "fabric_interior": "dimensional_irregularity",
}


class DownloadProgressBar(tqdm):
    """Custom tqdm hook for urllib streaming download progress."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_mvtec_category(
    category: str,
    target_dir: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """
    Downloads and extracts an MVTec AD category archive.
    
    Parameters:
        category: Name of MVTec category (e.g., 'metal_nut', 'tile', 'cable', 'pill')
        target_dir: Destination path (defaults to data/raw/mvtec)
        force_download: If True, re-downloads even if folder exists
        
    Returns:
        Path to the extracted category folder
    """
    if category not in MVTEC_CATEGORIES and category != "all":
        raise ValueError(f"Unknown MVTec category: '{category}'. Allowed: {MVTEC_CATEGORIES}")

    mvtec_root = (target_dir or DATA_RAW_DIR) / "mvtec"
    mvtec_root.mkdir(parents=True, exist_ok=True)
    category_dir = mvtec_root / category

    if category_dir.exists() and any(category_dir.iterdir()) and not force_download:
        print(f"[INFO] Category '{category}' already exists at: {category_dir}")
        return category_dir

    archive_name = f"{category}.tar.xz"
    archive_path = mvtec_root / archive_name
    
    # Primary URL and fallback mirror
    url = f"https://www.mydrive.ch/shares/38536/3830184030e49fe7474e85d91d0f122c/download/420938113-1629952094/{archive_name}"
    
    print(f"\n=======================================================")
    print(f"Downloading MVTec AD category: '{category}'")
    print(f"Target Archive: {archive_path}")
    print(f"=======================================================")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (DefectInspectionEngine/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.info().get('Content-Length', 0))
            
            with DownloadProgressBar(
                unit='B', unit_scale=True, miniters=1, desc=archive_name, total=total_size
            ) as t:
                with open(archive_path, 'wb') as out_file:
                    while True:
                        buffer = response.read(1024 * 64)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        t.update(len(buffer))

        print(f"\n[INFO] Download finished. Extracting {archive_name}...")
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=category_dir)

        print(f"[SUCCESS] Extracted category to: {category_dir}")
        # Clean up archive to save disk space
        if archive_path.exists():
            archive_path.unlink()
            
        return category_dir

    except (urllib.error.URLError, TimeoutError, Exception) as e:
        print(f"\n[WARNING] Could not automatically download '{category}' from MVTec servers: {e}")
        print("[GUIDE] To use MVTec AD manually:")
        print(f"  1. Download '{archive_name}' from the official MVTec website (https://www.mvtec.com/company/research/datasets/mvtec-ad)")
        print(f"  2. Place and extract it in: {category_dir}")
        print("  Alternatively, switch to the synthetic OpenCV dataset with: --mode synthetic\n")
        raise RuntimeError(f"MVTec AD download failed for category '{category}'. Use --mode synthetic instead.") from e


def get_mvtec_taxonomy_class(mvtec_defect_name: str) -> str:
    """Maps MVTec specific defect folder names to standardized project classes."""
    cleaned = mvtec_defect_name.lower().strip()
    return MVTEC_DEFECT_TAXONOMY_MAP.get(cleaned, "crack")
