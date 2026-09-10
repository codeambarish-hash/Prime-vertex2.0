"""
Inspectra AI — Defect Severity Scoring Module
=============================================
Calculates standardized industrial defect severity scores (0 - 100) combining:
  1. Defect Type Inherent Risk Weight (structural flaws like cracks vs cosmetic stains)
  2. Defect Area Footprint (% of total product surface area)
  3. Deep Classifier Confidence & Anomaly Intensity

Categorization:
  - Minor (< 30): Aesthetic imperfections, superficial discoloration -> Action: "Log only"
  - Major (30 - 70): Observable flaws, deep scratches, dents -> Action: "Flag for review"
  - Critical (> 70): Structural cracks, severe dimensional failures -> Action: "Reject immediately"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

# =============================================================================
# Inherent Severity Weight Configuration by Defect Taxonomy Class
# =============================================================================
# Structural integrity and functional failures carry higher weight than surface cosmetic marks.
DEFECT_TYPE_WEIGHTS: Dict[str, float] = {
    "normal": 0.00,
    "discoloration": 0.25,             # Purely cosmetic color variance
    "stain": 0.35,                     # Superficial residue / washable spot
    "scratch": 0.55,                   # Surface abrasion without deep substrate penetration
    "dent": 0.72,                      # Mechanical deformation impacting form factor
    "dimensional_irregularity": 0.88,  # Tolerance breach impacting mating & assembly
    "crack": 0.95,                     # Catastrophic structural fracture risk
}

# Severity Category Boundaries (0 - 100 scale)
SEVERITY_THRESHOLDS = {
    "minor_max": 30.0,
    "critical_min": 70.0,
}

# Industrial Color-Coding for Visual Overlays (RGB & Hex)
SEVERITY_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "Minor": (34, 197, 94),     # Emerald Green
    "Major": (234, 179, 8),     # Industrial Amber / Yellow
    "Critical": (239, 68, 68),  # Signal Red
    "None": (148, 163, 184),    # Slate Neutral (Pristine pass)
}

SEVERITY_COLORS_HEX: Dict[str, str] = {
    "Minor": "#22c55e",
    "Major": "#eab308",
    "Critical": "#ef4444",
    "None": "#94a3b8",
}

# Standardized Recommended Industrial Quality Actions
SEVERITY_RECOMMENDED_ACTIONS: Dict[str, str] = {
    "Minor": "Log only",
    "Major": "Flag for review",
    "Critical": "Reject immediately",
    "None": "Pass",
}


@dataclass
class SeverityResult:
    """Structured severity evaluation container."""
    score: float
    category: str
    recommended_action: str
    area_percentage: float
    color_rgb: Tuple[int, int, int]
    color_hex: str
    type_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity_score": round(self.score, 2),
            "severity_category": self.category,
            "recommended_action": self.recommended_action,
            "area_percentage": round(self.area_percentage, 3),
            "severity_color_rgb": list(self.color_rgb),
            "severity_color_hex": self.color_hex,
            "type_weight": self.type_weight,
        }


class DefectSeverityScorer:
    """
    Evaluates multi-factor industrial defect severity combining:
      - Surface area percentage coverage
      - Defect type inherent risk profile
      - Model detection confidence
    """

    def __init__(
        self,
        type_weights: Optional[Dict[str, float]] = None,
        minor_threshold: float = 30.0,
        critical_threshold: float = 70.0,
        weight_type: float = 0.50,
        weight_area: float = 0.30,
        weight_conf: float = 0.20,
    ):
        self.type_weights = type_weights or DEFECT_TYPE_WEIGHTS
        self.minor_threshold = minor_threshold
        self.critical_threshold = critical_threshold
        
        # Normalize component weights to sum to 1.0
        total_w = weight_type + weight_area + weight_conf
        self.w_type = weight_type / total_w
        self.w_area = weight_area / total_w
        self.w_conf = weight_conf / total_w

    def evaluate(
        self,
        defect_type: str,
        confidence: float,
        defect_area_px: float,
        total_surface_px: float = 256.0 * 256.0,
        is_defective: bool = True,
        anomaly_score: Optional[float] = None
    ) -> SeverityResult:
        """
        Computes calibrated severity score (0 - 100) and maps to Minor/Major/Critical.

        Args:
            defect_type: Category name (e.g. 'crack', 'dent', 'stain', 'normal')
            confidence: Model classification confidence [0.0, 1.0]
            defect_area_px: Area of defect footprint in pixels
            total_surface_px: Total area of inspected specimen in pixels (default 256x256)
            is_defective: Boolean flag indicating if item has a confirmed defect
            anomaly_score: Optional unsupervised reconstruction anomaly score

        Returns:
            SeverityResult with score, category, recommended_action, and visual palette.
        """
        if not is_defective or defect_type.lower() in ("normal", "pristine", "none"):
            return SeverityResult(
                score=0.0,
                category="Minor",
                recommended_action=SEVERITY_RECOMMENDED_ACTIONS["Minor"],
                area_percentage=0.0,
                color_rgb=SEVERITY_COLORS_RGB["Minor"],
                color_hex=SEVERITY_COLORS_HEX["Minor"],
                type_weight=0.0
            )

        # 1. Defect Type Inherent Weight Component (0.0 - 1.0)
        norm_type = defect_type.lower().strip()
        type_weight = self.type_weights.get(norm_type, 0.50)

        # 2. Defect Area Footprint Component (0.0 - 1.0)
        # Industrial rule: A defect covering >= 5.0% of total product surface is massive.
        # We compute area percentage and scale with saturation at 6.0%.
        surface_area = max(1.0, float(total_surface_px))
        area_pct = (float(defect_area_px) / surface_area) * 100.0
        # Scaled area footprint: 0% -> 0.0, >= 5% -> 1.0
        scaled_area = min(1.0, area_pct / 5.0)

        # 3. Model Confidence Component (0.0 - 1.0)
        clamped_conf = min(1.0, max(0.0, float(confidence)))

        # 4. Multi-Factor Severity Computation (0 - 100)
        # Base linear combination
        raw_score = (
            (self.w_type * type_weight) +
            (self.w_area * scaled_area) +
            (self.w_conf * clamped_conf)
        ) * 100.0

        # Structural multiplier: Cracks and dimensional irregularities with high confidence
        # have elevated baseline severity to safeguard against mechanical breakdown.
        if norm_type in ("crack", "dimensional_irregularity") and clamped_conf >= 0.75:
            raw_score = max(raw_score, 72.0)  # Guarantees Critical categorization
        elif norm_type in ("discoloration", "stain") and scaled_area < 0.35:
            raw_score = min(raw_score, 29.5)  # Restricts micro-cosmetic spots to Minor

        # Bound to [0.0, 100.0]
        final_score = round(min(100.0, max(0.0, raw_score)), 2)

        # 5. Categorize based on configured thresholds
        if final_score < self.minor_threshold:
            category = "Minor"
        elif final_score <= self.critical_threshold:
            category = "Major"
        else:
            category = "Critical"

        recommended_action = SEVERITY_RECOMMENDED_ACTIONS[category]
        color_rgb = SEVERITY_COLORS_RGB[category]
        color_hex = SEVERITY_COLORS_HEX[category]

        return SeverityResult(
            score=final_score,
            category=category,
            recommended_action=recommended_action,
            area_percentage=round(area_pct, 4),
            color_rgb=color_rgb,
            color_hex=color_hex,
            type_weight=type_weight
        )


# Global singleton instance for quick pipeline access
DEFAULT_SEVERITY_SCORER = DefectSeverityScorer()


def compute_defect_severity(
    defect_type: str,
    confidence: float,
    defect_area_px: float,
    total_surface_px: float = 256.0 * 256.0,
    is_defective: bool = True,
    anomaly_score: Optional[float] = None
) -> SeverityResult:
    """
    Public utility function for computing severity score and category directly.
    """
    return DEFAULT_SEVERITY_SCORER.evaluate(
        defect_type=defect_type,
        confidence=confidence,
        defect_area_px=defect_area_px,
        total_surface_px=total_surface_px,
        is_defective=is_defective,
        anomaly_score=anomaly_score
    )
