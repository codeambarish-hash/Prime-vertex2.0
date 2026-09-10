"""
Synthetic Industrial Defect & Surface Generator using OpenCV and NumPy
======================================================================
Generates realistic manufacturing substrates (brushed metal, cast iron, machined parts,
ceramic tiles, carbon composite) and introduces 6 industrial defect categories:
  1. Crack: Random walk fractal line with displacement and shadow groove
  2. Scratch: High-aspect linear specular cut with light bounce
  3. Dent: Radial depression with asymmetric illumination shading
  4. Stain: Multi-octave fluid diffusion blob with opacity blending
  5. Discoloration: Oxidation / heat tint patch with color temperature shift
  6. Dimensional Irregularity: Edge notch or protrusion violating manufacturing tolerances

Each defective sample generates:
  - Synthetic RGB Image (with environmental lighting, orientation, & background noise)
  - Pixel-Level Ground Truth Binary Mask (for precise defect localization)
  - Defect Classification Label & Bounding Box [x, y, w, h]
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from src.config import DEFECT_CLASSES, CLASS_TO_IDX


class SyntheticSurfaceGenerator:
    """Generates realistic manufacturing surface substrates with industrial textures."""

    @staticmethod
    def generate_brushed_metal(width: int, height: int) -> np.ndarray:
        """Simulates brushed aluminum/steel with linear machining grains."""
        base_val = np.random.randint(160, 200)
        # Create directional linear noise
        noise = np.random.normal(0, 18, (height, width)).astype(np.float32)
        # Blur heavily along the horizontal brush axis to create grain
        kernel_w = np.random.randint(35, 65)
        kernel = np.ones((1, kernel_w), dtype=np.float32) / kernel_w
        brushed = cv2.filter2D(noise, -1, kernel)
        
        surface = np.clip(base_val + brushed, 0, 255).astype(np.uint8)
        # Return 3-channel image
        return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def generate_cast_iron(width: int, height: int) -> np.ndarray:
        """Simulates cast iron / sand-cast metal with granular porosity."""
        base_val = np.random.randint(90, 130)
        coarse_noise = np.random.normal(0, 25, (height // 4, width // 4)).astype(np.float32)
        coarse_noise = cv2.resize(coarse_noise, (width, height), interpolation=cv2.INTER_CUBIC)
        fine_noise = np.random.normal(0, 12, (height, width)).astype(np.float32)
        combined = base_val + coarse_noise * 0.7 + fine_noise * 0.3
        surface = np.clip(combined, 0, 255).astype(np.uint8)
        # Add subtle cool metallic tint
        bgr = cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)
        bgr[:, :, 0] = np.clip(bgr[:, :, 0] + 5, 0, 255) # slight blue tone
        return bgr

    @staticmethod
    def generate_machined_part(width: int, height: int) -> np.ndarray:
        """Simulates lathe/milling toolmarks with circular or raster tool paths."""
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        # Tool pass ripples
        wavelength = random.uniform(8.0, 16.0)
        ripples = np.sin(dist_from_center * (2 * np.pi / wavelength)) * 14.0
        grain = np.random.normal(0, 8, (height, width))
        
        base_val = np.random.randint(140, 180)
        surface = np.clip(base_val + ripples + grain, 0, 255).astype(np.uint8)
        return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def generate_ceramic_tile(width: int, height: int) -> np.ndarray:
        """Simulates smooth glazed or matte industrial ceramic substrate."""
        base_color = np.array([
            random.randint(190, 230),  # Blue
            random.randint(195, 235),  # Green
            random.randint(200, 240)   # Red
        ], dtype=np.float32)
        
        # Subtle non-uniformity
        grad_x = np.linspace(-8, 8, width)
        grad_y = np.linspace(-8, 8, height)
        gx, gy = np.meshgrid(grad_x, grad_y)
        noise = np.random.normal(0, 4, (height, width, 3))
        
        surface = np.zeros((height, width, 3), dtype=np.float32)
        for c in range(3):
            surface[:, :, c] = base_color[c] + gx + gy + noise[:, :, c]
            
        return np.clip(surface, 0, 255).astype(np.uint8)

    @classmethod
    def get_random_substrate(cls, width: int, height: int) -> np.ndarray:
        """Randomly samples a realistic manufacturing substrate."""
        substrates = [
            cls.generate_brushed_metal,
            cls.generate_cast_iron,
            cls.generate_machined_part,
            cls.generate_ceramic_tile
        ]
        chosen = random.choice(substrates)
        return chosen(width, height)


class SyntheticDefectInjector:
    """Injects physically grounded defect anomalies and generates exact ground truth masks."""

    @staticmethod
    def inject_crack(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects a jagged branching crack defect:
        Dark micro-fracture line with displaced segments and adjacent specular edge.
        """
        h, w, _ = image.shape
        result = image.copy().astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random start point avoiding outer borders
        curr_x = random.randint(int(w * 0.2), int(w * 0.8))
        curr_y = random.randint(int(h * 0.2), int(h * 0.8))
        
        num_steps = random.randint(45, 110)
        angle = random.uniform(0, 2 * math.pi)
        crack_points = [(curr_x, curr_y)]
        
        for _ in range(num_steps):
            # Jagged angle wander
            angle += random.uniform(-0.45, 0.45)
            step_len = random.uniform(2.0, 4.5)
            curr_x = int(curr_x + step_len * math.cos(angle))
            curr_y = int(curr_y + step_len * math.sin(angle))
            
            # Boundary check
            if 5 <= curr_x < w - 5 and 5 <= curr_y < h - 5:
                crack_points.append((curr_x, curr_y))
            else:
                break

        if len(crack_points) < 10:
            # Fallback if crack hit border too soon
            return SyntheticDefectInjector.inject_crack(image)

        # Draw crack into mask and apply shadow / highlight to image
        thickness = random.randint(2, 3)
        pts = np.array(crack_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

        # Create depth: dark core line and slight white specular reflection on right
        crack_mask_bool = mask > 0
        result[crack_mask_bool] = result[crack_mask_bool] * random.uniform(0.15, 0.35)
        
        # Dilate slightly for the specular ridge
        ridge_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        specular_area = (ridge_mask > 0) & (~crack_mask_bool)
        result[specular_area] = np.clip(result[specular_area] * 1.25 + 15, 0, 255)

        x, y, bw, bh = cv2.boundingRect(mask)
        info = {"defect_type": "crack", "bbox": [x, y, bw, bh], "pixel_count": int(np.sum(mask > 0))}
        return np.clip(result, 0, 255).astype(np.uint8), mask, info

    @staticmethod
    def inject_scratch(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects a linear or slightly curved mechanical scratch:
        Shallow cut with specular highlight on light-incident edge and shadow furrow.
        """
        h, w, _ = image.shape
        result = image.copy().astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        start_x = random.randint(int(w * 0.15), int(w * 0.85))
        start_y = random.randint(int(h * 0.15), int(h * 0.85))
        length = random.randint(int(w * 0.2), int(w * 0.55))
        angle = random.uniform(0, 2 * math.pi)

        # Quadratic bezier scratch with mild curvature
        mid_x = start_x + (length // 2) * math.cos(angle) + random.randint(-15, 15)
        mid_y = start_y + (length // 2) * math.sin(angle) + random.randint(-15, 15)
        end_x = int(start_x + length * math.cos(angle))
        end_y = int(start_y + length * math.sin(angle))

        t = np.linspace(0, 1, 60)
        pts = []
        for val in t:
            px = int((1 - val) ** 2 * start_x + 2 * (1 - val) * val * mid_x + val ** 2 * end_x)
            py = int((1 - val) ** 2 * start_y + 2 * (1 - val) * val * mid_y + val ** 2 * end_y)
            if 0 <= px < w and 0 <= py < h:
                pts.append((px, py))

        if len(pts) < 10:
            return SyntheticDefectInjector.inject_scratch(image)

        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 2)
        cv2.polylines(mask, [pts_arr], isClosed=False, color=255, thickness=thickness)

        # Specular bright furrow line
        mask_bool = mask > 0
        result[mask_bool] = np.clip(result[mask_bool] * 1.6 + 45, 0, 255)

        # Parallel shadow furrow
        shadow_pts = pts_arr + np.array([[[1, 1]]], dtype=np.int32)
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(shadow_mask, [shadow_pts], isClosed=False, color=255, thickness=1)
        shadow_bool = (shadow_mask > 0) & (~mask_bool)
        result[shadow_bool] = result[shadow_bool] * 0.4

        x, y, bw, bh = cv2.boundingRect(mask)
        info = {"defect_type": "scratch", "bbox": [x, y, bw, bh], "pixel_count": int(np.sum(mask > 0))}
        return np.clip(result, 0, 255).astype(np.uint8), mask, info

    @staticmethod
    def inject_dent(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects a physical depression / impact indentation:
        Radial depression with 3D gradient shading (shadow on one side, light bounce on other).
        """
        h, w, _ = image.shape
        result = image.copy().astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        radius = random.randint(14, 38)
        cx = random.randint(radius + 15, w - radius - 15)
        cy = random.randint(radius + 15, h - radius - 15)

        cv2.circle(mask, (cx, cy), radius, 255, -1)

        # Generate directional lighting indentation
        light_angle = random.uniform(0, 2 * math.pi)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        within_dent = dist <= radius

        # Surface normal gradient simulation
        dx = (x - cx) / (radius + 1e-5)
        dy = (y - cy) / (radius + 1e-5)
        directional_shading = dx * math.cos(light_angle) + dy * math.sin(light_angle)
        
        falloff = np.cos((dist / (radius + 1e-5)) * (math.pi / 2))
        dent_effect = directional_shading * falloff * 55.0

        for c in range(3):
            channel = result[:, :, c]
            channel[within_dent] = np.clip(channel[within_dent] + dent_effect[within_dent], 0, 255)
            result[:, :, c] = channel

        x, y, bw, bh = cv2.boundingRect(mask)
        info = {"defect_type": "dent", "bbox": [x, y, bw, bh], "pixel_count": int(np.sum(mask > 0))}
        return np.clip(result, 0, 255).astype(np.uint8), mask, info

    @staticmethod
    def inject_stain(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects chemical, oil, or coolant liquid stain:
        Irregular semi-transparent organic fluid blob with diffusion boundary.
        """
        h, w, _ = image.shape
        result = image.copy().astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        blob_radius = random.randint(20, 48)

        # Build irregular blob via multi-ellipses
        num_sub_blobs = random.randint(4, 7)
        for _ in range(num_sub_blobs):
            offset_x = cx + random.randint(-blob_radius // 2, blob_radius // 2)
            offset_y = cy + random.randint(-blob_radius // 2, blob_radius // 2)
            rx = random.randint(blob_radius // 2, blob_radius)
            ry = random.randint(blob_radius // 2, blob_radius)
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (offset_x, offset_y), (rx, ry), angle, 0, 360, 255, -1)

        # Smooth boundary to emulate fluid diffusion
        blurred_mask = cv2.GaussianBlur(mask, (25, 25), 0)
        mask_binary = (blurred_mask > 70).astype(np.uint8) * 255
        alpha = (blurred_mask.astype(np.float32) / 255.0) * random.uniform(0.45, 0.75)

        # Stain color: dark oil (yellow-brownish) or coolant (bluish-purple)
        stain_type = random.choice(["oil", "coolant"])
        if stain_type == "oil":
            stain_bgr = np.array([20, 50, 75], dtype=np.float32) # brown/amber
        else:
            stain_bgr = np.array([85, 30, 40], dtype=np.float32) # chemical dark stain

        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1.0 - alpha) + stain_bgr[c] * alpha

        x, y, bw, bh = cv2.boundingRect(mask_binary)
        info = {"defect_type": "stain", "bbox": [x, y, bw, bh], "pixel_count": int(np.sum(mask_binary > 0))}
        return np.clip(result, 0, 255).astype(np.uint8), mask_binary, info

    @staticmethod
    def inject_discoloration(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects thermal oxidation, chemical bleaching, or heat tint:
        Soft-gradient color shift in HSV color space without harsh edges.
        """
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        ax = random.randint(25, 60)
        ay = random.randint(20, 50)
        rot = random.randint(0, 180)

        cv2.ellipse(mask, (cx, cy), (ax, ay), rot, 0, 360, 255, -1)
        soft_mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_binary = (soft_mask > 60).astype(np.uint8) * 255

        # Transform to HSV to perform realistic thermal spectrum shift (blue/violet/straw yellow)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        norm_mask = soft_mask.astype(np.float32) / 255.0
        
        target_hue_shift = random.choice([15.0, 90.0, 130.0]) # heat discoloration hues
        hsv[:, :, 0] = (hsv[:, :, 0] + norm_mask * target_hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + norm_mask * 90.0, 0, 255) # increased saturation
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        x, y, bw, bh = cv2.boundingRect(mask_binary)
        info = {"defect_type": "discoloration", "bbox": [x, y, bw, bh], "pixel_count": int(np.sum(mask_binary > 0))}
        return result, mask_binary, info

    @staticmethod
    def inject_dimensional_irregularity(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Injects tolerance violation, edge notch cutout, or protrusion / burr:
        Geometric defect where material is missing or extra flash exists on borders.
        """
        h, w, _ = image.shape
        result = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Decide whether notch (missing material) or protrusion (burr)
        is_notch = random.choice([True, False])
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge in ["top", "bottom"]:
            pos_x = random.randint(int(w * 0.25), int(w * 0.75))
            width_defect = random.randint(25, 55)
            depth_defect = random.randint(15, 35)
            pos_y = 0 if edge == "top" else h - depth_defect
            pts = np.array([
                [pos_x - width_defect // 2, 0 if edge == "top" else h],
                [pos_x, depth_defect if edge == "top" else h - depth_defect],
                [pos_x + width_defect // 2, 0 if edge == "top" else h]
            ], np.int32)
        else:
            pos_y = random.randint(int(h * 0.25), int(h * 0.75))
            width_defect = random.randint(15, 35)
            depth_defect = random.randint(25, 55)
            pts = np.array([
                [0 if edge == "left" else w, pos_y - depth_defect // 2],
                [width_defect if edge == "left" else w - width_defect, pos_y],
                [0 if edge == "left" else w, pos_y + depth_defect // 2]
            ], np.int32)

        cv2.fillPoly(mask, [pts], 255)

        if is_notch:
            # Missing material: replace with dark fixture background
            bg_color = np.array([30, 30, 32], dtype=np.uint8)
            result[mask > 0] = bg_color
        else:
            # Material burr / protrusion: metal flash with rough edge
            flash_color = np.array([170, 175, 180], dtype=np.uint8)
            result[mask > 0] = flash_color
            noise = np.random.normal(0, 15, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        x, y, bw, bh = cv2.boundingRect(mask)
        info = {
            "defect_type": "dimensional_irregularity",
            "bbox": [x, y, bw, bh],
            "pixel_count": int(np.sum(mask > 0)),
            "subtype": "notch" if is_notch else "burr"
        }
        return result, mask, info


class EnvironmentalPerturbationSimulator:
    """
    Simulates factory floor variability:
    Varying lighting conditions (glare, shadows, underexposure),
    orientation rotations, and conveyor background shifts.
    """

    @staticmethod
    def apply_lighting_gradient(image: np.ndarray) -> np.ndarray:
        """Applies non-uniform factory illumination (e.g. angle work lamp, vignette)."""
        h, w, _ = image.shape
        illumination_types = ["linear_gradient", "spotlight", "vignette"]
        choice = random.choice(illumination_types)
        
        if choice == "linear_gradient":
            # Gradient across diagonal or horizontal
            angle = random.uniform(0, 2 * math.pi)
            x_vals = np.linspace(-1, 1, w)
            y_vals = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x_vals, y_vals)
            grad = (xx * math.cos(angle) + yy * math.sin(angle)) * random.uniform(0.18, 0.35)
            multiplier = 1.0 + grad
            multiplier = np.expand_dims(multiplier, axis=2)
            return np.clip(image.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)
            
        elif choice == "spotlight":
            # Central or offset inspection spotlight
            cx = random.randint(int(w * 0.3), int(w * 0.7))
            cy = random.randint(int(h * 0.3), int(h * 0.7))
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            sigma = random.uniform(w * 0.4, w * 0.8)
            spot = np.exp(-dist_sq / (2 * (sigma ** 2)))
            spot = 0.75 + 0.5 * spot # range 0.75 to 1.25
            spot = np.expand_dims(spot, axis=2)
            return np.clip(image.astype(np.float32) * spot, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def apply_random_rotation(
        image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Applies random in-plane orientation changes commonly found on conveyor parts."""
        angle = random.choice([0, 90, 180, 270, random.uniform(-15, 15)])
        if angle == 0:
            return image, mask

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(
            image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        rotated_mask = None
        if mask is not None:
            rotated_mask = cv2.warpAffine(
                mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        return rotated_img, rotated_mask


def generate_synthetic_sample(
    width: int = 256,
    height: int = 256,
    defect_type: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a complete annotated synthetic manufacturing sample.
    If defect_type is None, randomly selects between 'normal' and one of the 6 defect classes.
    """
    if defect_type is None:
        defect_type = random.choice(DEFECT_CLASSES)

    # 1. Base substrate
    base = SyntheticSurfaceGenerator.get_random_substrate(width, height)

    # 2. Inject defect (or keep normal)
    if defect_type == "normal":
        mask = np.zeros((height, width), dtype=np.uint8)
        info = {
            "defect_type": "normal",
            "is_defective": False,
            "bbox": [0, 0, 0, 0],
            "pixel_count": 0,
            "class_idx": CLASS_TO_IDX["normal"]
        }
        sample_img = base
    else:
        injector_map = {
            "crack": SyntheticDefectInjector.inject_crack,
            "scratch": SyntheticDefectInjector.inject_scratch,
            "dent": SyntheticDefectInjector.inject_dent,
            "stain": SyntheticDefectInjector.inject_stain,
            "discoloration": SyntheticDefectInjector.inject_discoloration,
            "dimensional_irregularity": SyntheticDefectInjector.inject_dimensional_irregularity,
        }
        injector = injector_map[defect_type]
        sample_img, mask, info = injector(base)
        info["is_defective"] = True
        info["class_idx"] = CLASS_TO_IDX[defect_type]

    # 3. Apply environmental illumination and orientation perturbations
    sample_img = EnvironmentalPerturbationSimulator.apply_lighting_gradient(sample_img)
    sample_img, mask = EnvironmentalPerturbationSimulator.apply_random_rotation(sample_img, mask)

    # Recalculate bounding box if rotated
    if info["is_defective"] and mask is not None and np.sum(mask) > 0:
        x, y, bw, bh = cv2.boundingRect(mask)
        info["bbox"] = [x, y, bw, bh]
        info["pixel_count"] = int(np.sum(mask > 0))

    return sample_img, mask if mask is not None else np.zeros((height, width), dtype=np.uint8), info
