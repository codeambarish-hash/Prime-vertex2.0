#!/usr/bin/env python3
"""
High-Resolution Synthetic Industrial Surface Defect Generator
============================================================
Generates 1024x1024 high-fidelity manufacturing surface specimens with:
  1. Realistic metallic substrates (brushed aluminum, cold-rolled steel, cast iron, machined lathe)
  2. Physically grounded industrial defect anomalies:
     - Metallic branching stress cracks with shadow grooves & specular highlights
     - Abrasive linear scratches with bright furrows & gouge edges
     - Hydrocarbon oil / coolant stains with organic fluid diffusion
     - 3D impact dents with asymmetric illumination shading
     - Thermal oxidation discoloration with temper color spectra
     - Dimensional edge notch & burr irregularities
     - Pristine baseline surface
  3. Ground-truth binary localization masks (0=normal, 255=defect)
  4. CAD/HUD annotated inspection overlays with bounding boxes, crosshairs, and telemetry
  5. JSON metadata catalog for the inspection pipeline

Zero external dependencies: uses Python standard library (struct, zlib, math, random, json).
"""

import os
import math
import random
import struct
import zlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

WIDTH = 1024
HEIGHT = 1024

def save_png(width: int, height: int, rgb_bytes: bytearray, file_path: Path):
    """Encodes raw RGB bytearray into a compliant PNG and writes to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data))
    ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + ihdr_crc

    raw_scanlines = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw_scanlines.append(0)  # Filter type 0 (None)
        idx = y * row_bytes
        raw_scanlines.extend(rgb_bytes[idx:idx + row_bytes])

    compressed = zlib.compress(bytes(raw_scanlines), level=5)
    idat_crc = struct.pack('>I', zlib.crc32(b'IDAT' + compressed))
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + idat_crc

    iend_crc = struct.pack('>I', zlib.crc32(b'IEND'))
    iend = struct.pack('>I', 0) + b'IEND' + iend_crc

    data = sig + ihdr + idat + iend
    with open(file_path, 'wb') as f:
        f.write(data)


# -----------------------------------------------------------------------------
# Fast Procedural Substrate Generators (1024x1024)
# -----------------------------------------------------------------------------

def generate_brushed_stainless_steel(rng: random.Random) -> bytearray:
    """Simulates brushed 316 stainless steel with directional linear grain."""
    buf = bytearray(WIDTH * HEIGHT * 3)
    base_val = 175
    # Generate horizontal grain streaks
    # Pre-generate 1024 row grain profiles
    row_grains = [rng.randint(-18, 18) for _ in range(HEIGHT)]
    
    # Large wave lighting gradient across the surface
    for y in range(HEIGHT):
        row_g = row_grains[y]
        y_grad = math.sin(y / 160.0) * 8.0
        for x in range(WIDTH):
            # Horizontal streaks with micro variations
            x_var = rng.randint(-6, 6)
            specular_flare = math.sin((x + y * 0.15) / 220.0) * 12.0
            val = int(base_val + row_g + y_grad + x_var + specular_flare)
            val = max(0, min(255, val))
            idx = (y * WIDTH + x) * 3
            buf[idx] = val        # R
            buf[idx + 1] = val    # G
            buf[idx + 2] = min(255, val + 4) # slight cool metallic tint
    return buf


def generate_cast_iron_porous(rng: random.Random) -> bytearray:
    """Simulates cast iron with granular porosity and dark matte finish."""
    buf = bytearray(WIDTH * HEIGHT * 3)
    base_val = 95
    for y in range(HEIGHT):
        for x in range(WIDTH):
            noise = rng.randint(-16, 16)
            # Occasional micro-pore
            if rng.random() < 0.015:
                noise -= rng.randint(25, 45)
            val = max(0, min(255, base_val + noise))
            idx = (y * WIDTH + x) * 3
            buf[idx] = val
            buf[idx + 1] = min(255, val + 2)
            buf[idx + 2] = min(255, val + 6) # dark slate tint
    return buf


def generate_machined_lathe_steel(rng: random.Random) -> bytearray:
    """Simulates lathe-turned steel with circular concentric toolmarks."""
    buf = bytearray(WIDTH * HEIGHT * 3)
    cx, cy = WIDTH // 2, HEIGHT // 2
    base_val = 160
    for y in range(HEIGHT):
        dy = y - cy
        for x in range(WIDTH):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)
            # Tool pass ripples
            toolmark = math.sin(dist * 0.45) * 16.0
            grain = rng.randint(-5, 5)
            # Radial angle glare
            angle = math.atan2(dy, dx)
            sheen = math.sin(angle * 2.0) * 10.0
            val = int(base_val + toolmark + grain + sheen)
            val = max(0, min(255, val))
            idx = (y * WIDTH + x) * 3
            buf[idx] = val
            buf[idx + 1] = val
            buf[idx + 2] = min(255, val + 3)
    return buf


# -----------------------------------------------------------------------------
# High-Resolution Defect Injectors
# -----------------------------------------------------------------------------

def inject_crack_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects high-resolution branching fracture crack with specular edge."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3) # 0=normal, 255=defect

    start_x = 420
    start_y = 260
    num_nodes = 280
    angle = 1.1  # diagonal orientation
    curr_x, curr_y = float(start_x), float(start_y)

    points: List[Tuple[int, int]] = []
    branches: List[List[Tuple[int, int]]] = []

    min_x, max_x = WIDTH, 0
    min_y, max_y = HEIGHT, 0

    for i in range(num_nodes):
        angle += rng.uniform(-0.35, 0.35)
        step = rng.uniform(2.5, 4.2)
        curr_x += math.cos(angle) * step
        curr_y += math.sin(angle) * step
        ix, iy = int(curr_x), int(curr_y)
        if 20 <= ix < WIDTH - 20 and 20 <= iy < HEIGHT - 20:
            points.append((ix, iy))
            min_x, max_x = min(min_x, ix), max(max_x, ix)
            min_y, max_y = min(min_y, iy), max(max_y, iy)
            # Branch off occasionally
            if i in (85, 170) and len(branches) < 2:
                b_pts = []
                bx, by = curr_x, curr_y
                b_ang = angle + (0.75 if len(branches) == 0 else -0.75)
                for _ in range(60):
                    b_ang += rng.uniform(-0.3, 0.3)
                    bx += math.cos(b_ang) * 2.8
                    by += math.sin(b_ang) * 2.8
                    ibx, iby = int(bx), int(by)
                    if 20 <= ibx < WIDTH - 20 and 20 <= iby < HEIGHT - 20:
                        b_pts.append((ibx, iby))
                        min_x, max_x = min(min_x, ibx), max(max_x, ibx)
                        min_y, max_y = min(min_y, iby), max(max_y, iby)
                branches.append(b_pts)

    all_paths = [points] + branches

    # Draw dark crack fissure and bright specular ridge
    defect_pixels = 0
    for path in all_paths:
        for idx, (px, py) in enumerate(path):
            # Draw crack core (radius 2)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx*dx + dy*dy <= 4:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                            p_idx = (ny * WIDTH + nx) * 3
                            # Dark fissure core
                            raw[p_idx] = rng.randint(20, 45)
                            raw[p_idx + 1] = rng.randint(20, 45)
                            raw[p_idx + 2] = rng.randint(22, 48)
                            mask[p_idx] = 255
                            mask[p_idx + 1] = 255
                            mask[p_idx + 2] = 255
                            defect_pixels += 1
            # Specular light edge on the right
            for dy in range(-1, 2):
                nx, ny = px + 3, py + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    p_idx = (ny * WIDTH + nx) * 3
                    if mask[p_idx] == 0:
                        raw[p_idx] = min(255, int(raw[p_idx] * 1.45 + 50))
                        raw[p_idx + 1] = min(255, int(raw[p_idx + 1] * 1.45 + 50))
                        raw[p_idx + 2] = min(255, int(raw[p_idx + 2] * 1.45 + 55))

    pad = 12
    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(WIDTH - 1, max_x + pad) - max(0, min_x - pad), min(HEIGHT - 1, max_y + pad) - max(0, min_y - pad)]
    info = {
        "defect_type": "crack",
        "severity": "CRITICAL",
        "confidence": 0.984,
        "anomaly_score": 0.941,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "High-tensile stress fracture with micro-branching fissures and specular edge deformation."
    }
    return raw, mask, info


def inject_scratch_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects high-aspect abrasive linear scratch with specular light bounce."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3)

    start_x, start_y = 280, 220
    end_x, end_y = 780, 740
    mid_x, mid_y = 510, 460  # slight quadratic arc

    steps = 800
    min_x, max_x = WIDTH, 0
    min_y, max_y = HEIGHT, 0
    defect_pixels = 0

    for i in range(steps):
        t = i / float(steps)
        # Bezier curve
        px = int((1 - t)**2 * start_x + 2 * (1 - t) * t * mid_x + t**2 * end_x)
        py = int((1 - t)**2 * start_y + 2 * (1 - t) * t * mid_y + t**2 * end_y)
        
        min_x, max_x = min(min_x, px), max(max_x, px)
        min_y, max_y = min(min_y, py), max(max_y, py)

        # Specular furrow center (high brightness)
        for d in (-1, 0, 1):
            nx, ny = px + d, py + d
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                idx = (ny * WIDTH + nx) * 3
                raw[idx] = min(255, raw[idx] + 85 + rng.randint(-8, 8))
                raw[idx + 1] = min(255, raw[idx + 1] + 85 + rng.randint(-8, 8))
                raw[idx + 2] = min(255, raw[idx + 2] + 90 + rng.randint(-8, 8))
                mask[idx] = 255
                mask[idx + 1] = 255
                mask[idx + 2] = 255
                defect_pixels += 1

        # Gouge shadow trench on opposite flank
        nx_s, ny_s = px - 2, py + 2
        if 0 <= nx_s < WIDTH and 0 <= ny_s < HEIGHT:
            idx_s = (ny_s * WIDTH + nx_s) * 3
            raw[idx_s] = int(raw[idx_s] * 0.42)
            raw[idx_s + 1] = int(raw[idx_s + 1] * 0.42)
            raw[idx_s + 2] = int(raw[idx_s + 2] * 0.45)

    pad = 14
    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(WIDTH - 1, max_x + pad) - max(0, min_x - pad), min(HEIGHT - 1, max_y + pad) - max(0, min_y - pad)]
    info = {
        "defect_type": "scratch",
        "severity": "MAJOR",
        "confidence": 0.962,
        "anomaly_score": 0.887,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "Mechanical abrasive score mark with burnished specular furrow and gouge shadow trench."
    }
    return raw, mask, info


def inject_stain_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects organic hydrocarbon lubricant fluid stain with feathered diffusion."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3)

    cx, cy = 520, 500
    # Multi-centered organic fluid lobes
    lobes = [
        (cx, cy, 140, 110, 0.72),
        (cx - 50, cy + 40, 95, 80, 0.65),
        (cx + 65, cy - 35, 85, 95, 0.58),
        (cx + 30, cy + 70, 70, 65, 0.50),
    ]

    min_x, max_x = WIDTH, 0
    min_y, max_y = HEIGHT, 0
    defect_pixels = 0

    stain_r, stain_g, stain_b = 68, 48, 22  # dark amber hydrocarbon oil

    for y in range(cy - 180, cy + 190):
        if not (0 <= y < HEIGHT):
            continue
        for x in range(cx - 180, cx + 190):
            if not (0 <= x < WIDTH):
                continue
            # Evaluate fluid intensity from all lobes
            max_alpha = 0.0
            for lx, ly, rx, ry, max_int in lobes:
                dx = (x - lx) / float(rx)
                dy = (y - ly) / float(ry)
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1.0:
                    alpha = (1.0 - dist_sq) * max_int
                    if alpha > max_alpha:
                        max_alpha = alpha

            if max_alpha > 0.05:
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                idx = (y * WIDTH + x) * 3
                
                # Blend oil color with substrate
                a = min(0.82, max_alpha)
                raw[idx] = int(raw[idx] * (1.0 - a) + stain_r * a)
                raw[idx + 1] = int(raw[idx + 1] * (1.0 - a) + stain_g * a)
                raw[idx + 2] = int(raw[idx + 2] * (1.0 - a) + stain_b * a)

                if a > 0.20:
                    mask[idx] = 255
                    mask[idx + 1] = 255
                    mask[idx + 2] = 255
                    defect_pixels += 1

    pad = 15
    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(WIDTH - 1, max_x + pad) - max(0, min_x - pad), min(HEIGHT - 1, max_y + pad) - max(0, min_y - pad)]
    info = {
        "defect_type": "stain",
        "severity": "MODERATE",
        "confidence": 0.947,
        "anomaly_score": 0.865,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "Industrial cutting lubricant / synthetic oil residue with capillary fluid diffusion boundary."
    }
    return raw, mask, info


def inject_dent_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects 3D radial impact crater dent with directional shadow and highlight."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3)

    cx, cy = 512, 490
    radius = 125
    defect_pixels = 0
    light_angle = 0.785  # ~45 degrees illumination from top-left

    min_x, max_x = cx - radius, cx + radius
    min_y, max_y = cy - radius, cy + radius

    for dy in range(-radius, radius + 1):
        y = cy + dy
        if not (0 <= y < HEIGHT):
            continue
        for dx in range(-radius, radius + 1):
            x = cx + dx
            if not (0 <= x < WIDTH):
                continue
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                # Radial falloff
                factor = math.cos((dist / radius) * (math.pi / 2.0))
                # Directional surface normal shading
                norm_dx = dx / (radius + 1e-4)
                norm_dy = dy / (radius + 1e-4)
                shading = (norm_dx * math.cos(light_angle) + norm_dy * math.sin(light_angle)) * factor * 68.0

                idx = (y * WIDTH + x) * 3
                raw[idx] = max(0, min(255, int(raw[idx] + shading)))
                raw[idx + 1] = max(0, min(255, int(raw[idx + 1] + shading)))
                raw[idx + 2] = max(0, min(255, int(raw[idx + 2] + shading)))

                mask[idx] = 255
                mask[idx + 1] = 255
                mask[idx + 2] = 255
                defect_pixels += 1

    pad = 12
    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(WIDTH - 1, max_x + pad) - max(0, min_x - pad), min(HEIGHT - 1, max_y + pad) - max(0, min_y - pad)]
    info = {
        "defect_type": "dent",
        "severity": "MAJOR",
        "confidence": 0.958,
        "anomaly_score": 0.892,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "Spherical tooling impact impression with asymmetric Lambertian directional shading."
    }
    return raw, mask, info


def inject_discoloration_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects thermal oxidation heat affected zone (HAZ) with temper color gradient."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3)

    cx, cy = 490, 510
    rx, ry = 180, 130
    defect_pixels = 0

    min_x, max_x = cx - rx, cx + rx
    min_y, max_y = cy - ry, cy + ry

    for dy in range(-ry, ry + 1):
        y = cy + dy
        if not (0 <= y < HEIGHT):
            continue
        for dx in range(-rx, rx + 1):
            x = cx + dx
            if not (0 <= x < WIDTH):
                continue
            dist_sq = (dx / float(rx))**2 + (dy / float(ry))**2
            if dist_sq <= 1.0:
                intensity = (1.0 - dist_sq)
                idx = (y * WIDTH + x) * 3
                # Heat temper colors: blue perimeter, magenta mid-ring, bronze core
                r_orig = raw[idx]
                g_orig = raw[idx + 1]
                b_orig = raw[idx + 2]

                if intensity > 0.65:
                    # Bronze / straw yellow core
                    raw[idx] = min(255, int(r_orig * 1.35 + 30 * intensity))
                    raw[idx + 1] = min(255, int(g_orig * 1.15 + 15 * intensity))
                    raw[idx + 2] = max(0, int(b_orig * 0.60))
                elif intensity > 0.35:
                    # Peacock magenta / violet
                    raw[idx] = min(255, int(r_orig * 1.25 + 40 * intensity))
                    raw[idx + 1] = max(0, int(g_orig * 0.70))
                    raw[idx + 2] = min(255, int(b_orig * 1.30 + 35 * intensity))
                else:
                    # Electric heat blue fringe
                    raw[idx] = max(0, int(r_orig * 0.65))
                    raw[idx + 1] = min(255, int(g_orig * 0.90 + 20 * intensity))
                    raw[idx + 2] = min(255, int(b_orig * 1.45 + 50 * intensity))

                mask[idx] = 255
                mask[idx + 1] = 255
                mask[idx + 2] = 255
                defect_pixels += 1

    pad = 15
    bbox = [max(0, min_x - pad), max(0, min_y - pad), min(WIDTH - 1, max_x + pad) - max(0, min_x - pad), min(HEIGHT - 1, max_y + pad) - max(0, min_y - pad)]
    info = {
        "defect_type": "discoloration",
        "severity": "MODERATE",
        "confidence": 0.932,
        "anomaly_score": 0.814,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "Thermal oxidation heat-affected zone exhibiting multi-spectral temper color oxidation."
    }
    return raw, mask, info


def inject_dimensional_irregularity_1024(img_buf: bytearray, rng: random.Random) -> Tuple[bytearray, bytearray, Dict[str, Any]]:
    """Injects precision perimeter notch defect violating tolerance boundaries."""
    raw = bytearray(img_buf)
    mask = bytearray(WIDTH * HEIGHT * 3)

    notch_w = 90
    notch_h = 130
    notch_x = 460
    notch_y = 0  # top edge cutout
    defect_pixels = 0

    for dy in range(notch_h):
        y = notch_y + dy
        # Triangular notch shape
        curr_w = int(notch_w * (1.0 - dy / float(notch_h)))
        for dx in range(-curr_w, curr_w + 1):
            x = notch_x + dx
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                idx = (y * WIDTH + x) * 3
                # Replace with dark optical inspection stage fixture
                raw[idx] = 16
                raw[idx + 1] = 18
                raw[idx + 2] = 22
                mask[idx] = 255
                mask[idx + 1] = 255
                mask[idx + 2] = 255
                defect_pixels += 1

    bbox = [notch_x - notch_w - 6, 0, (notch_w * 2) + 12, notch_h + 8]
    info = {
        "defect_type": "dimensional_irregularity",
        "severity": "CRITICAL",
        "confidence": 0.979,
        "anomaly_score": 0.935,
        "bbox": bbox,
        "dimensions_mm": {"length": round(bbox[2] * 0.1, 2), "width": round(bbox[3] * 0.1, 2)},
        "defect_pixels": defect_pixels,
        "description": "Perimeter geometry violation: material excision notch breaching CAD envelope tolerance."
    }
    return raw, mask, info


# -----------------------------------------------------------------------------
# CAD / HUD Annotation Overlay Engine
# -----------------------------------------------------------------------------

def create_annotated_hud(raw_buf: bytearray, info: Dict[str, Any]) -> bytearray:
    """Creates a high-precision CAD inspection HUD overlay with bounding box, reticle, & telemetry."""
    hud = bytearray(raw_buf)
    bbox = info["bbox"]
    is_defective = info.get("defect_type") != "normal"
    
    # Border color: Rose (#f43f5e) for defect, Emerald (#10b981) for normal
    box_r, box_g, box_b = (244, 63, 94) if is_defective else (16, 185, 129)

    if is_defective and bbox[2] > 0 and bbox[3] > 0:
        bx, by, bw, bh = bbox
        # Draw 2px bounding box
        for t in range(3):
            # Top & bottom lines
            for x in range(bx, bx + bw):
                for py in (by + t, by + bh - t):
                    if 0 <= x < WIDTH and 0 <= py < HEIGHT:
                        idx = (py * WIDTH + x) * 3
                        hud[idx] = box_r
                        hud[idx + 1] = box_g
                        hud[idx + 2] = box_b
            # Left & right lines
            for y in range(by, by + bh):
                for px in (bx + t, bx + bw - t):
                    if 0 <= px < WIDTH and 0 <= y < HEIGHT:
                        idx = (y * WIDTH + px) * 3
                        hud[idx] = box_r
                        hud[idx + 1] = box_g
                        hud[idx + 2] = box_b

        # Corner bracket accents (length 24px)
        corner_len = 28
        for cy_pos in (by, by + bh):
            for cx_pos in (bx, bx + bw):
                dx_dir = 1 if cx_pos == bx else -1
                dy_dir = 1 if cy_pos == by else -1
                for l in range(corner_len):
                    for w_thick in (-1, 0, 1):
                        # Horizontal arm
                        px = cx_pos + (l * dx_dir)
                        py = cy_pos + w_thick
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            idx = (py * WIDTH + px) * 3
                            hud[idx], hud[idx+1], hud[idx+2] = 255, 255, 255
                        # Vertical arm
                        px2 = cx_pos + w_thick
                        py2 = cy_pos + (l * dy_dir)
                        if 0 <= px2 < WIDTH and 0 <= py2 < HEIGHT:
                            idx2 = (py2 * WIDTH + px2) * 3
                            hud[idx2], hud[idx2+1], hud[idx2+2] = 255, 255, 255

        # Center target reticle
        cx = bx + bw // 2
        cy = by + bh // 2
        reticle_r = 18
        for angle_deg in range(0, 360, 4):
            rad = math.radians(angle_deg)
            rx = int(cx + math.cos(rad) * reticle_r)
            ry = int(cy + math.sin(rad) * reticle_r)
            if 0 <= rx < WIDTH and 0 <= ry < HEIGHT:
                idx = (ry * WIDTH + rx) * 3
                hud[idx], hud[idx+1], hud[idx+2] = box_r, box_g, box_b

    # Upper inspection banner bar (translucent dark bar)
    banner_h = 44
    for y in range(banner_h):
        for x in range(WIDTH):
            idx = (y * WIDTH + x) * 3
            hud[idx] = int(hud[idx] * 0.25 + 15)
            hud[idx + 1] = int(hud[idx + 1] * 0.25 + 20)
            hud[idx + 2] = int(hud[idx + 2] * 0.25 + 28)

    # Status indicator square in upper left
    for y in range(12, 32):
        for x in range(16, 36):
            idx = (y * WIDTH + x) * 3
            hud[idx], hud[idx + 1], hud[idx + 2] = box_r, box_g, box_b

    return hud


# -----------------------------------------------------------------------------
# Main Generation Orchestrator
# -----------------------------------------------------------------------------

def generate_all_highres_samples():
    """Generates the comprehensive set of 1024x1024 sample assets."""
    print("=" * 72)
    print("  INSPECTRA AI — HIGH-RESOLUTION SYNTHETIC SPECIMEN GENERATOR (1024x1024)")
    print("=" * 72)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "test_samples"
    public_dir = project_root / "public" / "samples"

    data_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    rng = random.Random(2026)

    specimen_configs = [
        {
            "id": "specimen_highres_00_normal",
            "class": "normal",
            "label": "Pristine Brushed 316L Stainless Steel",
            "substrate": "brushed_stainless_steel",
            "injector": None,
            "generator": generate_brushed_stainless_steel
        },
        {
            "id": "specimen_highres_01_crack",
            "class": "crack",
            "label": "Metallic Micro-Branching Stress Crack",
            "substrate": "brushed_stainless_steel",
            "injector": inject_crack_1024,
            "generator": generate_brushed_stainless_steel
        },
        {
            "id": "specimen_highres_02_scratch",
            "class": "scratch",
            "label": "Abrasive High-Aspect Specular Scratch",
            "substrate": "machined_lathe_steel",
            "injector": inject_scratch_1024,
            "generator": generate_machined_lathe_steel
        },
        {
            "id": "specimen_highres_03_stain",
            "class": "stain",
            "label": "Hydrocarbon Lubricant Oil Diffusion Stain",
            "substrate": "cast_iron_porous",
            "injector": inject_stain_1024,
            "generator": generate_cast_iron_porous
        },
        {
            "id": "specimen_highres_04_dent",
            "class": "dent",
            "label": "3D Directional Impact Dent Depression",
            "substrate": "brushed_stainless_steel",
            "injector": inject_dent_1024,
            "generator": generate_brushed_stainless_steel
        },
        {
            "id": "specimen_highres_05_discoloration",
            "class": "discoloration",
            "label": "Thermal Oxidation Heat Tint Temper HAZ",
            "substrate": "brushed_stainless_steel",
            "injector": inject_discoloration_1024,
            "generator": generate_brushed_stainless_steel
        },
        {
            "id": "specimen_highres_06_dimensional_irregularity",
            "class": "dimensional_irregularity",
            "label": "Precision Edge Notch Tolerance Violation",
            "substrate": "machined_lathe_steel",
            "injector": inject_dimensional_irregularity_1024,
            "generator": generate_machined_lathe_steel
        },
    ]

    for idx, cfg in enumerate(specimen_configs):
        name = cfg["id"]
        c_name = cfg["class"]
        print(f"[*] [{idx+1}/{len(specimen_configs)}] Generating {name} (1024x1024) — {cfg['label']}...")

        # 1. Generate base substrate
        base = cfg["generator"](rng)

        # 2. Inject defect or keep pristine
        if cfg["injector"] is None:
            raw = base
            mask = bytearray(WIDTH * HEIGHT * 3)
            info = {
                "defect_type": "normal",
                "severity": "NONE",
                "confidence": 0.991,
                "anomaly_score": 0.082,
                "bbox": [0, 0, 0, 0],
                "dimensions_mm": {"length": 0.0, "width": 0.0},
                "defect_pixels": 0,
                "description": "Defect-free pristine manufacturing substrate with standard grain structure."
            }
        else:
            raw, mask, info = cfg["injector"](base, rng)

        # 3. Create annotated HUD
        hud = create_annotated_hud(raw, info)

        # 4. Save raw image, mask, and annotated HUD to both directories
        raw_filename = f"{name}.png"
        mask_filename = f"{name}_mask.png"
        hud_filename = f"{name}_annotated.png"

        for target_folder in (data_dir, public_dir):
            save_png(WIDTH, HEIGHT, raw, target_folder / raw_filename)
            save_png(WIDTH, HEIGHT, mask, target_folder / mask_filename)
            save_png(WIDTH, HEIGHT, hud, target_folder / hud_filename)

        catalog_entry = {
            "id": name,
            "title": cfg["label"],
            "defect_class": c_name,
            "is_defective": c_name != "normal",
            "resolution": f"{WIDTH}x{HEIGHT}",
            "substrate": cfg["substrate"],
            "severity": info["severity"],
            "confidence": info["confidence"],
            "anomaly_score": info["anomaly_score"],
            "dimensions_mm": info["dimensions_mm"],
            "bbox": info["bbox"],
            "defect_pixels": info["defect_pixels"],
            "description": info["description"],
            "files": {
                "raw": f"/samples/{raw_filename}",
                "mask": f"/samples/{mask_filename}",
                "annotated": f"/samples/{hud_filename}"
            }
        }
        catalog.append(catalog_entry)

    # Write JSON catalog to both outputs and public
    catalog_data = {
        "generator_version": "2.4.0-highres",
        "resolution": {"width": WIDTH, "height": HEIGHT},
        "total_specimens": len(catalog),
        "specimens": catalog
    }

    cat_json_path1 = data_dir / "specimen_highres_catalog.json"
    cat_json_path2 = public_dir / "specimen_highres_catalog.json"
    src_json_path = project_root / "src" / "data" / "highres_samples_catalog.json"
    src_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cat_json_path1, 'w') as f:
        json.dump(catalog_data, f, indent=2)
    with open(cat_json_path2, 'w') as f:
        json.dump(catalog_data, f, indent=2)
    with open(src_json_path, 'w') as f:
        json.dump(catalog_data, f, indent=2)

    print(f"\n[+] Successfully generated {len(catalog)} high-resolution synthetic defect specimens (1024x1024)!")
    print(f"    Raw images, masks, and annotated overlays saved to:")
    print(f"      - {data_dir}")
    print(f"      - {public_dir}")
    print(f"      - {src_json_path}")
    print("=" * 72)


if __name__ == "__main__":
    generate_all_highres_samples()
