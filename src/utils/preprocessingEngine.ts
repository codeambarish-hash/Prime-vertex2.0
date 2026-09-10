/**
 * Inspectra AI — Preprocessing Simulation Engine
 * Matches Python/OpenCV operations in src/preprocessing.py:
 * 1. Aspect-Ratio Preserving Letterbox Resizing & Symmetric Padding
 * 2. CLAHE (Contrast-Limited Adaptive Histogram Equalization) on L-channel in LAB space
 * 3. Edge-Preserving Bilateral Denoising (preserves sharp cracks & scratches)
 * 4. Background Normalization & Conveyor Suppression
 * 5. Luminance Histogram Extraction (Before vs After CLAHE)
 */

export interface PreprocessingConfig {
  targetSize: [number, number];
  applyBgNorm: boolean;
  applyClahe: boolean;
  claheClipLimit: number;
  claheTileSize: number;
  applyBilateral: boolean;
  bilateralRadius: number;
  bilateralSigmaColor: number;
  bilateralSigmaSpace: number;
  applyRoiCrop: boolean;
  rotationDeg: number;
  flipH: boolean;
  flipV: boolean;
}

export interface PreprocessingStages {
  raw: ImageData;
  bgNormalized: ImageData;
  clahe: ImageData;
  bilateral: ImageData;
  letterbox: ImageData;
  meta: {
    origSize: [number, number];
    scaledSize: [number, number];
    targetSize: [number, number];
    scale: number;
    padTop: number;
    padBottom: number;
    padLeft: number;
    padRight: number;
  };
  histogramBefore: number[];
  histogramAfter: number[];
}

/**
 * Converts RGB [0..255] to LAB space
 */
function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  // 1. Normalize and apply sRGB gamma correction to linear RGB
  let R = r / 255;
  let G = g / 255;
  let B = b / 255;

  R = R > 0.04045 ? Math.pow((R + 0.055) / 1.055, 2.4) : R / 12.92;
  G = G > 0.04045 ? Math.pow((G + 0.055) / 1.055, 2.4) : G / 12.92;
  B = B > 0.04045 ? Math.pow((B + 0.055) / 1.055, 2.4) : B / 12.92;

  // 2. Convert RGB to CIE XYZ (Observer = 2°, Illuminant = D65)
  const X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
  const Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 1.00000;
  const Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;

  // 3. Convert XYZ to CIELAB
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X);
  const fy = f(Y);
  const fz = f(Z);

  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const bVal = 200 * (fy - fz);

  return [Math.max(0, Math.min(100, L)), a, bVal];
}

/**
 * Converts LAB space back to RGB [0..255]
 */
function labToRgb(L: number, a: number, bVal: number): [number, number, number] {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - bVal / 200;

  const fInv = (t: number) => (t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787);

  const X = fInv(fx) * 0.95047;
  const Y = fInv(fy) * 1.00000;
  const Z = fInv(fz) * 1.08883;

  // CIE XYZ to sRGB
  let R = X * 3.2406 + Y * -1.5372 + Z * -0.4986;
  let G = X * -0.9689 + Y * 1.8758 + Z * 0.0415;
  let B = X * 0.0557 + Y * -0.2040 + Z * 1.0570;

  const gammaCorrect = (c: number) =>
    c > 0.0031308 ? 1.055 * Math.pow(c, 1 / 2.4) - 0.055 : 12.92 * c;

  const rFinal = Math.max(0, Math.min(255, Math.round(gammaCorrect(R) * 255)));
  const gFinal = Math.max(0, Math.min(255, Math.round(gammaCorrect(G) * 255)));
  const bFinal = Math.max(0, Math.min(255, Math.round(gammaCorrect(B) * 255)));

  return [rFinal, gFinal, bFinal];
}

/**
 * Computes a 256-bin luminance histogram from ImageData
 */
export function computeLuminanceHistogram(imgData: ImageData): number[] {
  const hist = new Array(256).fill(0);
  const data = imgData.data;
  const len = data.length;

  for (let i = 0; i < len; i += 4) {
    // Standard perceptual luminance: 0.299*R + 0.587*G + 0.114*B
    const lum = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    hist[lum] = (hist[lum] || 0) + 1;
  }
  return hist;
}

/**
 * Simulates CLAHE on the L-channel in LAB space across grid tiles
 */
export function simulateClahe(
  src: ImageData,
  clipLimit: number = 2.5,
  gridSize: number = 8
): ImageData {
  const width = src.width;
  const height = src.height;
  const dst = new ImageData(new Uint8ClampedArray(src.data), width, height);

  // Extract L channel arrays
  const LArr = new Float32Array(width * height);
  const aArr = new Float32Array(width * height);
  const bArr = new Float32Array(width * height);

  const srcData = src.data;
  for (let i = 0, p = 0; i < srcData.length; i += 4, p++) {
    const [l, a, bVal] = rgbToLab(srcData[i], srcData[i + 1], srcData[i + 2]);
    LArr[p] = l;
    aArr[p] = a;
    bArr[p] = bVal;
  }

  // Divide into grid tiles and compute contextual equalized mappings
  const tileW = width / gridSize;
  const tileH = height / gridSize;
  const numBins = 64; // Quantized bins for tile histogram
  const tileLut: Float32Array[] = [];

  for (let ty = 0; ty < gridSize; ty++) {
    for (let tx = 0; tx < gridSize; tx++) {
      const hist = new Float32Array(numBins);
      let count = 0;

      const startX = Math.floor(tx * tileW);
      const endX = Math.min(width, Math.floor((tx + 1) * tileW));
      const startY = Math.floor(ty * tileH);
      const endY = Math.min(height, Math.floor((ty + 1) * tileH));

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const lVal = LArr[y * width + x];
          const bin = Math.min(numBins - 1, Math.floor((lVal / 100) * numBins));
          hist[bin]++;
          count++;
        }
      }

      // Clip histogram according to clipLimit
      const clipThreshold = (clipLimit * count) / numBins;
      let excess = 0;
      for (let b = 0; b < numBins; b++) {
        if (hist[b] > clipThreshold) {
          excess += hist[b] - clipThreshold;
          hist[b] = clipThreshold;
        }
      }
      const excessPerBin = excess / numBins;
      for (let b = 0; b < numBins; b++) {
        hist[b] += excessPerBin;
      }

      // Compute CDF (Cumulative Distribution Function) mapping
      const lut = new Float32Array(numBins);
      let sum = 0;
      for (let b = 0; b < numBins; b++) {
        sum += hist[b];
        lut[b] = (sum / count) * 100.0;
      }
      tileLut.push(lut);
    }
  }

  // Bilinear interpolation between neighboring tile mappings for seamless boundaries
  const dstData = dst.data;
  for (let y = 0; y < height; y++) {
    const ty = y / tileH - 0.5;
    const ty0 = Math.max(0, Math.min(gridSize - 1, Math.floor(ty)));
    const ty1 = Math.max(0, Math.min(gridSize - 1, ty0 + 1));
    const dy = ty - ty0;

    for (let x = 0; x < width; x++) {
      const tx = x / tileW - 0.5;
      const tx0 = Math.max(0, Math.min(gridSize - 1, Math.floor(tx)));
      const tx1 = Math.max(0, Math.min(gridSize - 1, tx0 + 1));
      const dx = tx - tx0;

      const idx = y * width + x;
      const originalL = LArr[idx];
      const bin = Math.min(numBins - 1, Math.max(0, Math.floor((originalL / 100) * numBins)));

      const lut00 = tileLut[ty0 * gridSize + tx0][bin];
      const lut10 = tileLut[ty0 * gridSize + tx1][bin];
      const lut01 = tileLut[ty1 * gridSize + tx0][bin];
      const lut11 = tileLut[ty1 * gridSize + tx1][bin];

      const top = lut00 * (1 - dx) + lut10 * dx;
      const bottom = lut01 * (1 - dx) + lut11 * dx;
      const newL = top * (1 - dy) + bottom * dy;

      // Recombine with original chromaticity (a, b)
      const [r, g, b] = labToRgb(newL, aArr[idx], bArr[idx]);
      const pixelIdx = idx * 4;
      dstData[pixelIdx] = r;
      dstData[pixelIdx + 1] = g;
      dstData[pixelIdx + 2] = b;
      dstData[pixelIdx + 3] = 255;
    }
  }

  return dst;
}

/**
 * Simulates Bilateral Edge-Preserving Denoising
 * Weighted combination of geometric distance and radiometric intensity difference
 */
export function simulateBilateralDenoising(
  src: ImageData,
  radius: number = 3,
  sigmaColor: number = 40.0,
  sigmaSpace: number = 30.0
): ImageData {
  const width = src.width;
  const height = src.height;
  const dst = new ImageData(new Uint8ClampedArray(src.data), width, height);

  const srcData = src.data;
  const dstData = dst.data;

  const twoSigmaSpaceSq = 2 * sigmaSpace * sigmaSpace;
  const twoSigmaColorSq = 2 * sigmaColor * sigmaColor;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerIdx = (y * width + x) * 4;
      const cR = srcData[centerIdx];
      const cG = srcData[centerIdx + 1];
      const cB = srcData[centerIdx + 2];

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let sumWeight = 0;

      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;

        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;

          const nIdx = (ny * width + nx) * 4;
          const nR = srcData[nIdx];
          const nG = srcData[nIdx + 1];
          const nB = srcData[nIdx + 2];

          // Geometric spatial distance weight
          const spaceDistSq = dx * dx + dy * dy;
          const spaceWeight = Math.exp(-spaceDistSq / twoSigmaSpaceSq);

          // Photometric radiometric color difference weight
          const colorDistSq =
            (cR - nR) * (cR - nR) +
            (cG - nG) * (cG - nG) +
            (cB - nB) * (cB - nB);
          const colorWeight = Math.exp(-colorDistSq / twoSigmaColorSq);

          const weight = spaceWeight * colorWeight;

          sumR += nR * weight;
          sumG += nG * weight;
          sumB += nB * weight;
          sumWeight += weight;
        }
      }

      if (sumWeight > 0) {
        dstData[centerIdx] = Math.round(sumR / sumWeight);
        dstData[centerIdx + 1] = Math.round(sumG / sumWeight);
        dstData[centerIdx + 2] = Math.round(sumB / sumWeight);
      }
    }
  }

  return dst;
}

/**
 * Background Normalization: Detects workpiece boundaries and standardizes conveyor background
 */
export function simulateBackgroundNormalization(
  src: ImageData,
  bgFill: [number, number, number] = [18, 20, 26]
): ImageData {
  const width = src.width;
  const height = src.height;
  const dst = new ImageData(new Uint8ClampedArray(src.data), width, height);
  const srcData = src.data;
  const dstData = dst.data;

  // Detect component footprint using luminance gradient and distance from perimeter
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = srcData[idx];
      const g = srcData[idx + 1];
      const b = srcData[idx + 2];

      // Simulated conveyor belt test:
      // Conveyor rubber borders have dark/flat tone or outer edge location
      const isConveyorMargin =
        (x < 24 || x > width - 24 || y < 24 || y > height - 24) &&
        (r < 55 && g < 55 && b < 60);

      if (isConveyorMargin) {
        dstData[idx] = bgFill[0];
        dstData[idx + 1] = bgFill[1];
        dstData[idx + 2] = bgFill[2];
        dstData[idx + 3] = 255;
      }
    }
  }

  return dst;
}

/**
 * Aspect-Ratio Preserving Letterbox Resizing
 */
export function simulateLetterboxResize(
  src: ImageData,
  targetSize: [number, number] = [256, 256],
  padColor: [number, number, number] = [15, 15, 20]
): {
  imageData: ImageData;
  meta: PreprocessingStages['meta'];
} {
  const srcW = src.width;
  const srcH = src.height;
  const [targetW, targetH] = targetSize;

  const scale = Math.min(targetW / srcW, targetH / srcH);
  const scaledW = Math.max(1, Math.round(srcW * scale));
  const scaledH = Math.max(1, Math.round(srcH * scale));

  const padTop = Math.floor((targetH - scaledH) / 2);
  const padBottom = targetH - scaledH - padTop;
  const padLeft = Math.floor((targetW - scaledW) / 2);
  const padRight = targetW - scaledW - padLeft;

  // Create target canvas
  const dst = new ImageData(targetW, targetH);
  const dstData = dst.data;

  // Initialize with pad color
  for (let i = 0; i < dstData.length; i += 4) {
    dstData[i] = padColor[0];
    dstData[i + 1] = padColor[1];
    dstData[i + 2] = padColor[2];
    dstData[i + 3] = 255;
  }

  // Draw scaled image into center with nearest/bilinear interpolation
  const srcData = src.data;
  for (let dy = 0; dy < scaledH; dy++) {
    const sy = Math.min(srcH - 1, Math.floor(dy / scale));
    const targetY = padTop + dy;

    for (let dx = 0; dx < scaledW; dx++) {
      const sx = Math.min(srcW - 1, Math.floor(dx / scale));
      const targetX = padLeft + dx;

      const srcIdx = (sy * srcW + sx) * 4;
      const dstIdx = (targetY * targetW + targetX) * 4;

      dstData[dstIdx] = srcData[srcIdx];
      dstData[dstIdx + 1] = srcData[srcIdx + 1];
      dstData[dstIdx + 2] = srcData[srcIdx + 2];
      dstData[dstIdx + 3] = 255;
    }
  }

  return {
    imageData: dst,
    meta: {
      origSize: [srcW, srcH],
      scaledSize: [scaledW, scaledH],
      targetSize: [targetW, targetH],
      scale,
      padTop,
      padBottom,
      padLeft,
      padRight,
    },
  };
}
