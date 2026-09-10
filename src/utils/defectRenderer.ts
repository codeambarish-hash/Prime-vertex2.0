import { DefectClass, SubstrateType, LightingCondition, BoundingBox, InspectionResult, DefectRegionDetail } from '../types/inspection';

/**
 * Procedural canvas rendering engine matching Python/OpenCV synthetic defect generator.
 * Produces:
 *   1. Raw Simulated Product Image
 *   2. Ground Truth Localization Mask
 *   3. Colorized Anomaly Heatmap
 */
export function renderSimulation(
  canvas: HTMLCanvasElement,
  maskCanvas: HTMLCanvasElement,
  heatmapCanvas: HTMLCanvasElement,
  substrate: SubstrateType,
  defect: DefectClass,
  lighting: LightingCondition,
  rotationDeg: number,
  seed: number = 42
): InspectionResult {
  const ctx = canvas.getContext('2d');
  const maskCtx = maskCanvas.getContext('2d');
  const heatCtx = heatmapCanvas.getContext('2d');

  if (!ctx || !maskCtx || !heatCtx) {
    throw new Error('Canvas 2D context not available');
  }

  const width = canvas.width;
  const height = canvas.height;

  // Simple deterministic PRNG based on seed
  let s = seed + 1;
  const rand = () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };

  // 1. Draw Base Substrate
  ctx.save();
  ctx.clearRect(0, 0, width, height);
  maskCtx.clearRect(0, 0, width, height);
  maskCtx.fillStyle = '#000000';
  maskCtx.fillRect(0, 0, width, height);

  if (substrate === 'brushed_metal') {
    ctx.fillStyle = '#c5c9ce';
    ctx.fillRect(0, 0, width, height);
    // Brushed linear grains
    for (let i = 0; i < 450; i++) {
      const y = rand() * height;
      const len = 40 + rand() * 180;
      const x = rand() * width - 40;
      const alpha = 0.03 + rand() * 0.08;
      ctx.strokeStyle = rand() > 0.5 ? `rgba(255,255,255,${alpha})` : `rgba(40,45,50,${alpha})`;
      ctx.lineWidth = 1 + rand() * 1.5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + len, y);
      ctx.stroke();
    }
  } else if (substrate === 'cast_iron') {
    ctx.fillStyle = '#64686e';
    ctx.fillRect(0, 0, width, height);
    // Sand cast micro-cavities
    for (let i = 0; i < 900; i++) {
      const x = rand() * width;
      const y = rand() * height;
      const r = 0.8 + rand() * 2.2;
      const isDark = rand() > 0.4;
      ctx.fillStyle = isDark ? 'rgba(25,28,32,0.35)' : 'rgba(210,215,220,0.25)';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (substrate === 'machined_part') {
    ctx.fillStyle = '#9da3aa';
    ctx.fillRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    // Lathe circular toolmarks
    for (let r = 8; r < width * 0.8; r += 5 + rand() * 4) {
      ctx.strokeStyle = rand() > 0.5 ? 'rgba(240,245,250,0.22)' : 'rgba(35,40,45,0.22)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  } else if (substrate === 'ceramic_tile') {
    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, '#e5e7eb');
    grad.addColorStop(1, '#d1d5db');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);
    // Fine glaze grain
    for (let i = 0; i < 300; i++) {
      ctx.fillStyle = rand() > 0.5 ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.08)';
      ctx.fillRect(rand() * width, rand() * height, 1.5, 1.5);
    }
  } else {
    // Carbon composite
    ctx.fillStyle = '#1e2229';
    ctx.fillRect(0, 0, width, height);
    const tileSize = 14;
    for (let x = 0; x < width; x += tileSize) {
      for (let y = 0; y < height; y += tileSize) {
        const isOdd = ((x / tileSize) + (y / tileSize)) % 2 === 0;
        ctx.fillStyle = isOdd ? '#282e38' : '#181b20';
        ctx.fillRect(x, y, tileSize, tileSize);
      }
    }
  }

  // 2. Inject Selected Defect and generate Ground Truth Mask
  let bbox: BoundingBox = { x: 0, y: 0, w: 0, h: 0 };
  let defectPixels = 0;

  if (defect !== 'normal') {
    if (defect === 'crack') {
      let cx = width * 0.3 + rand() * (width * 0.4);
      let cy = height * 0.3 + rand() * (height * 0.4);
      const points: [number, number][] = [[cx, cy]];
      let angle = rand() * Math.PI * 2;
      let minX = cx, maxX = cx, minY = cy, maxY = cy;

      for (let i = 0; i < 55; i++) {
        angle += (rand() - 0.5) * 0.9;
        const step = 2.5 + rand() * 3.5;
        cx += Math.cos(angle) * step;
        cy += Math.sin(angle) * step;
        if (cx > 10 && cx < width - 10 && cy > 10 && cy < height - 10) {
          points.push([cx, cy]);
          minX = Math.min(minX, cx);
          maxX = Math.max(maxX, cx);
          minY = Math.min(minY, cy);
          maxY = Math.max(maxY, cy);
        }
      }

      // Draw onto image (dark core + specular white highlight)
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#181a1b';
      ctx.lineCap = 'round';
      ctx.beginPath();
      points.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();

      // Specular ridge
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.75)';
      ctx.beginPath();
      points.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px + 1.5, py + 1.5) : ctx.lineTo(px + 1.5, py + 1.5)));
      ctx.stroke();

      // Draw onto mask
      maskCtx.lineWidth = 3.5;
      maskCtx.strokeStyle = '#ffffff';
      maskCtx.lineCap = 'round';
      maskCtx.beginPath();
      points.forEach(([px, py], i) => (i === 0 ? maskCtx.moveTo(px, py) : maskCtx.lineTo(px, py)));
      maskCtx.stroke();

      bbox = {
        x: Math.max(0, Math.floor(minX - 4)),
        y: Math.max(0, Math.floor(minY - 4)),
        w: Math.min(width, Math.ceil(maxX - minX + 8)),
        h: Math.min(height, Math.ceil(maxY - minY + 8)),
      };
      defectPixels = Math.floor(points.length * 9);

    } else if (defect === 'scratch') {
      const sx = width * 0.2 + rand() * (width * 0.4);
      const sy = height * 0.2 + rand() * (height * 0.4);
      const angle = rand() * Math.PI * 2;
      const len = 60 + rand() * 80;
      const ex = sx + Math.cos(angle) * len;
      const ey = sy + Math.sin(angle) * len;
      const midx = (sx + ex) / 2 + (rand() - 0.5) * 20;
      const midy = (sy + ey) / 2 + (rand() - 0.5) * 20;

      // Bright groove cut
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.quadraticCurveTo(midx, midy, ex, ey);
      ctx.stroke();

      // Shadow companion
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(20, 20, 25, 0.65)';
      ctx.beginPath();
      ctx.moveTo(sx + 1.2, sy + 1.2);
      ctx.quadraticCurveTo(midx + 1.2, midy + 1.2, ex + 1.2, ey + 1.2);
      ctx.stroke();

      // Mask
      maskCtx.lineWidth = 2.5;
      maskCtx.strokeStyle = '#ffffff';
      maskCtx.beginPath();
      maskCtx.moveTo(sx, sy);
      maskCtx.quadraticCurveTo(midx, midy, ex, ey);
      maskCtx.stroke();

      const minX = Math.min(sx, ex, midx);
      const maxX = Math.max(sx, ex, midx);
      const minY = Math.min(sy, ey, midy);
      const maxY = Math.max(sy, ey, midy);
      bbox = {
        x: Math.max(0, Math.floor(minX - 3)),
        y: Math.max(0, Math.floor(minY - 3)),
        w: Math.ceil(maxX - minX + 6),
        h: Math.ceil(maxY - minY + 6),
      };
      defectPixels = Math.floor(len * 4);

    } else if (defect === 'dent') {
      const cx = width * 0.35 + rand() * (width * 0.3);
      const cy = height * 0.35 + rand() * (height * 0.3);
      const radius = 22 + rand() * 20;

      // 3D illumination gradient (shadow on upper left, light bounce on lower right)
      const dentGrad = ctx.createLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius);
      dentGrad.addColorStop(0, 'rgba(15, 18, 22, 0.65)');
      dentGrad.addColorStop(0.5, 'rgba(80, 85, 92, 0.1)');
      dentGrad.addColorStop(1, 'rgba(255, 255, 255, 0.6)');

      ctx.fillStyle = dentGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Mask
      maskCtx.fillStyle = '#ffffff';
      maskCtx.beginPath();
      maskCtx.arc(cx, cy, radius, 0, Math.PI * 2);
      maskCtx.fill();

      bbox = {
        x: Math.max(0, Math.floor(cx - radius)),
        y: Math.max(0, Math.floor(cy - radius)),
        w: Math.ceil(radius * 2),
        h: Math.ceil(radius * 2),
      };
      defectPixels = Math.floor(Math.PI * radius * radius);

    } else if (defect === 'stain') {
      const cx = width * 0.35 + rand() * (width * 0.3);
      const cy = height * 0.35 + rand() * (height * 0.3);
      const baseR = 24 + rand() * 18;

      ctx.fillStyle = 'rgba(75, 45, 20, 0.55)'; // organic oil stain
      maskCtx.fillStyle = '#ffffff';

      for (let i = 0; i < 5; i++) {
        const ox = cx + (rand() - 0.5) * baseR;
        const oy = cy + (rand() - 0.5) * baseR;
        const r = baseR * (0.5 + rand() * 0.5);
        ctx.beginPath();
        ctx.arc(ox, oy, r, 0, Math.PI * 2);
        ctx.fill();

        maskCtx.beginPath();
        maskCtx.arc(ox, oy, r, 0, Math.PI * 2);
        maskCtx.fill();
      }

      bbox = {
        x: Math.max(0, Math.floor(cx - baseR * 1.3)),
        y: Math.max(0, Math.floor(cy - baseR * 1.3)),
        w: Math.min(width, Math.ceil(baseR * 2.6)),
        h: Math.min(height, Math.ceil(baseR * 2.6)),
      };
      defectPixels = Math.floor(Math.PI * baseR * baseR * 1.6);

    } else if (defect === 'discoloration') {
      const cx = width * 0.4 + rand() * (width * 0.2);
      const cy = height * 0.4 + rand() * (height * 0.2);
      const rx = 35 + rand() * 25;
      const ry = 25 + rand() * 20;

      // Thermal oxidation tint (amber/violet gradient)
      const heatGrad = ctx.createRadialGradient(cx, cy, 5, cx, cy, rx);
      heatGrad.addColorStop(0, 'rgba(65, 80, 240, 0.48)'); // blue core
      heatGrad.addColorStop(0.5, 'rgba(180, 50, 180, 0.42)'); // purple ring
      heatGrad.addColorStop(1, 'rgba(230, 140, 30, 0)'); // straw amber feather

      ctx.fillStyle = heatGrad;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, rand() * Math.PI, 0, Math.PI * 2);
      ctx.fill();

      // Mask
      maskCtx.fillStyle = '#ffffff';
      maskCtx.beginPath();
      maskCtx.ellipse(cx, cy, rx * 0.85, ry * 0.85, 0, 0, Math.PI * 2);
      maskCtx.fill();

      bbox = {
        x: Math.max(0, Math.floor(cx - rx)),
        y: Math.max(0, Math.floor(cy - ry)),
        w: Math.ceil(rx * 2),
        h: Math.ceil(ry * 2),
      };
      defectPixels = Math.floor(Math.PI * rx * ry * 0.8);

    } else if (defect === 'dimensional_irregularity') {
      // Edge notch cutout or flash burr
      const isNotch = rand() > 0.5;
      const edge = 'top';
      const pw = 45 + rand() * 25;
      const ph = 25 + rand() * 15;
      const px = width * 0.5 - pw / 2;
      const py = 0;

      if (isNotch) {
        // Cutout: dark conveyor background showing through
        ctx.fillStyle = '#1c1e21';
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + pw / 2, ph);
        ctx.lineTo(px + pw, py);
        ctx.closePath();
        ctx.fill();
      } else {
        // Burr flash: protruding extra metal
        ctx.fillStyle = '#cbd5e1';
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + pw / 2, py + ph);
        ctx.lineTo(px + pw, py);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#475569';
        ctx.stroke();
      }

      maskCtx.fillStyle = '#ffffff';
      maskCtx.beginPath();
      maskCtx.moveTo(px, py);
      maskCtx.lineTo(px + pw / 2, ph);
      maskCtx.lineTo(px + pw, py);
      maskCtx.closePath();
      maskCtx.fill();

      bbox = {
        x: Math.floor(px),
        y: 0,
        w: Math.ceil(pw),
        h: Math.ceil(ph),
      };
      defectPixels = Math.floor((pw * ph) / 2);
    }
  }

  // 3. Apply Lighting Condition Perturbation
  if (lighting === 'angle_glare') {
    const glareGrad = ctx.createLinearGradient(0, 0, width, height);
    glareGrad.addColorStop(0, 'rgba(255, 255, 255, 0.45)');
    glareGrad.addColorStop(0.4, 'rgba(255, 255, 255, 0.08)');
    glareGrad.addColorStop(1, 'rgba(10, 15, 20, 0.25)');
    ctx.fillStyle = glareGrad;
    ctx.fillRect(0, 0, width, height);
  } else if (lighting === 'low_light') {
    ctx.fillStyle = 'rgba(0, 5, 15, 0.42)';
    ctx.fillRect(0, 0, width, height);
  } else if (lighting === 'spotlight_vignette') {
    const spot = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, width * 0.7);
    spot.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
    spot.addColorStop(0.7, 'rgba(0, 0, 0, 0.15)');
    spot.addColorStop(1, 'rgba(0, 0, 0, 0.65)');
    ctx.fillStyle = spot;
    ctx.fillRect(0, 0, width, height);
  }

  // 4. Generate Colormapped Anomaly Heatmap (Inferno / Thermal simulation)
  heatCtx.clearRect(0, 0, width, height);
  heatCtx.fillStyle = '#05021a'; // Deep indigo background
  heatCtx.fillRect(0, 0, width, height);

  if (defect !== 'normal' && defectPixels > 0) {
    const rawMaskData = maskCtx.getImageData(0, 0, width, height);
    const heatData = heatCtx.createImageData(width, height);
    const pixels = rawMaskData.data;

    for (let i = 0; i < pixels.length; i += 4) {
      const isDefect = pixels[i] > 120;
      if (isDefect) {
        // Inferno color palette for anomaly density (Yellow -> Orange -> Red)
        heatData.data[i] = 252;     // R
        heatData.data[i + 1] = 160; // G
        heatData.data[i + 2] = 44;  // B
        heatData.data[i + 3] = 240; // A
      } else {
        // Ambient background noise
        heatData.data[i] = 10;
        heatData.data[i + 1] = 8;
        heatData.data[i + 2] = 35;
        heatData.data[i + 3] = 255;
      }
    }
    heatCtx.putImageData(heatData, 0, 0);
  }

  ctx.restore();

  const totalArea = width * height;
  const coveragePct = Number(((defectPixels / totalArea) * 100).toFixed(2));
  const isDefective = defect !== 'normal';

  let severity: 'None' | 'Minor' | 'Moderate' | 'Critical' = 'None';
  if (isDefective) {
    if (coveragePct < 2.0) severity = 'Minor';
    else if (coveragePct < 6.0) severity = 'Moderate';
    else severity = 'Critical';
  }

  const confidence = isDefective ? Number((0.92 + rand() * 0.07).toFixed(3)) : Number((0.97 + rand() * 0.02).toFixed(3));

  // Compute realistic 7-class softmax distribution
  const defectTypes: DefectClass[] = ['normal', 'crack', 'scratch', 'dent', 'stain', 'discoloration', 'dimensional_irregularity'];
  const remainingProb = Math.max(0.001, 1.0 - confidence);
  const rawWeights: Record<DefectClass, number> = {} as any;
  let weightSum = 0;

  for (const dt of defectTypes) {
    if (dt === defect) {
      rawWeights[dt] = 0;
    } else {
      const w = 0.01 + rand() * 0.05;
      rawWeights[dt] = w;
      weightSum += w;
    }
  }

  const probabilities: Record<DefectClass, number> = {} as any;
  for (const dt of defectTypes) {
    if (dt === defect) {
      probabilities[dt] = confidence;
    } else {
      probabilities[dt] = Number(((rawWeights[dt] / weightSum) * remainingProb).toFixed(4));
    }
  }

  const pixelToMm = 0.1; // 0.1 mm / px calibration standard
  const regions: DefectRegionDetail[] = [];
  let totalAreaMm2 = 0;

  if (isDefective && defectPixels > 0 && bbox.w > 0 && bbox.h > 0) {
    const areaMm2 = Number((defectPixels * (pixelToMm * pixelToMm)).toFixed(2));
    totalAreaMm2 = areaMm2;
    const widthMm = Number((bbox.w * pixelToMm).toFixed(2));
    const heightMm = Number((bbox.h * pixelToMm).toFixed(2));
    const aspectRatio = Number((bbox.w / Math.max(bbox.h, 1)).toFixed(2));
    const extent = Number((defectPixels / Math.max(bbox.w * bbox.h, 1)).toFixed(2));

    regions.push({
      id: 1,
      bbox,
      areaPx: defectPixels,
      areaMm2,
      widthMm,
      heightMm,
      centroid: { x: Math.round(bbox.x + bbox.w / 2), y: Math.round(bbox.y + bbox.h / 2) },
      aspectRatio,
      extent: Math.min(extent, 1.0)
    });
  }

  return {
    isDefective,
    predictedClass: defect,
    confidence,
    severity,
    defectPixelCount: defectPixels,
    coveragePct,
    bbox,
    processingTimeMs: Number((11.4 + rand() * 4.2).toFixed(1)),
    probabilities,
    regions,
    pixelToMm,
    totalAreaMm2
  };
}
