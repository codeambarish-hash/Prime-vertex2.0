import React, { useState, useRef, useEffect } from 'react';
import {
  Sliders,
  Sparkles,
  Layers,
  ArrowRight,
  SunMedium,
  CheckCircle2,
  Maximize2,
  RefreshCw,
  RotateCw,
  FlipHorizontal,
  FlipVertical,
  Activity,
  Zap,
  Info
} from 'lucide-react';
import {
  simulateClahe,
  simulateBilateralDenoising,
  simulateBackgroundNormalization,
  simulateLetterboxResize,
  computeLuminanceHistogram
} from '../utils/preprocessingEngine';

interface SamplePreset {
  id: string;
  name: string;
  substrate: string;
  defect: string;
  defectDesc: string;
  lightingChallenge: string;
  drawRaw: (ctx: CanvasRenderingContext2D, width: number, height: number) => void;
}

export const PreprocessingTab: React.FC = () => {
  // Config state
  const [selectedSample, setSelectedSample] = useState<string>('brushed_crack');
  const [applyBgNorm, setApplyBgNorm] = useState<boolean>(true);
  const [applyClahe, setApplyClahe] = useState<boolean>(true);
  const [claheClipLimit, setClaheClipLimit] = useState<number>(2.5);
  const [applyBilateral, setApplyBilateral] = useState<boolean>(true);
  const [bilateralSigmaColor, setBilateralSigmaColor] = useState<number>(45);
  const [rotAngle, setRotAngle] = useState<number>(0);
  const [flipH, setFlipH] = useState<boolean>(false);
  const [flipV, setFlipV] = useState<boolean>(false);
  const [activeStageView, setActiveStageView] = useState<'stages' | 'split'>('stages');
  const [splitSliderPos, setSplitSliderPos] = useState<number>(50);

  // Canvases for intermediate stages
  const rawCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const claheCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bilateralCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const finalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const splitCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Histogram states
  const [histBefore, setHistBefore] = useState<number[]>([]);
  const [histAfter, setHistAfter] = useState<number[]>([]);
  const [metaInfo, setMetaInfo] = useState<{
    origW: number;
    origH: number;
    scale: number;
    padTop: number;
    padLeft: number;
  }>({ origW: 320, origH: 240, scale: 0.8, padTop: 28, padLeft: 0 });

  // Manufacturing presets
  const SAMPLES: SamplePreset[] = [
    {
      id: 'brushed_crack',
      name: 'Brushed Aluminum Part',
      substrate: 'Brushed Metal',
      defect: 'Hairline Fracture / Crack',
      defectDesc: 'Delicate fracture traversing grain with shadow furrow',
      lightingChallenge: 'Harsh Directional Spotlight Glare + Conveyor Edge',
      drawRaw: (ctx, w, h) => {
        // 1. Dark conveyor belt background border
        ctx.fillStyle = '#1c1e24';
        ctx.fillRect(0, 0, w, h);

        // 2. Manufactured metallic part in center
        const padX = 24;
        const padY = 20;
        const pw = w - padX * 2;
        const ph = h - padY * 2;

        ctx.fillStyle = '#b0b5bd';
        ctx.fillRect(padX, padY, pw, ph);

        // Brushed streaks
        ctx.lineWidth = 1.2;
        for (let i = 0; i < 350; i++) {
          const y = padY + Math.random() * ph;
          const len = 30 + Math.random() * (pw * 0.7);
          const x = padX + Math.random() * (pw - len);
          const alpha = 0.05 + Math.random() * 0.12;
          ctx.strokeStyle = Math.random() > 0.5 ? `rgba(255,255,255,${alpha})` : `rgba(30,35,40,${alpha})`;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + len, y);
          ctx.stroke();
        }

        // Spotlight glare hotspot (high contrast challenge)
        const glare = ctx.createRadialGradient(padX + pw * 0.3, padY + ph * 0.35, 10, padX + pw * 0.3, padY + ph * 0.35, 85);
        glare.addColorStop(0, 'rgba(255, 255, 255, 0.75)');
        glare.addColorStop(0.5, 'rgba(255, 255, 255, 0.3)');
        glare.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = glare;
        ctx.fillRect(padX, padY, pw, ph);

        // Shadow falloff on opposite side
        const shadow = ctx.createLinearGradient(padX + pw * 0.5, padY, padX + pw, padY + ph);
        shadow.addColorStop(0, 'rgba(0, 0, 0, 0)');
        shadow.addColorStop(1, 'rgba(0, 0, 0, 0.45)');
        ctx.fillStyle = shadow;
        ctx.fillRect(padX, padY, pw, ph);

        // Hairline Crack defect
        ctx.save();
        ctx.strokeStyle = '#1a1c20';
        ctx.lineWidth = 1.8;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'bevel';
        ctx.beginPath();
        const startX = padX + pw * 0.45;
        const startY = padY + ph * 0.25;
        ctx.moveTo(startX, startY);
        ctx.lineTo(startX + 14, startY + 22);
        ctx.lineTo(startX + 9, startY + 45);
        ctx.lineTo(startX + 28, startY + 70);
        ctx.lineTo(startX + 22, startY + 95);
        ctx.stroke();

        // White specular rim of the crack
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 0.9;
        ctx.beginPath();
        ctx.moveTo(startX + 1, startY + 1);
        ctx.lineTo(startX + 15, startY + 23);
        ctx.lineTo(startX + 10, startY + 46);
        ctx.lineTo(startX + 29, startY + 71);
        ctx.lineTo(startX + 23, startY + 96);
        ctx.stroke();
        ctx.restore();
      }
    },
    {
      id: 'machined_scratch',
      name: 'Lathe Machined Disc',
      substrate: 'Machined Steel',
      defect: 'Deep Linear Scratch',
      defectDesc: 'Abrasive tool mark gouging concentric toolpaths',
      lightingChallenge: 'Low-Light Underexposure + Off-Center Shadow',
      drawRaw: (ctx, w, h) => {
        ctx.fillStyle = '#16181d';
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2;
        const radius = Math.min(w, h) * 0.42;

        // Dark underexposed base
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = '#555b63';
        ctx.fill();

        // Concentric lathe grooves
        for (let r = radius - 4; r > 10; r -= 6) {
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.strokeStyle = r % 12 === 0 ? 'rgba(255,255,255,0.14)' : 'rgba(15,18,22,0.25)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Deep Scratch cutting diagonally across the disc
        ctx.beginPath();
        ctx.moveTo(cx - radius * 0.65, cy - radius * 0.35);
        ctx.lineTo(cx + radius * 0.55, cy + radius * 0.45);
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 2.2;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(cx - radius * 0.65 + 1.5, cy - radius * 0.35 + 1.5);
        ctx.lineTo(cx + radius * 0.55 + 1.5, cy + radius * 0.45 + 1.5);
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.restore();
      }
    },
    {
      id: 'ceramic_stain',
      name: 'Glazed Ceramic Substrate',
      substrate: 'White Ceramic',
      defect: 'Contaminant Fluid Stain',
      defectDesc: 'Multi-octave chemical oil residue with translucent perimeter',
      lightingChallenge: 'Sensor Grain Noise + High Specular Flatness',
      drawRaw: (ctx, w, h) => {
        ctx.fillStyle = '#1f242c';
        ctx.fillRect(0, 0, w, h);

        const padX = 22;
        const padY = 22;
        ctx.fillStyle = '#e8ebed';
        ctx.fillRect(padX, padY, w - padX * 2, h - padY * 2);

        // High frequency sensor noise simulation
        const imgData = ctx.getImageData(padX, padY, w - padX * 2, h - padY * 2);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
          const noise = (Math.random() - 0.5) * 35;
          data[i] = Math.min(255, Math.max(0, data[i] + noise));
          data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
          data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
        }
        ctx.putImageData(imgData, padX, padY);

        // Fluid Stain Blob
        const sx = w * 0.52;
        const sy = h * 0.48;
        const stainGrad = ctx.createRadialGradient(sx, sy, 5, sx, sy, 35);
        stainGrad.addColorStop(0, 'rgba(100, 75, 45, 0.75)');
        stainGrad.addColorStop(0.7, 'rgba(140, 110, 70, 0.45)');
        stainGrad.addColorStop(1, 'rgba(180, 150, 100, 0)');

        ctx.save();
        ctx.fillStyle = stainGrad;
        ctx.beginPath();
        ctx.ellipse(sx, sy, 38, 22, Math.PI / 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  ];

  // Pipeline execution effect
  useEffect(() => {
    const rawC = rawCanvasRef.current;
    if (!rawC) return;

    const rawCtx = rawC.getContext('2d', { willReadFrequently: true });
    if (!rawCtx) return;

    const curSample = SAMPLES.find((s) => s.id === selectedSample) || SAMPLES[0];

    // Clear and draw raw sensor canvas
    rawC.width = 320;
    rawC.height = 240;
    curSample.drawRaw(rawCtx, 320, 240);

    // Apply orientation transforms (rot, flip) if active
    if (rotAngle !== 0 || flipH || flipV) {
      const tempC = document.createElement('canvas');
      tempC.width = 320;
      tempC.height = 240;
      const tctx = tempC.getContext('2d');
      if (tctx) {
        tctx.drawImage(rawC, 0, 0);
        rawCtx.clearRect(0, 0, 320, 240);
        rawCtx.save();
        rawCtx.translate(160, 120);
        if (rotAngle !== 0) rawCtx.rotate((rotAngle * Math.PI) / 180);
        if (flipH) rawCtx.scale(-1, 1);
        if (flipV) rawCtx.scale(1, -1);
        rawCtx.drawImage(tempC, -160, -120);
        rawCtx.restore();
      }
    }

    const rawData = rawCtx.getImageData(0, 0, 320, 240);
    const histRaw = computeLuminanceHistogram(rawData);
    setHistBefore(histRaw);

    // Stage 1: Background Normalization (Conveyor suppression)
    let currentData = rawData;
    if (applyBgNorm) {
      currentData = simulateBackgroundNormalization(currentData, [15, 17, 23]);
    }
    const bgC = bgCanvasRef.current;
    if (bgC) {
      bgC.width = 320;
      bgC.height = 240;
      const bgCtx = bgC.getContext('2d');
      bgCtx?.putImageData(currentData, 0, 0);
    }

    // Stage 2: CLAHE on L-channel in LAB space
    if (applyClahe) {
      currentData = simulateClahe(currentData, claheClipLimit, 8);
    }
    const claheC = claheCanvasRef.current;
    if (claheC) {
      claheC.width = 320;
      claheC.height = 240;
      const claheCtx = claheC.getContext('2d');
      claheCtx?.putImageData(currentData, 0, 0);
    }
    const histClahe = computeLuminanceHistogram(currentData);
    setHistAfter(histClahe);

    // Stage 3: Bilateral Edge-Preserving Denoising
    if (applyBilateral) {
      currentData = simulateBilateralDenoising(currentData, 3, bilateralSigmaColor, 30.0);
    }
    const bilC = bilateralCanvasRef.current;
    if (bilC) {
      bilC.width = 320;
      bilC.height = 240;
      const bilCtx = bilC.getContext('2d');
      bilCtx?.putImageData(currentData, 0, 0);
    }

    // Stage 4: Aspect-Ratio Preserved Letterbox Resizing (256x256)
    const letterboxResult = simulateLetterboxResize(currentData, [256, 256], [12, 14, 18]);
    const finalC = finalCanvasRef.current;
    if (finalC) {
      finalC.width = 256;
      finalC.height = 256;
      const finalCtx = finalC.getContext('2d');
      finalCtx?.putImageData(letterboxResult.imageData, 0, 0);
    }

    setMetaInfo({
      origW: letterboxResult.meta.origSize[0],
      origH: letterboxResult.meta.origSize[1],
      scale: letterboxResult.meta.scale,
      padTop: letterboxResult.meta.padTop,
      padLeft: letterboxResult.meta.padLeft,
    });

    // Render Split-Screen Comparison Canvas
    const splitC = splitCanvasRef.current;
    if (splitC) {
      splitC.width = 320;
      splitC.height = 240;
      const splitCtx = splitC.getContext('2d');
      if (splitCtx) {
        // Draw raw
        splitCtx.putImageData(rawData, 0, 0);

        // Draw processed over top with clip
        const splitX = (splitSliderPos / 100) * 320;
        splitCtx.save();
        splitCtx.beginPath();
        splitCtx.rect(splitX, 0, 320 - splitX, 240);
        splitCtx.clip();
        splitCtx.putImageData(currentData, 0, 0);
        splitCtx.restore();

        // Draw dividing vertical line
        splitCtx.strokeStyle = '#38bdf8';
        splitCtx.lineWidth = 2;
        splitCtx.beginPath();
        splitCtx.moveTo(splitX, 0);
        splitCtx.lineTo(splitX, 240);
        splitCtx.stroke();
      }
    }
  }, [
    selectedSample,
    applyBgNorm,
    applyClahe,
    claheClipLimit,
    applyBilateral,
    bilateralSigmaColor,
    rotAngle,
    flipH,
    flipV,
    splitSliderPos
  ]);

  const activeSampleObj = SAMPLES.find((s) => s.id === selectedSample) || SAMPLES[0];

  return (
    <div className="space-y-6">
      {/* Top Banner / Concept Card */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-indigo-50 text-indigo-700 font-bold text-xs flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                src/preprocessing.py
              </span>
              <span className="text-xs text-slate-500 font-mono">Inspectra AI Pipeline</span>
            </div>
            <h2 className="text-lg font-black text-slate-900 mt-1">
              Industrial Image Preprocessing & Lighting Normalization
            </h2>
            <p className="text-xs text-slate-500 max-w-3xl leading-relaxed">
              Standardizes raw factory camera streams before deep learning inference. Solves the 4 primary
              industrial vision bottlenecks: aspect-ratio distortion, extreme illumination gradients,
              high-frequency sensor grain, and conveyor background confusion.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveStageView('stages')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                activeStageView === 'stages'
                  ? 'bg-indigo-600 text-white shadow-2xs'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              Stage-by-Stage Flow
            </button>
            <button
              onClick={() => setActiveStageView('split')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5 ${
                activeStageView === 'split'
                  ? 'bg-indigo-600 text-white shadow-2xs'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              <Maximize2 className="w-3.5 h-3.5" />
              Split Comparison
            </button>
          </div>
        </div>

        {/* Preset Sample Selector */}
        <div>
          <label className="text-xs font-bold text-slate-700 block mb-2">
            Select Inspection Sample Substrate & Defect Challenge:
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                onClick={() => setSelectedSample(s.id)}
                className={`text-left p-3 rounded-lg border transition-all ${
                  selectedSample === s.id
                    ? 'border-indigo-500 bg-indigo-50/50 shadow-2xs'
                    : 'border-slate-200 hover:border-slate-300 bg-slate-50/40'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900">{s.name}</span>
                  {selectedSample === s.id && (
                    <span className="w-2 h-2 rounded-full bg-indigo-600" />
                  )}
                </div>
                <div className="text-[11px] font-semibold text-rose-600 mt-0.5">
                  Defect: {s.defect}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">
                  Challenge: {s.lightingChallenge}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Interactive Stage Grid or Split-Screen View */}
      {activeStageView === 'stages' ? (
        <div className="space-y-4">
          {/* 5-Step Pipeline Grid */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {/* Step 1: Raw Sensor */}
            <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-2xs space-y-2 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700">1. Raw Sensor Input</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                  320x240
                </span>
              </div>
              <div className="relative aspect-4/3 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                <canvas ref={rawCanvasRef} className="w-full h-full object-contain" />
              </div>
              <div className="text-[11px] text-slate-500 leading-tight">
                <strong className="text-slate-700">Input Artifacts:</strong> Glare hotspots, underexposure, and conveyor belt edge.
              </div>
            </div>

            {/* Step 2: Background Normalized */}
            <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-2xs space-y-2 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700">2. Background Norm.</span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${applyBgNorm ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-500'}`}>
                  {applyBgNorm ? 'Active' : 'Bypassed'}
                </span>
              </div>
              <div className="relative aspect-4/3 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                <canvas ref={bgCanvasRef} className="w-full h-full object-contain" />
              </div>
              <div className="text-[11px] text-slate-500 leading-tight">
                <strong className="text-slate-700">Conveyor Mask:</strong> Isolates workpiece footprint and zeroes out fixture background.
              </div>
            </div>

            {/* Step 3: LAB CLAHE */}
            <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-2xs space-y-2 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700">3. LAB Space CLAHE</span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${applyClahe ? 'bg-indigo-100 text-indigo-800' : 'bg-slate-100 text-slate-500'}`}>
                  Clip {claheClipLimit.toFixed(1)}
                </span>
              </div>
              <div className="relative aspect-4/3 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                <canvas ref={claheCanvasRef} className="w-full h-full object-contain" />
              </div>
              <div className="text-[11px] text-slate-500 leading-tight">
                <strong className="text-slate-700">Luminance Balance:</strong> Flattens glare and reveals shadow crack detail without color drift.
              </div>
            </div>

            {/* Step 4: Bilateral Denoising */}
            <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-2xs space-y-2 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700">4. Bilateral Denoised</span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${applyBilateral ? 'bg-purple-100 text-purple-800' : 'bg-slate-100 text-slate-500'}`}>
                  σ_c={bilateralSigmaColor}
                </span>
              </div>
              <div className="relative aspect-4/3 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                <canvas ref={bilateralCanvasRef} className="w-full h-full object-contain" />
              </div>
              <div className="text-[11px] text-slate-500 leading-tight">
                <strong className="text-slate-700">Edge-Preserving:</strong> Smooths substrate noise while keeping fracture margins razor-sharp.
              </div>
            </div>

            {/* Step 5: Letterbox (256x256) */}
            <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-2xs space-y-2 flex flex-col">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-700">5. Final Letterboxed</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold">
                  256x256
                </span>
              </div>
              <div className="relative aspect-4/3 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                <canvas ref={finalCanvasRef} className="w-full h-full object-contain" />
              </div>
              <div className="text-[11px] text-slate-500 leading-tight">
                <strong className="text-slate-700">Standardized Input:</strong> Aspect preserved with symmetric padding (Top {metaInfo.padTop}px).
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Split Screen Comparison Mode */
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-900">Before & After Split View</span>
              <span className="text-xs text-slate-500">
                (Left: Raw Sensor | Right: Preprocessed Pipeline)
              </span>
            </div>
            <span className="text-xs font-mono font-bold text-indigo-600">
              Split: {splitSliderPos}%
            </span>
          </div>

          <div className="relative max-w-xl mx-auto aspect-4/3 bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-inner">
            <canvas ref={splitCanvasRef} className="w-full h-full object-contain" />
            <div
              className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-900/80 text-white text-[10px] font-mono font-bold backdrop-blur-xs"
            >
              RAW SENSOR
            </div>
            <div
              className="absolute top-2 right-2 px-2 py-0.5 rounded bg-indigo-900/80 text-indigo-200 text-[10px] font-mono font-bold backdrop-blur-xs"
            >
              PREPROCESSED
            </div>
          </div>

          <div className="max-w-xl mx-auto flex items-center gap-3">
            <span className="text-xs text-slate-500 font-medium shrink-0">Raw</span>
            <input
              type="range"
              min="0"
              max="100"
              value={splitSliderPos}
              onChange={(e) => setSplitSliderPos(Number(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <span className="text-xs text-slate-500 font-medium shrink-0">Preprocessed</span>
          </div>
        </div>
      )}

      {/* Interactive Parameter Controls & Live Luminance Histogram */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Preprocessing Tuning Controls (7 cols) */}
        <div className="lg:col-span-7 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Pipeline Tuning Controls
              </h3>
            </div>
            <span className="text-[11px] text-slate-400">Live Simulation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Background Normalization Toggle */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800">
                  Background Normalization
                </label>
                <input
                  type="checkbox"
                  checked={applyBgNorm}
                  onChange={(e) => setApplyBgNorm(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500 w-4 h-4"
                />
              </div>
              <p className="text-[11px] text-slate-500">
                Detects workpiece contours and standardizes conveyor rubber/metal margins.
              </p>
            </div>

            {/* CLAHE Lighting Normalization */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800">
                  LAB Space CLAHE
                </label>
                <input
                  type="checkbox"
                  checked={applyClahe}
                  onChange={(e) => setApplyClahe(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500 w-4 h-4"
                />
              </div>
              <div>
                <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                  <span>Clip Limit:</span>
                  <span className="font-mono font-bold text-slate-800">{claheClipLimit.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="5.0"
                  step="0.5"
                  disabled={!applyClahe}
                  value={claheClipLimit}
                  onChange={(e) => setClaheClipLimit(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer disabled:opacity-50"
                />
              </div>
            </div>

            {/* Bilateral Denoising */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-slate-800">
                  Bilateral Denoising
                </label>
                <input
                  type="checkbox"
                  checked={applyBilateral}
                  onChange={(e) => setApplyBilateral(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500 w-4 h-4"
                />
              </div>
              <div>
                <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                  <span>Sigma Color (Edge sensitivity):</span>
                  <span className="font-mono font-bold text-slate-800">{bilateralSigmaColor}</span>
                </div>
                <input
                  type="range"
                  min="15"
                  max="80"
                  step="5"
                  disabled={!applyBilateral}
                  value={bilateralSigmaColor}
                  onChange={(e) => setBilateralSigmaColor(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer disabled:opacity-50"
                />
              </div>
            </div>

            {/* Orientation Invariance (Albumentations simulation) */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <label className="text-xs font-bold text-slate-800 block">
                Orientation Invariance (Albumentations)
              </label>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setRotAngle((prev) => (prev + 90) % 360)}
                  className="flex-1 py-1.5 bg-white border border-slate-300 rounded text-xs font-medium text-slate-700 hover:bg-slate-100 flex items-center justify-center gap-1"
                >
                  <RotateCw className="w-3 h-3 text-indigo-600" />
                  Rotate {rotAngle}°
                </button>
                <button
                  onClick={() => setFlipH(!flipH)}
                  className={`p-1.5 border rounded ${flipH ? 'bg-indigo-100 border-indigo-300 text-indigo-800' : 'bg-white border-slate-300 text-slate-700'}`}
                  title="Horizontal Flip"
                >
                  <FlipHorizontal className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setFlipV(!flipV)}
                  className={`p-1.5 border rounded ${flipV ? 'bg-indigo-100 border-indigo-300 text-indigo-800' : 'bg-white border-slate-300 text-slate-700'}`}
                  title="Vertical Flip"
                >
                  <FlipVertical className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          <div className="pt-2 text-xs text-slate-500 flex items-center justify-between border-t border-slate-100">
            <span>Aspect-Ratio Letterbox Target: <strong className="text-slate-800">256 x 256 px</strong></span>
            <span>Scaling factor: <strong className="text-indigo-600 font-mono">{metaInfo.scale.toFixed(3)}</strong></span>
          </div>
        </div>

        {/* Right: Luminance Histogram Comparison (5 cols) */}
        <div className="lg:col-span-5 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Luminance Histogram Comparison
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                CLAHE Verification
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Adaptive histogram equalization distributes pixel intensities across the full 0–255 range, eliminating sensor saturation.
            </p>

            {/* Side-by-side micro histograms */}
            <div className="mt-4 space-y-3">
              {/* Before Histogram */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="font-semibold text-slate-600">Before CLAHE (Raw Sensor)</span>
                  <span className="text-slate-400">Compressed Dynamic Range</span>
                </div>
                <div className="h-14 bg-slate-950 rounded-lg p-1.5 flex items-end gap-[1px] border border-slate-800">
                  {histBefore.length > 0 &&
                    Array.from({ length: 32 }).map((_, i) => {
                      const chunk = histBefore.slice(i * 8, (i + 1) * 8);
                      const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
                      const maxVal = Math.max(...histBefore, 1);
                      const heightPct = Math.min(100, Math.max(4, (avg / maxVal) * 350));
                      return (
                        <div
                          key={i}
                          className="flex-1 bg-amber-400/80 rounded-t-xs"
                          style={{ height: `${heightPct}%` }}
                          title={`Bin ${i * 8}: ${Math.round(avg)} pixels`}
                        />
                      );
                    })}
                </div>
              </div>

              {/* After Histogram */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="font-semibold text-indigo-600">After CLAHE (L-Channel LAB)</span>
                  <span className="text-emerald-600 font-semibold">Expanded & Balanced</span>
                </div>
                <div className="h-14 bg-slate-950 rounded-lg p-1.5 flex items-end gap-[1px] border border-slate-800">
                  {histAfter.length > 0 &&
                    Array.from({ length: 32 }).map((_, i) => {
                      const chunk = histAfter.slice(i * 8, (i + 1) * 8);
                      const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
                      const maxVal = Math.max(...histAfter, 1);
                      const heightPct = Math.min(100, Math.max(4, (avg / maxVal) * 260));
                      return (
                        <div
                          key={i}
                          className="flex-1 bg-indigo-400 rounded-t-xs"
                          style={{ height: `${heightPct}%` }}
                          title={`Bin ${i * 8}: ${Math.round(avg)} pixels`}
                        />
                      );
                    })}
                </div>
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-600 space-y-1 mt-3">
            <div className="flex items-center gap-1.5 font-bold text-slate-800">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              <span>Inspection Advantage:</span>
            </div>
            <p className="text-[11px] leading-relaxed text-slate-500">
              By preserving the chroma channels (A and B) while modifying only the L-channel,
              thermal discoloration and copper oxidation defects are never falsified.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
