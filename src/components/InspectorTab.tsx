import React, { useState, useEffect, useRef } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  RefreshCw, 
  SunMedium, 
  RotateCw, 
  Maximize2, 
  Gauge, 
  Eye, 
  Cpu, 
  Crosshair,
  Sliders,
  Layers,
  BarChart2,
  X,
  Sparkles,
  Check,
  Ruler,
  Target,
  Scan,
  ShieldCheck,
  Filter,
  AlertOctagon,
  History
} from 'lucide-react';
import { DefectClass, SubstrateType, LightingCondition, InspectionResult } from '../types/inspection';
import { renderSimulation } from '../utils/defectRenderer';
import { SampleAssetsGallery } from './SampleAssetsGallery';

const DEFECT_OPTIONS: { id: DefectClass; label: string; desc: string; badgeColor: string }[] = [
  { id: 'normal', label: 'Normal (Pass)', desc: 'Pristine surface with standard manufacturing grain', badgeColor: 'bg-emerald-100 text-emerald-800' },
  { id: 'crack', label: 'Crack', desc: 'Fracture line with shadow depth and specular edge', badgeColor: 'bg-red-100 text-red-800' },
  { id: 'scratch', label: 'Scratch', desc: 'Linear mechanical abrasion with bright furrow', badgeColor: 'bg-orange-100 text-orange-800' },
  { id: 'dent', label: 'Dent', desc: 'Radial 3D impact depression with shading gradient', badgeColor: 'bg-amber-100 text-amber-800' },
  { id: 'stain', label: 'Stain', desc: 'Chemical or lubricant fluid diffusion blob', badgeColor: 'bg-purple-100 text-purple-800' },
  { id: 'discoloration', label: 'Discoloration', desc: 'Thermal oxidation or chemical hue shift', badgeColor: 'bg-blue-100 text-blue-800' },
  { id: 'dimensional_irregularity', label: 'Dim. Irregularity', desc: 'Edge notch or material burr tolerance defect', badgeColor: 'bg-pink-100 text-pink-800' },
];

const SUBSTRATES: { id: SubstrateType; label: string }[] = [
  { id: 'brushed_metal', label: 'Brushed Aluminum' },
  { id: 'cast_iron', label: 'Cast Iron Porous' },
  { id: 'machined_part', label: 'Lathe Machined Steel' },
  { id: 'ceramic_tile', label: 'Ceramic Substrate' },
  { id: 'carbon_composite', label: 'Carbon Composite' },
];

const LIGHTING_OPTIONS: { id: LightingCondition; label: string }[] = [
  { id: 'uniform', label: 'Uniform Inspection Light' },
  { id: 'angle_glare', label: 'Directional Glare Angle' },
  { id: 'low_light', label: 'Low Ambient Illumination' },
  { id: 'spotlight_vignette', label: 'Spotlight / Vignette' },
];

export const InspectorTab: React.FC = () => {
  const [substrate, setSubstrate] = useState<SubstrateType>('brushed_metal');
  const [defect, setDefect] = useState<DefectClass>('crack');
  const [lighting, setLighting] = useState<LightingCondition>('uniform');
  const [rotation, setRotation] = useState<number>(0);
  const [seed, setSeed] = useState<number>(101);
  const [activeViewMode, setActiveViewMode] = useState<'overlay' | 'raw' | 'heatmap' | 'mask' | 'quad'>('overlay');
  const [anomalyModel, setAnomalyModel] = useState<'autoencoder' | 'transfer_learning'>('autoencoder');
  const [classifierBackbone, setClassifierBackbone] = useState<'efficientnet_b0' | 'mobilenet_v2'>('efficientnet_b0');
  const [showConfusionMatrixModal, setShowConfusionMatrixModal] = useState<boolean>(false);
  const [localizationMethod, setLocalizationMethod] = useState<'gradcam' | 'unet' | 'hybrid'>('gradcam');
  const [pixelToMm, setPixelToMm] = useState<number>(0.1);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [showContours, setShowContours] = useState<boolean>(true);
  const [showHudLabels, setShowHudLabels] = useState<boolean>(true);
  const [showLocalizationModal, setShowLocalizationModal] = useState<boolean>(false);

  // Stage 5 Refinement States (False-Positive & Missed-Detection Reduction)
  const [enableDualFiltering, setEnableDualFiltering] = useState<boolean>(true);
  const [anomalyThreshold, setAnomalyThreshold] = useState<number>(0.50);
  const [classifierThreshold, setClassifierThreshold] = useState<number>(0.65);
  const [enableMorphFilter, setEnableMorphFilter] = useState<boolean>(true);
  const [minAreaThreshold, setMinAreaThreshold] = useState<number>(25);
  const [enableTTA, setEnableTTA] = useState<boolean>(true);
  const [borderlineLow, setBorderlineLow] = useState<number>(0.45);
  const [borderlineHigh, setBorderlineHigh] = useState<number>(0.70);
  const [showRefinementModal, setShowRefinementModal] = useState<boolean>(false);
  const [manualReviewList, setManualReviewList] = useState<Array<{
    id: string;
    timestamp: string;
    class: string;
    confidence: number;
    anomalyScore: number;
    reason: string;
    status: 'PENDING' | 'RESOLVED_PASS' | 'RESOLVED_DEFECT';
  }>>([
    {
      id: 'AUDIT-8041',
      timestamp: '14:08:12',
      class: 'scratch',
      confidence: 0.52,
      anomalyScore: 0.61,
      reason: 'Borderline classifier confidence (52.0%) in [45%, 70%] review window',
      status: 'PENDING'
    },
    {
      id: 'AUDIT-8042',
      timestamp: '14:10:45',
      class: 'normal',
      confidence: 0.88,
      anomalyScore: 0.74,
      reason: 'Dual model disagreement: Anomaly detector flags defect (0.74), classifier says normal',
      status: 'PENDING'
    }
  ]);

  const [result, setResult] = useState<InspectionResult | null>(null);

  const rawCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Trigger simulation render
  useEffect(() => {
    if (rawCanvasRef.current && maskCanvasRef.current && heatmapCanvasRef.current) {
      const res = renderSimulation(
        rawCanvasRef.current,
        maskCanvasRef.current,
        heatmapCanvasRef.current,
        substrate,
        defect,
        lighting,
        rotation,
        seed
      );
      setResult(res);
    }
  }, [substrate, defect, lighting, rotation, seed]);

  const handleRandomize = () => {
    setSeed(Math.floor(Math.random() * 100000));
  };

  // Derived Refinement & False-Positive Reduction Logic
  const rawDefectAreaPx = result?.defectPixelCount ?? 0;
  const rawConfidence = result?.confidence ?? 0;
  // Anomaly score: estimated from coverage / intensity or simulation
  const simulatedAnomalyScore = result?.isDefective 
    ? Math.min(0.96, Math.max(0.48, ((result?.coveragePct ?? 0) * 0.15) + (rawConfidence * 0.55))) 
    : 0.18;

  // 1. Confidence-based filtering rule: defect if BOTH anomaly detector AND classifier agree above thresholds
  const anomalyFlagsDefect = simulatedAnomalyScore >= anomalyThreshold;
  const classifierFlagsDefect = (result?.predictedClass ?? 'normal') !== 'normal' && rawConfidence >= classifierThreshold;
  const dualDefectAgreement = anomalyFlagsDefect && classifierFlagsDefect;
  const dualCleanAgreement = !anomalyFlagsDefect && !classifierFlagsDefect;
  const dualAgreed = (!enableDualFiltering) || (anomalyFlagsDefect === classifierFlagsDefect);

  // 2. Morphological filtering (opening/closing): purge tiny noise blobs under minAreaThreshold
  const isNoiseBlob = enableMorphFilter && rawDefectAreaPx > 0 && rawDefectAreaPx < minAreaThreshold;
  const cleanedDefectAreaPx = isNoiseBlob ? 0 : rawDefectAreaPx;
  const purgedBlobsCount = isNoiseBlob ? 1 : 0;
  const survivingBlobsCount = cleanedDefectAreaPx > 0 ? 1 : 0;

  // 3. Test-time augmentation (TTA) voting
  const ttaVoteRatio = !enableTTA ? 1.0 : (
    result?.isDefective && !isNoiseBlob ? (rawConfidence > 0.80 ? 1.0 : 0.83) : (isNoiseBlob ? 0.17 : 0.0)
  );
  const ttaConsensus = ttaVoteRatio >= 0.60;

  // 4. Borderline review criteria
  const isBorderlineConfidence = rawConfidence >= borderlineLow && rawConfidence < borderlineHigh;
  const requiresManualReview = (enableDualFiltering && !dualAgreed) || isBorderlineConfidence || (enableTTA && ttaVoteRatio > 0.20 && !ttaConsensus);

  let finalDecision: 'CONFIRMED_DEFECT' | 'CLEAN_PASS' | 'SUPPRESSED_FALSE_POSITIVE' | 'MANUAL_REVIEW' = 'CLEAN_PASS';
  if (requiresManualReview) {
    finalDecision = 'MANUAL_REVIEW';
  } else if (result?.isDefective && !isNoiseBlob && ttaConsensus && (!enableDualFiltering || dualDefectAgreement)) {
    finalDecision = 'CONFIRMED_DEFECT';
  } else if (isNoiseBlob || (enableDualFiltering && !classifierFlagsDefect && rawDefectAreaPx > 0)) {
    finalDecision = 'SUPPRESSED_FALSE_POSITIVE';
  } else {
    finalDecision = 'CLEAN_PASS';
  }

  const handleLoadIntoInspector = (newSubstrate: SubstrateType, newDefect: DefectClass) => {
    setSubstrate(newSubstrate);
    setDefect(newDefect);
  };

  return (
    <div className="space-y-6">
      {/* High-Resolution Synthetic Industrial Defect Asset Suite (1024x1024) */}
      <SampleAssetsGallery onLoadIntoInspector={handleLoadIntoInspector} />

      {/* Top Banner / Quick Metric Pill Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-medium text-slate-500 block">Raw State</span>
          <div className="flex items-center gap-2 mt-1">
            {result?.isDefective ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-rose-50 text-rose-700 border border-rose-200">
                <AlertTriangle className="w-3.5 h-3.5" /> RAW DEFECT
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5" /> RAW PASS
              </span>
            )}
          </div>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-medium text-slate-500 block">Refined Gate Decision</span>
          <div className="flex items-center gap-2 mt-1">
            {finalDecision === 'CONFIRMED_DEFECT' && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-rose-50 text-rose-700 border border-rose-200">
                <AlertTriangle className="w-3.5 h-3.5" /> REJECT
              </span>
            )}
            {finalDecision === 'CLEAN_PASS' && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5" /> CLEAN PASS
              </span>
            )}
            {finalDecision === 'SUPPRESSED_FALSE_POSITIVE' && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-amber-50 text-amber-800 border border-amber-200">
                <ShieldCheck className="w-3.5 h-3.5" /> NOISE PURGED
              </span>
            )}
            {finalDecision === 'MANUAL_REVIEW' && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-yellow-50 text-yellow-800 border border-yellow-300">
                <AlertOctagon className="w-3.5 h-3.5" /> AUDIT QUEUE
              </span>
            )}
          </div>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-medium text-slate-500 block">Classification Confidence</span>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className="text-lg font-bold text-slate-900">
              {result ? `${(result.confidence * 100).toFixed(1)}%` : '--'}
            </span>
            <span className="text-xs text-slate-500 uppercase">{result?.predictedClass}</span>
          </div>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-medium text-slate-500 block">Cleaned Defect Surface</span>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className="text-lg font-bold text-slate-900">
              {cleanedDefectAreaPx > 0 && result ? `${result.coveragePct}%` : '0.0%'}
            </span>
            <span className="text-xs text-slate-500">
              ({cleanedDefectAreaPx} px²)
            </span>
          </div>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-medium text-slate-500 block">TTA Consensus Vote</span>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className={`text-lg font-bold ${ttaConsensus ? 'text-emerald-600' : 'text-amber-600'}`}>
              {(ttaVoteRatio * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-slate-500">
              ({enableTTA ? '6 views' : 'Single view'})
            </span>
          </div>
        </div>
      </div>

      {/* Main Workbench Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Parameter & Defect Controls (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-600" />
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                  Test Simulation Rig
                </h2>
              </div>
              <button
                id="btn-randomize-seed"
                onClick={handleRandomize}
                className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors"
                title="Generate new random seed"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Re-Seed
              </button>
            </div>

            {/* Defect Category Selection */}
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-2">
                Defect Target Category
              </label>
              <div className="grid grid-cols-1 gap-1.5">
                {DEFECT_OPTIONS.map((item) => (
                  <button
                    key={item.id}
                    id={`defect-opt-${item.id}`}
                    onClick={() => setDefect(item.id)}
                    className={`flex items-center justify-between px-3 py-2 text-left rounded-lg text-xs font-medium border transition-all ${
                      defect === item.id
                        ? 'border-emerald-600 bg-emerald-50/50 text-emerald-950 font-bold shadow-2xs'
                        : 'border-slate-200 text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <span>{item.label}</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${item.badgeColor}`}>
                      {item.id === 'normal' ? '0' : 'GT Mask'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Substrate Selection */}
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-1.5">
                Manufacturing Substrate
              </label>
              <select
                id="select-substrate"
                value={substrate}
                onChange={(e) => setSubstrate(e.target.value as SubstrateType)}
                className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {SUBSTRATES.map((s) => (
                  <option key={s.id} value={s.id}>{s.label}</option>
                ))}
              </select>
            </div>

            {/* Anomaly Detection Model Architecture Selector */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                  <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Detection Model</span>
                </div>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-800 font-semibold">
                  src/anomaly_detection.py
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() => setAnomalyModel('autoencoder')}
                  className={`px-2 py-1.5 rounded text-[11px] font-semibold border text-center transition-all ${
                    anomalyModel === 'autoencoder'
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-2xs'
                      : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  Autoencoder (Unsup.)
                </button>
                <button
                  type="button"
                  onClick={() => setAnomalyModel('transfer_learning')}
                  className={`px-2 py-1.5 rounded text-[11px] font-semibold border text-center transition-all ${
                    anomalyModel === 'transfer_learning'
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-2xs'
                      : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  ResNet Classifier
                </button>
              </div>

              {/* Threshold & Reconstruction Metric Bar */}
              <div className="pt-2 border-t border-slate-200/80 text-[11px] space-y-1">
                <div className="flex justify-between text-slate-600">
                  <span>Recon. Error (MSE+SSIM):</span>
                  <span className="font-mono font-bold text-slate-900">
                    {result?.isDefective ? '0.0842' : '0.0128'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-600">
                  <span>Decision Threshold (τ = μ+2.8σ):</span>
                  <span className="font-mono font-bold text-indigo-600">0.0380</span>
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1 flex">
                  <div
                    className={`h-full transition-all duration-300 ${
                      result?.isDefective ? 'bg-rose-500 w-[85%]' : 'bg-emerald-500 w-[28%]'
                    }`}
                  />
                </div>
              </div>
            </div>

            {/* Factory Lighting Perturbation */}
            <div>
              <div className="flex items-center gap-1.5 mb-1.5">
                <SunMedium className="w-3.5 h-3.5 text-amber-500" />
                <label className="text-xs font-bold text-slate-700">
                  Illumination Condition
                </label>
              </div>
              <select
                id="select-lighting"
                value={lighting}
                onChange={(e) => setLighting(e.target.value as LightingCondition)}
                className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              >
                {LIGHTING_OPTIONS.map((l) => (
                  <option key={l.id} value={l.id}>{l.label}</option>
                ))}
              </select>
            </div>

            {/* Part Orientation Rotation */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-1.5">
                  <RotateCw className="w-3.5 h-3.5 text-blue-500" />
                  <label className="text-xs font-bold text-slate-700">
                    Conveyor Rotation
                  </label>
                </div>
                <span className="text-xs font-mono font-bold text-slate-600">{rotation}°</span>
              </div>
              <div className="grid grid-cols-4 gap-1.5">
                {[0, 90, 180, 270].map((deg) => (
                  <button
                    key={deg}
                    onClick={() => setRotation(deg)}
                    className={`py-1.5 text-xs font-mono font-medium rounded-md border transition-colors ${
                      rotation === deg
                        ? 'bg-blue-50 border-blue-500 text-blue-800 font-bold'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {deg}°
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Defect Classifier (Transfer Learning) Card */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-indigo-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Defect Classifier
                </h3>
              </div>
              <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-md text-[10px] font-mono">
                <button
                  type="button"
                  onClick={() => setClassifierBackbone('efficientnet_b0')}
                  className={`px-2 py-0.5 rounded transition-colors ${classifierBackbone === 'efficientnet_b0' ? 'bg-white text-indigo-700 font-bold shadow-2xs' : 'text-slate-600'}`}
                >
                  EfficientNet-B0
                </button>
                <button
                  type="button"
                  onClick={() => setClassifierBackbone('mobilenet_v2')}
                  className={`px-2 py-0.5 rounded transition-colors ${classifierBackbone === 'mobilenet_v2' ? 'bg-white text-indigo-700 font-bold shadow-2xs' : 'text-slate-600'}`}
                >
                  MobileNetV2
                </button>
              </div>
            </div>

            {/* Loss / Imbalance Badge */}
            <div className="flex items-center justify-between text-[11px] bg-slate-50 p-2 rounded-lg border border-slate-200 text-slate-600">
              <span>Loss Criterion:</span>
              <span className="font-mono font-bold text-indigo-700">Focal Loss (γ=2.0, balanced α)</span>
            </div>

            {/* 7-Class Softmax Probabilities Breakdown */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-bold text-slate-700">
                <span>Softmax Probabilities (7 Classes)</span>
                <span className="text-[10px] font-mono font-normal text-slate-500">Output Logits</span>
              </div>
              {DEFECT_OPTIONS.map((opt) => {
                const prob = result?.probabilities?.[opt.id] ?? (result?.predictedClass === opt.id ? result.confidence : 0.01);
                const isTop = result?.predictedClass === opt.id;
                return (
                  <div key={opt.id} className="space-y-0.5">
                    <div className="flex justify-between text-[11px]">
                      <span className={`font-medium flex items-center gap-1 ${isTop ? 'text-indigo-950 font-bold' : 'text-slate-600'}`}>
                        {isTop && <Check className="w-3 h-3 text-indigo-600 shrink-0" />}
                        {opt.label}
                      </span>
                      <span className="font-mono font-semibold text-slate-700">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isTop
                            ? opt.id === 'normal'
                              ? 'bg-emerald-500'
                              : 'bg-indigo-600'
                            : 'bg-slate-300'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Confusion Matrix Action Button */}
            <button
              id="btn-view-confusion-matrix"
              type="button"
              onClick={() => setShowConfusionMatrixModal(true)}
              className="w-full mt-2 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-200 text-xs font-bold transition-colors shadow-2xs"
            >
              <BarChart2 className="w-3.5 h-3.5" />
              View Confusion Matrix & PR/F1 Scorecard
            </button>
          </div>

          {/* Defect Localization Stage Card (Grad-CAM / U-Net / Calibration) */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-rose-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Defect Localization
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-200 font-bold uppercase">
                Stage 4
              </span>
            </div>

            {/* Localization Method Switch */}
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-1.5">
                Localization Engine
              </label>
              <div className="grid grid-cols-3 gap-1 p-1 bg-slate-100 rounded-lg text-xs font-medium">
                <button
                  type="button"
                  onClick={() => setLocalizationMethod('gradcam')}
                  className={`py-1.5 px-2 rounded-md transition-colors text-[11px] font-semibold text-center ${
                    localizationMethod === 'gradcam'
                      ? 'bg-white text-rose-700 shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Grad-CAM++
                </button>
                <button
                  type="button"
                  onClick={() => setLocalizationMethod('unet')}
                  className={`py-1.5 px-2 rounded-md transition-colors text-[11px] font-semibold text-center ${
                    localizationMethod === 'unet'
                      ? 'bg-white text-rose-700 shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  U-Net Mask
                </button>
                <button
                  type="button"
                  onClick={() => setLocalizationMethod('hybrid')}
                  className={`py-1.5 px-2 rounded-md transition-colors text-[11px] font-semibold text-center ${
                    localizationMethod === 'hybrid'
                      ? 'bg-white text-rose-700 shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Hybrid HUD
                </button>
              </div>
            </div>

            {/* Physical Calibration Ratio (px to mm) */}
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-slate-700">
                <div className="flex items-center gap-1.5">
                  <Ruler className="w-3.5 h-3.5 text-blue-600" />
                  <span>Physical Calibration (mm/px)</span>
                </div>
                <span className="font-mono text-blue-700 font-bold">{pixelToMm.toFixed(3)} mm/px</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0.02"
                  max="0.30"
                  step="0.01"
                  value={pixelToMm}
                  onChange={(e) => setPixelToMm(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>
              <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                <button
                  type="button"
                  onClick={() => setPixelToMm(0.05)}
                  className={`px-1.5 py-0.5 rounded border ${pixelToMm === 0.05 ? 'bg-blue-100 text-blue-800 border-blue-300 font-bold' : 'hover:bg-slate-200 border-slate-200'}`}
                >
                  50μm (Macro)
                </button>
                <button
                  type="button"
                  onClick={() => setPixelToMm(0.10)}
                  className={`px-1.5 py-0.5 rounded border ${pixelToMm === 0.10 ? 'bg-blue-100 text-blue-800 border-blue-300 font-bold' : 'hover:bg-slate-200 border-slate-200'}`}
                >
                  100μm (Std)
                </button>
                <button
                  type="button"
                  onClick={() => setPixelToMm(0.20)}
                  className={`px-1.5 py-0.5 rounded border ${pixelToMm === 0.20 ? 'bg-blue-100 text-blue-800 border-blue-300 font-bold' : 'hover:bg-slate-200 border-slate-200'}`}
                >
                  200μm (Wide)
                </button>
              </div>
            </div>

            {/* Overlay Elements Toggle */}
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-1.5">
                CAD HUD Elements
              </label>
              <div className="grid grid-cols-3 gap-1.5 text-xs">
                <button
                  type="button"
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`py-1 px-2 rounded-md border text-center transition-colors font-medium text-[11px] ${
                    showBoundingBoxes
                      ? 'bg-rose-50 border-rose-300 text-rose-800 font-bold'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Bounding Box
                </button>
                <button
                  type="button"
                  onClick={() => setShowContours(!showContours)}
                  className={`py-1 px-2 rounded-md border text-center transition-colors font-medium text-[11px] ${
                    showContours
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-800 font-bold'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Contours
                </button>
                <button
                  type="button"
                  onClick={() => setShowHudLabels(!showHudLabels)}
                  className={`py-1 px-2 rounded-md border text-center transition-colors font-medium text-[11px] ${
                    showHudLabels
                      ? 'bg-blue-50 border-blue-300 text-blue-800 font-bold'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Metric Tags
                </button>
              </div>
            </div>

            {/* Calibrated Defect Regions Telemetry */}
            {result?.isDefective ? (
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                  <div className="flex items-center gap-1.5">
                    <Scan className="w-3.5 h-3.5 text-rose-600" />
                    <span>Physical Measurements</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">
                    FOV: {(256 * pixelToMm).toFixed(1)} x {(256 * pixelToMm).toFixed(1)} mm
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                  <div className="bg-slate-50 p-2 rounded border border-slate-200">
                    <span className="text-slate-500 block text-[10px]">Defect Area</span>
                    <span className="font-bold text-slate-900">
                      {(result.defectPixelCount * pixelToMm * pixelToMm).toFixed(2)} mm²
                    </span>
                    <span className="text-slate-400 text-[10px] block">({result.defectPixelCount} px²)</span>
                  </div>
                  <div className="bg-slate-50 p-2 rounded border border-slate-200">
                    <span className="text-slate-500 block text-[10px]">Bounding Dimensions</span>
                    <span className="font-bold text-slate-900">
                      {(result.bbox.w * pixelToMm).toFixed(2)} x {(result.bbox.h * pixelToMm).toFixed(2)} mm
                    </span>
                    <span className="text-slate-400 text-[10px] block">({result.bbox.w}x{result.bbox.h} px)</span>
                  </div>
                  <div className="bg-slate-50 p-2 rounded border border-slate-200">
                    <span className="text-slate-500 block text-[10px]">Defect Centroid</span>
                    <span className="font-bold text-slate-900">
                      ({result.bbox.x + Math.round(result.bbox.w / 2)}, {result.bbox.y + Math.round(result.bbox.h / 2)})
                    </span>
                  </div>
                  <div className="bg-slate-50 p-2 rounded border border-slate-200">
                    <span className="text-slate-500 block text-[10px]">Aspect Ratio</span>
                    <span className="font-bold text-slate-900">
                      {(result.bbox.w / Math.max(1, result.bbox.h)).toFixed(2)}:1
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 text-center text-xs text-emerald-800 font-medium">
                Part classified Normal: 0 defect contours or saliency activations detected.
              </div>
            )}

            {/* Scorecard Action Button */}
            <button
              id="btn-view-localization-modal"
              type="button"
              onClick={() => setShowLocalizationModal(true)}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 text-xs font-bold transition-colors shadow-2xs"
            >
              <Target className="w-3.5 h-3.5" />
              View 4-Panel Localization Scorecard
            </button>
          </div>

          {/* False-Positive / Missed-Detection Reduction Module (Stage 5) */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-blue-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Refinement & TTA Engine
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 font-bold uppercase">
                Stage 5
              </span>
            </div>

            {/* 1. Confidence-based filtering rule */}
            <div className="space-y-2 p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800 flex items-center gap-1.5">
                  <Filter className="w-3.5 h-3.5 text-blue-600" />
                  Dual-Model Agreement Gate
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableDualFiltering}
                    onChange={(e) => setEnableDualFiltering(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-8 h-4 bg-slate-300 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              <p className="text-[11px] text-slate-500">
                Only flags defect if BOTH Anomaly Detector (&tau;<sub>a</sub>) and Classifier (&tau;<sub>c</sub>) agree above thresholds.
              </p>
              <div className="grid grid-cols-2 gap-2 pt-1">
                <div>
                  <div className="flex justify-between text-[10px] text-slate-600 mb-0.5">
                    <span>Anomaly Threshold &tau;<sub>a</sub></span>
                    <span className="font-mono font-bold">{anomalyThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.30"
                    max="0.85"
                    step="0.05"
                    value={anomalyThreshold}
                    onChange={(e) => setAnomalyThreshold(parseFloat(e.target.value))}
                    className="w-full accent-blue-600 h-1 bg-slate-200 rounded-lg cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] text-slate-600 mb-0.5">
                    <span>Classifier Conf &tau;<sub>c</sub></span>
                    <span className="font-mono font-bold">{(classifierThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.40"
                    max="0.90"
                    step="0.05"
                    value={classifierThreshold}
                    onChange={(e) => setClassifierThreshold(parseFloat(e.target.value))}
                    className="w-full accent-blue-600 h-1 bg-slate-200 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
              <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-200">
                <span className="text-slate-600">Gate Consensus:</span>
                <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                  dualAgreed 
                    ? 'bg-emerald-100 text-emerald-800' 
                    : 'bg-amber-100 text-amber-800'
                }`}>
                  {dualAgreed ? 'MODELS AGREE' : 'CONFLICT (SUPPRESSED/REVIEW)'}
                </span>
              </div>
            </div>

            {/* 2. Morphological Filtering (Opening / Closing) */}
            <div className="space-y-2 p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-600" />
                  Morphological Noise Pruning
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableMorphFilter}
                    onChange={(e) => setEnableMorphFilter(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-8 h-4 bg-slate-300 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600"></div>
                </label>
              </div>
              <p className="text-[11px] text-slate-500">
                Applies morphological opening/closing and purges noise speckles under minimum area threshold.
              </p>
              <div>
                <div className="flex justify-between text-[10px] text-slate-600 mb-0.5">
                  <span>Minimum Area Threshold</span>
                  <span className="font-mono font-bold">{minAreaThreshold} px ({((minAreaThreshold * pixelToMm * pixelToMm)).toFixed(3)} mm²)</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="80"
                  step="5"
                  value={minAreaThreshold}
                  onChange={(e) => setMinAreaThreshold(parseInt(e.target.value))}
                  className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg cursor-pointer"
                />
              </div>
              <div className="flex items-center justify-between gap-1 pt-1">
                <span className="text-[10px] text-slate-500">Presets:</span>
                <div className="flex gap-1">
                  <button
                    type="button"
                    onClick={() => setMinAreaThreshold(10)}
                    className={`px-1.5 py-0.5 text-[10px] rounded border ${minAreaThreshold === 10 ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-200'}`}
                  >
                    10 px (Speckle)
                  </button>
                  <button
                    type="button"
                    onClick={() => setMinAreaThreshold(25)}
                    className={`px-1.5 py-0.5 text-[10px] rounded border ${minAreaThreshold === 25 ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-200'}`}
                  >
                    25 px (Standard)
                  </button>
                  <button
                    type="button"
                    onClick={() => setMinAreaThreshold(50)}
                    className={`px-1.5 py-0.5 text-[10px] rounded border ${minAreaThreshold === 50 ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-200'}`}
                  >
                    50 px (Aggressive)
                  </button>
                </div>
              </div>
              <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-200 font-mono">
                <span className="text-slate-600 font-sans">Noise Blob Filter:</span>
                <span className={`font-bold text-[10px] px-1.5 py-0.5 rounded ${isNoiseBlob ? 'bg-amber-100 text-amber-800' : 'bg-slate-200 text-slate-800'}`}>
                  {isNoiseBlob ? '1 NOISE SPECKLE PURGED' : '0 BLOBS PURGED'}
                </span>
              </div>
            </div>

            {/* 3. Test-Time Augmentation (TTA) */}
            <div className="space-y-2 p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-800 flex items-center gap-1.5">
                  <RotateCw className="w-3.5 h-3.5 text-emerald-600" />
                  Test-Time Augmentation (TTA)
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableTTA}
                    onChange={(e) => setEnableTTA(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-8 h-4 bg-slate-300 peer-focus:outline-hidden rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-emerald-600"></div>
                </label>
              </div>
              <p className="text-[11px] text-slate-500">
                Runs inference across 6 augmented views (Original + H-Flip + V-Flip + &plusmn;5&deg; Rot + 90&deg;) and votes on consistency.
              </p>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-600">Multi-View Consensus:</span>
                <span className="font-mono font-bold text-slate-900">
                  {(ttaVoteRatio * 100).toFixed(0)}% ({Math.round(ttaVoteRatio * 6)}/6 views)
                </span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${ttaConsensus ? 'bg-emerald-500' : 'bg-amber-500'}`}
                  style={{ width: `${ttaVoteRatio * 100}%` }}
                />
              </div>
            </div>

            {/* 4. Borderline Review Queue Status */}
            <div className="p-3 bg-yellow-50/60 rounded-lg border border-yellow-200 text-xs space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-yellow-900 flex items-center gap-1.5">
                  <AlertOctagon className="w-3.5 h-3.5 text-yellow-600" />
                  Borderline Review Logger
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-800 font-bold">
                  {manualReviewList.length} In Queue
                </span>
              </div>
              <p className="text-[11px] text-yellow-800">
                Borderline confidence [{(borderlineLow * 100).toFixed(0)}% - {(borderlineHigh * 100).toFixed(0)}%] or model disagreements are routed to <code>outputs/manual_review_queue.json</code> instead of hard decisions.
              </p>
            </div>

            {/* Scorecard Action Button */}
            <button
              id="btn-view-refinement-modal"
              type="button"
              onClick={() => setShowRefinementModal(true)}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 text-xs font-bold transition-colors shadow-2xs cursor-pointer"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              View Refinement & Audit Scorecard
            </button>
          </div>
        </div>

        {/* Right Column: Visual Stage & Inspection Heatmaps (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs">
            {/* View Mode Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-emerald-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Optical Inspection Viewport
                </h3>
              </div>

              {/* View Mode Selector Tabs */}
              <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200 text-xs">
                <button
                  onClick={() => setActiveViewMode('overlay')}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    activeViewMode === 'overlay' ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Localized Overlay
                </button>
                <button
                  onClick={() => setActiveViewMode('raw')}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    activeViewMode === 'raw' ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Raw Camera
                </button>
                <button
                  onClick={() => setActiveViewMode('heatmap')}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    activeViewMode === 'heatmap' ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Anomaly Heatmap
                </button>
                <button
                  onClick={() => setActiveViewMode('mask')}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    activeViewMode === 'mask' ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Ground Truth Mask
                </button>
                <button
                  onClick={() => setActiveViewMode('quad')}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    activeViewMode === 'quad' ? 'bg-white shadow-2xs text-slate-900' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  4-Panel Report
                </button>
              </div>
            </div>

            {/* Canvas Stage Containers */}
            <div className="bg-slate-950 p-4 rounded-xl flex items-center justify-center relative min-h-[380px] overflow-hidden">
              {/* Corner Grid Crosshairs */}
              <div className="absolute top-2 left-2 text-slate-600 font-mono text-[10px] select-none">
                [CAM-01 • 256x256 • RAW 8-BIT]
              </div>
              <div className="absolute top-2 right-2 text-slate-600 font-mono text-[10px] select-none">
                TRIGGER: SYNC_PULSE
              </div>

              {/* Single View: Overlay / Raw / Heatmap / Mask */}
              {activeViewMode !== 'quad' && (
                <div className="relative border border-slate-700 rounded-lg overflow-hidden shadow-xl">
                  {/* Canvas Elements */}
                  <canvas
                    ref={rawCanvasRef}
                    width={256}
                    height={256}
                    className={`w-[320px] h-[320px] sm:w-[360px] sm:h-[360px] object-contain ${
                      activeViewMode === 'raw' || activeViewMode === 'overlay' ? 'block' : 'hidden'
                    }`}
                  />
                  <canvas
                    ref={heatmapCanvasRef}
                    width={256}
                    height={256}
                    className={`w-[320px] h-[320px] sm:w-[360px] sm:h-[360px] object-contain ${
                      activeViewMode === 'heatmap' ? 'block' : 'hidden'
                    }`}
                  />
                  <canvas
                    ref={maskCanvasRef}
                    width={256}
                    height={256}
                    className={`w-[320px] h-[320px] sm:w-[360px] sm:h-[360px] object-contain ${
                      activeViewMode === 'mask' ? 'block' : 'hidden'
                    }`}
                  />

                  {/* High-Precision Bounding Box Overlay on top of canvas */}
                  {activeViewMode === 'overlay' && result?.isDefective && result.bbox.w > 0 && showBoundingBoxes && (
                    <div
                      className="absolute border-2 border-rose-500 pointer-events-none transition-all duration-150"
                      style={{
                        left: `${(result.bbox.x / 256) * 100}%`,
                        top: `${(result.bbox.y / 256) * 100}%`,
                        width: `${(result.bbox.w / 256) * 100}%`,
                        height: `${(result.bbox.h / 256) * 100}%`,
                        backgroundColor: showContours ? 'rgba(239, 68, 68, 0.18)' : 'transparent',
                      }}
                    >
                      {/* Industrial CAD Corner Notches */}
                      <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-rose-400" />
                      <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-rose-400" />
                      <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-rose-400" />
                      <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-rose-400" />

                      {/* Contour dashed perimeter if enabled */}
                      {showContours && (
                        <div className="absolute inset-0 border border-dashed border-amber-400/80 rounded-xs pointer-events-none" />
                      )}

                      {/* Centroid Reticle */}
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 border border-cyan-400 rounded-full bg-cyan-500/50" />

                      {/* Tag label */}
                      {showHudLabels && (
                        <div className="absolute -top-7 left-0 bg-slate-950/90 border border-rose-500 text-white text-[9px] font-mono px-2 py-0.5 rounded font-bold uppercase tracking-wider whitespace-nowrap shadow-lg flex items-center gap-1.5">
                          <span className="text-rose-400">{result.predictedClass}</span>
                          <span className="text-slate-400">|</span>
                          <span className="text-emerald-300">{(result.bbox.w * pixelToMm).toFixed(1)}x{(result.bbox.h * pixelToMm).toFixed(1)}mm</span>
                          <span className="text-slate-400">|</span>
                          <span className="text-blue-300">{(result.defectPixelCount * pixelToMm * pixelToMm).toFixed(1)}mm²</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* HUD Telemetry Overlay on Canvas */}
                  {activeViewMode === 'overlay' && (
                    <div className="absolute bottom-0 inset-x-0 bg-slate-950/90 backdrop-blur-xs p-2 text-white font-mono text-[10px] flex items-center justify-between border-t border-slate-800">
                      <div>
                        {result?.isDefective ? (
                          <span className="text-rose-400 font-bold flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                            DEFECT: {result.predictedClass.toUpperCase()} [{(result.confidence * 100).toFixed(1)}%]
                          </span>
                        ) : (
                          <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                            NORMAL • PASS (ZERO DEFECT)
                          </span>
                        )}
                      </div>
                      <div className="text-slate-400 flex items-center gap-2">
                        <span>FOV: {(256 * pixelToMm).toFixed(1)}mm</span>
                        <span>•</span>
                        <span>CALIB: {pixelToMm.toFixed(2)}mm/px</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 4-Panel Quad Report View */}
              {activeViewMode === 'quad' && (
                <div className="grid grid-cols-2 gap-2 w-full max-w-[540px]">
                  {/* Panel 1: Raw Sensor */}
                  <div className="bg-slate-900 rounded-lg p-1.5 border border-slate-800">
                    <div className="text-[10px] font-mono text-amber-400 mb-1">1. RAW SENSOR IMAGE</div>
                    <canvas ref={rawCanvasRef} width={256} height={256} className="w-full aspect-square rounded" />
                  </div>

                  {/* Panel 2: Heatmap */}
                  <div className="bg-slate-900 rounded-lg p-1.5 border border-slate-800">
                    <div className="text-[10px] font-mono text-cyan-400 mb-1">2. ANOMALY DENSITY HEATMAP</div>
                    <canvas ref={heatmapCanvasRef} width={256} height={256} className="w-full aspect-square rounded" />
                  </div>

                  {/* Panel 3: Mask */}
                  <div className="bg-slate-900 rounded-lg p-1.5 border border-slate-800">
                    <div className="text-[10px] font-mono text-emerald-400 mb-1">3. GROUND TRUTH MASK</div>
                    <canvas ref={maskCanvasRef} width={256} height={256} className="w-full aspect-square rounded" />
                  </div>

                  {/* Panel 4: HUD */}
                  <div className="bg-slate-900 rounded-lg p-1.5 border border-slate-800 flex flex-col justify-between">
                    <div>
                      <div className="text-[10px] font-mono text-rose-400 mb-1">4. INSPECTION TELEMETRY</div>
                      <div className="space-y-1.5 text-slate-300 font-mono text-[11px] p-2 bg-slate-950 rounded">
                        <div>VERDICT: <span className={result?.isDefective ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                          {result?.isDefective ? 'REJECT' : 'PASS'}
                        </span></div>
                        <div>CLASS: <span className="text-white font-bold">{result?.predictedClass.toUpperCase()}</span></div>
                        <div>CONF: <span className="text-emerald-400">{((result?.confidence ?? 0) * 100).toFixed(1)}%</span></div>
                        <div>SEVERITY: <span className="text-amber-400">{result?.severity}</span></div>
                        <div>AREA: <span className="text-blue-400">{result?.coveragePct}% ({result?.defectPixelCount} px)</span></div>
                        <div>LATENCY: <span className="text-purple-400">{result?.processingTimeMs} ms</span></div>
                      </div>
                    </div>
                    <div className="text-[9px] text-slate-500 font-mono text-center">
                      Auto-Logged to outputs/
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Telemetry Strip */}
            <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <span className="text-slate-500 block text-[11px]">Active Substrate</span>
                <span className="font-semibold text-slate-800">{SUBSTRATES.find(s => s.id === substrate)?.label}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">Illumination Mode</span>
                <span className="font-semibold text-slate-800">{LIGHTING_OPTIONS.find(l => l.id === lighting)?.label}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">Defect Severity</span>
                <span className={`font-semibold ${
                  result?.severity === 'Critical' ? 'text-rose-600' :
                  result?.severity === 'Moderate' ? 'text-amber-600' :
                  result?.severity === 'Minor' ? 'text-blue-600' : 'text-emerald-600'
                }`}>
                  {result?.severity}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">BBox Dimensions</span>
                <span className="font-mono font-semibold text-slate-800">
                  {result?.isDefective ? `${result.bbox.w}x${result.bbox.h} px` : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Classification Metrics Modal */}
      {showConfusionMatrixModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto border border-slate-200 shadow-xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <div className="p-2 bg-indigo-50 text-indigo-700 rounded-lg">
                  <BarChart2 className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Defect Classifier Evaluation Scorecard
                  </h3>
                  <p className="text-xs text-slate-500">
                    Trained on MVTec AD + procedural samples with Focal Loss (γ=2.0)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowConfusionMatrixModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* High-level performance cards */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                <span className="text-[11px] font-medium text-slate-500 block">Overall Accuracy</span>
                <span className="text-lg font-bold text-emerald-600">96.5%</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                <span className="text-[11px] font-medium text-slate-500 block">Macro F1-Score</span>
                <span className="text-lg font-bold text-indigo-600">0.961</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center">
                <span className="text-[11px] font-medium text-slate-500 block">Validation Samples</span>
                <span className="text-lg font-bold text-slate-800">230 items</span>
              </div>
            </div>

            {/* 7x7 Confusion Matrix Grid */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  7-Class Confusion Matrix (Rows: Ground Truth, Cols: Predicted)
                </h4>
                <span className="text-[11px] text-slate-500">Normal + 6 Defect Types</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-center text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700">
                      <th className="p-2 text-left font-semibold">True \ Pred</th>
                      {['Norm', 'Crack', 'Scratch', 'Dent', 'Stain', 'Discol', 'DimIrr'].map((h) => (
                        <th key={h} className="p-2 font-semibold">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: 'Normal', row: [48, 1, 0, 0, 1, 0, 0] },
                      { label: 'Crack', row: [0, 29, 1, 0, 0, 0, 0] },
                      { label: 'Scratch', row: [1, 1, 28, 0, 0, 0, 0] },
                      { label: 'Dent', row: [0, 0, 0, 29, 0, 0, 1] },
                      { label: 'Stain', row: [0, 0, 0, 0, 30, 0, 0] },
                      { label: 'Discoloration', row: [0, 0, 0, 0, 0, 29, 1] },
                      { label: 'Dim. Irreg.', row: [0, 0, 1, 1, 0, 0, 28] },
                    ].map((r, rIdx) => (
                      <tr key={r.label} className="border-b border-slate-100">
                        <td className="p-2 text-left font-medium text-slate-700 bg-slate-50">{r.label}</td>
                        {r.row.map((val, cIdx) => {
                          const isDiag = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className={`p-2 font-mono font-semibold ${
                                isDiag
                                  ? 'bg-indigo-50 text-indigo-700 font-bold'
                                  : val > 0
                                  ? 'bg-rose-50 text-rose-600'
                                  : 'text-slate-300'
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Per-Class Precision / Recall / F1 Table */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Per-Class Precision, Recall & F1 Metrics
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-100 text-slate-700">
                    <tr>
                      <th className="p-2">Defect Class</th>
                      <th className="p-2 text-right">Precision</th>
                      <th className="p-2 text-right">Recall</th>
                      <th className="p-2 text-right">F1-Score</th>
                      <th className="p-2 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {[
                      { name: 'Normal', precision: 0.98, recall: 0.96, f1: 0.97, support: 50 },
                      { name: 'Crack', precision: 0.94, recall: 0.97, f1: 0.95, support: 30 },
                      { name: 'Scratch', precision: 0.97, recall: 0.93, f1: 0.95, support: 30 },
                      { name: 'Dent', precision: 0.97, recall: 0.97, f1: 0.97, support: 30 },
                      { name: 'Stain', precision: 0.97, recall: 1.00, f1: 0.98, support: 30 },
                      { name: 'Discoloration', precision: 1.00, recall: 0.97, f1: 0.98, support: 30 },
                      { name: 'Dimensional Irreg.', precision: 0.93, recall: 0.93, f1: 0.93, support: 30 },
                    ].map((row) => (
                      <tr key={row.name} className="hover:bg-slate-50">
                        <td className="p-2 font-sans font-medium text-slate-800">{row.name}</td>
                        <td className="p-2 text-right text-emerald-600">{(row.precision * 100).toFixed(1)}%</td>
                        <td className="p-2 text-right text-blue-600">{(row.recall * 100).toFixed(1)}%</td>
                        <td className="p-2 text-right font-bold text-indigo-700">{row.f1.toFixed(2)}</td>
                        <td className="p-2 text-right text-slate-500">{row.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={() => setShowConfusionMatrixModal(false)}
                className="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors cursor-pointer"
              >
                Close Scorecard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 4-Panel Defect Localization Scorecard Modal */}
      {showLocalizationModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[92vh] overflow-y-auto border border-slate-200 shadow-2xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 bg-rose-50 text-rose-700 rounded-lg">
                  <Target className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Defect Localization & Physical Metric Scorecard
                  </h3>
                  <p className="text-xs text-slate-500">
                    Grad-CAM++ Saliency • U-Net Segmentation Mask • OpenCV Contours & mm Calibration ({pixelToMm.toFixed(3)} mm/px)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowLocalizationModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* 4-Panel Quad Visual Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 bg-slate-950 p-3 rounded-xl border border-slate-800">
              {/* Panel 1: Raw Sensor */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-amber-400">
                  <span>1. RAW SENSOR</span>
                  <span className="text-slate-500">256x256</span>
                </div>
                <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-900 relative">
                  <canvas ref={rawCanvasRef} width={256} height={256} className="w-full h-full object-contain" />
                  <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 text-white font-mono text-[9px]">
                    Substrate: {substrate}
                  </span>
                </div>
              </div>

              {/* Panel 2: Grad-CAM Saliency */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-cyan-400">
                  <span>2. GRAD-CAM++ ATTENTION</span>
                  <span className="text-slate-500">Conv5</span>
                </div>
                <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-900 relative">
                  <canvas ref={heatmapCanvasRef} width={256} height={256} className="w-full h-full object-contain" />
                  <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 text-cyan-300 font-mono text-[9px]">
                    Saliency Activation
                  </span>
                </div>
              </div>

              {/* Panel 3: Binary Segmentation Mask */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-emerald-400">
                  <span>3. U-NET BINARY MASK</span>
                  <span className="text-slate-500">Otsu Morph</span>
                </div>
                <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-900 relative">
                  <canvas ref={maskCanvasRef} width={256} height={256} className="w-full h-full object-contain" />
                  <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 text-emerald-300 font-mono text-[9px]">
                    {result?.isDefective ? `${result.defectPixelCount} px detected` : '0 px defect'}
                  </span>
                </div>
              </div>

              {/* Panel 4: Calibrated HUD Composite */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-rose-400">
                  <span>4. CALIBRATED CAD HUD</span>
                  <span className="text-slate-500">{pixelToMm.toFixed(2)} mm/px</span>
                </div>
                <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-slate-900 relative">
                  <canvas ref={rawCanvasRef} width={256} height={256} className="w-full h-full object-contain opacity-90" />
                  {result?.isDefective && result.bbox.w > 0 && (
                    <div
                      className="absolute border-2 border-rose-500 bg-rose-500/20"
                      style={{
                        left: `${(result.bbox.x / 256) * 100}%`,
                        top: `${(result.bbox.y / 256) * 100}%`,
                        width: `${(result.bbox.w / 256) * 100}%`,
                        height: `${(result.bbox.h / 256) * 100}%`,
                      }}
                    >
                      <div className="absolute -top-4 left-0 bg-rose-600 text-white text-[8px] font-mono px-1 rounded font-bold whitespace-nowrap">
                        {(result.bbox.w * pixelToMm).toFixed(1)}x{(result.bbox.h * pixelToMm).toFixed(1)}mm
                      </div>
                    </div>
                  )}
                  <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 text-rose-300 font-mono text-[9px]">
                    Composite Overlay
                  </span>
                </div>
              </div>
            </div>

            {/* Comprehensive Physical Defect Metrics Table */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Industrial Metrology & Physical Calibration Telemetry
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border border-slate-200 rounded-lg overflow-hidden">
                  <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
                    <tr>
                      <th className="p-2.5">Parameter</th>
                      <th className="p-2.5">Pixel Dimension</th>
                      <th className="p-2.5">Calibrated Metric ({pixelToMm} mm/px)</th>
                      <th className="p-2.5">Status / Quality Gate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                    <tr>
                      <td className="p-2.5 font-sans font-medium text-slate-900">Defect Classification</td>
                      <td className="p-2.5 uppercase font-bold text-rose-600">{result?.predictedClass}</td>
                      <td className="p-2.5 font-bold text-indigo-600">{((result?.confidence ?? 0) * 100).toFixed(2)}% softmax confidence</td>
                      <td className="p-2.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${result?.isDefective ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'}`}>
                          {result?.isDefective ? 'REJECT' : 'PASS'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-sans font-medium text-slate-900">Total Defect Surface Area</td>
                      <td className="p-2.5">{result?.defectPixelCount} px² ({result?.coveragePct}%)</td>
                      <td className="p-2.5 font-bold text-slate-900">
                        {result?.isDefective ? `${((result?.defectPixelCount ?? 0) * pixelToMm * pixelToMm).toFixed(2)} mm²` : '0.00 mm²'}
                      </td>
                      <td className="p-2.5">
                        <span className="text-[11px] text-slate-600 font-sans">
                          {result?.severity} Severity ({result?.coveragePct}% of FOV)
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-sans font-medium text-slate-900">Bounding Box Extent (W x H)</td>
                      <td className="p-2.5">{result?.isDefective ? `${result.bbox.w} x ${result.bbox.h} px` : 'N/A'}</td>
                      <td className="p-2.5 font-bold text-slate-900">
                        {result?.isDefective ? `${(result.bbox.w * pixelToMm).toFixed(2)} x ${(result.bbox.h * pixelToMm).toFixed(2)} mm` : '0.00 x 0.00 mm'}
                      </td>
                      <td className="p-2.5">
                        <span className="text-[11px] text-slate-600 font-sans">
                          Aspect ratio: {result?.isDefective ? (result.bbox.w / Math.max(1, result.bbox.h)).toFixed(2) : '1.00'}:1
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-sans font-medium text-slate-900">Defect Centroid (Center of Mass)</td>
                      <td className="p-2.5">
                        {result?.isDefective ? `(${result.bbox.x + Math.round(result.bbox.w / 2)}, ${result.bbox.y + Math.round(result.bbox.h / 2)}) px` : 'N/A'}
                      </td>
                      <td className="p-2.5 font-bold text-slate-900">
                        {result?.isDefective ? `(${((result.bbox.x + result.bbox.w / 2) * pixelToMm).toFixed(2)}, ${((result.bbox.y + result.bbox.h / 2) * pixelToMm).toFixed(2)}) mm` : 'N/A'}
                      </td>
                      <td className="p-2.5">
                        <span className="text-[11px] text-slate-600 font-sans">
                          Spatial offset relative to substrate origin (0,0)
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-sans font-medium text-slate-900">Camera Optical Field of View</td>
                      <td className="p-2.5">256 x 256 px</td>
                      <td className="p-2.5 font-bold text-slate-900">
                        {(256 * pixelToMm).toFixed(1)} x {(256 * pixelToMm).toFixed(1)} mm ({((256 * pixelToMm) * (256 * pixelToMm)).toFixed(0)} mm²)
                      </td>
                      <td className="p-2.5">
                        <span className="text-[11px] text-slate-600 font-sans">
                          Telecentric lens calibrated @ {pixelToMm * 1000} μm/pixel
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex justify-between items-center pt-2">
              <div className="text-xs text-slate-500 font-mono">
                Log target: <code>outputs/defect_localization_sample.png</code>
              </div>
              <button
                type="button"
                onClick={() => setShowLocalizationModal(false)}
                className="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors cursor-pointer"
              >
                Close Localization Scorecard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Industrial Refinement & Manual Review Audit Scorecard Modal */}
      {showRefinementModal && (
        <div 
          id="modal-refinement-scorecard"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto"
        >
          <div className="bg-white rounded-2xl border border-slate-200 shadow-2xl max-w-5xl w-full p-6 space-y-6 my-8">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-blue-50 text-blue-700 border border-blue-200">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-slate-900">
                      Industrial Refinement & Quality Triage Scorecard
                    </h3>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-100 text-blue-800 font-bold">
                      src/refinement.py
                    </span>
                  </div>
                  <p className="text-xs text-slate-500">
                    Dual Model Gating &bull; Morphological Noise Pruning &bull; Test-Time Augmentation (TTA) &bull; Human Audit Queue
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowRefinementModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* 4-Panel Refinement Stage Visual Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Panel 1: Dual Confidence Agreement */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Filter className="w-3.5 h-3.5 text-blue-600" />
                    1. Dual Agreement
                  </span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                    dualAgreed ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                  }`}>
                    {dualAgreed ? 'PASS' : 'CONFLICT'}
                  </span>
                </div>
                
                <div className="space-y-2 text-xs">
                  <div>
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>Anomaly Score</span>
                      <span className="font-mono font-bold">{simulatedAnomalyScore.toFixed(2)} (Req: &ge; {anomalyThreshold.toFixed(2)})</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-1.5 mt-1 overflow-hidden">
                      <div 
                        className={`h-full ${simulatedAnomalyScore >= anomalyThreshold ? 'bg-rose-500' : 'bg-emerald-500'}`}
                        style={{ width: `${Math.min(100, simulatedAnomalyScore * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] text-slate-600">
                      <span>Classifier Confidence</span>
                      <span className="font-mono font-bold">{((result?.confidence ?? 0) * 100).toFixed(1)}% (Req: &ge; {(classifierThreshold * 100).toFixed(0)}%)</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-1.5 mt-1 overflow-hidden">
                      <div 
                        className={`h-full ${((result?.confidence ?? 0) >= classifierThreshold) ? 'bg-indigo-500' : 'bg-amber-500'}`}
                        style={{ width: `${(result?.confidence ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 border-t border-slate-200 pt-2">
                  {dualAgreed 
                    ? 'Both models agree on state. No false-positive contradiction detected.' 
                    : 'Disagreement between unsupervised anomaly detector and supervised classifier.'}
                </p>
              </div>

              {/* Panel 2: Morphological Filter */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-indigo-600" />
                    2. Morphological
                  </span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                    isNoiseBlob ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {isNoiseBlob ? 'PURGED' : 'CLEAN'}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600">Raw Defect Area:</span>
                    <span className="font-mono font-bold text-slate-800">{rawDefectAreaPx} px²</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600">Min Area Threshold:</span>
                    <span className="font-mono font-bold text-indigo-600">{minAreaThreshold} px²</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-600">Cleaned Area:</span>
                    <span className="font-mono font-bold text-emerald-700">{cleanedDefectAreaPx} px²</span>
                  </div>
                  <div className="flex justify-between text-[11px] border-t border-slate-200 pt-1">
                    <span className="text-slate-600">Noise Blobs Purged:</span>
                    <span className="font-mono font-bold text-amber-700">{purgedBlobsCount}</span>
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 border-t border-slate-200 pt-2">
                  cv2.morphologyEx with 3x3 elliptical structuring element removes micro-speckles and optical dust.
                </p>
              </div>

              {/* Panel 3: Test-Time Augmentation (TTA) */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <RotateCw className="w-3.5 h-3.5 text-emerald-600" />
                    3. TTA Voting
                  </span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                    ttaConsensus ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                  }`}>
                    {(ttaVoteRatio * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
                  <div className="p-1.5 bg-white rounded border border-slate-200 text-center">
                    <span className="text-slate-500 block">Original</span>
                    <span className="font-bold text-slate-800">{result?.isDefective ? 'Defect' : 'Clean'}</span>
                  </div>
                  <div className="p-1.5 bg-white rounded border border-slate-200 text-center">
                    <span className="text-slate-500 block">H-Flip</span>
                    <span className="font-bold text-slate-800">{result?.isDefective && !isNoiseBlob ? 'Defect' : 'Clean'}</span>
                  </div>
                  <div className="p-1.5 bg-white rounded border border-slate-200 text-center">
                    <span className="text-slate-500 block">V-Flip</span>
                    <span className="font-bold text-slate-800">{result?.isDefective && !isNoiseBlob ? 'Defect' : 'Clean'}</span>
                  </div>
                  <div className="p-1.5 bg-white rounded border border-slate-200 text-center">
                    <span className="text-slate-500 block">Rotated</span>
                    <span className="font-bold text-slate-800">{result?.isDefective && !isNoiseBlob ? 'Defect' : 'Clean'}</span>
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 border-t border-slate-200 pt-2">
                  Inverse geometric transform maps coordinate frame back to canonical camera coordinates.
                </p>
              </div>

              {/* Panel 4: Authoritative Decision */}
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Check className="w-3.5 h-3.5 text-slate-700" />
                    4. Final Decision
                  </span>
                </div>

                <div className="p-2.5 rounded-lg text-center border font-bold text-xs" style={{
                  backgroundColor: finalDecision === 'CONFIRMED_DEFECT' ? '#fef2f2' : finalDecision === 'CLEAN_PASS' ? '#f0fdf4' : finalDecision === 'SUPPRESSED_FALSE_POSITIVE' ? '#fffbeb' : '#fefce8',
                  borderColor: finalDecision === 'CONFIRMED_DEFECT' ? '#fecaca' : finalDecision === 'CLEAN_PASS' ? '#bbf7d0' : finalDecision === 'SUPPRESSED_FALSE_POSITIVE' ? '#fde68a' : '#fef08a',
                  color: finalDecision === 'CONFIRMED_DEFECT' ? '#b91c1c' : finalDecision === 'CLEAN_PASS' ? '#15803d' : finalDecision === 'SUPPRESSED_FALSE_POSITIVE' ? '#92400e' : '#854d0e',
                }}>
                  {finalDecision}
                </div>

                <div className="space-y-1 text-[11px] text-slate-600">
                  <div className="flex justify-between">
                    <span>Part Metric:</span>
                    <span className="font-mono font-bold text-slate-800">
                      {((cleanedDefectAreaPx * pixelToMm * pixelToMm)).toFixed(3)} mm²
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Manual Review:</span>
                    <span className="font-mono font-bold text-slate-800">
                      {requiresManualReview ? 'REQUIRED' : 'NONE'}
                    </span>
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 border-t border-slate-200 pt-2">
                  {finalDecision === 'CONFIRMED_DEFECT' && 'Pneumatic rejection arm instructed to discard part.'}
                  {finalDecision === 'CLEAN_PASS' && 'Part cleared for downstream automated packaging.'}
                  {finalDecision === 'SUPPRESSED_FALSE_POSITIVE' && 'Micro-speckle suppressed. Yield preserved.'}
                  {finalDecision === 'MANUAL_REVIEW' && 'Borderline case routed to inspection audit queue.'}
                </p>
              </div>
            </div>

            {/* Manual Review Audit Queue Section (outputs/manual_review_queue.json) */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                    <History className="w-4 h-4 text-amber-600" />
                    Manual Review Audit Queue (outputs/manual_review_queue.json)
                  </h4>
                  <p className="text-xs text-slate-500">
                    Samples with borderline confidence [45% - 70%], dual model disagreement, or split TTA voting.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const newId = `AUDIT-${Math.floor(1000 + Math.random() * 9000)}`;
                    const now = new Date().toLocaleTimeString();
                    setManualReviewList(prev => [
                      {
                        id: newId,
                        timestamp: now,
                        class: result?.predictedClass ?? 'stain',
                        confidence: result?.confidence ?? 0.54,
                        anomalyScore: simulatedAnomalyScore,
                        reason: requiresManualReview 
                          ? 'Live inspection trigger: Borderline confidence / model disagreement' 
                          : 'Manual engineer flag: Review surface texture under microscope',
                        status: 'PENDING'
                      },
                      ...prev
                    ]);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-yellow-50 text-yellow-800 border border-yellow-300 text-xs font-bold hover:bg-yellow-100 transition-colors flex items-center gap-1.5 cursor-pointer shadow-2xs"
                >
                  <AlertOctagon className="w-3.5 h-3.5" />
                  Log Current Sample to Audit Ledger
                </button>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
                    <tr>
                      <th className="p-3">Sample ID</th>
                      <th className="p-3">Timestamp</th>
                      <th className="p-3">Predicted Class</th>
                      <th className="p-3">Confidence & Anomaly</th>
                      <th className="p-3">Audit Reason</th>
                      <th className="p-3">Triage Status</th>
                      <th className="p-3 text-right">Engineer Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {manualReviewList.map(item => (
                      <tr key={item.id} className="hover:bg-slate-50 transition-colors">
                        <td className="p-3 font-mono font-bold text-slate-900">{item.id}</td>
                        <td className="p-3 font-mono text-slate-500">{item.timestamp}</td>
                        <td className="p-3 uppercase font-semibold text-slate-700">{item.class}</td>
                        <td className="p-3 font-mono">
                          <span className="text-indigo-600 font-bold">{(item.confidence * 100).toFixed(1)}%</span>
                          <span className="text-slate-400 mx-1">/</span>
                          <span className="text-slate-600">{item.anomalyScore.toFixed(2)}</span>
                        </td>
                        <td className="p-3 text-slate-600 max-w-xs">{item.reason}</td>
                        <td className="p-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            item.status === 'PENDING' ? 'bg-yellow-100 text-yellow-800 border border-yellow-300' :
                            item.status === 'RESOLVED_PASS' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                          }`}>
                            {item.status}
                          </span>
                        </td>
                        <td className="p-3 text-right space-x-1 whitespace-nowrap">
                          {item.status === 'PENDING' ? (
                            <>
                              <button
                                type="button"
                                onClick={() => {
                                  setManualReviewList(prev => prev.map(r => r.id === item.id ? { ...r, status: 'RESOLVED_PASS' } : r));
                                }}
                                className="px-2 py-1 rounded bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 text-[10px] font-bold cursor-pointer"
                              >
                                Clear (Pass)
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  setManualReviewList(prev => prev.map(r => r.id === item.id ? { ...r, status: 'RESOLVED_DEFECT' } : r));
                                }}
                                className="px-2 py-1 rounded bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 text-[10px] font-bold cursor-pointer"
                              >
                                Reject (Defect)
                              </button>
                            </>
                          ) : (
                            <span className="text-[10px] text-slate-400 font-mono italic">Audit Signed Off</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex justify-between items-center pt-2 border-t border-slate-100">
              <div className="text-xs text-slate-500 font-mono">
                Ledger file: <code>outputs/manual_review_queue.json</code> &bull; Scorecard: <code>outputs/refinement_scorecard.png</code>
              </div>
              <button
                type="button"
                onClick={() => setShowRefinementModal(false)}
                className="px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors cursor-pointer"
              >
                Close Refinement Scorecard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
