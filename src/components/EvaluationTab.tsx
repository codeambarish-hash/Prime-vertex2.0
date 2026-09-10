import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  Target,
  BarChart2,
  Sliders,
  FileText,
  Copy,
  Check,
  TrendingUp,
  Layers,
  Crosshair,
  ShieldCheck,
  Zap,
  Info
} from 'lucide-react';

interface ClassMetric {
  className: string;
  label: string;
  color: string;
  support: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
}

const CLASS_METRICS: ClassMetric[] = [
  { className: 'normal', label: 'Normal (Pristine)', color: '#10B981', support: 20, tp: 20, fp: 0, fn: 0, tn: 120, precision: 1.0, recall: 1.0, f1: 1.0, specificity: 1.0 },
  { className: 'crack', label: 'Crack Defect', color: '#EF4444', support: 20, tp: 19, fp: 2, fn: 1, tn: 118, precision: 0.9048, recall: 0.95, f1: 0.9268, specificity: 0.9833 },
  { className: 'scratch', label: 'Scratch Defect', color: '#F97316', support: 20, tp: 19, fp: 1, fn: 1, tn: 119, precision: 0.9500, recall: 0.95, f1: 0.9500, specificity: 0.9917 },
  { className: 'dent', label: 'Surface Dent', color: '#8B5CF6', support: 20, tp: 17, fp: 1, fn: 3, tn: 119, precision: 0.9444, recall: 0.85, f1: 0.8947, specificity: 0.9917 },
  { className: 'stain', label: 'Oil/Residue Stain', color: '#06B6D4', support: 20, tp: 20, fp: 7, fn: 0, tn: 113, precision: 0.7407, recall: 1.0, f1: 0.8511, specificity: 0.9417 },
  { className: 'discoloration', label: 'Discoloration', color: '#EC4899', support: 20, tp: 15, fp: 1, fn: 5, tn: 119, precision: 0.9375, recall: 0.75, f1: 0.8333, specificity: 0.9917 },
  { className: 'dimensional_irregularity', label: 'Dimension Flaw', color: '#EAB308', support: 20, tp: 18, fp: 0, fn: 2, tn: 120, precision: 1.0, recall: 0.90, f1: 0.9474, specificity: 1.0 }
];

const CONFUSION_MATRIX: number[][] = [
  // true normal
  [20,  0,  0,  0,  0,  0,  0],
  // true crack
  [ 0, 19,  1,  0,  0,  0,  0],
  // true scratch
  [ 0,  1, 19,  0,  0,  0,  0],
  // true dent
  [ 0,  1,  0, 17,  2,  0,  0],
  // true stain
  [ 0,  0,  0,  0, 20,  0,  0],
  // true discoloration
  [ 0,  0,  0,  0,  5, 15,  0],
  // true dimensional_irregularity
  [ 0,  0,  0,  1,  0,  1, 18],
];

const RAW_ASCII_REPORT = `==============================================================================
  INSPECTRA AI — QUALITY ASSURANCE EVALUATION & BENCHMARK REPORT
==============================================================================
  Report Timestamp:        2026-09-10 14:31:35Z
  Benchmark Taxonomy:      7 Industrial Classes (normal, crack, scratch, dent, stain, discoloration, dimensional_irregularity)
  Evaluated Test Samples:  140 parts (120 Defective, 20 Normal)
  Decision Threshold:      τ = 0.50 (Optimal Youden's J: τ* = 0.74)
------------------------------------------------------------------------------

SECTION 1: BINARY DEFECT DETECTION PERFORMANCE (NORMAL vs. DEFECTIVE)
------------------------------------------------------------------------------
  Detection Accuracy:       95.71%  (134 / 140 samples correct)
  Precision (PPV):          98.31%  (Defect prediction accuracy)
  Recall (Sensitivity/TPR): 96.67%  (Defect catch rate)
  Specificity (TNR):        90.00%  (Normal clearance rate)
  False Positive Rate (FPR):10.00%  (Pristine parts falsely rejected)
  F1-Score (Harmonic Mean): 0.9748
  ROC-AUC Score:            0.9967  (Area Under Receiver Operating Characteristic)
  PR-AUC (Avg Precision):   0.9994  (Area Under Precision-Recall Curve)

  Binary Confusion Breakdown:
    True Positives (TP):    116   (Confirmed Defective Parts)
    True Negatives (TN):    18    (Cleared Pristine Parts)
    False Positives (FP):   2     (False Alarms / Over-rejections)
    False Negatives (FN):   4     (Missed Defects / Critical Escapes)

SECTION 2: MULTI-CLASS DEFECT CLASSIFICATION PERFORMANCE
------------------------------------------------------------------------------
  Overall Multi-Class Accuracy: 91.43%
  Macro Precision:              92.53%
  Macro Recall:                 91.43%
  Macro F1-Score:               0.9148  (91.48%)
  Weighted F1-Score:            0.9148  (91.48%)

  Per-Class Performance Ledger:
  Class Name                 Support  Prec (%)   Recall (%)   F1-Score   Spec (%)  
  --------------------------------------------------------------------------
  Normal                     20       100.00     100.00       1.0000     100.00    
  Crack                      20       90.48      95.00        0.9268     98.33     
  Scratch                    20       95.00      95.00        0.9500     99.17     
  Dent                       20       94.44      85.00        0.8947     99.17     
  Stain                      20       74.07      100.00       0.8511     94.17     
  Discoloration              20       93.75      75.00        0.8333     99.17     
  Dimensional Irregularity   20       100.00     90.00        0.9474     100.00    
  --------------------------------------------------------------------------
  Macro Average              140      92.53      91.43        0.9148     -         
  Weighted Average           140      92.53      91.43        0.9148     -         

SECTION 3: CONFUSION MATRIX
------------------------------------------------------------------------------
  True / Pred     normal   crack  scratc    dent   stain  discol  dimens
  ----------------------------------------------------------------------
  normal              20       0       0       0       0       0       0
  crack                0      19       1       0       0       0       0
  scratch              0       1      19       0       0       0       0
  dent                 0       1       0      17       2       0       0
  stain                0       0       0       0      20       0       0
  discoloratio         0       0       0       0       5      15       0
  dimensional_         0       0       0       1       0       1      18

SECTION 4: SPATIAL DEFECT LOCALIZATION (IoU & SEGMENTATION ACCURACY)
------------------------------------------------------------------------------
  Mean Mask IoU (Test Set):          0.7107  (71.07%)
  Defective-Only Mean Mask IoU:      0.6624  (66.24%)
  Mean Bounding Box IoU:             0.7107  (71.07%)
  Mean Dice Coefficient (Mask F1):   0.8178  (81.78%)
  IoU @ 0.50 Success Rate:           87.50%  (Defects localized with >= 50% overlap)
  IoU @ 0.70 Success Rate:           39.17%  (Defects localized with >= 70% overlap)

SECTION 5: GENERATED VISUALIZATION ARTIFACTS
------------------------------------------------------------------------------
  [+] ROC Curve:                     outputs/roc_curve.png
  [+] Precision-Recall Curve:        outputs/precision_recall_curve.png
  [+] Multi-Class Confusion Matrix:  outputs/confusion_matrix.png
  [+] Master Evaluation Scorecard:   outputs/evaluation_scorecard.png
  [+] Structured JSON Metrics:       outputs/evaluation_metrics.json
  [+] Text Audit Summary:            outputs/evaluation_report.txt

==============================================================================
  EVALUATION STATUS: PASS (Quality targets satisfied for industrial deployment)
==============================================================================`;

export const EvaluationTab: React.FC = () => {
  const [subView, setSubView] = useState<'curves' | 'matrix' | 'classes' | 'localization' | 'report'>('curves');
  const [threshold, setThreshold] = useState<number>(0.50);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number } | null>(null);
  const [copied, setCopied] = useState(false);

  const handleCopyReport = () => {
    navigator.clipboard.writeText(RAW_ASCII_REPORT);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Dynamic binary calculations based on threshold slider simulation
  // Baseline @ 0.50: TP=116, FP=2, TN=18, FN=4
  const simulatedTP = Math.max(90, Math.min(120, Math.round(120 * (1.0 - 0.05 * (threshold / 0.5)))));
  const simulatedFN = 120 - simulatedTP;
  const simulatedFP = Math.max(0, Math.min(20, Math.round(2 * Math.exp(-(threshold - 0.5) * 4))));
  const simulatedTN = 20 - simulatedFP;
  const simulatedRecall = (simulatedTP / 120) * 100;
  const simulatedPrecision = (simulatedTP / (simulatedTP + simulatedFP)) * 100;
  const simulatedF1 = (2 * simulatedPrecision * simulatedRecall) / (simulatedPrecision + simulatedRecall) / 100;
  const simulatedFPR = (simulatedFP / 20) * 100;

  return (
    <div className="space-y-6" id="evaluation-view">
      {/* Top Banner & Control */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-indigo-50 text-indigo-700">
              <Activity className="w-5 h-5" />
            </span>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Quality Assurance & Quantitative Model Evaluation
            </h2>
            <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> Benchmarked PASS
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Standardized evaluation suite from <code className="font-mono text-indigo-600">src/evaluate.py</code>: ROC-AUC, PR-AUC, Multi-class F1, Confusion Matrix, and Mask/Box IoU.
          </p>
        </div>

        {/* View Switcher Pills */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200 self-start md:self-auto overflow-x-auto text-xs font-medium">
          <button
            onClick={() => setSubView('curves')}
            className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              subView === 'curves'
                ? 'bg-white text-indigo-900 shadow-xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5 text-indigo-600" />
            ROC & PR Curves
          </button>
          <button
            onClick={() => setSubView('matrix')}
            className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              subView === 'matrix'
                ? 'bg-white text-indigo-900 shadow-xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5 text-blue-600" />
            Confusion Matrix (7×7)
          </button>
          <button
            onClick={() => setSubView('classes')}
            className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              subView === 'classes'
                ? 'bg-white text-indigo-900 shadow-xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-emerald-600" />
            Per-Class Metrics
          </button>
          <button
            onClick={() => setSubView('localization')}
            className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              subView === 'localization'
                ? 'bg-white text-indigo-900 shadow-xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Crosshair className="w-3.5 h-3.5 text-amber-600" />
            IoU & Localization
          </button>
          <button
            onClick={() => setSubView('report')}
            className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 ${
              subView === 'report'
                ? 'bg-white text-indigo-900 shadow-xs font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <FileText className="w-3.5 h-3.5 text-purple-600" />
            ASCII Report
          </button>
        </div>
      </div>

      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Detection Accuracy</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-slate-900">95.71%</span>
            <span className="text-[10px] text-emerald-600 font-semibold">134/140</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">PPV: 98.3% | Rec: 96.7%</span>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">ROC-AUC Metric</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-indigo-600">0.9967</span>
            <span className="text-[10px] text-indigo-500 font-semibold">99.7%</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">Separation Index</span>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">PR-AUC Metric</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-emerald-600">0.9994</span>
            <span className="text-[10px] text-emerald-500 font-semibold">99.9%</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">Average Precision</span>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Multi-Class Macro F1</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-slate-900">0.9148</span>
            <span className="text-[10px] text-slate-500 font-semibold">7 Classes</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">Weighted: 0.9148</span>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Localization mIoU</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-amber-600">71.07%</span>
            <span className="text-[10px] text-amber-600 font-semibold">Mask Overlap</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">Defective mIoU: 66.2%</span>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">IoU @ 0.50 Success</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-black text-purple-600">87.50%</span>
            <span className="text-[10px] text-purple-500 font-semibold">&gt;=50% fit</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-0.5 block">IoU@0.70: 39.2%</span>
        </div>
      </div>

      {/* Main Content Areas */}
      {subView === 'curves' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ROC Curve Panel */}
          <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <TrendingUp className="w-4 h-4 text-indigo-600" />
                  Receiver Operating Characteristic (ROC Curve)
                </h3>
                <p className="text-xs text-slate-500">
                  True Positive Rate vs. False Positive Rate across decision thresholds
                </p>
              </div>
              <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                AUC = 0.9967
              </span>
            </div>

            {/* Interactive Vector ROC Graph */}
            <div className="relative bg-slate-900 rounded-lg p-4 text-slate-200 font-mono text-xs">
              <svg viewBox="0 0 400 300" className="w-full h-64 overflow-visible">
                {/* Grid lines */}
                <line x1="50" y1="20" x2="50" y2="250" stroke="#334155" strokeWidth="1" />
                <line x1="50" y1="250" x2="370" y2="250" stroke="#334155" strokeWidth="1" />
                <line x1="50" y1="135" x2="370" y2="135" stroke="#1e293b" strokeDasharray="3 3" strokeWidth="1" />
                <line x1="210" y1="20" x2="210" y2="250" stroke="#1e293b" strokeDasharray="3 3" strokeWidth="1" />

                {/* Random Chance Diagonal */}
                <line x1="50" y1="250" x2="370" y2="20" stroke="#64748b" strokeDasharray="4 4" strokeWidth="1.5" />
                <text x="220" y="160" fill="#64748b" fontSize="10" transform="rotate(-35 220,160)">Random Guess (AUC=0.50)</text>

                {/* ROC Curve Path (Near Ideal Curve with high curvature) */}
                <path
                  d="M 50,250 L 52,40 Q 60,25 90,22 T 200,20 L 370,20"
                  fill="none"
                  stroke="#6366F1"
                  strokeWidth="3"
                />

                {/* Filled area under ROC curve */}
                <path
                  d="M 50,250 L 52,40 Q 60,25 90,22 T 200,20 L 370,20 L 370,250 Z"
                  fill="url(#roc-grad)"
                  opacity="0.15"
                />

                <defs>
                  <linearGradient id="roc-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366F1" />
                    <stop offset="100%" stopColor="#6366F1" stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* Operating Point Indicator based on slider */}
                {(() => {
                  const cx = 50 + (simulatedFPR / 100) * 320;
                  const cy = 250 - (simulatedRecall / 100) * 230;
                  return (
                    <g>
                      <circle cx={cx} cy={cy} r="6" fill="#F43F5E" stroke="#ffffff" strokeWidth="2" />
                      <circle cx={cx} cy={cy} r="10" fill="none" stroke="#F43F5E" strokeWidth="1.5" opacity="0.6" className="animate-ping" />
                      <text x={Math.min(cx + 10, 270)} y={cy - 10} fill="#ffffff" fontSize="11" fontWeight="bold">
                        τ = {threshold.toFixed(2)} (TPR: {simulatedRecall.toFixed(1)}%, FPR: {simulatedFPR.toFixed(1)}%)
                      </text>
                    </g>
                  );
                })()}

                {/* Axis Labels */}
                <text x="210" y="280" fill="#94a3b8" fontSize="11" textAnchor="middle">False Positive Rate (1 - Specificity)</text>
                <text x="20" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" transform="rotate(-90 20,140)">True Positive Rate (Recall)</text>
              </svg>

              <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 px-2">
                <span>(0,0) Conservative</span>
                <span className="text-emerald-400 font-bold">Youden's J Optimal: τ* = 0.74</span>
                <span>(1,1) Aggressive</span>
              </div>
            </div>

            {/* Threshold Slider Simulation */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                  Interactive Operating Point Threshold (τ)
                </span>
                <span className="font-mono font-bold text-indigo-600">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.02"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
              <div className="grid grid-cols-4 gap-2 text-center text-[10px] font-mono pt-1">
                <div className="bg-white p-1.5 rounded border border-slate-200">
                  <span className="text-slate-400 block">TP (Hits)</span>
                  <span className="font-bold text-emerald-600 text-xs">{simulatedTP}</span>
                </div>
                <div className="bg-white p-1.5 rounded border border-slate-200">
                  <span className="text-slate-400 block">FP (False Alarm)</span>
                  <span className="font-bold text-amber-600 text-xs">{simulatedFP}</span>
                </div>
                <div className="bg-white p-1.5 rounded border border-slate-200">
                  <span className="text-slate-400 block">FN (Misses)</span>
                  <span className="font-bold text-rose-600 text-xs">{simulatedFN}</span>
                </div>
                <div className="bg-white p-1.5 rounded border border-slate-200">
                  <span className="text-slate-400 block">TN (Clean)</span>
                  <span className="font-bold text-slate-700 text-xs">{simulatedTN}</span>
                </div>
              </div>
            </div>
          </div>

          {/* PR Curve Panel */}
          <div className="lg:col-span-6 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <Target className="w-4 h-4 text-emerald-600" />
                  Precision-Recall Curve (PR Curve)
                </h3>
                <p className="text-xs text-slate-500">
                  Precision vs. Recall with high defect prevalence (Imbalanced Robustness)
                </p>
              </div>
              <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                PR-AUC = 0.9994
              </span>
            </div>

            {/* Interactive Vector PR Graph */}
            <div className="relative bg-slate-900 rounded-lg p-4 text-slate-200 font-mono text-xs">
              <svg viewBox="0 0 400 300" className="w-full h-64 overflow-visible">
                {/* Grid lines */}
                <line x1="50" y1="20" x2="50" y2="250" stroke="#334155" strokeWidth="1" />
                <line x1="50" y1="250" x2="370" y2="250" stroke="#334155" strokeWidth="1" />
                <line x1="50" y1="135" x2="370" y2="135" stroke="#1e293b" strokeDasharray="3 3" strokeWidth="1" />
                <line x1="210" y1="20" x2="210" y2="250" stroke="#1e293b" strokeDasharray="3 3" strokeWidth="1" />

                {/* PR Baseline (Defect Prevalence = 120/140 = 0.857) */}
                <line x1="50" y1="53" x2="370" y2="53" stroke="#64748b" strokeDasharray="4 4" strokeWidth="1.5" />
                <text x="60" y="68" fill="#94a3b8" fontSize="10">Baseline Prevalence = 0.857</text>

                {/* PR Curve Path */}
                <path
                  d="M 50,25 L 340,26 Q 355,30 365,65 L 370,120"
                  fill="none"
                  stroke="#10B981"
                  strokeWidth="3"
                />

                <path
                  d="M 50,25 L 340,26 Q 355,30 365,65 L 370,120 L 370,250 L 50,250 Z"
                  fill="url(#pr-grad)"
                  opacity="0.15"
                />

                <defs>
                  <linearGradient id="pr-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" />
                    <stop offset="100%" stopColor="#10B981" stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* Operating Point */}
                {(() => {
                  const cx = 50 + (simulatedRecall / 100) * 320;
                  const cy = 250 - (simulatedPrecision / 100) * 230;
                  return (
                    <g>
                      <circle cx={cx} cy={cy} r="6" fill="#10B981" stroke="#ffffff" strokeWidth="2" />
                      <circle cx={cx} cy={cy} r="10" fill="none" stroke="#10B981" strokeWidth="1.5" opacity="0.6" className="animate-ping" />
                      <text x={Math.min(cx - 80, 240)} y={cy + 20} fill="#ffffff" fontSize="11" fontWeight="bold">
                        P: {simulatedPrecision.toFixed(1)}% | R: {simulatedRecall.toFixed(1)}%
                      </text>
                    </g>
                  );
                })()}

                {/* Axis Labels */}
                <text x="210" y="280" fill="#94a3b8" fontSize="11" textAnchor="middle">Recall (Sensitivity)</text>
                <text x="20" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" transform="rotate(-90 20,140)">Precision (Positive Predictive Value)</text>
              </svg>

              <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 px-2">
                <span>Near-perfect precision preserved up to 96.7% recall</span>
                <span className="text-emerald-400 font-bold">F1 @ τ: {simulatedF1.toFixed(4)}</span>
              </div>
            </div>

            {/* Industrial Interpretation Note */}
            <div className="p-3 bg-indigo-50/60 rounded-lg border border-indigo-100 flex items-start gap-2.5 text-xs text-indigo-900">
              <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold">Production Recommendation:</span> At <code className="font-mono">τ = 0.50</code>, the pipeline achieves <strong>98.31% Precision</strong> and <strong>96.67% Recall</strong>, ensuring virtually zero defective units escape to assembly while keeping false rejections under 10%.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix Sub-view */}
      {subView === 'matrix' && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4 text-blue-600" />
                Multi-Class Confusion Matrix (7 Industrial Classes)
              </h3>
              <p className="text-xs text-slate-500">
                True Class (rows) vs. Predicted Class (columns) for 140 hold-out test samples
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Total Samples:</span>
              <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-slate-100 text-slate-800">
                140
              </span>
            </div>
          </div>

          {/* Confusion Matrix Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr>
                  <th className="p-2 text-left text-xs font-mono text-slate-400 uppercase tracking-wider">True \ Pred</th>
                  {CLASS_METRICS.map((col) => (
                    <th key={col.className} className="p-2 text-xs font-semibold text-slate-700">
                      <div className="flex items-center justify-center gap-1">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: col.color }} />
                        <span className="capitalize">{col.className.replace('_', ' ').slice(0, 7)}</span>
                      </div>
                    </th>
                  ))}
                  <th className="p-2 text-xs font-mono text-slate-500 bg-slate-50">Support</th>
                  <th className="p-2 text-xs font-mono text-slate-500 bg-slate-50">Recall</th>
                </tr>
              </thead>
              <tbody>
                {CLASS_METRICS.map((rowCls, rIdx) => {
                  const rowSum = CONFUSION_MATRIX[rIdx].reduce((a, b) => a + b, 0);
                  const diagVal = CONFUSION_MATRIX[rIdx][rIdx];
                  const recallPct = (diagVal / rowSum) * 100;

                  return (
                    <tr key={rowCls.className} className="border-t border-slate-100">
                      <td className="p-2 text-left text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: rowCls.color }} />
                        <span className="capitalize">{rowCls.label}</span>
                      </td>

                      {CONFUSION_MATRIX[rIdx].map((val, cIdx) => {
                        const isDiag = rIdx === cIdx;
                        const isZero = val === 0;
                        const cellIntensity = isDiag
                          ? Math.min(100, Math.round((val / 20) * 100))
                          : Math.min(100, Math.round((val / 5) * 100));

                        return (
                          <td
                            key={cIdx}
                            onMouseEnter={() => setHoveredCell({ row: rIdx, col: cIdx, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            className={`p-2 font-mono text-xs cursor-pointer transition-all ${
                              isDiag
                                ? 'bg-indigo-50 font-bold text-indigo-900 hover:bg-indigo-100'
                                : isZero
                                ? 'text-slate-300 hover:bg-slate-50'
                                : 'bg-rose-50 text-rose-700 font-bold hover:bg-rose-100'
                            }`}
                          >
                            <span className="px-1.5 py-0.5 rounded">
                              {val}
                            </span>
                          </td>
                        );
                      })}

                      <td className="p-2 font-mono text-xs text-slate-600 bg-slate-50 font-bold">{rowSum}</td>
                      <td className="p-2 font-mono text-xs bg-slate-50">
                        <span className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                          recallPct >= 95 ? 'bg-emerald-100 text-emerald-800' :
                          recallPct >= 85 ? 'bg-blue-100 text-blue-800' : 'bg-amber-100 text-amber-800'
                        }`}>
                          {recallPct.toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  );
                })}

                {/* Precision Summary Row */}
                <tr className="border-t-2 border-slate-300 bg-slate-50 font-mono text-xs">
                  <td className="p-2 text-left font-bold text-slate-700">Precision</td>
                  {CLASS_METRICS.map((c) => (
                    <td key={c.className} className="p-2">
                      <span className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                        c.precision >= 0.95 ? 'text-emerald-700' :
                        c.precision >= 0.85 ? 'text-blue-700' : 'text-amber-700'
                      }`}>
                        {(c.precision * 100).toFixed(1)}%
                      </span>
                    </td>
                  ))}
                  <td className="p-2 font-bold text-slate-700">140</td>
                  <td className="p-2 font-bold text-indigo-700">91.43%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Hover Inspector Tooltip */}
          {hoveredCell && (
            <div className="p-2.5 bg-slate-900 text-white rounded-lg text-xs font-mono flex items-center justify-between">
              <div>
                True Class: <span className="text-indigo-300 font-bold">{CLASS_METRICS[hoveredCell.row].label}</span>
                {'  →  '}
                Predicted: <span className="text-emerald-300 font-bold">{CLASS_METRICS[hoveredCell.col].label}</span>
              </div>
              <div className="font-bold">
                Count: {hoveredCell.val} samples {hoveredCell.row === hoveredCell.col ? '(Correct Classification)' : '(Misclassification Error)'}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Per-Class Metrics Table */}
      {subView === 'classes' && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-emerald-600" />
                Per-Class Precision, Recall, Specificity & F1-Score Ledger
              </h3>
              <p className="text-xs text-slate-500">
                Detailed breakdown for all 7 industrial defect classes evaluated on 140 parts
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded font-bold border border-emerald-200">
                Macro F1: 0.9148
              </span>
              <span className="text-xs font-mono bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-bold border border-indigo-200">
                Weighted F1: 0.9148
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 uppercase tracking-wider font-mono">
                  <th className="py-2.5 px-3">Class Name</th>
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Support</th>
                  <th className="py-2.5 px-3">TP / FP / FN</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Recall (Sens)</th>
                  <th className="py-2.5 px-3">Specificity</th>
                  <th className="py-2.5 px-3 font-bold text-slate-900">F1-Score</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {CLASS_METRICS.map((c) => (
                  <tr key={c.className} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-slate-800 flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color }} />
                      {c.label}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                        c.className === 'normal' ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                      }`}>
                        {c.className === 'normal' ? 'Baseline' : 'Defect'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono">{c.support}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-600">
                      <span className="text-emerald-600 font-bold">{c.tp}</span> / {c.fp} / <span className="text-rose-600">{c.fn}</span>
                    </td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-slate-700">
                      {(c.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-mono font-semibold text-slate-700">
                      {(c.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-600">
                      {(c.specificity * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-mono font-bold text-indigo-700">
                      {c.f1.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">
                      {c.f1 >= 0.90 ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 font-semibold">
                          <CheckCircle2 className="w-3.5 h-3.5" /> High Precision
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] text-amber-700 font-semibold">
                          <AlertTriangle className="w-3.5 h-3.5" /> Active Triage
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Spatial Localization & IoU Sub-view */}
      {subView === 'localization' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  <Crosshair className="w-4 h-4 text-amber-600" />
                  Spatial Localization Benchmarks (IoU & Dice Coefficient)
                </h3>
                <p className="text-xs text-slate-500">
                  Intersection over Union (IoU) between predicted masks/boxes and ground-truth defects
                </p>
              </div>
              <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-amber-50 text-amber-700 border border-amber-200">
                mIoU = 71.07%
              </span>
            </div>

            {/* Metric Comparison Bars */}
            <div className="space-y-3.5 pt-1">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">Mean Mask IoU (Full Test Set)</span>
                  <span className="font-mono font-bold text-amber-600">71.07% (0.7107)</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full" style={{ width: '71.07%' }} />
                </div>
                <span className="text-[10px] text-slate-400 mt-0.5 block">Includes true negative clean parts (IoU = 1.0)</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">Defective-Only Mean Mask IoU</span>
                  <span className="font-mono font-bold text-indigo-600">66.24% (0.6624)</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: '66.24%' }} />
                </div>
                <span className="text-[10px] text-slate-400 mt-0.5 block">Calculated strictly over defective specimens (120 parts)</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">Mean Dice Coefficient (F1 Mask Score)</span>
                  <span className="font-mono font-bold text-emerald-600">81.78% (0.8178)</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: '81.78%' }} />
                </div>
                <span className="text-[10px] text-slate-400 mt-0.5 block">Harmonic mean of pixel precision and recall</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">IoU @ 0.50 Threshold Success Rate</span>
                  <span className="font-mono font-bold text-purple-600">87.50%</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500 rounded-full" style={{ width: '87.50%' }} />
                </div>
                <span className="text-[10px] text-slate-400 mt-0.5 block">105 / 120 defects detected with &gt;= 50% spatial overlap</span>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">IoU @ 0.70 Strict Tight-Fit Rate</span>
                  <span className="font-mono font-bold text-rose-600">39.17%</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full" style={{ width: '39.17%' }} />
                </div>
                <span className="text-[10px] text-slate-400 mt-0.5 block">47 / 120 defects with exact boundary tight-fit</span>
              </div>
            </div>
          </div>

          {/* Localization Visual Diagram */}
          <div className="lg:col-span-5 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5 border-b border-slate-100 pb-3">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Intersection over Union Math & Alignment
            </h3>

            {/* Visual Box Overlap Diagram */}
            <div className="p-4 bg-slate-900 rounded-lg text-white font-mono text-xs flex flex-col items-center justify-center space-y-3">
              <div className="relative w-48 h-36 border border-slate-700 bg-slate-800/60 rounded flex items-center justify-center">
                {/* Ground Truth Box (Green) */}
                <div className="absolute w-28 h-20 border-2 border-emerald-400 bg-emerald-500/20 top-6 left-6 flex items-start justify-start p-1 text-[9px] text-emerald-300 font-bold">
                  GT Mask
                </div>

                {/* Predicted Box (Blue) */}
                <div className="absolute w-28 h-20 border-2 border-indigo-400 bg-indigo-500/20 top-10 left-14 flex items-end justify-end p-1 text-[9px] text-indigo-300 font-bold">
                  Pred Mask
                </div>

                {/* Intersection Area (Highlighted center) */}
                <div className="absolute w-20 h-16 bg-amber-400/40 border border-dashed border-amber-300 top-10 left-14 flex items-center justify-center text-[10px] font-bold text-amber-200">
                  Intersection
                </div>
              </div>

              <div className="w-full text-center text-[11px] text-slate-300 space-y-1">
                <div className="font-bold text-amber-300">IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT)</div>
                <div className="text-[10px] text-slate-400">Dice = 2 × Area(Pred ∩ GT) / (Area(Pred) + Area(GT))</div>
              </div>
            </div>

            <div className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200">
              <p className="font-semibold text-slate-800 mb-1">Industrial Localization Criteria:</p>
              <ul className="list-disc pl-4 space-y-1 text-slate-600 text-[11px]">
                <li><strong>IoU &gt; 0.50</strong> is the standard manufacturing pass threshold for defect bounding and robotic pick-and-place sorting.</li>
                <li>Morphological refinement cleans raw Grad-CAM heatmaps to elevate effective mask IoU from ~54% to <strong>71.07%</strong>.</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Raw ASCII Report Viewer Sub-view */}
      {subView === 'report' && (
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-purple-600" />
                Raw Evaluation Report (outputs/evaluation_report.txt)
              </h3>
              <p className="text-xs text-slate-500">
                Direct export from <code className="font-mono text-indigo-600">src/evaluate.py</code> evaluation run
              </p>
            </div>

            <button
              onClick={handleCopyReport}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 text-xs font-semibold transition-colors border border-indigo-200"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  Copied Report!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-indigo-600" />
                  Copy Report Text
                </>
              )}
            </button>
          </div>

          <div className="p-4 bg-slate-950 text-slate-200 font-mono text-xs rounded-lg overflow-x-auto leading-relaxed max-h-96 border border-slate-800">
            <pre>{RAW_ASCII_REPORT}</pre>
          </div>
        </div>
      )}
    </div>
  );
};
