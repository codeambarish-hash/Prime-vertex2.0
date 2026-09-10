import React, { useState } from 'react';
import { Terminal, Copy, Check, Play, BookOpen, Layers, CheckCircle2 } from 'lucide-react';

export const QuickStartGuideTab: React.FC = () => {
  const [selectedMode, setSelectedMode] = useState<'synthetic' | 'mvtec'>('synthetic');
  const [numSamples, setNumSamples] = useState<number>(50);
  const [imgSize, setImgSize] = useState<number>(256);
  const [mvtecCategory, setMvtecCategory] = useState<string>('metal_nut');
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const buildCommand = () => {
    if (selectedMode === 'synthetic') {
      return `python main.py --mode synthetic --num-samples ${numSamples} --img-size ${imgSize}`;
    } else {
      return `python main.py --mode mvtec --mvtec-category ${mvtecCategory} --img-size ${imgSize}`;
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Interactive CLI Command Builder */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Terminal className="w-5 h-5 text-emerald-600" />
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Interactive Execution Command Generator
            </h3>
            <p className="text-xs text-slate-500">
              Configure parameters to run dataset generation, statistical profiling, and inspection verification
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Mode Switch */}
          <div>
            <label className="text-xs font-bold text-slate-700 block mb-1.5">
              Dataset Mode (--mode)
            </label>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-100 rounded-lg">
              <button
                onClick={() => setSelectedMode('synthetic')}
                className={`py-1.5 text-xs font-medium rounded-md transition-colors ${
                  selectedMode === 'synthetic'
                    ? 'bg-white text-slate-900 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Synthetic (OpenCV)
              </button>
              <button
                onClick={() => setSelectedMode('mvtec')}
                className={`py-1.5 text-xs font-medium rounded-md transition-colors ${
                  selectedMode === 'mvtec'
                    ? 'bg-white text-slate-900 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                MVTec AD Benchmark
              </button>
            </div>
          </div>

          {/* Num Samples / Category */}
          {selectedMode === 'synthetic' ? (
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-1.5">
                Samples Per Class (--num-samples)
              </label>
              <select
                value={numSamples}
                onChange={(e) => setNumSamples(Number(e.target.value))}
                className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900"
              >
                <option value={20}>20 per class (140 total)</option>
                <option value={50}>50 per class (350 total - default)</option>
                <option value={100}>100 per class (700 total)</option>
                <option value={200}>200 per class (1,400 total)</option>
              </select>
            </div>
          ) : (
            <div>
              <label className="text-xs font-bold text-slate-700 block mb-1.5">
                MVTec Category (--mvtec-category)
              </label>
              <select
                value={mvtecCategory}
                onChange={(e) => setMvtecCategory(e.target.value)}
                className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900"
              >
                <option value="metal_nut">metal_nut (textures & thread defects)</option>
                <option value="tile">tile (cracks, discoloration, scratches)</option>
                <option value="bottle">bottle (broken, contamination)</option>
                <option value="screw">screw (bent, scratch, thread)</option>
                <option value="capsule">capsule (crack, squeeze, scratch)</option>
                <option value="cable">cable (bent, cut, poke)</option>
              </select>
            </div>
          )}

          {/* Image Size */}
          <div>
            <label className="text-xs font-bold text-slate-700 block mb-1.5">
              Image Size (--img-size)
            </label>
            <select
              value={imgSize}
              onChange={(e) => setImgSize(Number(e.target.value))}
              className="w-full text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-900"
            >
              <option value={256}>256 x 256 px (Optimal Speed/Memory)</option>
              <option value={512}>512 x 512 px (High Resolution Inspection)</option>
            </select>
          </div>
        </div>

        {/* Generated Terminal Box */}
        <div className="bg-slate-950 p-3.5 rounded-lg flex items-center justify-between font-mono text-xs text-slate-200 border border-slate-800">
          <div className="flex items-center gap-2 overflow-x-auto pr-4">
            <span className="text-emerald-400 font-bold">$</span>
            <span className="text-emerald-200">{buildCommand()}</span>
          </div>
          <button
            onClick={() => handleCopy(buildCommand(), 'builder')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs shrink-0 transition-colors"
          >
            {copiedCmd === 'builder' ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Step-by-Step Production Guide */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
          <div className="w-7 h-7 rounded-full bg-emerald-100 text-emerald-800 font-bold flex items-center justify-center text-xs">
            1
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Environment & Dependencies
          </h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Install OpenCV, PyTorch, Albumentations, and scikit-learn pinned in requirements.txt:
          </p>
          <div className="bg-slate-950 p-2.5 rounded font-mono text-[11px] text-slate-200 flex items-center justify-between">
            <code className="truncate">pip install -r requirements.txt</code>
            <button
              onClick={() => handleCopy('pip install -r requirements.txt', 'step1')}
              className="text-slate-400 hover:text-white shrink-0 ml-2"
            >
              <Copy className="w-3 h-3" />
            </button>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
          <div className="w-7 h-7 rounded-full bg-blue-100 text-blue-800 font-bold flex items-center justify-center text-xs">
            2
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Generate Data & Print Stats
          </h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Run the generator to create train/val/test splits and view tabular class statistics:
          </p>
          <div className="bg-slate-950 p-2.5 rounded font-mono text-[11px] text-slate-200 flex items-center justify-between">
            <code className="truncate">python main.py --mode synthetic</code>
            <button
              onClick={() => handleCopy('python main.py --mode synthetic', 'step2')}
              className="text-slate-400 hover:text-white shrink-0 ml-2"
            >
              <Copy className="w-3 h-3" />
            </button>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
          <div className="w-7 h-7 rounded-full bg-purple-100 text-purple-800 font-bold flex items-center justify-center text-xs">
            3
          </div>
          <h4 className="text-sm font-bold text-slate-900">
            Interactive Notebook Exploration
          </h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Inspect ground truth pixel masks and overlay heatmaps in Jupyter:
          </p>
          <div className="bg-slate-950 p-2.5 rounded font-mono text-[11px] text-slate-200 flex items-center justify-between">
            <code className="truncate">jupyter notebook notebooks/</code>
            <button
              onClick={() => handleCopy('jupyter notebook notebooks/01_dataset_exploration.ipynb', 'step3')}
              className="text-slate-400 hover:text-white shrink-0 ml-2"
            >
              <Copy className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
