import React, { useState } from 'react';
import { BarChart3, Database, FileSpreadsheet, PieChart, CheckCircle2, AlertCircle, Layers } from 'lucide-react';
import { DefectClass } from '../types/inspection';

interface ClassStat {
  name: DefectClass;
  label: string;
  count: number;
  pct: number;
  color: string;
  isDefect: boolean;
}

const DATASET_CLASSES: ClassStat[] = [
  { name: 'normal', label: 'Normal (Pristine)', count: 50, pct: 14.3, color: '#10b981', isDefect: false },
  { name: 'crack', label: 'Crack', count: 50, pct: 14.3, color: '#ef4444', isDefect: true },
  { name: 'scratch', label: 'Scratch', count: 50, pct: 14.3, color: '#f97316', isDefect: true },
  { name: 'dent', label: 'Dent', count: 50, pct: 14.3, color: '#f59e0b', isDefect: true },
  { name: 'stain', label: 'Stain', count: 50, pct: 14.3, color: '#a855f7', isDefect: true },
  { name: 'discoloration', label: 'Discoloration', count: 50, pct: 14.3, color: '#3b82f6', isDefect: true },
  { name: 'dimensional_irregularity', label: 'Dimensional Irreg.', count: 50, pct: 14.3, color: '#ec4899', isDefect: true },
];

export const DatasetStatsTab: React.FC = () => {
  const [selectedSplit, setSelectedSplit] = useState<'all' | 'train' | 'val' | 'test'>('all');

  const totalImages = 350;
  const trainCount = 245; // 70%
  const valCount = 52;   // 15%
  const testCount = 53;  // 15%
  const normalCount = 50;
  const defectiveCount = 300;

  return (
    <div className="space-y-6">
      {/* Top Stat Highlight Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Total Samples</span>
            <Database className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-black text-slate-900">{totalImages}</span>
            <span className="text-xs text-slate-500">labeled images</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">Standardized 256x256x3 RGB</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Defect vs Normal</span>
            <PieChart className="w-4 h-4 text-blue-600" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-black text-slate-900">{defectiveCount} / {normalCount}</span>
            <span className="text-xs text-rose-600 font-semibold">85.7% Anomaly</span>
          </div>
          <span className="text-[11px] text-emerald-600 mt-1 block">14.3% Pristine Baseline</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Mask Coverage Area</span>
            <BarChart3 className="w-4 h-4 text-purple-600" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-black text-slate-900">3.84%</span>
            <span className="text-xs text-slate-500">mean surface</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">Min: 0.35% | Max: 12.60%</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Ground Truth Masks</span>
            <Layers className="w-4 h-4 text-amber-600" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-black text-slate-900">300</span>
            <span className="text-xs text-slate-500">pixel masks</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">100% paired with bounding boxes</span>
        </div>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Class Distribution Bar Chart (7 cols) */}
        <div className="lg:col-span-7 bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Defect Taxonomy Distribution
              </h3>
              <p className="text-xs text-slate-500">
                Balanced across baseline normal and all 6 manufacturing defect categories
              </p>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-bold">
              7 Classes
            </span>
          </div>

          <div className="space-y-3 pt-2">
            {DATASET_CLASSES.map((cls) => (
              <div key={cls.name} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: cls.color }}
                    />
                    <span className="font-semibold text-slate-800">{cls.label}</span>
                    <span className="text-[10px] text-slate-400 uppercase font-mono">
                      {cls.isDefect ? 'Anomaly' : 'Baseline'}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-slate-600">{cls.count} imgs</span>
                    <span className="font-bold text-slate-900 font-mono w-10 text-right">
                      {cls.pct}%
                    </span>
                  </div>
                </div>
                {/* Progress bar */}
                <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${cls.pct * 4}%`, // scaled for visual prominence
                      backgroundColor: cls.color,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Tabular summary snippet */}
          <div className="pt-3 border-t border-slate-100 text-xs text-slate-600 flex items-center justify-between">
            <span>Stratified Splitting: 70% Train / 15% Validation / 15% Test</span>
            <span className="font-mono text-slate-400">Seed: 42</span>
          </div>
        </div>

        {/* Right: Split Breakdown & Directory Tree (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Split Ratio Box */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900">
              Dataset Partition Splits
            </h3>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[11px] font-bold text-slate-500 uppercase block">Train</span>
                <span className="text-lg font-black text-slate-900">{trainCount}</span>
                <span className="text-[10px] text-slate-400 block">70.0%</span>
              </div>

              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[11px] font-bold text-slate-500 uppercase block">Validation</span>
                <span className="text-lg font-black text-slate-900">{valCount}</span>
                <span className="text-[10px] text-slate-400 block">15.0%</span>
              </div>

              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[11px] font-bold text-slate-500 uppercase block">Test</span>
                <span className="text-lg font-black text-slate-900">{testCount}</span>
                <span className="text-[10px] text-slate-400 block">15.0%</span>
              </div>
            </div>

            {/* Split Visualization Bar */}
            <div className="w-full h-3 rounded-md overflow-hidden flex">
              <div className="bg-emerald-500 h-full" style={{ width: '70%' }} title="Train 70%" />
              <div className="bg-blue-500 h-full" style={{ width: '15%' }} title="Val 15%" />
              <div className="bg-amber-500 h-full" style={{ width: '15%' }} title="Test 15%" />
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" /> Train</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /> Validation</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500" /> Test</span>
            </div>
          </div>

          {/* Directory Hierarchy Card */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-2xs space-y-3">
            <h3 className="text-sm font-bold text-slate-900">
              Processed Dataset Disk Structure
            </h3>
            <div className="p-3 bg-slate-900 text-slate-200 font-mono text-xs rounded-lg overflow-x-auto leading-relaxed">
              <div>data/processed/</div>
              <div className="text-slate-400">├── images/</div>
              <div className="text-slate-400">│   ├── train/  <span className="text-emerald-400">(245 .png)</span></div>
              <div className="text-slate-400">│   ├── val/    <span className="text-blue-400">(52 .png)</span></div>
              <div className="text-slate-400">│   └── test/   <span className="text-amber-400">(53 .png)</span></div>
              <div className="text-slate-400">├── masks/</div>
              <div className="text-slate-400">│   ├── train/  <span className="text-emerald-400">(245 _mask.png)</span></div>
              <div className="text-slate-400">│   ├── val/    <span className="text-blue-400">(52 _mask.png)</span></div>
              <div className="text-slate-400">│   └── test/   <span className="text-amber-400">(53 _mask.png)</span></div>
              <div>├── annotations.json</div>
              <div>└── dataset_stats.json</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
