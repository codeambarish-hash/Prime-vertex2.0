import React, { useState } from 'react';
import { 
  Sparkles, 
  Download, 
  ExternalLink, 
  ZoomIn, 
  CheckCircle2, 
  AlertTriangle, 
  Crosshair, 
  Layers, 
  Eye, 
  ShieldCheck, 
  Activity, 
  Maximize2, 
  X,
  FileText,
  Sliders,
  Check
} from 'lucide-react';
import { HighResSpecimen, DefectClass, SubstrateType } from '../types/inspection';
import catalogData from '../data/highres_samples_catalog.json';

interface SampleAssetsGalleryProps {
  onLoadIntoInspector?: (substrate: SubstrateType, defect: DefectClass) => void;
}

export const SampleAssetsGallery: React.FC<SampleAssetsGalleryProps> = ({ onLoadIntoInspector }) => {
  const specimens: HighResSpecimen[] = catalogData.specimens as HighResSpecimen[];
  const [selectedSpecimen, setSelectedSpecimen] = useState<HighResSpecimen>(specimens[1]); // Default to crack
  const [viewMode, setViewMode] = useState<'raw' | 'annotated' | 'mask'>('annotated');
  const [filterClass, setFilterClass] = useState<string>('all');
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredSpecimens = filterClass === 'all' 
    ? specimens 
    : specimens.filter(s => filterClass === 'defects' ? s.is_defective : s.defect_class === filterClass);

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-rose-100 text-rose-800 border-rose-200';
      case 'MAJOR':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      case 'MODERATE':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
    }
  };

  const getCurrentImageUrl = (specimen: HighResSpecimen) => {
    switch (viewMode) {
      case 'raw':
        return specimen.files.raw;
      case 'mask':
        return specimen.files.mask;
      case 'annotated':
      default:
        return specimen.files.annotated;
    }
  };

  const handleCopyPath = (path: string, id: string) => {
    navigator.clipboard?.writeText(path);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const mapSubstrateToSimulator = (subStr: string): SubstrateType => {
    if (subStr.includes('cast_iron')) return 'cast_iron';
    if (subStr.includes('machined')) return 'machined_part';
    return 'brushed_metal';
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-8" id="sample-assets-showcase">
      {/* Header bar */}
      <div className="p-5 sm:p-6 border-b border-slate-200 bg-gradient-to-r from-slate-900 to-slate-800 text-white">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-400/30 flex items-center justify-center text-indigo-300">
                <Sparkles className="w-4 h-4 text-indigo-400" />
              </div>
              <h2 className="text-lg sm:text-xl font-bold tracking-tight text-white">
                High-Resolution Synthetic Industrial Defect Assets (1024×1024)
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Pipeline Calibrated
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl">
              Procedurally synthesized 1024×1024 industrial test specimens with authentic metallic micro-grain, physically accurate surface defects (cracks, scratches, oil stains, impact dents), ground truth masks, and CAD/HUD telemetry.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-300 hidden sm:inline">Format:</span>
            <span className="px-2 py-1 bg-slate-800 text-slate-200 text-xs font-mono rounded border border-slate-700">
              1024×1024 RGB PNG
            </span>
            <span className="px-2 py-1 bg-slate-800 text-slate-200 text-xs font-mono rounded border border-slate-700">
              Zero-Noise Binary Mask
            </span>
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-2 mt-4 pt-3 border-t border-slate-700/60 overflow-x-auto pb-1 text-xs">
          <span className="text-slate-400 text-xs font-medium mr-1 whitespace-nowrap">Defect Filter:</span>
          {[
            { id: 'all', label: 'All 7 Specimens' },
            { id: 'defects', label: 'Defects Only' },
            { id: 'crack', label: 'Metallic Crack' },
            { id: 'scratch', label: 'Abrasive Scratch' },
            { id: 'stain', label: 'Oil Stain' },
            { id: 'dent', label: 'Impact Dent' },
            { id: 'discoloration', label: 'Thermal Tint' },
            { id: 'dimensional_irregularity', label: 'Edge Notch' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilterClass(tab.id)}
              className={`px-3 py-1 rounded-md font-medium whitespace-nowrap transition-colors ${
                filterClass === tab.id
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* Left: Specimen Selector List */}
        <div className="lg:col-span-4 p-4 border-r border-slate-200 bg-slate-50/60 max-h-[620px] overflow-y-auto space-y-2.5">
          <div className="flex items-center justify-between text-xs text-slate-500 px-1 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[10px]">Test Asset Catalog</span>
            <span>{filteredSpecimens.length} items</span>
          </div>

          {filteredSpecimens.map((item) => {
            const isSelected = selectedSpecimen.id === item.id;
            return (
              <div
                key={item.id}
                onClick={() => setSelectedSpecimen(item)}
                className={`group p-3 rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-white border-indigo-500 shadow-sm ring-2 ring-indigo-500/10'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:shadow-2xs'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="relative w-16 h-16 rounded-lg overflow-hidden border border-slate-200 bg-slate-100 shrink-0">
                    <img
                      src={item.files.annotated}
                      alt={item.title}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                    <div className="absolute top-0.5 right-0.5">
                      {item.is_defective ? (
                        <span className="w-2 h-2 rounded-full bg-rose-500 block" />
                      ) : (
                        <span className="w-2 h-2 rounded-full bg-emerald-500 block" />
                      )}
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${getSeverityBadge(item.severity)}`}>
                        {item.defect_class.replace('_', ' ')}
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">1024px</span>
                    </div>
                    <h4 className="text-xs font-bold text-slate-900 mt-1 truncate group-hover:text-indigo-600 transition-colors">
                      {item.title}
                    </h4>
                    <div className="flex items-center gap-2 mt-1 text-[11px] text-slate-500">
                      <span>Conf: {(item.confidence * 100).toFixed(1)}%</span>
                      {item.is_defective && (
                        <>
                          <span>•</span>
                          <span>{item.dimensions_mm.length}×{item.dimensions_mm.width}mm</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: High-Res Specimen Inspection Viewport */}
        <div className="lg:col-span-8 p-5 sm:p-6 flex flex-col justify-between bg-white">
          <div>
            {/* Viewport Header Controls */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
              <div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider border ${getSeverityBadge(selectedSpecimen.severity)}`}>
                    {selectedSpecimen.defect_class.replace('_', ' ')}
                  </span>
                  <h3 className="text-base font-bold text-slate-900">
                    {selectedSpecimen.title}
                  </h3>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  Substrate: <span className="font-semibold text-slate-700 capitalize">{selectedSpecimen.substrate.replace(/_/g, ' ')}</span> • Resolution: <span className="font-mono text-slate-700">1024×1024</span> • Target Metric: <span className="font-mono text-slate-700">{selectedSpecimen.dimensions_mm.length}mm × {selectedSpecimen.dimensions_mm.width}mm</span>
                </p>
              </div>

              {/* View mode toggle pills */}
              <div className="flex items-center p-1 bg-slate-100 rounded-lg border border-slate-200 self-start sm:self-auto text-xs">
                <button
                  type="button"
                  onClick={() => setViewMode('annotated')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                    viewMode === 'annotated'
                      ? 'bg-white text-slate-900 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Crosshair className="w-3.5 h-3.5 text-indigo-600" />
                  CAD / HUD Telemetry
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('raw')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                    viewMode === 'raw'
                      ? 'bg-white text-slate-900 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5 text-emerald-600" />
                  Raw Specimen
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('mask')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                    viewMode === 'mask'
                      ? 'bg-white text-slate-900 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5 text-purple-600" />
                  Binary Mask
                </button>
              </div>
            </div>

            {/* Specimen Visual Canvas Area */}
            <div className="relative mt-4 bg-slate-900 rounded-xl overflow-hidden border border-slate-800 group shadow-inner">
              <div className="aspect-square max-h-[460px] w-full flex items-center justify-center overflow-hidden relative">
                <img
                  src={getCurrentImageUrl(selectedSpecimen)}
                  alt={`${selectedSpecimen.title} - ${viewMode}`}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-contain"
                />

                {/* Corner reticle marks */}
                <div className="absolute top-3 left-3 flex items-center gap-2 bg-slate-900/80 backdrop-blur-xs px-2.5 py-1 rounded border border-slate-700/80 text-[10px] text-slate-300 font-mono pointer-events-none">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>1024×1024 PX</span>
                  <span>•</span>
                  <span>VIEW: {viewMode.toUpperCase()}</span>
                </div>

                <div className="absolute top-3 right-3 flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setIsModalOpen(true)}
                    className="p-1.5 rounded-md bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/80 transition-colors"
                    title="Expand Full Resolution"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>

                {/* Bounding box coordinate tag */}
                {selectedSpecimen.is_defective && viewMode === 'annotated' && (
                  <div className="absolute bottom-3 left-3 bg-slate-900/85 backdrop-blur-xs px-2.5 py-1.5 rounded border border-slate-700 text-[11px] text-slate-200 font-mono space-x-2">
                    <span className="text-indigo-400">BBOX:</span>
                    <span>[{selectedSpecimen.bbox.join(', ')}]</span>
                    <span className="text-slate-500">|</span>
                    <span className="text-emerald-400">AREA:</span>
                    <span>{selectedSpecimen.defect_pixels.toLocaleString()} px²</span>
                  </div>
                )}
              </div>
            </div>

            {/* Specimen Telemetry Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
              <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">AI Confidence</span>
                <span className="text-base font-bold text-slate-900">
                  {(selectedSpecimen.confidence * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] text-emerald-600 block mt-0.5">High Confidence</span>
              </div>

              <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Anomaly Score</span>
                <span className={`text-base font-bold ${selectedSpecimen.anomaly_score > 0.5 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {selectedSpecimen.anomaly_score.toFixed(3)}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  {selectedSpecimen.anomaly_score > 0.5 ? 'Defect Alert' : 'Normal Substrate'}
                </span>
              </div>

              <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Physical Extent</span>
                <span className="text-base font-bold text-slate-900">
                  {selectedSpecimen.is_defective ? `${selectedSpecimen.dimensions_mm.length} mm` : 'None'}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  {selectedSpecimen.is_defective ? `Width: ${selectedSpecimen.dimensions_mm.width} mm` : 'Pristine Surface'}
                </span>
              </div>

              <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Pipeline Gate</span>
                <span className={`text-xs font-bold px-2 py-1 rounded inline-block mt-0.5 ${
                  selectedSpecimen.is_defective ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {selectedSpecimen.is_defective ? 'CONFIRMED REJECT' : 'CLEAN PASS'}
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-600 mt-3 bg-slate-50 p-3 rounded-lg border border-slate-200/80 leading-relaxed">
              <span className="font-semibold text-slate-800">Physical Assessment: </span>
              {selectedSpecimen.description}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 mt-4 border-t border-slate-200">
            <div className="flex items-center gap-2">
              <a
                href={selectedSpecimen.files.raw}
                download={`${selectedSpecimen.id}_raw.png`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 transition-colors shadow-xs"
              >
                <Download className="w-3.5 h-3.5" />
                Download Raw PNG (1024px)
              </a>
              <a
                href={selectedSpecimen.files.mask}
                download={`${selectedSpecimen.id}_mask.png`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 transition-colors"
              >
                <Download className="w-3.5 h-3.5 text-purple-600" />
                Binary Mask PNG
              </a>
            </div>

            <div className="flex items-center gap-2">
              {onLoadIntoInspector && (
                <button
                  type="button"
                  onClick={() => {
                    onLoadIntoInspector(
                      mapSubstrateToSimulator(selectedSpecimen.substrate),
                      selectedSpecimen.defect_class
                    );
                  }}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 hover:bg-indigo-100 transition-colors"
                >
                  <Sliders className="w-3.5 h-3.5" />
                  Load into Simulation Controls
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Full-Resolution Modal View */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-900/90 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-slate-800 max-w-4xl w-full overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between text-white">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <span>{selectedSpecimen.title}</span>
                  <span className="text-xs font-mono text-slate-400 font-normal">1024×1024 High-Resolution View</span>
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 overflow-auto flex items-center justify-center bg-black/40">
              <img
                src={getCurrentImageUrl(selectedSpecimen)}
                alt={selectedSpecimen.title}
                referrerPolicy="no-referrer"
                className="max-h-[65vh] w-auto object-contain rounded border border-slate-800 shadow-lg"
              />
            </div>

            <div className="p-4 border-t border-slate-800 bg-slate-950 flex flex-wrap items-center justify-between text-xs text-slate-300 gap-3">
              <div className="flex items-center gap-4 font-mono text-[11px]">
                <span>FILE: {selectedSpecimen.id}.png</span>
                <span>SUBSTRATE: {selectedSpecimen.substrate}</span>
                <span>SEVERITY: {selectedSpecimen.severity}</span>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={getCurrentImageUrl(selectedSpecimen)}
                  download={`${selectedSpecimen.id}_${viewMode}.png`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-500 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Save {viewMode.toUpperCase()} Image
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
