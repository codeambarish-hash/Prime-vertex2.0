import React, { useRef } from 'react';
import {
  LayoutDashboard,
  ShieldCheck,
  Layers,
  BarChart3,
  Code2,
  Terminal,
  Sliders,
  Activity,
  Cpu,
  ImagePlus
} from 'lucide-react';

export type TabType = 'dashboard' | 'inspector' | 'keras' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide';

interface HeaderProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  onOpenDashboard: () => void;
  onImageUpload?: (file: File, dataUrl: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, onOpenDashboard, onImageUpload }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const dataUrl = event.target?.result as string;
        if (onImageUpload) {
          onImageUpload(file, dataUrl);
        } else {
          setActiveTab('keras');
        }
      };
      reader.readAsDataURL(file);
    }
  };
  return (
    <header className="border-b border-slate-200 bg-white shadow-xs sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between py-3.5 gap-3">
          
          {/* Left Side Top Corner: Dashboard Launcher & Header Icons */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* Dashboard Icon in Left-Side Top Corner */}
            <button
              id="dashboard-top-left-trigger"
              onClick={onOpenDashboard}
              className="flex items-center gap-2 p-2 sm:px-3 sm:py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs shadow-sm hover:shadow-md transition-all group shrink-0 cursor-pointer"
              title="Dashboard: Display All Page Headers"
              aria-label="Open Dashboard (All Page Headers)"
            >
              <LayoutDashboard className="w-5 h-5 group-hover:rotate-6 transition-transform" />
              <span className="hidden sm:inline font-semibold">Dashboard</span>
              <span className="hidden md:inline px-1.5 py-0.2 text-[10px] rounded-full bg-indigo-500/60 font-mono">
                7 Headers
              </span>
            </button>

            {/* Quick Header Icons Strip in Left Top Corner */}
            <div className="hidden xl:flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200" title="Quick Header Shortcuts">
              <button
                onClick={() => setActiveTab('inspector')}
                title="Header: Live Inspector"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'inspector' ? 'bg-white text-emerald-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Layers className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('keras')}
                title="Header: TensorFlow / Keras Pipeline"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'keras' ? 'bg-white text-amber-500 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Cpu className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('preprocessing')}
                title="Header: Preprocessing Pipeline"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'preprocessing' ? 'bg-white text-indigo-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Sliders className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('dataset')}
                title="Header: Dataset & Statistics"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'dataset' ? 'bg-white text-blue-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <BarChart3 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('evaluation')}
                title="Header: Model Evaluation & QA"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'evaluation' ? 'bg-white text-rose-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Activity className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('code')}
                title="Header: Project Source Code"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'code' ? 'bg-white text-purple-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Code2 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('guide')}
                title="Header: Quick Start Guide"
                className={`p-1.5 rounded-md transition-colors ${activeTab === 'guide' ? 'bg-white text-teal-600 shadow-xs' : 'text-slate-500 hover:text-slate-900'}`}
              >
                <Terminal className="w-4 h-4" />
              </button>
              <button
                id="header-quick-upload-icon-btn"
                onClick={handleUploadClick}
                title="Upload Image for Inspection"
                className="p-1.5 rounded-md text-amber-600 hover:text-amber-700 hover:bg-amber-50 transition-colors"
              >
                <ImagePlus className="w-4 h-4" />
              </button>
            </div>

            <div className="h-6 w-[1px] bg-slate-200 hidden xl:block" />

            {/* App Title & Branding */}
            <div className="flex items-center space-x-2.5">
              <div className="w-9 h-9 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600 shadow-2xs">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight leading-none">
                    Inspectra AI
                  </h1>
                  <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Online
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 hidden sm:block mt-0.5">
                  Visual Inspection & Deep Learning Architecture
                </p>
              </div>
            </div>
          </div>

          {/* Main Navigation Tabs & Action Bar */}
          <div className="flex items-center gap-2.5 self-start md:self-auto flex-wrap sm:flex-nowrap">
            {/* Hidden Input for Header Upload Button */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.webp,image/*"
              onChange={handleFileChange}
              className="hidden"
              id="header-hidden-file-input"
            />

            <nav className="flex items-center space-x-1 p-1 bg-slate-100 rounded-lg border border-slate-200 overflow-x-auto text-xs font-medium">
            <button
              id="tab-dashboard"
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'dashboard'
                  ? 'bg-white text-indigo-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <LayoutDashboard className="w-3.5 h-3.5 text-indigo-600" />
              Dashboard
            </button>

            <button
              id="tab-inspector"
              onClick={() => setActiveTab('inspector')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'inspector'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-emerald-600" />
              Live Inspector
            </button>

            <button
              id="tab-keras"
              onClick={() => setActiveTab('keras')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'keras'
                  ? 'bg-white text-slate-900 shadow-xs font-bold ring-1 ring-amber-400'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Cpu className="w-3.5 h-3.5 text-amber-500" />
              TensorFlow / Keras
            </button>

            <button
              id="tab-preprocessing"
              onClick={() => setActiveTab('preprocessing')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'preprocessing'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Sliders className="w-3.5 h-3.5 text-indigo-600" />
              Preprocessing
            </button>

            <button
              id="tab-dataset"
              onClick={() => setActiveTab('dataset')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'dataset'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5 text-blue-600" />
              Dataset
            </button>

            <button
              id="tab-evaluation"
              onClick={() => setActiveTab('evaluation')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'evaluation'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Activity className="w-3.5 h-3.5 text-rose-600" />
              Evaluation
            </button>

            <button
              id="tab-code"
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'code'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Code2 className="w-3.5 h-3.5 text-purple-600" />
              Files
            </button>

            <button
              id="tab-guide"
              onClick={() => setActiveTab('guide')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                activeTab === 'guide'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <Terminal className="w-3.5 h-3.5 text-teal-600" />
              Guide
            </button>
          </nav>

          {/* Prominent Header Image Upload Icon & Button */}
          <button
            id="header-upload-image-btn"
            onClick={handleUploadClick}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-600 active:bg-amber-700 text-white font-bold text-xs shadow-xs hover:shadow-sm transition-all whitespace-nowrap cursor-pointer shrink-0 group"
            title="Upload Image for Inspection"
            aria-label="Upload Image for Inspection"
          >
            <ImagePlus className="w-4 h-4 group-hover:scale-110 transition-transform" />
            <span className="hidden sm:inline">Upload Image</span>
          </button>
        </div>
        </div>
      </div>
    </header>
  );
};

