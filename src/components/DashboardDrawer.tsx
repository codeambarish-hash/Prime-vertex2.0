import React, { useState } from 'react';
import {
  LayoutDashboard,
  X,
  Layers,
  Cpu,
  Sliders,
  BarChart3,
  Activity,
  Code2,
  Terminal,
  ShieldCheck,
  Search,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  ListFilter,
  Bookmark
} from 'lucide-react';
import { ALL_PAGE_HEADERS, PageHeaderItem } from '../data/pageHeaders';

interface DashboardDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: 'dashboard' | 'inspector' | 'keras' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide';
  setActiveTab: (tab: 'dashboard' | 'inspector' | 'keras' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide') => void;
}

const getIcon = (iconName: string, className: string = 'w-5 h-5') => {
  switch (iconName) {
    case 'Layers':
      return <Layers className={className} />;
    case 'Cpu':
      return <Cpu className={className} />;
    case 'Sliders':
      return <Sliders className={className} />;
    case 'BarChart3':
      return <BarChart3 className={className} />;
    case 'Activity':
      return <Activity className={className} />;
    case 'Code2':
      return <Code2 className={className} />;
    case 'Terminal':
      return <Terminal className={className} />;
    default:
      return <ShieldCheck className={className} />;
  }
};

const getSubHeaderIcon = (iconName: string, className: string = 'w-3.5 h-3.5') => {
  switch (iconName) {
    case 'Eye':
      return <Layers className={className} />;
    case 'ShieldCheck':
      return <ShieldCheck className={className} />;
    case 'Crosshair':
      return <Layers className={className} />;
    case 'Sliders':
      return <Sliders className={className} />;
    case 'Cpu':
      return <Cpu className={className} />;
    case 'BarChart3':
      return <BarChart3 className={className} />;
    case 'Activity':
      return <Activity className={className} />;
    case 'Code2':
      return <Code2 className={className} />;
    case 'Terminal':
      return <Terminal className={className} />;
    default:
      return <CheckCircle2 className={className} />;
  }
};

export const DashboardDrawer: React.FC<DashboardDrawerProps> = ({
  isOpen,
  onClose,
  activeTab,
  setActiveTab,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  if (!isOpen) return null;

  const categories = ['all', ...Array.from(new Set(ALL_PAGE_HEADERS.map((h) => h.category)))];

  const filteredHeaders = ALL_PAGE_HEADERS.filter((item) => {
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    const term = searchTerm.toLowerCase().trim();
    if (!term) return matchesCategory;

    const matchesTitle = item.title.toLowerCase().includes(term);
    const matchesPrimary = item.primaryHeader.toLowerCase().includes(term);
    const matchesTagline = item.tagline.toLowerCase().includes(term);
    const matchesSubheaders = item.subHeaders.some(
      (sh) => sh.title.toLowerCase().includes(term) || sh.description.toLowerCase().includes(term)
    );

    return matchesCategory && (matchesTitle || matchesPrimary || matchesTagline || matchesSubheaders);
  });

  const handleSelectTab = (tabId: PageHeaderItem['id']) => {
    setActiveTab(tabId);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex"
      id="dashboard-drawer-overlay"
      aria-modal="true"
      role="dialog"
    >
      {/* Backdrop overlay */}
      <div
        className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
        onClick={onClose}
      />

      {/* Slide-out Panel from Top-Left */}
      <div className="relative w-full max-w-2xl bg-white shadow-2xl flex flex-col h-full z-10 animate-in slide-in-from-left duration-250 border-r border-slate-200">
        {/* Drawer Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-50/80 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md">
              <LayoutDashboard className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-900">
                  Inspectra AI Dashboard
                </h2>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
                  All Page Headers
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Directory of all modules, primary headers, and functional components in the application
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-200/70 transition-colors"
            aria-label="Close dashboard"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Quick Search & Category Filter */}
        <div className="p-4 border-b border-slate-100 bg-white space-y-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search across all headers, pipelines, and sub-sections..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
              autoFocus
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600"
              >
                Clear
              </button>
            )}
          </div>

          {/* Categories Pill Bar */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs no-scrollbar">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 rounded-md font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === cat
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                }`}
              >
                {cat === 'all' ? 'All Headers (7)' : cat}
              </button>
            ))}
          </div>
        </div>

        {/* Headers List / Dashboard Grid */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-slate-50/40">
          <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
            <span className="font-semibold uppercase tracking-wider text-[10px] text-slate-400">
              Web Page Sections ({filteredHeaders.length} Available)
            </span>
            <span className="text-[11px]">Click any card to jump directly</span>
          </div>

          {filteredHeaders.length > 0 ? (
            filteredHeaders.map((item) => {
              const isActive = activeTab === item.id;

              return (
                <div
                  key={item.id}
                  onClick={() => handleSelectTab(item.id)}
                  className={`group bg-white rounded-xl border p-4 transition-all duration-150 cursor-pointer shadow-2xs hover:shadow-md ${
                    isActive
                      ? 'border-indigo-500 ring-2 ring-indigo-500/10'
                      : 'border-slate-200 hover:border-slate-300'
                  }`}
                >
                  {/* Top Bar of Card */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center space-x-3">
                      <div
                        className={`w-9 h-9 rounded-lg flex items-center justify-center ${item.badgeBg} ${item.accentColor} border ${item.badgeBorder}`}
                      >
                        {getIcon(item.iconName, 'w-5 h-5')}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                            {item.primaryHeader}
                          </h3>
                          {isActive && (
                            <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-indigo-600 text-white flex items-center gap-1 shadow-2xs">
                              <CheckCircle2 className="w-3 h-3" /> ACTIVE
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] font-medium text-slate-400 block uppercase tracking-wider mt-0.5">
                          {item.category} • {item.shortTitle}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <span
                        className={`px-2 py-0.5 text-[11px] font-semibold rounded-md border ${item.badgeBg} ${item.badgeText} ${item.badgeBorder}`}
                      >
                        {item.statusBadge}
                      </span>
                      <div className="w-6 h-6 rounded-md bg-slate-50 group-hover:bg-indigo-50 flex items-center justify-center text-slate-400 group-hover:text-indigo-600 transition-colors">
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </div>
                  </div>

                  {/* Tagline */}
                  <p className="text-xs text-slate-600 mt-2.5 leading-relaxed">
                    {item.tagline}
                  </p>

                  {/* Sub-headers grid */}
                  <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {item.subHeaders.map((sub, sIdx) => (
                      <div
                        key={sIdx}
                        className="bg-slate-50/80 hover:bg-slate-100/80 p-2.5 rounded-lg border border-slate-100 flex items-start justify-between gap-2 transition-colors"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span className={`p-0.5 rounded ${item.badgeBg} ${item.accentColor} shrink-0`}>
                              {getSubHeaderIcon(sub.iconName, 'w-3 h-3')}
                            </span>
                            <span className="font-semibold text-[11px] text-slate-800 truncate">
                              {sub.title}
                            </span>
                          </div>
                          <div className="text-[10px] text-slate-500 line-clamp-1">
                            {sub.description}
                          </div>
                        </div>
                        {sub.badge && (
                          <span className="text-[9px] font-mono font-medium px-1.5 py-0.5 rounded bg-white text-indigo-700 border border-slate-200 shrink-0">
                            {sub.badge}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Jump Button Footer */}
                  <div className="mt-3 flex items-center justify-between text-xs text-indigo-600 font-semibold group-hover:translate-x-0.5 transition-transform pt-1">
                    <span className="text-[11px] text-slate-400 font-normal">
                      Click anywhere to switch to this section
                    </span>
                    <span className="flex items-center gap-1">
                      Open Section <ArrowRight className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-center py-12 bg-white rounded-xl border border-slate-200 text-slate-500 space-y-2">
              <Search className="w-8 h-8 text-slate-300 mx-auto" />
              <p className="text-xs font-semibold text-slate-700">No page headers match &ldquo;{searchTerm}&rdquo;</p>
              <p className="text-xs text-slate-400">Try searching for &ldquo;Keras&rdquo;, &ldquo;CLAHE&rdquo;, &ldquo;Evaluation&rdquo;, or &ldquo;Loss&rdquo;</p>
              <button
                onClick={() => { setSearchTerm(''); setSelectedCategory('all'); }}
                className="mt-2 text-xs font-bold text-indigo-600 hover:text-indigo-800"
              >
                Reset Filters
              </button>
            </div>
          )}
        </div>

        {/* Drawer Footer */}
        <div className="p-4 border-t border-slate-200 bg-white flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="font-medium text-slate-700">7 Active Web Page Headers</span>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
          >
            Close Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};
