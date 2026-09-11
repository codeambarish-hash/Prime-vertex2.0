import React, { useState } from 'react';
import {
  LayoutDashboard,
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
  ChevronRight,
  TrendingUp,
  FolderTree,
  LayoutGrid,
  Award,
  Eye,
  Crosshair,
  Ruler,
  Target,
  SunMedium,
  Crop,
  SplitSquareVertical,
  Tags,
  PieChart,
  Wand2,
  DownloadCloud,
  Grid3X3,
  History,
  FileText,
  FolderGit2,
  FileCode,
  Archive,
  Play,
  Globe,
  Filter,
  ExternalLink
} from 'lucide-react';
import { ALL_PAGE_HEADERS, PageHeaderItem, SubHeaderItem } from '../data/pageHeaders';

interface DashboardOverviewTabProps {
  setActiveTab: (tab: 'inspector' | 'keras' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide') => void;
}

const getMainIcon = (iconName: string, className: string = 'w-6 h-6') => {
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

const getSubHeaderIcon = (iconName: string, className: string = 'w-4 h-4') => {
  switch (iconName) {
    case 'Eye':
      return <Eye className={className} />;
    case 'ShieldCheck':
      return <ShieldCheck className={className} />;
    case 'Crosshair':
      return <Crosshair className={className} />;
    case 'Ruler':
      return <Ruler className={className} />;
    case 'Target':
      return <Target className={className} />;
    case 'Cpu':
      return <Cpu className={className} />;
    case 'Sparkles':
      return <Sparkles className={className} />;
    case 'Sliders':
      return <Sliders className={className} />;
    case 'SunMedium':
      return <SunMedium className={className} />;
    case 'Crop':
      return <Crop className={className} />;
    case 'SplitSquareVertical':
      return <SplitSquareVertical className={className} />;
    case 'Tags':
      return <Tags className={className} />;
    case 'PieChart':
      return <PieChart className={className} />;
    case 'Wand2':
      return <Wand2 className={className} />;
    case 'DownloadCloud':
      return <DownloadCloud className={className} />;
    case 'TrendingUp':
      return <TrendingUp className={className} />;
    case 'Grid3X3':
      return <Grid3X3 className={className} />;
    case 'Award':
      return <Award className={className} />;
    case 'History':
      return <History className={className} />;
    case 'FileText':
      return <FileText className={className} />;
    case 'FolderGit2':
      return <FolderGit2 className={className} />;
    case 'FileCode':
      return <FileCode className={className} />;
    case 'Archive':
      return <Archive className={className} />;
    case 'Terminal':
      return <Terminal className={className} />;
    case 'Play':
      return <Play className={className} />;
    case 'Globe':
      return <Globe className={className} />;
    default:
      return <CheckCircle2 className={className} />;
  }
};

export const DashboardOverviewTab: React.FC<DashboardOverviewTabProps> = ({ setActiveTab }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = ['all', ...Array.from(new Set(ALL_PAGE_HEADERS.map((h) => h.category)))];

  const filteredHeaders = ALL_PAGE_HEADERS.filter((item) => {
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    const term = searchTerm.toLowerCase().trim();
    if (!term) return matchesCategory;

    const matchesTitle = item.title.toLowerCase().includes(term);
    const matchesPrimary = item.primaryHeader.toLowerCase().includes(term);
    const matchesTagline = item.tagline.toLowerCase().includes(term);
    const matchesSub = item.subHeaders.some(
      (sh) =>
        sh.title.toLowerCase().includes(term) ||
        sh.description.toLowerCase().includes(term) ||
        (sh.tag && sh.tag.toLowerCase().includes(term))
    );

    return matchesCategory && (matchesTitle || matchesPrimary || matchesTagline || matchesSub);
  });

  return (
    <div className="space-y-6" id="dashboard-overview-page">
      {/* Top Architecture Banner */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-2xs space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="flex items-start sm:items-center space-x-4">
            <div className="w-12 h-12 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shrink-0">
              <LayoutDashboard className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">
                  Application Architecture & Header Directory
                </h1>
                <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-600" /> Formatted Cards View
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1 max-w-2xl">
                Comprehensive blueprint of all web page headers, feature modules, and technical pipelines with descriptive iconography, KPI chips, and direct navigation links.
              </p>
            </div>
          </div>

          {/* Formatted Top KPI Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full lg:w-auto">
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center sm:text-left flex flex-col justify-between">
              <div className="flex items-center justify-center sm:justify-start gap-1.5 text-slate-500 mb-1">
                <LayoutGrid className="w-3.5 h-3.5 text-indigo-600" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Headers</span>
              </div>
              <div className="text-lg font-black text-slate-900">7 Modules</div>
              <div className="text-[10px] text-slate-400">Primary UI views</div>
            </div>

            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center sm:text-left flex flex-col justify-between">
              <div className="flex items-center justify-center sm:justify-start gap-1.5 text-slate-500 mb-1">
                <FolderTree className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Sections</span>
              </div>
              <div className="text-lg font-black text-emerald-600">28 Cards</div>
              <div className="text-[10px] text-slate-400">Descriptive sub-headers</div>
            </div>

            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center sm:text-left flex flex-col justify-between">
              <div className="flex items-center justify-center sm:justify-start gap-1.5 text-slate-500 mb-1">
                <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Coverage</span>
              </div>
              <div className="text-lg font-black text-blue-600">100%</div>
              <div className="text-[10px] text-slate-400">End-to-end pipeline</div>
            </div>

            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200 text-center sm:text-left flex flex-col justify-between">
              <div className="flex items-center justify-center sm:justify-start gap-1.5 text-slate-500 mb-1">
                <Award className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-[10px] font-bold uppercase tracking-wider">Accuracy</span>
              </div>
              <div className="text-lg font-black text-amber-600">0.9967</div>
              <div className="text-[10px] text-slate-400">ROC-AUC metric</div>
            </div>
          </div>
        </div>

        {/* Search & Category Filter Toolbar */}
        <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search across all headers, cards, or parameters..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
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

          {/* Category Filter Pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto text-xs pb-1 sm:pb-0">
            <span className="text-slate-400 text-[11px] font-semibold flex items-center gap-1 shrink-0 mr-1">
              <Filter className="w-3.5 h-3.5" /> Filter:
            </span>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === cat
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                }`}
              >
                {cat === 'all' ? 'All Modules (7)' : cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Formatted Grid of All Page Headers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredHeaders.map((item) => (
          <div
            key={item.id}
            className="group bg-white rounded-2xl border border-slate-200 hover:border-indigo-300 p-5 sm:p-6 shadow-2xs hover:shadow-md transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              {/* Top Bar of Header Card */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start space-x-3.5">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center ${item.badgeBg} ${item.accentColor} border ${item.badgeBorder} shadow-2xs shrink-0`}
                  >
                    {getMainIcon(item.iconName, 'w-6 h-6')}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                        {item.category}
                      </span>
                      <span className="text-slate-300">•</span>
                      <span className="text-[10px] font-mono font-semibold text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                        /{item.id}
                      </span>
                    </div>
                    <h2 className="text-base sm:text-lg font-bold text-slate-900 group-hover:text-indigo-600 transition-colors mt-0.5">
                      {item.primaryHeader}
                    </h2>
                  </div>
                </div>

                {/* Key Metric & Status Chip */}
                <div className="flex flex-col items-end gap-1.5 shrink-0">
                  <span
                    className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${item.badgeBg} ${item.badgeText} ${item.badgeBorder}`}
                  >
                    {item.statusBadge}
                  </span>
                  <span className="text-[10px] text-slate-500 font-medium flex items-center gap-1">
                    <span className="font-bold text-slate-800">{item.keyMetric}</span> {item.metricLabel}
                  </span>
                </div>
              </div>

              {/* Tagline Description */}
              <p className="text-xs text-slate-600 mt-3 leading-relaxed">
                {item.tagline}
              </p>

              {/* Formatted Sub-Header Cards Section */}
              <div className="mt-5 pt-4 border-t border-slate-100 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <FolderTree className="w-3.5 h-3.5 text-slate-400" />
                    Sub-Headers & Architectural Components ({item.subHeaders.length})
                  </span>
                  <span className="text-[10px] text-slate-400 font-medium">
                    Detailed specs & icons
                  </span>
                </div>

                {/* Grid of Formatted Cards for Each Sub-Header */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {item.subHeaders.map((sub, sIdx) => (
                    <div
                      key={sIdx}
                      className="bg-slate-50/80 hover:bg-slate-100/80 p-3 rounded-xl border border-slate-100 transition-all duration-150 flex flex-col justify-between gap-2"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-1.5">
                          <div className="flex items-center gap-2 min-w-0">
                            <span
                              className={`p-1 rounded-md ${item.badgeBg} ${item.accentColor} border ${item.badgeBorder} shrink-0`}
                            >
                              {getSubHeaderIcon(sub.iconName, 'w-3.5 h-3.5')}
                            </span>
                            <span className="text-xs font-bold text-slate-800 truncate">
                              {sub.title}
                            </span>
                          </div>
                        </div>

                        <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2">
                          {sub.description}
                        </p>
                      </div>

                      {/* Micro-badge & Technical Tag */}
                      <div className="flex items-center justify-between pt-1 border-t border-slate-200/60 text-[10px]">
                        {sub.tag && (
                          <span className="font-semibold text-slate-600">
                            {sub.tag}
                          </span>
                        )}
                        {sub.badge && (
                          <span className="font-mono font-semibold px-1.5 py-0.5 rounded bg-white text-indigo-700 border border-indigo-100 text-[10px] shadow-2xs ml-auto">
                            {sub.badge}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Formatted Card Action Footer */}
            <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Ready for inspection</span>
              </div>

              <button
                onClick={() => setActiveTab(item.id)}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 text-xs font-bold transition-all group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-600 cursor-pointer shadow-2xs"
              >
                <span>Open {item.shortTitle}</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
