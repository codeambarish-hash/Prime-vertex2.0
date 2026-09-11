/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header, TabType } from './components/Header';
import { DashboardDrawer } from './components/DashboardDrawer';
import { DashboardOverviewTab } from './components/DashboardOverviewTab';
import { InspectorTab } from './components/InspectorTab';
import { KerasInspectionTab } from './components/KerasInspectionTab';
import { PreprocessingTab } from './components/PreprocessingTab';
import { DatasetStatsTab } from './components/DatasetStatsTab';
import { EvaluationTab } from './components/EvaluationTab';
import { CodeExplorerTab } from './components/CodeExplorerTab';
import { QuickStartGuideTab } from './components/QuickStartGuideTab';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('keras');
  const [isDashboardDrawerOpen, setIsDashboardDrawerOpen] = useState(false);
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedFileObj, setUploadedFileObj] = useState<File | null>(null);

  const handleImageUpload = (file: File, dataUrl: string) => {
    setUploadedFileObj(file);
    setUploadedFileName(file.name);
    setUploadedImageSrc(dataUrl);
    setActiveTab('keras');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans antialiased">
      {/* Top Application Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenDashboard={() => setIsDashboardDrawerOpen(true)}
        onImageUpload={handleImageUpload}
      />

      {/* Slide-out Dashboard Drawer from Top-Left */}
      <DashboardDrawer
        isOpen={isDashboardDrawerOpen}
        onClose={() => setIsDashboardDrawerOpen(false)}
        activeTab={activeTab}
        setActiveTab={(t) => {
          setActiveTab(t);
          setIsDashboardDrawerOpen(false);
        }}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <DashboardOverviewTab setActiveTab={(t) => setActiveTab(t)} />
        )}
        {activeTab === 'inspector' && <InspectorTab />}
        {activeTab === 'keras' && (
          <KerasInspectionTab
            initialFile={uploadedFileObj}
            initialImageSrc={uploadedImageSrc}
            initialFileName={uploadedFileName}
          />
        )}
        {activeTab === 'preprocessing' && <PreprocessingTab />}
        {activeTab === 'dataset' && <DatasetStatsTab />}
        {activeTab === 'evaluation' && <EvaluationTab />}
        {activeTab === 'code' && <CodeExplorerTab />}
        {activeTab === 'guide' && <QuickStartGuideTab />}
      </main>

      {/* Industrial Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
            <span className="font-semibold text-slate-700">Inspectra AI</span>
            <span>•</span>
            <span>AI Powered Visual Inspection Pipeline</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>LAB CLAHE Normalization</span>
            <span>•</span>
            <span>Bilateral Denoising</span>
            <span>•</span>
            <span>Aspect Letterboxing</span>
            <span>•</span>
            <span>Albumentations Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

