/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header } from './components/Header';
import { InspectorTab } from './components/InspectorTab';
import { PreprocessingTab } from './components/PreprocessingTab';
import { DatasetStatsTab } from './components/DatasetStatsTab';
import { EvaluationTab } from './components/EvaluationTab';
import { CodeExplorerTab } from './components/CodeExplorerTab';
import { QuickStartGuideTab } from './components/QuickStartGuideTab';

export default function App() {
  const [activeTab, setActiveTab] = useState<'inspector' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide'>('inspector');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans antialiased">
      {/* Top Application Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'inspector' && <InspectorTab />}
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

