export interface SubHeaderItem {
  title: string;
  description: string;
  badge?: string;
  iconName: string;
  tag?: string;
}

export interface PageHeaderItem {
  id: 'inspector' | 'keras' | 'preprocessing' | 'dataset' | 'evaluation' | 'code' | 'guide';
  title: string;
  shortTitle: string;
  tagline: string;
  iconName: string;
  category: string;
  statusBadge: string;
  accentColor: string;
  badgeBg: string;
  badgeBorder: string;
  badgeText: string;
  primaryHeader: string;
  keyMetric: string;
  metricLabel: string;
  subHeaders: SubHeaderItem[];
}

export const ALL_PAGE_HEADERS: PageHeaderItem[] = [
  {
    id: 'inspector',
    title: 'Live Defect Inspector & Spatial Localization',
    shortTitle: 'Live Inspector',
    tagline: 'Real-time multi-task inspection: classification, Grad-CAM localization, millimeter sizing & severity classification.',
    iconName: 'Layers',
    category: 'Computer Vision Inference',
    statusBadge: 'Multi-Task Active',
    accentColor: 'text-emerald-600',
    badgeBg: 'bg-emerald-50',
    badgeBorder: 'border-emerald-200',
    badgeText: 'text-emerald-700',
    primaryHeader: 'Industrial Defect Inspection & Localization',
    keyMetric: '12ms',
    metricLabel: 'Inference Latency',
    subHeaders: [
      {
        title: 'Specimen Selection & High-Resolution Samples',
        description: '7 canonical test specimens with ground-truth defect masks and CAD dimension envelopes.',
        badge: '1024×1024',
        iconName: 'Eye',
        tag: 'Specimen Lab',
      },
      {
        title: 'Multi-Task Defect Classification & Quality Gate',
        description: 'Predicts defect class, confidence score, anomaly index, and acceptable vs rejected state.',
        badge: 'τ = 0.50',
        iconName: 'ShieldCheck',
        tag: 'Quality Gate',
      },
      {
        title: 'Grad-CAM Heatmap & Morphological Localization',
        description: 'Class-activation mapping refined with Otsu thresholding and morphological cleanup.',
        badge: 'CAM + Mask',
        iconName: 'Crosshair',
        tag: 'Grad-CAM',
      },
      {
        title: 'Metric Measurement & Severity Matrix',
        description: 'Calculates physical length, width, pixel area, and risk rating (CRITICAL, MAJOR, MODERATE).',
        badge: 'Physical mm',
        iconName: 'Ruler',
        tag: 'Dimensional',
      },
    ],
  },
  {
    id: 'keras',
    title: 'TensorFlow / Keras Defect Classifier',
    shortTitle: 'TensorFlow / Keras',
    tagline: 'Production-ready transfer learning backbone based on EfficientNetB0 with dual-phase training.',
    iconName: 'Cpu',
    category: 'Deep Learning Model',
    statusBadge: 'EfficientNetB0 (ImageNet)',
    accentColor: 'text-amber-600',
    badgeBg: 'bg-amber-50',
    badgeBorder: 'border-amber-200',
    badgeText: 'text-amber-700',
    primaryHeader: 'TensorFlow / Keras Defect Classifier (EfficientNetB0)',
    keyMetric: '95.7%',
    metricLabel: 'Top-1 Accuracy',
    subHeaders: [
      {
        title: 'Inference Engine & Quality Gate Confidence Gate',
        description: 'Enforces 60% confidence gate threshold; flags low-confidence predictions as Uncertain.',
        badge: 'Conf ≥ 0.60',
        iconName: 'Target',
        tag: 'Gate Threshold',
      },
      {
        title: 'Transfer Learning Backbone Architecture',
        description: 'EfficientNetB0 feature extractor with global average pooling, dropout, and Dense head.',
        badge: '224×224 RGB',
        iconName: 'Cpu',
        tag: 'Backbone',
      },
      {
        title: 'Dual-Phase Fine-Tuning Schedule',
        description: 'Phase 1 frozen backbone (1e-3 LR) + Phase 2 top 20 layers unfreezing (1e-4 LR).',
        badge: '2-Phase Fit',
        iconName: 'Sparkles',
        tag: 'Fine-Tuning',
      },
      {
        title: 'YAML Training Configuration Ingestion',
        description: 'Dynamically reads image_size, batch_size, epochs, splits, and quality rules from training_config.yaml.',
        badge: 'YAML Config',
        iconName: 'Sliders',
        tag: 'Hyperparameters',
      },
    ],
  },
  {
    id: 'preprocessing',
    title: 'Industrial Image Preprocessing Pipeline',
    shortTitle: 'Preprocessing Pipeline',
    tagline: 'Standardized 4-stage optical pipeline: LAB CLAHE normalization, bilateral denoising, letterboxing.',
    iconName: 'Sliders',
    category: 'Optical Conditioning',
    statusBadge: '4-Stage Pipeline',
    accentColor: 'text-indigo-600',
    badgeBg: 'bg-indigo-50',
    badgeBorder: 'border-indigo-200',
    badgeText: 'text-indigo-700',
    primaryHeader: 'Industrial Image Preprocessing & Contrast Enhancement',
    keyMetric: '4 Stages',
    metricLabel: 'Optical Conditioning',
    subHeaders: [
      {
        title: 'LAB Color Space CLAHE Normalization',
        description: 'Contrast Limited Adaptive Histogram Equalization applied exclusively to the luminance channel.',
        badge: 'L-Channel',
        iconName: 'SunMedium',
        tag: 'Histogram',
      },
      {
        title: 'Bilateral Edge-Preserving Denoising',
        description: 'Smooths surface machining grain while strictly preserving sharp micro-fracture boundaries.',
        badge: 'Edge-Preserving',
        iconName: 'Sparkles',
        tag: 'Denoising',
      },
      {
        title: 'Aspect-Ratio Preserved Letterboxing',
        description: 'Resizes any rectangular workpiece to 224×224 square tensor with neutral zero padding.',
        badge: '224×224 Tensor',
        iconName: 'Crop',
        tag: 'Letterboxing',
      },
      {
        title: 'Interactive Image Comparison & Stage Inspection',
        description: 'Real-time interactive split slider comparing raw input versus fully preprocessed output.',
        badge: 'A/B Split',
        iconName: 'SplitSquareVertical',
        tag: 'Visual Split',
      },
    ],
  },
  {
    id: 'dataset',
    title: 'Dataset Distribution & Taxonomy Analytics',
    shortTitle: 'Dataset & Statistics',
    tagline: 'Comprehensive taxonomy breakdown across 7 industrial classes with synthetic generator studio.',
    iconName: 'BarChart3',
    category: 'Dataset Intelligence',
    statusBadge: '7-Class Taxonomy',
    accentColor: 'text-blue-600',
    badgeBg: 'bg-blue-50',
    badgeBorder: 'border-blue-200',
    badgeText: 'text-blue-700',
    primaryHeader: 'Dataset Distribution & Taxonomy Analytics',
    keyMetric: '140 Parts',
    metricLabel: 'Curated Specimens',
    subHeaders: [
      {
        title: '7 Canonical Manufacturing Defect Classes',
        description: 'Normal, Crack, Scratch, Dent, Stain, Discoloration, and Dimensional Irregularity.',
        badge: '7 Classes',
        iconName: 'Tags',
        tag: 'Taxonomy',
      },
      {
        title: 'Class Distribution & Imbalance Analytics',
        description: 'Visual distribution bars, train/val/test splits (70% / 15% / 15%), and support metrics.',
        badge: 'Stratified',
        iconName: 'PieChart',
        tag: 'Distributions',
      },
      {
        title: 'Synthetic Defect Augmentation Studio',
        description: 'Procedural generator for physics-based metallic fractures, gouges, and lubricant stains.',
        badge: 'Generator',
        iconName: 'Wand2',
        tag: 'Synthetic',
      },
      {
        title: 'MVTec Anomaly Benchmark Downloader',
        description: 'Automated ingestion pipelines for official MVTec industrial anomaly detection datasets.',
        badge: 'MVTec Gold',
        iconName: 'DownloadCloud',
        tag: 'Benchmarks',
      },
    ],
  },
  {
    id: 'evaluation',
    title: 'Quality Assurance & Quantitative Evaluation',
    shortTitle: 'Model Evaluation',
    tagline: 'Benchmark scorecard: ROC curves (0.9967), 7×7 confusion matrix, IoU localization & training history.',
    iconName: 'Activity',
    category: 'Benchmarking & QA',
    statusBadge: 'Benchmarked PASS',
    accentColor: 'text-rose-600',
    badgeBg: 'bg-rose-50',
    badgeBorder: 'border-rose-200',
    badgeText: 'text-rose-700',
    primaryHeader: 'Quality Assurance & Quantitative Model Evaluation',
    keyMetric: '0.9967',
    metricLabel: 'ROC-AUC Score',
    subHeaders: [
      {
        title: 'ROC & Precision-Recall Performance Curves',
        description: 'Receiver Operating Characteristic (ROC-AUC 0.9967) and Precision-Recall (PR-AUC 0.9994).',
        badge: 'AUC 0.9967',
        iconName: 'TrendingUp',
        tag: 'ROC & PR',
      },
      {
        title: '7×7 Multi-Class Confusion Matrix',
        description: 'Full confusion cross-tabulation detailing per-class True Positives, False Positives, and escapes.',
        badge: '7×7 Grid',
        iconName: 'Grid3X3',
        tag: 'Matrix',
      },
      {
        title: 'Per-Class Precision, Recall, Specificity & F1-Score',
        description: 'Tabulated classification metrics with interactive threshold simulation slider (τ = 0.10–0.90).',
        badge: 'F1 91.48%',
        iconName: 'Award',
        tag: 'F1 Scoreboard',
      },
      {
        title: 'Spatial Defect Localization (IoU & Dice)',
        description: 'Mean Mask IoU (71.07%), Defective-only IoU (66.24%), and IoU@0.50 pass rates.',
        badge: 'IoU 71.07%',
        iconName: 'Crosshair',
        tag: 'IoU & Mask',
      },
      {
        title: 'Training History (Epochs vs Loss & Accuracy)',
        description: 'Reads training_history.json, rendering table of 50 epochs versus loss, accuracy, and delta.',
        badge: '50 Epochs',
        iconName: 'History',
        tag: 'Epoch Table',
      },
      {
        title: 'Raw ASCII Audit Report Export',
        description: 'Standardized CLI evaluation scorecard export from src/evaluate.py ready for regulatory audits.',
        badge: 'Audit Ready',
        iconName: 'FileText',
        tag: 'ASCII Export',
      },
    ],
  },
  {
    id: 'code',
    title: 'Project Source Code & Architecture Explorer',
    shortTitle: 'Project Files',
    tagline: 'Interactive source code repository viewer covering Python vision pipeline, training scripts & configs.',
    iconName: 'Code2',
    category: 'Engineering Architecture',
    statusBadge: 'Complete Source Tree',
    accentColor: 'text-purple-600',
    badgeBg: 'bg-purple-50',
    badgeBorder: 'border-purple-200',
    badgeText: 'text-purple-700',
    primaryHeader: 'Project Source Code & Architecture Explorer',
    keyMetric: '18 Files',
    metricLabel: 'Engineered Modules',
    subHeaders: [
      {
        title: 'src/ Core Computer Vision & Inference Modules',
        description: 'pipeline.py, preprocessing.py, keras_model.py, defect_classifier.py, localization.py, app.py.',
        badge: 'src/',
        iconName: 'FolderGit2',
        tag: 'Core Python',
      },
      {
        title: 'training/ TensorFlow Model Training Pipeline',
        description: 'train_keras.py, evaluate_keras.py, transfer learning scripts with YAML parser integration.',
        badge: 'training/',
        iconName: 'Cpu',
        tag: 'Training Scripts',
      },
      {
        title: 'training_config.yaml & Environment Declarations',
        description: 'Central configuration defining image_size: 224, batch_size: 32, epochs: 50, and quality gates.',
        badge: 'YAML Config',
        iconName: 'FileCode',
        tag: 'Hyperparameters',
      },
      {
        title: 'Data & Model Artifact Repositories',
        description: 'data/test_samples, models/keras/, and outputs/keras/ classification reports and confusion matrices.',
        badge: 'Artifacts',
        iconName: 'Archive',
        tag: 'Checkpoints',
      },
    ],
  },
  {
    id: 'guide',
    title: 'Inspectra AI Deployment & CLI Quick Start Guide',
    shortTitle: 'Quick Start Guide',
    tagline: 'End-to-end command reference: Python virtualenv installation, model training, single-image CLI & API.',
    iconName: 'Terminal',
    category: 'Developer Documentation',
    statusBadge: 'CLI & REST Ready',
    accentColor: 'text-teal-600',
    badgeBg: 'bg-teal-50',
    badgeBorder: 'border-teal-200',
    badgeText: 'text-teal-700',
    primaryHeader: 'Inspectra AI Deployment & CLI Quick Start Guide',
    keyMetric: '4 Steps',
    metricLabel: 'Deployment Runbook',
    subHeaders: [
      {
        title: 'Step 1: Environment & Dependencies Installation',
        description: 'Setup Python virtualenv and install TensorFlow, PyYAML, OpenCV, and Scikit-Learn.',
        badge: 'pip install',
        iconName: 'Terminal',
        tag: 'Virtualenv',
      },
      {
        title: 'Step 2: Model Training with Custom YAML Config',
        description: 'Run python training/train_keras.py --config training_config.yaml to train and evaluate.',
        badge: 'CLI Command',
        iconName: 'Play',
        tag: 'Train CLI',
      },
      {
        title: 'Step 3: Single-Image Inference Script',
        description: 'Execute python scripts/test_keras_image.py <image_path> for terminal quality gate verdicts.',
        badge: 'scripts/',
        iconName: 'FileCode',
        tag: 'Inference CLI',
      },
      {
        title: 'Step 4: Full-Stack Flask & Vite REST Endpoints',
        description: 'Documentation for /api/keras/inspect, /api/keras/evaluation, and /api/keras/training-history.',
        badge: 'REST APIs',
        iconName: 'Globe',
        tag: 'HTTP Endpoints',
      },
    ],
  },
];
