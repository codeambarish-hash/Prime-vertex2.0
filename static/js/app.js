/**
 * Inspectra AI — Web Inspection Studio Client Controller
 * Pure Vanilla JavaScript (Zero External Dependencies)
 */

(function () {
  'use strict';

  // Configuration & Constraints
  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp'];
  const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/x-ms-bmp'];

  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileTray = document.getElementById('fileTray');
  const fileChips = document.getElementById('fileChips');
  const fileCount = document.getElementById('fileCount');
  const clearFilesBtn = document.getElementById('clearFilesBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const loadSampleBtn = document.getElementById('loadSampleBtn');
  const uploadForm = document.getElementById('uploadForm');

  const errorBanner = document.getElementById('errorBanner');
  const errorTitle = document.getElementById('errorTitle');
  const errorMessage = document.getElementById('errorMessage');
  const closeErrorBtn = document.getElementById('closeErrorBtn');

  const loadingCard = document.getElementById('loadingCard');
  const loadingStage = document.getElementById('loadingStage');

  const singleResultPanel = document.getElementById('singleResultPanel');
  const singleStatusBadge = document.getElementById('singleStatusBadge');
  const singleCategoryPill = document.getElementById('singleCategoryPill');
  const singleFilename = document.getElementById('singleFilename');
  const singleLatencyTag = document.getElementById('singleLatencyTag');
  const singleStatusDecision = document.getElementById('singleStatusDecision');
  const annotatedImage = document.getElementById('annotatedImage');
  const rawPreviewImage = document.getElementById('rawPreviewImage');
  const btnShowAnnotated = document.getElementById('btnShowAnnotated');
  const btnShowOriginal = document.getElementById('btnShowOriginal');
  const singleConfidenceVal = document.getElementById('singleConfidenceVal');
  const singleConfidenceBar = document.getElementById('singleConfidenceBar');
  const singleAnomalyScoreVal = document.getElementById('singleAnomalyScoreVal');
  const singleAnomalyBar = document.getElementById('singleAnomalyBar');
  const singleSeverityVal = document.getElementById('singleSeverityVal');
  const singleSeverityBar = document.getElementById('singleSeverityBar');
  const singleSeverityCategory = document.getElementById('singleSeverityCategory');
  const singleActionVal = document.getElementById('singleActionVal');
  const singleAreaVal = document.getElementById('singleAreaVal');
  const singleBboxVal = document.getElementById('singleBboxVal');
  const singlePixelAreaVal = document.getElementById('singlePixelAreaVal');
  const singleExplanation = document.getElementById('singleExplanation');
  const singleDualAgreementTag = document.getElementById('singleDualAgreementTag');
  const singleAreaFilterTag = document.getElementById('singleAreaFilterTag');

  const batchResultPanel = document.getElementById('batchResultPanel');
  const batchTotalCount = document.getElementById('batchTotalCount');
  const batchDefectCount = document.getElementById('batchDefectCount');
  const batchPassCount = document.getElementById('batchPassCount');
  const batchAvgLatency = document.getElementById('batchAvgLatency');
  const batchTableBody = document.getElementById('batchTableBody');
  const downloadCsvBtn = document.getElementById('downloadCsvBtn');

  const specimenModal = document.getElementById('specimenModal');
  const modalFilename = document.getElementById('modalFilename');
  const modalCategorySubtitle = document.getElementById('modalCategorySubtitle');
  const modalAnnotatedImage = document.getElementById('modalAnnotatedImage');
  const modalStatsGrid = document.getElementById('modalStatsGrid');
  const closeModalBtn = document.getElementById('closeModalBtn');

  // Internal State
  let stagedFiles = [];
  let currentResults = [];
  let currentBatchId = null;
  let loadingInterval = null;

  // --------------------------------------------------------------------------
  // Utility Functions
  // --------------------------------------------------------------------------

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function showError(title, message) {
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorBanner.classList.remove('hidden');
    errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function clearError() {
    errorBanner.classList.add('hidden');
  }

  // --------------------------------------------------------------------------
  // File Validation & Staging
  // --------------------------------------------------------------------------

  function validateFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_MIME_TYPES.includes(file.type)) {
      return {
        valid: false,
        error: `File "${file.name}" has an unsupported format (${ext}). Please upload .jpg, .png, or .bmp files.`
      };
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return {
        valid: false,
        error: `File "${file.name}" is ${formatBytes(file.size)}, exceeding the 10 MB limit.`
      };
    }
    return { valid: true };
  }

  function addFiles(filesList) {
    clearError();
    let addedCount = 0;
    const errors = [];

    Array.from(filesList).forEach(file => {
      const validation = validateFile(file);
      if (!validation.valid) {
        errors.push(validation.error);
        return;
      }
      // Check for duplicates
      if (!stagedFiles.some(f => f.name === file.name && f.size === file.size)) {
        stagedFiles.push(file);
        addedCount++;
      }
    });

    if (errors.length > 0) {
      showError('File Validation Warning', errors[0]);
    }

    renderStagedFiles();
  }

  function removeFile(index) {
    stagedFiles.splice(index, 1);
    renderStagedFiles();
  }

  function clearAllFiles() {
    stagedFiles = [];
    fileInput.value = '';
    renderStagedFiles();
    clearError();
  }

  function renderStagedFiles() {
    fileChips.innerHTML = '';
    if (stagedFiles.length === 0) {
      fileTray.classList.add('hidden');
      analyzeBtn.disabled = true;
      fileCount.textContent = '0';
      return;
    }

    fileTray.classList.remove('hidden');
    analyzeBtn.disabled = false;
    fileCount.textContent = stagedFiles.length.toString();

    stagedFiles.forEach((file, idx) => {
      const chip = document.createElement('div');
      chip.className = 'file-chip';
      chip.innerHTML = `
        <span class="file-chip-name" title="${file.name}">${file.name}</span>
        <span class="file-chip-size">${formatBytes(file.size)}</span>
        <button type="button" class="file-chip-remove" data-index="${idx}" aria-label="Remove file">&times;</button>
      `;
      fileChips.appendChild(chip);
    });

    // Attach remove handlers
    fileChips.querySelectorAll('.file-chip-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(e.currentTarget.getAttribute('data-index'), 10);
        removeFile(index);
      });
    });
  }

  // --------------------------------------------------------------------------
  // Drag & Drop Event Listeners
  // --------------------------------------------------------------------------

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      addFiles(dt.files);
    }
  });

  dropzone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
  });

  clearFilesBtn.addEventListener('click', clearAllFiles);
  closeErrorBtn.addEventListener('click', clearError);

  // --------------------------------------------------------------------------
  // Sample Test Batch Generation (Client-Side Helper)
  // --------------------------------------------------------------------------

  loadSampleBtn.addEventListener('click', async () => {
    clearError();
    try {
      // 1. Attempt to fetch pre-generated specimen samples from server
      try {
        const resp = await fetch('/api/samples');
        if (resp.ok) {
          const sdata = await resp.json();
          if (sdata.samples && sdata.samples.length > 0) {
            const fetchedFiles = [];
            for (const sname of sdata.samples) {
              const fileResp = await fetch(`/samples/${encodeURIComponent(sname)}`);
              if (fileResp.ok) {
                const blob = await fileResp.blob();
                fetchedFiles.push(new File([blob], sname, { type: 'image/png' }));
              }
            }
            if (fetchedFiles.length > 0) {
              addFiles(fetchedFiles);
              return;
            }
          }
        }
      } catch (fetchErr) {
        // Fallback to client-side canvas generation
      }

      // 2. Client-side procedural sample generator fallback
      const sampleNames = [
        'specimen_sample_crack.png',
        'specimen_sample_scratch.png',
        'specimen_sample_dent.png',
        'specimen_sample_normal.png'
      ];

      const sampleFiles = [];
      for (const name of sampleNames) {
        // Create 256x256 test canvas
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        // Background surface metal texture
        ctx.fillStyle = '#94a3b8';
        ctx.fillRect(0, 0, 256, 256);

        // Add visual texture
        for (let i = 0; i < 600; i++) {
          ctx.fillStyle = Math.random() > 0.5 ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
          ctx.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
        }

        // Draw flaw shape based on name
        if (name.includes('crack')) {
          ctx.strokeStyle = '#1e293b';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(110, 100);
          ctx.lineTo(135, 130);
          ctx.lineTo(125, 155);
          ctx.lineTo(150, 180);
          ctx.stroke();
        } else if (name.includes('scratch')) {
          ctx.strokeStyle = '#f8fafc';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(60, 60);
          ctx.lineTo(190, 190);
          ctx.stroke();
        } else if (name.includes('dent')) {
          const grad = ctx.createRadialGradient(128, 128, 5, 128, 128, 30);
          grad.addColorStop(0, '#475569');
          grad.addColorStop(1, '#94a3b8');
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(128, 128, 25, 0, Math.PI * 2);
          ctx.fill();
        }

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        const file = new File([blob], name, { type: 'image/png' });
        sampleFiles.push(file);
      }

      addFiles(sampleFiles);
    } catch (err) {
      showError('Sample Batch Load Failed', 'Unable to generate sample images: ' + err.message);
    }
  });

  // --------------------------------------------------------------------------
  // Pipeline Execution (POST /analyze)
  // --------------------------------------------------------------------------

  const STAGE_MESSAGES = [
    'Standardizing image format & letterbox preprocessing...',
    'Performing unsupervised surface reconstruction & anomaly scoring...',
    'Evaluating 7-class deep defect taxonomy classifier...',
    'Extracting Grad-CAM spatial activation & bounding boxes...',
    'Enforcing dual-confidence gate & morphological noise pruning...'
  ];

  function startLoadingAnimation() {
    loadingCard.classList.remove('hidden');
    singleResultPanel.classList.add('hidden');
    batchResultPanel.classList.add('hidden');
    analyzeBtn.disabled = true;

    let stageIdx = 0;
    loadingStage.textContent = STAGE_MESSAGES[0];
    loadingInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % STAGE_MESSAGES.length;
      loadingStage.textContent = STAGE_MESSAGES[stageIdx];
    }, 450);
  }

  function stopLoadingAnimation() {
    clearInterval(loadingInterval);
    loadingCard.classList.add('hidden');
    analyzeBtn.disabled = stagedFiles.length === 0;
  }

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (stagedFiles.length === 0) return;

    clearError();
    startLoadingAnimation();

    const formData = new FormData();
    stagedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        let errDetail = `Server returned status ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson.error) errDetail = errJson.error;
        } catch (ignored) {}
        throw new Error(errDetail);
      }

      const data = await response.json();

      if (!data.results || data.results.length === 0) {
        throw new Error('No inspection results were returned by the server.');
      }

      currentResults = data.results;
      currentBatchId = data.batch_id || 'latest';

      stopLoadingAnimation();
      renderInspectionResults(data);
    } catch (err) {
      stopLoadingAnimation();
      showError('Analysis Execution Error', err.message || 'Failed to inspect specimen(s). Check backend server logs.');
    }
  });

  // --------------------------------------------------------------------------
  // Results Rendering
  // --------------------------------------------------------------------------

  function renderInspectionResults(data) {
    const results = data.results;

    // 1. Single View (display the first specimen in detail)
    const primary = results[0];
    displaySingleInspection(primary);

    // 2. If multiple specimens (or batch), render batch summary table
    if (results.length > 1) {
      renderBatchLedger(data);
    } else {
      batchResultPanel.classList.add('hidden');
    }
  }

  function displaySingleInspection(res) {
    singleResultPanel.classList.remove('hidden');

    const isDefective = res.is_defective;
    const defectType = res.defect_type || 'normal';
    const conf = res.confidence || 0.0;
    const score = res.anomaly_score || 0.0;
    const area = res.defect_area || 0.0;
    const areaPx = res.defect_area_px || 0.0;
    const bboxes = res.bounding_boxes || [];

    // Header Status Badge
    singleStatusBadge.className = 'status-badge ' + (isDefective ? 'defective' : 'normal');
    singleStatusBadge.textContent = isDefective ? `REJECT: DEFECTIVE` : `PASS: PRISTINE`;

    // Defect Type Pill
    singleCategoryPill.textContent = defectType.replace('_', ' ').toUpperCase();
    singleFilename.textContent = res.filename;
    singleLatencyTag.textContent = `${res.inference_time_ms.toFixed(1)} ms`;
    singleStatusDecision.textContent = res.decision_status || (isDefective ? 'CONFIRMED_DEFECT' : 'CLEAN_PASS');

    // Images
    const annotatedSrc = res.annotated_image_base64
      ? `data:image/png;base64,${res.annotated_image_base64}`
      : (res.annotated_image_path || '');

    annotatedImage.src = annotatedSrc;

    // Raw input preview
    if (res.raw_image_base64) {
      rawPreviewImage.src = `data:image/png;base64,${res.raw_image_base64}`;
    } else {
      rawPreviewImage.src = annotatedSrc;
    }

    // Toggle states
    btnShowAnnotated.classList.add('active');
    btnShowOriginal.classList.remove('active');
    annotatedImage.classList.remove('hidden');
    rawPreviewImage.classList.add('hidden');

    // Confidence
    const confPct = (conf * 100).toFixed(1);
    singleConfidenceVal.textContent = `${confPct}%`;
    singleConfidenceBar.style.width = `${Math.min(100, Math.max(5, conf * 100))}%`;

    // Anomaly Score
    singleAnomalyScoreVal.textContent = score.toFixed(4);
    singleAnomalyBar.style.width = `${Math.min(100, Math.max(5, score * 100))}%`;

    // Area & Bboxes
    singleAreaVal.textContent = `${area.toFixed(2)} mm²`;
    singlePixelAreaVal.textContent = `${areaPx.toFixed(0)} px`;

    // Severity & Action Telemetry
    const sevScore = typeof res.severity_score === 'number' ? res.severity_score : 0.0;
    const sevCategory = res.severity_category || (isDefective ? 'Major' : 'Minor');
    const recAction = res.recommended_action || (isDefective ? 'Flag for review' : 'Log only');
    const sevHex = res.severity_color_hex || (sevCategory === 'Critical' ? '#ef4444' : (sevCategory === 'Major' ? '#eab308' : '#22c55e'));

    if (singleSeverityVal) {
      singleSeverityVal.textContent = isDefective ? `${sevCategory} (${sevScore.toFixed(1)})` : `Minor (0.0)`;
      singleSeverityVal.style.color = sevHex;
    }
    if (singleSeverityBar) {
      singleSeverityBar.style.width = `${Math.min(100, Math.max(4, sevScore))}%`;
      singleSeverityBar.style.backgroundColor = sevHex;
    }
    if (singleSeverityCategory) {
      singleSeverityCategory.textContent = sevCategory;
      singleSeverityCategory.style.color = sevHex;
    }
    if (singleActionVal) {
      singleActionVal.textContent = recAction;
    }

    if (bboxes.length > 0) {
      const box = bboxes[0];
      singleBboxVal.textContent = `[x:${box[0]}, y:${box[1]}, w:${box[2]}, h:${box[3]}]`;
    } else {
      singleBboxVal.textContent = 'None (Clear)';
    }

    // Audit Explanation
    singleExplanation.textContent = res.explanation || (isDefective ? 'Dual-confidence consensus verified presence of localized surface defect.' : 'Surface uniform and within nominal tolerance thresholds.');
    singleDualAgreementTag.textContent = res.dual_agreement ? 'Dual-Gate: CONSENSUS' : 'Dual-Gate: EVALUATED';
    singleAreaFilterTag.textContent = (isDefective && areaPx >= 25) ? 'Morphology: ADEQUATE' : 'Morphology: NOMINAL';

    singleResultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Toggle Viewport Buttons
  btnShowAnnotated.addEventListener('click', () => {
    btnShowAnnotated.classList.add('active');
    btnShowOriginal.classList.remove('active');
    annotatedImage.classList.remove('hidden');
    rawPreviewImage.classList.add('hidden');
  });

  btnShowOriginal.addEventListener('click', () => {
    btnShowOriginal.classList.add('active');
    btnShowAnnotated.classList.remove('active');
    annotatedImage.classList.add('hidden');
    rawPreviewImage.classList.remove('hidden');
  });

  // --------------------------------------------------------------------------
  // Batch Ledger & Table Rendering
  // --------------------------------------------------------------------------

  function renderBatchLedger(data) {
    batchResultPanel.classList.remove('hidden');

    const results = data.results;
    const summary = data.summary || {};
    const total = results.length;
    const defects = results.filter(r => r.is_defective).length;
    const passes = total - defects;
    const avgLatency = (results.reduce((acc, r) => acc + (r.inference_time_ms || 0), 0) / total).toFixed(1);

    batchTotalCount.textContent = total.toString();
    batchDefectCount.textContent = defects.toString();
    batchPassCount.textContent = passes.toString();
    batchAvgLatency.textContent = `${avgLatency} ms`;

    // Configure CSV Download Link
    const batchId = data.batch_id || 'latest';
    downloadCsvBtn.href = `/report/${batchId}`;

    // Render Table Rows
    batchTableBody.innerHTML = '';
    results.forEach((res, idx) => {
      const tr = document.createElement('tr');

      const isDef = res.is_defective;
      const thumbSrc = res.annotated_image_base64
        ? `data:image/png;base64,${res.annotated_image_base64}`
        : '';

      const statusHtml = isDef
        ? `<span class="table-status-pill defect">DEFECTIVE</span>`
        : `<span class="table-status-pill pass">NORMAL</span>`;

      const typeLabel = res.defect_type.replace('_', ' ').toUpperCase();
      const sevScore = typeof res.severity_score === 'number' ? res.severity_score : 0.0;
      const sevCategory = res.severity_category || (isDef ? 'Major' : 'Minor');
      const recAction = res.recommended_action || (isDef ? 'Flag for review' : 'Log only');
      const sevHex = res.severity_color_hex || (sevCategory === 'Critical' ? '#ef4444' : (sevCategory === 'Major' ? '#eab308' : '#22c55e'));

      tr.innerHTML = `
        <td>
          <img src="${thumbSrc}" alt="Thumbnail" class="table-thumb">
        </td>
        <td>
          <strong>${res.filename}</strong>
        </td>
        <td>${statusHtml}</td>
        <td>
          <span class="font-semibold">${typeLabel}</span>
        </td>
        <td>
          <span style="display:inline-block; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px; background:${sevHex}22; color:${sevHex}; border:1px solid ${sevHex}66;">
            ${sevCategory.toUpperCase()} (${sevScore.toFixed(0)})
          </span>
        </td>
        <td>
          <span style="font-size:12px; font-weight:500; color:#94a3b8;">${recAction}</span>
        </td>
        <td>
          <span class="text-mono">${(res.confidence * 100).toFixed(1)}%</span>
        </td>
        <td>
          <span class="text-mono">${res.defect_area.toFixed(2)} mm²</span>
        </td>
        <td>
          <span class="text-mono">${res.anomaly_score.toFixed(3)}</span>
        </td>
        <td style="text-align: right;">
          <button type="button" class="btn-ghost-sm inspect-row-btn" data-index="${idx}">
            Inspect
          </button>
        </td>
      `;
      batchTableBody.appendChild(tr);
    });

    // Attach inspect buttons
    batchTableBody.querySelectorAll('.inspect-row-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const index = parseInt(e.currentTarget.getAttribute('data-index'), 10);
        openSpecimenModal(results[index]);
      });
    });
  }

  // --------------------------------------------------------------------------
  // Inspection Detail Modal
  // --------------------------------------------------------------------------

  function openSpecimenModal(res) {
    modalFilename.textContent = res.filename;
    modalCategorySubtitle.textContent = `Status: ${res.is_defective ? 'DEFECTIVE' : 'NORMAL'} (${res.defect_type.toUpperCase()})`;

    const imgSrc = res.annotated_image_base64
      ? `data:image/png;base64,${res.annotated_image_base64}`
      : '';
    modalAnnotatedImage.src = imgSrc;

    const modalSevCategory = res.severity_category || (res.is_defective ? 'Major' : 'Minor');
    const modalSevScore = typeof res.severity_score === 'number' ? res.severity_score : 0.0;
    const modalSevHex = res.severity_color_hex || (modalSevCategory === 'Critical' ? '#ef4444' : (modalSevCategory === 'Major' ? '#eab308' : '#22c55e'));
    const modalAction = res.recommended_action || (res.is_defective ? 'Flag for review' : 'Log only');

    modalStatsGrid.innerHTML = `
      <div class="stat-pill">
        <span class="stat-label">Defect Class</span>
        <span class="stat-val">${res.defect_type.toUpperCase()}</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Severity</span>
        <span class="stat-val" style="color: ${modalSevHex};">${modalSevCategory} (${modalSevScore.toFixed(1)})</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Recommended Action</span>
        <span class="stat-val">${modalAction}</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Confidence</span>
        <span class="stat-val text-primary">${(res.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Anomaly Score</span>
        <span class="stat-val text-amber">${res.anomaly_score.toFixed(4)}</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Defect Area</span>
        <span class="stat-val text-emerald">${res.defect_area.toFixed(2)} mm²</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Inference Time</span>
        <span class="stat-val text-mono">${res.inference_time_ms.toFixed(1)} ms</span>
      </div>
      <div class="stat-pill">
        <span class="stat-label">Audit Decision</span>
        <span class="stat-val">${res.decision_status || 'CONFIRMED'}</span>
      </div>
    `;

    specimenModal.classList.remove('hidden');
  }

  closeModalBtn.addEventListener('click', () => {
    specimenModal.classList.add('hidden');
  });

  specimenModal.addEventListener('click', (e) => {
    if (e.target === specimenModal) {
      specimenModal.classList.add('hidden');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !specimenModal.classList.contains('hidden')) {
      specimenModal.classList.add('hidden');
    }
  });

})();
