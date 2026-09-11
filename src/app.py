"""
Inspectra AI — Industrial Quality Inspection Web Application
============================================================
Lightweight Flask web interface for interactive vision-based defect inspection.

Features:
  - Single-page responsive UI served from templates/index.html & static/
  - Drag-and-drop & multi-image batch uploading (.png, .jpg, .bmp)
  - POST /analyze: Executes run_inspection() and returns base64 annotated visualizations
  - GET /report/<batch_id>: Downloadable CSV report for batch runs
  - Basic input validation (file extension, 10MB max size)
  - Runnable directly with `python src/app.py` on http://localhost:5000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Remove the script directory (src/) from sys.path to prevent shadowing standard library 'inspect'
_src_dir = str(Path(__file__).resolve().parent)
while _src_dir in sys.path:
    sys.path.remove(_src_dir)

# Add project root to sys.path to enable cleanly importing src.* modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

import base64
import csv
import io
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    url_for,
    abort
)
from werkzeug.utils import secure_filename

from src.config import DEFECT_CLASSES, OUTPUTS_DIR, DEFAULT_IMAGE_SIZE
from src.pipeline import run_inspection, visualize_inspection_result
from src.keras_model import run_keras_inference, check_gpu_status, get_keras_model
from src.keras_config import (
    KERAS_CONFIDENCE_THRESHOLD,
    SAVED_MODEL_PATH,
    EVALUATION_METRICS_PATH,
    CLASSIFICATION_REPORT_PATH,
    TRAINING_HISTORY_PATH,
    DEFECT_CLASSES as KERAS_DEFECT_CLASSES,
)


# =============================================================================
# Flask Application Setup & Configuration
# =============================================================================

app = Flask(
    __name__,
    template_folder=str(_PROJECT_ROOT / "templates"),
    static_folder=str(_PROJECT_ROOT / "static")
)

# 10 MB maximum request size
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Allow up to 16MB for multi-image payload

# Staging directories
UPLOAD_DIR = _PROJECT_ROOT / "temp" / "uploads"
REPORT_DIR = _PROJECT_ROOT / "outputs" / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# In-memory batch report cache: batch_id -> CSV file path
BATCH_REPORTS: Dict[str, Path] = {}


def is_allowed_filename(filename: str) -> bool:
    """Verifies filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# Web Routes
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    """Serves the single-page industrial inspection frontend."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Endpoint reporting system status, active taxonomy, and operational parameters."""
    return jsonify({
        "status": "online",
        "service": "Inspectra AI Inspection Studio",
        "version": "2.4.0",
        "taxonomy": DEFECT_CLASSES,
        "max_upload_bytes": MAX_FILE_SIZE_BYTES
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns list of pre-generated specimen samples in data/test_samples."""
    sample_dir = _PROJECT_ROOT / "data" / "test_samples"
    if not sample_dir.exists():
        return jsonify({"samples": []})
    samples = sorted([p.name for p in sample_dir.glob("*.png")])
    return jsonify({"samples": samples})


@app.route("/samples/<filename>", methods=["GET"])
def get_sample_file(filename: str):
    """Serves a specimen file from data/test_samples."""
    sample_path = _PROJECT_ROOT / "data" / "test_samples" / secure_filename(filename)
    if sample_path.exists():
        return send_file(str(sample_path), mimetype="image/png")
    return abort(404)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Accepts one or more image files, validates each, executes run_inspection(),
    generates color-coded CAD/HUD annotated images, and returns complete structured
    JSON telemetry including base64-encoded visualizations for instant client display.
    """
    # 1. Retrieve uploaded files from request
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        # Fallback to 'file' singular parameter
        single_file = request.files.get("file")
        if single_file and single_file.filename != "":
            uploaded_files = [single_file]
        else:
            return jsonify({
                "error": "No image files were supplied in the upload request. Please select an image."
            }), 400

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    batch_upload_folder = UPLOAD_DIR / batch_id
    batch_upload_folder.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_defects = 0
    total_latency = 0.0

    # 2. Process each specimen through the inspection pipeline
    for file_obj in uploaded_files:
        orig_filename = file_obj.filename or "unknown.png"
        
        # Validation 1: Check file extension
        if not is_allowed_filename(orig_filename):
            ext = orig_filename.rsplit(".", 1)[1] if "." in orig_filename else "unknown"
            return jsonify({
                "error": f"Invalid file type for '{orig_filename}'. Allowed formats: .png, .jpg, .jpeg, .bmp (Received: .{ext})"
            }), 400

        # Save to disk
        safe_name = secure_filename(orig_filename) or f"specimen_{uuid.uuid4().hex[:8]}.png"
        save_path = batch_upload_folder / safe_name
        file_obj.save(save_path)

        # Validation 2: Check file size
        file_size = save_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            save_path.unlink(missing_ok=True)
            return jsonify({
                "error": f"File '{orig_filename}' ({file_size / (1024*1024):.1f} MB) exceeds the 10 MB maximum allowed limit."
            }), 400

        # Validation 3: Check minimum file integrity (> 16 bytes)
        if file_size < 16:
            save_path.unlink(missing_ok=True)
            return jsonify({
                "error": f"File '{orig_filename}' appears corrupted or empty (0 bytes)."
            }), 400

        try:
            # 3. Execute End-to-End Inspection Pipeline
            inspection_res = run_inspection(save_path)

            # 4. Generate Publication-Grade Annotated HUD Visualization
            ann_filename = f"annotated_{save_path.stem}.png"
            ann_path = batch_upload_folder / ann_filename
            visualize_inspection_result(save_path, inspection_res, save_path=ann_path)

            # 5. Base64 encode the annotated image for client-side rendering
            with open(ann_path, "rb") as f_ann:
                b64_annotated = base64.b64encode(f_ann.read()).decode("utf-8")

            # Also encode the raw input image for raw/annotated toggle in UI
            with open(save_path, "rb") as f_raw:
                b64_raw = base64.b64encode(f_raw.read()).decode("utf-8")

            is_defective = bool(inspection_res["is_defective"])
            if is_defective:
                total_defects += 1
            total_latency += float(inspection_res["inference_time_ms"])

            clean_result = {
                "filename": orig_filename,
                "is_defective": is_defective,
                "defect_type": str(inspection_res["defect_type"]),
                "confidence": float(round(inspection_res["confidence"], 4)),
                "bounding_boxes": [list(b) for b in inspection_res["bounding_boxes"]],
                "defect_area": float(inspection_res["defect_area"]),
                "defect_area_px": float(inspection_res["defect_area_px"]),
                "defect_area_mm2": float(inspection_res["defect_area_mm2"]),
                "anomaly_score": float(inspection_res["anomaly_score"]),
                "inference_time_ms": float(inspection_res["inference_time_ms"]),
                "decision_status": str(inspection_res["decision_status"]),
                "dual_agreement": bool(inspection_res.get("dual_agreement", True)),
                "explanation": str(inspection_res.get("explanation", "")),
                "severity_score": float(inspection_res.get("severity_score", 0.0)),
                "severity_category": str(inspection_res.get("severity_category", "Minor")),
                "recommended_action": str(inspection_res.get("recommended_action", "Log only")),
                "severity_color_rgb": list(inspection_res.get("severity_color_rgb", [34, 197, 94])),
                "severity_color_hex": str(inspection_res.get("severity_color_hex", "#22c55e")),
                "annotated_image_base64": b64_annotated,
                "raw_image_base64": b64_raw,
            }
            results.append(clean_result)

        except Exception as e:
            return jsonify({
                "error": f"Failed while processing '{orig_filename}': {str(e)}"
            }), 500

    # 6. Generate Batch CSV Report and Store in Cache
    csv_filename = f"inspection_report_{batch_id}.csv"
    csv_path = REPORT_DIR / csv_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "filename",
            "is_defective",
            "defect_type",
            "confidence",
            "severity_score",
            "severity_category",
            "recommended_action",
            "defect_area_mm2",
            "defect_area_px",
            "anomaly_score",
            "inference_time_ms",
            "decision_status"
        ])
        for r in results:
            writer.writerow([
                r["filename"],
                r["is_defective"],
                r["defect_type"],
                f"{r['confidence']:.4f}",
                f"{r.get('severity_score', 0.0):.2f}",
                r.get("severity_category", "Minor"),
                r.get("recommended_action", "Log only"),
                f"{r['defect_area_mm2']:.2f}",
                f"{r['defect_area_px']:.1f}",
                f"{r['anomaly_score']:.4f}",
                f"{r['inference_time_ms']:.2f}",
                r["decision_status"]
            ])

    BATCH_REPORTS[batch_id] = csv_path
    BATCH_REPORTS["latest"] = csv_path

    # Also keep latest in OUTPUTS_DIR / inspection_summary.csv
    try:
        import shutil
        shutil.copyfile(csv_path, OUTPUTS_DIR / "inspection_summary.csv")
    except Exception:
        pass

    avg_latency = round(total_latency / len(results), 2) if results else 0.0

    summary = {
        "batch_id": batch_id,
        "total_specimens": len(results),
        "defects_found": total_defects,
        "pristine_cleared": len(results) - total_defects,
        "average_latency_ms": avg_latency,
        "csv_download_url": f"/report/{batch_id}"
    }

    return jsonify({
        "batch_id": batch_id,
        "summary": summary,
        "results": results
    })


@app.route("/report/<batch_id>", methods=["GET"])
def download_report(batch_id: str):
    """
    GET /report/<batch_id>
    Streams a downloadable CSV audit report of the batch inspection results.
    """
    csv_path: Optional[Path] = None

    if batch_id in BATCH_REPORTS and BATCH_REPORTS[batch_id].exists():
        csv_path = BATCH_REPORTS[batch_id]
    else:
        # Check report directory on disk
        candidate = REPORT_DIR / f"inspection_report_{batch_id}.csv"
        if candidate.exists():
            csv_path = candidate
        elif (OUTPUTS_DIR / "inspection_summary.csv").exists():
            csv_path = OUTPUTS_DIR / "inspection_summary.csv"

    if not csv_path or not csv_path.exists():
        return jsonify({
            "error": f"No audit report found for batch ID '{batch_id}'."
        }), 404

    return send_file(
        str(csv_path),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"inspection_report_{batch_id}.csv"
    )


# =============================================================================
# TensorFlow / Keras Pipeline Endpoints
# =============================================================================

@app.route("/api/keras/status", methods=["GET"])
def keras_status():
    """
    GET /api/keras/status
    Reports the operational status of the TensorFlow / Keras pipeline,
    including model availability, GPU hardware detection, active classes,
    and confidence threshold settings.
    """
    gpu_info = check_gpu_status()
    model_loaded = SAVED_MODEL_PATH.exists()

    return jsonify({
        "status": "ready",
        "framework": "TensorFlow / Keras",
        "model_file_exists": model_loaded,
        "model_path": str(SAVED_MODEL_PATH),
        "confidence_threshold": KERAS_CONFIDENCE_THRESHOLD,
        "classes": KERAS_DEFECT_CLASSES,
        "num_classes": len(KERAS_DEFECT_CLASSES),
        "hardware": gpu_info,
        "training_script": "python training/train_keras.py",
        "evaluation_script": "python training/evaluate_keras.py"
    })


@app.route("/api/keras/predict", methods=["POST"])
@app.route("/keras/predict", methods=["POST"])
def keras_predict():
    """
    POST /api/keras/predict
    Executes deep defect classification using the TensorFlow/Keras pipeline.
    Accepts single image via multipart/form-data ('file' or 'image').
    Supports custom confidence threshold via form parameter 'threshold'.
    """
    image_file = request.files.get("file") or request.files.get("image")
    if not image_file or image_file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Please supply a 'file' parameter."
        }), 400

    filename = secure_filename(image_file.filename) or "specimen.png"
    if not is_allowed_filename(filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported format for '{filename}'. Allowed: png, jpg, jpeg, bmp."
        }), 400

    # Read threshold parameter
    threshold = request.form.get("threshold", type=float)
    if threshold is None or threshold <= 0 or threshold >= 1.0:
        threshold = KERAS_CONFIDENCE_THRESHOLD

    # Read image bytes
    raw_bytes = image_file.read()
    if len(raw_bytes) < 16:
        return jsonify({
            "success": False,
            "error": "Image file is empty or corrupted."
        }), 400

    # Run inference
    result = run_keras_inference(raw_bytes, confidence_threshold=threshold, generate_gradcam=True)

    # Encode raw image for client preview
    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
    result["image_preview_b64"] = f"data:image/png;base64,{b64_img}"
    result["filename"] = filename

    # Standardized response structure as specified in Requirement 23
    prediction_obj = {
        "class": result.get("class"),
        "confidence": result.get("confidence", 0.0),
        "decision": result.get("decision", "defective" if result.get("is_defective") else "normal"),
        "is_defective": result.get("is_defective", False),
        "similarity_score": result.get("similarity_score", 0.0),
        "similar_images": result.get("similar_images", []),
        "probabilities": result.get("probabilities", {}),
        "severity": result.get("severity", "NONE"),
        "recommended_action": result.get("recommended_action", ""),
        "inference_time_ms": result.get("inference_time_ms", result.get("processing_time_ms", 0.0))
    }

    response_payload = {
        "success": True,
        "prediction": prediction_obj,
        **result
    }

    return jsonify(response_payload)


@app.route("/keras/reference-image/<path:image_name>", methods=["GET"])
@app.route("/reference_images/<path:image_name>", methods=["GET"])
def get_reference_image(image_name: str):
    """Safe serving of reference images for UI similarity visualization."""
    clean_name = Path(image_name).name
    ref_paths = [
        _PROJECT_ROOT / "public" / "reference_images" / clean_name,
        _PROJECT_ROOT / "data" / "reference_images" / clean_name
    ]
    for p in ref_paths:
        if p.exists():
            return send_file(str(p), mimetype="image/png")
    return jsonify({"error": "Reference image not found"}), 404


@app.route("/api/keras/evaluation", methods=["GET"])
def keras_evaluation():
    """
    GET /api/keras/evaluation
    Returns the latest training and evaluation metrics from outputs/keras/.
    """
    if EVALUATION_METRICS_PATH.exists():
        try:
            import json
            with open(EVALUATION_METRICS_PATH, "r") as f:
                metrics = json.load(f)
            return jsonify({"status": "success", "metrics": metrics})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "pending",
        "message": "Model evaluation metrics not generated yet. Run 'python training/train_keras.py' to generate."
    })


@app.route("/api/keras/training-history", methods=["GET"])
@app.route("/api/training-history", methods=["GET"])
def keras_training_history():
    """
    GET /api/keras/training-history
    Streams the parsed training_history.json for visualization in EvaluationTab.
    """
    search_paths = [
        TRAINING_HISTORY_PATH,
        Path("public/training_history.json"),
        Path("outputs/keras/training_history.json"),
        Path("models/keras/training_history.json")
    ]
    for p in search_paths:
        if p and p.exists():
            try:
                import json
                with open(p, "r") as f:
                    history = json.load(f)
                return jsonify({
                    "status": "success",
                    "source": str(p),
                    "history": history
                })
            except Exception as err:
                return jsonify({"status": "error", "message": str(err)}), 500

    return jsonify({
        "status": "not_found",
        "message": "training_history.json not found. Run training/train_keras.py to generate."
    }), 404




@app.errorhandler(413)
def request_entity_too_large(error):
    """Handler for payload exceeding server threshold."""
    return jsonify({
        "error": "The total upload payload exceeds the server threshold (~16 MB). Please reduce file sizes or count."
    }), 413


# =============================================================================
# CLI Server Runner
# =============================================================================

def run_app():
    """Runs the Flask inspection web app."""
    # Read port from command line or environment
    port = 5000
    for idx, arg in enumerate(sys.argv):
        if arg == "--port" and idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])

    if "FLASK_PORT" in os.environ:
        port = int(os.environ["FLASK_PORT"])

    print("=" * 78)
    print("  INSPECTRA AI — INDUSTRIAL QUALITY CONTROL WEB STUDIO")
    print("=" * 78)
    print(f"[*] Serving locally on:       http://localhost:{port}")
    print(f"[*] Accessible across host:   http://0.0.0.0:{port}")
    print(f"[*] Templates Root:           {_PROJECT_ROOT / 'templates'}")
    print(f"[*] Static Assets Root:       {_PROJECT_ROOT / 'static'}")
    print(f"[*] Debug Mode:               Enabled (debug=True)")
    print("=" * 78)

    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    run_app()
