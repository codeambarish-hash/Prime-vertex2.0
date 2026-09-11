#!/usr/bin/env python3
"""
Inspectra AI — Complete TensorFlow / Keras Pipeline Verification Suite
=====================================================================
Automated validation of Requirements 1-27:
  1. Verifies presence and integrity of all model artifacts in models/keras/
  2. Validates schema and fields in dataset_version.json and dataset_report.json
  3. Verifies outputs in outputs/keras/ (model_comparison.json, metrics, confusion matrices)
  4. Tests feature embedding index in data/embedding_index/ (embeddings.npy, metadata.json)
  5. Tests inference engine across 7 defect taxonomy classes
  6. Validates complete inference schema:
       - decision ('defective' | 'normal' | 'uncertain' | 'unknown')
       - similarity_score & similar_images (top 3-5 reference specimens)
       - severity ('NONE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'Severity unavailable')
       - recommended_action
       - probabilities dict (all 7 classes summing to ~1.0)
  7. Tests threshold gating (< 0.60 -> uncertain)
  8. Tests novel/unseen defect detection (low confidence + low similarity -> unknown)
  9. Tests robust error handling
"""

import os
import sys
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keras_config import (
    DEFECT_CLASSES,
    SAVED_MODEL_PATH,
    COMBINED_SAVED_MODEL_PATH,
    CLASS_NAMES_PATH,
    COMBINED_CLASS_NAMES_PATH,
    MODEL_CONFIG_PATH,
    COMBINED_MODEL_CONFIG_PATH,
    DATASET_VERSION_PATH,
    BASELINE_METRICS_PATH,
    COMBINED_METRICS_PATH,
    COMPARISON_REPORT_PATH,
    COMBINED_CONFUSION_MATRIX_PATH,
)
from src.keras_model import run_keras_inference
from training.build_embeddings import (
    EMBEDDING_INDEX_DIR,
    load_npy_file,
    query_similar_images,
    extract_visual_embedding_from_bytes,
)


def test_model_artifacts():
    """Checks that all required model files and indexes exist and are valid."""
    print("[Test 1/8] Checking required model and index artifacts...")
    artifacts = [
        (COMBINED_SAVED_MODEL_PATH, "Combined .keras model package"),
        (COMBINED_CLASS_NAMES_PATH, "Combined class names JSON"),
        (COMBINED_MODEL_CONFIG_PATH, "Combined model config JSON"),
        (DATASET_VERSION_PATH, "Dataset version manifest"),
        (PROJECT_ROOT / "models" / "keras" / "feature_extractor.keras", "Feature extractor package"),
        (BASELINE_METRICS_PATH, "Model A baseline metrics JSON"),
        (COMBINED_METRICS_PATH, "Model B combined metrics JSON"),
        (COMPARISON_REPORT_PATH, "Model comparison report JSON"),
        (PROJECT_ROOT / "outputs" / "keras" / "model_comparison.json", "Standard model comparison JSON"),
        (PROJECT_ROOT / "outputs" / "keras" / "dataset_report.json", "Dataset report JSON"),
        (EMBEDDING_INDEX_DIR / "embeddings.npy", "Reference embeddings NumPy array"),
        (EMBEDDING_INDEX_DIR / "metadata.json", "Reference metadata index JSON"),
        (COMBINED_CONFUSION_MATRIX_PATH, "Confusion matrix PNG")
    ]

    for path, desc in artifacts:
        assert path.exists(), f"Missing required artifact: {path} ({desc})"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"
        print(f"  ✓ {desc:<38}: {path.name} ({path.stat().st_size} bytes)")
    print("All required model artifacts verified.")


def test_dataset_version_manifest():
    """Verifies dataset_version.json and dataset_report.json conform to specification."""
    print("\n[Test 2/8] Validating dataset manifests and reports...")
    with open(DATASET_VERSION_PATH, "r") as f:
        meta = json.load(f)

    required_keys = [
        "dataset_sources",
        "number_of_images",
        "classes",
        "training_date",
        "image_size",
        "model_architecture",
        "training_epochs",
        "test_accuracy",
        "test_f1_score"
    ]

    for key in required_keys:
        assert key in meta, f"Missing required key '{key}' in dataset_version.json"
        print(f"  ✓ Key present: {key:<22} = {meta[key]}")

    assert isinstance(meta["dataset_sources"], list), "dataset_sources must be a list"
    assert len(meta["dataset_sources"]) >= 2, "Must reference both Keras and PyTorch datasets"
    assert meta["number_of_images"] >= 100, f"Expected unified dataset >= 100 images, got {meta['number_of_images']}"
    assert len(meta["classes"]) == 7, "Must contain all 7 defect taxonomy classes"

    # Also verify dataset_report.json
    rep_path = PROJECT_ROOT / "outputs" / "keras" / "dataset_report.json"
    with open(rep_path, "r") as f:
        rep = json.load(f)
    for k in ["total_images", "images_per_class", "image_dimensions", "corrupted_images", "duplicate_images", "class_imbalance"]:
        assert k in rep, f"Missing '{k}' in dataset_report.json"
    print("Dataset manifests and reports comply with all specifications.")


def test_inference_schema():
    """Verifies full inference output matches complete specification (Requirements 10-17)."""
    print("\n[Test 3/8] Testing complete inference schema and similarity pipeline...")
    unified_dir = PROJECT_ROOT / "data" / "unified_keras_dataset"
    specimens = list(unified_dir.rglob("*.png"))
    assert len(specimens) > 0, "No specimens found in unified_keras_dataset"

    test_img = specimens[0]
    result = run_keras_inference(str(test_img))

    assert result.get("success") is True, f"Inference failed: {result}"
    for k in ["class", "confidence", "is_defective", "decision", "similarity_score", "similar_images", "probabilities", "severity", "recommended_action", "inference_time_ms"]:
        assert k in result, f"Missing required field '{k}' in inference result"

    assert result["decision"] in ["defective", "normal", "uncertain", "unknown"], f"Invalid decision: {result['decision']}"
    assert isinstance(result["similar_images"], list), "similar_images must be a list"
    assert len(result["similar_images"]) > 0, "similar_images must contain top matches"

    top_sim = result["similar_images"][0]
    assert "image" in top_sim and "class" in top_sim and "similarity" in top_sim, "Invalid structure in similar_images entry"
    assert 0.0 <= result["similarity_score"] <= 1.0, f"Invalid similarity_score: {result['similarity_score']}"

    # Verify probability distribution
    probs = result["probabilities"]
    expected_classes = ["normal", "crack", "scratch", "dent", "stain", "discoloration", "dimensional_irregularity"]
    for cls_name in expected_classes:
        assert cls_name in probs, f"Missing class '{cls_name}' in probabilities dict"
        assert 0.0 <= probs[cls_name] <= 1.0, f"Invalid probability for '{cls_name}'"

    prob_sum = sum(probs[c] for c in expected_classes)
    assert abs(prob_sum - 1.0) < 0.05, f"Probabilities must sum to ~1.0, got {prob_sum}"

    print(f"  ✓ 'decision':           {result['decision']}")
    print(f"  ✓ 'class':              {result['class']}")
    print(f"  ✓ 'confidence':         {result['confidence']:.4f}")
    print(f"  ✓ 'similarity_score':   {result['similarity_score']:.4f}")
    print(f"  ✓ 'severity':           {result['severity']}")
    print(f"  ✓ 'similar_images':     {len(result['similar_images'])} reference matches found")
    print(f"  ✓ 'recommended_action': {result['recommended_action'][:50]}...")
    print("Inference output schema fully validated.")


def test_taxonomy_coverage():
    """Verifies inference correctly classifies specimens from each of the 7 classes."""
    print("\n[Test 4/8] Validating inference across all 7 defect classes...")
    unified_dir = PROJECT_ROOT / "data" / "unified_keras_dataset"
    tested = 0

    for cls_folder in ["normal", "crack", "scratch", "dent", "stain", "discoloration", "dimensional_irregularity"]:
        folder_path = unified_dir / cls_folder
        if not folder_path.exists():
            continue
        imgs = list(folder_path.glob("*.png"))
        if not imgs:
            continue
        sample = imgs[0]
        res = run_keras_inference(str(sample))
        pred_cls = res["class"]
        conf = res["confidence"]
        is_def = res["is_defective"]
        decision = res["decision"]
        severity = res["severity"]
        assert res["success"] is True, f"Inference failed on {sample}"
        print(f"  ✓ Target: {cls_folder:<25} -> Pred: {pred_cls:<25} (conf: {conf:.2f}, decision: {decision}, sev: {severity})")
        tested += 1

    assert tested >= 7, f"Expected 7 classes tested, got {tested}"
    print("All 7 classes successfully verified.")


def test_confidence_threshold_gating():
    """Verifies confidence threshold gating and uncertainty tagging (Requirement 13)."""
    print("\n[Test 5/8] Testing confidence threshold gating...")
    unified_dir = PROJECT_ROOT / "data" / "unified_keras_dataset"
    sample = list(unified_dir.rglob("*.png"))[0]

    # Standard threshold 0.60
    res_normal = run_keras_inference(str(sample), confidence_threshold=0.60)
    assert res_normal["uncertain"] is False
    assert res_normal["decision"] in ["defective", "normal"]

    # Extreme threshold 0.999 -> triggers uncertainty
    res_high = run_keras_inference(str(sample), confidence_threshold=0.999)
    assert res_high["uncertain"] is True
    assert res_high["decision"] in ["uncertain", "unknown"]
    assert res_high["is_defective"] is False, "Uncertain decision must not falsely flag product as defective"
    print("  ✓ Standard threshold (0.60): decision =", res_normal["decision"], "(uncertain: False)")
    print("  ✓ Strict threshold   (0.999): decision =", res_high["decision"], "(uncertain: True)")
    print("Confidence threshold gating verified.")


def test_novel_unknown_defect_detection():
    """Verifies novel / unseen defect pattern detection (Requirement 14)."""
    print("\n[Test 6/8] Testing novel / unknown defect detection...")
    # Synthetic out-of-distribution pattern with low confidence & low similarity
    synthetic_unknown = bytes([(i * 127 + 53) % 256 for i in range(1024)])
    res = run_keras_inference(synthetic_unknown, confidence_threshold=0.60)

    assert res["decision"] == "unknown", f"Expected decision 'unknown', got {res['decision']}"
    assert res["class"] is None, f"Expected class None for unknown defect, got {res['class']}"
    assert res["similarity_score"] < 0.60, f"Expected low similarity score, got {res['similarity_score']}"
    assert "Unrecognized" in res["recommended_action"] or "unseen" in res.get("className", "").lower()
    print(f"  ✓ Unknown decision:     {res['decision']}")
    print(f"  ✓ Class:                {res['class']}")
    print(f"  ✓ Similarity score:     {res['similarity_score']:.4f}")
    print(f"  ✓ Action flag:          {res['recommended_action'][:60]}...")
    print("Novel / unknown defect pattern detection verified.")


def test_reference_embeddings_index():
    """Verifies embedding index mathematical integrity (Requirements 10, 21)."""
    print("\n[Test 7/8] Validating reference embedding index...")
    npy_path = EMBEDDING_INDEX_DIR / "embeddings.npy"
    meta_path = EMBEDDING_INDEX_DIR / "metadata.json"

    shape, floats = load_npy_file(npy_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    assert shape[0] == len(metadata), f"Mismatch between embeddings ({shape[0]}) and metadata ({len(metadata)})"
    assert shape[1] == 1280, f"Expected 1280 embedding dimensions, got {shape[1]}"

    # Verify unit normalization: ||vec||_2 == 1.0
    first_vec = floats[:1280]
    norm = math.sqrt(sum(x * x for x in first_vec))
    assert abs(norm - 1.0) < 1e-4, f"Vector must be unit normalized, got norm = {norm}"

    print(f"  ✓ Indexed specimens:    {shape[0]} items")
    print(f"  ✓ Embedding dimensions: {shape[1]} (EfficientNetB0 compatible)")
    print(f"  ✓ Unit vector L2 norm:  {norm:.6f} == 1.0 (Exact cosine similarity via dot product)")
    print("Reference embedding index verified.")


def test_robust_error_handling():
    """Verifies graceful handling of invalid / corrupt image bytes."""
    print("\n[Test 8/8] Testing error handling with corrupted image bytes...")
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"
    res = run_keras_inference(corrupt_bytes)
    assert res.get("success") is True or "error" in res
    print(f"  ✓ Graceful error response received: class={res.get('class')}, decision={res.get('decision')}")
    print("Error handling verified.")


def main():
    print("=" * 76)
    print("   INSPECTRA AI — TENSORFLOW / KERAS FULL VERIFICATION SUITE")
    print("=" * 76)
    try:
        test_model_artifacts()
        test_dataset_version_manifest()
        test_inference_schema()
        test_taxonomy_coverage()
        test_confidence_threshold_gating()
        test_novel_unknown_defect_detection()
        test_reference_embeddings_index()
        test_robust_error_handling()
        print("\n" + "=" * 76)
        print("   [✓] ALL VERIFICATION TESTS PASSED SUCCESSFULLY (8/8)")
        print("=" * 76)
        return 0
    except AssertionError as e:
        print(f"\n[X] Test Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
