"""
Inspectra AI — AI Powered Visual Inspection
============================================
Main Execution Pipeline Orchestrator:
  1. Automated Preprocessing Verification (CLAHE, Bilateral Denoising, Letterbox)
  2. Dataset Preparation (Synthetic procedural OpenCV engine or MVTec AD benchmark)
  3. Calculation and printing of dataset statistics (class counts, split balance, mask areas)
  4. Generation of sample inspection report and preprocessing comparison in outputs/
"""

import argparse
import sys
from pathlib import Path

# Ensure root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PROCESSED_DIR,
    OUTPUTS_DIR,
    DEFECT_CLASSES,
    DEFAULT_IMAGE_SIZE,
)


def run_pipeline(
    mode: str = "synthetic",
    num_samples_per_class: int = 50,
    img_size: int = 256,
    mvtec_category: str = "metal_nut",
    run_preprocessing_check: bool = True,
    anomaly_method: str = "autoencoder",
    classifier_backbone: str = "efficientnet_b0",
    localization_method: str = "gradcam",
    pixel_to_mm: float = 0.1
):
    """Executes preprocessing verification, dataset preparation, anomaly detection, defect classification, and inspection reporting."""
    print("=" * 72)
    print("  INSPECTRA AI — AI POWERED VISUAL INSPECTION PIPELINE")
    print("=" * 72)
    print(f"[*] Execution Mode:       {mode.upper()}")
    print(f"[*] Anomaly Method:       {anomaly_method.upper()}")
    print(f"[*] Classifier Backbone:  {classifier_backbone.upper()}")
    print(f"[*] Target Resolution:    {img_size}x{img_size}")
    print(f"[*] Defect Taxonomy:      {', '.join(DEFECT_CLASSES)}")

    # Verify runtime dependencies
    missing_deps = []
    for pkg in ["cv2", "numpy", "torch", "albumentations", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_deps.append(pkg)

    if missing_deps:
        print("\n" + "!" * 72)
        print("  DEPENDENCY NOTICE: Full deep learning runtime packages required")
        print("!" * 72)
        print(f"  Missing libraries: {', '.join(missing_deps)}")
        print("\n  To execute the complete PyTorch training & inspection pipeline, run:")
        print("    pip install -r requirements.txt")
        print("\n  Web Interface & Direct Evaluation:")
        print("    - Evaluation Suite: python3 main.py --evaluate   (or: python3 src/evaluate.py)")
        print("    - Web Inspector:    Open the active browser preview for live inspection.")
        print("=" * 72)
        return

    import cv2
    import numpy as np
    from src.prepare_dataset import (
        prepare_synthetic_dataset,
        prepare_mvtec_dataset,
        compute_and_print_dataset_stats,
    )
    from src.inspect import create_inspection_quad_report
    from src.synthetic_generator import generate_synthetic_sample
    from src.preprocessing import (
        InspectraPreprocessor,
        visualize_preprocessing_batch,
        create_demo_manufacturing_samples
    )
    from src.anomaly_detection import (
        AnomalyDetector,
        ThresholdCalibrator,
        render_anomaly_inspection_figure
    )
    from src.defect_classifier import DefectClassifierEngine
    from src.localization import (
        DefectLocalizationEngine,
        create_localization_report
    )
    print("=" * 72)

    res = (img_size, img_size)
    records = []

    # Optional Preprocessing Pipeline Check
    if run_preprocessing_check:
        print("\n[Step 1/4] Running Inspectra AI Preprocessing Verification...")
        preprocessor = InspectraPreprocessor(target_size=res)
        demo_samples = create_demo_manufacturing_samples()
        prep_report = OUTPUTS_DIR / "preprocessing_comparison.png"
        visualize_preprocessing_batch(
            sample_images=[s[0] for s in demo_samples],
            sample_titles=[s[1] for s in demo_samples],
            preprocessor=preprocessor,
            save_path=prep_report
        )
        print(f"[SUCCESS] Preprocessing report saved to: {prep_report}")

    # Dataset Acquisition / Generation
    if mode == "mvtec":
        try:
            print(f"\n[Step 2/4] Downloading & Standardizing MVTec AD category: '{mvtec_category}'...")
            records = prepare_mvtec_dataset(category=mvtec_category, output_dir=DATA_PROCESSED_DIR, img_size=res)
        except Exception as err:
            print(f"[!] MVTec download failed: {err}")
            print("[*] Falling back to OpenCV synthetic surface & defect engine...")
            records = prepare_synthetic_dataset(output_dir=DATA_PROCESSED_DIR, num_samples_per_class=num_samples_per_class, img_size=res)
    else:
        print(f"\n[Step 2/4] Generating Synthetic Defect Dataset via OpenCV ({num_samples_per_class} per class)...")
        records = prepare_synthetic_dataset(output_dir=DATA_PROCESSED_DIR, num_samples_per_class=num_samples_per_class, img_size=res)

    # Compute and Print Dataset Statistics
    print("\n[Step 3/4] Analyzing Dataset Distribution & Mask Geometries...")
    stats = compute_and_print_dataset_stats(records)

    # Run Sample Inspection Demonstration & Anomaly Prediction
    print(f"[Step 4/4] Generating Sample Inspection Report & Anomaly Prediction ({anomaly_method})...")
    sample_img, sample_mask, sample_info = generate_synthetic_sample(
        width=img_size, height=img_size, defect_type="crack"
    )

    detector = AnomalyDetector(method=anomaly_method, img_size=res)
    pred_res = detector.predict(sample_img)
    print(f"[*] Anomaly Diagnosis: {pred_res['label'].upper()} (Score: {pred_res['anomaly_score']:.4f}, Threshold: {pred_res['threshold']:.4f}, Confidence: {pred_res['confidence']*100:.1f}%)")

    # Multi-Class Defect Categorization
    print(f"\n[*] Running Multi-Class Defect Categorization ({classifier_backbone.upper()} Backbone)...")
    classifier_engine = DefectClassifierEngine(backbone_name=classifier_backbone, img_size=res)
    cls_pred = classifier_engine.predict(sample_img)
    print(f"[+] Predicted Defect Class:  {cls_pred['predicted_class'].upper()} ({cls_pred['confidence']*100:.2f}% confidence)")
    print("[+] Per-Class Softmax Probabilities:")
    for cls_name, prob in cls_pred["probabilities"].items():
        bar = "█" * int(prob * 25)
        print(f"     - {cls_name:<26}: {prob*100:5.1f}% | {bar}")

    quad_report = create_inspection_quad_report(
        raw_image=sample_img,
        mask=sample_mask,
        predicted_class=cls_pred["predicted_class"],
        confidence=cls_pred["confidence"],
        bbox=tuple(sample_info["bbox"])
    )

    report_path = OUTPUTS_DIR / "sample_inspection_report.png"
    cv2.imwrite(str(report_path), quad_report)
    print(f"[SUCCESS] Sample 4-panel inspection report saved to: {report_path}")

    # Generate anomaly diagnostic figure if residual map is available
    if pred_res.get("residual_map") is not None and pred_res.get("reconstruction") is not None:
        anomaly_diag_path = OUTPUTS_DIR / "anomaly_detection_demo.png"
        render_anomaly_inspection_figure(
            original_img=pred_res["preprocessed_image"],
            reconstructed_img=pred_res["reconstruction"],
            residual_map=pred_res["residual_map"],
            prediction=pred_res,
            save_path=anomaly_diag_path
        )
        print(f"[SUCCESS] Autoencoder diagnostic panel saved to: {anomaly_diag_path}")

    # Step 5: Defect Localization (Grad-CAM / U-Net / OpenCV Contours & Calibration)
    print(f"\n[*] Running Defect Localization Engine ({localization_method.upper()}, pixel_to_mm={pixel_to_mm})...")
    loc_engine = DefectLocalizationEngine(
        classifier_model=None,
        unet_model=None,
        pixel_to_mm=pixel_to_mm,
        min_area_px=15.0
    )
    loc_result = loc_engine.localize(
        image=sample_img,
        method=localization_method,
        target_class=CLASS_TO_IDX.get(cls_pred["predicted_class"], 1),
        confidence=cls_pred["confidence"],
        defect_class=cls_pred["predicted_class"]
    )
    loc_summary = loc_result["summary"]
    print(f"[+] Localized Defect Regions: {loc_summary['region_count']} zone(s)")
    print(f"[+] Total Defect Surface:    {loc_summary['total_area_px']} px² ({loc_summary['total_area_mm2']} mm²)")

    loc_report_path = OUTPUTS_DIR / "defect_localization_sample.png"
    create_localization_report(
        original_image=sample_img,
        heatmap=loc_result["heatmap"],
        binary_mask=loc_result["binary_mask"],
        overlay_image=loc_result["overlay_image"],
        defect_class=loc_summary["defect_class"],
        confidence=loc_summary["confidence"],
        defect_regions=loc_result["regions"],
        save_path=loc_report_path,
        pixel_to_mm=pixel_to_mm
    )
    print(f"[SUCCESS] Defect localization scorecard saved to: {loc_report_path}")

    # Stage 5: False-Positive & Missed-Detection Reduction Refinement
    print("\n[Step 5/5] Running Refinement: Dual Agreement, Morphological Filter, TTA & Manual Review...")
    from src.refinement import (
        InspectionRefiner,
        RefinementConfig,
        DecisionStatus,
        render_refinement_scorecard
    )
    refine_config = RefinementConfig(
        pixel_to_mm=pixel_to_mm,
        min_area_px=25.0,
        tta_vote_threshold=0.60
    )
    refiner = InspectionRefiner(config=refine_config)
    refined_res = refiner.refine_inspection(
        image=sample_img,
        anomaly_score=anom_summary["anomaly_score"],
        predicted_class=loc_summary["defect_class"],
        confidence=loc_summary["confidence"],
        raw_mask=loc_result["binary_mask"],
        sample_id="SAMPLE_INSPECTION_RUN"
    )
    print(f"[+] Dual Confidence Agreement: {refined_res.dual_agreement} ({refined_res.agreement_reason})")
    print(f"[+] Morphological Filtering:   Raw {refined_res.raw_defect_area_px} px -> Cleaned {refined_res.filtered_defect_area_px} px ({refined_res.purged_blobs_count} noise blobs purged)")
    print(f"[+] TTA Ensemble Consistency:  {refined_res.tta_vote_ratio:.1%} votes (Consensus: {refined_res.tta_consensus})")
    print(f"[+] Authoritative Decision:    {refined_res.final_decision.value} (Manual Review: {refined_res.manual_review_required})")

    refine_scorecard_path = OUTPUTS_DIR / "refinement_scorecard.png"
    render_refinement_scorecard(sample_img, refined_res, save_path=refine_scorecard_path)
    print(f"[SUCCESS] Refinement scorecard saved to: {refine_scorecard_path}")

    # =========================================================================
    # STAGE 6: Quantitative Model & Pipeline Evaluation (Accuracy, F1, ROC-AUC, IoU)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  STAGE 6/6: QUANTITATIVE MODEL EVALUATION & BENCHMARK REPORTING")
    print("=" * 72)
    from src.evaluate import (
        evaluate_inspection_pipeline,
        format_evaluation_report
    )
    eval_report = evaluate_inspection_pipeline(
        output_dir=OUTPUTS_DIR,
        decision_threshold=0.50,
        save_plots=True
    )
    print(f"[+] Detection Accuracy:       {eval_report.detection.accuracy * 100:.2f}% (ROC-AUC: {eval_report.detection.roc_auc:.4f}, F1: {eval_report.detection.f1_score:.4f})")
    print(f"[+] Multi-Class Accuracy:     {eval_report.classification.accuracy * 100:.2f}% (Macro F1: {eval_report.classification.macro_f1:.4f}, Weighted F1: {eval_report.classification.weighted_f1:.4f})")
    print(f"[+] Defect Localization mIoU: {eval_report.localization.mean_mask_iou * 100:.2f}% (IoU@0.50 Success: {eval_report.localization.iou_at_50_accuracy * 100:.1f}%)")
    print(f"[SUCCESS] Evaluation report saved to: outputs/evaluation_report.txt")
    print(f"[SUCCESS] ROC curve saved to:         outputs/roc_curve.png")
    print(f"[SUCCESS] PR curve saved to:          outputs/precision_recall_curve.png")
    print(f"[SUCCESS] Confusion matrix saved to:  outputs/confusion_matrix.png")
    print(f"[SUCCESS] Evaluation scorecard saved: outputs/evaluation_scorecard.png")

    print("\n[INSPECTRA AI COMPLETE] End-to-end pipeline: Preprocessing -> Anomaly -> Classifier -> Localizer -> Refinement -> Evaluation verified.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspectra AI — AI Powered Visual Inspection Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to an image file or folder for end-to-end visual inspection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "mvtec"],
        default="synthetic",
        help="Dataset preparation source ('synthetic' with OpenCV or 'mvtec' for MVTec AD benchmark)"
    )
    parser.add_argument(
        "--anomaly-method",
        type=str,
        choices=["autoencoder", "transfer_learning"],
        default="autoencoder",
        help="Anomaly detection model: 'autoencoder' (unsupervised normal-only) or 'transfer_learning' (ResNet/EfficientNet)"
    )
    parser.add_argument(
        "--classifier-backbone",
        type=str,
        choices=["efficientnet_b0", "mobilenet_v2"],
        default="efficientnet_b0",
        help="Backbone architecture for defect classification transfer learning"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples per defect class for synthetic generation (total = 7 * num_samples)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Target image dimension in pixels"
    )
    parser.add_argument(
        "--mvtec-category",
        type=str,
        default="metal_nut",
        help="Category when mode='mvtec' (e.g. metal_nut, tile, bottle, cable)"
    )
    parser.add_argument(
        "--localization-method",
        type=str,
        choices=["gradcam", "unet"],
        default="gradcam",
        help="Localization method: 'gradcam' (CNN attention saliency) or 'unet' (direct segmentation mask)"
    )
    parser.add_argument(
        "--pixel-to-mm",
        type=float,
        default=0.1,
        help="Calibration ratio in millimeters per pixel (e.g. 0.1 mm/px)"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=True,
        help="Run preprocessing pipeline verification and save comparison report"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run quantitative model evaluation suite directly (Accuracy, F1, ROC-AUC, IoU, curves & reports)"
    )

    args = parser.parse_args()

    # 1. Direct Evaluation Mode
    if args.evaluate:
        from src.evaluate import evaluate_inspection_pipeline, format_evaluation_report
        report = evaluate_inspection_pipeline(output_dir=OUTPUTS_DIR, save_plots=True)
        print(format_evaluation_report(report))
        return

    # 2. End-to-End Inspection Mode (--input <image_or_folder> or default inspection)
    if args.input is not None or len(sys.argv) == 1:
        from src.pipeline import (
            run_inspection,
            run_batch_inspection,
            visualize_inspection_result,
            format_pipeline_summary,
            create_sample_test_images,
        )

        input_target = Path(args.input if args.input is not None else "data/test_samples")

        if input_target.is_file():
            # Single Image Inspection
            print("=" * 78)
            print(f"  INSPECTRA AI — SINGLE SPECIMEN INSPECTION")
            print("=" * 78)
            print(f"[*] Input Specimen:  {input_target}")
            
            res = run_inspection(input_target, pixel_to_mm=args.pixel_to_mm)
            ann_path = visualize_inspection_result(input_target, res, output_dir=OUTPUTS_DIR)
            res["annotated_image_path"] = str(ann_path)

            # Print single-image inspection details
            status_str = "[REJECT: DEFECTIVE]" if res["is_defective"] else "[PASS: PRISTINE]"
            print(f"[+] Status:          {status_str}")
            print(f"[+] Defect Type:     {res['defect_type'].upper()}")
            print(f"[+] Confidence:      {res['confidence']*100:.2f}%")
            print(f"[+] Anomaly Score:   {res['anomaly_score']:.4f}")
            print(f"[+] Bounding Boxes:  {res['bounding_boxes']}")
            print(f"[+] Defect Area:     {res['defect_area']:.2f} mm² ({res['defect_area_px']:.1f} px)")
            print(f"[+] Latency:         {res['inference_time_ms']:.2f} ms")
            print(f"[+] Annotated Image: {ann_path}")

            # Append/Write to summary CSV
            csv_path = OUTPUTS_DIR / "inspection_summary.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                import csv
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["filename", "is_defective", "defect_type", "confidence", "area", "anomaly_score", "inference_time_ms", "decision_status", "annotated_image_path"])
                writer.writerow([
                    res["filename"],
                    res["is_defective"],
                    res["defect_type"],
                    f"{res['confidence']:.4f}",
                    f"{res['defect_area']:.2f}",
                    f"{res['anomaly_score']:.4f}",
                    f"{res['inference_time_ms']:.2f}",
                    res["decision_status"],
                    str(ann_path)
                ])

            # Print Final Run Summary
            print("\n" + "=" * 78)
            print("  FINAL RUN SUMMARY")
            print("=" * 78)
            print("  Total Images Processed:       1")
            print(f"  Defects Found:                {1 if res['is_defective'] else 0}")
            print(f"  Defect Type Breakdown:        {res['defect_type']}: 1")
            print(f"  Average Inference Time:       {res['inference_time_ms']:.2f} ms / image")
            print(f"  Summary Ledger (CSV):         {csv_path}")
            print("=" * 78 + "\n")
            return

        else:
            # Batch Folder Inspection
            if not input_target.exists():
                print(f"[*] Input folder '{input_target}' does not exist. Creating sample test batch...")
                create_sample_test_images(target_dir=input_target)

            batch_output = run_batch_inspection(input_target, output_dir=OUTPUTS_DIR)
            summary = batch_output["summary"]

            # Print Final Run Summary
            print("\n" + "=" * 78)
            print("  FINAL RUN SUMMARY")
            print("=" * 78)
            print(f"  Total Images Processed:       {summary['total_images_processed']}")
            print(f"  Defects Found:                {summary['defects_found']}")
            print("  Defect Type Breakdown:")
            for dtype, count in summary["defect_type_breakdown"].items():
                if count > 0:
                    label = dtype.replace('_', ' ').title()
                    print(f"    - {label:<24}: {count} image{'s' if count != 1 else ''}")
            print(f"  Average Inference Time:       {summary['average_inference_time_ms']:.2f} ms / image")
            print(f"  Summary Ledger (CSV):         {summary['csv_path']}")
            print("=" * 78 + "\n")
            return

    run_pipeline(
        mode=args.mode,
        num_samples_per_class=args.num_samples,
        img_size=args.img_size,
        mvtec_category=args.mvtec_category,
        run_preprocessing_check=args.preprocess,
        anomaly_method=args.anomaly_method,
        classifier_backbone=args.classifier_backbone,
        localization_method=args.localization_method,
        pixel_to_mm=args.pixel_to_mm
    )


if __name__ == "__main__":
    main()
