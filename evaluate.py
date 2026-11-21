"""
VibeShift Standalone AI Accuracy Evaluator (CleanBTNVit Phase 12).

Console-only scientific accuracy evaluation CLI supporting:
  - Registered local datasets in `datasets/dataset_manifest.json` (`--list-datasets`)
  - Manual user test folder `dataset/{angry,disgust,fear,happy,neutral,sad,surprise}/`
  - Exact Phase 12 single-view (`--mode single`), horizontal-flip (`--mode flip`),
    or both (`--mode both`)
  - Strictly separated `pre_cropped` vs `full_photo` (YuNet) protocols
  - Complete per-run reproducible reports in `results/<run_id>/`
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector import DEFAULT_YUNET_PATH, YuNetFaceDetector, verify_yunet_integrity
from evaluator import (
    DatasetNotFoundError,
    EvalSample,
    LoadedDataset,
    PredictionRecord,
    UnsupportedAdapterError,
    compute_metrics,
    format_classification_report_text,
    format_dataset_listing,
    load_dataset,
    load_manifest,
    make_run_id,
    write_evaluation_artifacts,
)
from model import (
    CANONICAL_LABELS,
    DEFAULT_MODEL_PATH,
    MODEL_DISPLAY_NAME,
    CleanBTNVitEvaluatorModel,
    crop_face,
    preprocess_face_for_clean_btn,
    verify_checkpoint_integrity,
)


def _read_sample_image(sample: EvalSample) -> Optional[np.ndarray]:
    if sample.image_array is not None:
        return sample.image_array
    if sample.image_path is None or not sample.image_path.is_file():
        return None
    try:
        raw = sample.image_path.read_bytes()
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


def evaluate_loaded_dataset(
    loaded: LoadedDataset,
    model: CleanBTNVitEvaluatorModel,
    mode: str,
    protocol: str,
    output_base_dir: Path,
    detector: Optional[YuNetFaceDetector] = None,
    progress_every: int = 250,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[PredictionRecord], Path]:
    """
    Runs real CleanBTNVit inference on a LoadedDataset for one mode ('single' or 'flip')
    and writes all 6 reproducible evaluation artifacts into a unique results/<run_id>/ folder.
    """
    if mode not in ("single", "flip"):
        raise ValueError(f"Unsupported evaluation mode '{mode}'. Expected 'single' or 'flip'.")
    if protocol not in ("pre_cropped", "full_photo"):
        raise ValueError(f"Unsupported protocol '{protocol}'. Expected 'pre_cropped' or 'full_photo'.")

    if protocol == "full_photo" and detector is None:
        detector = YuNetFaceDetector(model_path=DEFAULT_YUNET_PATH)

    use_flip = mode == "flip"
    mode_label = "horizontal_flip" if use_flip else "single_view"
    records: List[PredictionRecord] = []
    total = len(loaded.samples)

    for idx, sample in enumerate(loaded.samples, 1):
        img = _read_sample_image(sample)
        if img is None:
            records.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    image_path=sample.display_path,
                    ground_truth=sample.ground_truth,
                    original_label=sample.original_label,
                    predicted=None,
                    correct=None,
                    status="unreadable_image",
                    detection_status="not_applicable",
                    num_faces_detected=None,
                    confidence=None,
                    inference_ms=0.0,
                    probabilities=None,
                )
            )
            continue

        face_img = img
        det_status = "not_applicable"
        num_faces: Optional[int] = None

        if protocol == "full_photo":
            assert detector is not None
            det_status, bbox, num_faces = detector.select_evaluation_face(
                img,
                target_box_xywh=sample.target_box_xywh,
                target_iou_min=0.3,
            )
            if bbox is None:
                records.append(
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        image_path=sample.display_path,
                        ground_truth=sample.ground_truth,
                        original_label=sample.original_label,
                        predicted=None,
                        correct=None,
                        status=det_status,
                        detection_status=det_status,
                        num_faces_detected=num_faces,
                        confidence=None,
                        inference_ms=0.0,
                        probabilities=None,
                    )
                )
                continue
            cropped = crop_face(img, bbox, margin_ratio=0.06)
            if cropped is None:
                records.append(
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        image_path=sample.display_path,
                        ground_truth=sample.ground_truth,
                        original_label=sample.original_label,
                        predicted=None,
                        correct=None,
                        status="preprocess_failed",
                        detection_status=det_status,
                        num_faces_detected=num_faces,
                        confidence=None,
                        inference_ms=0.0,
                        probabilities=None,
                    )
                )
                continue
            face_img = cropped
        elif sample.crop_from_target_box and sample.target_box_xywh is not None:
            det_status = "annotated_box"
            cropped = crop_face(img, sample.target_box_xywh, margin_ratio=0.0)
            if cropped is None:
                records.append(
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        image_path=sample.display_path,
                        ground_truth=sample.ground_truth,
                        original_label=sample.original_label,
                        predicted=None,
                        correct=None,
                        status="preprocess_failed",
                        detection_status=det_status,
                        num_faces_detected=None,
                        confidence=None,
                        inference_ms=0.0,
                        probabilities=None,
                    )
                )
                continue
            face_img = cropped

        tensor = preprocess_face_for_clean_btn(face_img)
        if tensor is None:
            records.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    image_path=sample.display_path,
                    ground_truth=sample.ground_truth,
                    original_label=sample.original_label,
                    predicted=None,
                    correct=None,
                    status="preprocess_failed",
                    detection_status=det_status,
                    num_faces_detected=num_faces,
                    confidence=None,
                    inference_ms=0.0,
                    probabilities=None,
                )
            )
            continue

        probs, inf_ms = model.predict_tensor(tensor, horizontal_flip=use_flip)
        pred_idx = int(np.argmax(probs))
        pred_label = CANONICAL_LABELS[pred_idx]
        conf = float(probs[pred_idx])
        prob_dict = {lbl: float(probs[i]) for i, lbl in enumerate(CANONICAL_LABELS)}

        records.append(
            PredictionRecord(
                sample_id=sample.sample_id,
                image_path=sample.display_path,
                ground_truth=sample.ground_truth,
                original_label=sample.original_label,
                predicted=pred_label,
                correct=(pred_label == sample.ground_truth),
                status="evaluated",
                detection_status=det_status,
                num_faces_detected=num_faces,
                confidence=conf,
                inference_ms=inf_ms,
                probabilities=prob_dict,
            )
        )

        if not quiet and progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"  [{mode_label}] Processed {idx} / {total} images...", flush=True)

    metrics = compute_metrics(records, protocol=protocol)
    run_id = make_run_id(loaded.dataset_name, loaded.split_name, mode_label)
    run_dir = output_base_dir / run_id

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": MODEL_DISPLAY_NAME,
        "checkpoint_path": str(model.weights_path),
        "checkpoint_sha256": model.checkpoint_sha256,
        "device": model.device_name,
        "precision": model.precision_label,
        "dataset_name": loaded.dataset_name,
        "dataset_display_name": loaded.display_name,
        "split_name": loaded.split_name,
        "adapter": loaded.adapter,
        "protocol": protocol,
        "inference_mode": mode_label,
        "label_mapping": loaded.label_mapping,
        "provenance": loaded.provenance,
        "license_status": loaded.license_status,
        "independence_status": loaded.independence_status,
        "suitable_for_independent_test": loaded.suitable_for_independent_test,
        "used_for_training": loaded.used_for_training,
        "used_for_checkpoint_selection": loaded.used_for_checkpoint_selection,
        "previously_inspected": loaded.previously_inspected,
        "disclosure": loaded.disclosure,
        "dataset_warnings": loaded.warnings,
        "output_dir": str(run_dir),
    }

    write_evaluation_artifacts(run_dir, metadata, metrics, records)
    return metadata, metrics, records, run_dir


def run_environment_self_check(project_root: Path) -> int:
    """Verifies model checkpoint SHA-256, YuNet SHA-256, imports, and dataset registry without running benchmarks."""
    print("[Check 1/4] Verifying CleanBTNVit Phase 12 checkpoint SHA-256...")
    ok_m, sha_m, err_m = verify_checkpoint_integrity(DEFAULT_MODEL_PATH)
    if not ok_m:
        print(f"[ERROR] {err_m}")
        return 1
    print(f"[OK] CleanBTNVit checkpoint verified: {sha_m}")

    print("[Check 2/4] Verifying YuNet ONNX detector SHA-256...")
    ok_y, sha_y, err_y = verify_yunet_integrity(DEFAULT_YUNET_PATH)
    if not ok_y:
        print(f"[ERROR] {err_y}")
        return 1
    print(f"[OK] YuNet ONNX verified: {sha_y}")

    print("[Check 3/4] Verifying dataset manifest registry...")
    manifest = load_manifest(project_root)
    ds_keys = list(manifest.get("datasets", {}).keys())
    print(f"[OK] Loaded dataset_manifest.json ({len(ds_keys)} registered datasets: {', '.join(ds_keys)}).")

    print("[Check 4/4] Verifying CleanBTNVit PyTorch load + synthetic forward pass...")
    model = CleanBTNVitEvaluatorModel(weights_path=DEFAULT_MODEL_PATH, device="auto")
    model.warmup()
    print(f"[OK] CleanBTNVit loaded & warmed up cleanly on {model.device_name} ({model.precision_label}).")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VibeShift Standalone AI Accuracy Evaluator (CleanBTNVit Phase 12)."
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all registered datasets, splits, protocols, and independence status.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Verify model/detector SHA-256, imports, and dataset registry without evaluating any dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Registered dataset name (e.g., FER2013, ExpW_Eval1400, JAFFE, CREMA_D_Frames144, dataset) or folder path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (e.g., PrivateTest, PublicTest, DevHoldout4015, full_photo, annotated_crops, all).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "flip", "both"],
        default="single",
        help="Inference mode: 'single' (1 forward pass), 'flip' (horizontal-flip probability average), or 'both'.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["auto", "pre_cropped", "full_photo"],
        default="auto",
        help="Image evaluation protocol ('auto' selects dataset default; 'full_photo' runs YuNet detection).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device: 'auto' (CUDA if available else CPU), 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to clean_btn_vit_phase12.pt (default: model/clean_btn_vit_phase12.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Base directory where per-run evaluation folders are saved (default: results/).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        print(format_dataset_listing(PROJECT_ROOT))
        return 0

    if args.check_env:
        try:
            return run_environment_self_check(PROJECT_ROOT)
        except Exception as exc:
            print(f"[ERROR] Environment self-check failed: {exc}")
            return 1

    if not args.dataset:
        parser.print_help()
        print("\nTip: Run 'python evaluate.py --list-datasets' to see all available datasets and splits.")
        return 1

    try:
        loaded = load_dataset(PROJECT_ROOT, args.dataset, split_name=args.split)
    except (DatasetNotFoundError, UnsupportedAdapterError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if len(loaded.samples) == 0:
        print("==========================================")
        print(f" Dataset '{loaded.dataset_name}' (split '{loaded.split_name}') contains 0 images.")
        print("==========================================")
        if loaded.dataset_name == "dataset":
            print("Add labeled images (.jpg, .jpeg, .png, .bmp, .webp) into:")
            print("  dataset/angry/, dataset/disgust/, dataset/fear/, dataset/happy/,")
            print("  dataset/neutral/, dataset/sad/, dataset/surprise/")
            print("Then re-run: python evaluate.py --dataset dataset --mode single")
        else:
            print(f"Check dataset directory: {loaded.local_root}")
        return 2

    protocol = loaded.default_protocol if args.protocol == "auto" else args.protocol
    if loaded.warnings:
        for w in loaded.warnings:
            print(f"[NOTICE] {w}")

    if loaded.used_for_training:
        print("\n[WARNING] TRAINING DATA LEAKAGE NOTICE:")
        print(f"          {loaded.disclosure}\n")
    elif not loaded.suitable_for_independent_test:
        print(f"\n[NOTICE] INDEPENDENCE DISCLOSURE: {loaded.disclosure}\n")

    print(f"[1/3] Loading and SHA-256 verifying {MODEL_DISPLAY_NAME}...")
    try:
        model = CleanBTNVitEvaluatorModel(weights_path=args.weights, device=args.device)
        model.warmup()
    except Exception as exc:
        print(f"[ERROR] Failed to load CleanBTNVit model: {exc}")
        return 1
    print(f"[OK] Model loaded on {model.device_name} ({model.precision_label}) | SHA-256 {model.checkpoint_sha256[:12]}...")

    detector: Optional[YuNetFaceDetector] = None
    if protocol == "full_photo":
        print("[2/3] Loading YuNet ONNX face detector for full_photo protocol...")
        detector = YuNetFaceDetector(model_path=DEFAULT_YUNET_PATH)
        print(f"[OK] YuNet ready | SHA-256 {detector.sha256[:12]}...")
    else:
        print("[2/3] Pre-cropped protocol active (YuNet face detection skipped).")

    modes = ["single", "flip"] if args.mode == "both" else [args.mode]
    print(
        f"[3/3] Evaluating '{loaded.dataset_name}' / '{loaded.split_name}' "
        f"({len(loaded.samples)} images, protocol={protocol}, modes={modes})..."
    )

    for m in modes:
        metadata, metrics, _, _ = evaluate_loaded_dataset(
            loaded=loaded,
            model=model,
            mode=m,
            protocol=protocol,
            output_base_dir=args.output_dir.resolve(),
            detector=detector,
        )
        print()
        print(format_classification_report_text(metadata, metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())
