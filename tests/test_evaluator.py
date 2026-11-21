"""
Comprehensive unit, parity, and synthetic smoke test suite for VibeShift-AI-Accuracy-Test.

Strictly verifies software correctness using controlled synthetic inputs and dataset
metadata/file-count validation ONLY — never executes a scientific accuracy benchmark
on the discovered real datasets.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate as eval_cli
from detector import (
    DEFAULT_YUNET_PATH,
    YUNET_EXPECTED_SHA256,
    YuNetFaceDetector,
    compute_iou_xywh,
    verify_yunet_integrity,
)
from evaluator import (
    CREMAD_VALID_LABELS,
    EXPW_NUMERIC_TO_CANONICAL,
    JAFFE_CODE_TO_CANONICAL,
    DatasetNotFoundError,
    PredictionRecord,
    UnsupportedAdapterError,
    compute_metrics,
    format_dataset_listing,
    load_dataset,
    load_manifest,
    recompute_metrics_from_predictions_csv,
)
from model import (
    CANONICAL_LABELS,
    DEFAULT_MODEL_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    CleanBTNVitEvaluatorModel,
    is_git_lfs_pointer,
    preprocess_face_for_clean_btn,
    verify_checkpoint_integrity,
)


def _draw_synthetic_face(canvas: np.ndarray, cx: int, cy: int, scale: float = 1.0) -> None:
    rx, ry = int(90 * scale), int(120 * scale)
    cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, (180, 150, 130), -1)
    cv2.ellipse(canvas, (cx - int(35 * scale), cy - int(30 * scale)), (int(18 * scale), int(10 * scale)), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(canvas, (cx + int(35 * scale), cy - int(30 * scale)), (int(18 * scale), int(10 * scale)), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(canvas, (cx - int(35 * scale), cy - int(50 * scale)), (int(25 * scale), int(5 * scale)), 0, 0, 360, (40, 30, 20), -1)
    cv2.ellipse(canvas, (cx + int(35 * scale), cy - int(50 * scale)), (int(25 * scale), int(5 * scale)), 0, 0, 360, (40, 30, 20), -1)
    cv2.line(canvas, (cx, cy - int(25 * scale)), (cx, cy + int(20 * scale)), (140, 110, 90), max(1, int(3 * scale)))
    cv2.ellipse(canvas, (cx, cy + int(25 * scale)), (int(15 * scale), int(6 * scale)), 0, 0, 360, (140, 110, 90), -1)
    cv2.ellipse(canvas, (cx, cy + int(60 * scale)), (int(35 * scale), int(12 * scale)), 0, 0, 360, (70, 50, 150), -1)


def _make_single_face_portrait() -> np.ndarray:
    canvas = np.full((400, 400, 3), 220, dtype=np.uint8)
    _draw_synthetic_face(canvas, 200, 200, scale=1.0)
    return canvas


def _make_two_face_portrait() -> np.ndarray:
    canvas = np.full((420, 820, 3), 220, dtype=np.uint8)
    _draw_synthetic_face(canvas, 210, 210, scale=0.95)
    _draw_synthetic_face(canvas, 610, 210, scale=0.95)
    return canvas


def test_independent_path_isolation():
    leaked = [
        (name, str(getattr(mod, "__file__", "")))
        for name, mod in sys.modules.items()
        if getattr(mod, "__file__", None)
        and (
            "VibeShift-Final" in str(mod.__file__)
            or "VibeShift-local" in str(mod.__file__)
        )
    ]
    assert leaked == [], f"Evaluator leaked imports from external projects: {leaked}"


def test_checkpoint_and_yunet_sha256_verification():
    ok_m, sha_m, err_m = verify_checkpoint_integrity(DEFAULT_MODEL_PATH)
    assert ok_m is True and err_m is None
    assert sha_m == EXPECTED_CHECKPOINT_SHA256

    ok_y, sha_y, err_y = verify_yunet_integrity(DEFAULT_YUNET_PATH)
    assert ok_y is True and err_y is None
    assert sha_y == YUNET_EXPECTED_SHA256

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "corrupt.pt"
        bad.write_bytes(b"corrupted-model-bytes")
        ok_bad, _, err_bad = verify_checkpoint_integrity(bad)
        assert ok_bad is False and "SHA-256 mismatch" in (err_bad or "")

        lfs = Path(td) / "lfs_pointer.pt"
        lfs.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:0316ea2a\nsize 257223791\n",
            encoding="utf-8",
        )
        assert is_git_lfs_pointer(lfs) is True
        ok_lfs, _, err_lfs = verify_checkpoint_integrity(lfs)
        assert ok_lfs is False and "Git LFS pointer" in (err_lfs or "")


def test_preprocessing_exact_phase12_formula():
    sample_gray = np.arange(48 * 48, dtype=np.uint8).reshape(48, 48)
    sample_bgr = cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR)
    t_gray = preprocess_face_for_clean_btn(sample_gray)
    t_bgr = preprocess_face_for_clean_btn(sample_bgr)
    assert t_gray is not None and t_bgr is not None
    assert tuple(t_gray.shape) == (1, 3, 112, 112)
    assert torch.equal(t_gray, t_bgr)

    ref = torch.from_numpy(sample_gray).unsqueeze(0).unsqueeze(0).float() / 255.0
    ref = torch.nn.functional.interpolate(ref, size=(112, 112), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    ref = (ref.mul(255.0).round().div(255.0) - mean) / std
    assert (t_gray - ref).abs().max().item() == 0.0


def test_label_mappings_and_env_helper():
    assert CANONICAL_LABELS == ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
    # ExpW differs from canonical order on indices 4, 5, 6
    assert EXPW_NUMERIC_TO_CANONICAL[4] == "sad"
    assert EXPW_NUMERIC_TO_CANONICAL[5] == "surprise"
    assert EXPW_NUMERIC_TO_CANONICAL[6] == "neutral"
    assert set(JAFFE_CODE_TO_CANONICAL.values()) == set(CANONICAL_LABELS)
    assert CREMAD_VALID_LABELS == {"angry", "disgust", "fear", "happy", "neutral", "sad"}

    spec = importlib.util.spec_from_file_location("env_check", PROJECT_ROOT / "scripts" / "env_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_cuda_driver_ok("| NVIDIA-SMI 560.94  CUDA Version: 12.6 |") is True
    assert mod.parse_cuda_driver_ok("| NVIDIA-SMI 610.91  CUDA UMD Version: 13.3 |") is True
    assert mod.parse_cuda_driver_ok("| NVIDIA-SMI 525.00  CUDA Version: 11.8 |") is False


def test_dataset_registry_scanning_duplicates_and_class_counts():
    manifest = load_manifest(PROJECT_ROOT)
    assert "FER2013" in manifest["datasets"]
    assert "ExpW_Eval1400" in manifest["datasets"]
    assert "ExpW_Phase12_TrainClean5962" in manifest["datasets"]
    assert "JAFFE" in manifest["datasets"]
    assert "CREMA_D_Frames144" in manifest["datasets"]
    assert "dataset" in manifest["datasets"]

    listing = format_dataset_listing(PROJECT_ROOT)
    assert "FER2013" in listing and "PrivateTest" in listing and "JAFFE" in listing

    # Verify duplicate metadata is explicitly tracked
    assert len(manifest["datasets"]["FER2013"]["duplicate_locations"]) >= 1
    assert len(manifest["datasets"]["JAFFE"]["duplicate_locations"]) >= 1

    # Validate exact class counts on disk WITHOUT running model inference
    expected_checks = [
        ("FER2013", "PrivateTest", 3589, 7),
        ("FER2013", "PublicTest", 3589, 7),
        ("FER2013", "PublicTest_Clean3536", 3536, 7),
        ("FER2013", "DevHoldout4015", 4015, 7),
        ("FER2013", "Train22757", 22757, 7),
        ("ExpW_Eval1400", "full_photo", 1400, 7),
        ("ExpW_Eval1400", "annotated_crops", 1400, 7),
        ("ExpW_Phase12_TrainClean5962", "train_clean", 5962, 7),
        ("JAFFE", "all", 213, 7),
        ("CREMA_D_Frames144", "all", 144, 6),
    ]
    for ds_name, sp_name, exp_total, exp_active_classes in expected_checks:
        loaded = load_dataset(PROJECT_ROOT, ds_name, sp_name)
        assert len(loaded.samples) == exp_total, f"{ds_name}/{sp_name} expected {exp_total}, got {len(loaded.samples)}"
        counts = Counter(s.ground_truth for s in loaded.samples)
        assert len(counts) == exp_active_classes
        expected_cc = manifest["datasets"][ds_name]["splits"][sp_name]["class_counts"]
        for lbl, c in expected_cc.items():
            assert counts.get(lbl, 0) == c, f"Mismatch for {ds_name}/{sp_name} class {lbl}"

    # Verify unsupported dataset raises cleanly
    with pytest.raises(DatasetNotFoundError):
        load_dataset(PROJECT_ROOT, "NON_EXISTENT_DATASET")

    # Verify unknown adapter raises UnsupportedAdapterError
    with tempfile.TemporaryDirectory() as td:
        fake_root = Path(td)
        (fake_root / "datasets").mkdir()
        (fake_root / "datasets" / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "datasets": {
                        "FakeDS": {
                            "adapter": "unknown_magic_adapter",
                            "local_path": "datasets/fake",
                            "splits": {"all": {}},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(UnsupportedAdapterError):
            load_dataset(fake_root, "FakeDS", "all")


def test_empty_manual_dataset_handling():
    loaded = load_dataset(PROJECT_ROOT, "dataset")
    assert len(loaded.samples) == 0
    rc = eval_cli.main(["--dataset", "dataset", "--mode", "single"])
    assert rc == 2


def test_metrics_accuracy_precision_recall_f1_and_confusion_matrix():
    # Known synthetic records: 4 evaluated (3 correct, 1 incorrect) + 1 unreadable + 1 no_face
    def _rec(sid, gt, pred, status="evaluated", det="detected_single_face"):
        probs = {c: (0.8 if c == pred else 0.2 / 6.0) for c in CANONICAL_LABELS} if status == "evaluated" else None
        return PredictionRecord(
            sample_id=sid,
            image_path=f"synth/{sid}",
            ground_truth=gt,
            original_label=gt,
            predicted=pred if status == "evaluated" else None,
            correct=(gt == pred) if status == "evaluated" else None,
            status=status,
            detection_status=det,
            num_faces_detected=1 if status == "evaluated" else 0,
            confidence=0.8 if status == "evaluated" else None,
            inference_ms=10.0 if status == "evaluated" else 0.0,
            probabilities=probs,
        )

    records = [
        _rec("1", "happy", "happy"),
        _rec("2", "happy", "happy"),
        _rec("3", "sad", "sad"),
        _rec("4", "sad", "happy"),
        _rec("5", "angry", None, status="no_face_detected", det="no_face_detected"),
        _rec("6", "neutral", None, status="unreadable_image", det="not_applicable"),
    ]
    m = compute_metrics(records, protocol="full_photo")
    assert m["total_images"] == 6
    assert m["evaluated"] == 4
    assert m["correct"] == 3
    assert m["incorrect"] == 1
    assert m["skipped"] == 2
    assert m["accuracy_pct"] == pytest.approx(75.0)
    assert m["accuracy_fraction"] == "3/4"

    # happy: TP=2, FP=1, FN=0 -> P=2/3, R=1.0, F1=4/5=80%
    assert m["per_class"]["happy"]["precision"] == pytest.approx(66.6667, abs=1e-3)
    assert m["per_class"]["happy"]["recall"] == pytest.approx(100.0)
    assert m["per_class"]["happy"]["f1"] == pytest.approx(80.0)
    # sad: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3=66.6667%
    assert m["per_class"]["sad"]["precision"] == pytest.approx(100.0)
    assert m["per_class"]["sad"]["recall"] == pytest.approx(50.0)
    assert m["per_class"]["sad"]["f1"] == pytest.approx(66.6667, abs=1e-3)
    # Active classes = 2 (happy, sad) -> macro F1 = (80 + 66.6667)/2 = 73.3333%
    assert m["active_ground_truth_classes"] == 2
    assert m["macro_f1_pct"] == pytest.approx(73.3333, abs=1e-3)
    assert m["weighted_f1_pct"] == pytest.approx(73.3333, abs=1e-3)

    # Full-photo detection metrics: readable=5, valid=4 (80%), conditional=3/4 (75%), end-to-end=3/5 (60%)
    fp = m["full_photo_detection"]
    assert fp["readable_images"] == 5
    assert fp["detection_success_rate_pct"] == pytest.approx(80.0)
    assert fp["conditional_classification_accuracy_pct"] == pytest.approx(75.0)
    assert fp["end_to_end_accuracy_pct"] == pytest.approx(60.0)

    # Zero-evaluated safety
    m_empty = compute_metrics([], protocol="pre_cropped")
    assert m_empty["accuracy_pct"] == 0.0 and m_empty["macro_f1_pct"] == 0.0


def test_model_cpu_and_cuda_smoke_single_flip_and_batch_independence():
    # CPU smoke test
    model_cpu = CleanBTNVitEvaluatorModel(weights_path=DEFAULT_MODEL_PATH, device="cpu")
    rng = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn((1, 3, 112, 112), generator=rng)

    p_orig, ms_single = model_cpu.predict_tensor(x, horizontal_flip=False)
    p_flipped_only, _ = model_cpu.predict_tensor(x.flip(-1).contiguous(), horizontal_flip=False)
    p_flip_avg, ms_flip = model_cpu.predict_tensor(x, horizontal_flip=True)

    assert p_orig.shape == (7,) and abs(float(p_orig.sum()) - 1.0) < 1e-4
    assert p_flip_avg.shape == (7,) and abs(float(p_flip_avg.sum()) - 1.0) < 1e-4
    assert ms_single > 0 and ms_flip > 0
    expected_avg = 0.5 * (p_orig + p_flipped_only)
    assert np.max(np.abs(p_flip_avg - expected_avg)) < 1e-5

    # Batch-independence check on output_VIT
    batch_x = torch.randn((3, 3, 112, 112), generator=rng)
    peers = batch_x.clone()
    peers[1:] = -3.0 * peers[1:]
    with torch.inference_mode():
        out_a = model_cpu.model(batch_x)[0]
        out_b = model_cpu.model(peers)[0]
        out_1 = model_cpu.model(batch_x[:1])[0]
    assert (out_a - out_b).abs().max().item() < 1e-5
    assert (out_a - out_1).abs().max().item() < 1e-4

    # CUDA smoke test if available
    if torch.cuda.is_available():
        model_cuda = CleanBTNVitEvaluatorModel(weights_path=DEFAULT_MODEL_PATH, device="cuda")
        p_cuda_s, _ = model_cuda.predict_tensor(x, horizontal_flip=False)
        p_cuda_f, _ = model_cuda.predict_tensor(x, horizontal_flip=True)
        assert p_cuda_s.shape == (7,) and abs(float(p_cuda_s.sum()) - 1.0) < 1e-3
        assert int(np.argmax(p_cuda_s)) == int(np.argmax(p_orig))
        assert int(np.argmax(p_cuda_f)) == int(np.argmax(p_flip_avg))


def test_full_end_to_end_synthetic_pre_cropped_and_full_photo_reports_and_reproducibility():
    """
    Tests invalid-image handling, single-view + horizontal-flip, full-photo YuNet single/no/multi-face
    disclosure, all 6 output artifacts, and exact reproducibility from predictions.csv using
    strictly synthetic images in a temporary directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CleanBTNVitEvaluatorModel(weights_path=DEFAULT_MODEL_PATH, device=device)
    detector = YuNetFaceDetector(model_path=DEFAULT_YUNET_PATH)

    assert compute_iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert compute_iou_xywh((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ds_dir = root / "synth_dataset"
        for c in CANONICAL_LABELS:
            (ds_dir / c).mkdir(parents=True)

        # 1. Valid synthetic single-face portrait
        single_face = _make_single_face_portrait()
        cv2.imwrite(str(ds_dir / "happy" / "single_face.jpg"), single_face)
        # 2. Blank image (no face)
        cv2.imwrite(str(ds_dir / "neutral" / "blank.png"), np.zeros((240, 240, 3), dtype=np.uint8))
        # 3. Two-face synthetic image (ambiguous multi-face -> must NOT guess randomly!)
        two_face = _make_two_face_portrait()
        cv2.imwrite(str(ds_dir / "sad" / "two_faces.jpg"), two_face)
        # 4. Corrupt unreadable image
        (ds_dir / "angry" / "corrupt.jpg").write_bytes(b"not-valid-jpeg-bytes")

        loaded = load_dataset(PROJECT_ROOT, str(ds_dir))
        assert len(loaded.samples) == 4

        out_base = root / "results"
        # Full-photo YuNet run (single mode)
        meta_fp, met_fp, recs_fp, run_dir_fp = eval_cli.evaluate_loaded_dataset(
            loaded=loaded,
            model=model,
            mode="single",
            protocol="full_photo",
            output_base_dir=out_base,
            detector=detector,
            quiet=True,
        )
        assert met_fp["total_images"] == 4
        assert met_fp["skip_breakdown"]["unreadable_image"] == 1
        assert met_fp["skip_breakdown"]["no_face_detected"] == 1
        assert met_fp["skip_breakdown"]["ambiguous_multi_face"] == 1
        assert met_fp["evaluated"] == 1
        assert "full_photo_detection" in met_fp

        # Verify all 6 artifacts created
        for fname in (
            "metrics.json",
            "predictions.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "classification_report.txt",
            "evaluation_summary.md",
        ):
            fp = run_dir_fp / fname
            assert fp.is_file() and fp.stat().st_size > 0, f"Missing or empty artifact: {fname}"

        # Reproducibility check: recomputing metrics directly from predictions.csv must match metrics.json
        recomputed = recompute_metrics_from_predictions_csv(
            run_dir_fp / "predictions.csv", protocol="full_photo"
        )
        assert recomputed["accuracy_pct"] == met_fp["accuracy_pct"]
        assert recomputed["macro_f1_pct"] == met_fp["macro_f1_pct"]
        assert recomputed["confusion_matrix"] == met_fp["confusion_matrix"]
        assert recomputed["full_photo_detection"] == met_fp["full_photo_detection"]

        # Pre-cropped horizontal-flip run (evaluates the 3 readable images, skips 1 corrupt)
        _, met_pc, _, run_dir_pc = eval_cli.evaluate_loaded_dataset(
            loaded=loaded,
            model=model,
            mode="flip",
            protocol="pre_cropped",
            output_base_dir=out_base,
            quiet=True,
        )
        assert met_pc["total_images"] == 4
        assert met_pc["evaluated"] == 3
        assert met_pc["skip_breakdown"]["unreadable_image"] == 1
        assert (run_dir_pc / "confusion_matrix.png").is_file()
