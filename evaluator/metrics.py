"""
Exact scientific classification and face-detection metrics for VibeShift-AI-Accuracy-Test.

Formulas:
  - Accuracy = (Correctly Classified Images / Total Evaluated Images) * 100
  - Per-class Precision = TP / (TP + FP)
  - Per-class Recall    = TP / (TP + FN)
  - Per-class F1        = 2 * P * R / (P + R)
  - Macro F1            = mean F1 across active ground-truth classes (and 7-class macro reported explicitly)
  - Weighted F1         = sum(support_c * F1_c) / sum(support_c)
  - Full-photo end-to-end accuracy = (Correct / Total Readable Images) * 100 (disclosed separately from
    conditional classification accuracy among valid detected faces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from model.classifier import CANONICAL_LABELS, LABEL_TO_INDEX


@dataclass
class PredictionRecord:
    sample_id: str
    image_path: str
    ground_truth: str
    original_label: str
    predicted: Optional[str]
    correct: Optional[bool]
    status: str  # "evaluated" | "unreadable_image" | "preprocess_failed" | "no_face_detected" | "ambiguous_multi_face" | "no_target_match"
    detection_status: str  # "not_applicable" | "annotated_box" | "detected_single_face" | "detected_target_match" | "no_face_detected" | "ambiguous_multi_face" | "no_target_match"
    num_faces_detected: Optional[int]
    confidence: Optional[float]
    inference_ms: float
    probabilities: Optional[Dict[str, float]]


def compute_metrics(
    records: Sequence[PredictionRecord],
    protocol: str = "pre_cropped",
    labels: Sequence[str] = CANONICAL_LABELS,
) -> Dict[str, Any]:
    label_list = list(labels)
    n_cls = len(label_list)
    idx_map = {lbl: i for i, lbl in enumerate(label_list)}

    total_images = len(records)
    evaluated_records = [r for r in records if r.status == "evaluated"]
    unreadable_count = sum(1 for r in records if r.status == "unreadable_image")
    preprocess_failed_count = sum(1 for r in records if r.status == "preprocess_failed")
    no_face_count = sum(1 for r in records if r.status == "no_face_detected")
    ambiguous_multi_face_count = sum(1 for r in records if r.status == "ambiguous_multi_face")
    no_target_match_count = sum(1 for r in records if r.status == "no_target_match")
    skipped_count = total_images - len(evaluated_records)

    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    correct_count = 0
    for r in evaluated_records:
        gt_i = idx_map[r.ground_truth]
        pr_i = idx_map[r.predicted]  # type: ignore[index]
        cm[gt_i, pr_i] += 1
        if gt_i == pr_i:
            correct_count += 1

    evaluated_count = len(evaluated_records)
    incorrect_count = evaluated_count - correct_count
    accuracy_pct = (correct_count / evaluated_count * 100.0) if evaluated_count > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    active_f1s: List[float] = []
    all_f1s: List[float] = []
    weighted_f1_num = 0.0

    for i, lbl in enumerate(label_list):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        pred_total = int(cm[:, i].sum())
        fp = pred_total - tp
        fn = support - tp

        precision = (tp / pred_total) if pred_total > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[lbl] = {
            "support": support,
            "predicted_count": pred_total,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision * 100.0, 4),
            "recall": round(recall * 100.0, 4),
            "f1": round(f1 * 100.0, 4),
        }
        all_f1s.append(f1)
        if support > 0:
            active_f1s.append(f1)
            weighted_f1_num += support * f1

    macro_f1_active_pct = (sum(active_f1s) / len(active_f1s) * 100.0) if active_f1s else 0.0
    macro_f1_all7_pct = (sum(all_f1s) / len(all_f1s) * 100.0) if all_f1s else 0.0
    weighted_f1_pct = (weighted_f1_num / evaluated_count * 100.0) if evaluated_count > 0 else 0.0

    # Latency stats over evaluated images
    latencies = [r.inference_ms for r in evaluated_records if r.inference_ms >= 0.0]
    median_ms = float(np.median(latencies)) if latencies else 0.0
    mean_ms = float(np.mean(latencies)) if latencies else 0.0

    result: Dict[str, Any] = {
        "labels": label_list,
        "protocol": protocol,
        "total_images": total_images,
        "evaluated": evaluated_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "skipped": skipped_count,
        "skip_breakdown": {
            "unreadable_image": unreadable_count,
            "preprocess_failed": preprocess_failed_count,
            "no_face_detected": no_face_count,
            "ambiguous_multi_face": ambiguous_multi_face_count,
            "no_target_match": no_target_match_count,
        },
        "accuracy_pct": round(accuracy_pct, 4),
        "accuracy_fraction": f"{correct_count}/{evaluated_count}",
        "macro_f1_pct": round(macro_f1_active_pct, 4),
        "macro_f1_all7_pct": round(macro_f1_all7_pct, 4),
        "active_ground_truth_classes": len(active_f1s),
        "weighted_f1_pct": round(weighted_f1_pct, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "inference_ms_mean": round(mean_ms, 3),
        "inference_ms_median": round(median_ms, 3),
    }

    if protocol == "full_photo":
        readable_total = total_images - unreadable_count
        detection_failures = no_face_count + no_target_match_count
        valid_detections = evaluated_count + preprocess_failed_count
        det_success_pct = (valid_detections / readable_total * 100.0) if readable_total > 0 else 0.0
        e2e_acc_pct = (correct_count / readable_total * 100.0) if readable_total > 0 else 0.0
        result["full_photo_detection"] = {
            "readable_images": readable_total,
            "valid_face_detections": valid_detections,
            "detection_success_rate_pct": round(det_success_pct, 4),
            "detection_failures_total": detection_failures,
            "no_face_detected": no_face_count,
            "no_target_match": no_target_match_count,
            "ambiguous_multi_face": ambiguous_multi_face_count,
            "conditional_classification_accuracy_pct": round(accuracy_pct, 4),
            "conditional_accuracy_fraction": f"{correct_count}/{evaluated_count}",
            "end_to_end_accuracy_pct": round(e2e_acc_pct, 4),
            "end_to_end_accuracy_fraction": f"{correct_count}/{readable_total}",
        }

    return result


def recompute_metrics_from_predictions_csv(csv_path, protocol: str = "pre_cropped") -> Dict[str, Any]:
    """Recomputes all metrics directly from a saved predictions.csv to verify reproducibility."""
    import csv as _csv

    records: List[PredictionRecord] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            status = row["status"]
            pred = row["predicted"] or None
            corr = None if row["correct"] == "" else (row["correct"].lower() == "true")
            conf = None if row["confidence"] == "" else float(row["confidence"])
            n_faces = None if row["num_faces_detected"] == "" else int(row["num_faces_detected"])
            probs = None
            if status == "evaluated":
                probs = {lbl: float(row[f"prob_{lbl}"]) for lbl in CANONICAL_LABELS}
            records.append(
                PredictionRecord(
                    sample_id=row["sample_id"],
                    image_path=row["image_path"],
                    ground_truth=row["ground_truth"],
                    original_label=row["original_label"],
                    predicted=pred,
                    correct=corr,
                    status=status,
                    detection_status=row["detection_status"],
                    num_faces_detected=n_faces,
                    confidence=conf,
                    inference_ms=float(row["inference_ms"] or 0.0),
                    probabilities=probs,
                )
            )
    return compute_metrics(records, protocol=protocol)
