"""
Terminal output and structured artifact report generation for VibeShift-AI-Accuracy-Test.

Generates inside results/<run_id>/:
  - metrics.json
  - predictions.csv
  - confusion_matrix.csv
  - confusion_matrix.png
  - classification_report.txt
  - evaluation_summary.md
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.classifier import CANONICAL_LABELS
from .metrics import PredictionRecord


def make_run_id(dataset_name: str, split_name: str, mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    safe_ds = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
    safe_sp = "".join(c if c.isalnum() or c in "-_" else "_" for c in split_name)
    return f"{ts}_{safe_ds}_{safe_sp}_{mode}"


def format_confusion_matrix_text(cm: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    col_w = 10
    header = f"{'True \\ Pred':<12}" + "".join(f"{lbl[:9]:>{col_w}}" for lbl in labels)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, lbl in enumerate(labels):
        row_str = f"{lbl:<12}" + "".join(f"{int(cm[i][j]):>{col_w}}" for j in range(len(labels)))
        lines.append(row_str)
    return "\n".join(lines)


def format_classification_report_text(
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    labels = metrics["labels"]
    lines = [
        "==========================================",
        " VIBESHIFT - AI ACCURACY EVALUATION",
        "==========================================",
        "",
        f"Model: {metadata['model_name']}",
        f"Checkpoint SHA-256: {metadata['checkpoint_sha256']}",
        f"Device / Precision: {metadata['device']} ({metadata['precision']})",
        f"Dataset: {metadata['dataset_name']} ({metadata['dataset_display_name']})",
        f"Split: {metadata['split_name']}",
        f"Protocol: {metadata['protocol']}",
        f"Inference Mode: {metadata['inference_mode']}",
        f"Independence Status: {metadata['independence_status']}",
        "",
        f"Total Images: {metrics['total_images']}",
        f"Evaluated: {metrics['evaluated']}",
        f"Correct: {metrics['correct']}",
        f"Incorrect: {metrics['incorrect']}",
        f"Skipped: {metrics['skipped']}",
    ]

    if metrics["skipped"] > 0:
        sb = metrics["skip_breakdown"]
        lines.append(
            f"  Skip Breakdown: unreadable={sb['unreadable_image']}, "
            f"no_face={sb['no_face_detected']}, ambiguous_multi_face={sb['ambiguous_multi_face']}, "
            f"no_target_match={sb['no_target_match']}, preprocess_failed={sb['preprocess_failed']}"
        )

    lines.extend(
        [
            "",
            f"Accuracy: {metrics['accuracy_pct']:.2f}% ({metrics['correct']} / {metrics['evaluated']})",
            f"Macro F1 (active {metrics['active_ground_truth_classes']} classes): {metrics['macro_f1_pct']:.2f}%",
        ]
    )
    if metrics["active_ground_truth_classes"] < len(labels):
        lines.append(f"Macro F1 (all 7 classes): {metrics['macro_f1_all7_pct']:.2f}%")
    lines.append(f"Weighted F1: {metrics['weighted_f1_pct']:.2f}%")
    lines.append(
        f"Inference Latency: median {metrics['inference_ms_median']:.2f} ms | mean {metrics['inference_ms_mean']:.2f} ms"
    )

    if "full_photo_detection" in metrics:
        fp = metrics["full_photo_detection"]
        lines.extend(
            [
                "",
                "Full-Photograph YuNet Detection Disclosure:",
                f"  Readable Images: {fp['readable_images']}",
                f"  Valid Face Detections: {fp['valid_face_detections']} ({fp['detection_success_rate_pct']:.2f}%)",
                f"  Detection Failures (no face / no target match): {fp['detection_failures_total']} "
                f"(no_face={fp['no_face_detected']}, no_target_match={fp['no_target_match']})",
                f"  Ambiguous Multi-Face Images: {fp['ambiguous_multi_face']}",
                f"  Conditional Accuracy (valid detected faces): {fp['conditional_classification_accuracy_pct']:.2f}% ({fp['conditional_accuracy_fraction']})",
                f"  End-to-End Accuracy (all readable images):   {fp['end_to_end_accuracy_pct']:.2f}% ({fp['end_to_end_accuracy_fraction']})",
            ]
        )

    lines.extend(
        [
            "",
            "Per-Class Precision / Recall / F1 / Support:",
            f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
            "-" * 56,
        ]
    )
    for lbl in labels:
        pc = metrics["per_class"][lbl]
        lines.append(
            f"{lbl:<12} {pc['precision']:>9.2f}% {pc['recall']:>9.2f}% {pc['f1']:>9.2f}% {pc['support']:>10}"
        )

    lines.extend(
        [
            "",
            "Confusion Matrix (Rows = Ground Truth, Cols = Predicted):",
            format_confusion_matrix_text(metrics["confusion_matrix"], labels),
            "",
            f"Leakage / Independence Disclosure: {metadata['disclosure']}",
            f"Results saved to: {metadata['output_dir']}",
            "==========================================",
        ]
    )
    return "\n".join(lines)


def save_confusion_matrix_png(
    cm: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    arr = np.asarray(cm, dtype=np.int64)
    row_sums = arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, arr / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=150)
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Recall (%)", fontsize=9)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground-Truth Label", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    thresh = 50.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            count = int(arr[i, j])
            pct = float(norm[i, j])
            color = "white" if pct >= thresh else "#111827"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
            )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def write_evaluation_artifacts(
    output_dir: Path,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    records: Sequence[PredictionRecord],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["output_dir"] = str(output_dir)

    # 1. predictions.csv (complete per-image audit trail)
    pred_csv = output_dir / "predictions.csv"
    fieldnames = [
        "sample_id",
        "image_path",
        "ground_truth",
        "original_label",
        "predicted",
        "correct",
        "status",
        "detection_status",
        "num_faces_detected",
        "confidence",
        "inference_ms",
    ] + [f"prob_{lbl}" for lbl in CANONICAL_LABELS]

    with pred_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row: Dict[str, Any] = {
                "sample_id": r.sample_id,
                "image_path": r.image_path,
                "ground_truth": r.ground_truth,
                "original_label": r.original_label,
                "predicted": "" if r.predicted is None else r.predicted,
                "correct": "" if r.correct is None else str(r.correct),
                "status": r.status,
                "detection_status": r.detection_status,
                "num_faces_detected": "" if r.num_faces_detected is None else r.num_faces_detected,
                "confidence": "" if r.confidence is None else f"{r.confidence:.6f}",
                "inference_ms": f"{r.inference_ms:.3f}",
            }
            for lbl in CANONICAL_LABELS:
                row[f"prob_{lbl}"] = (
                    "" if r.probabilities is None else f"{r.probabilities[lbl]:.6f}"
                )
            writer.writerow(row)

    # 2. metrics.json
    metrics_json = output_dir / "metrics.json"
    payload = {"metadata": metadata, "metrics": metrics}
    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 3. confusion_matrix.csv
    cm_csv = output_dir / "confusion_matrix.csv"
    labels = metrics["labels"]
    with cm_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ground_truth\\predicted"] + list(labels))
        for i, lbl in enumerate(labels):
            w.writerow([lbl] + list(metrics["confusion_matrix"][i]))

    # 4. confusion_matrix.png
    cm_png = output_dir / "confusion_matrix.png"
    save_confusion_matrix_png(
        metrics["confusion_matrix"],
        labels,
        title=f"CleanBTNVit Phase 12 — {metadata['dataset_name']} / {metadata['split_name']} ({metadata['inference_mode']})",
        out_path=cm_png,
    )

    # 5. classification_report.txt
    report_txt = output_dir / "classification_report.txt"
    report_text = format_classification_report_text(metadata, metrics)
    report_txt.write_text(report_text + "\n", encoding="utf-8")

    # 6. evaluation_summary.md
    summary_md = output_dir / "evaluation_summary.md"
    md_lines = [
        f"# VibeShift AI Accuracy Evaluation — {metadata['dataset_name']} ({metadata['split_name']})",
        "",
        "## 1. Run Configuration & Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Timestamp** | `{metadata['timestamp']}` |",
        f"| **Model** | `{metadata['model_name']}` (single model, `output_VIT` only) |",
        f"| **Checkpoint SHA-256** | `{metadata['checkpoint_sha256']}` |",
        f"| **Device / Precision** | `{metadata['device']}` (`{metadata['precision']}`) |",
        f"| **Dataset** | `{metadata['dataset_name']}` — {metadata['dataset_display_name']} |",
        f"| **Provenance** | {metadata['provenance']} |",
        f"| **License Status** | {metadata['license_status']} |",
        f"| **Split** | `{metadata['split_name']}` |",
        f"| **Protocol** | `{metadata['protocol']}` |",
        f"| **Inference Mode** | `{metadata['inference_mode']}` |",
        f"| **Preprocessing** | `uint8 grayscale -> float32 [0,1] -> 112x112 bilinear (align_corners=False) -> mul(255).round().div(255) -> 3ch ImageNet mean/std` |",
        f"| **Class Mapping** | `{json.dumps(metadata['label_mapping'])}` |",
        "",
        "## 2. Dataset Leakage & Independence Disclosure",
        "",
        f"- **Independence Status:** `{metadata['independence_status']}`",
        f"- **Used in CleanBTNVit Phase 12 Training:** `{metadata['used_for_training']}`",
        f"- **Used for Checkpoint Selection:** `{metadata['used_for_checkpoint_selection']}`",
        f"- **Previously Inspected in Research:** `{metadata['previously_inspected']}`",
        f"- **Disclosure:** {metadata['disclosure']}",
        "",
        "## 3. Genuine Accuracy & Classification Metrics",
        "",
        f"- **Total Images:** `{metrics['total_images']}`",
        f"- **Evaluated Images:** `{metrics['evaluated']}`",
        f"- **Correct:** `{metrics['correct']}`",
        f"- **Incorrect:** `{metrics['incorrect']}`",
        f"- **Skipped / Excluded:** `{metrics['skipped']}` (`{json.dumps(metrics['skip_breakdown'])}`)",
        f"- **Accuracy (`Correct / Evaluated × 100`):** **`{metrics['accuracy_pct']:.2f}%`** (`{metrics['correct']} / {metrics['evaluated']}`)",
        f"- **Macro F1 (active {metrics['active_ground_truth_classes']} classes):** **`{metrics['macro_f1_pct']:.2f}%`**",
        f"- **Weighted F1:** **`{metrics['weighted_f1_pct']:.2f}%`**",
        f"- **Inference Latency:** median `{metrics['inference_ms_median']:.2f} ms`, mean `{metrics['inference_ms_mean']:.2f} ms`",
    ]

    if "full_photo_detection" in metrics:
        fp = metrics["full_photo_detection"]
        md_lines.extend(
            [
                "",
                "### Full-Photograph Face Detection Protocol Metrics (YuNet)",
                "",
                f"- **Readable Images:** `{fp['readable_images']}`",
                f"- **Face Detection Success Rate:** `{fp['detection_success_rate_pct']:.2f}%` (`{fp['valid_face_detections']} / {fp['readable_images']}`)",
                f"- **Detection Failures (`no_face_detected` + `no_target_match`):** `{fp['detection_failures_total']}`",
                f"- **Ambiguous Multi-Face Images:** `{fp['ambiguous_multi_face']}`",
                f"- **Conditional Classification Accuracy (valid detected faces):** `{fp['conditional_classification_accuracy_pct']:.2f}%` (`{fp['conditional_accuracy_fraction']}`)",
                f"- **End-to-End Accuracy (all readable images):** `{fp['end_to_end_accuracy_pct']:.2f}%` (`{fp['end_to_end_accuracy_fraction']}`)",
            ]
        )

    md_lines.extend(
        [
            "",
            "## 4. Per-Class Metrics",
            "",
            "| Emotion | Precision (%) | Recall (%) | F1-Score (%) | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for lbl in labels:
        pc = metrics["per_class"][lbl]
        md_lines.append(
            f"| `{lbl}` | {pc['precision']:.2f}% | {pc['recall']:.2f}% | {pc['f1']:.2f}% | {pc['support']} |"
        )

    md_lines.extend(
        [
            "",
            "## 5. Confusion Matrix (Rows = Ground Truth, Columns = Predicted)",
            "",
            "```text",
            format_confusion_matrix_text(metrics["confusion_matrix"], labels),
            "```",
            "",
            "![Confusion Matrix](confusion_matrix.png)",
            "",
            "## 6. Reproducibility",
            "",
            "All metrics in this report are computed directly from `predictions.csv` without post-hoc filtering or modification.",
        ]
    )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "predictions_csv": pred_csv,
        "metrics_json": metrics_json,
        "confusion_matrix_csv": cm_csv,
        "confusion_matrix_png": cm_png,
        "classification_report_txt": report_txt,
        "evaluation_summary_md": summary_md,
    }
