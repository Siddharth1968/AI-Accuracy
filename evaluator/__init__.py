"""Standalone dataset loading, metrics, and reporting package for VibeShift-AI-Accuracy-Test."""

from .adapters import (
    CREMAD_VALID_LABELS,
    EXPW_NUMERIC_TO_CANONICAL,
    JAFFE_CODE_TO_CANONICAL,
    SUPPORTED_LIBRARY_IMAGE_EXTS,
    SUPPORTED_MANUAL_IMAGE_EXTS,
    DatasetNotFoundError,
    EvalSample,
    LoadedDataset,
    UnsupportedAdapterError,
    format_dataset_listing,
    load_dataset,
    load_manifest,
)
from .metrics import PredictionRecord, compute_metrics, recompute_metrics_from_predictions_csv
from .reporting import (
    format_classification_report_text,
    format_confusion_matrix_text,
    make_run_id,
    write_evaluation_artifacts,
)

__all__ = [
    "CREMAD_VALID_LABELS",
    "EXPW_NUMERIC_TO_CANONICAL",
    "JAFFE_CODE_TO_CANONICAL",
    "SUPPORTED_LIBRARY_IMAGE_EXTS",
    "SUPPORTED_MANUAL_IMAGE_EXTS",
    "DatasetNotFoundError",
    "EvalSample",
    "LoadedDataset",
    "PredictionRecord",
    "UnsupportedAdapterError",
    "compute_metrics",
    "format_classification_report_text",
    "format_confusion_matrix_text",
    "format_dataset_listing",
    "load_dataset",
    "load_manifest",
    "make_run_id",
    "recompute_metrics_from_predictions_csv",
    "write_evaluation_artifacts",
]
