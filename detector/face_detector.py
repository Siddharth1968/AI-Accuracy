"""
Standalone OpenCV YuNet ONNX face detector for full-photograph evaluation protocols.
Uses detector/yunet-2023mar.onnx with strict SHA-256 verification.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

YUNET_EXPECTED_SHA256: str = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEFAULT_YUNET_PATH: Path = Path(__file__).resolve().parent / "yunet-2023mar.onnx"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_yunet_integrity(
    model_path: Path = DEFAULT_YUNET_PATH,
    expected_sha256: Optional[str] = YUNET_EXPECTED_SHA256,
) -> Tuple[bool, Optional[str], Optional[str]]:
    if not model_path.is_file():
        return False, None, f"YuNet ONNX model not found at '{model_path}'."
    actual = sha256_file(model_path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        return False, actual, f"YuNet SHA-256 mismatch: expected {expected_sha256}, got {actual}."
    return True, actual, None


def compute_iou_xywh(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Computes IoU between two (x, y, w, h) bounding boxes."""
    ax1, ay1, aw, ah = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bw, bh = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


class YuNetFaceDetector:
    """
    Standalone OpenCV YuNet ONNX face detector (`yunet-2023mar.onnx`).
    Supports:
      - Unambiguous single-face detection for full photographs (no random guessing when >1 face).
      - Ground-truth bounding-box IoU matching for multi-face scenes with annotated target boxes
        (such as ExpW, using `target_match_iou_min=0.3`).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.9,
        fallback_threshold: Optional[float] = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_side: int = 1280,
        expected_sha256: Optional[str] = YUNET_EXPECTED_SHA256,
    ):
        self.model_path = Path(model_path).resolve() if model_path is not None else DEFAULT_YUNET_PATH
        self.score_threshold = float(score_threshold)
        self.fallback_threshold = None if fallback_threshold is None else float(fallback_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_side = int(max_side)
        self._lock = threading.Lock()
        self.detector: Optional[Any] = None
        self.is_loaded: bool = False
        self.sha256: Optional[str] = None

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Installed OpenCV build lacks cv2.FaceDetectorYN (YuNet) support.")

        ok, actual_sha, err = verify_yunet_integrity(self.model_path, expected_sha256=expected_sha256)
        if not ok:
            raise RuntimeError(err)

        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )
        self.sha256 = actual_sha
        self.is_loaded = True

    def _run_once(self, image_bgr: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        orig_h, orig_w = image_bgr.shape[:2]
        scale = 1.0
        work = image_bgr
        if self.max_side > 0 and max(orig_h, orig_w) > self.max_side:
            scale = self.max_side / float(max(orig_h, orig_w))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            work = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        wh, ww = work.shape[:2]
        self.detector.setInputSize((ww, wh))
        self.detector.setScoreThreshold(float(threshold))
        _, raw_faces = self.detector.detect(work)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        inv = 1.0 / scale
        min_dim = 16 if min(orig_w, orig_h) < 96 else 24
        detections: List[Dict[str, Any]] = []
        for row in raw_faces:
            fx, fy, fw, fh = float(row[0]) * inv, float(row[1]) * inv, float(row[2]) * inv, float(row[3]) * inv
            score = float(row[-1])
            x1 = max(0, int(round(fx)))
            y1 = max(0, int(round(fy)))
            x2 = min(orig_w, int(round(fx + fw)))
            y2 = min(orig_h, int(round(fy + fh)))
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_dim or bh < min_dim:
                continue
            detections.append({"bbox": (x1, y1, bw, bh), "score": score})

        detections.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        return detections

    def detect_faces(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return [], self.score_threshold

        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            image_bgr = image
        elif image.ndim == 3 and image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            return [], self.score_threshold

        with self._lock:
            dets = self._run_once(image_bgr, self.score_threshold)
            if dets or self.fallback_threshold is None:
                return dets, self.score_threshold
            dets = self._run_once(image_bgr, self.fallback_threshold)
            return dets, self.fallback_threshold

    def select_evaluation_face(
        self,
        image: np.ndarray,
        target_box_xywh: Optional[Tuple[int, int, int, int]] = None,
        target_iou_min: float = 0.3,
    ) -> Tuple[str, Optional[Tuple[int, int, int, int]], int]:
        """
        Selects the face to classify under strict scientific full-photo rules:
          - If target_box_xywh is provided (e.g., ExpW annotated target): select the highest-IoU
            YuNet detection with IoU >= target_iou_min; otherwise report 'no_target_match' / 'no_face_detected'.
          - If NO target_box_xywh is provided:
              * 0 faces -> ('no_face_detected', None, 0)
              * 1 face  -> ('detected_single_face', bbox, 1)
              * >1 face -> ('ambiguous_multi_face', None, n)  (never guesses a random face!)
        """
        dets, _ = self.detect_faces(image)
        n = len(dets)
        if n == 0:
            return "no_face_detected", None, 0

        if target_box_xywh is not None:
            best_det = None
            best_iou = -1.0
            for d in dets:
                iou = compute_iou_xywh(d["bbox"], target_box_xywh)
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            if best_det is not None and best_iou >= target_iou_min:
                return "detected_target_match", best_det["bbox"], n
            return "no_target_match", None, n

        if n == 1:
            return "detected_single_face", dets[0]["bbox"], 1
        return "ambiguous_multi_face", None, n


__all__ = [
    "DEFAULT_YUNET_PATH",
    "YUNET_EXPECTED_SHA256",
    "YuNetFaceDetector",
    "compute_iou_xywh",
    "verify_yunet_integrity",
]
