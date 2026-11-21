"""
Standalone CleanBTNVit Phase 12 evaluator model wrapper.

Strictly single-model inference using ONLY the verified Phase 12 CleanBTNVit
checkpoint (`output_VIT` batch-independent pathway) and exact Phase 12
preprocessing. Zero dependencies on VibeShift-Final or VibeShift-local.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .btn_runtime.clean_btn_vit import CleanBTNVit
from .preprocess import preprocess_face_for_clean_btn

CANONICAL_LABELS: Tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
NUM_CLASSES: int = len(CANONICAL_LABELS)

MODEL_DISPLAY_NAME: str = "CleanBTNVit Phase 12"
EXPECTED_CHECKPOINT_SHA256: str = (
    "0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58"
)

DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent / "clean_btn_vit_phase12.pt"


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_git_lfs_pointer(path: Union[str, Path]) -> bool:
    p = Path(path)
    try:
        if p.stat().st_size > 1024:
            return False
        return p.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def verify_checkpoint_integrity(
    weights_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    expected_sha256: Optional[str] = EXPECTED_CHECKPOINT_SHA256,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verifies file existence, rejects Git LFS pointer files, and validates SHA-256.
    Returns (ok, actual_sha256, error_message).
    """
    path = Path(weights_path)
    if not path.is_file():
        return (
            False,
            None,
            (
                f"CleanBTNVit checkpoint not found at '{path}'. "
                "Obtain an authorized copy of clean_btn_vit_phase12.pt and place it in model/."
            ),
        )
    if is_git_lfs_pointer(path):
        return (
            False,
            None,
            f"File '{path}' is a Git LFS pointer file, not the 245.3 MiB CleanBTNVit weights.",
        )

    actual = sha256_file(path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        return (
            False,
            actual,
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}.",
        )
    return True, actual, None


class CleanBTNVitEvaluatorModel:
    """
    Exact standalone Phase 12 CleanBTNVit inference runtime supporting:
      - Strict SHA-256 verification and `weights_only=True` state_dict loading.
      - Single-view inference (1 forward pass) and exact horizontal-flip inference
        (0.5 * (softmax(logits) + softmax(logits_flip))).
      - CUDA FP16 autocast or CPU FP32 execution.
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: str = "auto",
        expected_sha256: Optional[str] = EXPECTED_CHECKPOINT_SHA256,
    ):
        self.weights_path = Path(weights_path).resolve()
        self.expected_sha256 = expected_sha256
        self.device = self._resolve_device(device)
        self.model: Optional[CleanBTNVit] = None
        self.checkpoint_sha256: Optional[str] = None
        self.is_loaded: bool = False
        self._load()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        req = (device or "auto").strip().lower()
        if req == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if req.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")
            return torch.device(req)
        if req == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'cuda'.")

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                return f"cuda ({torch.cuda.get_device_name(self.device)})"
            except Exception:
                return "cuda"
        return "cpu"

    @property
    def precision_label(self) -> str:
        return "FP16 autocast (CUDA)" if self.device.type == "cuda" else "FP32 (CPU)"

    def _load(self) -> None:
        ok, actual_sha, err = verify_checkpoint_integrity(
            self.weights_path, expected_sha256=self.expected_sha256
        )
        if not ok:
            raise RuntimeError(err)

        state_dict = torch.load(str(self.weights_path), map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state_dict mapping.")

        net = CleanBTNVit(head_drop=0.0)
        net.load_state_dict(state_dict, strict=True)
        net.to(self.device)
        net.eval()

        self.model = net
        self.checkpoint_sha256 = actual_sha
        self.is_loaded = True

    def warmup(self) -> None:
        dummy = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
        self.predict_tensor(dummy, horizontal_flip=False)

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        horizontal_flip: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Runs CleanBTNVit on a preprocessed tensor of shape (1, 3, 112, 112) or (B, 3, 112, 112).
        Returns (probs_np, inference_ms) where probs_np has shape (7,) for B=1 or (B, 7).
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("CleanBTNVit model is not loaded.")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1:] != (3, 112, 112):
            raise ValueError(f"Expected tensor shape (B, 3, 112, 112), got {tuple(tensor.shape)}")

        t0 = time.perf_counter()
        with torch.inference_mode():
            x = tensor.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(x).float()
                    probs = torch.softmax(logits, dim=1)
                    if horizontal_flip:
                        logits_flip = self.model(x.flip(-1).contiguous()).float()
                        probs = 0.5 * (probs + torch.softmax(logits_flip, dim=1))
                torch.cuda.synchronize(self.device)
            else:
                logits = self.model(x).float()
                probs = torch.softmax(logits, dim=1)
                if horizontal_flip:
                    logits_flip = self.model(x.flip(-1).contiguous()).float()
                    probs = 0.5 * (probs + torch.softmax(logits_flip, dim=1))

            probs_np = probs.detach().cpu().numpy().astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if probs_np.shape[0] == 1:
            return probs_np[0], elapsed_ms
        return probs_np, elapsed_ms

    def predict_face_image(
        self,
        face_image: np.ndarray,
        horizontal_flip: bool = False,
    ) -> Tuple[Optional[np.ndarray], float]:
        tensor = preprocess_face_for_clean_btn(face_image)
        if tensor is None:
            return None, 0.0
        return self.predict_tensor(tensor, horizontal_flip=horizontal_flip)


__all__ = [
    "CANONICAL_LABELS",
    "DEFAULT_MODEL_PATH",
    "EXPECTED_CHECKPOINT_SHA256",
    "LABEL_TO_INDEX",
    "MODEL_DISPLAY_NAME",
    "NUM_CLASSES",
    "CleanBTNVitEvaluatorModel",
    "is_git_lfs_pointer",
    "sha256_file",
    "verify_checkpoint_integrity",
]
