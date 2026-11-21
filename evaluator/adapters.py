"""
Explicit dataset registry and annotation adapters for VibeShift-AI-Accuracy-Test.

Every supported dataset format uses an explicit, documented adapter — ground-truth
labels are never guessed from arbitrary filenames unless the dataset specification
explicitly defines that convention (e.g., JAFFE official README_FIRST.txt).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from model.classifier import CANONICAL_LABELS

SUPPORTED_MANUAL_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_LIBRARY_IMAGE_EXTS = SUPPORTED_MANUAL_IMAGE_EXTS | {".tif", ".tiff"}

# Explicit ExpW official numeric label order (from expw-readme.txt)
EXPW_NUMERIC_TO_CANONICAL: Dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Explicit JAFFE official pose-code convention (from README_FIRST.txt)
JAFFE_CODE_TO_CANONICAL: Dict[str, str] = {
    "AN": "angry",
    "DI": "disgust",
    "FE": "fear",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise",
}
JAFFE_FILENAME_RE = re.compile(r"^[A-Z]{2}\.(AN|DI|FE|HA|NE|SA|SU)\d+\.\d+\.tiff?$", re.IGNORECASE)

# Explicit CREMA-D 6-class label set (surprise is absent)
CREMAD_VALID_LABELS = {"angry", "disgust", "fear", "happy", "neutral", "sad"}


class UnsupportedAdapterError(ValueError):
    """Raised when a manifest dataset specifies an adapter that is not implemented."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset or split is missing from disk or from dataset_manifest.json."""


@dataclass
class EvalSample:
    sample_id: str
    display_path: str
    ground_truth: str
    original_label: str
    image_path: Optional[Path] = None
    image_array: Optional[np.ndarray] = None
    target_box_xywh: Optional[Tuple[int, int, int, int]] = None
    crop_from_target_box: bool = False


@dataclass
class LoadedDataset:
    dataset_name: str
    display_name: str
    split_name: str
    adapter: str
    local_root: Path
    default_protocol: str
    supported_protocols: List[str]
    samples: List[EvalSample]
    label_mapping: Dict[str, str]
    provenance: str
    license_status: str
    independence_status: str
    suitable_for_independent_test: bool
    used_for_training: bool
    used_for_checkpoint_selection: bool
    previously_inspected: bool
    disclosure: str
    warnings: List[str] = field(default_factory=list)


def load_manifest(project_root: Path) -> Dict[str, Any]:
    manifest_path = project_root / "datasets" / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise DatasetNotFoundError(f"Dataset manifest not found at '{manifest_path}'.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _scan_image_folder(
    folder: Path,
    allowed_exts: set[str] = SUPPORTED_MANUAL_IMAGE_EXTS,
    prefix_filter: Optional[str] = None,
) -> Tuple[List[EvalSample], List[str]]:
    if not folder.is_dir():
        raise DatasetNotFoundError(f"Dataset folder not found: '{folder}'")

    warnings: List[str] = []
    samples: List[EvalSample] = []
    canonical_set = set(CANONICAL_LABELS)

    for entry in sorted(folder.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            continue
        cls_name = entry.name.lower()
        if cls_name not in canonical_set:
            warnings.append(f"Ignored unrecognized class directory: '{entry.name}'")
            continue
        for img_file in sorted(entry.iterdir()):
            if not img_file.is_file() or img_file.name.startswith("."):
                continue
            if img_file.suffix.lower() not in allowed_exts:
                warnings.append(f"Ignored unsupported file extension: '{cls_name}/{img_file.name}'")
                continue
            if prefix_filter and not img_file.name.startswith(prefix_filter):
                continue
            rel = f"{cls_name}/{img_file.name}"
            samples.append(
                EvalSample(
                    sample_id=rel,
                    display_path=str(img_file),
                    ground_truth=cls_name,
                    original_label=entry.name,
                    image_path=img_file,
                )
            )
    return samples, warnings


def _load_fer2013_split(
    dataset_root: Path,
    split_cfg: Dict[str, Any],
) -> Tuple[List[EvalSample], List[str]]:
    mode = split_cfg.get("mode")
    if mode == "prefix":
        subdir = dataset_root / split_cfg["subdir"]
        return _scan_image_folder(
            subdir,
            allowed_exts=SUPPORTED_LIBRARY_IMAGE_EXTS,
            prefix_filter=split_cfg["prefix"],
        )
    if mode == "folder":
        subdir = dataset_root / split_cfg["subdir"]
        return _scan_image_folder(subdir, allowed_exts=SUPPORTED_LIBRARY_IMAGE_EXTS)
    if mode == "csv":
        csv_path = dataset_root / split_cfg["csv_path"]
        if not csv_path.is_file():
            raise DatasetNotFoundError(f"FER2013 split CSV missing: '{csv_path}'")
        want_split = split_cfg["csv_split_value"]
        samples: List[EvalSample] = []
        warnings: List[str] = []
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("split") != want_split:
                    continue
                rel_path = row["path"].replace("\\", "/")
                label = row["label"].strip().lower()
                if label not in CANONICAL_LABELS:
                    warnings.append(f"Skipped invalid label '{label}' in {csv_path.name}")
                    continue
                full_path = dataset_root / Path(rel_path)
                samples.append(
                    EvalSample(
                        sample_id=rel_path,
                        display_path=str(full_path),
                        ground_truth=label,
                        original_label=label,
                        image_path=full_path,
                    )
                )
        return samples, warnings
    raise UnsupportedAdapterError(f"Unsupported FER2013 split mode: '{mode}'")


def _load_expw_manifest_split(
    dataset_root: Path,
    split_cfg: Dict[str, Any],
) -> Tuple[List[EvalSample], List[str]]:
    manifest_file = dataset_root / "sample-manifest.json"
    if not manifest_file.is_file():
        raise DatasetNotFoundError(f"ExpW sample-manifest.json missing at '{manifest_file}'")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    crop_mode = split_cfg.get("crop_mode", "yunet_full_photo")
    crop_from_box = crop_mode == "annotated_box"
    origin_dir = dataset_root / "images" / "origin"

    samples: List[EvalSample] = []
    warnings: List[str] = []
    for item in data.get("samples", []):
        expw_idx = int(item["expw_label"])
        mapped_label = EXPW_NUMERIC_TO_CANONICAL.get(expw_idx)
        declared_label = str(item.get("label", "")).strip().lower()
        if mapped_label is None or mapped_label != declared_label:
            raise ValueError(
                f"ExpW label mismatch on line {item.get('annotation_line')}: "
                f"expw_label={expw_idx} -> {mapped_label}, manifest label={declared_label}"
            )
        box = item.get("box")
        box_tuple = (int(box[0]), int(box[1]), int(box[2]), int(box[3])) if box and len(box) == 4 else None
        img_name = item["image"]
        img_path = origin_dir / img_name
        samples.append(
            EvalSample(
                sample_id=f"{img_name}#face{item.get('face_id', 0)}",
                display_path=str(img_path),
                ground_truth=mapped_label,
                original_label=f"expw_{expw_idx}_{mapped_label}",
                image_path=img_path,
                target_box_xywh=box_tuple,
                crop_from_target_box=crop_from_box,
            )
        )
    return samples, warnings


def _load_expw_npz_split(
    dataset_root: Path,
    split_cfg: Dict[str, Any],
) -> Tuple[List[EvalSample], List[str]]:
    npz_path = dataset_root / split_cfg["npz_file"]
    if not npz_path.is_file():
        raise DatasetNotFoundError(f"ExpW NPZ archive missing at '{npz_path}'")
    with np.load(str(npz_path), allow_pickle=False) as data:
        images = data["images"]
        labels = data["labels"]

    samples: List[EvalSample] = []
    for idx in range(len(labels)):
        lbl_idx = int(labels[idx])
        if not (0 <= lbl_idx < len(CANONICAL_LABELS)):
            raise ValueError(f"Invalid canonical label index {lbl_idx} at row {idx} in {npz_path.name}")
        lbl = CANONICAL_LABELS[lbl_idx]
        samples.append(
            EvalSample(
                sample_id=f"{npz_path.name}[{idx:05d}]",
                display_path=f"{npz_path}#{idx}",
                ground_truth=lbl,
                original_label=str(lbl_idx),
                image_array=images[idx],
            )
        )
    return samples, []


def _load_jaffe_split(
    dataset_root: Path,
    split_cfg: Dict[str, Any],
) -> Tuple[List[EvalSample], List[str]]:
    img_dir = dataset_root / split_cfg.get("subdir", "images")
    if not img_dir.is_dir():
        raise DatasetNotFoundError(f"JAFFE images directory missing at '{img_dir}'")

    samples: List[EvalSample] = []
    warnings: List[str] = []
    for img_file in sorted(img_dir.iterdir()):
        if not img_file.is_file() or img_file.name.startswith("."):
            continue
        m = JAFFE_FILENAME_RE.match(img_file.name)
        if not m:
            warnings.append(f"Skipped file not matching official JAFFE convention: '{img_file.name}'")
            continue
        code = m.group(1).upper()
        lbl = JAFFE_CODE_TO_CANONICAL[code]
        samples.append(
            EvalSample(
                sample_id=img_file.name,
                display_path=str(img_file),
                ground_truth=lbl,
                original_label=code,
                image_path=img_file,
            )
        )
    return samples, warnings


def _load_cremad_split(
    dataset_root: Path,
    split_cfg: Dict[str, Any],
) -> Tuple[List[EvalSample], List[str]]:
    manifest_file = dataset_root / split_cfg.get("manifest_file", "frame-manifest.json")
    if not manifest_file.is_file():
        raise DatasetNotFoundError(f"CREMA-D frame-manifest.json missing at '{manifest_file}'")
    entries = json.loads(manifest_file.read_text(encoding="utf-8"))
    frames_dir = dataset_root / "frames"

    samples: List[EvalSample] = []
    warnings: List[str] = [
        "CREMA-D contains 6 emotion classes only (angry, disgust, fear, happy, neutral, sad); 'surprise' has 0 ground-truth samples."
    ]
    for item in entries:
        lbl = str(item["label"]).strip().lower()
        if lbl not in CREMAD_VALID_LABELS:
            raise ValueError(f"Unexpected CREMA-D label '{lbl}' for id '{item.get('id')}'")
        frame_name = Path(item["frame"]).name
        frame_path = frames_dir / frame_name
        samples.append(
            EvalSample(
                sample_id=item["id"],
                display_path=str(frame_path),
                ground_truth=lbl,
                original_label=lbl,
                image_path=frame_path,
            )
        )
    return samples, warnings


def load_dataset(
    project_root: Path,
    dataset_name: str,
    split_name: Optional[str] = None,
) -> LoadedDataset:
    """
    Loads a registered dataset & split from `datasets/dataset_manifest.json`,
    OR loads a user-supplied custom 7-class folder path directly.
    """
    manifest = load_manifest(project_root)
    datasets_cfg: Dict[str, Any] = manifest.get("datasets", {})

    # Allow case-insensitive lookup against registered dataset keys
    matched_key = None
    for k in datasets_cfg:
        if k.lower() == dataset_name.lower():
            matched_key = k
            break

    if matched_key is None:
        # Check if user passed a direct path to an unregistered 7-class image folder
        custom_path = Path(dataset_name)
        if not custom_path.is_absolute():
            custom_path = (project_root / custom_path).resolve()
        if custom_path.is_dir():
            samples, warnings = _scan_image_folder(custom_path, allowed_exts=SUPPORTED_MANUAL_IMAGE_EXTS)
            return LoadedDataset(
                dataset_name=custom_path.name,
                display_name=f"Custom Folder ({custom_path})",
                split_name=split_name or "default",
                adapter="image_folder",
                local_root=custom_path,
                default_protocol="pre_cropped",
                supported_protocols=["pre_cropped", "full_photo"],
                samples=samples,
                label_mapping={c: c for c in CANONICAL_LABELS},
                provenance="Unregistered custom local folder",
                license_status="User-supplied / unknown",
                independence_status="UNVERIFIED_CUSTOM_FOLDER",
                suitable_for_independent_test=True,
                used_for_training=False,
                used_for_checkpoint_selection=False,
                previously_inspected=False,
                disclosure="Custom unregistered directory. Verify provenance and independence manually.",
                warnings=warnings,
            )
        available = ", ".join(sorted(datasets_cfg.keys()))
        raise DatasetNotFoundError(
            f"Dataset '{dataset_name}' not found in dataset_manifest.json (available: {available})."
        )

    cfg = datasets_cfg[matched_key]
    splits = cfg.get("splits", {})
    if not splits:
        raise DatasetNotFoundError(f"Dataset '{matched_key}' defines no splits in dataset_manifest.json.")

    if split_name is None:
        if len(splits) == 1:
            resolved_split = next(iter(splits.keys()))
        elif "PrivateTest" in splits:
            raise ValueError(
                f"Dataset '{matched_key}' has multiple splits: {', '.join(splits.keys())}. "
                "Please specify --split explicitly (e.g., --split PrivateTest)."
            )
        else:
            raise ValueError(
                f"Dataset '{matched_key}' has multiple splits: {', '.join(splits.keys())}. "
                "Please specify --split explicitly."
            )
    else:
        resolved_split = None
        for sk in splits:
            if sk.lower() == split_name.lower():
                resolved_split = sk
                break
        if resolved_split is None:
            raise DatasetNotFoundError(
                f"Split '{split_name}' not found for dataset '{matched_key}'. "
                f"Available splits: {', '.join(splits.keys())}"
            )

    split_cfg = splits[resolved_split]
    local_root = (project_root / cfg["local_path"]).resolve()
    adapter = cfg.get("adapter")

    if adapter == "image_folder":
        samples, warnings = _scan_image_folder(local_root, allowed_exts=SUPPORTED_MANUAL_IMAGE_EXTS)
    elif adapter == "fer2013":
        samples, warnings = _load_fer2013_split(local_root, split_cfg)
    elif adapter == "expw_manifest":
        samples, warnings = _load_expw_manifest_split(local_root, split_cfg)
    elif adapter == "expw_npz":
        samples, warnings = _load_expw_npz_split(local_root, split_cfg)
    elif adapter == "jaffe_filename":
        samples, warnings = _load_jaffe_split(local_root, split_cfg)
    elif adapter == "cremad_manifest":
        samples, warnings = _load_cremad_split(local_root, split_cfg)
    else:
        raise UnsupportedAdapterError(
            f"Dataset '{matched_key}' requires adapter '{adapter}', which is not implemented."
        )

    default_protocol = cfg.get("default_protocol", "pre_cropped")
    if adapter == "expw_manifest" and split_cfg.get("crop_mode") == "annotated_box":
        default_protocol = "pre_cropped"

    return LoadedDataset(
        dataset_name=matched_key,
        display_name=cfg.get("display_name", matched_key),
        split_name=resolved_split,
        adapter=adapter,
        local_root=local_root,
        default_protocol=default_protocol,
        supported_protocols=list(cfg.get("supported_protocols", ["pre_cropped"])),
        samples=samples,
        label_mapping=dict(cfg.get("label_mapping", {c: c for c in CANONICAL_LABELS})),
        provenance=cfg.get("provenance", "Unknown"),
        license_status=cfg.get("license_status", "Unknown"),
        independence_status=split_cfg.get("independence_status", "UNCERTAIN"),
        suitable_for_independent_test=bool(split_cfg.get("suitable_for_independent_test", False)),
        used_for_training=bool(split_cfg.get("used_for_training", False)),
        used_for_checkpoint_selection=bool(split_cfg.get("used_for_checkpoint_selection", False)),
        previously_inspected=bool(split_cfg.get("previously_inspected", False)),
        disclosure=split_cfg.get("disclosure", "Independence status not established."),
        warnings=warnings,
    )


def format_dataset_listing(project_root: Path) -> str:
    manifest = load_manifest(project_root)
    lines = [
        "==========================================================================================",
        " VIBESHIFT — REGISTERED LOCAL DATASETS (from datasets/dataset_manifest.json)",
        "==========================================================================================",
        f"{'Dataset':<28} {'Split':<22} {'Images':>7} {'Protocol':<12} {'Independence Status'}",
        "-" * 90,
    ]
    for ds_name, cfg in manifest.get("datasets", {}).items():
        local_root = (project_root / cfg["local_path"]).resolve()
        present = local_root.exists()
        for sp_name, sp_cfg in cfg.get("splits", {}).items():
            proto = cfg.get("default_protocol", "pre_cropped")
            if cfg.get("adapter") == "expw_manifest" and sp_cfg.get("crop_mode") == "annotated_box":
                proto = "pre_cropped"
            cnt = sp_cfg.get("expected_images", 0)
            if ds_name == "dataset" and present:
                try:
                    found, _ = _scan_image_folder(local_root)
                    cnt = len(found)
                except Exception:
                    cnt = 0
            status = sp_cfg.get("independence_status", "UNCERTAIN")
            if not present:
                status = f"MISSING_LOCALLY ({status})"
            lines.append(f"{ds_name:<28} {sp_name:<22} {cnt:>7} {proto:<12} {status}")

    not_copied = manifest.get("not_copied_archives", [])
    if not_copied:
        lines.append("-" * 90)
        lines.append("Archives documented in inventory but NOT copied locally:")
        for item in not_copied:
            gb = item.get("size_bytes", 0) / (1024 ** 3)
            lines.append(f"  - {item['name']} ({gb:.2f} GB) at {item['original_path']}")

    absent = manifest.get("verified_absent_locally", [])
    if absent:
        lines.append("Verified absent locally: " + ", ".join(x["name"] for x in absent))

    lines.extend(
        [
            "==========================================================================================",
            "Example commands:",
            "  python evaluate.py --dataset FER2013 --split PrivateTest --mode single",
            "  python evaluate.py --dataset FER2013 --split PrivateTest --mode flip",
            "  python evaluate.py --dataset ExpW_Eval1400 --split full_photo --mode single",
            "  python evaluate.py --dataset JAFFE --split all --mode single",
            "  python evaluate.py --dataset dataset --mode single",
        ]
    )
    return "\n".join(lines)
