# VibeShift Standalone AI Accuracy Evaluator — Completion Report

**Date:** 2026-11-21 (IST)  
**Model:** `CleanBTNVit Phase 12` (Single emotion classifier only — no ensembles, no retraining, no legacy models)

---

## 1. Exact New Folder Path

`C:\Users\asus\OneDrive\Desktop\VibeShift-AI-Accuracy-Test`

Resolved dynamically via Windows `[Environment]::GetFolderPath('Desktop')` (OneDrive-redirected Desktop). Directory did not previously exist, so no existing work was overwritten. Zero symbolic links or junctions exist anywhere inside the project (`TRUE_SYMLINKS_OR_JUNCTIONS=0`).

---

## 2. Files Copied from `VibeShift-Final` (Strictly Read-Only Source)

| Source File (`VibeShift-Final`) | Independent Copy (`VibeShift-AI-Accuracy-Test`) | Notes |
|---|---|---|
| `backend/ai/models/clean_btn_vit_phase12.pt` | `model/clean_btn_vit_phase12.pt` | Verbatim physical copy (`257,223,791` bytes) |
| `backend/ai/btn_runtime/__init__.py` | `model/btn_runtime/__init__.py` | Relative import adjustment only (zero `backend.*` dependency) |
| `backend/ai/btn_runtime/BatchTransformer.py` | `model/btn_runtime/BatchTransformer.py` | Exact verbatim copy |
| `backend/ai/btn_runtime/clean_btn_vit.py` | `model/btn_runtime/clean_btn_vit.py` | Relative import adjustment only (preserves MobileFaceNet, IR-50, 3-stage cross-attention, ViT head, batch-independent `output_VIT`) |
| `backend/ai/btn_runtime/ir50.py` | `model/btn_runtime/ir50.py` | Exact verbatim copy |
| `backend/ai/btn_runtime/mobilefacenet.py` | `model/btn_runtime/mobilefacenet.py` | Exact verbatim copy |
| `backend/ai/btn_runtime/vit_model.py` | `model/btn_runtime/vit_model.py` | Exact verbatim copy |
| `backend/ai/preprocess.py` | `model/preprocess.py` | Exact verbatim copy |
| `backend/ai/models/yunet-2023mar.onnx` | `detector/yunet-2023mar.onnx` | Verbatim physical copy (`232,589` bytes, MIT License, optional for `full_photo` protocol) |

No website, frontend, FastAPI, React, Spotify, SQLite database, GUI, EfficientNet-B2, EVA-02, FaRL, or ensemble code was copied.

---

## 3. Model Checkpoint & Detector Hashes

- **CleanBTNVit Phase 12 (`model/clean_btn_vit_phase12.pt`):**  
  `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58` (**MATCH = True**)
- **YuNet ONNX (`detector/yunet-2023mar.onnx`):**  
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` (**MATCH = True**)

---

## 4. Model-Loading Verification

- Strict SHA-256 verification passed (and unit-tested to reject corrupted checksums and 134-byte Git LFS pointer files).
- Loaded with `torch.load(..., map_location="cpu", weights_only=True)` and `CleanBTNVit(head_drop=0.0).load_state_dict(..., strict=True)`.
- Verified batch-independence of `output_VIT`: max delta under perturbed batch peers `< 1e-5`, max delta singleton vs batch `< 1e-4`.
- Verified both **CPU (FP32)** and **CUDA (FP16 autocast on NVIDIA GeForce RTX 3050 Laptop GPU)** forward passes in single-view and horizontal-flip modes.

---

## 5. Preprocessing & Inference Parity Verification

Controlled synthetic parity check across 5 deterministic synthetic test images (`(48,48)` grayscale, `(48,48,3)` RGB, `(128,128,3)` RGB, `(256,256)` grayscale, `(96,112,4)` BGRA) comparing `VibeShift-AI-Accuracy-Test` directly against `VibeShift-Final` on CUDA:
- **Preprocessing tensor `max_abs_diff`:** **`0.0000000000`**
- **Single-view 7-class probabilities `max_abs_diff`:** **`0.0000000000`**
- **Horizontal-flip 7-class probabilities `max_abs_diff`:** **`0.0000000000`**
- **Canonical 7-class order verified:** `0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise`.

---

## 6. Every Relevant Local Dataset Discovered (Read-Only Scan)

1. **`FER2013`** (`C:\VibeShift-Datasets\FER2013\` + `split.csv` + `selection.csv` + metadata): `35,887` 48×48 grayscale `.jpg` images (`28,709` train + `3,589` PublicTest + `3,589` PrivateTest). Duplicate archive at `C:\VibeShift-Datasets\downloads\msambare-fer2013-v1.zip` (`60.34 MB`).
2. **`ExpW_Eval1400`** (`VibeShift-local\experiments\cross-dataset\dataset\images\origin\` + `expw-label.lst` + `expw-readme.txt` + `sample-manifest.json`): `1,400` balanced full-scene RGB `.jpg` images (`200` per class across all 7 classes; original ExpW order `4=sad, 5=surprise, 6=neutral` explicitly remapped).
3. **`ExpW_Phase12_TrainClean5962`** (`VibeShift-local\experiments\maximum-accuracy\phase11_expw\`): `5,962` deduplicated 48×48 grayscale crops in `expw_clean_48.npz` (`audit.json`, `manifest.json`).
4. **`ExpW_RawArchive_7z`** (`VibeShift-local\experiments\cross-dataset\dataset\archives\origin.7z.001..008`): `8` split `.7z` files (`7.56 GB` / `8,113,576,419` bytes), unextracted archive covering `91,793` face annotations.
5. **`JAFFE`** (`VibeShift-local\experiments\kdef-evaluation\dataset\jaffe\` + `README_FIRST.txt` + `manifest.json`): `213` official v2 `256×256` `.tiff` portraits across 10 female posers. Duplicate archive at `experiments\kdef-evaluation\jaffe.zip` (`12.29 MB`).
6. **`CREMA_D_Frames144`** (`VibeShift-local\experiments\phase4\data\frames\` + `LICENSE.txt` + CSVs + `frame-manifest.json`): `144` extracted `480×360` RGB `.png` frames across **6 classes** (`surprise` absent; source `.flv` clips `38.29 MB`).
7. **Explicitly Verified Absent Locally:** **FER+**, **RAF-DB** (`access-status.json: PERMISSION_REQUIRED, dataset_found: false`), **AffectNet** (no image files on disk; only external pretrained weights existed upstream), **KDEF** (`dataset-access.json: found: false, registration required`).

Full per-class counts, SHA-256 hashes, provenance, and duplicate maps are recorded in `datasets/DATASET_INVENTORY.md` and `datasets/dataset_manifest.json`.

---

## 7. Datasets Physically Copied into `datasets/local_library/`

All **5 ready-to-evaluate local image datasets** were physically copied (independent copies, preserving original structure, labels, splits, and metadata):
1. `datasets/local_library/FER2013/` (`35,887` images + `5` metadata files)
2. `datasets/local_library/ExpW_Eval1400/` (`1,400` images + `3` official annotation/manifest files)
3. `datasets/local_library/ExpW_Phase12_TrainClean5962/` (`5,962` crops in `.npz` + `audit.json` + `manifest.json`)
4. `datasets/local_library/JAFFE/` (`213` `.tiff` images + `README_FIRST.txt` + `manifest.json`)
5. `datasets/local_library/CREMA_D_Frames144/` (`144` `.png` frames + `LICENSE.txt`, `SentenceFilenames.csv`, `tabulatedVotes.csv`, `frame-manifest.json`)

An empty 7-class manual folder `dataset/{angry,disgust,fear,happy,neutral,sad,surprise}/` was also created (`0` images).

---

## 8. Image Counts & Storage Used

| Component | Files / Images | Measured Disk Size |
|---|---:|---:|
| `datasets/local_library/FER2013` | `35,887` images (`35,892` files) | `56.64 MB` |
| `datasets/local_library/ExpW_Eval1400` | `1,400` images (`1,403` files) | `88.19 MB` |
| `datasets/local_library/ExpW_Phase12_TrainClean5962` | `5,962` crops (`3` files) | `13.25 MB` |
| `datasets/local_library/JAFFE` | `213` images (`215` files) | `13.46 MB` |
| `datasets/local_library/CREMA_D_Frames144` | `144` frames (`148` files) | `26.23 MB` |
| **Subtotal — Copied Local Dataset Library** | **`43,606` labeled samples (`37,661` files)** | **`197.77 MB`** |
| `model/` (`clean_btn_vit_phase12.pt` + runtime) | `10` files | `245.36 MB` |
| `detector/` (`yunet-2023mar.onnx` + wrapper) | `3` files | `0.23 MB` |
| `.venv/` (Python 3.12 + CUDA 12.6 PyTorch + deps) | isolated venv | `4,326.02 MB` |
| **Total `VibeShift-AI-Accuracy-Test` Project Size** | — | **`4,769.52 MB` (`~4.66 GB`)** |

System drive `C:` had `135.79 GB` free before copying.

---

## 9. Datasets & Artifacts Skipped and Why

- **`ExpW_RawArchive_7z` (`origin.7z.001..008`, `7.56 GB`):** Unextracted multipart 7-Zip archive skipped to avoid duplicating 7.56 GB of compressed raw archives inside OneDrive Desktop; documented in full in `datasets/DATASET_INVENTORY.md` and `datasets/dataset_manifest.json` with original path.
- **`msambare-fer2013-v1.zip` (`60.34 MB`) & `jaffe.zip` (`12.29 MB`):** Exact duplicate zip archives of already copied extracted `FER2013` and `JAFFE` datasets.
- **`FER2013/development*/development-pixels.npz` (`94.21 MB` combined):** Derived pixel caches of `FER2013`, not distinct datasets.
- **`phase4/data/clips/*.flv` (`38.29 MB`):** Raw Flash video source clips from which `CREMA_D_Frames144` PNG frames were extracted.
- **Teacher prediction caches (`*.pt` / `*.npz`) & synthetic QA images (`webcam-reliability/`, `phase4/tmp/qa/`):** Excluded because teacher predictions and synthetic QA fixtures are not verified ground-truth datasets.

---

## 10. Dataset Licensing & Sharing Eligibility

- **`datasets/shareable/`:** Intentionally contains **zero** restricted datasets (`README.md` explains policy).
- **`datasets/local_library/` (excluded from Git via `.gitignore`):**
  - `JAFFE`: **Public & teammate redistribution explicitly prohibited** (`README_FIRST.txt`).
  - `ExpW_Eval1400` & `ExpW_Phase12_TrainClean5962`: Non-commercial academic research only; **no public redistribution grant**.
  - `FER2013`: Kaggle `DbCL-1.0` metadata, web-scraped face images lack verified model releases; **excluded from public Git**.
  - `CREMA_D_Frames144`: `ODbL 1.0 + DbCL 1.0` (`LICENSE.txt` preserved); held cautiously in `local_library/` pending explicit attribution/human-subject redistribution review.
- **`model/clean_btn_vit_phase12.pt` (excluded from Git via `.gitignore`):** Upstream AffectNet / ExpW / RAF-DB / MS-Celeb-1M lineage lacks confirmed public redistribution rights.
- **`detector/yunet-2023mar.onnx`:** MIT-licensed (shareable).

---

## 11. Training / Development / Test Split Distinctions

| Dataset / Split | Category | Independent Final Test for Phase 12? |
|---|---|---|
| `FER2013 / PrivateTest` (`3,589`) | Held-out FER2013 test split | **YES** (Untouched in Phase 12; disclose 53 corpus-inherent PublicTest/PrivateTest overlaps in `dataset-report.json`) |
| `FER2013 / PublicTest` (`3,589`) & `PublicTest_Clean3536` (`3,536`) | Previously inspected test split | **NO (Partial)** — untouched in Phase 12, inspected in Phases 6 & 8 |
| `FER2013 / DevHoldout4015` (`4,015`) | Reused development holdout | **NO** — reused for checkpoint selection & soup tuning across Phases 8–12 |
| `FER2013 / Train22757` (`22,757`) & `AllTrain28709` (`28,709`) | Training data | **NO (Training Leakage)** |
| `ExpW_Phase12_TrainClean5962 / train_clean` (`5,962`) | Supplementary training data | **NO (Training Leakage — used in Phase 12 Stage 1 soft-KD)** |
| `ExpW_Eval1400 / full_photo` & `annotated_crops` (`1,400`) | Cross-dataset evaluation subset | **Conditional YES** — strictly excluded from Phase 12 training; earlier project models were evaluated here |
| `JAFFE / all` (`213`) | Cross-dataset posed subset | **Conditional YES** — untouched in Phase 12; earlier project models were evaluated here |
| `CREMA_D_Frames144 / all` (`144`, 6 classes) | Cross-dataset video-frame subset | **Conditional YES** — untouched in Phase 12; 6 classes only (`surprise` has 0 support) |
| `dataset / default` (`0` images) | Manual user-supplied folder | Depends on images added by user |

---

## 12. Available Evaluation Commands

```powershell
# List all registered datasets, splits, protocols, and independence status
.\.venv\Scripts\python.exe evaluate.py --list-datasets

# Interactive menu launcher
run_evaluation.bat

# Primary held-out FER2013 PrivateTest (single-view, horizontal-flip, or both)
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode flip
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode both

# Cross-dataset benchmarks
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split full_photo --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split annotated_crops --mode flip
.\.venv\Scripts\python.exe evaluate.py --dataset JAFFE --split all --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset CREMA_D_Frames144 --split all --mode single

# Manual folder (after adding images to dataset/<emotion>/)
.\.venv\Scripts\python.exe evaluate.py --dataset dataset --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset dataset --mode flip --protocol full_photo
```

---

## 13. Tests Passed and Failed

- **`setup.bat`:** **PASSED** (`exit 0` — `.venv` built, CUDA 12.6 PyTorch + minimal `requirements.txt` installed, `evaluate.py --check-env` passed).
- **`pytest tests/ -v`:** **9 / 9 PASSED, 0 FAILED** (`14.27s`), covering:
  1. Independent path isolation (`LEAKED_MODULES == []`)
  2. Checkpoint & YuNet SHA-256 verification (+ bad checksum & Git LFS pointer rejection)
  3. Exact Phase 12 preprocessing formula (`max_abs_diff == 0.0`)
  4. Label mappings (canonical, ExpW 4/5/6 remap, JAFFE regex, CREMA-D 6-class) & CUDA driver regex (`CUDA Version:` + `CUDA UMD Version:`)
  5. Dataset registry, `--list-datasets`, duplicate tracking, unsupported adapter rejection, and exact on-disk class-count validation across all 5 copied datasets (without running inference)
  6. Empty manual `dataset/` handling (`exit 2`)
  7. Accuracy, precision, recall, active/7-class Macro F1, Weighted F1, confusion matrix, and full-photo detection metrics
  8. CPU + CUDA smoke tests, single-view + horizontal-flip averaging verification, and `output_VIT` batch-independence
  9. End-to-end synthetic `full_photo` & `pre_cropped` evaluation (valid face, corrupt image, blank no-face, ambiguous two-face disclosure, all 6 output artifacts, and exact reproducibility from `predictions.csv`)
- **Controlled Synthetic Parity Check vs. `VibeShift-Final`:** **PASSED** (`0.0000000000` max diff across preprocessing, single-view, and horizontal-flip).
- **`run_evaluation.bat --list-datasets`:** **PASSED** (`exit 0`).

---

## 14. CPU and GPU Support

- **GPU:** Verified on `NVIDIA GeForce RTX 3050 Laptop GPU` (CUDA 12.6, FP16 autocast); auto-detected by `setup.bat` and `--device auto` / `--device cuda`.
- **CPU:** Verified via `CleanBTNVitEvaluatorModel(device="cpu")` (FP32) in unit tests; supported automatically on non-GPU machines via `setup.bat` (or forced via `VIBESHIFT_FORCE_CPU=1` / `--device cpu`).

---

## 15. ZIP / GitHub Readiness

- `.gitignore` strictly excludes `.venv/`, `__pycache__/`, `results/*` (keeping `.gitkeep`), `model/clean_btn_vit_phase12.pt`, `datasets/local_library/`, and private user images in `dataset/*/*` (keeping `dataset/*/.gitkeep`).
- `.vscode/settings.json` pre-configures VS Code for `.venv\Scripts\python.exe` and `pytest`.
- `README.md`, `FRIEND_SETUP_GUIDE.md`, `datasets/DATASET_INVENTORY.md`, `datasets/dataset_manifest.json`, and `datasets/DATASET_USAGE_GUIDE.md` document everything needed for future authorized sharing.
- No Git repository was initialized or pushed automatically.

---

## 16. Remaining Limitations

1. Real accuracy numbers on `FER2013 PrivateTest`, `ExpW_Eval1400`, `JAFFE`, and `CREMA_D_Frames144` have intentionally **not** been executed yet — they await your explicit dataset selection.
2. `FER+`, `RAF-DB`, `AffectNet`, and `KDEF` are not present locally on this machine; if acquired later under their official licenses, explicit adapters can be registered in `datasets/dataset_manifest.json` and `evaluator/adapters.py`.
3. `ExpW_RawArchive_7z` (`7.56 GB`) remains unextracted at its original source location rather than duplicated into OneDrive Desktop.
4. Teammates cloning a future GitHub copy must obtain `model/clean_btn_vit_phase12.pt` privately under academic authorization and download restricted benchmark datasets directly from official sources.

---

## 17. Confirmation That Original Projects Were Unchanged

- **`C:\Users\asus\OneDrive\Desktop\VibeShift-Final`:** Verified completely unchanged (`git status -s` empty, `HEAD` unchanged at `ec0a95f4fb7ce054324771a1ec2af2b3c558a4e1`).
- **`C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local`:** Verified `0` files modified (`VIBESHIFT_LOCAL_MODIFIED_SINCE_2245=0`).
- **`C:\VibeShift-Datasets`:** Accessed strictly read-only; `0` files altered, moved, or deleted.

---

## 18. Confirmation That No GitHub Repository Was Modified

- Confirmed: **Zero** Git commits, pushes, API writes, or remote interactions were performed on any GitHub repository (and your friend's GitHub repository was never accessed).

---

## 19. Confirmation That No Scientific Evaluation Result Was Fabricated

- Confirmed: `results/` contains **0** evaluation runs (`RESULTS_NON_GITKEEP=0`), `dataset/` contains **0** images (`MANUAL_DATASET_NON_GITKEEP=0`), no real dataset accuracy benchmark was run or fabricated, and all verification tests ran strictly on synthetic inputs and metadata checks.
