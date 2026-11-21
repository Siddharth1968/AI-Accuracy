# VibeShift — Standalone AI Accuracy Evaluator (`CleanBTNVit Phase 12`)

A completely independent, Python-only console project for measuring genuine facial-expression recognition accuracy of VibeShift's finalized **CleanBTNVit Phase 12** model across locally available labeled datasets or manually supplied image folders.

- **Single Emotion Classifier Only:** `CleanBTNVit Phase 12` (`output_VIT` strictly batch-independent pathway, SHA-256 `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`). No ensembles, no EfficientNet-B2, no retraining.
- **Optional Full-Photo Face Detector:** OpenCV **YuNet** (`detector/yunet-2023mar.onnx`, MIT License, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`), used only for full-photograph evaluation protocols.
- **Console-Only:** Pure Python CLI (`evaluate.py`) — no React, no FastAPI, no web server, no database, no GUI.

---

## 1. Quick Start in VS Code (Windows)

1. Open this folder (`VibeShift-AI-Accuracy-Test`) in **VS Code**.
2. Run **`setup.bat`** once (from VS Code's terminal or by double-clicking it):
   ```cmd
   setup.bat
   ```
   - Creates `.venv` using Python `3.10–3.12`.
   - Auto-detects NVIDIA CUDA GPU (installs `torch==2.10.0+cu126`) or automatically falls back to CPU PyTorch (`set VIBESHIFT_FORCE_CPU=1` forces CPU).
   - Verifies checkpoint SHA-256, YuNet SHA-256, Python imports, and `datasets/dataset_manifest.json`.
3. In VS Code, confirm the Python interpreter is set to **`.venv\Scripts\python.exe`** (`Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`). `.vscode/settings.json` pre-configures this automatically.

---

## 2. Listing & Evaluating Datasets

### Discover All Available Local Datasets
```powershell
.\.venv\Scripts\python.exe evaluate.py --list-datasets
```

### Run via Interactive Console Menu
```cmd
run_evaluation.bat
```

### Direct Command-Line Examples

```powershell
# 1. Primary independent held-out FER2013 PrivateTest (3,589 pre-cropped faces)
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode flip
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode both

# 2. Cross-dataset ExpW 1,400 balanced subset (full-photo YuNet vs annotated face crops)
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split full_photo --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split annotated_crops --mode flip

# 3. Cross-dataset JAFFE Official v2 (213 posed portraits, full-photo YuNet)
.\.venv\Scripts\python.exe evaluate.py --dataset JAFFE --split all --mode single

# 4. Cross-dataset CREMA-D 144 extracted frames (6 classes only — surprise has 0 support)
.\.venv\Scripts\python.exe evaluate.py --dataset CREMA_D_Frames144 --split all --mode single

# 5. Manual user-supplied folder (place images into dataset/<emotion>/)
.\.venv\Scripts\python.exe evaluate.py --dataset dataset --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset dataset --mode flip --protocol full_photo

# 6. Force CPU evaluation explicitly
.\.venv\Scripts\python.exe evaluate.py --dataset JAFFE --split all --mode single --device cpu
```

---

## 3. Exact Verified Preprocessing & Canonical Labels

Every evaluated face crop passes through the exact deterministic Phase 12 pipeline (`model/preprocess.py`):
1. Read image and convert consistently to `uint8` grayscale (preserving exact pixels if already grayscale).
2. Convert to `float32` in `[0.0, 1.0]` with shape `(1, 1, H, W)`.
3. Resize to `112 × 112` using PyTorch bilinear interpolation (`align_corners=False`).
4. Quantize via `.mul(255.0).round().div(255.0)`.
5. Broadcast-normalize across 3 channels with ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
6. Forward pass through `CleanBTNVit` (`output_VIT`), converting logits into 7-class softmax probabilities (`horizontal_flip` averages `softmax(logits)` and `softmax(logits_flip)`).

| Index | Emotion |
|---:|---|
| `0` | `angry` |
| `1` | `disgust` |
| `2` | `fear` |
| `3` | `happy` |
| `4` | `neutral` |
| `5` | `sad` |
| `6` | `surprise` |

---

## 4. Datasets Included Locally vs. GitHub Readiness

| Folder | Included Locally? | Included on GitHub? | Why |
|---|---|---|---|
| `datasets/local_library/` | **YES (`197.77 MB`, 5 datasets)** | **NO (`.gitignore`d)** | Contains `FER2013` (35,887), `ExpW_Eval1400` (1,400), `ExpW_Phase12_TrainClean5962` (5,962), `JAFFE` (213), and `CREMA_D_Frames144` (144). These carry non-commercial academic or explicit non-redistribution terms (e.g., JAFFE forbids GitHub redistribution). |
| `datasets/shareable/` | Empty | YES (README only) | Reserved strictly for datasets with verified public redistribution rights. |
| `model/clean_btn_vit_phase12.pt` | **YES (`245.3 MiB`)** | **NO (`.gitignore`d)** | Upstream AffectNet/ExpW/RAF-DB/MS-Celeb-1M lineage lacks confirmed public redistribution rights. |
| `detector/yunet-2023mar.onnx` | **YES (`227 KB`)** | **YES** | MIT-licensed OpenCV Zoo model (Shiqi Yu). |
| `dataset/<emotion>/` | Empty 7-class template | YES (`.gitkeep` only) | User-supplied images are ignored by `.gitignore` to protect privacy. |

See [datasets/DATASET_INVENTORY.md](datasets/DATASET_INVENTORY.md), [datasets/DATASET_USAGE_GUIDE.md](datasets/DATASET_USAGE_GUIDE.md), and [FRIEND_SETUP_GUIDE.md](FRIEND_SETUP_GUIDE.md) for full provenance, split independence disclosures, and authorized teammate instructions.

---

## 5. Output Artifacts Per Run

Every run writes a unique timestamped folder under `results/<run_id>/` containing:
- `metrics.json` — full run configuration, SHA-256, accuracy, macro/weighted F1, per-class metrics, confusion matrix, detection stats, and leakage disclosure.
- `predictions.csv` — per-image audit trail (ground truth, prediction, correctness, status, detection status, confidence, and all 7 class probabilities; all metrics are reproducible directly from this file).
- `confusion_matrix.csv` & `confusion_matrix.png` — raw 7×7 table and row-normalized visual heatmap.
- `classification_report.txt` & `evaluation_summary.md` — human-readable terminal and Markdown summaries.

---

## 6. Running the Software Verification Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest tests/ -v
```
