# VibeShift AI Accuracy Evaluator — Friend & Teammate Setup Guide

This guide walks engineering students and teammates step-by-step through installing and running the **VibeShift CleanBTNVit Phase 12** accuracy evaluator on Windows.

> **Important Asset-Rights Notice:** Before cloning or using this repository, note that third-party facial-expression datasets (`FER2013`, `ExpW`, `JAFFE`) and the derived `CleanBTNVit Phase 12` weights (`clean_btn_vit_phase12.pt`) are **blocked from GitHub redistribution** unless explicit written permissions are confirmed. See `BLOCKED_ASSETS_REPORT.md` for exact licensing requirements and legal download instructions below.

---

## 1. Install Git

1. Download **Git for Windows** (64-bit) from https://git-scm.com/download/win.
2. Run the installer with default options (make sure **Git from the command line** is enabled).
3. Verify in PowerShell or Command Prompt:
   ```powershell
   git --version
   ```

## 2. Install Git LFS

1. Download **Git LFS** from https://git-lfs.com/ (or verify if already bundled with Git for Windows).
2. Enable Git LFS once for your user account:
   ```powershell
   git lfs version
   git lfs install
   ```

## 3. Clone the Private GitHub Repository (After Being Added as an Authorized Collaborator)

1. Accept the collaborator invitation to the private repository on GitHub (`Siddharth1968/vibeshift-ai-accuracy-evaluator` once published).
2. Clone the repository into a clean local folder (e.g., `C:\Projects`):
   ```powershell
   git clone https://github.com/Siddharth1968/vibeshift-ai-accuracy-evaluator.git
   cd vibeshift-ai-accuracy-evaluator
   git lfs pull
   ```
3. **If blocked assets are not hosted in Git:**
   - Obtain an authorized copy of `clean_btn_vit_phase12.pt` privately from your academic project team (once faculty/institutional permission is confirmed) and place it at `model\clean_btn_vit_phase12.pt`.
   - Download benchmark datasets directly from their official sources:
     - **FER2013:** Kaggle `msambare/fer2013` → extract `train/` and `test/` into `datasets\local_library\FER2013\`.
     - **JAFFE (Official v2):** Zenodo DOI `10.5281/zenodo.14974867` → extract `.tiff` images into `datasets\local_library\JAFFE\images\`. *(Never upload JAFFE to GitHub or Kaggle.)*
     - **ExpW:** CUHK MMLab ExpW → place under `datasets\local_library\ExpW_Eval1400\`.
     - **CREMA-D:** `CheyneyComputerScience/CREMA-D` (ODbL 1.0 / DbCL 1.0) → place under `datasets\local_library\CREMA_D_Frames144\`.
     - **Manual Test Folder (No Download Needed):** Place your own labeled photos into `dataset\angry\`, `dataset\disgust\`, `dataset\fear\`, `dataset\happy\`, `dataset\neutral\`, `dataset\sad\`, `dataset\surprise\`.

## 4. Install Python (3.10 – 3.12, 64-bit)

1. Download **Python 3.12.10 (64-bit)** from https://www.python.org/downloads/release/python-31210/.
   *(Do not use Python 3.13 or 3.14 yet — PyTorch wheels require Python 3.10–3.12.)*
2. Check **"Add python.exe to PATH"** on the first installer screen, then click **Install Now**.
3. Verify in a new terminal:
   ```powershell
   py -3.12 --version
   ```

## 5. Open the Project in Antigravity IDE (or VS Code)

1. Launch **Antigravity IDE** (or **VS Code**).
2. Click **File → Open Folder...** and select the cloned `vibeshift-ai-accuracy-evaluator` directory.
3. Open the integrated terminal (`Ctrl + \``).
4. After running `setup.bat` below, ensure the workspace Python interpreter is set to **`.venv\Scripts\python.exe`** (`Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`). `.vscode/settings.json` pre-configures this path automatically.

## 6. Run Automated Setup (`setup.bat`)

From the project root in your terminal (or by double-clicking `setup.bat`):

```cmd
setup.bat
```

`setup.bat` creates an isolated `.venv`, installs minimal Python dependencies (`requirements.txt`), verifies the model and YuNet checksums, and checks the dataset manifest registry.

## 7. Selecting CPU vs. CUDA (NVIDIA GPU)

- **Automatic detection (Default):** `setup.bat` runs `nvidia-smi` and installs **PyTorch CUDA 12.6** automatically on laptops with an NVIDIA GPU (CUDA driver >= 12.1), or installs **CPU PyTorch** automatically on non-NVIDIA laptops.
- **Force CPU installation during setup (smaller ~0.9 GB `.venv`):**
  ```cmd
  set VIBESHIFT_FORCE_CPU=1
  setup.bat
  ```
- **Choose compute device at evaluation time:**
  ```powershell
  # Auto (uses CUDA GPU if available, otherwise CPU)
  .\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode single --device auto

  # Force CPU explicitly
  .\.venv\Scripts\python.exe evaluate.py --dataset JAFFE --split all --mode single --device cpu

  # Force NVIDIA GPU explicitly
  .\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode flip --device cuda
  ```

## 8. List Available Datasets

```powershell
.\.venv\Scripts\python.exe evaluate.py --list-datasets
```

This displays every registered dataset, split, image count, default protocol (`pre_cropped` vs. `full_photo`), and dataset-leakage / independence status.

## 9. Run FER2013 Accuracy Evaluation

```powershell
# Evaluate both Single-View and Horizontal-Flip on the held-out FER2013 PrivateTest split (3,589 images):
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode both

# Or run a single mode:
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode single
.\.venv\Scripts\python.exe evaluate.py --dataset FER2013 --split PrivateTest --mode flip
```

*(You can also double-click `run_evaluation.bat` for an interactive console menu.)*

## 10. Evaluate ExpW and JAFFE

```powershell
# ExpW 1,400 balanced full-scene photos (YuNet face detection + CleanBTNVit):
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split full_photo --mode single

# ExpW 1,400 annotated ground-truth face crops (isolates classifier from face detector):
.\.venv\Scripts\python.exe evaluate.py --dataset ExpW_Eval1400 --split annotated_crops --mode flip

# JAFFE Official v2 (213 posed portraits, full-photo YuNet protocol):
.\.venv\Scripts\python.exe evaluate.py --dataset JAFFE --split all --mode single

# CREMA-D 144 extracted video frames (6 classes — surprise has 0 support):
.\.venv\Scripts\python.exe evaluate.py --dataset CREMA_D_Frames144 --split all --mode single
```

## 11. Locate the Confusion Matrix & Saved Reports

Every run creates a unique timestamped directory inside **`results\`**:

```text
results\<YYYYMMDD_HHMMSS_Dataset_Split_Mode>\
├── confusion_matrix.png        # Row-normalized visual 7x7 heatmap with raw counts
├── confusion_matrix.csv        # Raw 7x7 confusion matrix (rows = truth, cols = predicted)
├── metrics.json                # Complete machine-readable metrics + provenance + SHA-256
├── predictions.csv             # Per-image predictions & 7-class probabilities
├── classification_report.txt   # Formatted terminal report
└── evaluation_summary.md       # Full Markdown summary with independence disclosures
```

## 12. How to Interpret the Accuracy Report

- **Accuracy (`Correct / Evaluated × 100`):** Always read the exact numerator and denominator (`Correct / Evaluated`), not just the percentage.
- **Macro F1 vs. Weighted F1:** **Macro F1** averages F1 equally across active emotion classes (important when minority classes like `disgust` have fewer images); **Weighted F1** weights each class by its sample count.
- **Full-Photograph Datasets (`ExpW_Eval1400/full_photo`, `JAFFE`, `CREMA_D_Frames144`):**
  - **Conditional Classification Accuracy** measures accuracy only among images where YuNet detected a valid unambiguous face.
  - **End-to-End Accuracy** counts detection failures (`no_face_detected`, `no_target_match`) and `ambiguous_multi_face` images against the total readable denominator. Never confuse conditional accuracy with end-to-end accuracy.
- **Independence / Leakage Status:** Never report `TRAINING_DATA` (`Train22757`, `ExpW_Phase12_TrainClean5962`) or `REUSED_DEV_HOLDOUT` (`DevHoldout4015`) as unseen independent test accuracy.

## 13. Verify Model & Detector Checksums

Run the built-in self-check:

```powershell
.\.venv\Scripts\python.exe evaluate.py --check-env
```

Or check SHA-256 directly in PowerShell:

```powershell
Get-FileHash .\model\clean_btn_vit_phase12.pt -Algorithm SHA256
Get-FileHash .\detector\yunet-2023mar.onnx -Algorithm SHA256
```

Expected values:
- **`model\clean_btn_vit_phase12.pt`** (`257,223,791` bytes): `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`
- **`detector\yunet-2023mar.onnx`** (`232,589` bytes): `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

## 14. Troubleshooting Incomplete Git LFS Downloads

If `evaluate.py --check-env` or `setup.bat` reports:

> `File 'model\clean_btn_vit_phase12.pt' is a Git LFS pointer file, not the 245.3 MiB CleanBTNVit weights.`

This means the repository was cloned before Git LFS was installed, so only a ~134-byte text pointer was checked out. Fix it with:

```powershell
git lfs install
git lfs pull
.\.venv\Scripts\python.exe evaluate.py --check-env
```

If Git LFS pull fails because of repository LFS quota or because weights are intentionally excluded from Git, obtain the authorized `clean_btn_vit_phase12.pt` file privately from your project lead, copy it into `model\clean_btn_vit_phase12.pt`, and re-run `.\.venv\Scripts\python.exe evaluate.py --check-env`.
