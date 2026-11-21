# VibeShift AI Accuracy Evaluator — Teammate & Faculty Setup Guide

This guide explains how an authorized teammate or faculty reviewer can set up and run **VibeShift-AI-Accuracy-Test** on a fresh Windows laptop after receiving the code from GitHub (or an authorized source archive).

---

## 1. Prerequisites

- **Windows 10 or Windows 11 (64-bit)**
- **Python 3.10, 3.11, or 3.12 (64-bit)** — Python 3.12.10 recommended: https://www.python.org/downloads/release/python-31210/ (check **"Add python.exe to PATH"**; do not use Python 3.13/3.14 yet).
- **VS Code** (optional, recommended): https://code.visualstudio.com/ with the Microsoft **Python** extension.
- **GPU optional:** NVIDIA GPU with CUDA 12.1+ driver gives ~22–43 ms/image; CPU-only laptops work automatically via CPU PyTorch (~180–360 ms/image).

---

## 2. Why Model Weights & Local Datasets Are Not in Public Git

When this folder is published to GitHub, `.gitignore` intentionally excludes:
1. **`model/clean_btn_vit_phase12.pt`** (`245.3 MiB`, SHA-256 `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`) — upstream academic training lineage (AffectNet, ExpW, RAF-DB, MS-Celeb-1M) does not establish public redistribution rights.
2. **`datasets/local_library/`** — academic datasets carry non-commercial or explicit non-redistribution rules (e.g., JAFFE forbids GitHub/Kaggle redistribution).

---

## 3. Obtaining Required Assets Legally

### A. Authorized Model Checkpoint (`model/clean_btn_vit_phase12.pt`)
1. Confirm with your faculty advisor / Lead Researcher that private academic-team transfer is permitted within your non-commercial course project.
2. Obtain `clean_btn_vit_phase12.pt` via a private team-only channel (USB drive or access-controlled institutional drive — never a public link).
3. Copy it to:
   ```text
   VibeShift-AI-Accuracy-Test\model\clean_btn_vit_phase12.pt
   ```
4. Verify SHA-256 equals `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58` (automatically verified by `setup.bat` and `evaluate.py`).

### B. Datasets (Obtain Directly from Official Sources)
- **Manual Testing (Works Immediately Without Downloading Datasets):** Place your own labeled images into `dataset/angry/`, `dataset/disgust/`, `dataset/fear/`, `dataset/happy/`, `dataset/neutral/`, `dataset/sad/`, `dataset/surprise/`.
- **FER2013:** Download `msambare/fer2013` directly from Kaggle and extract `train/` and `test/` into `datasets/local_library/FER2013/`.
- **JAFFE (Official v2):** Download directly from Zenodo DOI `10.5281/zenodo.14974867` under non-commercial scientific research terms and extract `.tiff` files into `datasets/local_library/JAFFE/images/`. (Never re-upload JAFFE to GitHub.)
- **ExpW:** Obtain directly from CUHK MMLab under non-commercial academic research terms.
- **CREMA-D:** Obtain directly from `CheyneyComputerScience/CREMA-D` under ODbL 1.0 / DbCL 1.0.

---

## 4. Authorized Academic-Team Packaging Workflow (When Approved)

If your faculty advisor approves private non-commercial team transfer of the evaluator + checkpoint (without prohibited datasets like JAFFE):
1. Share the clean Git-tracked source repository (which excludes `.venv/`, `results/`, `datasets/local_library/`, and `model/*.pt`).
2. Transfer `clean_btn_vit_phase12.pt` privately via access-controlled university storage or USB.
3. Have each teammate download any needed benchmark datasets directly from their official sources (Section 3B above) so individual dataset license terms (such as JAFFE's non-redistribution rule) are strictly respected.

---

## 5. Setup & Opening in VS Code

1. Open `VibeShift-AI-Accuracy-Test` in **VS Code** (`File → Open Folder...`).
2. Open the integrated terminal (`Ctrl + \``) and run:
   ```cmd
   setup.bat
   ```
   *(To force a smaller CPU-only PyTorch install on any laptop: `set VIBESHIFT_FORCE_CPU=1` then `setup.bat`.)*
3. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose **`.\.venv\Scripts\python.exe`**.
4. Verify available datasets:
   ```powershell
   .\.venv\Scripts\python.exe evaluate.py --list-datasets
   ```
5. Run an evaluation via `run_evaluation.bat` or directly via `evaluate.py`!
