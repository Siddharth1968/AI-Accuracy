# VibeShift AI Accuracy Evaluator — GitHub Deployment Completion Report

**Date:** 2026-11-21 (IST)  
**Staging Copy:** `C:\Users\asus\OneDrive\Desktop\VibeShift-Accuracy-GitHub-Release`  
**Read-Only Source Evaluator:** `C:\Users\asus\OneDrive\Desktop\VibeShift-AI-Accuracy-Test`

---

## 1. Repository

- **GitHub Account (Expected / Verified):** `Siddharth1968`
- **Proposed Repository Name:** `vibeshift-ai-accuracy-evaluator`
- **Repository URL:** **NOT CREATED** (`https://github.com/Siddharth1968/vibeshift-ai-accuracy-evaluator` checked read-only → HTTP `404`, does not exist publicly)
- **Private / Public Status:** **NOT CREATED** (Halted at mandatory pre-publication asset-rights gate per Sections 2, 4, 6, and 12 — zero writes made to GitHub)
- **Default Branch:** `main` (planned; no Git commits or pushes performed)
- **Final Commit SHA:** `NONE — publication workflow stopped before GitHub repository creation and push because required model weights and 4 of 5 local datasets lack established redistribution rights`

---

## 2. File Inventory (Staging Copy: `VibeShift-Accuracy-GitHub-Release`)

- **Total Staged Files (after removing temporary local `results/` run folder from staging and adding `yunet-LICENSE.txt`, `.gitattributes`, and reports):** `37,706` files (excluding `.venv/`, `__pycache__/`, `.pytest_cache/`, and personal `results/` runs)
- **Total Committed / Uploaded Files:** `0` (publication halted at asset-rights compliance gate)
- **Total Staging Directory Size:** `443.54 MB` (`model/` + `detector/` + `datasets/local_library/` + code/docs; `.venv/` excluded for portability)
- **Git LFS Tooling Local Check:** `git-lfs/3.7.0 (GitHub; windows amd64; go 1.24.4; git 92dddf56)` — verified installed locally; **`0` Git LFS objects uploaded**
- **Model Checkpoint Size (`model/clean_btn_vit_phase12.pt`):** `257,223,791` bytes (`245.31 MiB`) — retained locally in staging, **BLOCKED from GitHub upload**
- **Model Checkpoint SHA-256:** `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58` (**VERIFIED MATCH**)
- **Detector Size & SHA-256 (`detector/yunet-2023mar.onnx`):** `232,589` bytes, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` (**VERIFIED MATCH**, MIT License preserved in `detector/yunet-LICENSE.txt`)

### Staged Datasets & Rights Verification

| Dataset | Staged Path | Actual Files / Images Verified | Size on Disk | License / Terms Verified | GitHub Upload Eligibility |
|---|---|---:|---:|---|---|
| **FER2013** (`Train22757`, `DevHoldout4015`, `PublicTest`, `PrivateTest`, `PublicTest_Clean3536`, metadata) | `datasets/local_library/FER2013/` | `35,892` files (`35,887` `.jpg` images + `5` metadata files) | `56.64 MB` | Kaggle `msambare` `DbCL-1.0` repacker tag; original ICML 2013 web-scraped images lack verified individual copyright releases | **BLOCKED (Unverified)** |
| **ExpW_Eval1400** (balanced full-scene subset + `expw-label.lst` + `sample-manifest.json` + `expw-readme.txt`) | `datasets/local_library/ExpW_Eval1400/` | `1,403` files (`1,400` `.jpg` photos + `3` official files) | `88.19 MB` | CUHK MMLab non-commercial academic/educational research only; no redistribution grant | **BLOCKED (Unestablished)** |
| **ExpW_Phase12_TrainClean5962** (marked `TRAINING_DATA`) | `datasets/local_library/ExpW_Phase12_TrainClean5962/` | `3` files (`5,962` crops in `.npz` + `audit.json` + `manifest.json`) | `13.25 MB` | Derived from CUHK MMLab ExpW; non-commercial academic research only, no redistribution grant | **BLOCKED (Unestablished)** |
| **JAFFE** (Official v2 + `README_FIRST.txt` + `manifest.json`) | `datasets/local_library/JAFFE/` | `215` files (`213` `.tiff` portraits + `2` metadata files) | `13.46 MB` | `README_FIRST.txt`: **`Not Permitted: Redistribution of the JAFFE dataset (incl. via Github, Kaggle...)`** | **HARD BLOCK (Explicitly Prohibited)** |
| **CREMA_D_Frames144** (`frames/*.png`, CSVs, `frame-manifest.json`, `LICENSE.txt`) | `datasets/local_library/CREMA_D_Frames144/` | `148` files (`144` `.png` frames + `4` metadata/license files) | `26.23 MB` | Explicit **ODbL 1.0 + DbCL 1.0** (`LICENSE.txt` preserved) | **CONDITIONALLY AUTHORIZED (with ODbL attribution/share-alike)** |

### Blocked / Omitted Assets Summary
- **Blocked from GitHub upload (retained safely in local staging):** `model/clean_btn_vit_phase12.pt`, `JAFFE`, `ExpW_Eval1400`, `ExpW_Phase12_TrainClean5962`, `FER2013`. (Full legal details and required clearances are in `BLOCKED_ASSETS_REPORT.md`.)
- **Excluded from staging intentionally (Section 8 portability & privacy):** `.venv/` (`4,326 MB`), `__pycache__/`, `.pytest_cache/`, and the user's personal `results/20261121_234630_919_FER2013_PrivateTest_single_view/` evaluation output directory (which remains completely untouched in `VibeShift-AI-Accuracy-Test`).

---

## 3. Verification Results

| Check | Status | Details |
|---|---|---|
| **Local Staging Copy Creation & Completeness** | **PASS** | Created independently at `C:\Users\asus\OneDrive\Desktop\VibeShift-Accuracy-GitHub-Release` excluding `.venv`/caches; all `37,706` files (`443.54 MB`) verified; `HAS_VENV=False`. |
| **Model & Detector SHA-256 Verification (Staging)** | **PASS** | `clean_btn_vit_phase12.pt` (`257,223,791` bytes, `0316ea2a...`) and `yunet-2023mar.onnx` (`232,589` bytes, `8f2383e4...`) exact match. |
| **Dataset File Count & Metadata Verification (Staging)** | **PASS** | All 5 local datasets verified directly on disk (35,887 FER2013, 1,400 ExpW_Eval1400, 5,962 ExpW_TrainClean, 213 JAFFE, 144 CREMA-D). |
| **Pre-Publication Security Audit (Section 11)** | **PASS (Secrets) / ACTION NEEDED (Personal Paths)** | **`0` API keys, `0` GitHub tokens, `0` passwords, `0` SSH keys, `0` `.env` files, `0` unexpected `.exe`/`.dll`/`.db` files.** Flagged absolute personal Windows paths (`C:\Users\asus\...`) inside 3 documentation/manifest files (`STANDALONE_EVALUATOR_COMPLETION_REPORT.md`, `datasets/DATASET_INVENTORY.md`, `datasets/dataset_manifest.json`) that should be sanitized in staging before any future GitHub push. |
| **Pre-Publication Asset-Rights Audit (Sections 2, 4, 6)** | **BLOCKED** | **5 required assets blocked** (1 explicit prohibition: JAFFE; 4 unestablished redistribution rights: `clean_btn_vit_phase12.pt`, `ExpW_Eval1400`, `ExpW_Phase12_TrainClean5962`, `FER2013`). |
| **Friend Setup Guide (`FRIEND_SETUP_GUIDE.md`)** | **UPDATED** | Updated in staging with all 14 required sections (Git, Git LFS, private clone, Python 3.10–3.12, Antigravity IDE + VS Code, `setup.bat`, CPU/CUDA, `--list-datasets`, FER2013/ExpW/JAFFE commands, confusion matrices, report interpretation, checksums, Git LFS pointer troubleshooting, plus honest blocked-asset acquisition steps). |
| **Fresh-Clone Test** | **NOT EXECUTED (BLOCKED)** | Repository was not created or pushed because required model weights and datasets cannot legally be uploaded without permissions. |
| **Git LFS Remote Download Test** | **NOT EXECUTED (BLOCKED)** | Local `git-lfs/3.7.0` verified; no LFS objects uploaded. |
| **Model Load / Dependency / CPU & GPU / Automated Tests / Parity** | **VERIFIED ON READ-ONLY SOURCE (`9/9 pytest`, `0.0000000000` parity); NOT RE-RUN ON REMOTE CLONE (BLOCKED)** | Source evaluator code was copied verbatim without changing model weights, preprocessing, adapters, or calculations. Fresh-clone verification will run once authorized release scope is approved. |

---

## 4. Safety Confirmations

- **Original standalone evaluator (`C:\Users\asus\OneDrive\Desktop\VibeShift-AI-Accuracy-Test`) unchanged:** **CONFIRMED** (strictly read-only; user's existing `results/20261121_234630_919_FER2013_PrivateTest_single_view/` preserved intact).
- **`C:\Users\asus\OneDrive\Desktop\VibeShift-Final` unchanged:** **CONFIRMED** (`git status -s` clean, HEAD unchanged at `ec0a95f4fb7ce054324771a1ec2af2b3c558a4e1`).
- **`C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local` unchanged:** **CONFIRMED** (`0` files modified).
- **Existing GitHub repositories unchanged:** **CONFIRMED** (`Siddharth1968/vibe-shift` untouched; friend's original GitHub repository never accessed or modified; zero GitHub repositories created or pushed).
- **No credentials uploaded or exposed:** **CONFIRMED**.
- **No unauthorized assets uploaded:** **CONFIRMED** (neither model weights nor any restricted dataset were uploaded to GitHub, Git LFS, GitHub Releases, or external storage).
- **No training data presented as an independent test set:** **CONFIRMED** (`Train22757`, `AllTrain28709`, and `ExpW_Phase12_TrainClean5962` retain explicit `TRAINING_DATA` flags and `DevHoldout4015` retains `REUSED_DEV_HOLDOUT`).

---

## 5. Final Verdict

**BLOCKED — REQUIRED ASSETS CANNOT YET BE DISTRIBUTED**
