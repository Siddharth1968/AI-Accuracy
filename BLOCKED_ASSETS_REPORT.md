# VibeShift AI Accuracy Evaluator — Blocked Assets & Redistribution Rights Report

**Date:** 2026-11-21 (IST)  
**Staging Directory:** `C:\Users\asus\OneDrive\Desktop\VibeShift-Accuracy-GitHub-Release`  
**Proposed Repository:** `Siddharth1968/vibeshift-ai-accuracy-evaluator` (Private — **NOT CREATED YET**)  
**Publication Status:** **BLOCKED — REQUIRED ASSETS CANNOT YET BE DISTRIBUTED**

---

## 1. Why Complete GitHub Publication Was Stopped

The release objective requires that a teammate be able to clone a single GitHub repository and immediately run the exact `CleanBTNVit Phase 12` checkpoint across all 5 local datasets without manually locating weights or downloading datasets separately.

Per the mandatory pre-publication compliance rules:
- **Educational use, non-commercial use, Git LFS, ZIP packaging, or private GitHub hosting do NOT automatically grant legal redistribution rights for third-party datasets or restricted derived weights.**
- **Assets without established redistribution permission must NOT be uploaded, and must NOT be silently dropped while claiming full clone-only functionality.**

Audit of every primary license and provenance file on disk identified **5 required assets (the CleanBTNVit checkpoint + 4 of 5 local datasets)** that cannot currently be uploaded to GitHub.

---

## 2. Asset-by-Asset Redistribution Rights Audit

| # | Asset | Local Staging Path | Files / Size | Audit Outcome | Primary Evidence & License Clause | Specific Permission Required to Unblock |
|---|---|---|---:|---|---|---|
| **1** | **CleanBTNVit Phase 12 Weights** (`clean_btn_vit_phase12.pt`) | `model/clean_btn_vit_phase12.pt` | `1` file (`257,223,791` bytes, SHA-256 `0316ea2a...`) | **BLOCKED (Unestablished)** | Derived from upstream **AffectNet** (`AffectNet7_67.60.pth` — University of Denver EULA restricts access to registered faculty Lead Researcher + institutional Permitted Researchers and prohibits third-party transfer), **ExpW** (`5,962` images in Stage 1 soft-KD — non-commercial academic only, no redistribution grant), **RAF-DB** (Stage 1 `raf_btn` teacher — restricts derived data to same-site non-commercial research), **MS-Celeb-1M** (`ir50.pth` backbone — retracted June 2019; original terms prohibited corpus/derivative redistribution), and **`SeoulTech-HCIRLab/BTN`** Dropbox weights (no weights license file). | Written institutional/faculty **Lead Researcher** clearance confirming academic-team eligibility under AffectNet/ExpW/RAF-DB terms AND legal clearance (or non-MS-Celeb-1M / open-licensed re-training) establishing permission to host derived weights in a private GitHub LFS repo. |
| **2** | **JAFFE Official v2** | `datasets/local_library/JAFFE/` | `215` files (`213` `.tiff` images, `13.46 MB`) | **HARD BLOCK (Explicitly Prohibited)** | Verbatim quote in `JAFFE/README_FIRST.txt` (DOI `10.5281/zenodo.14974867`, Lyons, Kamachi, Gyoba): **`> Not Permitted: - Redistribution of the JAFFE dataset (incl. via Github, Kaggle, Colaboratory, GitCafe, CSDN etc.)`**. | Written redistribution waiver directly from **Dr. Michael J. Lyons** (Ritsumeikan University, copyright holder). Without written waiver, JAFFE **must NEVER be uploaded to GitHub** (even privately); teammates must download directly from Zenodo DOI `10.5281/zenodo.14974867`. |
| **3** | **ExpW Balanced Evaluation Subset (`ExpW_Eval1400`)** | `datasets/local_library/ExpW_Eval1400/` | `1,403` files (`1,400` `.jpg` photos, `88.19 MB`) | **BLOCKED (Unestablished)** | `expw-readme.txt` provides citation metadata only and grants **no redistribution right**. CUHK MMLab distributes ExpW strictly for non-commercial academic/educational research and disclaims copyright over the underlying web-sourced photographs. | Written redistribution permission from **CUHK MMLab** (and copyright clearance for re-hosting 1,400 web-sourced photographs on GitHub). Otherwise, teammates must obtain ExpW directly from CUHK MMLab. |
| **4** | **ExpW Cleaned Training Subset (`ExpW_Phase12_TrainClean5962`)** | `datasets/local_library/ExpW_Phase12_TrainClean5962/` | `3` files (`5,962` face crops in `.npz`, `13.25 MB`) | **BLOCKED (Unestablished)** | Derived directly from CUHK MMLab ExpW face photographs under the same non-commercial academic restriction with no redistribution grant. | Same written redistribution clearance from **CUHK MMLab** as `ExpW_Eval1400`. |
| **5** | **FER2013 Corpus** | `datasets/local_library/FER2013/` | `35,892` files (`35,887` `.jpg` images, `56.64 MB`) | **BLOCKED (Unverified Underlying Image Rights)** | `kaggle-msambare-fer2013-v1-metadata.json` carries a `DbCL-1.0` tag placed by third-party Kaggle repacker `msambare` (`userSpecifiedSources: ""`), but the original ICML 2013 FER-2013 corpus was compiled from Google Image Search without individual copyright clearances or subject releases. A third-party Kaggle repacker's `DbCL-1.0` label does not reliably grant copyright permission to re-host 35,887 face images on GitHub. | Verified copyright/redistribution authorization covering the underlying FER-2013 images. Otherwise, teammates should download `msambare/fer2013` directly from Kaggle under their own acceptance of Kaggle's terms. |
| **6** | **CREMA-D 144-Frame Subset (`CREMA_D_Frames144`)** | `datasets/local_library/CREMA_D_Frames144/` | `148` files (`144` `.png` frames, `26.23 MB`) | **CONDITIONALLY AUTHORIZED (ODbL 1.0 / DbCL 1.0)** | `CREMA_D_Frames144/LICENSE.txt` explicitly licenses the database under **ODbL 1.0** (`http://opendatacommons.org/licenses/odbl/1.0/`) and individual contents under **DbCL 1.0** (`http://opendatacommons.org/licenses/dbcl/1.0/`). | Authorized for redistribution **provided** `LICENSE.txt`, ODbL share-alike terms, and attribution to Cao et al. (*CREMA-D*, IEEE TAC 2014) are preserved. Retained locally in staging since overall release is blocked by items 1–5. |
| **7** | **YuNet ONNX Detector (`yunet-2023mar.onnx`)** | `detector/yunet-2023mar.onnx` + `detector/yunet-LICENSE.txt` | `2` files (`232,589` bytes, SHA-256 `8f2383e4...`) | **AUTHORIZED (MIT License)** | OpenCV Zoo / Shiqi Yu MIT License (`detector/yunet-LICENSE.txt` included in staging). | None — ready for GitHub publication. |
| **8** | **Evaluator Code, Adapters, Tests, Scripts & Docs** | `evaluate.py`, `model/btn_runtime/`, `model/classifier.py`, `model/preprocess.py`, `detector/face_detector.py`, `evaluator/`, `scripts/`, `tests/`, `setup.bat`, `run_evaluation.bat`, docs | `29` files (`< 0.3 MB`) | **AUTHORIZED (Subject to Personal-Path Cleanup)** | User's own code + MIT-licensed `SeoulTech-HCIRLab/BTN` architecture modules. Pre-publication security audit found **0 secrets/credentials**, and flagged `C:\Users\asus\...` personal path strings in 3 documentation/manifest files (`STANDALONE_EVALUATOR_COMPLETION_REPORT.md`, `datasets/DATASET_INVENTORY.md`, `datasets/dataset_manifest.json`). | Sanitize `C:\Users\asus\...` personal paths in those 3 docs inside the staging copy before any GitHub push. |

---

## 3. Recommended Legal Path Forward (User Decision Required)

Because **complete clone-with-weights-and-all-5-datasets on GitHub is blocked**, you have two clean, compliant options:

1. **Option A — Publish a Clean, Fully Authorized Code + YuNet Repository (`Siddharth1968/vibeshift-ai-accuracy-evaluator`):**
   - Sanitize the 3 staging docs containing `C:\Users\asus\...`.
   - Push all evaluator Python code, tests, `setup.bat`, `run_evaluation.bat`, MIT-licensed `detector/yunet-2023mar.onnx`, empty `dataset/` 7-class template, and (optionally, with ODbL attribution) `CREMA_D_Frames144`.
   - Keep `model/clean_btn_vit_phase12.pt` and the 4 restricted datasets (`FER2013`, `ExpW_Eval1400`, `ExpW_Phase12_TrainClean5962`, `JAFFE`) excluded via `.gitignore`.
   - Teammates clone the repo, run `setup.bat`, download each benchmark dataset directly from its official source (Kaggle / Zenodo / CUHK MMLab), and install `clean_btn_vit_phase12.pt` privately once faculty/institutional authorization is confirmed.
2. **Option B — Obtain Written Permissions First:**
   - Secure explicit written redistribution waivers from the relevant rights holders / institutional Lead Researcher before enabling Git LFS upload of any currently blocked asset.
