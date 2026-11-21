# VibeShift AI Accuracy Evaluator — Local Dataset Inventory

**Generated:** 2026-11-21 (IST)  
**Discovery Mode:** Strictly read-only inspection of known VibeShift dataset locations (`C:\VibeShift-Datasets`, `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local`, `C:\Users\asus\OneDrive\Desktop\VibeShift-Final`, and paths explicitly referenced in project documentation). No unrelated personal directories were scanned, and no new datasets were downloaded.

---

## 1. Storage & Copy Feasibility Summary

- **System Drive (`C:`) Free Space Before Copy:** `135.79 GB` (`313.78 GB` used)
- **Usable Local Image Datasets Copied into `datasets/local_library/`:** **5 dataset collections (`197.77 MB` measured on disk, `37,661` files total)**
  1. `FER2013` (`35,887` images + `5` official split/selection metadata files): **`56.64 MB`**
  2. `ExpW_Eval1400` (`1,400` full-scene photographs + `3` official annotation/manifest files): **`88.19 MB`**
  3. `ExpW_Phase12_TrainClean5962` (`5,962` cleaned 48×48 crops in `.npz` + `audit.json` + `manifest.json`): **`13.25 MB`**
  4. `JAFFE` (`213` original `.tiff` images + `README_FIRST.txt` + `manifest.json`): **`13.46 MB`**
  5. `CREMA_D_Frames144` (`144` extracted `.png` frames + `LICENSE.txt` + CSVs + `frame-manifest.json`): **`26.23 MB`**
- **Large Raw Archives & Duplicate / Non-Dataset Artifacts NOT Copied (Preserved Read-Only at Original Paths):**
  - `ExpW` raw multipart 7-Zip archive (`origin.7z.001..008`, `8,113,576,419` bytes / **`7.56 GB`**): Unextracted compressed archive at `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\cross-dataset\dataset\archives\`. Skipped to avoid creating an uncontrolled 7.56 GB duplicate compressed archive inside OneDrive Desktop.
  - `msambare-fer2013-v1.zip` (`60.34 MB`, SHA-256 `9f3422c7a45b676c2965ef4fc5f3099932aa98fd13eaaa2c254f620a9894a6e9`): Exact duplicate archive of the already extracted 35,887 FER2013 images.
  - `jaffe.zip` (`12.29 MB`, SHA-256 `6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b`): Exact duplicate archive of the already extracted 213 JAFFE images.
  - `FER2013/development/development-pixels.npz` (`25.99 MB`) and `FER2013/development-phase6/development-pixels.npz` (`68.22 MB`): Derived NumPy pixel caches of `FER2013`, not distinct datasets.
  - `phase4/data/clips/*.flv` (`144` video files, `38.29 MB`): Raw source Flash video clips from which `CREMA_D_Frames144/frames/*.png` were extracted.
  - Teacher logit/prediction caches (`*.pt` / `*.npz` under `experiments/maximum-accuracy/`) and synthetic QA images (`webcam-reliability/fer-sad-*.png`, `phase4/tmp/qa/`): Excluded because they are training caches / synthetic fixtures rather than independently labeled benchmark datasets.
- **Explicitly Verified Absent Locally (NOT Present on Disk):**
  - **FER+**: Not present locally.
  - **RAF-DB**: Not present locally (`experiments/rafdb-evaluation/access-status.json` confirms `status: PERMISSION_REQUIRED`, `dataset_found: false`).
  - **AffectNet**: Not present locally as an image dataset (only an externally trained AffectNet-7 initialization checkpoint was used in upstream research).
  - **KDEF**: Not present locally (`experiments/kdef-evaluation/dataset-access.json` confirms `found: false`, registration-gated).

---

## 2. Detailed Dataset Inventory

### Dataset 1: `FER2013` (Complete Local Corpus + Research Split Definitions)
- **Original Paths:**
  - Canonical extracted images: `C:\VibeShift-Datasets\FER2013\train\` (`28,709` images) & `C:\VibeShift-Datasets\FER2013\test\` (`7,178` images)
  - Research split manifest (22,757 train / 4,015 dev): `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\efficientnet-advanced\split.csv`
  - Phase 6 PublicTest selection (3,536 clean): `C:\VibeShift-Datasets\FER2013\development-phase6\selection.csv`
  - Dataset audit & metadata: `C:\VibeShift-Datasets\FER2013\development-phase6\dataset-report.json`, `C:\VibeShift-Datasets\downloads\kaggle-msambare-fer2013-v1-metadata.json`
- **Local Library Copy:** `datasets/local_library/FER2013/`
- **Version / Provenance:** ICML 2013 FER-2013 packaged via Kaggle `msambare/fer2013` v1 (`archive_sha256: 9f3422c7a45b676c2965ef4fc5f3099932aa98fd13eaaa2c254f620a9894a6e9`)
- **Total Images:** `35,887` (`28,709` Training + `3,589` PublicTest + `3,589` PrivateTest)
- **Resolution & Format:** `48 × 48` uint8 1-channel grayscale `.jpg` (pre-cropped faces)
- **Label Mapping:** Folder names directly match CleanBTNVit canonical 7-class order (`0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise`).
- **Ground-Truth Annotation Format:** Class-named subdirectories (`train/<emotion>/`, `test/<emotion>/`); `test/` filenames carry verified official prefixes (`PublicTest_*.jpg` and `PrivateTest_*.jpg`); research training/dev splits defined in `metadata/split.csv` and `metadata/selection_phase6.csv`.
- **Disk Size:** `~54.2 MB` copied (`43.09 MB` train + `10.80 MB` test + metadata CSVs/JSONs)
- **Duplicate Relationships:**
  - `C:\VibeShift-Datasets\downloads\msambare-fer2013-v1.zip` (`60.34 MB`) is an exact duplicate archive of these 35,887 images (not copied).
  - `C:\VibeShift-Datasets\FER2013\development\development-pixels.npz` and `development-phase6\development-pixels.npz` are derived pixel caches (not copied).
  - Internal FER2013 corpus duplicates documented in `dataset-report.json`: `1,035` exact duplicate groups in `train`, `81` in `test`, and `53` exact/near overlap images between `PublicTest` and `PrivateTest`.
- **Splits & Per-Class Counts:**
  - **`PrivateTest` (`3,589`):** `angry: 491, disgust: 55, fear: 528, happy: 879, neutral: 626, sad: 594, surprise: 416`
  - **`PublicTest` (`3,589`):** `angry: 467, disgust: 56, fear: 496, happy: 895, neutral: 607, sad: 653, surprise: 415`
  - **`PublicTest_Clean3536` (`3,536` — excludes 53 PrivateTest overlaps):** `angry: 460, disgust: 55, fear: 492, happy: 890, neutral: 601, sad: 648, surprise: 390`
  - **`DevHoldout4015` (`4,015` — Training-derived via `split.csv`):** `angry: 561, disgust: 52, fear: 564, happy: 1054, neutral: 721, sad: 698, surprise: 365`
  - **`Train22757` (`22,757` — CleanBTNVit Phase 12 training split via `split.csv`):** `angry: 3182, disgust: 298, fear: 3194, happy: 5974, neutral: 4087, sad: 3954, surprise: 2068`
  - **`AllTrain28709` (`28,709` — includes 1,937 quarantined duplicates/cross-split matches):** `angry: 3995, disgust: 436, fear: 4097, happy: 7215, neutral: 4965, sad: 4830, surprise: 3171`
- **Leakage & Research History:**
  - **Used to train CleanBTNVit Phase 12?** `Train22757`: **YES** (`AllTrain28709` contains `Train22757`). `DevHoldout4015`, `PublicTest`, `PrivateTest`: **NO**.
  - **Used for checkpoint selection?** `DevHoldout4015`: **YES** (heavily reused across Phases 8–12). `PublicTest_Clean3536`: **YES in earlier Phases 6 & 8**, untouched during Phase 12. `PrivateTest`: **NO** (strictly untouched in Phase 12 training and selection).
  - **Previously inspected during research?** `Train22757` & `DevHoldout4015`: **YES**. `PublicTest`: **YES** (Phases 6/8). `PrivateTest`: **NO for Phase 12 CleanBTNVit**.
  - **Suitable for independent final testing?**
    - `PrivateTest` (`3,589`): **YES** (Primary held-out FER2013 test split for CleanBTNVit Phase 12; disclose the 53 corpus-inherent PublicTest/PrivateTest overlap items documented in `dataset-report.json` and 439 train-to-private corpus matches quarantined out of `Train22757`).
    - `PublicTest` / `PublicTest_Clean3536`: **PARTIAL** (Untouched during Phase 12, but previously inspected in Phases 6/8).
    - `DevHoldout4015` / `Train22757` / `AllTrain28709`: **NO** (Reused dev / training leakage).
- **Licensing & Sharing Eligibility:**
  - Kaggle package lists `DbCL-1.0`, while original FER2013 images were harvested from web image searches without individual release agreements.
  - **Public redistribution in GitHub:** **NO / UNVERIFIED** (excluded via `.gitignore`; stored only in `datasets/local_library/`).
  - **Academic teammate sharing:** **UNVERIFIED — teammates should download directly from Kaggle `msambare/fer2013` under their own acceptance of terms.**

---

### Dataset 2: `ExpW_Eval1400` (Balanced 1,400-Image Cross-Dataset Full-Scene Subset)
- **Original Paths:**
  - Images: `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\cross-dataset\dataset\images\origin\` (`1,400` `.jpg` images)
  - Official annotations & readme: `...\experiments\cross-dataset\expw-label.lst` (`SHA-256: 0a08a33189c3e366a79ae6e147cfb29a98a2416ba00754f80f102bec24a907a5`), `expw-readme.txt`
  - Frozen balanced selection: `...\experiments\cross-dataset\sample-manifest.json` (`seed: 20261121`)
- **Local Library Copy:** `datasets/local_library/ExpW_Eval1400/`
- **Version / Provenance:** CUHK MMLab Expression in-the-Wild (ExpW, Zhang et al., arXiv:1609.06426v2)
- **Total Images:** `1,400` unique full-scene photographs (one annotated face per image)
- **Resolution & Format:** Variable-resolution uint8 RGB `.jpg` (ranging from ~200×200 to 1600×1600; full photographs)
- **Original Label Mapping (CRITICAL DIFFERENCE from CleanBTNVit!):**
  - ExpW `expw-readme.txt`: `0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral`
  - CleanBTNVit canonical: `0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise`
  - Explicit adapter remaps `4→sad (5)`, `5→surprise (6)`, `6→neutral (4)` using `expw-readme.txt` and `sample-manifest.json`.
- **Per-Class Counts:** `angry: 200, disgust: 200, fear: 200, happy: 200, neutral: 200, sad: 200, surprise: 200`
- **Disk Size:** `~95.0 MB` (`83.38 MB` images + `11.5 MB` `expw-label.lst` + manifest/readme)
- **Duplicate Relationships:** Sampled from the unextracted `origin.7z.001..008` archive (`7.56 GB`, not copied). Strictly disjoint from `ExpW_Phase12_TrainClean5962` (`excluded_eval_source_images: 1400` in `phase11_expw/audit.json`).
- **Leakage & Research History:**
  - **Used to train CleanBTNVit Phase 12?** **NO** (explicitly excluded by source filename before Phase 12 ExpW selection).
  - **Used for checkpoint selection?** **NO**.
  - **Previously inspected during research?** **YES** (evaluated by earlier baseline BTN & EfficientNet models in `VIBESHIFT_CROSS_DATASET_ACCURACY_REPORT.md`; NOT evaluated by Phase 12 CleanBTNVit).
  - **Suitable for independent final testing?** **CONDITIONAL YES** (Cross-dataset transfer benchmark untouched by Phase 12 training; note earlier project models were evaluated on it, and `full_photo` vs `annotated_crops` protocols must be reported separately).
- **Licensing & Sharing Eligibility:**
  - **License:** Non-commercial academic and educational research only (CUHK MMLab; authors disclaim copyright over web-sourced images).
  - **Public redistribution:** **NO**.
  - **Academic teammate sharing:** **NOT ESTABLISHED** (teammates should obtain ExpW directly from CUHK MMLab under non-commercial research terms).

---

### Dataset 3: `ExpW_Phase12_TrainClean5962` (Cleaned Supplementary Training Subset)
- **Original Path:** `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\maximum-accuracy\phase11_expw\` (`expw_clean_48.npz`, `audit.json`, `manifest.json`)
- **Local Library Copy:** `datasets/local_library/ExpW_Phase12_TrainClean5962/`
- **Total Images:** `5,962` pre-cropped 48×48 uint8 grayscale faces packed in `expw_clean_48.npz` (`SHA-256: 653ce501622123ee747ffae0870f19fb9c98a3a59c6fa5b63cdade522689c930`)
- **Label Mapping:** Already mapped to CleanBTNVit canonical order (`0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise`).
- **Per-Class Counts:** `angry: 941, disgust: 1001, fear: 500, happy: 749, neutral: 916, sad: 1087, surprise: 768`
- **Disk Size:** `~14.4 MB`
- **Leakage & Research History:**
  - **Used to train CleanBTNVit Phase 12?** **YES** (Stage 1 pure soft-KD training).
  - **Suitable for independent final testing?** **NO — TRAINING DATA (LEAKAGE).** Included strictly for audit/reproducibility and diagnostic inspection.
- **Licensing & Sharing Eligibility:** ExpW non-commercial academic research terms — **NO public redistribution**; **NOT ESTABLISHED** for third-party sharing.

---

### Dataset 4: `JAFFE` (Official v2 — Japanese Female Facial Expression Database)
- **Original Path:** `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\kdef-evaluation\dataset\jaffe\` (`213` `.tiff` images) + `README_FIRST.txt` + `manifest.json`
- **Local Library Copy:** `datasets/local_library/JAFFE/`
- **Version / Provenance:** Official JAFFE v2, Zenodo DOI `10.5281/zenodo.14974867` (Lyons, Kamachi, Gyoba, 1997; `jaffe.zip` SHA-256 `6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b`)
- **Total Images:** `213` posed portraits from 10 female subjects (`KA, KL, KM, KR, MK, NA, NM, TM, UY, YM`)
- **Resolution & Format:** `256 × 256` uint8 `.tiff` (`212` 1-channel grayscale, `1` 3-channel RGB; head-and-shoulders portraits)
- **Label Mapping (Official `README_FIRST.txt` Filename Pose Convention):**
  - `AN → angry`, `DI → disgust`, `FE → fear`, `HA → happy`, `NE → neutral`, `SA → sad`, `SU → surprise`
- **Per-Class Counts:** `angry: 30, disgust: 29, fear: 32, happy: 31, neutral: 30, sad: 31, surprise: 30`
- **Disk Size:** `~13.4 MB` (duplicate archive `jaffe.zip` `12.29 MB` skipped)
- **Leakage & Research History:**
  - **Used to train CleanBTNVit Phase 12?** **NO**.
  - **Used for checkpoint selection?** **NO**.
  - **Previously inspected during research?** **YES** (evaluated by earlier baseline BTN & EfficientNet models in `VIBESHIFT_KDEF_ACCURACY_REPORT.md`; NOT evaluated by Phase 12 CleanBTNVit).
  - **Suitable for independent final testing?** **CONDITIONAL YES** (Small posed cross-dataset benchmark untouched by Phase 12 training; full-photo YuNet detection protocol recommended by default).
- **Licensing & Sharing Eligibility:**
  - **License:** Non-commercial scientific research only; requires citing Lyons et al. (arXiv:2009.05938 & arXiv:2107.13998).
  - **Public redistribution:** **EXPLICITLY PROHIBITED** (`README_FIRST.txt`: *"Not Permitted: Redistribution of the JAFFE dataset (incl. via Github, Kaggle...)"*).
  - **Academic teammate sharing:** **PROHIBITED** — each teammate must download JAFFE directly from Zenodo DOI `10.5281/zenodo.14974867`.

---

### Dataset 5: `CREMA_D_Frames144` (144-Frame Extracted Subset of CREMA-D)
- **Original Paths:**
  - Extracted frames: `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\phase4\data\frames\` (`144` `.png` files)
  - License & CSV metadata: `...\experiments\phase4\data\LICENSE.txt`, `SentenceFilenames.csv`, `tabulatedVotes.csv`
  - Frame manifest: `...\experiments\phase4\results\frame-manifest.json`
- **Local Library Copy:** `datasets/local_library/CREMA_D_Frames144/`
- **Version / Provenance:** CREMA-D (`CheyneyComputerScience/CREMA-D`), middle video frame extracted per clip (`144` clips across actors)
- **Total Images:** `144` full-scene video frames
- **Resolution & Format:** `480 × 360` uint8 RGB `.png` (full photographs)
- **Label Mapping (6-Class Subset Only!):**
  - `ANG → angry (24)`, `DIS → disgust (24)`, `FEA → fear (24)`, `HAP → happy (24)`, `NEU → neutral (24)`, `SAD → sad (24)`
  - **`surprise` (`0` images):** CREMA-D does not include a `surprise` class! Predictions of `surprise` by the 7-class model count as errors in the 7-class confusion matrix, and active 6-class macro F1 is reported alongside 7-class metrics.
- **Disk Size:** `~22.2 MB` (source `.flv` video clips `38.29 MB` excluded as non-image video files)
- **Leakage & Research History:**
  - **Used to train CleanBTNVit Phase 12?** **NO**.
  - **Used for checkpoint selection?** **NO**.
  - **Previously inspected during research?** **YES** (used in Phase 4 face-detector/pipeline benchmark; NOT evaluated by Phase 12 CleanBTNVit).
  - **Suitable for independent final testing?** **CONDITIONAL YES** (Small 6-class cross-dataset video-frame test; requires full-photo YuNet detection and explicit disclosure of 0 `surprise` support).
- **Licensing & Sharing Eligibility:**
  - **License (`LICENSE.txt`):** Open Database License (`ODbL 1.0`) + Database Contents License (`DbCL 1.0`).
  - **Public redistribution:** **CAUTIOUS HOLD / UNVERIFIED FOR GITHUB** — kept in `datasets/local_library/` (excluded from Git) rather than `datasets/shareable/` until ODbL share-alike / attribution / human-subject video-frame terms are explicitly verified.
  - **Academic teammate sharing:** Teammates may obtain CREMA-D directly from the official `CheyneyComputerScience/CREMA-D` repository under ODbL 1.0 / DbCL 1.0.

---

### Dataset 6 (Not Copied — Archive Reference Only): `ExpW_RawArchive_7z`
- **Original Path:** `C:\Users\asus\OneDrive\Desktop\moodre\VibeShift-local\experiments\cross-dataset\dataset\archives\` (`origin.7z.001` through `origin.7z.008`)
- **Size:** `8,113,576,419` bytes (`7,737.71 MB` / `7.56 GB`)
- **Annotations Covered:** `91,793` face bounding boxes across `68,096` unique images (referenced by `expw-label.lst`)
- **Copy Status:** **SKIPPED** (unextracted multipart `.7z` archive; omitted from local copy to avoid duplicating 7.56 GB on OneDrive Desktop while preserving the ready-to-evaluate `ExpW_Eval1400` and `ExpW_Phase12_TrainClean5962` subsets).

---

### Manual User Dataset Folder: `dataset/`
- **Path:** `dataset/{angry,disgust,fear,happy,neutral,sad,surprise}/`
- **Status:** Intentionally **EMPTY** (`0` images) for user-supplied labeled evaluation images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
