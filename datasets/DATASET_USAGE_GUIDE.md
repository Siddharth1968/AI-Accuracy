# Dataset Usage Guide — Choosing the Right Split & Protocol

This guide helps you select the scientifically appropriate dataset, split, and image protocol when evaluating **CleanBTNVit Phase 12**.

---

## 1. Quick Evaluation Eligibility Table

| Dataset | Split | Images | Recommended Protocol | Independent Test for Phase 12? | Notes |
|---|---|---:|---|---|---|
| **`FER2013`** | `PrivateTest` | 3,589 | `pre_cropped` | **YES (Primary Held-Out FER2013 Test)** | Untouched during Phase 12 training & selection. Disclose the 53 corpus-inherent PublicTest/PrivateTest overlaps in `dataset-report.json`. |
| **`FER2013`** | `PublicTest` / `PublicTest_Clean3536` | 3,589 / 3,536 | `pre_cropped` | Partial | Untouched in Phase 12, but inspected in Phases 6 & 8. |
| **`FER2013`** | `DevHoldout4015` | 4,015 | `pre_cropped` | **NO (Reused Dev Holdout)** | Used across Phases 8–12 for checkpoint selection & soup tuning. Useful only for reproducing dev numbers (`74.10%` single / `74.37%` flip). |
| **`FER2013`** | `Train22757` / `AllTrain28709` | 22,757 / 28,709 | `pre_cropped` | **NO (Training Data)** | Contains CleanBTNVit Phase 12 training data — never report as test accuracy! |
| **`ExpW_Eval1400`** | `full_photo` | 1,400 | `full_photo` (YuNet) | **Conditional YES** | Cross-dataset real-world photos; strictly excluded from Phase 12 training. Earlier baseline models were evaluated here. |
| **`ExpW_Eval1400`** | `annotated_crops` | 1,400 | `pre_cropped` | **Conditional YES** | Uses official ExpW target bounding boxes (isolates classifier accuracy from YuNet detection). |
| **`ExpW_Phase12_TrainClean5962`** | `train_clean` | 5,962 | `pre_cropped` | **NO (Training Data)** | Used in Phase 12 Stage 1 soft-KD training — never report as test accuracy! |
| **`JAFFE`** | `all` | 213 | `full_photo` (YuNet) | **Conditional YES** | Small posed cross-dataset benchmark (10 female subjects); untouched in Phase 12 training. |
| **`CREMA_D_Frames144`** | `all` | 144 | `full_photo` (YuNet) | **Conditional YES (6 classes)** | 144 video frames across 6 classes (`surprise` has 0 samples!). Untouched in Phase 12 training. |
| **`dataset`** | `default` | User-supplied | `pre_cropped` or `full_photo` | User-dependent | Empty 7-class folder for your own manually added labeled images. |

---

## 2. Protocol Distinction: `pre_cropped` vs. `full_photo`

- **`pre_cropped` (Default for FER2013, ExpW `annotated_crops`, `ExpW_Phase12_TrainClean5962`, and `dataset/`):**
  - Feeds each face image directly into exact Phase 12 preprocessing (uint8 grayscale → `112×112` PyTorch bilinear interpolation → `.mul(255).round().div(255)` → 3-channel ImageNet normalization) without invoking YuNet.
- **`full_photo` (Default for `ExpW_Eval1400/full_photo`, `JAFFE`, `CREMA_D_Frames144`; optional for `dataset/` via `--protocol full_photo`):**
  - Runs OpenCV **YuNet** (`detector/yunet-2023mar.onnx`) first.
  - **No face detected:** Recorded explicitly as `no_face_detected` (not silently dropped).
  - **Multiple faces detected (without target bounding box):** Recorded explicitly as `ambiguous_multi_face` — the evaluator **never** picks a random face and claims valid ground truth. (For `ExpW_Eval1400/full_photo`, when an official target bounding box is provided in `sample-manifest.json`, the evaluator matches the highest-IoU YuNet detection with `IoU >= 0.3`, matching the project's documented cross-dataset protocol.)
  - Reports **both** conditional accuracy among valid detected faces **and** end-to-end accuracy across all readable images!

---

## 3. Sharing Rules (Local Library vs. GitHub)

- **`datasets/local_library/`** is excluded from Git by `.gitignore` because these datasets carry academic/non-commercial restrictions or explicit non-redistribution clauses (e.g., JAFFE forbids GitHub/Kaggle redistribution).
- **`datasets/shareable/`** intentionally contains no restricted datasets. Only copy datasets here if you have verified public redistribution permission.
- See `datasets/DATASET_INVENTORY.md` and `FRIEND_SETUP_GUIDE.md` for how teammates should obtain each dataset legally.
