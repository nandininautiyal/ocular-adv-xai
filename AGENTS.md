# Project Context: Ocular Adversarial XAI

This file provides critical context, strict coding guidelines, and the current state of the repository for any AI agent assisting on this project. **Always read this file before suggesting architectural changes.**

## Project Goal
This research project evaluates how adversarial attacks on ocular biometric models (specifically **Xception + ArcFace**) affect their Explainable AI (XAI) heatmaps, and whether defences can restore the original heatmaps (Divergence analysis). The README's longer-term question is whether heatmap shifts can *detect and attribute* attacks without labelled adversarial training data; note that clean-vs-attacked divergence needs the clean image, which a deployed system does not have, so detection needs reference-free signals (e.g. heatmap(x) vs heatmap(JPEG(x))).

## Team Roles & Workflow
- **Nandini**: Handles the heavy model training on Kaggle (e.g., `03_train_baseline.py` for 155M parameters, 10 epochs). She provides the `.keras` model checkpoints.
- **Deepanshu (You)**: Handles the evaluation pipeline, XAI heatmaps, attack wrappers, and defences.
- **Aindrila**: Collaborates on the repository.
- **Workflow**: All major features should be developed on personal branches, raised as Draft PRs, and merged into `main` after review. Keep PRs small: one module plus its tests.
- **Never push directly to `main` and never force-push or rewrite its history** (AI agents included). Commit with your real GitHub email (`git config user.email`); a placeholder email once forced a history rewrite of `main`.
- **No AI attribution in commits or PRs:** no `Co-Authored-By` lines for AI tools and no "Generated with ..." footers. Commits are authored by the team member only.

## Deadline & Deliverables
- **Deadline: before the end of November 2026.** Deliverables: a project report and a journal-level research paper.
- The supervisor expects novelty over the base paper (ABRobOcular), with explainability as a core part.
- **Planned contributions (proposed 2026-10-22, to be confirmed by the team):**
  1. **Corrected re-benchmark:** ABRobOcular's attack code has four bugs; the corrected attacks are much stronger (e.g. BIM success at eps 0.001: 6 % → 75 %).
  2. **Anatomy-grounded explanation analysis:** how LayerCAM/Eigen-CAM attention moves between pupil, iris, sclera and periocular skin under each attack and defence, against noise and defended-clean controls.
  3. **Main novelty: training-free detection by explanation consistency.** Score an input by how much its heatmap changes under a mild transform (JPEG, bit depth), which needs no clean reference and no adversarial training data. Evaluate with AUROC and TPR at 5 % FPR, on attacks never seen during calibration. Compare against Feature Squeezing (prediction-only consistency, Xu et al. 2018) and a supervised detector trained on seen attacks, as in ABRobOcular.
  4. **If time allows:** a second backbone, and an adaptive attack against the detector.

## Current Status (2026-10-22)
- Done: baseline training (97.94 % margin-free); clean evaluation (13,585 of 13,868 test images correct, 97.96 %); standard attacks checked on the real checkpoint (`08_check_attacks.py`, 100 images: success rises with epsilon, BIM/PGD/MIM reach 100 % at 2/255, FGSM 87 %); a 6-image heatmap demo (`09_heatmap_demo.py`); heuristic concept masks (`src/xai/masks.py`, PR #15) with a visual QC script (`10`).
- Not done yet: the full `05_run_experiment.py` run on the real checkpoint. Run a `--max-images 20` dry run first, then `--per-class 2`. Default epsilons are 0.001, 0.5/255, 1/255, 2/255, because larger budgets already succeed 100 %.
- `06_attack_smoke.py` uses the vendored attacks; its numbers (BIM 6 % < FGSM 37 % at eps 0.001) reflect their bugs and are for paper replication only.
- The concept masks are a classical OpenCV estimate; validate them against a few hand-annotated images before reporting region-level results.

## Where Things Live
- **Test split is pinned:** `splits/visob_test.csv` lists the 13,868 test images of the run that trained the checkpoint; `02_make_splits.py` always uses it (rows sorted, remaining images split 75/25 into train/validation). **Never evaluate the checkpoint on a freshly randomised split**: before 2026-10-22 the split depended on disk file order, and a new Kaggle session put about 70 % of the original train/validation images into "test" (accuracy 98.61 % instead of the true 97.96 %).
- **Checkpoint:** `Xception_ArcFace_full.keras` (1.7 GB, 300 classes, test accuracy 97.94 % margin-free) is a Kaggle output, shared as a private Kaggle Dataset. **Never commit `.keras` / `.h5` files** (GitHub's limit is 100 MB; `.gitignore` excludes them).
- **Kaggle checkpoint dataset:** `nandininautiyal/ocular-xception-arcface-full-baseline` (path `/kaggle/input/datasets/nandininautiyal/ocular-xception-arcface-full-baseline/trained_models/Xception_ArcFace_full.keras`).
- **Experiments run on Kaggle**, next to the checkpoint and the VISOB data: `01` → `02` → `04_eval_clean.py` (writes the correct-image list) → `05_run_experiment.py`. `notebooks/` holds the Kaggle notebooks with their printed outputs; result files are no longer committed under `outputs/` (removed in PR #13) and live in the Kaggle notebook outputs.
- **Environment:** TF 2.20 with `tf-keras` 2.20.x (the versions the checkpoint was trained with); `opencv-python-headless<5` (OpenCV 5 removed the Haar cascades that Monocle Blending needs).

## Architecture & Codebase Structure
- **Vendored Code:** The `third_party/ABRobOcular/` directory contains reference implementations of the ArcFace model, data loaders, and attacks. **NEVER modify files inside `third_party/`**.
- **Model Signature:** The core model is an Xception base with an ArcFace head. It requires **dual inputs**: `[image, label]`.
- **Inference Rules:** During inference, attacks, or XAI generation, the `label` input MUST be a zero vector of shape `(N, n_classes)` to bypass the ArcFace margin penalty. Read `n_classes` from the model (`model.inputs[1].shape[-1]`) instead of hard-coding 300.
- **The vendored ArcFace layer returns softmax probabilities, not logits.** With scale s = 64 the top probability is exactly 1.0 for confident predictions, so its gradient vanishes. Anything that needs gradients (XAI, attacks) must use the margin-free cosine logits from `src/models/arcface_inference.py` (`get_margin_free(model)`). In eager code, never compute cross-entropy on `model.output`: Keras clips probabilities to [1e-7, 1 - 1e-7] and the clip has zero gradient (`model.fit` only works because graph mode reads the logits out of the Softmax op).
- **Image Contract:** All functions handling images must expect/return shape `(256, 256, 3)` (or a batch `(N, 256, 256, 3)`), dtype `float32`, bounded in `[0, 1]`.

## Strict Coding Guidelines
1. **TensorFlow & Keras Legacy:**
   You MUST set the environment variable `TF_USE_LEGACY_KERAS = "1"` before importing TensorFlow to ensure compatibility with `tf_keras`.
   ```python
   import os
   os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
   import tensorflow as tf
   ```
2. **Attacks: standard implementations, vendored code only for replication.**
   `src/attacks/whitebox.py` has standard FGSM, BIM, PGD, MIM and CW-L2 on the margin-free logits; use these for the XAI study, with `epsilon` chosen explicitly. The default attack loss is the margin loss (best other logit minus true logit, which never saturates); `loss="ce"` keeps cross-entropy. CW accepts `epsilon=None` only for uniform calls and ignores it; the runner runs it once. The vendored `_face` attacks have bugs (double softmax; BIM/MIM project onto the epsilon ball only after the last step; PGD ignores `epsilon`; "CW" is sign steps plus N(0, 0.1) noise), so they are exposed only as `paper_*` wrappers to reproduce the paper's ASR table. Load the vendored module from its file (as `whitebox.py` does): the package `__init__` imports PyTorch.
3. **Pure Functions:**
   All attack, defence, and XAI attribution methods must be implemented as pure functions without side effects (seeded randomness, inputs never modified in place).
4. **XAI Targets:**
   CAM heatmaps target the last 4-D activation before the head: `block14_sepconv2_act` for Xception (`find_feature_layer`). Do **not** use the raw `SeparableConv2D` output `block14_sepconv2`: it is pre-BatchNorm/ReLU and ~95 % of its activations are negative. Explain the margin-free logit, never the softmax output.
5. **Tests:** every module ships with tests in `tests/` (pytest on CPU, using the tiny ArcFace model in `tests/conftest.py`). Run `python -m pytest tests -q` before opening a PR.

## Pipeline Overview
- `Phase 1 (XAI)`: Heatmap generation (`src/xai/attribution.py`) using LayerCAM and Eigen-CAM (sign-corrected). `size=None` returns the native 8 x 8 map.
- `Phase 2 (Attacks)`: Adversarial generation (`src/attacks/whitebox.py` & `blackbox.py`). Blackbox includes Monocle Blending (Haar eye box, procedural stand-in for the unreleased monocle image) and Patch Occlusion (positions are (row, col), matching the paper's Fig. 7; changed in PR #14).
- `Phase 3 (Defences)`: Inference-time defences (`src/defences/inference.py`): JPEG compression and bit-depth reduction. **Strip and Patch Augmentation are training-time defences** (`src/defences/training.py`): they need separately trained models, as does EAT.
- `Phase 4 (Compare)`: Heatmap divergence metrics (`src/compare/divergence.py`): Spearman and top-k intersection (primary), JS divergence, cosine, L2, global SSIM, centre-of-mass shift, computed on native-resolution maps. Region mass (`src/compare/regions.py`, runner flag `--concept-masks`): fraction of each heatmap on pupil / iris / sclera / periocular, using masks of the clean image (`src/xai/masks.py`) area-averaged to the 8 x 8 grid; `dmass_<region>` is the change from the clean map.
- `Phase 4b (Detection)`: Reference-free, training-free detection (`src/detect/`, `scripts/11_run_detection.py`): scores how much an input's heatmap changes under JPEG / bit depth (`xc_*`), against Feature Squeezing (`fs_l1_*`) and logit cosine (`pred_cos_*`). Thresholds and percentile fusion use clean calibration images only, disjoint from evaluation images. Reports AUROC, TPR at 5 % FPR, a supervised detector on held-out attacks (leave-one-attack-out), and attack clustering (ARI). Maps explain the class predicted for the input, never the true class.
- `Phase 5 (Orchestration)`: The end-to-end runner (`scripts/05_run_experiment.py`) runs on Kaggle with the `.keras` checkpoint and the correct-image list from `04`. It explains the **true class** at every stage and includes two controls (random noise of the same budget; defence applied to the clean image). Judge restoration against the defended clean map (`vs_defended_clean_*`), not only the clean map: JPEG moves heatmaps of clean images too.

- `Phase 2b (Attack ablation)`: `src/attacks/ablation.py` + `scripts/13_attack_ablation.py` run FGSM/BIM/MIM from the vendored settings (double softmax, 5x step, projection only at the end, raw momentum; reproduced exactly, see `tests/test_ablation.py`) to the standard ones, fixing one flaw at a time, plus vendored vs standard PGD and CW. This is the evidence behind contribution C1.
- Detection fusion is robust z-scores against clean calibration images (`RobustZFusion`, mean over scores); the earlier max-of-percentiles fusion saturated at 1.0, which made TPR at 5 % FPR read 0.
- `Phase 6 (Analysis)`: `scripts/12_analyse.py` (laptop, no model) turns `rows_*.csv` (05) and `features_*.csv` (11) into the paper's tables, figures and `report.md`. Statistics (`src/analysis/stats.py`): 95 % subject-cluster bootstrap CIs, Wilcoxon signed-rank tests on per-subject means, Holm correction, paired bootstrap for AUROC differences. Images of one subject are never treated as independent.

## 🤖 Agent Directives (Meta-Rules)
**To any AI agent reading this file:**
It is YOUR responsibility to help Deepanshu keep this file up to date.
1. Before concluding a major coding session, proactively ask Deepanshu: *"Should we update `AGENTS.md` to reflect the new changes we just made?"*
2. If the team's workflow, roles, or pipeline architecture changes, you must immediately suggest updating this file.
3. This file is our shared memory across all models (Claude, Gemini, etc.). Do not let it go stale.
4. Run code before claiming it works, and never push to `main` yourself.
5. **Check every stage before moving on.**
   - Start each session by finding out what changed since the last one: `git fetch`, `git log origin/main`, open PRs and their CI, and unmerged teammate branches that touch the same files.
   - Confirm a step's prerequisites before starting it, and check its output before building on it.
   - Begin every long run (Kaggle) with a 2–3 minute smoke run that checks its inputs and prints a time estimate.
   - If a step only makes sense after a team or supervisor decision, flag it instead of starting it.
   - Say which checks passed.
