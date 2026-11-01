# Ocular Adversarial Robustness x Explainable AI (ocular-adv-xai)

A B.Tech research project investigating adversarial robustness and explainability
in ocular biometric recognition systems, building on ABRobOcular
(Krishnamurthy and Rattani, 2025).

## Problem statement

Ocular (periocular) recognition models can be fooled by small adversarial
perturbations, and defences such as JPEG compression are usually judged only by
whether the predicted identity comes back. This project asks what the attacks do
to the model's explanations, and whether a defence restores the reasons for a
decision, not just the label:

1. How do the LayerCAM and Eigen-CAM heatmaps of an Xception + ArcFace ocular
   recognition model change under corrected white-box attacks (FGSM, BIM, PGD,
   MIM, CW) and the base paper's black-box attacks (patch occlusion, monocle
   blending), compared with noise of the same size?
2. Where on the eye does the evidence for the true identity go, and where does
   the evidence for the attacker's identity sit? This is measured with
   anatomical eye masks and against an area-share baseline.
3. When JPEG or bit-depth reduction restores the label, does it also restore
   the explanation, compared with the same defence applied to a noisy image?
   And does the defence still work against adaptive attacks that know it is there?

Along the way the project corrects flaws in the base paper's released attack
code. It evaluates recognition with verification metrics (EER, TAR at a fixed
FAR) on a cross-session and an open-set protocol, because the base paper's
random split puts near-identical video frames in both train and test.
Training-free attack detection from heatmap consistency was explored and
dropped: it is already published for generic images, and here it added nothing
over prediction-based detection. AGENTS.md has the current status and the open
decisions.

Pipeline: for each correctly classified test image we (1) get the clean
prediction and explanation maps, (2) attack the image and recompute them,
(3) apply a defence and recompute them, and (4) compare the maps with each
other and with control maps (noise of the same budget, the defended clean
image, the defended noisy image), over the whole map and per eye region.

## Repository structure

ocular-adv-xai/
- third_party/ABRobOcular/   vendored reference code, see Acknowledgements (never modified)
  - AB_Rob_Ocular/            core attack/defense library
  - Detection and Attribution/  ABRobOcular's own multi-detector (has its OWN
    config.yml/train.py/eval.py/predict.py, separate from the top-level ones)
  - Black-Box Defense/         Patch/Strip augmentation notebooks
  - train.py / eval.py / predict.py / dataset_loader.py   base recognition model scripts
  - config.yml
  - fgsm_sample_run.py
  - ORIGINAL_README.md         original repo's README, untouched
  - VENDORED_FROM.txt          provenance: commit hash, date, notes
- src/models/    wrapper around the reference model (Xception+ArcFace) and margin-free logits; anatomy-guided
  heads on the same backbone (anatomy_model.py: attention pooling with an eye-mask loss, a parts head whose
  score splits exactly over the eye parts, an AttNet-style gate) and their mask-supervised trainer
- src/data/      training data with eye masks: 03's augmentation applied to the image and its mask (masked_generator.py)
- src/attacks/   FGSM/BIM/PGD/MIM/CW, Patch Occlusion, Monocle Blending; the 8-bit budget
  helper (budget.py); adaptive attacks through the defences (adaptive.py); the ablation of
  the vendored attacks' flaws (ablation.py)
- src/defences/  JPEG and bit-depth reduction (inference time, with parameters); Strip and Patch
  Augmentation (training time). EAT and the unified defense are training-time too.
- src/xai/       LayerCAM + Eigen-CAM; heuristic eye masks (masks.py); learned eye masks (seg_masks.py);
  Anatomical Shapley Attribution, exact Shapley values over eye-part groups (shapley.py)
- src/compare/   heatmap divergence metrics; region mass, area, enrichment and entropy per eye region
- src/detect/    consistency-based attack detection (kept for reference)
- src/analysis/  statistics (subject-cluster bootstrap, Wilcoxon, Holm, permutation tests) and
  the area-baseline null
- src/eval/      verification and identification metrics (EER, TAR at FAR, rank-1, CIs)
- src/generated_masks.zip   legacy: 150 preliminary heuristic masks made before the PR #24 mask
  rewrite; no code uses it
- scripts/       00-23, see below
- notebooks/     next_run.ipynb (the next Kaggle session, end to end); baseline-run.ipynb
  (training record); train_cross_session.ipynb (cross-session retraining, 2 seeds, and its evaluation);
  train_anatomy.ipynb (the anatomy-guided models under the same protocol, their evaluation and ASA)
- docs/          mask_labelling.md (hand-labelling eye masks with local tools)
- splits/        visob_test.csv, the pinned test list of the checkpoint
- tests/         pytest suite (CPU, tiny ArcFace model)
- configs/       experiment configs
- data/          gitignored
- requirements.txt
- README.md

### Scripts

| Script | What it does | Runs on |
|---|---|---|
| 00_sanity_check.py | first plumbing check of the vendored model on the small HuggingFace ABRobOcular_Attacks subset (not VISOB) | any |
| 01_build_manifest.py | one row per VISOB image (relpath, subject, device, lighting, eye, phase, frame index), parsed from the paths | Kaggle |
| 02_make_splits.py | splits and symlink tree: `random` (paper protocol, test pinned to splits/visob_test.csv), `cross_session` (train on enrollment, test on verification images), `open_set` (subjects outside the 300 classes) | Kaggle |
| 03_train_baseline.py | trains Xception + ArcFace; writes `class_index_<tag>.json` next to the checkpoint | Kaggle GPU |
| 04_eval_clean.py | clean accuracy (overall, per device, lighting, eye, phase) and the list of correctly classified images | Kaggle GPU |
| 05_run_experiment.py | the main experiment: attacks, defences, controls, heatmaps, region masses, every map saved | Kaggle GPU |
| 06_attack_smoke.py | vendored attacks at eps = 0.001 (paper replication only) | Kaggle GPU |
| 07_check_patch_positions.py | renders the paper's Fig. 7 patch positions | any |
| 08_check_attacks.py | attack success against epsilon on the real checkpoint | Kaggle GPU |
| 09_heatmap_demo.py | clean, attacked and defended heatmaps of a few images | Kaggle GPU |
| 10_check_masks.py | QC overlays of the heuristic masks | Kaggle |
| 11_run_detection.py | consistency-based attack detection (reference; not a contribution) | Kaggle GPU |
| 12_analyse.py | statistics, tables, figures, report.md and report_v2.md from the CSVs of 05 and 11 | laptop |
| 13_attack_ablation.py | vendored to standard attacks, fixing one flaw at a time | Kaggle GPU |
| 14_adaptive_attacks.py | non-adaptive PGD vs BPDA-PGD vs transform-robust PGD through JPEG and bit depth | Kaggle GPU |
| 15_eval_verification.py | verification and identification: P1 (random split, reproduction only), P2 (cross-session, training subjects), P3 (open-set, unseen subjects) | Kaggle GPU |
| 16_check_dataset.py | checks of the Kaggle VISOB copy (counts, synthetic frames, test-train overlap) and per-image heuristic mask areas | Kaggle CPU |
| 17_make_eye_masks.py | learned eye masks (periorbital DeepLabV3, zero-shot) in model-input geometry | Kaggle GPU |
| 18_validate_masks.py | exports a labelling pack; scores eye masks against hand labels (IoU, Dice) | Kaggle or laptop |
| 20_prepare_ubipr.py | manifest of the UBIPr images for zero-shot cross-dataset verification (15 x1) | Kaggle |
| 21_make_train_masks.py | 64 x 64 training masks for every image (or converted from 17's file) and how often each eye part is in the crop | Kaggle GPU |
| 22_anatomical_shapley.py | Anatomical Shapley Attribution of clean, noisy, attacked and defended images, next to CAM and attention part masses and the parts model's exact receipt | Kaggle GPU |
| 23_analyse_asa.py | statistics, tables, figures and report_asa.md from the rows of 22 | laptop |

Each script's docstring gives its smoke-run and full-run commands.

## Setup

Python 3.11 or newer (requirements.txt pins the versions used for the tests and the analysis):

git clone https://github.com/nandininautiyal/ocular-adv-xai.git
cd ocular-adv-xai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

PyTorch and torchvision are needed only to make the eye masks (17); Kaggle has them preinstalled.

## How to run

- **Tests (laptop, CPU, no data):** set `TF_USE_LEGACY_KERAS=1`, then run `python -m pytest tests -q`.
  The suite uses a tiny ArcFace model and synthetic images.
- **Experiments (Kaggle):** open `notebooks/next_run.ipynb` in a private Kaggle notebook with GPU
  T4 x2, Internet on, the secret `github_token`, and the inputs `mubashirulhassan00/visob-augmented-recent`
  and `nandininautiyal/ocular-xception-arcface-full-baseline`. Run it once with `RUN_FULL = False`
  (checks, smoke runs of every script, time estimates: about 1 h 15 min), then set `RUN_FULL = True`
  and use Save Version > Save & Run All (about 4-5 h). It runs 01, 02, 04, 17, 15, 05, 14, 16 and 12
  on both GPUs and the CPU at the same time, guards Kaggle's 12-hour limit, and ends with a checklist.
  Download `results.zip` from the Output panel.
- **Anatomy-guided models (Kaggle):** `notebooks/train_anatomy.ipynb`, run like the notebook above
  (`RUN_FULL = False` first, then Save & Run All): eye masks (17, 21), `03 --arch ...` for every model,
  04, 15 and then 22 + 23 on the best mask model and the cross-session baseline. Resumable across sessions.
- **Single scripts:** the order is 01 -> 02 -> 04 (correct-image list) -> 05 / 14 / 15 / 16 / 17. Every
  TensorFlow script runs in its own process (`!python ...`); never import TensorFlow in a notebook cell.
  Start every long run with a 2-3 minute smoke run.
- **Analysis (laptop, no model):** `python scripts/12_analyse.py --experiment-rows rows_<tag>.csv --out-dir analysis_<tag>`;
  add `--areas` with 16's per-image area CSV for runs whose rows have no area columns.
- **Eye-mask validation:** label the exported images with a local tool and score them with
  `scripts/18_validate_masks.py`; see docs/mask_labelling.md.

## Data

Datasets, trained model weights and results are not committed to this repo (`.keras` / `.h5`
files are gitignored; the checkpoint is 1.7 GB). The current pipeline uses VISOB only.

- VISOB: the Kaggle copy `mubashirulhassan00/visob-augmented-recent` (v1, 134,258 crops of
  240 x 160 pixels, which the loaders stretch to 256 x 256). It is a third-party upload whose
  licence fields conflict; the official route is a signed VISOB Data Sharing Agreement, to be
  requested through the supervisor. `scripts/01_build_manifest.py` documents the expected layout.
  Never redistribute the images.
- Checkpoint `Xception_ArcFace_full.keras`: a private Kaggle Dataset shared by Nandini;
  ask her to share it with your Kaggle account.

## Where results live

- The Kaggle notebook outputs (each session's `results.zip`).
- Deepanshu's laptop, `D:\TempCLG\results`: the 27 Oct run (rows, summaries, image list, examples),
  its analyses (`analysis_full/`, `analysis_v2/`) and a copy of Nandini's 25 Oct run (`nandini_oct25/`).
- Never in the repo (result files were removed in PR #13) and never on a public page.

## Contribution workflow

- main is the stable branch, no direct commits.
- Each team member works on their own personal branch: deepanshu, aindrila, nandini.
- Merge into main via Pull Request; at least one other teammate reviews before merging.
- Keep commits scoped and messages descriptive.
- Nothing is published to the web: Kaggle notebooks and datasets stay private.

## Acknowledgements

Portions of this codebase are adapted from ABRobOcular (Krishnamurthy and Rattani, 2025).
The source repository displays an Apache-2.0 license badge; no separate LICENSE file
was present in the source repo at the time of vendoring. Source:
https://github.com/Bharath-K3/ABRobOcular

Krishnamurthy, B., and Rattani, A. (2025). ABRobOcular: Adversarial benchmarking and
robustness analysis of datasets and tools for ocular-based user recognition.
Neurocomputing, 130352.
