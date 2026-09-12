# Ocular Adversarial Robustness x Explainable AI (ocular-adv-xai)

A B.Tech research project investigating adversarial robustness and explainability
in ocular biometric recognition systems, building on ABRobOcular
(Krishnamurthy and Rattani, 2025).

## Problem statement

Existing adversarial-attack detectors for ocular biometrics, including
ABRobOcular's own supervised multi-detector, require labeled training examples
of every attack type they aim to catch. This project investigates whether an
ocular recognition model's internal explanation behavior (via LayerCAM and
Eigen-CAM heatmaps) shifts in detectable, attack-specific ways under adversarial
perturbation, and whether this can serve as a lightweight, architecture-agnostic
signal for detection and attribution, without depending on labeled adversarial
training data for every attack family.

Pipeline: for a given ocular image, we (1) get a clean prediction and explanation
map, (2) apply an adversarial attack and get a second prediction and explanation
map, (3) apply a defense and get a third prediction and explanation map, then (4)
compare all three explanation maps to study how attention shifts and whether it
is restored by the defense.

## Repository structure

ocular-adv-xai/
- third_party/ABRobOcular/   vendored reference code, see Acknowledgements
  - AB_Rob_Ocular/            core attack/defense library
  - Detection and Attribution/  ABRobOcular's own multi-detector (has its OWN
    config.yml/train.py/eval.py/predict.py, separate from the top-level ones)
  - Black-Box Defense/         Patch/Strip augmentation notebooks
  - train.py / eval.py / predict.py / dataset_loader.py   base recognition model scripts
  - config.yml
  - fgsm_sample_run.py
  - ORIGINAL_README.md         original repo's README, untouched
  - VENDORED_FROM.txt          provenance: commit hash, date, notes
- src/models/    our reproduction of the recognition model (Xception+ArcFace)
- src/attacks/    FGSM/BIM/MIM/PGD/CW, Patch Occlusion, Monocle Blending
- src/defenses/    JPEG/Strip/EAT, unified defense
- src/xai/          LayerCAM + Eigen-CAM wrappers, divergence metrics
- src/compare/       clean vs attacked vs defended comparison pipeline
- notebooks/     exploratory work
- configs/         experiment configs
- docs/             scoping report, proposal, meeting notes
- data/             gitignored, see Data section below
- outputs/           gitignored except tracked summary tables/figures
- requirements.txt
- README.md

## Setup

git clone https://github.com/nandininautiyal/ocular-adv-xai.git
cd ocular-adv-xai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Data

Datasets (VISOB, UFPR, MICHE, PolyU, CrossEyed) and trained model weights are
not committed to this repo. They are stored at: [team: fill in shared drive link here].
Download and place them under data/ locally before running any training or eval scripts.

## Contribution workflow

- main is the stable branch, no direct commits.
- Each team member works on their own personal branch: deepanshu, aindrila, nandini.
- Merge into main via Pull Request; at least one other teammate reviews before merging.
- Keep commits scoped and messages descriptive.

## Acknowledgements

Portions of this codebase are adapted from ABRobOcular (Krishnamurthy and Rattani, 2025).
The source repository displays an Apache-2.0 license badge; no separate LICENSE file
was present in the source repo at the time of vendoring. Source:
https://github.com/Bharath-K3/ABRobOcular

Krishnamurthy, B., and Rattani, A. (2025). ABRobOcular: Adversarial benchmarking and
robustness analysis of datasets and tools for ocular-based user recognition.
Neurocomputing, 130352.
