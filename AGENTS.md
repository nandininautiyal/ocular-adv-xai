# Project Context: Ocular Adversarial XAI

This file provides critical context, strict coding guidelines, and the current state of the repository for any AI agent assisting on this project. **Always read this file before suggesting architectural changes.**

## Project Goal
This research project evaluates how adversarial attacks on ocular biometric models (specifically **Xception + ArcFace**) affect their Explainable AI (XAI) heatmaps, and whether defences can restore the original heatmaps (Divergence analysis).

## Team Roles & Workflow
- **Nandini**: Handles the heavy model training on Kaggle (e.g., `03_train_baseline.py` for 155M parameters, 10 epochs). She provides the `.keras` model checkpoints.
- **Deepanshu (You)**: Handles the evaluation pipeline, XAI heatmaps, attack wrappers, and defences.
- **Andrilla**: Collaborates on the repository.
- **Workflow**: All major features should be developed on personal branches, raised as Draft PRs, and merged into `main` after review.

## Architecture & Codebase Structure
- **Vendored Code:** The `third_party/ABRobOcular/` directory contains reference implementations of the ArcFace model, data loaders, and attacks. **NEVER modify files inside `third_party/`**.
- **Model Signature:** The core model is an Xception base with an ArcFace head. It requires **dual inputs**: `[image, label]`. 
- **Inference Rules:** During inference, attacks, or XAI generation, the `label` input MUST be a zero vector of shape `(1, 300)` to bypass the ArcFace margin penalty.
- **Image Contract:** All functions handling images must expect/return shape `(256, 256, 3)`, dtype `float32`, bounded in `[0, 1]`.

## Strict Coding Guidelines
1. **TensorFlow & Keras Legacy:** 
   You MUST set the environment variable `TF_USE_LEGACY_KERAS = "1"` before importing TensorFlow to ensure compatibility with `tf_keras`.
   ```python
   import os
   os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
   import tensorflow as tf
   ```
2. **Re-use, Don't Re-invent:** 
   When implementing standard attacks (FGSM, BIM, PGD, CW, MIM), do not write the gradient logic yourself. Instead, write thin wrappers in `src/attacks/whitebox.py` that call the `_face` methods in `third_party.ABRobOcular.AB_Rob_Ocular.AB_Rob_Ocular_Tensor.OcularAdversarialAttacks`.
   *Note: The `label` input in these methods requires a batch dimension (e.g., `(1, 300)`).*
3. **Pure Functions:**
   All attack, defence, and XAI attribution methods must be implemented as pure functions without side effects.
4. **XAI Targets:**
   When generating CAM heatmaps, the target layer must dynamically find the last `Conv2D`, `DepthwiseConv2D`, or **`SeparableConv2D`** (critical for Xception) layer in the model.

## Pipeline Overview
- `Phase 1 (XAI)`: Heatmap generation (`src/xai/attribution.py`) using LayerCAM and Eigen-CAM.
- `Phase 2 (Attacks)`: Adversarial generation (`src/attacks/whitebox.py` & `blackbox.py`). Blackbox includes Monocle Blending and Patch Occlusion.
- `Phase 3 (Defences)`: Inference-time defences (`src/defences/inference.py`) like JPEG compression and Strip defence.
- `Phase 4 (Compare)`: Heatmap divergence metrics (`src/compare/divergence.py`) using SSIM, L2, KL Divergence, and Cosine Similarity.
- `Phase 5 (Orchestration)`: The end-to-end runner (`scripts/05_run_experiment.py`) requires a `.keras` checkpoint to execute.

## 🤖 Agent Directives (Meta-Rules)
**To any AI agent reading this file:** 
It is YOUR responsibility to help Deepanshu keep this file up to date. 
1. Before concluding a major coding session, proactively ask Deepanshu: *"Should we update `AGENTS.md` to reflect the new changes we just made?"*
2. If the team's workflow, roles, or pipeline architecture changes, you must immediately suggest updating this file.
3. This file is our shared memory across all models (Claude, Gemini, etc.). Do not let it go stale.
