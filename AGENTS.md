# Project Context: Ocular Adversarial XAI

This file provides critical context and strict coding guidelines for any AI agent working in this repository.

## Project Goal
This research project evaluates how adversarial attacks on ocular biometric models (specifically Xception + ArcFace) affect their Explainable AI (XAI) heatmaps, and whether defences can restore the original heatmaps.

## Architecture & Codebase Structure
- **Vendored Code:** The `third_party/ABRobOcular/` directory contains reference implementations of the ArcFace model, data loaders, and attacks. **NEVER modify files inside `third_party/`**.
- **Model Signature:** The core model is an Xception base with an ArcFace head. It requires **dual inputs**: `[image, label]`. 
- **Inference Rules:** During inference, attacks, or XAI generation, the `label` input MUST be a zero vector of shape `(1, 300)` to bypass the ArcFace margin penalty.
- **Image Contract:** All functions handling images must expect/return shape `(256, 256, 3)`, dtype `float32`, bounded in `[0, 1]`.

## Strict Coding Guidelines
1. **TensorFlow & Keras Legacy:** 
   You MUST set the environment variable `TF_USE_LEGACY_KERAS = "1"` before importing TensorFlow.
   ```python
   import os
   os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
   import tensorflow as tf
   ```
2. **Re-use, Don't Re-invent:** 
   When implementing standard attacks (FGSM, BIM, PGD, CW, MIM), do not write the gradient logic yourself. Instead, write thin wrappers in `src/attacks/whitebox.py` that call the `_face` methods in `third_party.ABRobOcular.AB_Rob_Ocular.AB_Rob_Ocular_Tensor.OcularAdversarialAttacks`.
3. **Pure Functions:**
   All attack, defence, and XAI attribution methods must be implemented as pure functions (e.g., `def attack(image, model): return adv_image`).
4. **XAI Targets:**
   When generating CAM heatmaps (e.g., LayerCAM, Eigen-CAM), the target layer should dynamically find the last `Conv2D`, `DepthwiseConv2D`, or `SeparableConv2D` layer in the model.

## Pipeline Overview
- `Phase 1 (XAI)`: Heatmap generation (`src/xai/attribution.py`)
- `Phase 2 (Attacks)`: Adversarial generation (`src/attacks/whitebox.py` & `blackbox.py`)
- `Phase 3 (Defences)`: Inference-time defences (`src/defences/inference.py`)
- `Phase 4 (Compare)`: Heatmap divergence metrics (`src/compare/divergence.py`)
- `Phase 5 (Orchestration)`: The end-to-end runner (`scripts/05_run_experiment.py`) requires a `.keras` checkpoint to execute.
