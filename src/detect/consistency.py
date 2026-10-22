"""
src/detect/consistency.py

Reference-free adversarial-example detection by explanation consistency.

A deployed recogniser never sees the clean version of an input, so it cannot
compare "clean vs attacked" heatmaps. What it can do is compare an input with
a mildly transformed copy of itself (JPEG, bit-depth reduction): a natural
image keeps its explanation, while an adversarial perturbation is fragile and
its explanation jumps. Every score here needs only the input, and no score is
fitted to adversarial examples.

Scores (higher = more suspicious), per transform ``t``:

    xc_spearman_<xai>_<t>   1 - Spearman(map(x), map(t(x)))       explanation consistency
    xc_js_<xai>_<t>         JS divergence between the two maps    explanation consistency
    fs_l1_<t>               L1 between softmax outputs            Feature Squeezing baseline
                                                                  (Xu et al., NDSS 2018)
    pred_cos_<t>            1 - cosine(logits(x), logits(t(x)))   prediction-consistency baseline

Both maps of an image explain the class predicted for ``x`` (the true class is
unknown at deployment). Maps are compared at native resolution (8 x 8).
"""

import numpy as np
from scipy.stats import spearmanr

from src.compare.divergence import js_divergence
from src.defences.inference import DEFENCE_REGISTRY
from src.models.arcface_inference import predict_logits
from src.xai.attribution import XAI_REGISTRY


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _heatmaps(model, xai, images, class_idx):
    fn = XAI_REGISTRY[xai]
    if xai == "eigencam":                       # class-agnostic
        return fn(model, images, size=None)
    return fn(model, images, class_idx=class_idx, size=None)


def rank_inconsistency(map_a, map_b):
    """1 - Spearman correlation, defined for constant maps too.

    Returns 0.0 if both maps are constant (nothing to rank, nothing changed) and
    1.0 if exactly one is (the explanation collapsed or appeared).
    """
    a, b = np.ravel(map_a), np.ravel(map_b)
    const_a, const_b = np.ptp(a) == 0, np.ptp(b) == 0
    if const_a and const_b:
        return 0.0
    if const_a or const_b:
        return 1.0
    return float(1.0 - spearmanr(a, b).correlation)


def detection_features(model, images, xai_names=("layercam", "eigencam"),
                       transforms=("jpeg", "bit_depth")):
    """Reference-free detection scores for a batch.

    Parameters
    ----------
    model : tf.keras.Model with dual inputs [image, label]
    images : ndarray (N, 256, 256, 3), float32, [0, 1]
    xai_names : names in ``XAI_REGISTRY``
    transforms : names in ``DEFENCE_REGISTRY``

    Returns
    -------
    features : dict name -> ndarray (N,); also ``pred`` (predicted class of x)
    """
    images = np.asarray(images, dtype=np.float32)
    logits = predict_logits(model, images)
    pred = logits.argmax(axis=1)
    prob = _softmax(logits)
    unit = logits / np.linalg.norm(logits, axis=1, keepdims=True)
    maps = {xai: _heatmaps(model, xai, images, pred) for xai in xai_names}

    features = {"pred": pred}
    for t in transforms:
        moved = DEFENCE_REGISTRY[t](images)
        moved_logits = predict_logits(model, moved)
        features[f"fs_l1_{t}"] = np.abs(_softmax(moved_logits) - prob).sum(axis=1)
        moved_unit = moved_logits / np.linalg.norm(moved_logits, axis=1, keepdims=True)
        features[f"pred_cos_{t}"] = 1.0 - (unit * moved_unit).sum(axis=1)
        for xai in xai_names:
            moved_maps = _heatmaps(model, xai, moved, pred)
            features[f"xc_spearman_{xai}_{t}"] = np.array(
                [rank_inconsistency(a, b) for a, b in zip(maps[xai], moved_maps)])
            features[f"xc_js_{xai}_{t}"] = np.array(
                [js_divergence(a, b) for a, b in zip(maps[xai], moved_maps)])
    return features


def score_columns(columns):
    """Detection-score columns among *columns*, grouped by family."""
    return {
        "explanation": [c for c in columns if c.startswith("xc_")],
        "feature_squeezing": [c for c in columns if c.startswith("fs_l1_")],
        "prediction": [c for c in columns if c.startswith("pred_cos_")],
    }


class PercentileFusion:
    """Training-free fusion: each score becomes its percentile among clean
    calibration images, and the fused score is the maximum percentile.

    Feature Squeezing fuses its squeezers with a max in the same way. Only
    clean images are used, so no adversarial example is ever seen.
    """

    def __init__(self, clean_scores):
        """clean_scores : dict name -> ndarray of calibration (clean) scores."""
        self.sorted = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in clean_scores.items()}

    def percentile(self, name, values):
        ref = self.sorted[name]
        return np.searchsorted(ref, np.asarray(values, dtype=np.float64), side="right") / len(ref)

    def __call__(self, scores, names):
        """Max percentile over *names*; scores : dict name -> ndarray."""
        return np.max(np.stack([self.percentile(n, scores[n]) for n in names]), axis=0)
