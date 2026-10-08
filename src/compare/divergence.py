"""
src/compare/divergence.py

Heatmap divergence metrics for the clean → attacked → defended comparison.

This is the **core experiment module**.  The research question is:

    "Does an adversarial attack shift the model's explanation (heatmap),
     and does a defence restore it?"

All functions accept two heatmaps of shape (H, W) with values in [0, 1]
and return a scalar divergence score.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Structural Similarity Index (SSIM)
# ---------------------------------------------------------------------------

def ssim(heatmap_a, heatmap_b, C1=1e-4, C2=9e-4):
    """Compute the Structural Similarity Index (SSIM) between two heatmaps.

    Parameters
    ----------
    heatmap_a, heatmap_b : ndarray (H, W), float32, [0, 1]
    C1, C2 : float — stabilisation constants

    Returns
    -------
    score : float in [-1, 1]
        1.0 = identical, lower = more divergent.
    """
    mu_a = heatmap_a.mean()
    mu_b = heatmap_b.mean()
    sigma_a = heatmap_a.var()
    sigma_b = heatmap_b.var()
    sigma_ab = ((heatmap_a - mu_a) * (heatmap_b - mu_b)).mean()

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)

    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(heatmap_a, heatmap_b):
    """Cosine similarity between two flattened heatmaps.

    Parameters
    ----------
    heatmap_a, heatmap_b : ndarray (H, W), float32, [0, 1]

    Returns
    -------
    score : float in [-1, 1]
        1.0 = identical direction, 0.0 = orthogonal.
    """
    a = heatmap_a.flatten()
    b = heatmap_b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# L2 Distance (normalised)
# ---------------------------------------------------------------------------

def l2_distance(heatmap_a, heatmap_b):
    """Normalised L2 (Euclidean) distance between two heatmaps.

    Normalised by the number of pixels so the value is comparable across
    different spatial resolutions.

    Parameters
    ----------
    heatmap_a, heatmap_b : ndarray (H, W), float32, [0, 1]

    Returns
    -------
    distance : float >= 0
        0.0 = identical.
    """
    diff = (heatmap_a - heatmap_b).flatten()
    return float(np.linalg.norm(diff) / np.sqrt(len(diff)))


# ---------------------------------------------------------------------------
# KL Divergence (discretised heatmaps → probability distributions)
# ---------------------------------------------------------------------------

def kl_divergence(heatmap_a, heatmap_b, eps=1e-10):
    """KL divergence D_KL(A || B) treating normalised heatmaps as
    probability distributions.

    Parameters
    ----------
    heatmap_a, heatmap_b : ndarray (H, W), float32, [0, 1]
    eps : float — smoothing to avoid log(0)

    Returns
    -------
    divergence : float >= 0
        0.0 = identical distributions.
    """
    a = heatmap_a.flatten() + eps
    b = heatmap_b.flatten() + eps
    a = a / a.sum()
    b = b / b.sum()
    return float(np.sum(a * np.log(a / b)))


# ---------------------------------------------------------------------------
# Aggregate comparison helper
# ---------------------------------------------------------------------------

def compare_heatmaps(heatmap_clean, heatmap_attacked, heatmap_defended=None):
    """Run all divergence metrics between the three heatmap stages.

    Parameters
    ----------
    heatmap_clean : ndarray (H, W)
    heatmap_attacked : ndarray (H, W)
    heatmap_defended : ndarray (H, W) or None

    Returns
    -------
    results : dict
        Keys like ``"clean_vs_attacked_ssim"``, ``"clean_vs_defended_cosine"``, etc.
    """
    metrics = {
        "ssim": ssim,
        "cosine": cosine_similarity,
        "l2": l2_distance,
        "kl": kl_divergence,
    }

    results = {}
    for name, fn in metrics.items():
        results[f"clean_vs_attacked_{name}"] = fn(heatmap_clean, heatmap_attacked)
        if heatmap_defended is not None:
            results[f"clean_vs_defended_{name}"] = fn(heatmap_clean, heatmap_defended)
            results[f"attacked_vs_defended_{name}"] = fn(heatmap_attacked, heatmap_defended)

    return results
