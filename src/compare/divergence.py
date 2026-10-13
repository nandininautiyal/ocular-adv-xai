"""
src/compare/divergence.py

Heatmap divergence metrics for the clean → attacked → defended comparison.

This is the **core experiment module**.  The research question is:

    "Does an adversarial attack shift the model's explanation (heatmap),
     and does a defence restore it?"

All functions accept two heatmaps of the same shape (H, W) with values in
[0, 1] and return a scalar. They work at any resolution: compare the native
8 x 8 maps (``size=None`` in src/xai) rather than 256 x 256 upsampled copies,
whose interpolated pixels are not independent.

Primary metrics follow the saliency-robustness literature (Ghorbani et al.,
2019, "Interpretation of Neural Networks is Fragile"): Spearman rank
correlation and top-k intersection. The rest are secondary.
"""

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Rank-based metrics (primary)
# ---------------------------------------------------------------------------

def spearman(heatmap_a, heatmap_b):
    """Spearman rank correlation between two heatmaps.

    Returns
    -------
    score : float in [-1, 1]
        1.0 = same ranking of locations. 0.0 if either map is constant.
    """
    a, b = np.ravel(heatmap_a), np.ravel(heatmap_b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def top_k_intersection(heatmap_a, heatmap_b, k=0.1):
    """Fraction of the top-*k* locations of A that are also in the top-*k* of B.

    Parameters
    ----------
    k : float in (0, 1] — fraction of locations (at least one)

    Returns
    -------
    score : float in [0, 1]
        1.0 = the most salient regions coincide.
    """
    a, b = np.ravel(heatmap_a), np.ravel(heatmap_b)
    n = max(1, int(round(k * a.size)))
    top_a = set(np.argsort(-a, kind="stable")[:n])
    top_b = set(np.argsort(-b, kind="stable")[:n])
    return len(top_a & top_b) / n


# ---------------------------------------------------------------------------
# Distribution and geometry metrics (secondary)
# ---------------------------------------------------------------------------

def _as_distribution(heatmap, eps):
    p = np.ravel(heatmap).astype(np.float64) + eps
    return p / p.sum()


def js_divergence(heatmap_a, heatmap_b, eps=1e-10):
    """Jensen-Shannon divergence (base 2) between heatmaps as distributions.

    Symmetric and bounded, unlike KL, so exact zeros (common after ReLU in
    LayerCAM) cannot blow it up.

    Returns
    -------
    divergence : float in [0, 1]
        0.0 = identical distributions.
    """
    p, q = _as_distribution(heatmap_a, eps), _as_distribution(heatmap_b, eps)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def kl_divergence(heatmap_a, heatmap_b, eps=1e-10):
    """KL divergence D_KL(A || B) treating normalised heatmaps as
    probability distributions.

    Asymmetric and unbounded: where B is ~0 and A is not, the value is dominated
    by *eps*. Prefer :func:`js_divergence`; this is kept for comparison.

    Returns
    -------
    divergence : float >= 0
        0.0 = identical distributions.
    """
    p, q = _as_distribution(heatmap_a, eps), _as_distribution(heatmap_b, eps)
    return float(np.sum(p * np.log(p / q)))


def cosine_similarity(heatmap_a, heatmap_b):
    """Cosine similarity between two flattened heatmaps.

    Returns
    -------
    score : float in [-1, 1]
        1.0 = identical direction, 0.0 = orthogonal.
    """
    a = np.ravel(heatmap_a).astype(np.float64)
    b = np.ravel(heatmap_b).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_distance(heatmap_a, heatmap_b):
    """Root-mean-square difference between two heatmaps.

    Returns
    -------
    distance : float >= 0
        0.0 = identical.
    """
    diff = np.ravel(heatmap_a).astype(np.float64) - np.ravel(heatmap_b).astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def global_ssim(heatmap_a, heatmap_b, C1=1e-4, C2=9e-4):
    """SSIM computed over the whole map as a single window.

    This is *not* the usual windowed SSIM (Wang et al., 2004), which averages
    local Gaussian windows; at 8 x 8 a local window would cover the whole map
    anyway. C1 and C2 are the standard constants for a dynamic range of 1.

    Returns
    -------
    score : float in [-1, 1]
        1.0 = identical, lower = more divergent.
    """
    a = np.asarray(heatmap_a, np.float64)
    b = np.asarray(heatmap_b, np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (a.var() + b.var() + C2)
    return float(numerator / denominator)


def centre_of_mass_shift(heatmap_a, heatmap_b):
    """Distance between the two heatmaps' centres of mass, as a fraction of the map diagonal.

    Returns
    -------
    shift : float in [0, 1]
        0.0 = attention centred on the same point.
    """
    def centre(m):
        m = np.asarray(m, np.float64)
        total = m.sum()
        if total <= 0:
            return np.array([(m.shape[0] - 1) / 2.0, (m.shape[1] - 1) / 2.0])
        rows, cols = np.indices(m.shape)
        return np.array([(rows * m).sum() / total, (cols * m).sum() / total])

    shape = np.shape(heatmap_a)
    diagonal = np.hypot(shape[0] - 1, shape[1] - 1) or 1.0
    return float(np.linalg.norm(centre(heatmap_a) - centre(heatmap_b)) / diagonal)


# ---------------------------------------------------------------------------
# Aggregate comparison helper
# ---------------------------------------------------------------------------

METRICS = {
    "spearman": spearman,
    "top10_intersection": top_k_intersection,
    "js": js_divergence,
    "cosine": cosine_similarity,
    "l2": l2_distance,
    "global_ssim": global_ssim,
    "com_shift": centre_of_mass_shift,
}


def compare_heatmaps(heatmap_ref, heatmap_other):
    """Run every metric between a reference heatmap (usually the clean one) and another.

    Parameters
    ----------
    heatmap_ref, heatmap_other : ndarray (H, W), float32, [0, 1]

    Returns
    -------
    results : dict
        Metric name -> float, keys as in :data:`METRICS`.
    """
    if np.shape(heatmap_ref) != np.shape(heatmap_other):
        raise ValueError(f"Shape mismatch: {np.shape(heatmap_ref)} vs {np.shape(heatmap_other)}")
    return {name: fn(heatmap_ref, heatmap_other) for name, fn in METRICS.items()}
