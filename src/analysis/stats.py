"""
src/analysis/stats.py

Statistics for the paper: confidence intervals and tests that respect the fact
that images of the same subject are not independent.

Every resampling here is a *cluster* bootstrap over subjects (all images of a
resampled subject come along), and paired tests are run on per-subject means.
"""

import numpy as np
from scipy.stats import wilcoxon

from src.detect.evaluate import auroc


def cluster_bootstrap_ci(values, groups, stat=np.median, n_boot=2000, level=0.95, seed=0):
    """Point estimate and percentile CI of ``stat(values)``, resampling whole groups.

    Returns
    -------
    (estimate, low, high) : floats; NaN if there are no finite values
    """
    values, groups = np.asarray(values, np.float64), np.asarray(groups)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(stat(values))
    ids = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ids]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        boots[b] = stat(values[np.concatenate([members[i] for i in pick])])
    alpha = (1.0 - level) / 2
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def paired_subject_test(a, b, groups):
    """Wilcoxon signed-rank test of a vs b on per-subject means (two-sided).

    Parameters
    ----------
    a, b : arrays of paired per-image values (same length)
    groups : subject of each pair

    Returns
    -------
    dict with ``n_subjects``, ``median_diff`` (a - b, per-subject means),
    ``rank_biserial`` (effect size in [-1, 1], positive = a > b) and ``p``
    """
    a, b, groups = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(groups)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b, groups = a[keep], b[keep], groups[keep]
    ids = np.unique(groups)
    diff = np.array([a[groups == g].mean() - b[groups == g].mean() for g in ids])
    out = {"n_subjects": int(len(ids)), "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
           "rank_biserial": float("nan"), "p": float("nan")}
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return out
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1.0
    out["rank_biserial"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    out["p"] = float(wilcoxon(nonzero).pvalue)
    return out


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaNs stay NaN and are not counted)."""
    p = np.asarray(pvalues, np.float64)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def auroc_ci(neg, pos, neg_groups, pos_groups, n_boot=2000, level=0.95, seed=0):
    """AUROC with a subject-cluster bootstrap CI (a subject's clean and adversarial rows move together)."""
    neg, pos = np.asarray(neg, np.float64), np.asarray(pos, np.float64)
    neg_groups, pos_groups = np.asarray(neg_groups), np.asarray(pos_groups)
    estimate = auroc(neg, pos)
    ids = np.unique(np.concatenate([neg_groups, pos_groups]))
    if len(ids) == 0 or np.isnan(estimate):
        return estimate, float("nan"), float("nan")
    neg_idx = {g: np.flatnonzero(neg_groups == g) for g in ids}
    pos_idx = {g: np.flatnonzero(pos_groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = ids[rng.integers(0, len(ids), len(ids))]
        n = np.concatenate([neg_idx[g] for g in pick])
        p = np.concatenate([pos_idx[g] for g in pick])
        if len(n) and len(p):
            boots.append(auroc(neg[n], pos[p]))
    if not boots:
        return estimate, float("nan"), float("nan")
    alpha = (1.0 - level) / 2
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def paired_auroc_difference(neg_a, pos_a, neg_b, pos_b, neg_groups, pos_groups,
                            n_boot=2000, level=0.95, seed=0):
    """AUROC(a) - AUROC(b) for two scores on the same images, with a paired cluster bootstrap.

    Returns
    -------
    dict with ``diff``, ``low``, ``high`` and a two-sided bootstrap ``p``
    (twice the smaller tail; resamples exactly at zero count half to each side)
    """
    arrays = [np.asarray(x, np.float64) for x in (neg_a, pos_a, neg_b, pos_b)]
    neg_groups, pos_groups = np.asarray(neg_groups), np.asarray(pos_groups)
    diff = auroc(arrays[0], arrays[1]) - auroc(arrays[2], arrays[3])
    ids = np.unique(np.concatenate([neg_groups, pos_groups]))
    neg_idx = {g: np.flatnonzero(neg_groups == g) for g in ids}
    pos_idx = {g: np.flatnonzero(pos_groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = ids[rng.integers(0, len(ids), len(ids))]
        n = np.concatenate([neg_idx[g] for g in pick])
        p = np.concatenate([pos_idx[g] for g in pick])
        if len(n) and len(p):
            boots.append(auroc(arrays[0][n], arrays[1][p]) - auroc(arrays[2][n], arrays[3][p]))
    if not boots or np.isnan(diff):
        return {"diff": float(diff), "low": float("nan"), "high": float("nan"), "p": float("nan")}
    boots = np.asarray(boots)
    alpha = (1.0 - level) / 2
    ties = 0.5 * (boots == 0).mean()
    p = min(1.0, 2 * min((boots < 0).mean() + ties, (boots > 0).mean() + ties))
    return {"diff": float(diff), "low": float(np.quantile(boots, alpha)),
            "high": float(np.quantile(boots, 1 - alpha)), "p": float(p)}
