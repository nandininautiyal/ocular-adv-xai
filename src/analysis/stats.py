"""
src/analysis/stats.py

Statistics for the paper: confidence intervals and tests that respect the fact
that images of the same subject are not independent.

Every resampling here is a *cluster* bootstrap over subjects (all images of a
resampled subject come along), and paired tests are run on per-subject means.
A CI should describe the quantity a test tests: :func:`paired_subject_ci`
gives the CI of the per-subject mean difference that
:func:`paired_subject_test` tests, next to the image-level medians of
:func:`cluster_bootstrap_ci`.
"""

import numpy as np
from scipy.stats import rankdata, wilcoxon

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
    if len(nonzero) <= 13 and len(np.unique(np.abs(nonzero))) < len(nonzero):
        out["p"] = _signflip_wilcoxon_p(nonzero)        # what scipy computes here, without its slow loop
    else:
        out["p"] = float(wilcoxon(nonzero).pvalue)
    return out


def _signflip_wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p from all 2^n sign flips of *d* (no zeros).

    scipy's ``wilcoxon(method="auto")`` does exactly this for n <= 13 values with tied
    absolute values (statistic: sum of average ranks of the positive values; p = twice the
    smaller tail, relative tolerance 100 eps), but one Python call per flip takes seconds.
    """
    d = np.asarray(d, np.float64)
    ranks = rankdata(np.abs(d))
    observed = ranks[d > 0].sum()
    flips = (np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1      # which values are positive
    null = flips @ ranks
    gamma = abs(np.finfo(np.float64).eps * 100 * observed)
    low = np.count_nonzero(null <= observed + gamma) / len(null)
    high = np.count_nonzero(null >= observed - gamma) / len(null)
    return float(np.clip(2 * min(low, high), 0.0, 1.0))


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


_TRUE, _FALSE = {"true", "1", "1.0"}, {"false", "0", "0.0"}


def parse_bool(values, name="values"):
    """True/False, "True"/"False" (any case) or 1/0 -> bool array; anything else raises.

    ``astype(bool)`` is not safe on a CSV column: the string "False" and NaN
    both become True, so a filter on them would silently keep every row.

    Raises
    ------
    ValueError if a value is missing or not one of the forms above
    """
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr.copy()
    out = np.empty(arr.shape, dtype=bool)
    flat, bad = out.reshape(-1), []
    for i, v in enumerate(arr.reshape(-1)):
        key = v.strip().lower() if isinstance(v, str) else v
        if isinstance(key, str):
            ok = key in _TRUE or key in _FALSE
            flat[i] = key in _TRUE
        else:
            try:
                ok = key is not None and (key == 1 or key == 0)
            except (TypeError, ValueError):
                ok = False
            flat[i] = bool(ok and key == 1)
        if not ok:
            bad.append(v)
    if bad:
        raise ValueError(f"{name}: {len(bad)} value(s) are not True/False or 1/0, e.g. {bad[:3]!r}")
    return out


def finite_median(values):
    """Median of the finite values; NaN (without numpy's all-NaN warning) if there are none."""
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def format_p(p, floor=1e-10):
    """A p-value for a report line: "< 1e-10" below *floor*, else two significant digits.

    With 300 subjects that all move the same way, scipy's normal approximation
    returns about 6e-51; such values carry no information beyond "very small".
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < floor:
        return f"< {floor:.0e}"
    return f"{p:.2g}"


def cluster_bootstrap_counts(groups, n_boot=2000, seed=0):
    """The subject draws of a cluster bootstrap, as counts.

    The draws are those :func:`cluster_bootstrap_ci` makes with the same seed,
    so statistics computed from these counts reproduce it exactly, but for
    many statistics at once and much faster.

    Returns
    -------
    codes : int array (N,) — index of each row's subject (sorted subject ids)
    counts : int array (n_boot, n_subjects) — how often each subject is drawn
        in each resample
    """
    ids, codes = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ids), (n_boot, len(ids)))
    flat = (np.arange(n_boot)[:, None] * len(ids) + picks).ravel()
    counts = np.bincount(flat, minlength=n_boot * len(ids)).reshape(n_boot, len(ids))
    return codes.reshape(-1), counts


def _resampled_medians(values, codes, counts):
    """Median of *values* in every resample, where row i appears counts[b, codes[i]] times
    in resample b (exactly np.median of the repeated values; NaN for an empty resample)."""
    values = np.asarray(values, np.float64)
    out = np.full(len(counts), np.nan)
    if len(values) == 0:
        return out
    order = np.argsort(values, kind="stable")
    v, c = values[order], np.asarray(codes)[order]
    step = max(1, 2_000_000 // len(v))
    for s in range(0, len(counts), step):
        cum = np.cumsum(counts[s:s + step][:, c], axis=1)
        total = cum[:, -1]
        lo = np.argmax(cum >= ((total + 1) // 2)[:, None], axis=1)     # the two middle ranks
        hi = np.argmax(cum >= (total // 2 + 1)[:, None], axis=1)
        med = (v[lo] + v[hi]) / 2
        med[total == 0] = np.nan
        out[s:s + step] = med
    return out


def _bootstrap_p(boots):
    """Two-sided bootstrap p for 'the statistic is 0': twice the smaller tail, ties count half,
    with the +1 correction so that it is never exactly 0."""
    boots = boots[np.isfinite(boots)]
    if not len(boots):
        return float("nan")
    ties = 0.5 * (boots == 0).sum()
    low = ((boots < 0).sum() + ties + 1) / (len(boots) + 1)
    high = ((boots > 0).sum() + ties + 1) / (len(boots) + 1)
    return float(min(1.0, 2 * min(low, high)))


def cluster_bootstrap_median(values, groups, n_boot=2000, level=0.95, seed=0):
    """Median with a subject-cluster bootstrap CI: the same numbers as
    ``cluster_bootstrap_ci(values, groups, np.median, ...)``, vectorised.

    Returns
    -------
    (estimate, low, high) : floats; NaN if there are no finite values
    """
    values, groups = np.asarray(values, np.float64), np.asarray(groups)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    codes, counts = cluster_bootstrap_counts(groups, n_boot, seed)
    boots = _resampled_medians(values, codes, counts)
    alpha = (1.0 - level) / 2
    return float(np.median(values)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def paired_subject_ci(a, b, groups, n_boot=2000, level=0.95, seed=0):
    """Mean over subjects of the per-subject mean difference a - b, with a subject-cluster
    bootstrap CI: the CI for the per-subject differences that :func:`paired_subject_test`
    tests (pass ``b = 0`` for a one-sample comparison).

    Returns
    -------
    (estimate, low, high) : floats; NaN if no pair is finite
    """
    a, b, groups = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(groups)
    a, b = np.broadcast_to(a, groups.shape), np.broadcast_to(b, groups.shape)
    keep = np.isfinite(a) & np.isfinite(b)
    diff, groups = a[keep] - b[keep], groups[keep]
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    codes, counts = cluster_bootstrap_counts(groups, n_boot, seed)
    per_subject = np.bincount(codes, weights=diff) / np.bincount(codes)
    boots = counts @ per_subject / len(per_subject)
    alpha = (1.0 - level) / 2
    return (float(per_subject.mean()), float(np.quantile(boots, alpha)),
            float(np.quantile(boots, 1 - alpha)))


def cluster_permutation_test(values, labels, groups, n_perm=2000, seed=0):
    """Two-sided permutation test of median(values[labels]) - median(values[~labels]),
    exchanging subjects rather than images.

    Each permutation gives every subject the label pattern of another subject with
    the same number of rows (in random order within the subject). This keeps how
    labels cluster within subjects (a subject whose images are all restored, say),
    which an image-level shuffle would break, making the test too liberal.

    Returns
    -------
    dict with ``diff`` (observed), ``p`` ((1 + #|null| >= |diff|) / (1 + n_perm)),
    ``n_perm``; NaN if a side is empty
    """
    values, groups = np.asarray(values, np.float64), np.asarray(groups)
    labels = parse_bool(labels, "labels")
    keep = np.isfinite(values)
    values, labels, groups = values[keep], labels[keep], groups[keep]
    out = {"diff": float("nan"), "p": float("nan"), "n_perm": int(n_perm)}
    if labels.all() or not labels.any():
        return out
    observed = float(np.median(values[labels]) - np.median(values[~labels]))
    out["diff"] = observed
    _, codes = np.unique(groups, return_inverse=True)
    codes = codes.reshape(-1)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    rng = np.random.default_rng(seed)
    permuted = np.empty((n_perm, len(values)), dtype=bool)
    for k in np.unique(sizes):
        subjects = np.flatnonzero(sizes == k)
        rows = order[starts[subjects][:, None] + np.arange(k)]           # (n_k, k) row indices
        pattern = labels[rows]
        who = np.argsort(rng.random((n_perm, len(subjects))), axis=1)    # which subject's pattern
        within = np.argsort(rng.random((n_perm, len(subjects), k)), axis=2)
        permuted[:, rows] = pattern[who[:, :, None], within]
    rows = np.arange(len(values))
    null = (_resampled_medians(values, rows, permuted.astype(np.int64))
            - _resampled_medians(values, rows, (~permuted).astype(np.int64)))
    out["p"] = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + n_perm))
    return out


def stratified_difference(values, labels, strata, groups, n_boot=2000, level=0.95, seed=0, min_group=1):
    """Pooled within-stratum difference median(values[labels]) - median(values[~labels]).

    Each stratum (e.g. one attack x epsilon x defence x XAI condition) contributes
    its own difference, weighted by n1 * n0 / (n1 + n0) (the inverse variance of a
    difference between groups of n1 and n0 rows, so strata where almost every row is
    in one group count little). A pooled comparison that ignores strata would
    instead mix "which strata have many labelled rows" into the difference.
    Subjects are resampled jointly across strata, since the same subjects appear in
    every stratum; one stratum gives a plain difference of medians with its CI.
    Strata where either group has fewer than *min_group* rows are left out.

    Returns
    -------
    dict with ``diff``, ``low``, ``high``, ``p`` (two-sided bootstrap p), ``n_strata``
    (strata used), ``n`` (rows in those strata), ``weight`` (sum of weights)
    """
    values, strata, groups = (np.asarray(values, np.float64), np.asarray(strata), np.asarray(groups))
    labels = parse_bool(labels, "labels")
    keep = np.isfinite(values)
    values, labels, strata, groups = values[keep], labels[keep], strata[keep], groups[keep]
    out = {"diff": float("nan"), "low": float("nan"), "high": float("nan"), "p": float("nan"),
           "n_strata": 0, "n": 0, "weight": 0.0}
    if len(values) == 0:
        return out
    codes, counts = cluster_bootstrap_counts(groups, n_boot, seed)
    n_subjects = counts.shape[1]
    num = den = 0.0
    boot_num, boot_den = np.zeros(n_boot), np.zeros(n_boot)
    for s in np.unique(strata):
        one, zero = (strata == s) & labels, (strata == s) & ~labels
        n1, n0 = int(one.sum()), int(zero.sum())
        if min(n1, n0) < max(1, min_group):
            continue
        w = n1 * n0 / (n1 + n0)
        num += w * (np.median(values[one]) - np.median(values[zero]))
        den += w
        out["n_strata"] += 1
        out["n"] += n1 + n0
        k1 = counts @ np.bincount(codes[one], minlength=n_subjects)      # resampled group sizes
        k0 = counts @ np.bincount(codes[zero], minlength=n_subjects)
        both = (k1 > 0) & (k0 > 0)
        wb = np.where(both, k1 * k0 / np.maximum(k1 + k0, 1), 0.0)
        d = _resampled_medians(values[one], codes[one], counts) - _resampled_medians(values[zero], codes[zero], counts)
        boot_num += np.where(both, wb * np.nan_to_num(d), 0.0)
        boot_den += wb
    if den == 0:
        return out
    boots = np.where(boot_den > 0, boot_num / np.where(boot_den > 0, boot_den, 1.0), np.nan)
    alpha = (1.0 - level) / 2
    out.update({"diff": float(num / den), "low": float(np.nanquantile(boots, alpha)),
                "high": float(np.nanquantile(boots, 1 - alpha)), "p": _bootstrap_p(boots),
                "weight": float(den)})
    return out
