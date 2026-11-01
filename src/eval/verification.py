"""
src/eval/verification.py

Biometric verification and identification metrics on the ArcFace embedding.

Why this module exists
----------------------
Closed-set accuracy over the 300 training classes (04_eval_clean.py) is not a
verification result. Biometrics reviewers expect comparison scores between two
samples, ROC / DET curves, the equal error rate (EER), the true accept rate at a
fixed false accept rate (TAR@FAR, also called GMR@FMR) and rank-1 / CMC
identification, with confidence intervals, as ISO/IEC 19795-1 describes. FAR
and FRR here are the algorithm-level false match and false non-match rates:
every image is compared and nothing fails to be acquired.

Scores
------
embed() returns the input of the ArcFace head (the Dense(1024) output; dropout
is off at inference), L2-normalised, so the score of two samples is their
cosine similarity, and a score >= threshold is an accept. Samples are compared
only within a group (the evaluation script uses the eye: left with left, right
with right), in one of two ways:
- all pairs within one set (pair_counts), e.g. the pinned test set;
- enrolment templates against probes (build_templates, template_scores): a
  template is the mean of one subject's enrolment embeddings of one eye,
  renormalised.

Large score sets
----------------
All same-eye pairs of the 13,868 pinned test images are about 10^8 scores, too
many to keep and resample. Scores are therefore counted per subject on a fixed
grid (bin width 1e-4 on [-1, 1]), genuine and impostor separately. Metrics from
the counts are the exact metrics of the scores rounded down to the grid, and a
subject bootstrap only has to add rows. roc, eer, tar_at_far and error_rates
take raw scores (small sets, tests); the *_from_counts versions take counts.

Confidence intervals
--------------------
Percentile CIs from a cluster bootstrap over probe subjects (all comparisons of
a resampled subject's probes come along), with subjects drawn exactly as
src/analysis/stats.py draws them; rank-1 uses its cluster_bootstrap_ci directly.
"""

import numpy as np

from src.analysis.stats import cluster_bootstrap_ci

BINS_PER_UNIT = 10000                                  # score grid step 1e-4
N_BINS = 2 * BINS_PER_UNIT                             # covers cosine scores in [-1, 1]
THRESHOLDS = np.arange(N_BINS) / BINS_PER_UNIT - 1.0   # lower edge of every bin
FARS = (1e-2, 1e-3)


def far_key(far):
    """Column-friendly label of a FAR target: 0.01 -> '1e-2'."""
    mantissa, exponent = f"{far:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# ------------------------------------------------------------------ embeddings

def l2_normalise(x):
    """Rows scaled to unit length (all-zero rows stay zero), like tf.nn.l2_normalize."""
    x = np.asarray(x, np.float32)
    return x / np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))


def embed(model, images, batch_size=64):
    """L2-normalised embeddings (N, D) at the input of the ArcFace head.

    For the Xception checkpoint this is the Dense(1024) output (D = 1024), the
    vector whose cosine with each class weight gives the margin-free logits
    (src/models/arcface_inference.py). Dropout is off at inference.

    Parameters
    ----------
    model : the dual-input [image, label] ArcFace model
    images : float32 (N, 256, 256, 3) or (256, 256, 3) in [0, 1]
    """
    from src.models.arcface_inference import get_margin_free   # imports TensorFlow; the metrics do not
    x = np.asarray(images, np.float32)
    if x.ndim == 3:
        x = x[np.newaxis]
    sub_model = get_margin_free(model)._sub_model   # image -> ArcFace input, built once per model
    parts = [np.asarray(sub_model(x[i:i + batch_size], training=False)) for i in range(0, len(x), batch_size)]
    if not parts:
        return np.zeros((0, int(sub_model.output.shape[-1])), np.float32)
    return l2_normalise(np.concatenate(parts))


def cosine_scores(a, b):
    """Score of every row of *a* against every row of *b* (both L2-normalised): (len(a), len(b))."""
    return np.asarray(a, np.float32) @ np.asarray(b, np.float32).T


def _groups(groups, n):
    return np.zeros(n, np.int64) if groups is None else np.asarray(groups)


# --------------------------------------------------------------- comparisons

def score_bins(scores):
    """Grid bin of each score: floor((score + 1) * 10^4), clipped to [0, N_BINS - 1]."""
    k = np.floor((np.asarray(scores, np.float64) + 1.0) * BINS_PER_UNIT)
    return np.clip(k, 0, N_BINS - 1).astype(np.int64)


def pair_counts(embeddings, subjects, groups=None, max_block=1 << 24):
    """Genuine and impostor score counts of all pairs within one set, per subject.

    Every ordered pair (i, j), i != j, of the same group is counted for the
    subject of i, so each unordered pair is counted twice in total. That leaves
    every rate unchanged and makes each subject's row a cluster for the bootstrap.

    Parameters
    ----------
    embeddings : (N, D) L2-normalised
    subjects : (N,) identity of each image; same subject = genuine pair
    groups : (N,) optional; only pairs within a group are compared (e.g. the eye)
    max_block : scores computed at once (memory bound)

    Returns
    -------
    dict with
      ``subjects``             sorted unique subjects (the rows)
      ``genuine``, ``impostor`` int64 counts (n_subjects, N_BINS)
      ``nn_correct``           per image: 1.0 if its most similar other image of the
                               group is of the same subject, 0.0 if not (ties count as
                               wrong), NaN if it has no same-subject partner; i.e.
                               leave-one-out rank-1 identification
    """
    emb = np.asarray(embeddings, np.float32)
    groups = _groups(groups, len(emb))
    ids, sidx = np.unique(np.asarray(subjects), return_inverse=True)
    size = len(ids) * N_BINS
    genuine, impostor = np.zeros(size, np.int64), np.zeros(size, np.int64)
    nn_correct = np.full(len(emb), np.nan)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        e, s = emb[members], sidx[members]
        step = max(1, max_block // len(members))
        for start in range(0, len(members), step):
            rows = np.arange(start, min(start + step, len(members)))
            scores = e[rows] @ e.T
            same = s[rows, None] == s[None, :]
            partner = same.copy()
            partner[np.arange(len(rows)), rows] = False          # an image is never its own partner
            key = s[rows, None] * N_BINS + score_bins(scores)
            genuine += np.bincount(key[partner], minlength=size)
            impostor += np.bincount(key[~same], minlength=size)
            best_genuine = np.where(partner, scores, -np.inf).max(axis=1)
            best_impostor = np.where(same, -np.inf, scores).max(axis=1)
            nn_correct[members[rows]] = np.where(np.isfinite(best_genuine),
                                                 (best_genuine > best_impostor).astype(np.float64), np.nan)
    return {"subjects": ids, "genuine": genuine.reshape(len(ids), N_BINS),
            "impostor": impostor.reshape(len(ids), N_BINS), "nn_correct": nn_correct}


def build_templates(embeddings, subjects, groups=None):
    """One template per (subject, group): the mean of its embeddings, renormalised.

    Returns
    -------
    (templates (T, D) float32, subject of each template (T,), group of each template (T,)),
    sorted by subject, then group
    """
    emb = np.asarray(embeddings, np.float32)
    subjects, groups = np.asarray(subjects), _groups(groups, len(emb))
    if len(emb) == 0:
        return np.zeros((0, emb.shape[-1]), np.float32), subjects[:0], groups[:0]
    s_ids, s_idx = np.unique(subjects, return_inverse=True)
    g_ids, g_idx = np.unique(groups, return_inverse=True)
    codes, inverse = np.unique(s_idx * len(g_ids) + g_idx, return_inverse=True)
    sums = np.zeros((len(codes), emb.shape[1]), np.float64)
    np.add.at(sums, inverse, emb)                 # the 1/n of the mean cancels in the renormalisation
    return l2_normalise(sums), s_ids[codes // len(g_ids)], g_ids[codes % len(g_ids)]


def template_scores(templates, template_subjects, probes, probe_subjects, template_groups=None,
                    probe_groups=None):
    """Score every probe against every template of its group.

    Returns
    -------
    scores : (n_probes, n_templates) float32, NaN for templates of another group
    own : (n_probes, n_templates) bool, True at the template of the probe's subject and group
    """
    same_group = (_groups(probe_groups, len(probes))[:, None]
                  == _groups(template_groups, len(templates))[None, :])
    scores = np.where(same_group, cosine_scores(probes, templates), np.nan).astype(np.float32)
    own = same_group & (np.asarray(probe_subjects)[:, None] == np.asarray(template_subjects)[None, :])
    return scores, own


def template_counts(scores, own, subjects, max_block=1 << 24):
    """Genuine and impostor score counts of a probe x template matrix, per probe subject.

    Parameters
    ----------
    scores, own : from template_scores (NaN scores are skipped)
    subjects : (n_probes,) subject of each probe (the bootstrap clusters)

    Returns
    -------
    dict with ``subjects`` (sorted unique, the rows) and int64 counts ``genuine``,
    ``impostor`` (n_subjects, N_BINS)
    """
    scores, own = np.asarray(scores), np.asarray(own, bool)
    ids, sidx = np.unique(np.asarray(subjects), return_inverse=True)
    size = len(ids) * N_BINS
    genuine, impostor = np.zeros(size, np.int64), np.zeros(size, np.int64)
    step = max(1, max_block // max(scores.shape[1], 1))
    for start in range(0, len(scores), step):
        block, is_own = scores[start:start + step], own[start:start + step]
        valid = ~np.isnan(block)
        key = sidx[start:start + step, None] * N_BINS + score_bins(np.where(valid, block, 0.0))
        genuine += np.bincount(key[valid & is_own], minlength=size)
        impostor += np.bincount(key[valid & ~is_own], minlength=size)
    return {"subjects": ids, "genuine": genuine.reshape(len(ids), N_BINS),
            "impostor": impostor.reshape(len(ids), N_BINS)}


def ranks(scores, own):
    """Rank of each probe's own template among the templates of its group (1 = best).

    Ties with the own score count against it; 0 where the probe has no own template.
    """
    scores = np.where(np.isnan(scores), -np.inf, scores)
    own_score = np.where(own, scores, -np.inf).max(axis=1)
    above = (scores >= own_score[:, None]) & ~own
    return np.where(own.any(axis=1), above.sum(axis=1) + 1, 0)


def cmc(ranks, max_rank=20):
    """Cumulative match characteristic: share of probes ranked k or better, k = 1..max_rank.

    Probes without an own template (rank 0) are left out.
    """
    r = np.asarray(ranks)
    r = r[r > 0]
    if len(r) == 0:
        return np.full(max_rank, np.nan)
    return np.array([float((r <= k).mean()) for k in range(1, max_rank + 1)])


def rank1(ranks):
    """Rank-1 identification rate: share of probes whose own template scores highest."""
    return float(cmc(ranks, 1)[0])


# ------------------------------------------------------------------- metrics

def _roc(genuine_counts, impostor_counts, thresholds):
    """ROC of scores grouped by value (*thresholds* ascending, with the counts at each value)."""
    g = np.asarray(genuine_counts, np.float64)[::-1]
    i = np.asarray(impostor_counts, np.float64)[::-1]
    t = np.asarray(thresholds, np.float64)[::-1]
    keep = (g + i) > 0
    g, i, t = g[keep], i[keep], t[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        tar = np.r_[0.0, np.cumsum(g) / g.sum()]
        far = np.r_[0.0, np.cumsum(i) / i.sum()]
    return far, tar, np.r_[np.inf, t]


def roc(genuine, impostor):
    """ROC of raw scores: arrays (far, tar, thresholds).

    The points run from the strictest threshold (+inf: nothing accepted, far = tar
    = 0) down to the lowest score (everything accepted, far = tar = 1); far and tar
    are the shares of impostor and genuine scores >= threshold.
    """
    gen = np.asarray(genuine, np.float64).ravel()
    imp = np.asarray(impostor, np.float64).ravel()
    gen, imp = gen[np.isfinite(gen)], imp[np.isfinite(imp)]
    values, inverse = np.unique(np.r_[gen, imp], return_inverse=True)
    g = np.bincount(inverse[:len(gen)], minlength=len(values))
    i = np.bincount(inverse[len(gen):], minlength=len(values))
    return _roc(g, i, values)


def roc_from_counts(genuine_counts, impostor_counts):
    """ROC of grid counts (length N_BINS each, e.g. the summed rows of pair_counts)."""
    return _roc(genuine_counts, impostor_counts, THRESHOLDS)


def eer_from_roc(far, tar, thresholds):
    """Equal error rate (FAR = FRR), interpolated linearly between the two ROC points around it.

    Returns (eer, threshold), where threshold is that of the first point with FRR <= FAR.
    """
    if len(far) < 2 or np.isnan(far[-1]) or np.isnan(tar[-1]):
        return float("nan"), float("nan")
    diff = (1.0 - tar) - far        # FRR - FAR: 1 at threshold +inf, -1 at the end, never increases
    k = int(np.argmax(diff <= 0))
    if diff[k] == 0:
        return float(far[k]), float(thresholds[k])
    w = diff[k - 1] / (diff[k - 1] - diff[k])
    return float(far[k - 1] + w * (far[k] - far[k - 1])), float(thresholds[k])


def tar_at_far_from_roc(far, tar, thresholds, target):
    """Highest TAR whose FAR does not exceed *target* (an operating point, no interpolation).

    Returns (tar, threshold, far achieved).
    """
    if len(far) < 2 or np.isnan(far[-1]) or np.isnan(tar[-1]):
        return float("nan"), float("nan"), float("nan")
    k = int(np.searchsorted(far, target * (1 + 1e-9), side="right")) - 1
    return float(tar[k]), float(thresholds[k]), float(far[k])


def eer(genuine, impostor):
    """Equal error rate of raw scores, interpolated: (eer, threshold)."""
    return eer_from_roc(*roc(genuine, impostor))


def tar_at_far(genuine, impostor, far):
    """TAR at the threshold where FAR first stays within *far*: (tar, threshold, far achieved)."""
    return tar_at_far_from_roc(*roc(genuine, impostor), far)


def error_rates(genuine, impostor, threshold):
    """(FRR, FAR) at *threshold*: share of genuine scores below it, of impostor scores at or above it.

    NaN for a NaN threshold (e.g. a TAR@FAR threshold that could not be set).
    """
    if np.isnan(threshold):
        return float("nan"), float("nan")
    gen, imp = np.asarray(genuine, np.float64), np.asarray(impostor, np.float64)
    frr = float((gen < threshold).mean()) if len(gen) else float("nan")
    far = float((imp >= threshold).mean()) if len(imp) else float("nan")
    return frr, far


def error_rates_from_counts(genuine_counts, impostor_counts, threshold):
    """(FRR, FAR) of grid counts at *threshold*; a bin is accepted when its lower edge is >= threshold.

    NaN for a NaN threshold: without it every bin would be rejected (FRR 1, FAR 0).
    """
    if np.isnan(threshold):
        return float("nan"), float("nan")
    accept = THRESHOLDS >= threshold - 1e-12
    g, i = np.asarray(genuine_counts, np.float64), np.asarray(impostor_counts, np.float64)
    frr = float(g[~accept].sum() / g.sum()) if g.sum() else float("nan")
    far = float(i[accept].sum() / i.sum()) if i.sum() else float("nan")
    return frr, far


def metrics_from_counts(genuine_counts, impostor_counts, fars=FARS):
    """EER and TAR@FAR of grid counts (length N_BINS each).

    Returns
    -------
    dict with n_genuine, n_impostor, eer, eer_threshold and, per FAR target
    (key = far_key(far), e.g. '1e-2'): tar_at_far_<key>, thr_at_far_<key>,
    far_achieved_<key>
    """
    far, tar, thr = roc_from_counts(genuine_counts, impostor_counts)
    out = {"n_genuine": int(np.sum(genuine_counts)), "n_impostor": int(np.sum(impostor_counts))}
    out["eer"], out["eer_threshold"] = eer_from_roc(far, tar, thr)
    for target in fars:
        key = far_key(target)
        out[f"tar_at_far_{key}"], out[f"thr_at_far_{key}"], out[f"far_achieved_{key}"] = \
            tar_at_far_from_roc(far, tar, thr, target)
    return out


# ------------------------------------------------------- confidence intervals

def bootstrap_picks(n_groups, n_boot, seed=0):
    """Resampled group indices, drawn as src/analysis/stats.cluster_bootstrap_ci draws them."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_groups, n_groups) for _ in range(n_boot)]


def _interval(values, level):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha))


def bootstrap_metrics(genuine_counts, impostor_counts, fars=FARS, n_boot=1000, level=0.95, seed=0):
    """Percentile CIs of the EER and TAR@FAR, resampling whole rows (subjects) of the count matrices.

    Parameters
    ----------
    genuine_counts, impostor_counts : (n_subjects, N_BINS), e.g. from pair_counts or template_counts

    Returns
    -------
    dict ``eer`` / ``tar_at_far_<key>`` -> (low, high); NaN without rows or with n_boot = 0
    """
    names = ["eer"] + [f"tar_at_far_{far_key(f)}" for f in fars]
    gen, imp = np.asarray(genuine_counts), np.asarray(impostor_counts)
    if len(gen) == 0 or n_boot <= 0:
        return {name: (float("nan"), float("nan")) for name in names}
    used = np.flatnonzero(gen.sum(axis=0) + imp.sum(axis=0))      # empty bins never move a ROC point
    gen, imp, thr = gen[:, used].astype(np.float64), imp[:, used].astype(np.float64), THRESHOLDS[used]
    picks = bootstrap_picks(len(gen), n_boot, seed)
    boots = {name: [] for name in names}
    chunk = max(1, (1 << 22) // max(len(used), 1))                  # resamples per matrix product
    for start in range(0, n_boot, chunk):
        weights = np.stack([np.bincount(p, minlength=len(gen)) for p in picks[start:start + chunk]])
        weights = weights.astype(np.float64)
        for g, i in zip(weights @ gen, weights @ imp):
            far, tar, t = _roc(g, i, thr)
            boots["eer"].append(eer_from_roc(far, tar, t)[0])
            for target in fars:
                boots[f"tar_at_far_{far_key(target)}"].append(tar_at_far_from_roc(far, tar, t, target)[0])
    return {name: _interval(values, level) for name, values in boots.items()}


def rank1_ci(correct, subjects, n_boot=1000, level=0.95, seed=0):
    """Rank-1 rate with a subject-cluster bootstrap CI: (estimate, low, high).

    correct : per probe 1.0 / 0.0; NaN (no own template or partner) is left out
    """
    if n_boot <= 0:
        values = np.asarray(correct, np.float64)
        values = values[np.isfinite(values)]
        return (float(values.mean()) if len(values) else float("nan")), float("nan"), float("nan")
    return cluster_bootstrap_ci(correct, subjects, stat=np.mean, n_boot=n_boot, level=level, seed=seed)
