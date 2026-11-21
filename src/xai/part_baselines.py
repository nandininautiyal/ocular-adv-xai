"""
src/xai/part_baselines.py

Sampling baselines for Anatomical Shapley Attribution (src/xai/shapley.py), and
the projection of every attribution onto one 256 x 256 map for agreement.

Why this module exists
----------------------
ASA is exact: with P = 5 eye parts all 2^5 = 32 coalitions are evaluated. The
usual way to get Shapley-like part attributions is to sample coalitions and fit
a surrogate model instead (KernelSHAP: Doh et al. 2024 on face regions; LIME:
most XAI papers on periocular recognition, on superpixels or patches). This
module implements both estimators with numpy only (no shap, lime or skimage),
on two kinds of players:

- part players: the same anatomical groups, hiding methods and score as ASA, with
  a budget of half the 2^P coalitions of the image's present parts, at most
  ``--baseline-samples`` in scripts/22 (default 16 = half of the 32 when all five
  parts are present; :func:`part_budget`), so their error against exact ASA
  measures what sampling costs;
- pixel players: a square grid of patches (default 32 px, 8 x 8 = 64 players),
  the conventional pixel-level explanation, hidden with the same method and
  aggregated to parts afterwards (:func:`patch_part_shares`): the attribution of
  part p is sum over patches j of w_j x (share of patch j's pixels in part p).

Estimators
----------
:func:`kernelshap_design` / :func:`kernelshap_fit` (Lundberg & Lee 2017): weighted
linear regression of the coalition values on the coalition indicators, with the
Shapley kernel pi(S) = (P - 1) / (C(P, |S|) |S| (P - |S|)).
- Sampling follows shap's KernelExplainer: subset sizes are visited in
  complementary pairs (s, P - s) from the smallest; a size pair is enumerated
  completely while the remaining budget times its share of the kernel mass covers
  it, and the rest of the budget samples sizes in proportion to their kernel mass,
  each sampled coalition with its complement. Unlike shap, which samples with
  replacement and adds a duplicate's weight to the first draw, coalitions are drawn
  WITHOUT replacement: a duplicate is redrawn and costs no forward pass. Within a
  size that is only partly sampled, the size's kernel mass is shared equally by its
  sampled coalitions (shap shares the random part's mass by draw count; in
  expectation the two agree), and the random part is rescaled to the mass left
  after the enumerated sizes, as in shap. With a budget of all 2^P coalitions every
  coalition gets its exact kernel weight and the estimate equals the exact Shapley
  values (Lundberg & Lee 2017, Theorem 2; tests/test_part_baselines.py checks it).
- The budget counts every evaluated coalition, including the empty and the full
  one, which shap evaluates separately (f(x) and the base value).
- Efficiency is enforced as in shap (``KernelExplainer.solve``): the last player's
  value is eliminated, phi_P = (v(full) - v(empty)) - sum of the others, and the
  others come from weighted least squares on the transformed problem, so
  sum(phi) = v(full) - v(empty) exactly. No L1 feature selection (shap's
  ``l1_reg``): every player keeps a value.

:func:`lime_design` / :func:`lime_fit` (Ribeiro et al. 2016, lime_image defaults):
coalitions drawn uniformly (each player kept with probability 1/2, with
replacement), the first sample is the full image; weights
sqrt(exp(-d^2 / w^2)) with d the cosine distance between the coalition vector and
the all-ones vector and kernel width w = 0.25 (lime_image's default for its
default cosine distance); ridge regression with intercept, alpha = 1 (lime's
default ``Ridge(alpha=1, fit_intercept=True)``), on all players (lime_image's
default ``num_features`` selects all of them). The coefficients are the
attribution; they are not constrained to sum to v(full) - v(empty). Duplicate
coalitions stay in the regression (as in lime) but go through the model once.

Agreement maps
--------------
:func:`player_value_map` spreads positive player values uniformly over the
player's pixels (ASA, the part baselines, the receipt, patch weights);
:func:`upsample_map` resizes a native CAM or attention map (bilinear) and clips
it at 0. :func:`pairwise_js` gives the Jensen-Shannon divergence (base 2) of
every pair after each map is normalised to a distribution; a map with no positive
mass has no distribution (NaN).

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
- Coalition images: float32 (K, 256, 256, 3), range [0, 1] (src/xai/shapley.hide)

TensorFlow is imported only by the functions that run a model.
"""

import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.special import entr

from src.xai.shapley import DEFAULT_WORKERS, hide, make_cache, model_outputs, output_kind, score_values

ESTIMATORS = ("kernelshap", "lime")
LIME_KERNEL_WIDTH = 0.25      # lime_image's default for its default (cosine) distance
LIME_ALPHA = 1.0              # lime's default Ridge(alpha=1)
DEFAULT_PART_SAMPLES = 16     # cap of part_budget: half of the 2^5 = 32 coalitions of the default groups
DEFAULT_PIXEL_SAMPLES = 500
DEFAULT_PATCH = 32            # pixels: 8 x 8 = 64 patch players on a 256 x 256 image


def part_budget(n_present, cap=DEFAULT_PART_SAMPLES):
    """Coalitions of a part baseline on an image with *n_present* part players: half of their 2^n
    coalitions, at most *cap*, and at least 2 (the empty and the full one).

    A fixed budget of 16 is half of the 2^5 coalitions only when all five parts are present. Without
    a brow (most VISOB images) four parts play, 16 is all 2^4 of their coalitions, and KernelSHAP then
    returns the exact Shapley values (Lundberg & Lee 2017, Theorem 2): its error against ASA would be 0
    by construction, not a measurement. Half of the present players' coalitions keeps the sampling
    baselines at the same relative budget on every image. With one player (2 coalitions) every
    estimator is exact anyway.
    """
    n = int(n_present)
    if n < 1:
        raise ValueError(f"need at least one present part, got {n}")
    return max(2, min(int(cap), (1 << n) // 2))


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

def player_map_from_masks(masks):
    """Player id of every pixel from disjoint masks (P, H, W); -1 = in no player (always kept)."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (P, H, W), got {masks.shape}")
    if (masks.sum(axis=0) > 1).any():
        raise ValueError("masks overlap; players must be disjoint")
    out = np.full(masks.shape[1:], -1, np.int32)
    for p, m in enumerate(masks):
        out[m] = p
    return out


def patch_map(shape=(256, 256), patch=DEFAULT_PATCH):
    """Player id of every pixel for a square grid of *patch* x *patch* patches, row-major
    (patches at the right and bottom edge are smaller when *patch* does not divide the size)."""
    if int(patch) < 1:
        raise ValueError(f"patch must be at least 1 pixel, got {patch}")
    h, w = int(shape[0]), int(shape[1])
    cols = -(-w // int(patch))
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy // int(patch)) * cols + xx // int(patch)).astype(np.int32)


def n_players(player_map):
    """Number of players of a player map (largest id + 1)."""
    return int(np.max(player_map)) + 1 if np.size(player_map) else 0


def player_keep(player_map, kept):
    """Pixels kept when the players in *kept* (bool (n,)) are kept: their pixels and every pixel
    in no player."""
    kept = np.asarray(kept, dtype=bool)
    return np.append(kept, True)[np.asarray(player_map)]          # id -1 indexes the appended True


def patch_part_shares(patches, masks):
    """Share of each patch's pixels in each part: (P, n_patches), entry (p, j) = |patch j and part p| / |patch j|.

    Pixel-level attributions w (n_patches,) aggregate to parts as ``shares @ w``; the parts then
    hold sum(w) minus the share of the patches outside every part (unknown pixels), so the total
    is conserved when the parts cover every pixel.
    """
    patches = np.asarray(patches)
    masks = np.asarray(masks, dtype=bool)
    n = n_players(patches)
    size = np.bincount(patches.ravel(), minlength=n).astype(np.float64)
    out = np.stack([np.bincount(patches[m], minlength=n) for m in masks]).astype(np.float64)
    return out / np.maximum(size, 1.0)


def aggregate_to_parts(weights, shares):
    """Part attributions (P,) from patch attributions (n_patches,) and :func:`patch_part_shares`."""
    return np.asarray(shares, np.float64) @ np.asarray(weights, np.float64)


# ---------------------------------------------------------------------------
# KernelSHAP
# ---------------------------------------------------------------------------

def _bits(members, n):
    return sum(1 << int(p) for p in members)


def _row(bitmask, n):
    return np.array([(bitmask >> p) & 1 for p in range(n)], bool)


def kernel_weight(n, size):
    """Shapley kernel weight of one coalition of *size* players out of *n* (infinite for the empty
    and the full coalition, which the efficiency constraint handles)."""
    if size <= 0 or size >= n:
        return math.inf
    return (n - 1) / (math.comb(n, size) * size * (n - size))


def kernelshap_design(n, budget, seed=0):
    """Coalitions KernelSHAP evaluates besides the empty and the full one, with their weights.

    Parameters
    ----------
    n : number of players
    budget : coalitions evaluated in all, the empty and the full one included (at least 2)
    seed : seed (int or sequence) of numpy's default_rng for the sampled sizes

    Returns
    -------
    dict with ``coalitions`` (K, n) bool (each distinct, none empty or full), ``weights`` (K,)
    (regression weights, summing to 1 when K > 0), ``n_enumerated`` (coalitions from completely
    enumerated sizes) and ``n_evaluated`` = K + 2 (K + 1 for a single player, whose only other
    coalition is the empty one)
    """
    n, budget = int(n), int(budget)
    if n < 1:
        raise ValueError(f"need at least one player, got {n}")
    if budget < 2:
        raise ValueError(f"the budget counts the empty and the full coalition: at least 2, got {budget}")
    if n == 1:
        return {"coalitions": np.zeros((0, 1), bool), "weights": np.zeros(0), "n_enumerated": 0, "n_evaluated": 2}
    full = (1 << n) - 1
    slots = min(budget - 2, full - 1)
    if slots == full - 1:                          # everything: exact kernel weights
        masks = list(range(1, full))
        w = np.array([kernel_weight(n, bin(m).count("1")) for m in masks])
        return {"coalitions": np.stack([_row(m, n) for m in masks]), "weights": w / w.sum(),
                "n_enumerated": len(masks), "n_evaluated": len(masks) + 2}
    n_sizes, n_paired = n // 2, (n - 1) // 2      # sizes 1 .. ceil((n - 1) / 2), as in shap
    size_weight = np.array([(n - 1) / (s * (n - s)) for s in range(1, n_sizes + 1)])
    size_weight[:n_paired] *= 2
    size_weight /= size_weight.sum()

    chosen, weights, left = [], [], slots
    remaining, n_full = size_weight.copy(), 0
    for k in range(n_sizes):                       # complete sizes, smallest first
        s, paired = k + 1, k < n_paired
        count = math.comb(n, s) * (2 if paired else 1)
        if left * remaining[k] / count < 1.0 - 1e-8:
            break
        n_full += 1
        left -= count
        if remaining[k] < 1.0:
            remaining = remaining / (1.0 - remaining[k])
        w = size_weight[k] / math.comb(n, s) / (2 if paired else 1)
        for members in itertools.combinations(range(n), s):
            m = _bits(members, n)
            chosen.append(m)
            weights.append(w)
            if paired:
                chosen.append(full ^ m)
                weights.append(w)
    n_enumerated = len(chosen)

    if left > 0 and n_full < n_sizes:              # the rest: sizes by kernel mass, without replacement
        rng = np.random.default_rng(seed)
        draw_p = size_weight.copy()
        draw_p[:n_paired] /= 2                     # each paired draw also adds its complement
        draw_p = draw_p[n_full:]
        drawn, by_size = set(), {}
        while left > 0:
            if draw_p.sum() <= 0:
                break
            k = n_full + int(rng.choice(len(draw_p), p=draw_p / draw_p.sum()))
            s = k + 1
            if by_size.get(s, 0) >= math.comb(n, s):          # this size is exhausted
                draw_p[k - n_full] = 0.0
                continue
            m = _bits(rng.choice(n, s, replace=False), n)
            if m in drawn:
                continue
            for member in ([m, full ^ m] if k < n_paired else [m])[:left]:
                drawn.add(member)
                chosen.append(member)
                size = bin(member).count("1")
                by_size[size] = by_size.get(size, 0) + 1
                left -= 1
        random_w = []
        for m in chosen[n_enumerated:]:            # a size's kernel mass, shared by its sampled coalitions
            size = bin(m).count("1")
            k = min(size, n - size) - 1
            side = size_weight[k] / (2 if k < n_paired else 1)
            random_w.append(side / by_size[size])
        random_w = np.array(random_w)
        if random_w.size:
            weights.extend(random_w * (size_weight[n_full:].sum() / random_w.sum()))
    coalitions = np.stack([_row(m, n) for m in chosen]) if chosen else np.zeros((0, n), bool)
    weights = np.asarray(weights, np.float64)
    if weights.size:
        weights = weights / weights.sum()
    return {"coalitions": coalitions, "weights": weights, "n_enumerated": n_enumerated,
            "n_evaluated": len(chosen) + 2}


def kernelshap_fit(coalitions, weights, values, v_empty, v_full):
    """KernelSHAP values from evaluated coalitions, with efficiency enforced as in shap.

    Parameters
    ----------
    coalitions : (K, n) bool — :func:`kernelshap_design`
    weights : (K,) regression weights
    values : (K,) value of each coalition
    v_empty, v_full : value of the empty and the full coalition

    Returns
    -------
    phi : ndarray (n,), float64 — sum(phi) == v_full - v_empty
    """
    z = np.asarray(coalitions, np.float64)
    if z.ndim != 2:
        raise ValueError(f"coalitions must be (K, n), got shape {z.shape}")
    n = z.shape[1]
    delta = float(v_full) - float(v_empty)
    if n == 1:
        return np.array([delta])
    if not len(z):
        return np.full(n, delta / n)               # nothing sampled: the constraint alone
    y = np.asarray(values, np.float64) - float(v_empty)
    x = z[:, :-1] - z[:, -1:]                      # phi_last = delta - sum(phi_others)
    y = y - z[:, -1] * delta
    sw = np.sqrt(np.asarray(weights, np.float64))
    beta = np.linalg.lstsq(x * sw[:, np.newaxis], y * sw, rcond=None)[0]
    return np.append(beta, delta - beta.sum())


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def lime_design(n, n_samples, seed=0):
    """lime_image's samples: (n_samples, n) bool, each player kept with probability 1/2 (with
    replacement), the first row the full coalition."""
    n, n_samples = int(n), int(n_samples)
    if n < 1 or n_samples < 1:
        raise ValueError(f"need at least one player and one sample, got {n} and {n_samples}")
    z = np.random.default_rng(seed).integers(0, 2, (n_samples, n)).astype(bool)
    z[0] = True
    return z


def lime_weights(coalitions, kernel_width=LIME_KERNEL_WIDTH):
    """lime's kernel sqrt(exp(-d^2 / w^2)) of the cosine distance d between each coalition and the
    full one (d = 1 - sqrt(|S| / n); 1 for the empty coalition, as sklearn's cosine distance)."""
    z = np.asarray(coalitions, bool)
    d = 1.0 - np.sqrt(z.sum(axis=1) / z.shape[1])
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def lime_fit(coalitions, values, kernel_width=LIME_KERNEL_WIDTH, alpha=LIME_ALPHA):
    """Weighted ridge regression of the values on the coalition indicators (sklearn's Ridge with
    sample weights and an unpenalised intercept).

    Returns
    -------
    dict with ``coef`` (n,) — the attribution, ``intercept`` and ``weights`` (K,)
    """
    x = np.asarray(coalitions, np.float64)
    y = np.asarray(values, np.float64)
    w = lime_weights(coalitions, kernel_width)
    xm = (w @ x) / w.sum()
    ym = float(w @ y) / w.sum()
    xc, yc = x - xm, y - ym
    a = (xc * w[:, np.newaxis]).T @ xc + float(alpha) * np.eye(x.shape[1])
    coef = np.linalg.lstsq(a, (xc * w[:, np.newaxis]).T @ yc, rcond=None)[0]
    return {"coef": coef, "intercept": ym - float(xm @ coef), "weights": w}


# ---------------------------------------------------------------------------
# Estimators on a game table, and on a model
# ---------------------------------------------------------------------------

def design(estimator, n, budget, seed=0):
    """Sampled coalitions of an estimator: dict with ``coalitions`` (K, n) bool (KernelSHAP: without
    the empty and the full one) and ``weights`` (KernelSHAP only)."""
    if estimator == "kernelshap":
        return kernelshap_design(n, budget, seed)
    if estimator == "lime":
        return {"coalitions": lime_design(n, budget, seed), "weights": None}
    raise ValueError(f"unknown estimator {estimator!r}; choose from {ESTIMATORS}")


def evaluated(estimator, plan):
    """Every coalition an estimator's plan needs a value for, (K', n) bool: KernelSHAP adds the
    empty (first) and the full (second) coalition."""
    z = np.asarray(plan["coalitions"], bool)
    if estimator == "kernelshap":
        n = z.shape[1]
        return np.concatenate([np.zeros((1, n), bool), np.ones((1, n), bool), z])
    return z


def fit(estimator, plan, values):
    """Attribution (n,) of an estimator from the values of :func:`evaluated` (same order)."""
    values = np.asarray(values, np.float64)
    if estimator == "kernelshap":
        return kernelshap_fit(plan["coalitions"], plan["weights"], values[2:], values[0], values[1])
    return lime_fit(plan["coalitions"], values)["coef"]


def n_distinct(coalitions):
    """Distinct coalitions (= forward passes) among (K, n) bool rows."""
    z = np.asarray(coalitions, bool)
    return int(len(np.unique(z, axis=0))) if len(z) else 0


def to_indices(coalitions, players):
    """Coalition indices of the full game (bit p = group p kept) for coalitions over a subset of
    the groups (*players*: their group indices); the other groups are hidden."""
    z = np.asarray(coalitions, bool)
    weights = np.array([1 << int(p) for p in players], dtype=np.int64)
    return (z.astype(np.int64) * weights).sum(axis=1) if z.shape[1] else np.zeros(len(z), np.int64)


def from_game(estimator, values, budget, seed=0, present=None):
    """An estimator run on a game given as all 2^P coalition values (the values are looked up, not
    computed), over the *present* players only (absent players get 0, as in ASA).

    Returns
    -------
    phi : ndarray (P,), float64
    """
    v = np.asarray(values, np.float64)
    n = len(v).bit_length() - 1
    if len(v) != 1 << n:
        raise ValueError(f"values must hold 2^P coalition values, got {len(v)}")
    players = np.flatnonzero(np.ones(n, bool) if present is None else np.asarray(present, bool))
    phi = np.zeros(n)
    if not len(players):
        return phi
    plan = design(estimator, len(players), budget, seed)
    phi[players] = fit(estimator, plan, v[to_indices(evaluated(estimator, plan), players)])
    return phi


def sampled_outputs(model, image, player_map, coalitions, method, output="logits", batch_size=64, cache=None,
                    workers=DEFAULT_WORKERS):
    """Model outputs of coalition images given by player indicator rows.

    Each distinct row goes through the model once (built and evaluated in chunks of *batch_size*,
    so 500 pixel coalitions never sit in memory together); duplicates copy its output.

    Parameters
    ----------
    model : the dual-input [image, label] ArcFace model
    image : ndarray (256, 256, 3), float32, [0, 1]
    player_map : int (256, 256) — :func:`player_map_from_masks` or :func:`patch_map`
    coalitions : (K, n) bool — players kept in each coalition image
    method : hiding method (src/xai/shapley.hide)
    output : "logits" (margin-free) or "embedding" (L2-normalised)
    cache, workers : as in src/xai/shapley.coalition_images

    Returns
    -------
    outputs : ndarray (K, C or D), float32
    n_forward : int — distinct coalition images that went through the model
    """
    z = np.asarray(coalitions, bool)
    unique, inverse = np.unique(z, axis=0, return_inverse=True)
    cache = make_cache(image, [method] if method != "inpaint" else []) if cache is None else cache
    build = lambda row: hide(image, player_keep(player_map, row), method, cache)      # noqa: E731
    parts = []
    for start in range(0, len(unique), batch_size):
        chunk = unique[start:start + batch_size]
        if method == "inpaint" and workers > 1 and len(chunk) > 1:
            with ThreadPoolExecutor(min(workers, len(chunk))) as pool:
                images = np.stack(list(pool.map(build, chunk)))
        else:
            images = np.stack([build(row) for row in chunk])
        parts.append(model_outputs(model, images, output, batch_size))
    out = np.concatenate(parts)
    return out[np.asarray(inverse).ravel()], int(len(unique))


def estimate(model, image, player_map, estimator, budget, method, score, target=None, template=None, seed=0,
             present=None, batch_size=64, cache=None, workers=DEFAULT_WORKERS):
    """KernelSHAP or LIME attribution of one image on a model.

    Parameters
    ----------
    player_map : int (256, 256) — players (parts or patches), -1 = always kept
    estimator : "kernelshap" or "lime"
    budget : coalitions (KernelSHAP: in all, empty and full included; LIME: samples)
    method, score, target, template : as in src/xai/shapley.asa
    seed : seed of the sampled coalitions
    present : bool (n,) — players with pixels (default: all with at least one pixel); absent ones get 0

    Returns
    -------
    dict with ``phi`` (n,) and ``n_forward`` (distinct coalition images through the model)
    """
    n = n_players(player_map)
    if present is None:
        present = np.bincount(np.asarray(player_map)[np.asarray(player_map) >= 0].ravel(), minlength=n) > 0
    players = np.flatnonzero(present)
    phi = np.zeros(n)
    if not len(players):
        return {"phi": phi, "n_forward": 0}
    plan = design(estimator, len(players), budget, seed)
    rows = evaluated(estimator, plan)
    full = np.zeros((len(rows), n), bool)
    full[:, players] = rows
    outputs, n_forward = sampled_outputs(model, image, player_map, full, method, output_kind(score), batch_size,
                                         cache, workers)
    phi[players] = fit(estimator, plan, score_values(outputs, score, target, template))
    return {"phi": phi, "n_forward": n_forward}


# ---------------------------------------------------------------------------
# Agreement: every attribution as one 256 x 256 distribution
# ---------------------------------------------------------------------------

def player_value_map(values, player_map):
    """Non-negative map (H, W): each player's positive value spread uniformly over its pixels
    (value / pixel count); negative and NaN values and pixels in no player get 0."""
    pm = np.asarray(player_map)
    n = max(n_players(pm), len(values))
    v = np.nan_to_num(np.asarray(values, np.float64), nan=0.0)
    v = np.maximum(np.pad(v, (0, n - len(v))), 0.0)
    count = np.bincount(pm[pm >= 0].ravel(), minlength=n).astype(np.float64)
    density = np.where(count > 0, v / np.maximum(count, 1.0), 0.0)
    return np.append(density, 0.0)[pm]


def upsample_map(native, shape=(256, 256)):
    """A native heatmap (h, w) resized to *shape* (bilinear) and clipped at 0."""
    native = np.nan_to_num(np.asarray(native, np.float32), nan=0.0)
    out = cv2.resize(native, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    return np.maximum(out.astype(np.float64), 0.0)


def _distribution(heatmap):
    m = np.asarray(heatmap, np.float64).ravel()
    total = m.sum() if np.isfinite(m).all() else np.nan
    if not np.isfinite(total) or total <= 0 or (m < 0).any():
        return None
    return m / total


def _entropy2(p):
    """Entropy in bits (0 log 0 = 0)."""
    return float(entr(p).sum() / math.log(2.0))


def js_divergence(map_a, map_b):
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two non-negative maps as distributions;
    NaN if either has no positive mass."""
    return pairwise_js({"a": map_a, "b": map_b})[("a", "b")]


def pairwise_js(maps):
    """Jensen-Shannon divergence (base 2) of every pair of maps.

    Parameters
    ----------
    maps : dict name -> non-negative map (H, W) of one shape, or None (missing)

    Returns
    -------
    dict (name_a, name_b) -> float, for every pair in the dict's order; NaN where a map is
    missing or has no positive mass
    """
    dist = {k: (None if m is None else _distribution(m)) for k, m in maps.items()}
    entropy = {k: (None if p is None else _entropy2(p)) for k, p in dist.items()}
    out = {}
    for a, b in itertools.combinations(maps, 2):
        p, q = dist[a], dist[b]
        if p is None or q is None:
            out[(a, b)] = float("nan")
            continue
        js = _entropy2(0.5 * (p + q)) - 0.5 * (entropy[a] + entropy[b])
        out[(a, b)] = float(min(max(js, 0.0), 1.0))
    return out
