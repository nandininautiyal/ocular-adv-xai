"""
src/xai/shapley.py

Anatomical Shapley Attribution (ASA): exact Shapley values of a recognition
score over a handful of eye parts.

Why this module exists
----------------------
LayerCAM and Eigen-CAM (src/xai/attribution.py) explain an 8 x 8 feature map,
and every cell of that map sees most of the 256 x 256 image, so "the map's mass
on the iris" is a statement about a grid cell, not about the iris. ASA asks the
model directly: with the eye-mask label maps of scripts/17 (masks_<tag>.npz)
the image is split into P anatomical parts (default: iris, sclera, lid, brow,
skin), every one of the 2^P subsets of parts is kept while the others are
hidden, and the score of each of these coalition images is the value of a
cooperative game. With P = 5 that is 32 forward passes, so the Shapley values
are exact: no sampling, no surrogate model (Doh et al. 2024 estimate them over
face regions with KernelSHAP instead).

Game
----
- Players: the part groups, ``groups`` = ordered mapping name -> mask classes,
  e.g. {"iris": ["iris"], "sclera": ["sclera", "caruncle"], ...}
  (:data:`DEFAULT_GROUPS`, as in config ``anatomy.part_groups``). Unknown pixels
  (label 255) and classes in no group belong to no player and are always kept.
- Coalition index: a bitmask of the KEPT groups, bit p = group p kept, so index
  0 hides every group and index 2^P - 1 is the unchanged image.
- Hiding (:func:`hide`): ``blur`` (Gaussian, sigma 8 px), ``mean`` (the image's
  mean colour) or ``inpaint`` (OpenCV Telea, radius 5, on the 8-bit image).
  Kept pixels are copied bit for bit; only hidden pixels change. The choice of
  baseline changes the game, so results are reported for all three.
- Value v(S): a score of the model on coalition image S (:func:`score_values`):
  ``logit`` (margin-free logit s * cos of a target class), ``margin`` (target
  logit minus the best other logit; negative = the target is not predicted) or
  ``verif`` (cosine between the image's embedding and a template embedding).

Outputs
-------
- :func:`exact_shapley`: phi_p = sum over S without p of
  |S|! (P - |S| - 1)! / P! * (v(S + p) - v(S)). Efficiency holds by
  construction: sum(phi) = v(full) - v(empty).
- :func:`shapley_interactions`: the Shapley interaction index of every pair
  (Grabisch 1997), I_pq = sum over S without p, q of
  |S|! (P - |S| - 2)! / (P - 1)! * (v(S + p + q) - v(S + p) - v(S + q) + v(S)),
  with phi on the diagonal (the order-1 index is the Shapley value). It is 0 for
  every pair of an additive game.
- A group with no pixel in an image is a null player. :func:`asa` evaluates
  only the coalitions that differ in a present group and copies the value to
  the rest, so an absent group gets phi = 0 and zero interactions exactly
  (and the forward passes for it are saved).

Faithfulness helpers compare other explanations with ASA on the same parts:
:func:`part_masses` (a heatmap's mass per group, with the area-averaging of
src/compare/regions.py), :func:`rank_agreement` (Spearman over present groups)
and :func:`deletion_curve` / :func:`deletion_from_values` (the score as groups
are hidden in a given order, and its normalised area). A deletion curve only
needs coalition values, so the curves of every order come free with the ASA
values; note that ranking groups by ASA (:func:`asa_order`) and deleting them
with the same baseline favours ASA (it is computed on the same game). Heatmaps
have no sign, so their orders delete the most salient part first whatever the
direction of the score change.

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
- Output : float32, (256, 256, 3) per image or (K, 256, 256, 3), range [0, 1]

TensorFlow is imported only by the functions that run a model, so the game
arithmetic and the faithfulness helpers work without it (scripts/23).
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.stats import spearmanr

from src.compare.regions import IGNORE_LABEL, masks_to_grids, region_mass_named

DEFAULT_GROUPS = {"iris": ("iris",), "sclera": ("sclera", "caruncle"), "lid": ("lid",),
                  "brow": ("brow",), "skin": ("skin",)}
METHODS = ("blur", "mean", "inpaint")
SCORES = ("logit", "margin", "verif")
BLUR_SIGMA = 8.0            # pixels
INPAINT_RADIUS = 5          # pixels
MAX_GROUPS = 10             # 2^10 coalition images per image and method is already a lot
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)     # threads for inpainting (OpenCV releases the GIL)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def normalise_groups(groups):
    """Ordered ((name, (class, ...)), ...) from a mapping or a sequence of pairs.

    Raises ValueError for an empty or duplicate name, a group without classes, a class
    in two groups, or more than :data:`MAX_GROUPS` groups.
    """
    items = list(groups.items()) if hasattr(groups, "items") else list(groups)
    out, seen = [], {}
    for item in items:
        try:
            name, classes = item
        except (TypeError, ValueError):
            raise ValueError(f"groups must map a name to its mask classes, got {item!r}") from None
        name = str(name).strip()
        classes = (classes,) if isinstance(classes, str) else tuple(classes)
        classes = tuple(str(c).strip() for c in classes)
        if not name or name in dict(out):
            raise ValueError(f"group names must be distinct and non-empty, got {name!r}")
        if not classes or not all(classes):
            raise ValueError(f"group {name!r} has no mask classes")
        for c in classes:
            if c in seen:
                raise ValueError(f"class {c!r} is in groups {seen[c]!r} and {name!r}; groups must not overlap")
            seen[c] = name
        out.append((name, classes))
    if not out:
        raise ValueError("need at least one group")
    if len(out) > MAX_GROUPS:
        raise ValueError(f"{len(out)} groups give 2^{len(out)} coalitions; use at most {MAX_GROUPS}")
    return tuple(out)


def group_names(groups):
    """Names of the groups, in player order."""
    return tuple(name for name, _ in normalise_groups(groups))


def parse_groups(spec):
    """Groups from a command-line spec such as ``iris=iris;sclera=sclera+caruncle;skin=skin``."""
    pairs = []
    for part in str(spec).split(";"):
        if not part.strip():
            continue
        name, sep, classes = part.partition("=")
        if not sep:
            raise ValueError(f"expected name=class+class in group spec {spec!r}, got {part!r}")
        pairs.append((name, [c for c in classes.split("+") if c.strip()]))
    return normalise_groups(pairs)


def format_groups(groups):
    """Inverse of :func:`parse_groups`."""
    return ";".join(f"{name}={'+'.join(classes)}" for name, classes in normalise_groups(groups))


def groups_from_config(cfg):
    """``anatomy.part_groups`` of a loaded config (a mapping, or a list of one-key mappings)."""
    part_groups = (cfg or {}).get("anatomy", {}).get("part_groups")
    if part_groups is None:
        raise ValueError("the config has no anatomy.part_groups")
    if isinstance(part_groups, list):
        part_groups = [pair for entry in part_groups for pair in dict(entry).items()]
    return normalise_groups(part_groups)


def group_masks(labels, class_names, groups):
    """Pixel masks of the part groups of one label map.

    Parameters
    ----------
    labels : ndarray (H, W) of integer label ids (index into *class_names*), 255 = unknown
    class_names : sequence of str, e.g. ("skin", "iris", "sclera", "lid", "caruncle", "brow")
    groups : mapping name -> classes, or sequence of pairs (:func:`normalise_groups`)

    Returns
    -------
    masks : ndarray (P, H, W), bool — disjoint; unknown pixels and classes outside every
        group are in no mask (they are always kept)
    """
    groups = normalise_groups(groups)
    class_names = [str(c) for c in class_names]
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be one integer label map (H, W), got {labels.dtype} {labels.shape}")
    valid = ((labels >= 0) & (labels < len(class_names))) | (labels == IGNORE_LABEL)
    if not valid.all():
        raise ValueError(f"labels must be in 0..{len(class_names) - 1} or {IGNORE_LABEL}, "
                         f"found {np.unique(labels[~valid])[:5]}")
    masks = np.zeros((len(groups),) + labels.shape, bool)
    for p, (name, classes) in enumerate(groups):
        for c in classes:
            if c not in class_names:
                raise ValueError(f"group {name!r}: class {c!r} is not in class_names {class_names}")
            masks[p] |= labels == class_names.index(c)
    return masks


# ---------------------------------------------------------------------------
# Hiding
# ---------------------------------------------------------------------------

def _check_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected one image (H, W, 3), got shape {image.shape}")
    return image


def make_cache(image, methods=METHODS):
    """Fills that do not depend on what is hidden, computed once per image: the blurred
    image (``blur``) and the mean colour (``mean``). Pass the dict as ``cache`` to
    :func:`hide` and :func:`coalition_images`; ``inpaint`` depends on the hidden pixels
    and is never cached."""
    image = _check_image(image)
    cache = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown hiding method {method!r}; choose from {METHODS}")
        if method == "blur":
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA,
                                       borderType=cv2.BORDER_REFLECT_101)
            cache["blur"] = np.clip(blurred, 0.0, 1.0).astype(np.float32)
        elif method == "mean":
            cache["mean"] = np.broadcast_to(image.reshape(-1, 3).mean(axis=0, dtype=np.float64)
                                            .astype(np.float32), image.shape)
    return cache


def _inpaint_fill(image, keep):
    """Telea inpainting of the hidden pixels on the 8-bit image, back on the [0, 1] scale.
    With no pixel kept there is nothing to fill from: the mean colour is used instead. This is
    not an edge case: when the groups cover every pixel (17's masks map the background to skin,
    and the default groups cover all six classes) it happens for the empty coalition of every
    image, so the inpaint game's v(empty) is the mean game's."""
    if not keep.any():
        return make_cache(image, ("mean",))["mean"]
    uint8 = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    hidden = (~keep).astype(np.uint8) * 255
    return cv2.inpaint(uint8, hidden, INPAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float32) / 255.0


def hide(image, keep, method, cache=None):
    """The image with every pixel outside *keep* replaced by a baseline.

    Parameters
    ----------
    image : ndarray (H, W, 3), float32, [0, 1]
    keep : ndarray (H, W), bool — pixels copied unchanged (bit for bit)
    method : "blur" (Gaussian blur of the image, sigma 8 px), "mean" (the image's mean
        colour) or "inpaint" (cv2.inpaint, Telea, radius 5, on the 8-bit image)
    cache : dict from :func:`make_cache` for this image, or None; missing fills are
        added to it (a memo: the result does not depend on it)

    Returns
    -------
    hidden : ndarray (H, W, 3), float32, [0, 1] — a new array; *image* is not modified
    """
    image = _check_image(image)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != image.shape[:2]:
        raise ValueError(f"keep must have shape {image.shape[:2]}, got {keep.shape}")
    if method not in METHODS:
        raise ValueError(f"unknown hiding method {method!r}; choose from {METHODS}")
    if keep.all():
        return image.copy()
    if method == "inpaint":
        fill = _inpaint_fill(image, keep)
    else:
        cache = {} if cache is None else cache
        if method not in cache:
            cache.update(make_cache(image, (method,)))
        fill = cache[method]
    return np.where(keep[..., np.newaxis], image, fill).astype(np.float32)


def coalition_keep(masks, index):
    """Pixels kept in coalition *index* (bit p set = group p kept): every pixel outside the
    groups, and the pixels of the kept groups."""
    masks = np.asarray(masks, dtype=bool)
    hidden = np.zeros(masks.shape[1:], bool)
    for p in range(len(masks)):
        if not (index >> p) & 1:
            hidden |= masks[p]
    return ~hidden


def coalition_images(image, masks, method, cache=None, indices=None, workers=DEFAULT_WORKERS):
    """Every coalition image of one image.

    Parameters
    ----------
    image : ndarray (H, W, 3), float32, [0, 1]
    masks : ndarray (P, H, W), bool — :func:`group_masks`
    method : hiding method, see :func:`hide`
    cache : see :func:`hide`
    indices : coalition indices to build (default: all 2^P, in order)
    workers : threads for ``inpaint`` (about 60 ms per coalition image on one core); the
        result does not depend on it

    Returns
    -------
    images : ndarray (2^P, H, W, 3) (or (len(indices), ...)), float32 — row k is coalition
        k: bit p of k set = group p kept; row 0 hides every group, row 2^P - 1 is the image
    """
    image = _check_image(image)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"masks must have shape (P, {image.shape[0]}, {image.shape[1]}), got {masks.shape}")
    n = 1 << len(masks)
    indices = range(n) if indices is None else [int(k) for k in indices]
    cache = {} if cache is None else cache
    build = lambda k: hide(image, coalition_keep(masks, k), method, cache)  # noqa: E731
    if method == "inpaint" and workers > 1 and len(indices) > 1:
        with ThreadPoolExecutor(min(workers, len(indices))) as pool:
            return np.stack(list(pool.map(build, indices)))
    return np.stack([build(k) for k in indices])


# ---------------------------------------------------------------------------
# Exact Shapley values and interactions
# ---------------------------------------------------------------------------

def _game(values):
    """(values as float64, number of players P) of a game given as 2^P coalition values."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or len(v) < 2 or len(v) & (len(v) - 1):
        raise ValueError(f"values must be a 1-D array of 2^P coalition values (P >= 1), got shape {v.shape}")
    return v, len(v).bit_length() - 1


def _sizes(n_players):
    """Number of players in every coalition index 0 .. 2^P - 1."""
    index = np.arange(1 << n_players)
    return sum((index >> p) & 1 for p in range(n_players))


def exact_shapley(values):
    """Exact Shapley values of a game.

    Parameters
    ----------
    values : array (2^P,) — v of every coalition, index = bitmask of the players in it

    Returns
    -------
    phi : ndarray (P,), float64 — sum(phi) == values[-1] - values[0] (efficiency)
    """
    v, n = _game(values)
    index, sizes = np.arange(1 << n), _sizes(n)
    weight = np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)])
    phi = np.empty(n)
    for p in range(n):
        without = index[((index >> p) & 1) == 0]
        phi[p] = np.sum(weight[sizes[without]] * (v[without | (1 << p)] - v[without]))
    return phi


def shapley_interactions(values):
    """Shapley interaction index of every pair of players (Grabisch 1997).

    Parameters
    ----------
    values : array (2^P,) — as for :func:`exact_shapley`

    Returns
    -------
    interactions : ndarray (P, P), float64, symmetric — entry (p, q) is the pair's
        interaction index (positive = the two parts are worth more together than apart);
        the diagonal holds the Shapley values (the order-1 index)
    """
    v, n = _game(values)
    out = np.diag(exact_shapley(v))
    if n < 2:
        return out
    index, sizes = np.arange(1 << n), _sizes(n)
    weight = np.array([math.factorial(s) * math.factorial(n - s - 2) / math.factorial(n - 1)
                       for s in range(n - 1)])
    for p in range(n):
        for q in range(p + 1, n):
            bp, bq = 1 << p, 1 << q
            rest = index[(index & (bp | bq)) == 0]
            # written as a difference of two marginal contributions, so that it is exactly 0
            # when p or q is a null player (v unchanged by adding it)
            delta = (v[rest | bp | bq] - v[rest | bq]) - (v[rest | bp] - v[rest])
            out[p, q] = out[q, p] = np.sum(weight[sizes[rest]] * delta)
    return out


def canonical_indices(n_players, present):
    """For every coalition, the coalition with the absent players removed.

    Coalitions that differ only in absent players have the same image, so only the
    canonical ones (no absent bit set) need a forward pass.

    Returns
    -------
    canonical : int array (2^P,)
    """
    present = np.asarray(present, dtype=bool)
    if present.shape != (n_players,):
        raise ValueError(f"present must have shape ({n_players},), got {present.shape}")
    keep_bits = sum(1 << p for p in range(n_players) if present[p])
    return np.arange(1 << n_players) & keep_bits


# ---------------------------------------------------------------------------
# Scores on a model
# ---------------------------------------------------------------------------

def _forward(model, images, output, batch_size):
    """Margin-free logits (N, C) or L2-normalised embeddings (N, D) of *images*, batched."""
    import tensorflow as tf                                             # noqa: F401 (TF only when needed)
    from src.eval.verification import embed
    from src.models.arcface_inference import get_margin_free

    images = np.asarray(images, dtype=np.float32)
    if output == "embedding":
        return embed(model, images, batch_size=batch_size)
    net = get_margin_free(model)
    parts = [net(tf.constant(images[i:i + batch_size])).numpy() for i in range(0, len(images), batch_size)]
    return np.concatenate(parts) if parts else np.zeros((0, net.n_classes), np.float32)


def model_outputs(model, images, output="logits", batch_size=64):
    """Margin-free logits (``output="logits"``, (N, C)) or L2-normalised embeddings
    (``"embedding"``, (N, D)) of a batch of images, *batch_size* images per forward pass."""
    if output not in ("logits", "embedding"):
        raise ValueError(f"output must be 'logits' or 'embedding', got {output!r}")
    return _forward(model, images, output, batch_size)


def output_kind(score):
    """What a score needs from the model: "logits" or "embedding"."""
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}; choose from {SCORES}")
    return "embedding" if score == "verif" else "logits"


def score_values(outputs, score, target=None, template=None):
    """Value of every coalition from the model outputs of its image.

    Parameters
    ----------
    outputs : ndarray (K, C) margin-free logits (``logit``, ``margin``) or (K, D)
        L2-normalised embeddings (``verif``)
    score : "logit" (logit of *target*), "margin" (logit of *target* minus the best other
        logit) or "verif" (cosine with *template*)
    target : int — class for "logit" and "margin"
    template : ndarray (D,) — embedding for "verif" (normalised here)

    Returns
    -------
    values : ndarray (K,), float64
    """
    out = np.asarray(outputs, dtype=np.float64)
    if score == "verif":
        if template is None:
            raise ValueError("score 'verif' needs a template embedding")
        t = np.asarray(template, dtype=np.float64).reshape(-1)
        if t.shape[0] != out.shape[1]:
            raise ValueError(f"template has {t.shape[0]} dimensions, the embeddings {out.shape[1]}")
        return out @ (t / max(np.linalg.norm(t), 1e-12))
    if score not in ("logit", "margin"):
        raise ValueError(f"unknown score {score!r}; choose from {SCORES}")
    if target is None or not 0 <= int(target) < out.shape[1]:
        raise ValueError(f"score {score!r} needs a target class in 0..{out.shape[1] - 1}, got {target!r}")
    target = int(target)
    value = out[:, target]
    if score == "logit":
        return value.copy()
    others = np.delete(out, target, axis=1)
    return value - others.max(axis=1)


def coalition_outputs(model, image, masks, methods, output="logits", batch_size=64, cache=None,
                      workers=DEFAULT_WORKERS):
    """Model outputs of every coalition image, for each hiding method, in one batched pass.

    Only coalitions that differ in a present group (a group with at least one pixel) go
    through the model; the others get the output of their canonical coalition
    (:func:`canonical_indices`), so absent groups are exact null players.

    Parameters
    ----------
    model : the dual-input [image, label] ArcFace model
    image : ndarray (256, 256, 3), float32, [0, 1]
    masks : ndarray (P, 256, 256), bool — :func:`group_masks`
    methods : sequence of hiding methods
    output : "logits" (margin-free) or "embedding" (L2-normalised)
    batch_size : images per forward pass
    cache : see :func:`hide`
    workers : threads for inpainting, see :func:`coalition_images`

    Returns
    -------
    outputs : dict method -> ndarray (2^P, C or D), float32
    present : ndarray (P,), bool
    n_forward : int — images that went through the model
    """
    masks = np.asarray(masks, dtype=bool)
    present = masks.reshape(len(masks), -1).any(axis=1)
    canonical = canonical_indices(len(masks), present)
    needed = np.unique(canonical)
    cache = make_cache(image, [m for m in methods if m != "inpaint"]) if cache is None else cache
    images = np.concatenate([coalition_images(image, masks, m, cache, needed, workers) for m in methods])
    outputs = _forward(model, images, output, batch_size)
    position = {k: i for i, k in enumerate(needed)}
    rows = np.array([position[k] for k in canonical])
    per_method = {m: outputs[j * len(needed):(j + 1) * len(needed)][rows] for j, m in enumerate(methods)}
    return per_method, present, int(len(images))


def asa(model, image, labels, class_names, groups=DEFAULT_GROUPS, target=None, method="blur", score="margin",
        template=None, batch_size=64, masks=None, cache=None, workers=DEFAULT_WORKERS):
    """Anatomical Shapley Attribution of one image.

    Parameters
    ----------
    model : the dual-input [image, label] ArcFace model
    image : ndarray (256, 256, 3), float32, [0, 1]
    labels : ndarray (256, 256) — the image's eye-mask label map (model-input geometry)
    class_names : names of the label ids
    groups : part groups (players), see :func:`normalise_groups`
    target : int — class whose score is explained ("logit", "margin")
    method : hiding method, see :func:`hide`
    score : "logit", "margin" or "verif", see :func:`score_values`
    template : ndarray (D,) — template embedding for "verif"
    batch_size : coalition images per forward pass (2^5 = 32 per image and method)
    masks : precomputed :func:`group_masks` of *labels* (optional)
    cache : see :func:`hide`
    workers : threads for inpainting, see :func:`coalition_images`

    Returns
    -------
    dict with ``phi`` (P,), ``interactions`` (P, P), ``v_full``, ``v_empty``, ``values``
    (2^P,), ``present`` (P,) bool, ``names`` (P,) and ``n_forward``
    """
    names = group_names(groups)
    if masks is None:
        masks = group_masks(labels, class_names, groups)
    outputs, present, n_forward = coalition_outputs(model, image, masks, [method], output_kind(score),
                                                    batch_size, cache, workers)
    values = score_values(outputs[method], score, target, template)
    return {"phi": exact_shapley(values), "interactions": shapley_interactions(values),
            "v_full": float(values[-1]), "v_empty": float(values[0]), "values": values,
            "present": present, "names": names, "n_forward": n_forward}


# ---------------------------------------------------------------------------
# Faithfulness helpers
# ---------------------------------------------------------------------------

def part_masses(heatmap, labels, class_names, groups=DEFAULT_GROUPS, masks=None):
    """Share of a native heatmap's mass on each part group.

    The group masks are area-averaged to the heatmap's grid exactly as
    src/compare/regions.py does for region masses (``masks_to_grids``,
    ``region_mass_named``). Unknown pixels belong to no group, so the shares sum to
    1 minus the weight on them.

    Parameters
    ----------
    heatmap : ndarray (h, w), non-negative (e.g. an 8 x 8 LayerCAM or attention map)
    labels : ndarray (256, 256) — eye-mask label map in model-input geometry
    class_names, groups : as for :func:`group_masks`
    masks : precomputed :func:`group_masks` (optional)

    Returns
    -------
    masses : ndarray (P,), float64 — NaN for an all-zero heatmap
    """
    names = group_names(groups)
    if masks is None:
        masks = group_masks(labels, class_names, groups)
    heatmap = np.asarray(heatmap, dtype=np.float64)
    grids = masks_to_grids(np.asarray(masks, dtype=bool), heatmap.shape)
    mass = region_mass_named(heatmap, grids, names)
    return np.array([mass[f"mass_{n}"] for n in names])


def rank_agreement(phi, other, present, min_groups=3):
    """Spearman correlation between two attributions over the present groups.

    Parameters
    ----------
    phi, other : arrays (P,) — e.g. ASA values and a heatmap's part masses
    present : array (P,) bool — groups with pixels; absent groups are left out
    min_groups : with fewer present groups (with finite values in both) the result is
        NaN: two points always correlate +-1

    Returns
    -------
    rho : float in [-1, 1], or NaN (too few groups, or a constant attribution)
    """
    phi, other = np.asarray(phi, np.float64), np.asarray(other, np.float64)
    use = np.asarray(present, bool) & np.isfinite(phi) & np.isfinite(other)
    if use.sum() < max(2, min_groups):
        return float("nan")
    a, b = phi[use], other[use]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def order_from_scores(scores, present):
    """Present groups from the highest score to the lowest (ties keep group order); None if a
    present group has no finite score."""
    scores = np.asarray(scores, np.float64)
    idx = np.flatnonzero(np.asarray(present, bool))
    if not np.isfinite(scores[idx]).all():
        return None
    return [int(p) for p in idx[np.argsort(-scores[idx], kind="stable")]]


def oriented_phi(phi, v_full, v_empty):
    """phi signed so that larger = a larger share of the gap v_full - v_empty: phi itself when the
    full image scores at least as high as the empty one, -phi when it scores lower (e.g. the true
    class of an attacked image). :func:`asa_order` ranks this quantity; compare other attributions
    with it (rank agreement) so that both use one convention."""
    sign = -1.0 if v_full < v_empty else 1.0
    return sign * np.asarray(phi, np.float64)


def asa_order(phi, v_full, v_empty, present):
    """Present groups by their share of the gap v_full - v_empty, largest first: phi from high
    to low when the full image scores higher than the empty one, from low to high when it
    scores lower (e.g. the true class of an attacked image; :func:`oriented_phi`). The deletion
    curve is rescaled to run from 1 to 0 either way, and for an additive game this order gives
    the lowest possible deletion AUC (every partial sum of the deleted contributions is largest)."""
    return order_from_scores(oriented_phi(phi, v_full, v_empty), present)


def deletion_from_values(values, order):
    """Deletion curve of an order, read off the coalition values.

    Step k hides the first k groups of *order* (all other groups kept), so the curve
    starts at v(full) and ends at the value with every group of *order* hidden.

    Parameters
    ----------
    values : array (2^P,) — coalition values (index = bitmask of kept groups)
    order : sequence of group indices, each at most once (usually the present groups)

    Returns
    -------
    dict with ``scores`` (len(order) + 1,) and ``auc``: the mean (trapezoid rule over
    k / len(order)) of the curve rescaled to run from 1 (nothing hidden) to 0 (every
    group of *order* hidden). Lower = the score falls sooner = a more faithful order.
    NaN when the order has fewer than 2 groups or the end points are equal.
    """
    v, n = _game(values)
    order = [int(p) for p in order]
    if len(set(order)) != len(order) or not all(0 <= p < n for p in order):
        raise ValueError(f"order must list distinct groups in 0..{n - 1}, got {order}")
    index = (1 << n) - 1
    scores = [v[index]]
    for p in order:
        index &= ~(1 << p)
        scores.append(v[index])
    scores = np.array(scores)
    span = scores[0] - scores[-1]
    if len(order) < 2 or span == 0:
        return {"scores": scores, "auc": float("nan")}
    curve = (scores - scores[-1]) / span
    return {"scores": scores, "auc": float(np.sum((curve[1:] + curve[:-1]) / 2) / len(order))}


def deletion_curve(model, image, masks, order, method, score, target=None, template=None, batch_size=64,
                   cache=None):
    """Deletion curve of an order on a model: the score as the groups of *order* are hidden
    one after another (len(order) + 1 forward passes). Same output as
    :func:`deletion_from_values` on the ASA values of the same image, method and score."""
    masks = np.asarray(masks, dtype=bool)
    n = len(masks)
    order = [int(p) for p in order]
    if len(set(order)) != len(order) or not all(0 <= p < n for p in order):
        raise ValueError(f"order must list distinct groups in 0..{n - 1}, got {order}")
    indices, index = [(1 << n) - 1], (1 << n) - 1
    for p in order:
        index &= ~(1 << p)
        indices.append(index)
    images = coalition_images(image, masks, method, cache, indices)
    scores = score_values(_forward(model, images, output_kind(score), batch_size), score, target, template)
    values = np.full(1 << n, np.nan)
    values[indices] = scores
    return deletion_from_values(values, order)
