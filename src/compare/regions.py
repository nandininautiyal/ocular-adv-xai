"""
src/compare/regions.py

Where on the eye does a heatmap put its weight?

Divergence metrics (src/compare/divergence.py) say *how much* an explanation
moves under an attack; region mass says *where to*: the fraction of the
heatmap that falls on each eye region. Region masks are pixel masks of the
clean image in model-input geometry, area-averaged to the heatmap's native
grid, so every cell holds the fraction of it that each region covers.

Two sources of regions:
- Heuristic masks (src/xai/masks.py), :data:`REGIONS` = pupil, iris, sclera,
  periocular, via :func:`region_grids`. They partition the image, so the four
  masses of a heatmap sum to 1. They are a classical OpenCV estimate, and
  "periocular" is the outside of one fixed ellipse (69 % of every image):
  validate them on hand-annotated images before reporting region-level results.
- Eye-mask label maps (masks_<tag>.npz: one label id per pixel, names from its
  class_names, e.g. skin, iris, sclera, lid, caruncle, brow), via
  :func:`grids_from_labels`. Label 255 (unknown) belongs to no region, so the
  masses sum to 1 minus the weight on unknown pixels.

A mass means little without the region's area: a map spread evenly over the
image puts 69 % of its weight on the heuristic periocular ring. So
:func:`region_areas` gives each region's area share, :func:`enrichment` the
mass / area ratio (1 = no preference) and :func:`normalised_entropy` how spread
out a map is. Occluders (patch, monocle) replace the pixels under them, so a
region shift in those images is partly the occlusion itself;
:func:`occlusion_grids` lets :func:`region_mass_named` recompute masses over
the pixels the occluder left unchanged.

Grids may be square (an int, e.g. 8 for Xception) or (h, w), e.g. the 5 x 8
maps of a 160 x 240 input.
"""

import cv2
import numpy as np

from src.xai.masks import get_concept_masks

REGIONS = ("pupil", "iris", "sclera", "periocular")
IGNORE_LABEL = 255          # "unknown" in eye-mask label maps; counted in no region


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def grid_shape(grid):
    """(h, w) of a grid given as an int (square) or as a pair (h, w)."""
    if np.ndim(grid) == 0:
        h = w = int(grid)
    else:
        h, w = (int(v) for v in grid)
    if h < 1 or w < 1:
        raise ValueError(f"grid must be positive, got {grid!r}")
    return h, w


def _names(names):
    names = tuple(str(n) for n in names)
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError(f"region names must be distinct and non-empty, got {names}")
    return names


def masks_to_grids(masks, grid):
    """Area-average pixel masks to a grid: the fraction of each cell each mask covers.

    The same arithmetic as ``src.xai.masks.masks_to_grid`` (OpenCV INTER_AREA),
    for any number of masks, any image size and square or (h, w) grids. It is an
    exact area average, also when the image size is not a multiple of the grid.

    Parameters
    ----------
    masks : ndarray (..., H, W), bool or float in [0, 1]
    grid : int or (h, w)

    Returns
    -------
    grids : ndarray (..., h, w), float32
    """
    masks = np.asarray(masks)
    if masks.ndim < 2:
        raise ValueError(f"masks must have shape (..., H, W), got {masks.shape}")
    h, w = grid_shape(grid)
    flat = masks.reshape((-1,) + masks.shape[-2:])
    out = np.empty((len(flat), h, w), np.float32)
    for k, mask in enumerate(flat):
        out[k] = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).reshape(h, w)
    return out.reshape(masks.shape[:-2] + (h, w))


def heuristic_masks(images):
    """Heuristic concept masks (src/xai/masks.py) of each image as pixel masks.

    Parameters
    ----------
    images : ndarray (N, 256, 256, 3), float32, [0, 1]

    Returns
    -------
    masks : ndarray (N, 4, 256, 256), bool — channel order as in :data:`REGIONS`
    """
    return np.stack([np.stack([m[r] for r in REGIONS]) for m in map(get_concept_masks, images)])


def region_grids(images, grid):
    """Concept masks of each image, area-averaged to the heatmap grid.

    Parameters
    ----------
    images : ndarray (N, 256, 256, 3), float32, [0, 1]
    grid : int or (h, w) — size of the heatmaps the masks will be compared with (8 for Xception)

    Returns
    -------
    grids : ndarray (N, 4, h, w), float32 — channel order as in :data:`REGIONS`
    """
    return masks_to_grids(heuristic_masks(images), grid)


def label_masks(labels, class_names):
    """Pixel masks of eye-mask label maps, one channel per class.

    Parameters
    ----------
    labels : ndarray (N, H, W) of integer label ids, the index into *class_names*;
        255 marks unknown pixels, which get no channel
    class_names : sequence of str, e.g. ("skin", "iris", "sclera", "lid", "caruncle", "brow")

    Returns
    -------
    masks : ndarray (N, R, H, W), bool, R = len(class_names)
    """
    names = _names(class_names)
    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integer label maps of shape (N, H, W), "
                         f"got {labels.dtype} {labels.shape}")
    valid = ((labels >= 0) & (labels < len(names))) | (labels == IGNORE_LABEL)
    if not valid.all():
        raise ValueError(f"labels must be in 0..{len(names) - 1} (class_names) or {IGNORE_LABEL}, "
                         f"found {np.unique(labels[~valid])[:5]}")
    return labels[:, np.newaxis] == np.arange(len(names)).reshape(1, -1, 1, 1)


def grids_from_labels(labels, class_names, grid):
    """Eye-mask label maps, area-averaged to the heatmap grid.

    Parameters
    ----------
    labels : ndarray (N, H, W), uint8 — label ids in model-input geometry (masks_<tag>.npz),
        255 = unknown
    class_names : sequence of str — index = label id
    grid : int or (h, w)

    Returns
    -------
    grids : ndarray (N, R, h, w), float32 — share of each cell covered by each class
    names : tuple of str — region names, in channel order (= class_names)
    """
    names = _names(class_names)
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape (N, H, W), got {labels.shape}")
    grids = np.empty((len(labels), len(names)) + grid_shape(grid), np.float32)
    for i, image_labels in enumerate(labels):        # one image at a time keeps memory small
        grids[i] = masks_to_grids(label_masks(image_labels[np.newaxis], names)[0], grid)
    return grids, names


def occlusion_mask(clean, attacked, tol=0.5 / 255.0):
    """Pixels an occluder changed: some channel differs from the clean image by more than *tol*.

    Parameters
    ----------
    clean, attacked : ndarray (..., H, W, 3), float32, [0, 1]
    tol : float — default half an 8-bit level

    Returns
    -------
    occluded : ndarray (..., H, W), bool; its mean is the occluded share of the image
    """
    diff = np.abs(np.asarray(attacked, dtype=np.float32) - np.asarray(clean, dtype=np.float32))
    return (diff > tol).any(axis=-1)


def occlusion_grids(region_masks, occluded, grid):
    """Region grids restricted to the pixels an occluder left unchanged.

    Pass both outputs to :func:`region_mass_named` (``visible=...``, e.g.
    ``prefix="massvis"``): the heatmap weight on occluded pixels is set to 0 and the
    masses are renormalised over the rest; a region the occluder covers completely
    gets NaN. With nothing occluded this equals the plain region mass (except that a
    region absent from the image gets NaN instead of 0).

    Parameters
    ----------
    region_masks : ndarray (..., R, H, W), bool — pixel masks of the clean image
        (:func:`heuristic_masks` or :func:`label_masks`)
    occluded : ndarray (..., H, W), bool — e.g. :func:`occlusion_mask`
    grid : int or (h, w)

    Returns
    -------
    grids : ndarray (..., R, h, w), float32 — share of each cell that is in the region and visible
    visible : ndarray (..., h, w), float32 — share of each cell that is visible
    """
    region_masks = np.asarray(region_masks, dtype=bool)
    occluded = np.asarray(occluded, dtype=bool)
    if region_masks.shape[-2:] != occluded.shape[-2:]:
        raise ValueError(f"region_masks {region_masks.shape} and occluded {occluded.shape} "
                         f"must cover the same pixels")
    visible = ~occluded
    return masks_to_grids(region_masks & visible[..., np.newaxis, :, :], grid), masks_to_grids(visible, grid)


# ---------------------------------------------------------------------------
# Per-map measures
# ---------------------------------------------------------------------------

def region_mass_named(heatmap, grids, names, visible=None, prefix="mass"):
    """Fraction of the heatmap's total weight on each named region.

    Parameters
    ----------
    heatmap : ndarray (h, w), non-negative
    grids : ndarray (R, h, w) — from :func:`region_grids`, :func:`grids_from_labels`
        or :func:`occlusion_grids`
    names : sequence of R region names
    visible : ndarray (h, w) or None — share of each cell an occluder left unchanged
        (:func:`occlusion_grids`); the total weight is then taken over visible pixels only,
        and a region with no visible pixel gets NaN: the occluder hid all of it, so its
        mass is not measured (0 would read as "no weight there", and its change from the
        clean map as "no shift")
    prefix : str — column prefix: "mass", or e.g. "massvis" for occlusion-aware masses

    Returns
    -------
    mass : dict ``<prefix>_<name>`` -> float; NaN for an all-zero heatmap (or no visible weight)
    """
    names = _names(names)
    heatmap = np.asarray(heatmap, dtype=np.float64)
    grids = np.asarray(grids, dtype=np.float64)
    if grids.shape != (len(names),) + heatmap.shape:
        raise ValueError(f"grids must have shape {(len(names),) + heatmap.shape}, got {grids.shape}")
    if (heatmap < 0).any():
        raise ValueError("heatmap must be non-negative")
    weights = heatmap
    if visible is not None:
        visible = np.asarray(visible, dtype=np.float64)
        if visible.shape != heatmap.shape:
            raise ValueError(f"visible must have shape {heatmap.shape}, got {visible.shape}")
        weights = heatmap * visible
    total = weights.sum()
    if total == 0:
        return {f"{prefix}_{n}": float("nan") for n in names}
    return {f"{prefix}_{n}": float((heatmap * g).sum() / total) if visible is None or g.any() else float("nan")
            for n, g in zip(names, grids)}


def region_mass(heatmap, grids):
    """Fraction of the heatmap's total weight on each heuristic region.

    Parameters
    ----------
    heatmap : ndarray (H, W), non-negative
    grids : ndarray (4, H, W) from :func:`region_grids`

    Returns
    -------
    mass : dict ``mass_<region>`` -> float; NaN for an all-zero heatmap
    """
    return region_mass_named(heatmap, grids, REGIONS)


def region_areas(grids, names):
    """Area share of each region on the grid: the mean of its cells.

    For grids from :func:`masks_to_grids` (an exact area average) this is the
    region's share of the image's pixels.

    Parameters
    ----------
    grids : ndarray (R, h, w) — one image
    names : sequence of R region names

    Returns
    -------
    areas : dict ``area_<name>`` -> float
    """
    names = _names(names)
    grids = np.asarray(grids, dtype=np.float64)
    if grids.ndim != 3 or len(grids) != len(names):
        raise ValueError(f"grids must have shape ({len(names)}, h, w), got {grids.shape}")
    return {f"area_{n}": float(g.mean()) for n, g in zip(names, grids)}


def enrichment(masses, areas, prefix="enr"):
    """Mass over area share for each region: 1 means the map weights the region by its area only.

    Parameters
    ----------
    masses : dict ``<mass prefix>_<name>`` -> float, from :func:`region_mass_named`
    areas : dict ``area_<name>`` -> float, from :func:`region_areas`
    prefix : str — column prefix of the result

    Returns
    -------
    enrichment : dict ``<prefix>_<name>`` -> float; NaN where the area is 0
    """
    out = {}
    for key, mass in masses.items():
        name = key.split("_", 1)[1]
        area = areas[f"area_{name}"]
        out[f"{prefix}_{name}"] = float(mass / area) if area > 0 else float("nan")
    return out


def normalised_entropy(heatmap):
    """Shannon entropy of a non-negative map, divided by its maximum log(number of cells).

    1 for a uniform map, 0 for a map with a single non-zero cell; NaN for an
    all-zero map (or a single cell). Use it on native maps (8 x 8), not on
    upsampled ones.

    Parameters
    ----------
    heatmap : ndarray (h, w), non-negative

    Returns
    -------
    entropy : float in [0, 1]
    """
    p = np.asarray(heatmap, dtype=np.float64).ravel()
    if (p < 0).any():
        raise ValueError("heatmap must be non-negative")
    total = p.sum()
    if p.size < 2 or not total > 0:
        return float("nan")
    p = p[p > 0] / total
    return float(np.clip(-(p * np.log(p)).sum() / np.log(np.size(heatmap)), 0.0, 1.0))
