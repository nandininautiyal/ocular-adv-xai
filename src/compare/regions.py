"""
src/compare/regions.py

Where on the eye does a heatmap put its weight?

Divergence metrics (src/compare/divergence.py) say *how much* an explanation
moves under an attack; region mass says *where to*: the fraction of the
heatmap that falls on the pupil, iris, sclera and periocular skin. Masks come
from the clean image (src/xai/masks.py) and are area-averaged to the heatmap's
native grid (``masks_to_grid``), so every cell holds the fraction of it that
each region covers. The four regions partition the image, so the four masses
of a heatmap sum to 1.

The masks are a classical OpenCV estimate; validate them on a few
hand-annotated images before reporting region-level results.
"""

import numpy as np

from src.xai.masks import get_concept_masks, masks_to_grid

REGIONS = ("pupil", "iris", "sclera", "periocular")


def region_grids(images, grid):
    """Concept masks of each image, area-averaged to a (grid x grid) map.

    Parameters
    ----------
    images : ndarray (N, 256, 256, 3), float32, [0, 1]
    grid : int — side of the heatmaps the masks will be compared with (8 for Xception)

    Returns
    -------
    grids : ndarray (N, 4, grid, grid), float32 — channel order as in :data:`REGIONS`
    """
    return np.stack([masks_to_grid(get_concept_masks(img), grid=grid) for img in images])


def region_mass(heatmap, grids):
    """Fraction of the heatmap's total weight on each region.

    Parameters
    ----------
    heatmap : ndarray (H, W), non-negative
    grids : ndarray (4, H, W) from :func:`region_grids`

    Returns
    -------
    mass : dict ``mass_<region>`` -> float; NaN for an all-zero heatmap
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    grids = np.asarray(grids, dtype=np.float64)
    if grids.shape != (len(REGIONS),) + heatmap.shape:
        raise ValueError(f"grids must have shape {(len(REGIONS),) + heatmap.shape}, got {grids.shape}")
    if (heatmap < 0).any():
        raise ValueError("heatmap must be non-negative")
    total = heatmap.sum()
    if total == 0:
        return {f"mass_{r}": float("nan") for r in REGIONS}
    return {f"mass_{r}": float((heatmap * g).sum() / total) for r, g in zip(REGIONS, grids)}
