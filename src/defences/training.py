"""
src/defences/training.py

Training-time augmentation defences from the ABRobOcular paper (Section 4.1,
item 4): Strip Augmentation and Patch Augmentation. They are applied to
*training* images, so the defence is a separately trained model (e.g. the
paper's Strip5 ... Strip25 and Patch16 / Patch32 / Patch64 models), not a
transform applied to a test image.

Ported from the vendored notebooks
(third_party/ABRobOcular/Black-Box Defense/Augmentation_Strip_ArcFace.ipynb and
Patch_Visualization.ipynb). Differences from the notebooks: the input is copied
instead of modified in place, and randomness comes from an explicit
``numpy.random.Generator`` instead of the global numpy state.

Image contract
--------------
- Input  : (H, W, 3) array (any value scale; removed pixels are set to 0)
- Output : same shape and dtype
"""

import numpy as np


def strip_augmentation(image, rng, strip_size=5, p_horizontal=0.15, p_vertical=0.15):
    """Zero one full-width or full-height strip, as in the vendored notebook.

    With probability *p_horizontal* a horizontal strip is removed, with
    probability *p_vertical* a vertical one, otherwise the image is unchanged.

    Parameters
    ----------
    image : ndarray (H, W, 3)
    rng : numpy.random.Generator
    strip_size : int — strip width in pixels (the paper uses 5, 10, 15, 20, 25)
    p_horizontal, p_vertical : float

    Returns
    -------
    augmented : ndarray, same shape and dtype as *image*
    """
    out = np.array(image, copy=True)
    draw = rng.random()
    if draw < p_horizontal:
        y = rng.integers(0, out.shape[0] - strip_size)
        out[y:y + strip_size, :, :] = 0
    elif draw < p_horizontal + p_vertical:
        x = rng.integers(0, out.shape[1] - strip_size)
        out[:, x:x + strip_size, :] = 0
    return out


def patch_augmentation(image, rng, patch_size=32, p=0.3):
    """Zero one square patch at a random location.

    The vendored repo only ships a visualisation for this; its training
    notebook says the Strip protocol applies, so the patch is removed with the
    same total probability (0.3) by default.

    Parameters
    ----------
    image : ndarray (H, W, 3)
    rng : numpy.random.Generator
    patch_size : int — side length in pixels (the paper uses 8, 16, 32, 64)
    p : float — probability of removing a patch

    Returns
    -------
    augmented : ndarray, same shape and dtype as *image*
    """
    out = np.array(image, copy=True)
    if rng.random() < p:
        x = rng.integers(0, out.shape[1] - patch_size)
        y = rng.integers(0, out.shape[0] - patch_size)
        out[y:y + patch_size, x:x + patch_size, :] = 0
    return out


def make_preprocessing_function(kind, size, seed=42):
    """Return a single-argument function for ``ImageDataGenerator(preprocessing_function=...)``.

    Parameters
    ----------
    kind : {"strip", "patch"}
    size : int — strip width or patch side length
    seed : int

    Returns
    -------
    preprocess : callable (H, W, 3) -> (H, W, 3)
    """
    rng = np.random.default_rng(seed)
    if kind == "strip":
        return lambda image: strip_augmentation(image, rng, strip_size=size)
    if kind == "patch":
        return lambda image: patch_augmentation(image, rng, patch_size=size)
    raise ValueError(f"kind must be 'strip' or 'patch', got {kind!r}")
