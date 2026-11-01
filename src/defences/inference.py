"""
src/defences/inference.py

Inference-time image-space defences.  Each defence is a **pure function**:

    defended_image = defence(image, ...)

Image contract
--------------
- Input  : float32, shape (256, 256, 3) or (N, 256, 256, 3), range [0, 1]
- Output : float32, same shape as the input, range [0, 1]

Defences implemented (both are in the ABRobOcular paper, Section 4.1):
  1. JPEG compression
  2. Bit-depth reduction

Not in this module, on purpose:
  - Strip / Patch Augmentation, EAT, FGSM-AT, PGD-AT are *training-time*
    defences: they produce a different model, so they cannot be applied to an
    image. Strip and Patch Augmentation live in ``src/defences/training.py``.
    (The earlier ``strip_defence`` here was STRIP, Gao et al. 2019, a backdoor
    detector unrelated to the paper's Strip Augmentation; it also returned a
    50/50 blend with another subject's image as the "defended" image.)
  - Geometric defences (random resize-and-pad, pixel deflection, RDG) move the
    image content, so a heatmap shift after them is not an attention shift
    unless the maps are re-aligned first.

Both defences quantise to 8 bits. Any L-inf perturbation smaller than half a
grey level (0.5 / 255 ~= 0.002, e.g. the paper's epsilon = 0.001) is removed by
that quantisation alone, before the defence itself does anything.

Parameters: the registry functions keep their defaults (JPEG quality 75,
4 bits), which the base paper does not report. ``make_defence("jpeg",
quality=50)`` and ``parse_defence("jpeg:quality=50")`` fix other values for a
parameter sweep and label them for result tables ("jpeg" for the defaults).
"""

import functools
import inspect
import numbers

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# 1.  JPEG Compression Defence
# ---------------------------------------------------------------------------

def jpeg_compress(image, quality=75):
    """Apply JPEG compression and decompression to remove high-frequency
    adversarial perturbations.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    quality : int in [1, 100] — JPEG quality factor (lower = more smoothing)

    Returns
    -------
    defended : ndarray, same shape as *image*, float32, [0, 1]
    """
    images = np.asarray(image, dtype=np.float32)
    batch = images if images.ndim == 4 else images[np.newaxis]
    out = []
    for img in batch:
        img_uint8 = tf.cast(tf.round(tf.clip_by_value(img * 255.0, 0.0, 255.0)), tf.uint8)
        decoded = tf.io.decode_jpeg(tf.io.encode_jpeg(img_uint8, quality=quality), channels=3)
        out.append(tf.cast(decoded, tf.float32).numpy() / 255.0)
    out = np.stack(out).astype(np.float32)
    return out if images.ndim == 4 else out[0]


# ---------------------------------------------------------------------------
# 2.  Bit-Depth Reduction
# ---------------------------------------------------------------------------

def bit_depth_reduction(image, bits=4):
    """Reduce the bit depth of an image to remove fine-grained perturbations.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    bits : int in [1, 8] — target bit depth

    Returns
    -------
    defended : ndarray, same shape as *image*, float32, [0, 1]
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    levels = 2 ** bits - 1
    quantised = np.round(np.clip(image, 0.0, 1.0) * levels) / levels
    return quantised.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry for easy iteration in evaluation scripts
# ---------------------------------------------------------------------------

DEFENCE_REGISTRY = {
    "jpeg": jpeg_compress,
    "bit_depth": bit_depth_reduction,
}

# Parameters each defence accepts, with their valid (inclusive) ranges.
PARAMETER_RANGES = {
    "jpeg": {"quality": (1, 100)},
    "bit_depth": {"bits": (1, 8)},
}


def make_defence(name, **params):
    """A registry defence with its parameters fixed, e.g. ``make_defence("jpeg", quality=50)``.

    Parameters
    ----------
    name : str — key of :data:`DEFENCE_REGISTRY`
    **params : whole numbers within :data:`PARAMETER_RANGES` (``quality`` for
        "jpeg", ``bits`` for "bit_depth"); omitted ones keep their defaults

    Returns
    -------
    defence : callable image -> defended image (same contract as the registry
        functions), with a ``label`` attribute for result tables: *name* when every
        parameter has its default ("jpeg"), else ``name:key=value`` ("jpeg:quality=50")

    Raises
    ------
    KeyError for an unknown *name*, TypeError for an unknown parameter, ValueError
    for a value that is not a whole number in range.
    """
    if name not in DEFENCE_REGISTRY:
        raise KeyError(f"unknown defence {name!r}; choose from {sorted(DEFENCE_REGISTRY)}")
    ranges = PARAMETER_RANGES[name]
    fixed = {}
    for key, value in params.items():
        if key not in ranges:
            raise TypeError(f"defence {name!r} has no parameter {key!r}; it takes {sorted(ranges)}")
        lo, hi = ranges[key]
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not float(value).is_integer() or not lo <= value <= hi):
            raise ValueError(f"{name} {key} must be a whole number in [{lo}, {hi}], got {value!r}")
        fixed[key] = int(value)

    fn = DEFENCE_REGISTRY[name]
    defaults = {key: param.default for key, param in inspect.signature(fn).parameters.items()}
    defence = functools.partial(fn, **fixed)
    defence.label = name + "".join(f":{key}={value}" for key, value in sorted(fixed.items())
                                   if value != defaults[key])
    return defence


def parse_defence(spec):
    """:func:`make_defence` from a command-line spec.

    Parameters
    ----------
    spec : str — "jpeg", "jpeg:quality=50", "bit_depth:bits=3"; the ``label`` of a
        defence parses back to the same defence

    Returns
    -------
    defence : callable with a ``label`` attribute, see :func:`make_defence`
    """
    name, *pairs = (part.strip() for part in str(spec).split(":"))
    params = {}
    for pair in pairs:
        key, sep, value = (part.strip() for part in pair.partition("="))
        if not sep or not key:
            raise ValueError(f"expected name:key=value in defence spec {spec!r}, got {pair!r}")
        try:
            params[key] = int(value)
        except ValueError:
            raise ValueError(f"{key} must be a whole number in defence spec {spec!r}, "
                             f"got {value!r}") from None
    return make_defence(name, **params)
