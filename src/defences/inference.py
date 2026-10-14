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
"""

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
