"""
src/defences/inference.py

Inference-time image-space defences.  Each defence is a **pure function**:

    defended_image = defence(image, ...)

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
- Output : float32, shape (256, 256, 3), range [0, 1]

Defences implemented (matching the README specification):
  1. JPEG compression
  2. Strip defence  (superposition of clean overlays)
  3. Bit-depth reduction (lightweight complement)

EAT (Ensemble Adversarial Training) is a *training-time* defence and
therefore does NOT belong in this module — it would be a separate
training script that augments the training set with adversarial examples.
"""

import io

import tensorflow as tf
import numpy as np

# ---------------------------------------------------------------------------
# 1.  JPEG Compression Defence
# ---------------------------------------------------------------------------

def jpeg_compress(image, quality=75):
    """Apply JPEG compression and decompression to remove high-frequency
    adversarial perturbations.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    quality : int in [1, 100] — JPEG quality factor (lower = more smoothing)

    Returns
    -------
    defended : ndarray (256, 256, 3), float32, [0, 1]
    """
    img_uint8 = tf.cast(tf.clip_by_value(image * 255.0, 0, 255), tf.uint8)
    jpeg = tf.io.encode_jpeg(img_uint8, quality=quality)
    decoded = tf.io.decode_jpeg(jpeg, channels=3)
    defended = tf.cast(decoded, tf.float32) / 255.0
    return defended.numpy()


# ---------------------------------------------------------------------------
# 2.  Strip Defence  (Gao et al., 2019)
# ---------------------------------------------------------------------------

def strip_defence(image, overlay_images, model, n_classes=300,
                  alpha=0.5, n_overlays=20, entropy_threshold=None):
    """Strip defence: superimpose clean overlay images onto the input and
    measure prediction entropy.

    If the input is adversarial, the prediction should be *strongly biased*
    toward the adversarial class regardless of the overlay, resulting in
    **low entropy**.  Clean inputs produce higher entropy because the
    overlays shift the prediction more.

    This function returns two things:
    - The blended image with the *highest-entropy* overlay (i.e. the one
      that most effectively disrupts a potential adversarial signal).
    - The mean Shannon entropy across all overlay experiments.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    overlay_images : list[ndarray]
        A pool of clean images, each (256, 256, 3) float32 [0, 1].
    model : tf.keras.Model
        The dual-input ArcFace model.
    n_classes : int
    alpha : float in (0, 1)
        Blending weight.  defended = alpha * image + (1 - alpha) * overlay.
    n_overlays : int
        Number of random overlays to sample from the pool.
    entropy_threshold : float or None
        If provided, and the mean entropy is *below* this value, the image
        is flagged as likely adversarial.

    Returns
    -------
    result : dict with keys
        - ``"defended_image"`` : ndarray (256, 256, 3), float32, [0, 1]
        - ``"mean_entropy"``   : float
        - ``"is_adversarial"`` : bool (only meaningful when entropy_threshold
          is set)
    """
    import tensorflow as tf  # deferred to avoid import-order issues

    rng = np.random.default_rng(42)
    indices = rng.choice(len(overlay_images), size=min(n_overlays, len(overlay_images)),
                          replace=False)
    dummy = np.zeros((1, n_classes), dtype="float32")

    entropies = []
    best_entropy = -1.0
    best_blended = image  # fallback

    for idx in indices:
        blended = (alpha * image + (1.0 - alpha) * overlay_images[idx]).astype(np.float32)
        blended = np.clip(blended, 0.0, 1.0)

        preds = model.predict([blended[np.newaxis], dummy], verbose=0)[0]
        preds = np.clip(preds, 1e-10, 1.0)  # avoid log(0)
        entropy = -np.sum(preds * np.log(preds))
        entropies.append(entropy)

        if entropy > best_entropy:
            best_entropy = entropy
            best_blended = blended

    mean_entropy = float(np.mean(entropies))
    is_adversarial = mean_entropy < entropy_threshold if entropy_threshold is not None else False

    return {
        "defended_image": best_blended,
        "mean_entropy": mean_entropy,
        "is_adversarial": is_adversarial,
    }


# ---------------------------------------------------------------------------
# 3.  Bit-Depth Reduction
# ---------------------------------------------------------------------------

def bit_depth_reduction(image, bits=4):
    """Reduce the bit depth of an image to remove fine-grained perturbations.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    bits : int in [1, 8] — target bit depth

    Returns
    -------
    defended : ndarray (256, 256, 3), float32, [0, 1]
    """
    levels = 2 ** bits - 1
    quantised = np.round(image * levels) / levels
    return quantised.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry for easy iteration in evaluation scripts
# ---------------------------------------------------------------------------

DEFENCE_REGISTRY = {
    "jpeg": jpeg_compress,
    "bit_depth": bit_depth_reduction,
    # strip_defence requires extra args (overlay pool) so it's called
    # separately, but listed here for discoverability.
    "strip": strip_defence,
}
