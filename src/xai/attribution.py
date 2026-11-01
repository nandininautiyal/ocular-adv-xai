"""
src/xai/attribution.py

LayerCAM and Eigen-CAM heatmap generation for the Xception + ArcFace
dual-input recognition model.

All functions are pure: they accept a model and preprocessed image(s) and
return heatmap(s) with values in [0, 1].

Image contract
--------------
- dtype : float32
- shape : (256, 256, 3), or (N, 256, 256, 3) for a batch
- range : [0, 1]

Model contract
--------------
The model has two inputs: [image_input, label_input]. Explanations use the
margin-free path (zero label vector) through ``src.models.arcface_inference``.

Choices that matter for the heatmap-divergence study
----------------------------------------------------
- Target layer: the last 4-D activation before the head (``block14_sepconv2_act``
  for Xception). The raw ``block14_sepconv2`` output is pre-BatchNorm and can be
  negative, which both methods assume it is not.
- Class score: the margin-free cosine logit ``s * cos(theta_k)``. The vendored
  ArcFace layer applies softmax inside call(); with s = 64 the top probability
  rounds to 1.0, so gradients of the softmax output are numerical noise.
- Eigen-CAM sign: an SVD singular vector is only defined up to sign, so the map
  can come out inverted. The component is flipped to correlate positively with
  the summed activations, which makes the map deterministic.
- Resolution: the Xception feature map is only 8 x 8 for a 256 x 256 input.
  Pass ``size=None`` to get the native map; compare metrics on native maps and
  upsample only for display.
- Magnitude: every map is min-max scaled on its own, which discards how strong
  the evidence was; a weak map looks as spread out as a strong one.
  ``return_raw=True`` also returns the sum and maximum of each native map before
  scaling, so "evidence removed" can be told apart from "evidence spread out".
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.arcface_inference import as_batch, find_feature_layer, get_margin_free

DEFAULT_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_max(maps):
    """Scale each map in a (N, h, w) array to [0, 1]; constant maps become 0."""
    lo = maps.min(axis=(1, 2), keepdims=True)
    span = maps.max(axis=(1, 2), keepdims=True) - lo
    return np.where(span > 0, (maps - lo) / np.where(span > 0, span, 1.0), 0.0).astype(np.float32)


def _finish(maps, size, single, return_raw=False):
    """Normalise, optionally resize (bilinear), and drop the batch axis for single images.

    With *return_raw*, also return the sum and maximum of each native map before
    normalisation (floats for a single image, (N,) float64 arrays for a batch).
    """
    raw = np.asarray(maps, dtype=np.float64)
    maps = _min_max(maps)
    if size is not None:
        maps = tf.image.resize(maps[..., np.newaxis], size)[..., 0].numpy()
        maps = np.clip(maps, 0.0, 1.0).astype(np.float32)
    maps = maps[0] if single else maps
    if not return_raw:
        return maps
    raw_sum, raw_max = raw.sum(axis=(1, 2)), raw.max(axis=(1, 2))
    if single:
        return maps, float(raw_sum[0]), float(raw_max[0])
    return maps, raw_sum, raw_max


# ---------------------------------------------------------------------------
# LayerCAM  (Jiang et al., 2021)
# ---------------------------------------------------------------------------

def layercam(model, image, class_idx=None, target_layer=None, size=DEFAULT_SIZE, return_raw=False):
    """Generate a LayerCAM heatmap for *image*.

    Parameters
    ----------
    model : tf.keras.Model
        The dual-input Xception + ArcFace model.
    image : np.ndarray, (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    class_idx : int, sequence of int, or None
        Class to explain (one per image for a batch). If *None*, the predicted
        class is used. To study attention shift, explain the true class at every
        stage (clean, attacked, defended) rather than each stage's prediction.
    target_layer : str or None
        Layer to explain. Defaults to the last 4-D activation (block14_sepconv2_act).
    size : (int, int) or None
        Output resolution. *None* returns the native feature-map resolution.
    return_raw : bool
        Also return the sum and maximum of the native map before min-max scaling
        (in units of logit gradient x activation), whatever *size* is.

    Returns
    -------
    heatmap : np.ndarray, (H, W) or (N, H, W), float32, range [0, 1]
    raw_sum, raw_max : float, or np.ndarray (N,) float64 for a batch — only with *return_raw*
    """
    x, single = as_batch(image)
    target_layer = target_layer or find_feature_layer(model)
    net = get_margin_free(model, target_layer)

    with tf.GradientTape() as tape:
        features, logits = net(x)
        if class_idx is None:
            idx = tf.argmax(logits, axis=1, output_type=tf.int32)
        else:
            idx = tf.reshape(tf.constant(class_idx, dtype=tf.int32), [-1])
            idx = tf.broadcast_to(idx, [tf.shape(x)[0]])
        score = tf.reduce_sum(tf.gather(logits, idx, axis=1, batch_dims=1))

    grads = tape.gradient(score, features)                        # (N, h, w, c)

    # LayerCAM: ReLU(gradients) weight each activation location, sum over channels.
    cams = tf.nn.relu(tf.reduce_sum(tf.nn.relu(grads) * features, axis=-1))
    return _finish(cams.numpy(), size, single, return_raw)


# ---------------------------------------------------------------------------
# Eigen-CAM  (Muhammad & Yeasin, 2020)
# ---------------------------------------------------------------------------

def _eigen_projection(activations):
    """Projection of one (h, w, c) activation map on its first singular vector.

    Uncentred, as in the Eigen-CAM paper. The singular vector's sign is
    arbitrary, so it is aligned with the total activation: the salient region is
    then always the high end of the map, whatever sign LAPACK returns.
    """
    h, w, c = activations.shape
    flat = activations.reshape(h * w, c).astype(np.float64)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    projection = flat @ vt[0]
    if np.dot(projection, flat.sum(axis=1)) < 0:
        projection = -projection
    return projection.reshape(h, w)


def eigencam(model, image, target_layer=None, size=DEFAULT_SIZE, return_raw=False):
    """Generate an Eigen-CAM heatmap for *image*.

    Eigen-CAM is gradient-free and class-agnostic: it projects the activations
    at the target layer onto their first principal component.

    Parameters
    ----------
    model : tf.keras.Model
        The dual-input Xception + ArcFace model.
    image : np.ndarray, (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    target_layer : str or None
        Layer to explain. Defaults to the last 4-D activation (block14_sepconv2_act).
    size : (int, int) or None
        Output resolution. *None* returns the native feature-map resolution.
    return_raw : bool
        Also return the sum and maximum of the native projection before min-max
        scaling (in activation units), whatever *size* is.

    Returns
    -------
    heatmap : np.ndarray, (H, W) or (N, H, W), float32, range [0, 1]
    raw_sum, raw_max : float, or np.ndarray (N,) float64 for a batch — only with *return_raw*
    """
    x, single = as_batch(image)
    target_layer = target_layer or find_feature_layer(model)
    features, _ = get_margin_free(model, target_layer)(x)
    maps = np.stack([_eigen_projection(a) for a in features.numpy()])
    return _finish(maps, size, single, return_raw)


# ---------------------------------------------------------------------------
# Registry for easy iteration in evaluation scripts
# ---------------------------------------------------------------------------

XAI_REGISTRY = {
    "layercam": layercam,
    "eigencam": eigencam,
}
