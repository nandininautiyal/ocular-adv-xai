"""
src/models/arcface_inference.py

Margin-free inference helpers for the trained Xception + ArcFace checkpoint.

Why this module exists
----------------------
The vendored ArcFace layer (third_party/ABRobOcular/train.py) applies softmax
*inside* call(), so ``model.output`` is a probability, not a logit. With the
ArcFace scale s = 64 the top probability rounds to exactly 1.0 in float32 for a
confident prediction, and its gradient becomes numerical noise. Gradient-based
XAI (LayerCAM) and attacks therefore need the logits that feed that softmax:

    logits = s * cos(theta) = s * normalize(embedding) @ normalize(W)

With a zero label vector the margin term vanishes, so
``softmax(logits) == model([image, zeros])`` exactly.

Image contract
--------------
- dtype : float32, range [0, 1]
- shape : (256, 256, 3) for one image or (N, 256, 256, 3) for a batch
"""

import os
import weakref

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.recognition_model import ArcFace


def load_checkpoint(path):
    """Load a trained ``.keras`` / ``.h5`` checkpoint: the vendored ArcFace layer plus the layers of
    the anatomy-guided heads (src/models/anatomy_model.CUSTOM_OBJECTS), so every arch loads."""
    from src.models.anatomy_model import CUSTOM_OBJECTS
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)


def get_n_classes(model):
    """Number of classes, read from the shape of the model's label input."""
    return int(model.inputs[1].shape[-1])


def find_arcface_layer(model):
    """Return the ArcFace head of *model* (matched by class name, so reloads work)."""
    for layer in model.layers:
        if type(layer).__name__ == "ArcFace":
            return layer
    raise ValueError("Model has no ArcFace layer")


def find_feature_layer(model):
    """Name of the last 4-D activation before the ArcFace head.

    For the Xception backbone this is ``block14_sepconv2_act``: the post-BN,
    post-ReLU feature map that feeds Flatten. It is the standard Grad-CAM target
    for Xception. The raw ``block14_sepconv2`` output is pre-BatchNorm and can be
    negative, which LayerCAM and Eigen-CAM both assume it is not.
    """
    names = [layer.name for layer in model.layers]
    if "block14_sepconv2_act" in names:
        return "block14_sepconv2_act"

    last = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            rank = len(layer.output.shape)
        except (AttributeError, ValueError):  # multi-node / multi-output layers
            continue
        if rank == 4:
            last = layer.name
    if last is None:
        raise ValueError("No 4-D feature layer found before the ArcFace head")
    return last


class MarginFreeLogits:
    """Single-input view of the dual-input model: images -> [features,] logits.

    Built once per (model, feature layer) through :func:`get_margin_free`.
    Calling it never runs the ArcFace margin and never applies softmax.
    """

    def __init__(self, model, feature_layer=None):
        arcface = find_arcface_layer(model)
        embedding = arcface.input[0]
        outputs = [embedding]
        if feature_layer is not None:
            outputs = [model.get_layer(feature_layer).output, embedding]
        self._sub_model = tf.keras.Model(inputs=model.inputs[0], outputs=outputs)
        self._arcface = arcface
        self.feature_layer = feature_layer
        self.n_classes = get_n_classes(model)

    def __call__(self, images):
        """Return ``logits`` (N, C), or ``(features, logits)`` if a feature layer was set."""
        out = self._sub_model(images, training=False)
        features, embedding = (out if self.feature_layer is not None else (None, out))
        weights = tf.nn.l2_normalize(self._arcface.W, axis=0)
        logits = self._arcface.s * tf.matmul(tf.nn.l2_normalize(embedding, axis=1), weights)
        return logits if features is None else (features, logits)


_CACHE = weakref.WeakKeyDictionary()


def get_margin_free(model, feature_layer=None):
    """Cached :class:`MarginFreeLogits` for *model* (rebuilding sub-models is slow)."""
    per_model = _CACHE.setdefault(model, {})
    if feature_layer not in per_model:
        per_model[feature_layer] = MarginFreeLogits(model, feature_layer)
    return per_model[feature_layer]


def as_batch(images):
    """Return ``(float32 tensor of shape (N, H, W, 3), was_single_image)``."""
    x = tf.convert_to_tensor(images, dtype=tf.float32)
    if x.shape.rank == 3:
        return x[tf.newaxis], True
    if x.shape.rank == 4:
        return x, False
    raise ValueError(f"Expected (H, W, 3) or (N, H, W, 3), got shape {x.shape}")


def predict_logits(model, images):
    """Margin-free logits as a numpy array (N, C) for one image or a batch."""
    x, _ = as_batch(images)
    return get_margin_free(model)(x).numpy()


def predict_labels(model, images):
    """Predicted class indices (N,) using margin-free inference."""
    return np.argmax(predict_logits(model, images), axis=1)
