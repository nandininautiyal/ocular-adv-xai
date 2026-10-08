"""
src/xai/attribution.py

LayerCAM and Eigen-CAM heatmap generation for the Xception + ArcFace
dual-input recognition model.

All functions are pure: they accept a model, a preprocessed image, and
return a heatmap ndarray of shape (H, W) with values in [0, 1].

Image contract
--------------
- dtype : float32
- shape : (256, 256, 3)
- range : [0, 1]

Model contract
--------------
The model has two inputs: [image_input, label_input].
During inference / explanation the label input is a zero vector of shape
(1, n_classes).  All helpers in this module handle that automatically.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_n_classes(model):
    """Return n_classes from the model's label-input shape."""
    # model.input is [image_input, label_input]
    label_shape = model.input[1].shape  # (None, n_classes)
    return int(label_shape[-1])


def _dummy_labels(n_classes, batch=1):
    """Zero label vector used during inference (no ArcFace margin)."""
    return np.zeros((batch, n_classes), dtype="float32")


def _find_last_conv_layer(model):
    """Walk the model and return the name of the last Conv2D layer.

    In the Xception + ArcFace architecture this is the final depthwise or
    pointwise convolution inside the Xception base, which is the correct
    target for class-activation-map methods.
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D,
                              tf.keras.layers.SeparableConv2D)):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D / DepthwiseConv2D / SeparableConv2D layer found in model")
    return last_conv


def _build_grad_model(model, target_layer_name):
    """Build a sub-model that outputs [conv_outputs, predictions].

    This avoids re-running a forward pass twice and lets GradientTape
    watch the conv output tensor.
    """
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )
    return grad_model


# ---------------------------------------------------------------------------
# LayerCAM  (Jiang et al., 2021)
# ---------------------------------------------------------------------------

def layercam(model, image, target_layer=None, class_idx=None):
    """Generate a LayerCAM heatmap for *image*.

    Parameters
    ----------
    model : tf.keras.Model
        The dual-input Xception + ArcFace model.
    image : np.ndarray, shape (256, 256, 3), dtype float32, range [0, 1]
        A single preprocessed ocular image.
    target_layer : str or None
        Name of the convolutional layer to explain.  If *None*, the last
        Conv2D layer in the model is used automatically.
    class_idx : int or None
        Class index to explain.  If *None*, the predicted class is used.

    Returns
    -------
    heatmap : np.ndarray, shape (256, 256), dtype float32, range [0, 1]
        The LayerCAM attribution map, resized to the input resolution.
    """
    if target_layer is None:
        target_layer = _find_last_conv_layer(model)

    n_classes = _get_n_classes(model)
    img_tensor = tf.cast(tf.expand_dims(image, 0), tf.float32)
    dummy = tf.constant(_dummy_labels(n_classes))

    grad_model = _build_grad_model(model, target_layer)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_tensor, dummy])
        if class_idx is None:
            class_idx = tf.argmax(predictions[0]).numpy()
        class_score = predictions[0, class_idx]

    grads = tape.gradient(class_score, conv_outputs)          # (1, h, w, c)
    conv_outputs = conv_outputs[0]                            # (h, w, c)
    grads = grads[0]                                          # (h, w, c)

    # LayerCAM: element-wise product of ReLU(gradients) and activations,
    # then sum over channels.
    weights = tf.nn.relu(grads)                               # (h, w, c)
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)      # (h, w)
    cam = tf.nn.relu(cam)                                     # discard negatives

    # Resize to input spatial dimensions
    cam = tf.image.resize(cam[..., tf.newaxis], (256, 256))[..., 0]

    # Normalise to [0, 1]
    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max

    return cam.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Eigen-CAM  (Muhammad & Yeasin, 2020)
# ---------------------------------------------------------------------------

def eigencam(model, image, target_layer=None):
    """Generate an Eigen-CAM heatmap for *image*.

    Eigen-CAM is gradient-free: it computes the first principal component of
    the activation map at the target convolutional layer.

    Parameters
    ----------
    model : tf.keras.Model
        The dual-input Xception + ArcFace model.
    image : np.ndarray, shape (256, 256, 3), dtype float32, range [0, 1]
        A single preprocessed ocular image.
    target_layer : str or None
        Name of the convolutional layer to explain.  If *None*, the last
        Conv2D layer in the model is used automatically.

    Returns
    -------
    heatmap : np.ndarray, shape (256, 256), dtype float32, range [0, 1]
        The Eigen-CAM attribution map, resized to the input resolution.
    """
    if target_layer is None:
        target_layer = _find_last_conv_layer(model)

    n_classes = _get_n_classes(model)
    img_tensor = tf.cast(tf.expand_dims(image, 0), tf.float32)
    dummy = tf.constant(_dummy_labels(n_classes))

    # Forward pass to get conv activations only
    activation_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(target_layer).output,
    )
    activations = activation_model([img_tensor, dummy])       # (1, h, w, c)
    activations = activations[0].numpy()                      # (h, w, c)

    h, w, c = activations.shape
    # Reshape to 2-D matrix: (h*w, c) — each spatial location is a sample
    reshaped = activations.reshape(h * w, c)

    # SVD → first right-singular vector is the first principal component
    # Project activations onto it to get a (h*w,) saliency vector
    _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
    projection = reshaped @ Vt[0]                             # (h*w,)
    cam = projection.reshape(h, w)

    # Normalise to [0, 1]
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    # Resize to input resolution
    cam = tf.image.resize(cam[..., np.newaxis], (256, 256))[..., 0]

    return cam.numpy().astype(np.float32)
