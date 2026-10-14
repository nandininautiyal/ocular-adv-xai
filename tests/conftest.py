"""
Shared fixtures: a tiny dual-input model built with the vendored ArcFace layer.

The real checkpoint is 1.7 GB and lives on Kaggle, so tests use a small CNN
with the same contract ([image, label] inputs, ArcFace head with s = 64,
softmax inside the layer). It is trained until its top softmax probability
rounds to 1.0 with top-1 / top-2 logit gaps of ~35-40, the regime the real
97.9 %-accurate model is in.
"""
import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.models.recognition_model import ArcFace  # noqa: E402

tf.config.experimental.enable_op_determinism()

N_CLASSES = 5


def make_images(n=N_CLASSES, seed=0):
    """Distinct synthetic 'eyes': a coloured disk at a different place in each image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256]
    images = []
    for k in range(n):
        img = 0.3 + rng.uniform(-0.05, 0.05, (256, 256, 3))
        disk = (yy - (48 + 40 * k)) ** 2 + (xx - (200 - 30 * k)) ** 2 < 30 ** 2
        img[disk] = np.eye(3)[k % 3] * 0.9 + 0.05
        images.append(img)
    return np.clip(np.stack(images), 0.0, 1.0).astype(np.float32)


def build_tiny_model(n_classes=N_CLASSES):
    image = tf.keras.Input((256, 256, 3))
    x = tf.keras.layers.Conv2D(8, 3, strides=4, padding="same")(image)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=4, padding="same")(x)
    x = tf.keras.layers.Activation("relu", name="last_act")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    label = tf.keras.Input((n_classes,))
    output = ArcFace(n_classes=n_classes, s=64.0, m=0.5)([x, label])
    return tf.keras.Model(inputs=[image, label], outputs=output)


def train_head(model, images, labels, steps=200, lr=3e-3):
    """Fit the Dense + ArcFace head with the ArcFace loss computed from logits.

    The conv layers stay at their initial weights, like the mostly frozen
    backbone of the real model; training them too kills most ReLUs and leaves
    gradients on ~2 % of pixels, which makes every gradient attack look weak.

    Cross-entropy on the layer's softmax output would not work here: in eager
    mode Keras clips probabilities to [1e-7, 1 - 1e-7], the clip has zero
    gradient, and an untrained ArcFace head is far below 1e-7 for the true class.
    """
    arcface = next(layer for layer in model.layers if type(layer).__name__ == "ArcFace")
    dense = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
    features = tf.keras.Model(model.inputs[0], dense.input)(images)
    y = tf.one_hot(labels, arcface.n_classes)
    optimizer = tf.keras.optimizers.Adam(lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            cos = tf.nn.l2_normalize(dense(features), 1) @ tf.nn.l2_normalize(arcface.W, 0)
            target = tf.cos(tf.acos(tf.clip_by_value(cos, -1 + 1e-7, 1 - 1e-7)) + arcface.m)
            logits = arcface.s * (cos * (1 - y) + target * y)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))
        variables = dense.trainable_variables + arcface.trainable_variables
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
    return model


@pytest.fixture(scope="session")
def images():
    return make_images()


@pytest.fixture(scope="session")
def labels():
    return np.arange(N_CLASSES)


@pytest.fixture(scope="session")
def model(images, labels):
    tf.keras.utils.set_random_seed(0)
    return train_head(build_tiny_model(), images, labels)


@pytest.fixture(scope="session")
def zeros():
    return np.zeros((N_CLASSES, N_CLASSES), np.float32)
