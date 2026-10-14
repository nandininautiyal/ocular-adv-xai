import numpy as np
import tensorflow as tf

from src.models.arcface_inference import (find_feature_layer, get_margin_free, get_n_classes,
                                          predict_labels, predict_logits)
from src.models.recognition_model import ArcFace


def test_softmax_of_margin_free_logits_equals_model_output(model, images, zeros):
    probs = model.predict([images, zeros], verbose=0)
    logits = predict_logits(model, images)
    np.testing.assert_allclose(tf.nn.softmax(logits).numpy(), probs, atol=1e-5)


def test_fixture_is_in_the_saturated_regime(model, images, zeros, labels):
    # Like the real checkpoint: the top probability is exactly 1.0 in float32.
    probs = model.predict([images, zeros], verbose=0)
    assert (probs.max(axis=1) == 1.0).all()
    np.testing.assert_array_equal(predict_labels(model, images), labels)


def test_single_image_and_batch_agree(model, images):
    np.testing.assert_allclose(predict_logits(model, images[1]), predict_logits(model, images)[1:2],
                               rtol=1e-5, atol=1e-4)


def test_n_classes_and_feature_layer(model):
    assert get_n_classes(model) == 5
    assert find_feature_layer(model) == "last_act"


def test_cache_returns_same_object(model):
    assert get_margin_free(model) is get_margin_free(model)
    assert get_margin_free(model, "last_act") is not get_margin_free(model)


def test_xception_feature_layer_is_post_relu():
    # Same graph as the vendored create_model(), without downloading ImageNet weights.
    base = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=(256, 256, 3))
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    label = tf.keras.Input((3,))
    model = tf.keras.Model([base.input, label], ArcFace(n_classes=3)([x, label]))
    assert find_feature_layer(model) == "block14_sepconv2_act"
