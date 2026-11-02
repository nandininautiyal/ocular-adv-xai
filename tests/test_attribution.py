import numpy as np
import pytest
import tensorflow as tf

from src.models.arcface_inference import find_feature_layer, get_margin_free
from src.xai.attribution import XAI_REGISTRY, _eigen_projection, eigencam, layercam


def class_kwargs(name, class_idx):
    return {"class_idx": class_idx} if name == "layercam" else {}


@pytest.mark.parametrize("name", sorted(XAI_REGISTRY))
def test_shapes_and_range(model, images, name):
    fn = XAI_REGISTRY[name]
    single = fn(model, images[0])
    batch = fn(model, images)
    native = fn(model, images, size=None)
    assert single.shape == (256, 256) and single.dtype == np.float32
    assert batch.shape == (5, 256, 256)
    assert native.shape == (5, 16, 16)
    for maps in (single, batch, native):
        assert maps.min() >= 0.0 and maps.max() <= 1.0


@pytest.mark.parametrize("name", sorted(XAI_REGISTRY))
def test_batch_matches_single(model, images, name):
    fn = XAI_REGISTRY[name]
    np.testing.assert_allclose(fn(model, images, size=None)[2], fn(model, images[2], size=None), atol=1e-4)


def test_layercam_is_informative_when_softmax_is_saturated(model, images, zeros):
    # The model's softmax output is exactly 1.0 here (see test_arcface_inference),
    # so its gradient is ~0; maps from the margin-free logit must still vary.
    assert (model.predict([images, zeros], verbose=0).max(axis=1) == 1.0).all()
    maps = layercam(model, images, class_idx=np.arange(5), size=None)
    assert all(np.ptp(m) > 0 for m in maps)


def test_softmax_gradient_vanishes_in_this_regime(model, images, zeros):
    # Documents why LayerCAM must not use model.output: d(prob)/d(features) ~ 0.
    feature = tf.keras.Model(model.inputs, [model.get_layer("last_act").output, model.output])
    with tf.GradientTape() as tape:
        acts, probs = feature([images[:1], zeros[:1]])
        score = probs[0, 0]
    assert np.abs(tape.gradient(score, acts).numpy()).max() < 1e-6


def test_layercam_class_argument(model, images):
    default = layercam(model, images[3], size=None)          # predicted class (3)
    explicit = layercam(model, images[3], class_idx=3, size=None)
    other = layercam(model, images[3], class_idx=0, size=None)
    np.testing.assert_allclose(default, explicit, atol=1e-6)
    assert not np.allclose(explicit, other)


def test_eigen_projection_sign_is_aligned_with_activation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        acts = np.maximum(rng.normal(size=(8, 8, 32)), 0.0)
        acts[2:5, 2:5] += 3.0                                 # a salient blob
        proj = _eigen_projection(acts)
        assert np.dot(proj.ravel(), acts.sum(-1).ravel()) >= 0
        assert proj[3, 3] > np.median(proj)                    # blob is at the high end


def test_eigencam_deterministic(model, images):
    np.testing.assert_array_equal(eigencam(model, images[0]), eigencam(model, images[0]))


@pytest.mark.parametrize("name", sorted(XAI_REGISTRY))
def test_return_raw_is_opt_in_and_keeps_the_maps(model, images, name):
    fn, kwargs = XAI_REGISTRY[name], class_kwargs(name, np.arange(5))
    plain = fn(model, images, size=None, **kwargs)
    maps, raw_sum, raw_max = fn(model, images, size=None, return_raw=True, **kwargs)
    np.testing.assert_array_equal(maps, plain)
    assert raw_sum.shape == raw_max.shape == (5,) and raw_sum.dtype == np.float64
    assert (raw_max > 0).all() and (raw_sum >= raw_max).all()
    # magnitudes always describe the native map, whatever the output size
    big, big_sum, big_max = fn(model, images, return_raw=True, **kwargs)
    np.testing.assert_array_equal(big, fn(model, images, **kwargs))
    np.testing.assert_array_equal(big_sum, raw_sum)
    np.testing.assert_array_equal(big_max, raw_max)
    # one image: plain floats, equal to that image's entry of the batch
    one, one_sum, one_max = fn(model, images[2], size=None, return_raw=True, **class_kwargs(name, 2))
    assert one.shape == (16, 16) and isinstance(one_sum, float) and isinstance(one_max, float)
    assert one_sum == pytest.approx(raw_sum[2], rel=1e-4) and one_max == pytest.approx(raw_max[2], rel=1e-4)


def test_raw_magnitudes_are_those_of_the_unscaled_maps(model, images):
    net = get_margin_free(model, find_feature_layer(model))
    x = tf.constant(images)
    features, _ = net(x)
    projections = np.stack([_eigen_projection(a) for a in features.numpy()])
    _, raw_sum, raw_max = eigencam(model, images, size=None, return_raw=True)
    np.testing.assert_allclose(raw_sum, projections.sum(axis=(1, 2)), rtol=1e-6)
    np.testing.assert_allclose(raw_max, projections.max(axis=(1, 2)), rtol=1e-6)

    with tf.GradientTape() as tape:                      # LayerCAM of class k for image k
        features, logits = net(x)
        score = tf.reduce_sum(tf.gather(logits, np.arange(5), axis=1, batch_dims=1))
    grads = tape.gradient(score, features)
    cams = tf.nn.relu(tf.reduce_sum(tf.nn.relu(grads) * features, axis=-1)).numpy()
    _, raw_sum, raw_max = layercam(model, images, class_idx=np.arange(5), size=None, return_raw=True)
    np.testing.assert_allclose(raw_sum, cams.sum(axis=(1, 2)), rtol=1e-5)
    np.testing.assert_allclose(raw_max, cams.max(axis=(1, 2)), rtol=1e-5)
