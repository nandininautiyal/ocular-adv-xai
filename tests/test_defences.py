import numpy as np
import pytest

from src.defences.inference import (DEFENCE_REGISTRY, bit_depth_reduction, jpeg_compress, make_defence,
                                    parse_defence)
from src.defences.training import make_preprocessing_function, patch_augmentation, strip_augmentation


@pytest.mark.parametrize("name", sorted(DEFENCE_REGISTRY))
def test_contract(images, name):
    single = DEFENCE_REGISTRY[name](images[0])
    batch = DEFENCE_REGISTRY[name](images)
    assert single.shape == (256, 256, 3) and single.dtype == np.float32
    assert batch.shape == images.shape
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    np.testing.assert_allclose(batch[0], single, atol=1e-6)


def test_registry_has_only_inference_time_defences():
    assert set(DEFENCE_REGISTRY) == {"jpeg", "bit_depth"}


def test_jpeg_high_quality_is_close_to_identity():
    # Smooth image: JPEG's chroma subsampling legitimately alters per-pixel colour noise.
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    img = np.stack([*np.meshgrid(ramp, ramp), np.full((256, 256), 0.5, np.float32)], axis=-1)
    assert np.abs(jpeg_compress(img, quality=100) - img).mean() < 0.01


def test_jpeg_rounds_instead_of_truncating():
    img = np.full((16, 16, 3), 100.6 / 255.0, np.float32)
    assert np.allclose(jpeg_compress(img, quality=100), 101 / 255.0, atol=1.5 / 255.0)


def test_bit_depth_levels(images):
    out = bit_depth_reduction(images[0], bits=2)
    assert len(np.unique(out)) <= 4
    with pytest.raises(ValueError):
        bit_depth_reduction(images[0], bits=0)


def test_make_defence_fixes_parameters(images):
    jpeg50 = make_defence("jpeg", quality=50)
    np.testing.assert_array_equal(jpeg50(images), jpeg_compress(images, quality=50))
    assert not np.array_equal(jpeg50(images), jpeg_compress(images))
    assert jpeg50(images[0]).shape == (256, 256, 3) and jpeg50.label == "jpeg:quality=50"
    bits3 = make_defence("bit_depth", bits=np.int64(3))
    np.testing.assert_array_equal(bits3(images), bit_depth_reduction(images, bits=3))
    assert bits3.label == "bit_depth:bits=3"


def test_make_defence_defaults_match_the_registry(images):
    for name, fn in DEFENCE_REGISTRY.items():
        defence = make_defence(name)
        assert defence.label == name
        np.testing.assert_array_equal(defence(images), fn(images))
    # spelling out a default keeps the registry name, so rows stay comparable with older runs
    assert make_defence("jpeg", quality=75).label == "jpeg"
    assert make_defence("bit_depth", bits=4).label == "bit_depth"


def test_make_defence_rejects_bad_parameters():
    with pytest.raises(KeyError):
        make_defence("blur")
    with pytest.raises(TypeError):
        make_defence("jpeg", bits=3)
    for bad in (0, 101, 50.5, True, "50", float("nan")):
        with pytest.raises(ValueError):
            make_defence("jpeg", quality=bad)
    with pytest.raises(ValueError):
        make_defence("bit_depth", bits=9)


def test_parse_defence(images):
    for spec, label in [("jpeg", "jpeg"), ("jpeg:quality=50", "jpeg:quality=50"),
                        (" bit_depth : bits = 3 ", "bit_depth:bits=3"), ("bit_depth:bits=4", "bit_depth")]:
        defence = parse_defence(spec)
        assert defence.label == label and parse_defence(defence.label).label == label
    np.testing.assert_array_equal(parse_defence("jpeg:quality=50")(images[0]),
                                  jpeg_compress(images[0], quality=50))
    for bad in ("jpeg:quality", "jpeg:quality=high", "jpeg:=50", "jpeg:quality=0"):
        with pytest.raises(ValueError):
            parse_defence(bad)
    with pytest.raises(KeyError):
        parse_defence("blur:sigma=1")


def test_strip_augmentation_matches_notebook_protocol(images):
    rng = np.random.default_rng(0)
    original = images[0].copy()
    out = strip_augmentation(images[0], rng, strip_size=10, p_horizontal=1.0, p_vertical=0.0)
    np.testing.assert_array_equal(images[0], original)            # input untouched
    zero_rows = np.where((out == 0).all(axis=(1, 2)))[0]
    assert len(zero_rows) == 10 and np.all(np.diff(zero_rows) == 1)
    assert (out == 0).sum() == 10 * 256 * 3


def test_strip_augmentation_probability():
    rng = np.random.default_rng(0)
    img = np.ones((32, 32, 3), np.float32)
    changed = [strip_augmentation(img, rng).min() == 0 for _ in range(4000)]
    assert abs(np.mean(changed) - 0.30) < 0.03


def test_patch_augmentation(images):
    out = patch_augmentation(images[0], np.random.default_rng(1), patch_size=32, p=1.0)
    assert (out == 0).all(axis=2).sum() == 32 * 32


def test_preprocessing_function():
    fn = make_preprocessing_function("strip", 5, seed=0)
    assert fn(np.ones((64, 64, 3))).shape == (64, 64, 3)
    with pytest.raises(ValueError):
        make_preprocessing_function("stripe", 5)
