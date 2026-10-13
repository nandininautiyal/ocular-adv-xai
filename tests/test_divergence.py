import numpy as np
import pytest

from src.compare.divergence import (METRICS, centre_of_mass_shift, compare_heatmaps, js_divergence,
                                    kl_divergence, spearman, top_k_intersection)


def blob(cy, cx, size=8):
    yy, xx = np.mgrid[0:size, 0:size]
    m = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)
    return (m / m.max()).astype(np.float32)


def test_identical_maps():
    m = blob(3, 3)
    r = compare_heatmaps(m, m)
    assert r["spearman"] == pytest.approx(1.0)
    assert r["top10_intersection"] == 1.0
    assert r["js"] == pytest.approx(0.0, abs=1e-12)
    assert r["cosine"] == pytest.approx(1.0)
    assert r["l2"] == 0.0
    assert r["global_ssim"] == pytest.approx(1.0)
    assert r["com_shift"] == 0.0
    assert set(r) == set(METRICS)


def test_shifted_attention_is_detected():
    a, b = blob(2, 2), blob(5, 6)
    r = compare_heatmaps(a, b)
    assert r["spearman"] < 0.5 and r["top10_intersection"] < 0.5
    assert 0.0 < r["js"] <= 1.0
    assert r["com_shift"] > 0.2


def test_constant_map_gives_no_nan():
    assert spearman(np.zeros((8, 8)), blob(3, 3)) == 0.0
    assert np.isfinite(kl_divergence(blob(3, 3), np.zeros((8, 8))))
    assert centre_of_mass_shift(np.zeros((8, 8)), np.zeros((8, 8))) == 0.0


def test_js_is_symmetric_and_bounded():
    a, b = blob(1, 1), np.zeros((8, 8), np.float32)
    b[7, 7] = 1.0
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))
    assert js_divergence(a, b) <= 1.0


def test_top_k_counts_at_least_one_location():
    a, b = blob(3, 3), blob(3, 3)
    assert top_k_intersection(a, b, k=0.001) == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare_heatmaps(np.zeros((8, 8)), np.zeros((16, 16)))
