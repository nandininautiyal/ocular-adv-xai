"""Sampling baselines and agreement maps (src/xai/part_baselines.py): KernelSHAP and LIME on synthetic games
with known answers, the patch players and their aggregation to parts, the 256 x 256 maps and their
Jensen-Shannon divergence, and the estimators on the tiny conftest model."""
import math

import numpy as np
import pytest

from src.compare.divergence import js_divergence as js_reference
from src.xai import part_baselines as pb
from src.xai.shapley import asa, exact_shapley, group_masks

CLASSES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")
GROUPS = {"iris": ("iris",), "sclera": ("sclera", "caruncle"), "lid": ("lid",), "brow": ("brow",), "skin": ("skin",)}


def additive_game(weights, constant=2.0):
    """v(S) = constant + sum of the weights of the players in S (index = bitmask of kept players)."""
    n = len(weights)
    return np.array([constant + sum(weights[p] for p in range(n) if (k >> p) & 1) for k in range(1 << n)])


def eye_labels(k=0):
    """conftest.make_images anatomy of image k: iris disk, sclera ring, brow rows, unknown bottom rows."""
    yy, xx = np.mgrid[0:256, 0:256]
    r2 = (yy - (48 + 40 * k)) ** 2 + (xx - (200 - 30 * k)) ** 2
    labels = np.zeros((256, 256), np.uint8)
    labels[r2 < 45 ** 2] = 2
    labels[r2 < 30 ** 2] = 1
    labels[:16] = 5
    labels[16:24] = 3
    labels[240:] = 255
    return labels


# ---------------------------------------------------------------------------
# KernelSHAP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_kernelshap_with_every_coalition_is_exact_shapley(n):
    """With all 2^P coalitions every coalition gets its exact Shapley kernel weight, and the constrained
    weighted regression gives the exact Shapley values (Lundberg & Lee 2017)."""
    for seed in range(3):
        v = np.random.default_rng(seed).normal(size=1 << n)
        np.testing.assert_allclose(pb.from_game("kernelshap", v, 1 << n, seed), exact_shapley(v), atol=1e-10)
        np.testing.assert_allclose(pb.from_game("kernelshap", v, 10 * (1 << n), seed), exact_shapley(v), atol=1e-10)
    plan = pb.kernelshap_design(5, 32, 0)
    assert plan["n_evaluated"] == 32 and len(plan["coalitions"]) == 30
    sizes = plan["coalitions"].sum(axis=1)
    expected = np.array([pb.kernel_weight(5, s) for s in sizes])
    np.testing.assert_allclose(plan["weights"], expected / expected.sum())


@pytest.mark.parametrize("budget", [6, 10, 16, 24, 31])
def test_kernelshap_design_samples_without_replacement(budget):
    plan = pb.kernelshap_design(5, budget, seed=7)
    z = plan["coalitions"]
    assert plan["n_evaluated"] == budget == len(z) + 2
    assert len(np.unique(z, axis=0)) == len(z)                         # no duplicates
    assert not (z.all(axis=1) | ~z.any(axis=1)).any()                   # neither the full nor the empty coalition
    assert (plan["weights"] > 0).all() and plan["weights"].sum() == pytest.approx(1.0)
    again = pb.kernelshap_design(5, budget, seed=7)
    np.testing.assert_array_equal(again["coalitions"], z)
    np.testing.assert_array_equal(again["weights"], plan["weights"])
    # a size is sampled together with its complement, as in shap
    rows = {tuple(r) for r in z}
    paired = sum(tuple(~np.array(r)) in rows for r in rows)
    assert paired >= len(rows) - 1                                      # at most the last draw lacks its complement


def test_kernelshap_follows_shaps_enumeration_rule():
    """shap enumerates a size pair completely when budget x its kernel share covers it: with 22 sampled
    coalitions of 5 players, sizes 1 and 4 (10 coalitions, 60 % of the mass) are enumerated; with 14 they are not."""
    full = pb.kernelshap_design(5, 24, seed=0)
    assert full["n_enumerated"] == 10
    enumerated = full["coalitions"][:10].sum(axis=1)
    assert sorted(enumerated) == [1] * 5 + [4] * 5
    np.testing.assert_allclose(full["weights"][:10].sum(), 0.6)        # their exact share of the kernel mass
    rest = full["weights"][10:]
    np.testing.assert_allclose(rest, rest[0])                           # one partly sampled size pair: equal weights
    assert pb.kernelshap_design(5, 16, seed=0)["n_enumerated"] == 0
    # 64 patch players, 500 coalitions: nothing is complete, everything is sampled by kernel mass
    big = pb.kernelshap_design(64, 500, seed=1)
    assert big["n_enumerated"] == 0 and big["n_evaluated"] == 500 and pb.n_distinct(big["coalitions"]) == 498
    sizes = np.minimum(big["coalitions"].sum(axis=1), 64 - big["coalitions"].sum(axis=1))
    assert np.mean(sizes <= 3) > 0.3                                    # the kernel favours small and large coalitions


@pytest.mark.parametrize("budget", [6, 9, 16, 20])
def test_kernelshap_efficiency_and_additive_games(budget):
    rng = np.random.default_rng(budget)
    v = rng.normal(size=32)
    phi = pb.from_game("kernelshap", v, budget, seed=1)
    assert phi.sum() == pytest.approx(v[-1] - v[0], abs=1e-12)          # efficiency is enforced exactly
    if budget >= 16:          # enough complement pairs to identify 5 players: a linear game is fitted exactly
        weights = np.array([1.5, -2.0, 0.25, 3.0, -0.5])
        np.testing.assert_allclose(pb.from_game("kernelshap", additive_game(weights), budget, seed=2), weights,
                                   atol=1e-10)


def test_estimators_skip_absent_players():
    weights = np.array([1.0, 2.0, 0.0, -1.0, 0.5])
    v = additive_game(weights)
    present = np.array([True, True, False, True, True])
    for estimator in pb.ESTIMATORS:
        phi = pb.from_game(estimator, v, 16, seed=0, present=present)
        assert phi[2] == 0.0
    np.testing.assert_allclose(pb.from_game("kernelshap", v, 16, seed=0, present=present), weights, atol=1e-10)
    np.testing.assert_array_equal(pb.to_indices(np.array([[True, False, True]]), [0, 3, 4]), [1 + 16])


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def test_lime_recovers_the_weights_of_an_additive_game():
    weights = np.array([1.0, -2.0, 3.0, 0.5, 0.0])
    v = additive_game(weights)
    z = pb.lime_design(5, 16, seed=0)
    assert z[0].all() and z.shape == (16, 5)
    values = v[pb.to_indices(z, range(5))]
    np.testing.assert_allclose(pb.lime_fit(z, values, alpha=1e-9)["coef"], weights, atol=1e-6)
    fitted = pb.lime_fit(z, values)          # lime's default ridge (alpha = 1) shrinks: order and signs survive
    assert np.argsort(fitted["coef"]).tolist() == np.argsort(weights).tolist()
    assert (np.sign(fitted["coef"][:4]) == np.sign(weights[:4])).all()
    assert (np.abs(fitted["coef"]) <= np.abs(weights) + 0.2).all()
    np.testing.assert_allclose(pb.from_game("lime", v, 400, seed=1), weights, atol=0.1)


def test_lime_kernel_and_ridge_match_lime_and_sklearn():
    """lime_image: cosine distance to the full coalition, kernel sqrt(exp(-d^2 / 0.25^2)), Ridge(alpha=1) with
    sample weights and an intercept."""
    z = pb.lime_design(6, 40, seed=3)
    z[5] = False                                                        # an empty coalition: distance 1
    d = 1.0 - (z.astype(float) @ np.ones(6)) / (np.sqrt(np.maximum(z.sum(axis=1), 1e-300)) * np.sqrt(6))
    d[~z.any(axis=1)] = 1.0
    np.testing.assert_allclose(pb.lime_weights(z), np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2)))
    sklearn = pytest.importorskip("sklearn.linear_model")
    y = np.random.default_rng(0).normal(size=40)
    ridge = sklearn.Ridge(alpha=1.0, fit_intercept=True).fit(z.astype(float), y, sample_weight=pb.lime_weights(z))
    fitted = pb.lime_fit(z, y)
    np.testing.assert_allclose(fitted["coef"], ridge.coef_, atol=1e-8)
    assert fitted["intercept"] == pytest.approx(ridge.intercept_, abs=1e-8)


# ---------------------------------------------------------------------------
# Players, aggregation and maps
# ---------------------------------------------------------------------------

def test_patch_players_and_aggregation_conserve_totals():
    patches = pb.patch_map((256, 256), 32)
    assert pb.n_players(patches) == 64 and np.bincount(patches.ravel()).tolist() == [32 * 32] * 64
    assert patches[0, 31] == 0 and patches[0, 32] == 1 and patches[32, 0] == 8
    assert pb.n_players(pb.patch_map((256, 256), 100)) == 9                 # edge patches are smaller
    masks = group_masks(eye_labels(), CLASSES, GROUPS)
    shares = pb.patch_part_shares(patches, masks)
    assert shares.shape == (5, 64) and (shares >= 0).all()
    unknown = np.bincount(patches[eye_labels() == 255], minlength=64) / (32 * 32)
    np.testing.assert_allclose(shares.sum(axis=0) + unknown, 1.0)        # every pixel in one part or unknown
    w = np.random.default_rng(0).normal(size=64)
    parts = pb.aggregate_to_parts(w, shares)
    assert parts.sum() == pytest.approx(w.sum() - (w * unknown).sum())   # the total moves only by the unknown share
    covered = group_masks(np.where(eye_labels() == 255, 0, eye_labels()).astype(np.uint8), CLASSES, GROUPS)
    assert pb.aggregate_to_parts(w, pb.patch_part_shares(patches, covered)).sum() == pytest.approx(w.sum())
    keep = pb.player_keep(patches, np.arange(64) == 9)
    assert keep.sum() == 32 * 32 and keep[32:64, 32:64].all()
    player_map = pb.player_map_from_masks(masks)
    assert (player_map[eye_labels() == 255] == -1).all()
    assert pb.player_keep(player_map, np.zeros(5, bool))[eye_labels() == 255].all()   # pixels in no player stay


def test_maps_and_jensen_shannon():
    masks = group_masks(eye_labels(), CLASSES, GROUPS)
    player_map = pb.player_map_from_masks(masks)
    values = np.array([2.0, -1.0, np.nan, 0.5, 1.0])
    m = pb.player_value_map(values, player_map)
    assert m.shape == (256, 256) and (m >= 0).all()
    np.testing.assert_allclose(m.sum(), 3.5)                    # the positive values, each spread over its part
    np.testing.assert_allclose(m[masks[0]], 2.0 / masks[0].sum())
    assert (m[masks[1]] == 0).all() and (m[masks[2]] == 0).all()
    up = pb.upsample_map(np.arange(64, dtype=np.float32).reshape(8, 8) - 10)
    assert up.shape == (256, 256) and up.min() == 0.0
    rng = np.random.default_rng(1)
    a, b = rng.uniform(size=(256, 256)), rng.uniform(size=(256, 256))
    assert pb.js_divergence(a, b) == pytest.approx(js_reference(a, b), abs=1e-6)
    assert pb.js_divergence(a, 3 * a) == pytest.approx(0.0, abs=1e-12)
    left, right = np.zeros((4, 4)), np.zeros((4, 4))
    left[:, :2], right[:, 2:] = 1.0, 1.0
    assert pb.js_divergence(left, right) == pytest.approx(1.0)
    js = pb.pairwise_js({"a": a, "b": b, "zero": np.zeros((256, 256)), "none": None})
    assert list(js) == [("a", "b"), ("a", "zero"), ("a", "none"), ("b", "zero"), ("b", "none"), ("zero", "none")]
    assert np.isnan(js[("a", "zero")]) and np.isnan(js[("b", "none")]) and 0 < js[("a", "b")] < 1


# ---------------------------------------------------------------------------
# On the tiny model
# ---------------------------------------------------------------------------

def test_estimators_on_a_model(model, images, labels):
    """KernelSHAP with every coalition reproduces exact ASA on the model; sampled estimators evaluate each
    distinct coalition once."""
    labels_map = eye_labels(0)
    masks = group_masks(labels_map, CLASSES, GROUPS)
    player_map = pb.player_map_from_masks(masks)
    exact = asa(model, images[0], labels_map, CLASSES, GROUPS, target=int(labels[0]), method="blur", score="margin")
    full = pb.estimate(model, images[0], player_map, "kernelshap", 32, "blur", "margin", target=int(labels[0]))
    assert full["n_forward"] == 32
    np.testing.assert_allclose(full["phi"], exact["phi"], atol=1e-3)
    sampled = pb.estimate(model, images[0], player_map, "kernelshap", 16, "blur", "margin", target=int(labels[0]),
                          seed=4)
    assert sampled["n_forward"] == 16
    assert sampled["phi"].sum() == pytest.approx(exact["v_full"] - exact["v_empty"], abs=1e-3)
    np.testing.assert_allclose(sampled["phi"], pb.from_game("kernelshap", exact["values"], 16, seed=4), atol=1e-3)
    lime = pb.estimate(model, images[0], player_map, "lime", 16, "mean", "margin", target=int(labels[0]), seed=4)
    plan = pb.lime_design(5, 16, seed=4)
    assert lime["n_forward"] == pb.n_distinct(plan) <= 16
    z = np.array([[True, False, True], [True, False, True], [False, False, False]])
    outputs, n = pb.sampled_outputs(model, images[0], pb.patch_map((256, 256), 128)[:, :], np.pad(z, ((0, 0), (0, 1))),
                                    "mean", batch_size=2)
    assert n == 2 and outputs.shape[0] == 3
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert math.isfinite(float(outputs.sum()))
