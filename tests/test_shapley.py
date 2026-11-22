"""Anatomical Shapley Attribution (src/xai/shapley.py): the game arithmetic on synthetic games,
the hiding baselines, and ASA on the tiny conftest model."""
import itertools
import math

import numpy as np
import pytest

from src.compare.regions import grids_from_labels, region_mass_named
from src.eval.verification import embed
from src.models.arcface_inference import predict_logits
from src.xai.shapley import (DEFAULT_GROUPS, METHODS, asa, asa_order, canonical_indices, coalition_images,
                             coalition_keep,
                             coalition_outputs, deletion_curve, deletion_from_values, exact_shapley,
                             format_groups, group_masks, groups_from_config, hide, make_cache,
                             normalise_groups, order_from_scores, parse_groups, part_masses, rank_agreement,
                             score_values, shapley_interactions, shift_part)

CLASSES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")
GROUP_NAMES = ("iris", "sclera", "lid", "brow", "skin")


def eye_labels(n):
    """Label maps for conftest.make_images: the disk is the iris, a ring around it the sclera,
    the top 16 rows the brow, the bottom 16 rows unknown (255); no lid, no caruncle."""
    yy, xx = np.mgrid[0:256, 0:256]
    out = []
    for k in range(n):
        r2 = (yy - (48 + 40 * k)) ** 2 + (xx - (200 - 30 * k)) ** 2
        labels = np.zeros((256, 256), np.uint8)
        labels[r2 < 45 ** 2] = 2
        labels[r2 < 30 ** 2] = 1
        labels[:16] = 5
        labels[240:] = 255
        out.append(labels)
    return np.stack(out)


def random_game(n, seed=0):
    return np.random.default_rng(seed).normal(size=1 << n)


def game_from_dividends(dividends, n):
    """v(S) = sum of the Harsanyi dividends d_T over T subset of S."""
    v = np.zeros(1 << n)
    for s in range(1 << n):
        v[s] = sum(d for t, d in dividends.items() if t & s == t)
    return v


def members(t, n):
    return [p for p in range(n) if (t >> p) & 1]


# ---------------------------------------------------------------------------
# Game arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_efficiency(n):
    v = random_game(n, seed=n)
    phi = exact_shapley(v)
    assert phi.shape == (n,) and phi.dtype == np.float64
    assert phi.sum() == pytest.approx(v[-1] - v[0], rel=1e-12, abs=1e-12)


def test_matches_the_permutation_definition():
    n = 4
    v = random_game(n, seed=3)
    expected = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        s = 0
        for p in perm:
            expected[p] += v[s | (1 << p)] - v[s]
            s |= 1 << p
    expected /= math.factorial(n)
    np.testing.assert_allclose(exact_shapley(v), expected, rtol=1e-12, atol=1e-12)


def test_symmetry():
    n, (i, j) = 4, (1, 3)
    base = random_game(n, seed=4)

    def swap(s):
        bi, bj = (s >> i) & 1, (s >> j) & 1
        return (s & ~((1 << i) | (1 << j))) | (bj << i) | (bi << j)
    v = np.array([base[s] + base[swap(s)] for s in range(1 << n)])     # symmetric in i and j
    phi = exact_shapley(v)
    assert phi[i] == pytest.approx(phi[j], rel=1e-12, abs=1e-12)


def test_null_player():
    n, null = 5, 2
    base = random_game(n, seed=5)
    v = np.array([base[s & ~(1 << null)] for s in range(1 << n)])      # player 2 never changes v
    phi, inter = exact_shapley(v), shapley_interactions(v)
    assert phi[null] == 0.0
    assert np.all(inter[null] == 0.0) and np.all(inter[:, null] == 0.0)
    assert phi.sum() == pytest.approx(v[-1] - v[0], rel=1e-12, abs=1e-12)


def test_linearity():
    n = 5
    v1, v2 = random_game(n, seed=6), random_game(n, seed=7)
    np.testing.assert_allclose(exact_shapley(2.5 * v1 - 0.75 * v2),
                               2.5 * exact_shapley(v1) - 0.75 * exact_shapley(v2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shapley_interactions(2.5 * v1 - 0.75 * v2),
                               2.5 * shapley_interactions(v1) - 0.75 * shapley_interactions(v2),
                               rtol=1e-12, atol=1e-12)


def test_additive_game_recovers_its_weights_and_has_no_interactions():
    n = 5
    weights = np.array([3.0, -1.5, 0.25, 7.0, -2.0])
    v = np.array([4.0 + sum(weights[p] for p in members(s, n)) for s in range(1 << n)])
    np.testing.assert_allclose(exact_shapley(v), weights, rtol=0, atol=1e-12)
    inter = shapley_interactions(v)
    off = ~np.eye(n, dtype=bool)
    assert np.abs(inter[off]).max() < 1e-12
    np.testing.assert_allclose(np.diag(inter), weights, atol=1e-12)


def test_pair_game():
    n = 4
    v = np.array([5.0 if (s & 0b0101) == 0b0101 else 0.0 for s in range(1 << n)])     # needs players 0 and 2
    np.testing.assert_allclose(exact_shapley(v), [2.5, 0, 2.5, 0], atol=1e-12)
    inter = shapley_interactions(v)
    assert inter[0, 2] == pytest.approx(5.0) and inter[2, 0] == pytest.approx(5.0)
    assert np.abs(inter[np.triu_indices(n, 1)][[0, 2, 3, 4, 5]]).max() < 1e-12


def test_harsanyi_dividends():
    """phi_p = sum over T containing p of d_T / |T|; I_pq = sum over T containing p and q of
    d_T / (|T| - 1) (Grabisch 1997)."""
    n = 5
    rng = np.random.default_rng(8)
    dividends = {t: rng.normal() for t in range(1, 1 << n)}
    v = game_from_dividends(dividends, n)
    phi = np.array([sum(d / len(members(t, n)) for t, d in dividends.items() if (t >> p) & 1) for p in range(n)])
    np.testing.assert_allclose(exact_shapley(v), phi, rtol=1e-10, atol=1e-12)
    inter = shapley_interactions(v)
    for p, q in itertools.combinations(range(n), 2):
        expected = sum(d / (len(members(t, n)) - 1) for t, d in dividends.items()
                       if (t >> p) & 1 and (t >> q) & 1)
        assert inter[p, q] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    np.testing.assert_array_equal(inter, inter.T)


def test_game_shape_is_checked():
    for bad in ([1.0], [1.0, 2.0, 3.0], np.zeros((2, 2))):
        with pytest.raises(ValueError):
            exact_shapley(bad)


def test_canonical_indices():
    canon = canonical_indices(3, [True, False, True])
    assert canon.tolist() == [0, 1, 0, 1, 4, 5, 4, 5]


# ---------------------------------------------------------------------------
# Groups, hiding and coalitions
# ---------------------------------------------------------------------------

def test_groups_parse_and_validate():
    groups = normalise_groups(DEFAULT_GROUPS)
    assert tuple(n for n, _ in groups) == GROUP_NAMES and groups[1] == ("sclera", ("sclera", "caruncle"))
    assert parse_groups(format_groups(DEFAULT_GROUPS)) == groups
    assert groups_from_config({"anatomy": {"part_groups": {"iris": ["iris"], "rest": ["skin", "brow"]}}}) == \
        (("iris", ("iris",)), ("rest", ("skin", "brow")))
    assert groups_from_config({"anatomy": {"part_groups": [{"iris": ["iris"]}, {"skin": "skin"}]}}) == \
        (("iris", ("iris",)), ("skin", ("skin",)))
    for bad in ({"a": ["iris"], "b": ["iris"]}, {"a": []}, {}, [("", ["iris"])]):
        with pytest.raises(ValueError):
            normalise_groups(bad)
    with pytest.raises(ValueError):
        parse_groups("iris")


def test_group_masks():
    labels = eye_labels(1)[0]
    masks = group_masks(labels, CLASSES, DEFAULT_GROUPS)
    assert masks.shape == (5, 256, 256) and masks.dtype == bool
    assert masks.sum(axis=0).max() == 1                                   # disjoint
    np.testing.assert_array_equal(masks.any(axis=0), labels != 255)      # unknown pixels in no group
    np.testing.assert_array_equal(masks[0], labels == 1)
    np.testing.assert_array_equal(masks[1], (labels == 2) | (labels == 4))
    assert not masks[2].any()                                            # no lid in these maps
    with pytest.raises(ValueError):
        group_masks(labels, CLASSES, {"pupil": ["pupil"]})
    with pytest.raises(ValueError):
        group_masks(np.full((4, 4), 9, np.uint8), CLASSES, DEFAULT_GROUPS)


def test_shift_part_moves_one_parts_edge():
    """22 --shift-part: a part grows over its neighbours (they take its class) or shrinks (its edge pixels take the
    class of the nearest other labelled pixel); unknown pixels never change; 0 changes nothing."""
    labels = eye_labels(1)[0].copy()
    labels[16:24] = 3                                                     # a lid band under the brow (rows 0-15)
    below = labels[24:26].copy()
    grown = shift_part(labels, CLASSES, DEFAULT_GROUPS, "lid", 2)
    assert (grown[14:16] == 3).all() and (grown[16:24] == 3).all()        # the brow's last 2 rows became lid
    np.testing.assert_array_equal(grown[24:26], np.where(below == 255, 255, 3))
    assert (grown[:14] == labels[:14]).all() and (grown[26:] == labels[26:]).all()
    shrunk = shift_part(labels, CLASSES, DEFAULT_GROUPS, "lid", -2)
    assert (shrunk[16:18] == 5).all() and (shrunk[18:22] == 3).all()      # the top 2 lid rows took the brow's class
    np.testing.assert_array_equal(shrunk[22:24], labels[[24, 24]])       # the bottom 2: the class of row 24 below
    gone = shift_part(labels, CLASSES, DEFAULT_GROUPS, "lid", -8)         # 8 rows: every lid pixel within 4
    assert not (gone == 3).any()
    same = shift_part(labels, CLASSES, DEFAULT_GROUPS, "lid", 0)
    np.testing.assert_array_equal(same, labels)
    assert same is not labels and same.dtype == labels.dtype
    everywhere = shift_part(labels, CLASSES, DEFAULT_GROUPS, "skin", 300)
    np.testing.assert_array_equal(everywhere == 255, labels == 255)      # unknown pixels never change
    sclera = shift_part(labels, CLASSES, DEFAULT_GROUPS, "sclera", 1)     # a two-class group keeps its classes
    assert set(np.unique(sclera[sclera != labels])) <= {2, 4}
    for bad in ("pupil", "caruncle"):                                     # groups, not mask classes
        with pytest.raises(ValueError):
            shift_part(labels, CLASSES, DEFAULT_GROUPS, bad, 2)


@pytest.fixture(scope="module")
def offgrid_image(images):
    """An image off the 8-bit grid, so that 'bit-identical' is a real check."""
    return np.clip(images[0] + np.float32(1e-4) * np.arange(3, dtype=np.float32), 0, 1).astype(np.float32)


@pytest.mark.parametrize("method", METHODS)
def test_hide_keeps_kept_pixels_bit_identical(method, offgrid_image):
    rng = np.random.default_rng(0)
    keep = rng.random((256, 256)) < 0.6
    keep[:40] = True
    before = offgrid_image.copy()
    out = hide(offgrid_image, keep, method)
    assert out.dtype == np.float32 and out.shape == (256, 256, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_array_equal(out[keep], offgrid_image[keep])
    np.testing.assert_array_equal(offgrid_image, before)                 # input never modified
    assert not np.array_equal(out[~keep], offgrid_image[~keep])
    np.testing.assert_array_equal(hide(offgrid_image, np.ones((256, 256), bool), method), offgrid_image)
    nothing = hide(offgrid_image, np.zeros((256, 256), bool), method)
    assert np.isfinite(nothing).all() and nothing.min() >= 0 and nothing.max() <= 1


def test_hiding_fills(offgrid_image):
    import cv2
    keep = np.zeros((256, 256), bool)
    keep[:, :128] = True
    mean = hide(offgrid_image, keep, "mean")
    colour = offgrid_image.reshape(-1, 3).mean(0, dtype=np.float64)
    np.testing.assert_allclose(mean[~keep], np.broadcast_to(colour, (128 * 256, 3)),
                               atol=1e-6)
    blur = hide(offgrid_image, keep, "blur")
    expected = cv2.GaussianBlur(offgrid_image, (0, 0), 8.0, sigmaY=8.0, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(blur[~keep], np.clip(expected, 0, 1)[~keep], atol=1e-7)
    inpaint = hide(offgrid_image, keep, "inpaint")
    on_grid = inpaint[~keep] * 255
    np.testing.assert_allclose(on_grid, np.round(on_grid), atol=1e-3)      # 8-bit inpainted values
    cache = make_cache(offgrid_image)
    np.testing.assert_array_equal(hide(offgrid_image, keep, "blur", cache), blur)
    assert set(cache) == {"blur", "mean"}
    with pytest.raises(ValueError):
        hide(offgrid_image, keep, "zero")


@pytest.mark.parametrize("method", METHODS)
def test_coalition_indexing(method, offgrid_image):
    masks = group_masks(eye_labels(1)[0], CLASSES, DEFAULT_GROUPS)
    images = coalition_images(offgrid_image, masks, method)
    assert images.shape == (32, 256, 256, 3) and images.dtype == np.float32
    np.testing.assert_array_equal(images[31], offgrid_image)               # every group kept
    ungrouped = ~masks.any(axis=0)
    for k in range(32):
        keep = coalition_keep(masks, k)
        np.testing.assert_array_equal(keep, ungrouped | masks[[p for p in range(5) if (k >> p) & 1]].any(axis=0))
        np.testing.assert_array_equal(images[k][keep], offgrid_image[keep])
        np.testing.assert_array_equal(images[k][ungrouped], offgrid_image[ungrouped])
    # bit 0 = iris: coalitions 1 and 0 differ on the iris pixels (and, for inpainting, whose fill
    # depends on every kept pixel, also elsewhere in the hidden region; never on ungrouped pixels)
    changed = (images[1] != images[0]).any(axis=-1)
    assert changed[masks[0]].mean() > 0.9 and not changed[ungrouped].any()
    if method != "inpaint":
        assert not changed[~masks[0]].any()
    # the lid group is empty: coalitions that differ only in it are the same image
    np.testing.assert_array_equal(images[0b00100], images[0])


# ---------------------------------------------------------------------------
# Scores and ASA on the tiny model
# ---------------------------------------------------------------------------

def test_score_values():
    logits = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    np.testing.assert_allclose(score_values(logits, "logit", target=2), [2.0, 4.0])
    np.testing.assert_allclose(score_values(logits, "margin", target=2), [-1.0, -1.0])
    np.testing.assert_allclose(score_values(logits, "margin", target=0), [1.0, -5.0])
    emb = np.array([[1.0, 0.0], [0.6, 0.8]])
    np.testing.assert_allclose(score_values(emb, "verif", template=[0.0, 2.0]), [0.0, 0.8])
    for kwargs in ({"score": "margin"}, {"score": "margin", "target": 3}, {"score": "verif"}, {"score": "prob"}):
        with pytest.raises(ValueError):
            score_values(logits, **kwargs)


@pytest.mark.parametrize("score", ["margin", "logit"])
def test_asa_end_to_end_on_the_tiny_model(model, images, labels, score):
    lab = eye_labels(len(images))
    image, label_map, target = images[1], lab[1], int(labels[1])
    out = asa(model, image, label_map, CLASSES, DEFAULT_GROUPS, target=target, method="blur", score=score,
              batch_size=16)
    assert out["names"] == GROUP_NAMES and out["values"].shape == (32,)
    assert out["phi"].shape == (5,) and out["interactions"].shape == (5, 5)
    assert out["phi"].sum() == pytest.approx(out["v_full"] - out["v_empty"], rel=1e-12, abs=1e-9)
    np.testing.assert_allclose(np.diag(out["interactions"]), out["phi"])
    # v_full is the score of the unchanged image
    logits = predict_logits(model, image[np.newaxis])
    full = score_values(logits, score, target=target)[0]
    assert out["v_full"] == pytest.approx(full, abs=1e-4)
    if score == "margin":
        assert out["v_full"] > 0                                           # correctly classified
    # the lid group has no pixel: exact null player, and fewer forward passes (16, not 32)
    assert out["present"].tolist() == [True, True, False, True, True]
    assert out["phi"][2] == 0.0 and np.all(out["interactions"][2] == 0.0)
    assert out["n_forward"] == 16
    # the values are the scores of the coalition images
    masks = group_masks(label_map, CLASSES, DEFAULT_GROUPS)
    direct = score_values(predict_logits(model, coalition_images(image, masks, "blur")), score, target=target)
    np.testing.assert_allclose(out["values"], direct, atol=1e-4)


def test_coalition_outputs_share_one_pass(model, images):
    masks = group_masks(eye_labels(1)[0], CLASSES, DEFAULT_GROUPS)
    outputs, present, n = coalition_outputs(model, images[0], masks, list(METHODS), batch_size=40)
    assert set(outputs) == set(METHODS) and n == 3 * 16 and present.sum() == 4
    for m in METHODS:
        assert outputs[m].shape == (32, 5)
        np.testing.assert_array_equal(outputs[m][0b00100], outputs[m][0])      # absent lid
    np.testing.assert_allclose(outputs["mean"][31], outputs["blur"][31], atol=1e-5)    # same full image


def test_asa_verification_score(model, images):
    lab = eye_labels(len(images))
    template = embed(model, images[2])[0]
    out = asa(model, images[2], lab[2], CLASSES, target=None, method="mean", score="verif", template=template)
    assert out["v_full"] == pytest.approx(1.0, abs=1e-5)                  # the image against its own embedding
    assert out["phi"].sum() == pytest.approx(out["v_full"] - out["v_empty"], abs=1e-12)
    masks = group_masks(lab[2], CLASSES, DEFAULT_GROUPS)
    emb = embed(model, coalition_images(images[2], masks, "mean"))
    np.testing.assert_allclose(out["values"], emb @ template, atol=1e-5)
    with pytest.raises(ValueError):
        asa(model, images[2], lab[2], CLASSES, method="mean", score="verif")


# ---------------------------------------------------------------------------
# Faithfulness helpers
# ---------------------------------------------------------------------------

def test_part_masses_match_region_masses():
    lab = eye_labels(1)[0]
    heatmap = np.random.default_rng(1).random((8, 8))
    masses = part_masses(heatmap, lab, CLASSES, DEFAULT_GROUPS)
    grids, _ = grids_from_labels(lab[np.newaxis], CLASSES, 8)
    region = region_mass_named(heatmap, grids[0], CLASSES)
    assert masses[0] == pytest.approx(region["mass_iris"])
    assert masses[1] == pytest.approx(region["mass_sclera"] + region["mass_caruncle"])
    assert masses[2] == 0.0
    unknown = (heatmap * (cv_area(lab == 255))).sum() / heatmap.sum()
    assert masses.sum() == pytest.approx(1 - unknown)
    assert np.isnan(part_masses(np.zeros((8, 8)), lab, CLASSES)).all()


def cv_area(mask):
    import cv2
    return cv2.resize(mask.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)


def test_rank_agreement():
    present = np.array([True, True, False, True, True])
    assert rank_agreement([4, 3, 0, 2, 1], [40, 30, -99, 20, 10], present) == pytest.approx(1.0)
    assert rank_agreement([4, 3, 0, 2, 1], [1, 2, 99, 3, 4], present) == pytest.approx(-1.0)
    assert np.isnan(rank_agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [True, True, False, False, False]))
    assert np.isnan(rank_agreement([1, 1, 0, 1, 1], [1, 2, 3, 4, 5], present))
    assert np.isnan(rank_agreement([1, 2, 3, 4, 5], [1, np.nan, 3, np.nan, np.nan], np.ones(5, bool)))


def test_deletion_from_values():
    n = 3
    weights = np.array([6.0, 3.0, 1.0])
    v = np.array([sum(weights[p] for p in members(s, n)) for s in range(1 << n)])
    best = deletion_from_values(v, [0, 1, 2])
    worst = deletion_from_values(v, [2, 1, 0])
    np.testing.assert_allclose(best["scores"], [10, 4, 1, 0])
    # curve 1, 0.4, 0.1, 0 -> trapezoid mean (0.7 + 0.25 + 0.05) / 3
    assert best["auc"] == pytest.approx(1.0 / 3)
    assert worst["auc"] > best["auc"]
    assert np.isnan(deletion_from_values(v, [1])["auc"])
    assert order_from_scores([0.5, 2.0, np.nan, 1.0], [True, True, False, True]) == [1, 3, 0]
    assert order_from_scores([0.5, np.nan, 1.0], [True, True, True]) is None
    with pytest.raises(ValueError):
        deletion_from_values(v, [0, 0])


@pytest.mark.parametrize("weights", [[6.0, -1.0, 3.0, 0.5], [-6.0, 1.0, -3.0, -0.5], [2.0, -4.0, 1.0, 0.0]])
def test_asa_order_is_the_best_deletion_order_of_an_additive_game(weights):
    n = len(weights)
    v = np.array([1.0 + sum(weights[p] for p in members(s, n)) for s in range(1 << n)])
    phi = exact_shapley(v)
    present = np.ones(n, bool)
    best = min(deletion_from_values(v, list(perm))["auc"] for perm in itertools.permutations(range(n)))
    order = asa_order(phi, v[-1], v[0], present)
    assert deletion_from_values(v, order)["auc"] == pytest.approx(best, abs=1e-12)


def test_threaded_inpainting_is_identical(offgrid_image):
    masks = group_masks(eye_labels(1)[0], CLASSES, DEFAULT_GROUPS)
    np.testing.assert_array_equal(coalition_images(offgrid_image, masks, "inpaint", workers=1),
                                  coalition_images(offgrid_image, masks, "inpaint", workers=4))


def test_deletion_curve_on_the_model_matches_the_asa_values(model, images, labels):
    lab = eye_labels(len(images))
    out = asa(model, images[3], lab[3], CLASSES, target=int(labels[3]), method="inpaint", score="margin")
    masks = group_masks(lab[3], CLASSES, DEFAULT_GROUPS)
    order = order_from_scores(out["phi"], out["present"])
    direct = deletion_curve(model, images[3], masks, order, "inpaint", "margin", target=int(labels[3]))
    free = deletion_from_values(out["values"], order)
    np.testing.assert_allclose(direct["scores"], free["scores"], atol=1e-4)
    assert direct["auc"] == pytest.approx(free["auc"], abs=1e-3)
