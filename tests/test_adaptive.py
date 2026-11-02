"""Adaptive attacks (src/attacks/adaptive.py) and scripts/14_adaptive_attacks.py, on the tiny model."""
import functools
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from conftest import REPO
from src.attacks.adaptive import (_grid_rounding, bpda_gradient, bpda_pgd, eot_bpda_pgd, identity,
                                  random_jpeg, transform_robust_pgd)
from src.attacks.whitebox import _loss_gradient, pgd
from src.compare.divergence import spearman
from src.defences.inference import bit_depth_reduction, jpeg_compress
from src.models.arcface_inference import get_margin_free, predict_logits

JPEG = functools.partial(jpeg_compress, quality=75)
BITS = functools.partial(bit_depth_reduction, bits=4)
EPS = 2 / 255
STEPS = 4

ATTACKS = {
    "bpda_jpeg": lambda x, m, eps, **kw: bpda_pgd(x, m, eps, JPEG, **kw),
    "bpda_bits": lambda x, m, eps, **kw: bpda_pgd(x, m, eps, BITS, **kw),
    "tr_jpeg": lambda x, m, eps, **kw: transform_robust_pgd(x, m, eps, [identity, JPEG], **kw),
    "eot_jpeg": lambda x, m, eps, **kw: eot_bpda_pgd(x, m, eps, random_jpeg(50, 90), n_samples=2, **kw),
}


def _script():
    path = os.path.join(REPO, "scripts", "14_adaptive_attacks.py")
    spec = importlib.util.spec_from_file_location("adaptive_attacks_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _script()


@pytest.fixture(scope="module")
def clean(images):
    """The synthetic images on the 8-bit grid, like every image loaded from a PNG."""
    return (np.round(images * 255.0) / 255.0).astype(np.float32)


def margin(model, x, labels):
    """True-class logit minus the best other logit (negative = misclassified)."""
    logits = predict_logits(model, x)
    true = logits[np.arange(len(labels)), labels].copy()
    logits[np.arange(len(labels)), labels] = -np.inf
    return true - logits.max(axis=1)


def on_grid(x):
    return np.allclose(x * 255.0, np.round(x * 255.0), atol=1e-4)


def recording(transform, calls):
    """*transform* that also keeps a copy of every batch it is given."""
    def record(images):
        calls.append(np.array(images, copy=True))
        return transform(images)
    return record


# ---------------------------------------------------------------------------
# BPDA gradient
# ---------------------------------------------------------------------------

def test_bpda_gradient_reaches_the_input_through_jpeg(model, clean, labels):
    x = tf.constant(clean[:2])
    net = get_margin_free(model)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(net(tf.constant(JPEG(x))))     # JPEG runs in numpy: the tape loses x
    assert tape.gradient(loss, x) is None

    gradient, margins = bpda_gradient(clean[:2], model, JPEG, label=labels[:2])
    assert gradient.shape == (2, 256, 256, 3) and np.isfinite(gradient).all()
    assert (np.abs(gradient).reshape(2, -1).max(axis=1) > 0).all()
    # BPDA: the ordinary gradient at the defended image, passed straight back to the input
    expected = _loss_gradient(net, tf.constant(JPEG(clean[:2])), tf.constant(labels[:2], tf.int32))
    np.testing.assert_allclose(gradient, expected.numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(margins, margin(model, JPEG(clean[:2]), labels[:2])[np.newaxis], atol=1e-4)


def test_transform_robust_objective_covers_both_transforms(model, clean, labels):
    both, margins_both = bpda_gradient(clean, model, [identity, JPEG], label=labels)
    plain, margins_plain = bpda_gradient(clean, model, identity, label=labels)
    defended, margins_defended = bpda_gradient(clean, model, JPEG, label=labels)
    np.testing.assert_allclose(both, plain + defended, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(margins_both, np.concatenate([margins_plain, margins_defended]), atol=1e-5)
    # through the identity, BPDA is the ordinary margin-loss gradient of whitebox.py
    expected = _loss_gradient(get_margin_free(model), tf.constant(clean), tf.constant(labels, tf.int32))
    np.testing.assert_allclose(plain, expected.numpy(), rtol=1e-5, atol=1e-7)

    adv, info = transform_robust_pgd(clean, model, EPS, [identity, JPEG], label=labels, steps=STEPS,
                                     return_info=True)
    worst = np.maximum(margin(model, adv, labels), margin(model, JPEG(adv), labels))
    np.testing.assert_allclose(info["margin"], worst, atol=1e-4)    # success = fooled under both
    np.testing.assert_array_equal(info["success"], worst < 0)


# ---------------------------------------------------------------------------
# 8-bit budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ATTACKS))
@pytest.mark.parametrize("eps, levels", [(0.0039216, 1), (0.0078431, 2), (0.03, 7)])
def test_outputs_are_8bit_and_within_whole_levels_of_the_budget(model, clean, labels, name, eps, levels):
    adv = ATTACKS[name](clean, model, eps, label=labels, steps=STEPS)
    assert adv.shape == clean.shape and adv.dtype == np.float32
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    assert on_grid(adv)
    linf = np.abs(adv - clean).max()
    assert 0 < linf <= eps + 1e-6
    assert linf * 255.0 <= levels + 1e-3


def test_rounding_matches_quantise_of_the_runner(script, images, clean):
    rng = np.random.default_rng(0)
    for eps in (0.0039216, 0.0078431, 0.0156863, 0.03):
        x = np.clip(clean + rng.uniform(-2 * eps, 2 * eps, clean.shape), 0.0, 1.0).astype(np.float32)
        ours = _grid_rounding(clean, eps)(x)
        runner = script.quantise(x, clean, eps)              # private copy of 05's quantise
        np.testing.assert_array_equal(np.round(ours * 255.0), np.round(runner * 255.0))
    # an image off the grid still gets 8-bit levels within the budget
    off = _grid_rounding(images, EPS)(np.clip(images + rng.uniform(-EPS, EPS, images.shape), 0.0, 1.0))
    assert on_grid(off) and np.abs(off - images).max() <= EPS + 1e-6


def test_budget_below_one_level_needs_float_mode(model, clean, labels):
    with pytest.raises(ValueError, match="below one 8-bit level"):
        bpda_pgd(clean[:1], model, 0.001, JPEG, label=labels[:1], steps=1)
    adv = bpda_pgd(clean[:1], model, 0.001, BITS, label=labels[:1], steps=2, quantise_8bit=False)
    assert 0 < np.abs(adv - clean[:1]).max() <= 0.001 + 1e-7


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def test_success_is_judged_after_the_defence(model, clean, labels):
    adv, info = bpda_pgd(clean, model, EPS, JPEG, label=labels, steps=STEPS, return_info=True)
    defended = margin(model, JPEG(adv), labels)
    np.testing.assert_allclose(info["margin"], defended, atol=1e-4)  # scored on D(returned 8-bit image)
    np.testing.assert_array_equal(info["success"], predict_logits(model, JPEG(adv)).argmax(1) != labels)
    assert not np.allclose(info["margin"], margin(model, adv, labels), atol=1e-2)
    assert info["history"].shape == (STEPS + 1, len(labels))
    np.testing.assert_array_equal(info["margin"], info["history"].min(axis=0))   # the best iterate


def test_the_defence_sees_every_iterate_as_an_8bit_image(model, clean, labels):
    # D runs outside the tape once per iterate (the start and every step), always on the rounded iterate
    calls = []
    adv = bpda_pgd(clean[:2], model, 0.03, recording(JPEG, calls), label=labels[:2], steps=STEPS)
    assert len(calls) == STEPS + 1
    for x in calls:
        assert on_grid(x) and x.min() >= 0.0 and x.max() <= 1.0
        assert np.abs(x - clean[:2]).max() * 255.0 <= 7 + 1e-3              # 0.03 allows 7 whole levels
    for i in range(2):                                                        # each result is a defended iterate
        assert any(np.array_equal(adv[i], x[i]) for x in calls)
    # float mode: every iterate is projected onto the epsilon ball and [0, 1]
    calls.clear()
    bpda_pgd(clean[:2], model, 0.001, recording(JPEG, calls), label=labels[:2], steps=STEPS, quantise_8bit=False)
    assert len(calls) == STEPS + 1 and not np.array_equal(calls[0], calls[-1])
    for x in calls:
        assert np.abs(x - clean[:2]).max() <= 0.001 + 1e-7 and x.min() >= 0.0 and x.max() <= 1.0
    # transform-robust: the undefended and the defended branch see the same 8-bit image at every step
    plain, defended = [], []
    transform_robust_pgd(clean[:2], model, EPS, [recording(identity, plain), recording(BITS, defended)],
                         label=labels[:2], steps=STEPS)
    assert len(plain) == len(defended) == STEPS + 1
    assert all(on_grid(a) and np.array_equal(a, b) for a, b in zip(plain, defended))


def test_bpda_attacks_the_defence_better_than_a_non_adaptive_attack(script, model, clean, labels):
    eps = 4 / 255
    plain = script.quantise(pgd(clean, model, eps, label=labels, steps=STEPS), clean, eps)
    adaptive = bpda_pgd(clean, model, eps, JPEG, label=labels, steps=STEPS)
    assert (margin(model, JPEG(adaptive), labels) < margin(model, JPEG(plain), labels)).all()


def test_attacks_are_deterministic_per_seed(model, clean, labels):
    for name in ("bpda_jpeg", "eot_jpeg"):
        a = ATTACKS[name](clean[:2], model, EPS, label=labels[:2], steps=STEPS, seed=1)
        b = ATTACKS[name](clean[:2], model, EPS, label=labels[:2], steps=STEPS, seed=1)
        c = ATTACKS[name](clean[:2], model, EPS, label=labels[:2], steps=STEPS, seed=2)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


def test_eot_with_a_fixed_defence_is_bpda(model, clean, labels):
    eot = eot_bpda_pgd(clean, model, EPS, lambda rng: JPEG, n_samples=2, label=labels, steps=STEPS)
    np.testing.assert_array_equal(eot, bpda_pgd(clean, model, EPS, JPEG, label=labels, steps=STEPS))


def test_random_jpeg_draws_seeded_qualities_in_range():
    sample = random_jpeg(60, 70)
    first = [sample(np.random.default_rng(3)).keywords["quality"] for _ in range(2)]
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    a = [sample(rng_a).keywords["quality"] for _ in range(50)]
    b = [sample(rng_b).keywords["quality"] for _ in range(50)]
    assert a == b and first[0] == first[1] == a[0]
    assert min(a) >= 60 and max(a) <= 70 and len(set(a)) > 1
    with pytest.raises(ValueError):
        random_jpeg(90, 60)


def test_single_image_and_input_untouched(model, clean, labels):
    original = clean.copy()
    assert bpda_pgd(clean[0], model, EPS, JPEG, label=labels[0], steps=2).shape == (256, 256, 3)
    assert transform_robust_pgd(clean[0], model, EPS, [identity, BITS], steps=2).shape == (256, 256, 3)
    np.testing.assert_array_equal(clean, original)


def test_invalid_arguments(model, clean):
    with pytest.raises(ValueError):
        bpda_pgd(clean[:1], model, EPS, JPEG, steps=0)
    with pytest.raises(ValueError):
        bpda_pgd(clean[:1], model, EPS, JPEG, loss="hinge")
    with pytest.raises(ValueError):
        transform_robust_pgd(clean[:1], model, EPS, [])
    with pytest.raises(ValueError):
        eot_bpda_pgd(clean[:1], model, EPS, random_jpeg(), n_samples=0)


# ---------------------------------------------------------------------------
# scripts/14_adaptive_attacks.py
# ---------------------------------------------------------------------------

def test_run_batch_and_summary(script, model, clean, labels):
    eps = 0.0078431
    defences = script.make_defences(["jpeg", "bit_depth"], jpeg_quality=75, bits=4)
    base = [{"image_id": 10 + i, "true_idx": int(label)} for i, label in enumerate(labels[:3])]
    timings = {}
    rows = pd.DataFrame(script.run_batch(model, clean[:3], labels[:3], base, [eps], defences, steps=2,
                                         seed=0, timings=timings))
    assert len(rows) == 3 * 3 * 2                                   # images x attacks x defences
    assert set(rows.attack) == {"pgd", "bpda_pgd", "tr_pgd"} and set(rows.defence) == {"jpeg", "bit_depth"}
    assert set(timings) == {"pgd", "bpda_pgd", "tr_pgd", "evaluate"}
    assert (rows.budget_levels == 2).all() and (rows.pert_linf <= eps + 1e-6).all()
    # whole levels, never more than the budget: float32 round-off must not show up in the column
    assert (rows.pert_linf_levels <= rows.budget_levels).all() and (rows.pert_l2 > 0).all()
    assert (rows.pert_linf_levels == rows.pert_linf_levels.round()).all()
    assert (rows.success_defended == (rows.pred_defended != rows.true_idx)).all()
    assert (rows.success_defended == (rows.margin_defended < 0)).all()
    assert (rows.success_undefended == (rows.pred_undefended != rows.true_idx)).all()
    assert rows.adaptive.tolist() == (rows.attack != "pgd").tolist()
    assert set(rows.defence_param) == {75, 4}
    # the non-adaptive attack is the same image under both defences
    assert (rows[rows.attack == "pgd"].groupby("image_id").pert_l2.nunique() == 1).all()
    for xai in ("layercam", "eigencam"):
        rho = rows[f"{xai}_vs_defended_clean_spearman"].dropna()
        top = rows[f"{xai}_vs_defended_clean_top10_intersection"]
        assert rho.between(-1, 1).all() and top.between(0, 1).all()

    summary = script.summarise(rows)
    keys = [f"{a}@0.0078431+{d}" for d in ("jpeg", "bit_depth") for a in ("pgd", "bpda_pgd", "tr_pgd")]
    assert list(summary["success_after_defence"]) == keys
    for r in summary["success_after_defence"].values():
        assert r["n"] == 3 and 0 <= r["ci_low"] <= r["rate"] <= r["ci_high"] <= 1
    for d in ("jpeg", "bit_depth"):
        any_attack = summary["success_after_defence_any_attack"][f"0.0078431+{d}"]["rate"]
        assert any_attack >= max(summary["success_after_defence"][f"{a}@0.0078431+{d}"]["rate"]
                                 for a in ("pgd", "bpda_pgd", "tr_pgd"))
        both = summary["success_with_and_without_defence"][f"tr_pgd@0.0078431+{d}"]["rate"]
        assert both <= summary["success_after_defence"][f"tr_pgd@0.0078431+{d}"]["rate"]
    assert set(summary["defence_on_clean_flip_rate"]) == {"jpeg", "bit_depth"}
    assert set(summary["median_map_similarity_to_defended_clean"]["bpda_pgd@0.0078431+jpeg"]) == {
        f"{x}_vs_defended_clean_{m}" for x in ("layercam", "eigencam") for m in ("spearman", "top10_intersection")}


def test_score_compares_defended_with_defended_clean(script, model, clean, labels):
    # an unchanged image: D(adv) = D(clean), so its maps equal the defended clean maps. A harsh defence (1 bit)
    # moves the maps and can flip the label, so maps or success taken before D would show up here.
    defence, _ = script.make_defences(["bit_depth"], bits=1)["bit_depth"]
    defended, xai = defence(clean), ("layercam", "eigencam")
    reference = {"correct": predict_logits(model, defended).argmax(1) == labels,
                 "maps": {x: script.true_class_maps(model, x, defended, labels) for x in xai}}
    undefended_maps = script.true_class_maps(model, "layercam", clean, labels)
    assert min(spearman(a, b) for a, b in zip(undefended_maps, reference["maps"]["layercam"])) < 0.9
    rows = pd.DataFrame(script.score(model, clean, labels, clean.copy(), defence, reference, xai,
                                     [{"image_id": i} for i in range(len(labels))], {"attack": "none"}))
    np.testing.assert_array_equal(rows.success_defended, ~reference["correct"])
    np.testing.assert_array_equal(rows.defended_clean_correct, reference["correct"])
    assert not rows.success_undefended.any()
    np.testing.assert_allclose(rows.margin_defended, margin(model, defended, labels), atol=1e-4)
    np.testing.assert_allclose(rows.margin_undefended, margin(model, clean, labels), atol=1e-4)
    assert (rows.pert_linf == 0).all() and (rows.pert_linf_levels == 0).all() and (rows.pert_l2 == 0).all()
    for x in xai:
        assert (rows[f"{x}_vs_defended_clean_top10_intersection"] == 1).all()
        assert np.allclose(rows[f"{x}_vs_defended_clean_spearman"], 1.0)


def test_summarise_counts_on_known_rows(script):
    # 4 images; the 1/255 block comes last in the table but first in the summary (ordered by budget)
    t, f = True, False
    blocks = [  # attack, epsilon, defence, success after D, success without D, D(clean) correct, a map similarity
        ("pgd", 0.0078431, "jpeg", [t, f, f, t], [t, t, f, t], [t, t, t, f], [0.1, 0.2, np.nan, 0.4]),
        ("bpda_pgd", 0.0078431, "jpeg", [f, t, f, t], [t, t, t, t], [t, t, t, f], [0.5, 0.6, 0.7, 0.8]),
        ("pgd", 0.0078431, "bit_depth", [f, t, t, f], [t, t, f, t], [t, f, f, t], [0.0, 0.0, 0.0, 0.0]),
        ("pgd", 0.0039216, "jpeg", [f, f, f, f], [t, t, t, t], [t, t, t, f], [0.9, 0.9, 0.9, 0.9]),
    ]
    rows = pd.concat([pd.DataFrame({"image_id": range(4), "attack": a, "epsilon": e, "defence": d,
                                    "success_defended": s, "success_undefended": u, "defended_clean_correct": c,
                                    "layercam_vs_defended_clean_spearman": r})
                      for a, e, d, s, u, c, r in blocks], ignore_index=True)
    summary = script.summarise(rows)

    def counts(name):
        return {key: (r["k"], r["n"]) for key, r in summary[name].items()}

    assert list(summary["success_after_defence"]) == [
        "pgd@0.0039216+jpeg", "pgd@0.0078431+jpeg", "bpda_pgd@0.0078431+jpeg", "pgd@0.0078431+bit_depth"]
    assert counts("success_after_defence") == {
        "pgd@0.0039216+jpeg": (0, 4), "pgd@0.0078431+jpeg": (2, 4), "bpda_pgd@0.0078431+jpeg": (2, 4),
        "pgd@0.0078431+bit_depth": (2, 4)}
    assert counts("success_after_defence_where_defended_clean_correct") == {
        "pgd@0.0039216+jpeg": (0, 3), "pgd@0.0078431+jpeg": (1, 3), "bpda_pgd@0.0078431+jpeg": (1, 3),
        "pgd@0.0078431+bit_depth": (0, 2)}
    assert counts("success_without_defence") == {
        "pgd@0.0039216+jpeg": (4, 4), "pgd@0.0078431+jpeg": (3, 4), "bpda_pgd@0.0078431+jpeg": (4, 4),
        "pgd@0.0078431+bit_depth": (3, 4)}
    assert counts("success_with_and_without_defence") == {
        "pgd@0.0039216+jpeg": (0, 4), "pgd@0.0078431+jpeg": (2, 4), "bpda_pgd@0.0078431+jpeg": (2, 4),
        "pgd@0.0078431+bit_depth": (1, 4)}
    assert counts("success_after_defence_any_attack") == {      # images 0, 1 and 3 fall to pgd or bpda_pgd
        "0.0039216+jpeg": (0, 4), "0.0078431+jpeg": (3, 4), "0.0078431+bit_depth": (2, 4)}
    assert counts("defence_on_clean_flip_rate") == {"jpeg": (1, 4), "bit_depth": (2, 4)}
    rate = summary["success_after_defence"]["pgd@0.0078431+jpeg"]
    assert rate["rate"] == 0.5 and (rate["ci_low"], rate["ci_high"]) == script.wilson(2, 4)
    similarity = summary["median_map_similarity_to_defended_clean"]["pgd@0.0078431+jpeg"]
    assert similarity == {"layercam_vs_defended_clean_spearman": pytest.approx(0.2)}      # NaN skipped


def test_wilson_interval(script):
    assert script.wilson(0, 5)[0] == 0.0 and script.wilson(5, 5)[1] == 1.0     # no round-off outside [0, 1]
    low, high = script.wilson(12, 100)
    assert low == pytest.approx(0.06999, abs=1e-5) and high == pytest.approx(0.19812, abs=1e-5)


def test_script_end_to_end(script, tmp_path, model, clean, labels):
    ckpt = tmp_path / "tiny.keras"
    model.save(ckpt)
    records = []
    for i, (img, label) in enumerate(zip(clean[:3], labels[:3])):
        path = tmp_path / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        records.append({"link_path": str(path), "relpath": f"test/{label}/{label}_{i}_l_c.png",
                        "true_idx": int(label), "true_subject": 1000 + int(label), "eye": "left"})
    correct_csv = tmp_path / "correct_test_tiny.csv"
    pd.DataFrame(records).to_csv(correct_csv, index=False)
    out_dir = tmp_path / "out"
    common = ["--ckpt", str(ckpt), "--correct-csv", str(correct_csv), "--out-dir", str(out_dir),
              "--epsilons", "0.0078431", "--batch-size", "2"]

    script.main(common + ["--max-images", "2", "--steps", "2", "--xai", "eigencam", "--tag", "tiny"])
    table = pd.read_csv(out_dir / "rows_adaptive_tiny.csv")
    assert table.image_id.nunique() == 2 and len(table) == 2 * 3 * 2
    assert {"relpath", "link_path", "true_subject", "pert_l2", "margin_defended",
            "eigencam_vs_defended_clean_spearman"} <= set(table.columns)
    assert not any(c.startswith("layercam") for c in table.columns)
    summary = json.loads((out_dir / "summary_adaptive_tiny.json").read_text())
    assert summary["n_images"] == 2 and summary["budget_levels"] == [2] and summary["steps"] == 2
    assert summary["defences"] == {"jpeg": {"quality": 75}, "bit_depth": {"bits": 4}}
    assert isinstance(summary["git_commit"], str) and summary["git_commit"]
    assert set(summary["cost_relative_to_pgd"]) == {"pgd", "bpda_pgd", "tr_pgd"}

    # --exclude-from with the rows of the first run leaves the third image only
    script.main(common + ["--exclude-from", str(out_dir / "rows_adaptive_tiny.csv"), "--attacks", "bpda_pgd",
                          "--defences", "jpeg", "--steps", "1", "--xai", "none", "--tag", "rest"])
    rest = pd.read_csv(out_dir / "rows_adaptive_rest.csv")
    assert len(rest) == 1 and rest.link_path[0] not in set(table.link_path)

    for bad in (["--epsilons", "0.001"],                # below one 8-bit level: refused before loading
                ["--steps", "0"], ["--epsilons", "0.0078431,-0.0039216", "--no-quantise"]):
        with pytest.raises(SystemExit):
            script.main(common[:4] + bad)
