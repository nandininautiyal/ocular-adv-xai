"""Gradient-masking check (src/attacks/masking.py, the AttentionPool temperature, scripts/24_gradient_masking.py
and 14 --attacks ...,aas_pgd) on tiny models with and without an attention pool."""
import importlib.util
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml
from PIL import Image

from conftest import REPO, make_images
from src.attacks.budget import quantise
from src.attacks.masking import (DEFAULT_TAUS, aas_pgd, attention_stats, check_taus, gradient_stats, has_attention,
                                 map_saturation, with_attention_temperature)
from src.attacks.whitebox import _loss_gradient, pgd
from src.models import anatomy_model as am
from src.models.arcface_inference import get_margin_free, load_checkpoint, predict_logits

N_CLASSES = 6
SATURATION = 400.0     # attention output weights x400: some maps put > 0.9 of their weight on one cell


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def masking_script():
    return _script("24_gradient_masking")


@pytest.fixture(scope="module")
def adaptive_script():
    return _script("14_adaptive_attacks")


def tiny_cfg(n_classes=N_CLASSES):
    """configs/anatomy.yml with the tiny backbone and small heads (as in test_anatomy_model)."""
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"backbone": "tiny", "fine_tune_at": 0, "dense_units": 16, "n_classes": n_classes})
    cfg["anatomy"]["hidden"] = 8
    cfg["training"]["learning_rate"] = 1e-3
    return cfg


def tiny_model(arch, scale=1.0, seed=0):
    """A tiny anatomy model; *scale* multiplies the attention MLP's output weights (large = saturated maps)."""
    tf.keras.utils.set_random_seed(seed)
    model = am.build_anatomy_model(tiny_cfg(), arch)
    if scale != 1.0:
        pool = model.get_layer("attention_pool")
        pool.w2.assign(pool.w2 * scale)
    return model


@pytest.fixture(scope="module")
def x():
    return (np.round(make_images(N_CLASSES, seed=1) * 255.0) / 255.0).astype(np.float32)


@pytest.fixture(scope="module")
def attn():
    return tiny_model("attn")


@pytest.fixture(scope="module")
def parts():
    return tiny_model("parts")


@pytest.fixture(scope="module")
def saturated():
    return tiny_model("attn", scale=SATURATION)


def margins(model, images, labels):
    logits = predict_logits(model, images)
    idx = np.arange(len(labels))
    true = logits[idx, labels].copy()
    logits[idx, labels] = -np.inf
    return true - logits.max(axis=1)


def pool_logits(model, images):
    """The attention MLP's output (N, H*W, K) before the softmax, recomputed from the layer weights."""
    pool = model.get_layer("attention_pool")
    features = tf.keras.Model(model.inputs[0], pool.input)(images)
    hidden = tf.nn.relu(tf.einsum("nhwc,cd->nhwd", features, pool.w1) + pool.b1)
    logits = tf.einsum("nhwd,dk->nhwk", hidden, pool.w2) + pool.b2
    return tf.reshape(logits, (len(images), -1, pool.n_maps)).numpy()


# ---------------------------------------------------------------------------
# AttentionPool temperature
# ---------------------------------------------------------------------------

def test_temperature_one_gives_identical_outputs_on_shared_weights(attn, parts, x):
    zeros = np.zeros((len(x), N_CLASSES), np.float32)
    for model in (attn, parts):
        scaled = with_attention_temperature(model, 1.0)
        assert scaled is not model and scaled.name == model.name and am.model_arch(scaled) == am.model_arch(model)
        np.testing.assert_array_equal(scaled.predict_on_batch([x, zeros]), model.predict_on_batch([x, zeros]))
        np.testing.assert_array_equal(predict_logits(scaled, x), predict_logits(model, x))
        np.testing.assert_array_equal(am.attention_maps(scaled, x), am.attention_maps(model, x))
        # the same variables, nothing copied
        assert {id(v) for v in scaled.weights} == {id(v) for v in model.weights}
        assert len(scaled.weights) == len(model.weights)
        assert with_attention_temperature(model, 1.0) is scaled          # cached per (model, tau)


def test_temperature_divides_the_attention_logits(parts, x):
    before = predict_logits(parts, x)
    logits = pool_logits(parts, x)
    for tau in (2.0, 16.0):
        maps = am.attention_maps(with_attention_temperature(parts, tau), x)
        want = tf.nn.softmax(logits / tau, axis=1).numpy().reshape(maps.shape)
        np.testing.assert_allclose(maps, want, atol=1e-6)
    # a higher temperature flattens the maps; the original model is unchanged
    ent = [map_saturation(am.attention_maps(with_attention_temperature(parts, t), x))[1].mean() for t in (1, 4, 64)]
    assert ent[0] < ent[1] < ent[2] <= 1.0
    np.testing.assert_array_equal(predict_logits(parts, x), before)


def test_scaled_model_follows_weight_updates_of_the_original(x):
    model = tiny_model("attn", seed=3)
    scaled = with_attention_temperature(model, 4.0)
    before = predict_logits(scaled, x)
    dense = model.get_layer("head_dense")
    dense.kernel.assign(dense.kernel * 2.0)
    pool = model.get_layer("attention_pool")
    pool.w2.assign(pool.w2 * 3.0)
    after = predict_logits(scaled, x)
    assert not np.allclose(before, after)
    np.testing.assert_array_equal(predict_logits(with_attention_temperature(model, 1.0), x), predict_logits(model, x))


def test_saved_models_keep_the_old_format_and_old_files_load(attn, x, tmp_path):
    # a model saved before the temperature existed has exactly these AttentionPool config keys and four weights;
    # at temperature 1 the layer still writes that format, so old and new files are the same and both load
    old_keys = set(tf.keras.layers.Layer(name="probe").get_config()) | {"hidden", "n_maps", "map_names"}
    path = tmp_path / "attn.keras"
    attn.save(path)
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    layer = next(lay for lay in config["config"]["layers"] if lay["config"]["name"] == "attention_pool")
    assert set(layer["config"]) == old_keys and "temperature" not in json.dumps(layer)
    assert [w.name.split("/")[-1].split(":")[0] for w in attn.get_layer("attention_pool").weights] == \
        ["w1", "b1", "w2", "b2"]
    loaded = load_checkpoint(str(path))
    assert loaded.get_layer("attention_pool").temperature == 1.0
    np.testing.assert_array_equal(predict_logits(loaded, x), predict_logits(attn, x))
    # a config written by the pre-temperature code (no key) builds a temperature-1 layer
    restored = am.AttentionPool.from_config({k: layer["config"][k] for k in old_keys})
    assert restored.temperature == 1.0 and "temperature" not in restored.get_config()


def test_temperature_config_and_checks():
    pool = am.AttentionPool(hidden=4, n_maps=2, temperature=4.0)
    assert pool.get_config()["temperature"] == 4.0
    assert am.AttentionPool.from_config(pool.get_config()).temperature == 4.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            am.AttentionPool(hidden=4, temperature=bad)
    with pytest.raises(ValueError, match="not built"):
        pool.shared_copy(2.0)
    assert check_taus([1, 2]) == (1.0, 2.0)
    for bad in ([], [0, 1], [1, 1], [float("inf")]):
        with pytest.raises(ValueError):
            check_taus(bad)


def test_models_without_attention(model, images):
    assert not has_attention(model)
    with pytest.raises(ValueError, match="attention_pool"):
        with_attention_temperature(model, 2.0)
    assert attention_stats(model, images) is None


# ---------------------------------------------------------------------------
# aas_pgd
# ---------------------------------------------------------------------------

def test_aas_pgd_is_pgd_without_attention(model, images, labels, x):
    clean = (np.round(images * 255.0) / 255.0).astype(np.float32)
    for eps in (1 / 255, 4 / 255):
        adv, info = aas_pgd(clean, model, eps, label=labels, steps=5, seed=3, return_info=True)
        np.testing.assert_array_equal(adv, pgd(clean, model, eps, label=labels, steps=5, seed=3))
        assert info["taus"].shape == (5, len(clean)) and (info["taus"] == 1.0).all()
    gap = tiny_model("gap")
    np.testing.assert_array_equal(aas_pgd(x, gap, 2 / 255, steps=4), pgd(x, gap, 2 / 255, steps=4))


def test_aas_pgd_with_the_grid_one_is_pgd(saturated, x):
    np.testing.assert_array_equal(aas_pgd(x, saturated, 2 / 255, steps=4, taus=(1,)),
                                  pgd(x, saturated, 2 / 255, steps=4))


def test_aas_pgd_takes_the_proposal_the_unscaled_model_likes_best(saturated, x):
    eps, steps = 4 / 255, 1
    labels = predict_logits(saturated, x).argmax(1)
    adv, info = aas_pgd(x, saturated, eps, label=labels, steps=steps, seed=5, return_info=True)
    x0 = tf.constant(x)
    start = tf.clip_by_value(tf.clip_by_value(x0 + tf.random.stateless_uniform(x0.shape, seed=[5, 0], minval=-eps,
                                                                                 maxval=eps), x0 - eps, x0 + eps), 0, 1)
    alpha = 2.5 * eps / steps
    proposals, losses = [], []
    for tau in DEFAULT_TAUS:
        net = get_margin_free(saturated if tau == 1 else with_attention_temperature(saturated, tau))
        g = _loss_gradient(net, start, tf.constant(labels, tf.int32))
        p = np.clip(np.clip((start + alpha * tf.sign(g)).numpy(), x - eps, x + eps), 0, 1)
        proposals.append(p)
        losses.append(-margins(saturated, p, labels))
    choice = np.argmax(np.stack(losses), axis=0)
    np.testing.assert_array_equal(info["taus"][0], np.asarray(DEFAULT_TAUS)[choice])
    np.testing.assert_allclose(adv, np.stack([proposals[c][i] for i, c in enumerate(choice)]), atol=1e-7)
    assert (choice > 0).any()                    # a rescaled model wins somewhere on the saturated model


def test_aas_pgd_is_at_least_as_strong_as_pgd_on_an_attention_model(saturated, x):
    labels = predict_logits(saturated, x).argmax(1)
    max_weight, _ = attention_stats(saturated, x)
    assert max_weight.max() > 0.9                 # some maps are saturated
    for levels in (1, 2, 4):
        eps = levels / 255
        p = quantise(pgd(x, saturated, eps, label=labels, steps=10), x, eps)
        a = quantise(aas_pgd(x, saturated, eps, label=labels, steps=10), x, eps)
        mp, ma = margins(saturated, p, labels), margins(saturated, a, labels)
        assert (ma < 0).sum() >= (mp < 0).sum()
        assert ma.mean() <= mp.mean() + 0.05
        assert np.abs(a - x).max() <= eps + 1e-6


def test_aas_pgd_arguments_and_purity(saturated, x):
    copy = x.copy()
    single = aas_pgd(x[0], saturated, 2 / 255, steps=2)
    assert single.shape == x[0].shape and single.dtype == np.float32
    np.testing.assert_array_equal(x, copy)
    np.testing.assert_array_equal(aas_pgd(x, saturated, 2 / 255, steps=2), aas_pgd(x, saturated, 2 / 255, steps=2))
    for kwargs in ({"steps": 0}, {"loss": "hinge"}, {"taus": ()}):
        with pytest.raises(ValueError):
            aas_pgd(x, saturated, 2 / 255, **kwargs)
    with pytest.raises(ValueError):
        aas_pgd(x, saturated, 0.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_map_saturation_on_known_maps():
    uniform = np.full((1, 8, 8, 1), 1 / 64)
    one_hot = np.zeros((1, 8, 8, 1))
    one_hot[0, 3, 4, 0] = 1.0
    max_weight, entropy = map_saturation(np.concatenate([uniform, one_hot], axis=0))
    np.testing.assert_allclose(max_weight[:, 0], [1 / 64, 1.0])
    np.testing.assert_allclose(entropy[:, 0], [1.0, 0.0], atol=1e-12)


def test_gradient_stats(saturated, x):
    labels = predict_logits(saturated, x).argmax(1)
    stats = gradient_stats(saturated, x, labels, taus=(1, 4))
    g = _loss_gradient(get_margin_free(saturated), tf.constant(x), tf.constant(labels, tf.int32)).numpy()
    g = g.reshape(len(x), -1)
    np.testing.assert_allclose(stats["zero_share"], (g == 0).mean(axis=1))
    np.testing.assert_allclose(stats["l2"], np.linalg.norm(g, axis=1), rtol=1e-5)
    np.testing.assert_allclose(stats["l2_tau1"], stats["l2"])
    assert (stats["nonfinite_share"] == 0).all() and set(stats) >= {"linf", "l1", "l2_tau4"}
    # a saturated softmax gives large, peaky gradients; the scaled model's are smaller
    assert np.median(stats["l2_tau4"]) < np.median(stats["l2"])


# ---------------------------------------------------------------------------
# scripts/24_gradient_masking.py
# ---------------------------------------------------------------------------

def _rate(k, n):
    return {"rate": k / n, "k": k, "n": n}


def _summary(steps=(0.2, 0.5, 0.9), budget=(0.3, 0.6, 1.0), compare=None, zero=0.1, nonfinite=0.0, all_zero=0,
             max_weight=None):
    compare = compare or {"pgd": 0.5, "aas_pgd": 0.5, "rr_pgd": 0.52}
    out = {"pgd_success_vs_steps": {"1levels": {str(n): _rate(int(r * 100), 100)
                                                for n, r in zip((10, 50, 200), steps)}},
           "pgd_success_vs_budget": {str(L): _rate(int(r * 100), 100) for L, r in zip((1, 4, 255), budget)},
           "attack_comparison": {f"{a}@1levels": _rate(int(r * 100), 100) for a, r in compare.items()},
           "gradient_stats": {"clean": {"max_nonfinite_share": nonfinite, "images_all_zero_gradient": all_zero,
                                        "median_zero_share": zero}},
           "attention_stats": {} if max_weight is None else {"clean": {"median_max_weight": max_weight}}}
    return out


def test_monotonicity_and_masking_flags(masking_script):
    s = masking_script.settings_from(levels=(1,), steps_levels=(1,), budget_grid=(1, 4, 255))
    flags, skipped = masking_script.masking_flags(_summary(), s)
    assert flags == [] and skipped == []
    flags, _ = masking_script.masking_flags(_summary(steps=(0.2, 0.6, 0.55)), s)
    assert flags == ["pgd_steps_not_monotone@1levels"]
    flags, _ = masking_script.masking_flags(_summary(steps=(0.2, 0.6, 0.595)), s)       # within the tolerance
    assert flags == []
    flags, _ = masking_script.masking_flags(_summary(budget=(0.3, 0.2, 0.98)), s)
    assert flags == ["pgd_budget_not_monotone", "unbounded_pgd_below_0.99"]
    flags, _ = masking_script.masking_flags(_summary(compare={"pgd": 0.5, "aas_pgd": 0.6, "rr_pgd": 0.7}), s)
    assert flags == ["aas_pgd_beats_pgd@1levels", "random_restarts_beat_pgd@1levels"]
    flags, _ = masking_script.masking_flags(_summary(zero=0.8, nonfinite=0.01, all_zero=2, max_weight=0.97), s)
    assert flags == ["nonfinite_gradients@clean", "all_zero_gradient_images@clean", "zero_gradients@clean",
                     "attention_saturated@clean"]
    no_unbounded = masking_script.settings_from(levels=(1,), steps_levels=(1,), budget_grid=(1, 4))
    summary = _summary()
    del summary["pgd_success_vs_budget"]["255"]
    flags, skipped = masking_script.masking_flags(summary, no_unbounded)
    assert flags == [] and skipped == ["unbounded_budget (add 255 to --budget-grid)"]
    assert masking_script.drops({"a": 0.5, "b": 0.4, "c": 0.6}, 0.01) == [("a", "b")]


def test_monotonicity_against_the_best_earlier_rate_on_sorted_grids(masking_script):
    """A slow slide (each step within the tolerance, the total beyond it) fails, and grids typed out of order
    are checked in ascending order."""
    drops = masking_script.drops
    assert drops({"10": 0.600, "50": 0.592, "200": 0.584}, 0.01) == [("10", "200")]
    assert drops({"10": 0.600, "50": 0.595, "200": 0.598}, 0.01) == []
    assert drops({"1": float("nan"), "2": 0.5, "4": 0.3}, 0.01) == [("2", "4")]
    s = masking_script.settings_from(levels=(1,), steps_levels=(1,), steps_grid=(200, 50, 10),
                                     budget_grid=(255, 1, 2, 4))
    records = []
    for test, levels, steps, rate in ([("steps", 1, n, r) for n, r in ((10, 0.55), (50, 0.6), (200, 0.5))]
                                      + [("budget", L, 50, r) for L, r in ((1, 0.3), (2, 0.5), (4, 0.7), (255, 1.0))]):
        records += [{"test": test, "attack": "pgd", "budget_levels": levels, "steps": steps, "success": k < rate * 20,
                     "condition": None, "tau_mean": np.nan, "tau_not1_share": np.nan} for k in range(20)]
    summary = masking_script.summarise(pd.DataFrame(records), s)
    assert list(summary["pgd_success_vs_budget"]) == ["1", "2", "4", "255"]
    assert list(summary["pgd_success_vs_steps"]["1levels"]) == ["10", "50", "200"]
    assert summary["monotonicity"]["budget_drops"] == []
    assert summary["monotonicity"]["steps_drops"] == {"1levels": [("50", "200")]}
    assert "pgd_budget_not_monotone" not in summary["masking_flags"]
    assert "pgd_steps_not_monotone@1levels" in summary["masking_flags"]


def _write_images(tmp_path, images, labels):
    records = []
    for i, (img, label) in enumerate(zip(images, labels)):
        path = tmp_path / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        records.append({"link_path": str(path), "relpath": f"test/{label}/{label}_{i}_l_c.png",
                        "true_idx": int(label), "true_subject": 1000 + int(label), "eye": "left"})
    path = tmp_path / "correct_test_tiny.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


SMALL = ["--steps-grid", "1,3", "--steps-levels", "2", "--budget-grid", "1,4,255", "--budget-steps", "3",
         "--levels", "1,2", "--steps", "2", "--restarts", "2", "--batch-size", "2"]


@pytest.mark.parametrize("kind", ["attn", "parts", "plain"])
def test_24_end_to_end(masking_script, tmp_path, kind, model, images, labels, x):
    if kind == "plain":                       # conftest's trained model, no attention pool
        net, clean, truth = model, (np.round(images * 255.0) / 255.0).astype(np.float32), labels
    else:
        net = tiny_model(kind, scale=SATURATION)
        clean, truth = x, predict_logits(net, x).argmax(1)
    ckpt = tmp_path / f"{kind}.keras"
    net.save(ckpt)
    correct = _write_images(tmp_path, clean[:3], truth[:3])
    out = tmp_path / "out"
    masking_script.main(["--ckpt", str(ckpt), "--correct-csv", str(correct), "--out-dir", str(out),
                         "--max-images", "3", "--per-class", "3", "--tag", kind] + SMALL)
    rows = pd.read_csv(out / f"rows_masking_{kind}.csv")
    summary = json.loads((out / f"summary_masking_{kind}.json").read_text())
    n = 3
    counts = rows.test.value_counts().to_dict()
    assert counts == {"steps": n * 2, "budget": n * 3, "compare": n * 3 * 2, "gradient": n * 3}
    assert rows.image_id.nunique() == n and {"relpath", "link_path", "true_subject"} <= set(rows.columns)
    attacked = rows[rows.test != "gradient"]
    assert (attacked.pert_linf_levels <= attacked.budget_levels).all()
    assert (attacked.success == (attacked.pred != attacked.true_idx)).all()
    compare = rows[rows.test == "compare"].set_index(["image_id", "budget_levels", "attack"])
    for (image, levels, attack), row in compare.iterrows():
        if attack == "rr_pgd":                # restart 0 is the pgd run: the restarts never lose an image
            assert row.margin <= compare.loc[(image, levels, "pgd")].margin + 1e-6
    gradient = rows[rows.test == "gradient"]
    assert set(gradient.condition) == {"clean", "pgd@1", "pgd@2"}
    assert gradient.grad_l2.notna().all() and (gradient.grad_nonfinite_share == 0).all()
    assert summary["git_commit"] and summary["n_images"] == n and summary["has_attention"] == (kind != "plain")
    assert summary["arch"] == ("baseline" if kind == "plain" else kind)
    assert isinstance(summary["masking_flags"], list)
    assert summary["masking_check_passed"] == (not summary["masking_flags"])
    assert list(summary["pgd_success_vs_budget"]) == ["1", "4", "255"]
    assert list(summary["pgd_success_vs_steps"]["2levels"]) == ["1", "3"]
    assert set(summary["attack_comparison"]) == {f"{a}@{L}levels" for a in ("pgd", "aas_pgd", "rr_pgd") for L in (1, 2)}
    assert summary["settings"]["taus"] == list(DEFAULT_TAUS) and summary["checks_skipped"] == []
    assert set(summary["seconds"]) == {"steps", "budget", "pgd", "aas_pgd", "rr_pgd", "gradient"}
    assert summary["batch_size"] == 2 and summary["n_batches"] == 2 and summary["seconds_per_batch"] > 0
    aas = rows[(rows.test == "compare") & (rows.attack == "aas_pgd")]
    if kind == "plain":
        assert summary["attention_stats"] == {} and "attn_max_weight" not in rows.columns
        assert (aas.tau_not1_share == 0).all()
        pgd_rows = rows[(rows.test == "compare") & (rows.attack == "pgd")]
        np.testing.assert_array_equal(aas.margin.to_numpy(), pgd_rows.margin.to_numpy())      # aas_pgd is pgd
        assert not any(f.startswith("aas_pgd_beats") for f in summary["masking_flags"])
    else:
        assert set(summary["attention_stats"]) == {"clean", "pgd@1", "pgd@2"}
        assert gradient.attn_max_weight.between(1 / 64 - 1e-6, 1).all()
        assert gradient.attn_entropy_norm.between(0, 1 + 1e-6).all()
        names = am.part_names(net) if kind == "parts" else ["eye"]
        assert {f"attn_max_{m}" for m in names} <= set(rows.columns)
        assert {f"grad_l2_tau{t:g}" for t in DEFAULT_TAUS} <= set(rows.columns)
        assert "median_l2_tau16" in summary["gradient_stats"]["clean"]


def test_24_refuses_bad_settings_and_three_input_models(masking_script, tmp_path, x):
    net = tiny_model("attnet_gate")
    ckpt = tmp_path / "gate.keras"
    net.save(ckpt)
    correct = _write_images(tmp_path, x[:2], [0, 1])
    common = ["--ckpt", str(ckpt), "--correct-csv", str(correct), "--out-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="dual-input"):
        masking_script.main(common + SMALL)
    for bad in (["--levels", "0"], ["--budget-grid", "1,256"], ["--steps-grid", "10,10"], ["--taus", "1,0"],
                ["--restarts", "0"], ["--step-size", "-1"]):
        with pytest.raises(SystemExit):
            masking_script.main(common + bad)


# ---------------------------------------------------------------------------
# scripts/14_adaptive_attacks.py --attacks ...,aas_pgd
# ---------------------------------------------------------------------------

def test_14_runs_aas_pgd(adaptive_script, model, images, labels, saturated, x):
    clean = (np.round(images * 255.0) / 255.0).astype(np.float32)
    defences = adaptive_script.make_defences(["jpeg"])
    base = [{"image_id": i, "true_idx": int(label)} for i, label in enumerate(labels[:3])]
    timings = {}
    rows = pd.DataFrame(adaptive_script.run_batch(model, clean[:3], labels[:3], base, [2 / 255], defences,
                                                  attacks=["pgd", "aas_pgd"], xai_names=(), steps=3,
                                                  timings=timings))
    assert set(rows.attack) == {"pgd", "aas_pgd"} and set(timings) == {"pgd", "aas_pgd", "evaluate"}
    assert not rows.adaptive.any()                     # aas_pgd does not know the defence, like pgd
    by_attack = {a: g.reset_index(drop=True) for a, g in rows.groupby("attack")}
    for col in ("margin_undefended", "margin_defended", "pert_l2"):       # no attention: aas_pgd is pgd
        np.testing.assert_array_equal(by_attack["aas_pgd"][col], by_attack["pgd"][col])
    summary = adaptive_script.summarise(rows)
    assert list(summary["success_after_defence"]) == ["pgd@0.00784314+jpeg", "aas_pgd@0.00784314+jpeg"]

    truth = predict_logits(saturated, x).argmax(1)
    rows = pd.DataFrame(adaptive_script.run_batch(saturated, x[:3], truth[:3], base, [2 / 255], defences,
                                                  attacks=["aas_pgd"], xai_names=(), steps=3, taus=(1, 8)))
    assert len(rows) == 3 and (rows.pert_linf_levels <= 2).all()


def test_14_end_to_end_with_aas_pgd(adaptive_script, tmp_path, x):
    net = tiny_model("attn", scale=SATURATION)
    truth = predict_logits(net, x).argmax(1)
    ckpt = tmp_path / "attn.keras"
    net.save(ckpt)
    correct = _write_images(tmp_path, x[:2], truth[:2])
    out = tmp_path / "out"
    adaptive_script.main(["--ckpt", str(ckpt), "--correct-csv", str(correct), "--out-dir", str(out),
                          "--epsilons", "0.0078431", "--attacks", "pgd,aas_pgd", "--defences", "bit_depth",
                          "--steps", "2", "--xai", "none", "--taus", "1,4", "--per-class", "2", "--tag", "aas"])
    rows = pd.read_csv(out / "rows_adaptive_aas.csv")
    summary = json.loads((out / "summary_adaptive_aas.json").read_text())
    assert set(rows.attack) == {"pgd", "aas_pgd"} and len(rows) == 4
    assert summary["aas_taus"] == [1.0, 4.0] and set(summary["cost_relative_to_pgd"]) == {"pgd", "aas_pgd"}
    with pytest.raises(SystemExit):
        adaptive_script.main(["--ckpt", str(ckpt), "--correct-csv", str(correct), "--taus", "0"])
