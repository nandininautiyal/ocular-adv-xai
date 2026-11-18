"""Anatomy-guided heads (src/models/anatomy_model.py) on the tiny CPU backbone: every arch builds, runs,
saves and reloads; the inference helpers (margin-free logits, LayerCAM, embeddings) work on each; the parts
receipt is exact; the mask loss trains the attention map; and 03 trains every arch end to end, including a
--resume continuation, writing the plain inference model as its checkpoint. 04 and 15 feed attnet_gate's
region input."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml
from PIL import Image

from conftest import REPO, make_images
from src.data import masked_generator as mg
from src.eval.verification import embed
from src.models import anatomy_model as am
from src.models.arcface_inference import get_margin_free, load_checkpoint, predict_logits
from src.xai.attribution import eigencam, layercam
from src.xai import seg_masks as sm

NEW_ARCHS = ("gap", "attn", "attn_mask", "parts", "attnet_gate")
N_CLASSES = 3


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load("03_train_baseline")


def tiny_cfg(n_classes=N_CLASSES, **training):
    """configs/anatomy.yml with the tiny backbone, every layer trainable and small heads."""
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"backbone": "tiny", "fine_tune_at": 0, "dense_units": 16, "n_classes": n_classes})
    cfg["anatomy"]["hidden"] = 8
    cfg["training"].update({"learning_rate": 1e-3, **training})
    return cfg


def regions_for(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, (n, 8, 8, 1)).astype(np.float32)


def inputs_for(model, x, label=None):
    label = np.zeros((len(x), am_n_classes(model)), np.float32) if label is None else label
    return [x, label] + ([regions_for(len(x))] if len(model.inputs) == 3 else [])


def am_n_classes(model):
    return int(model.inputs[1].shape[-1])


@pytest.fixture(scope="module")
def built():
    tf.keras.utils.set_random_seed(0)
    return {arch: am.build_anatomy_model(tiny_cfg(), arch) for arch in NEW_ARCHS}


@pytest.fixture(scope="module")
def x():
    return make_images(4)


# ------------------------------------------------------------------ models

def test_every_arch_builds_and_runs(built, x):
    for arch, model in built.items():
        assert am.model_arch(model) == arch and model.name == f"anatomy_{arch}"
        assert len(model.inputs) == (3 if arch == "attnet_gate" else 2)
        assert "block14_sepconv2_act" in [layer.name for layer in model.layers]
        probs = model.predict_on_batch(inputs_for(model, x))
        assert probs.shape == (4, N_CLASSES)
        np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)
        assert model.optimizer is not None and model.compiled_loss is not None
    assert am.part_names(built["parts"]) == ["iris", "sclera", "lid", "brow", "skin"]
    assert built["parts"].get_layer("attention_pool").n_maps == 5
    assert am.region_classes(built["attnet_gate"]) == ["iris", "sclera", "caruncle", "lid", "brow"]


def test_baseline_is_the_vendored_model(monkeypatch):
    monkeypatch.setattr(am, "build_model", lambda cfg: ("vendored", cfg["model"]["n_classes"]))
    assert am.build_anatomy_model(tiny_cfg(), "baseline") == ("vendored", N_CLASSES)
    with pytest.raises(ValueError, match="unknown arch"):
        am.build_anatomy_model(tiny_cfg(), "resnet")
    assert am.model_arch(_tiny_baseline({"model": {"n_classes": 3}, "training": {"learning_rate": 1e-3}})) == "baseline"


def test_save_and_reload_give_identical_outputs(built, x, tmp_path):
    for arch, model in built.items():
        path = tmp_path / f"{arch}.keras"
        model.save(path)
        loaded = load_checkpoint(str(path))
        assert am.model_arch(loaded) == arch          # attn and attn_mask told apart by the saved name
        np.testing.assert_array_equal(loaded.predict_on_batch(inputs_for(loaded, x)),
                                      model.predict_on_batch(inputs_for(model, x)))
        for got, want in zip(loaded.get_weights(), model.get_weights()):
            np.testing.assert_array_equal(got, want)
    assert am.part_names(load_checkpoint(str(tmp_path / "parts.keras"))) == am.part_names(built["parts"])
    assert am.region_classes(load_checkpoint(str(tmp_path / "attnet_gate.keras"))) == \
        am.region_classes(built["attnet_gate"])


def test_inference_helpers_work_on_every_arch(built, x):
    for arch, model in built.items():
        # attnet_gate: one region bound for every image (broadcast), so single images work too
        target = model if arch != "attnet_gate" else am.bind_region(model, regions_for(1))
        zeros = np.zeros((len(x), N_CLASSES), np.float32)
        logits = predict_logits(target, x)
        np.testing.assert_allclose(tf.nn.softmax(logits).numpy(), target.predict_on_batch([x, zeros]), atol=1e-5)
        cam = layercam(target, x[0], class_idx=0, target_layer="block14_sepconv2_act", size=None)
        assert cam.shape == (8, 8) and 0 <= cam.min() and cam.max() <= 1
        assert eigencam(target, x[:2], size=None).shape == (2, 8, 8)
        features, _ = get_margin_free(target, "block14_sepconv2_act")(x)
        assert features.shape == (len(x), 8, 8, 16)
        e = embed(target, x, batch_size=3)
        assert e.shape == (len(x), 16)
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, rtol=1e-5)
        cos = e @ tf.nn.l2_normalize(am._find_arcface(target).W, 0).numpy()
        np.testing.assert_allclose(cos * 64, logits, atol=1e-3)


def test_attnet_gate_binding_and_region_embeddings(built, x):
    model = built["attnet_gate"]
    regions = regions_for(len(x))
    zeros = np.zeros((len(x), N_CLASSES), np.float32)
    bound = am.bind_region(model, regions)
    np.testing.assert_allclose(bound.predict_on_batch([x, zeros]), model.predict_on_batch([x, zeros, regions]),
                               atol=1e-6)
    one = am.bind_region(model, regions[1])                     # one region, broadcast over the batch
    np.testing.assert_allclose(one.predict_on_batch([x[1:2], zeros[:1]]),
                               model.predict_on_batch([x[1:2], zeros[:1], regions[1:2]]), atol=1e-6)
    np.testing.assert_allclose(am.embed_regions(model, x, regions, batch_size=3), embed(bound, x), atol=1e-6)
    with pytest.raises(ValueError):
        am.embed_regions(built["gap"], x, regions)
    with pytest.raises(ValueError):
        am.bind_region(built["gap"], regions)


def test_region_gate_applies_attnets_gains():
    gate = am.RegionGate(5.0, 0.2)
    features = tf.ones((1, 8, 8, 2))
    region = np.zeros((1, 8, 8, 1), np.float32)
    region[0, :4] = 1
    region[0, 4, 0] = 0.5
    out = gate([features, region]).numpy()
    assert out[0, 0, 0, 0] == pytest.approx(5.0) and out[0, 7, 7, 1] == pytest.approx(0.2)
    assert out[0, 4, 0, 0] == pytest.approx(0.2 + 4.8 * 0.5)


def test_attention_maps(built, x):
    for arch in ("attn", "attn_mask", "parts"):
        maps = am.attention_maps(built[arch], x, batch_size=3)
        assert maps.shape == (len(x), 8, 8, 5 if arch == "parts" else 1) and maps.dtype == np.float32
        np.testing.assert_allclose(maps.sum(axis=(1, 2)), 1.0, rtol=1e-5)
    for arch in ("gap", "attnet_gate"):
        with pytest.raises(ValueError, match="attention_pool"):
            am.attention_maps(built[arch], x)


def test_part_receipt_sums_to_the_margin_free_cosine(built, x):
    model = built["parts"]
    tf.keras.utils.set_random_seed(1)
    model.set_weights([w + np.random.default_rng(1).normal(0, 0.3, w.shape).astype(np.float32)
                       for w in model.get_weights()])           # a non-trivial head
    class_ids = np.array([0, 1, 2, 1])
    receipt = am.part_receipt(model, x, class_ids, batch_size=3)
    assert receipt.shape == (len(x), 5) and receipt.dtype == np.float32
    cosine = predict_logits(model, x)[np.arange(len(x)), class_ids] / 64.0
    np.testing.assert_allclose(receipt.sum(axis=1), cosine, atol=2e-6)
    assert np.abs(receipt).max() > 1e-3                          # the terms are not all zero
    logit_terms = am.part_receipt(model, x, class_ids, batch_size=3, units="logit")
    assert am.arcface_scale(model) == 64.0
    np.testing.assert_allclose(logit_terms, receipt * 64.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(logit_terms.sum(axis=1), cosine * 64.0, atol=2e-4)
    with pytest.raises(ValueError, match="units"):
        am.part_receipt(model, x, class_ids, units="prob")
    with pytest.raises(ValueError):
        am.part_receipt(built["attn"], x, class_ids)
    with pytest.raises(ValueError, match="class ids"):
        am.part_receipt(model, x, class_ids[:2])


def test_settings_defaults_and_checks():
    s = am.anatomy_settings({})
    assert list(s["part_groups"]) == ["iris", "sclera", "lid", "brow", "skin"]
    assert (s["hidden"], s["mask_loss_weight"], s["eps"], s["min_part_area"]) == (256, 1.0, 1e-3, 0.005)
    assert (s["gate_inside"], s["gate_outside"]) == (5.0, 0.2)
    assert am.eye_groups(s) == ["iris", "sclera", "lid", "brow"]
    with pytest.raises(ValueError, match="is in part groups"):
        am.anatomy_settings({"anatomy": {"part_groups": {"a": ["iris"], "b": ["iris"], "skin": ["skin"]}}})
    with pytest.raises(ValueError, match="skin"):
        am.anatomy_settings({"anatomy": {"part_groups": {"a": ["iris"]}}})
    with pytest.raises(ValueError, match="unknown anatomy settings"):
        am.anatomy_settings({"anatomy": {"mask_loss_wieght": 2.0}})
    order = am.anatomy_settings({"anatomy": {"part_groups": {"brow": ["brow"], "skin": ["skin"], "iris": ["iris"]}}})
    assert list(order["part_groups"]) == ["brow", "skin", "iris"]


def test_anatomy_config_is_cross_session_plus_the_anatomy_section():
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        anatomy = yaml.safe_load(f)
    with open(os.path.join(REPO, "configs", "cross_session.yml")) as f:
        cross = yaml.safe_load(f)
    assert set(anatomy) == set(cross) | {"anatomy"}
    for section in cross:
        assert anatomy[section] == cross[section]
    assert am.anatomy_settings(anatomy) == am.anatomy_settings({})          # the file holds the defaults


# ------------------------------------------------------------------ mask loss

def test_mask_kl():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.01, 1, (3, 8, 8, 2))
    t /= t.sum(axis=(1, 2), keepdims=True)
    a = rng.uniform(0.01, 1, (3, 8, 8, 2))
    a /= a.sum(axis=(1, 2), keepdims=True)
    w = np.array([[1, 0], [1, 1], [0, 0]], np.float32)
    kl = (t * np.log(t / a)).sum(axis=(1, 2))
    assert float(am.mask_kl(t, a, w)) == pytest.approx((kl * w).sum() / w.sum(), rel=1e-5)
    assert float(am.mask_kl(t, t, np.ones((3, 2)))) == pytest.approx(0.0, abs=1e-6)
    assert float(am.mask_kl(t, a, np.zeros((3, 2)))) == 0.0


def _toy_masks(n):
    """Eye targets on the coloured disk of each conftest image (make_images): content the 1 x 1 attention
    MLP can find, as the eye in a real crop."""
    yy, xx = np.mgrid[0:256, 0:256]
    eye = np.stack([((yy - (48 + 40 * k)) ** 2 + (xx - (200 - 30 * k)) ** 2 < 30 ** 2) for k in range(n)])
    frac = eye.reshape(n, 8, 32, 8, 32).mean(axis=(2, 4))[..., None]
    targets = (frac + 1e-3) / (frac + 1e-3).sum(axis=(1, 2), keepdims=True)
    return targets.astype(np.float32), np.ones((n, 1), np.float32)


def test_mask_loss_pulls_the_attention_map_to_the_target(x):
    y = np.eye(N_CLASSES, dtype=np.float32)[[0, 1, 2, 0]]
    targets, weights = _toy_masks(len(x))
    after = {}
    for weight in (0.0, 10.0):                           # the same training without and with the mask loss
        tf.keras.utils.set_random_seed(0)
        cfg = tiny_cfg(learning_rate=1e-2)
        cfg["anatomy"]["hidden"] = 16
        net = am.build_anatomy_model(cfg, "attn_mask")
        trainer = am.make_trainer(net, cfg, mask_loss_weight=weight)
        before = float(am.mask_kl(targets, am.attention_maps(net, x), weights))
        hist = trainer.fit((x, y, targets, weights), y, batch_size=4, epochs=80, verbose=0)
        after[weight] = float(am.mask_kl(targets, am.attention_maps(net, x), weights))
        assert {"loss", "ce", "mask_loss", "accuracy"} <= set(hist.history)
        np.testing.assert_allclose(hist.history["loss"][-1], hist.history["ce"][-1] + weight * hist.history["mask_loss"][-1],
                                   rtol=1e-4)
        assert len(trainer.trainable_variables) == len(net.trainable_variables)
    assert after[10.0] < 0.6 * before and after[10.0] < 0.5 * after[0.0], (before, after)


def test_trainer_cross_entropy_reads_the_logits_like_fit(x):
    """In graph mode the compiled CE reads the logits of ArcFace's Softmax op: a true class far below 1e-7
    gives the CE of the logits (> 16.1), not the clipped -log(1e-7)."""
    tf.keras.utils.set_random_seed(0)
    cfg = tiny_cfg()
    net = am.build_anatomy_model(cfg, "parts")
    trainer = am.make_trainer(net, cfg, mask_loss_weight=0.0)
    arcface = am._find_arcface(net)
    sub = tf.keras.Model(net.inputs[0], arcface.input[0])
    e = sub(x[:2], training=False).numpy()
    w = np.zeros((e.shape[1], N_CLASSES), np.float32)
    w[:, 0], w[:, 1], w[:, 2] = -e.mean(0), e.mean(0), e[0] - e[1]     # the true class 0 points away
    arcface.W.assign(w)
    y = np.eye(N_CLASSES, dtype=np.float32)[[0, 0]]
    cos = tf.nn.l2_normalize(e, 1).numpy() @ tf.nn.l2_normalize(w, 0).numpy()
    target = np.cos(np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7)) + arcface.m)
    logits = arcface.s * (cos * (1 - y) + target * y)
    want = float(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits)))
    assert want > 20
    targets, weights = _toy_masks(2)
    targets = np.repeat(targets, 5, axis=-1)
    weights = np.ones((2, 5), np.float32)
    got = trainer.evaluate((x[:2], y, targets, weights), y, batch_size=2, verbose=0, return_dict=True)
    assert got["ce"] == pytest.approx(want, rel=1e-4)


def test_trainer_saves_and_restores_through_the_inference_model(x, tmp_path):
    tf.keras.utils.set_random_seed(0)
    cfg = tiny_cfg()
    net = am.build_anatomy_model(cfg, "attn_mask")
    trainer = am.make_trainer(net, cfg)
    y = np.eye(N_CLASSES, dtype=np.float32)[[0, 1, 2, 0]]
    trainer.fit((x, y, *_toy_masks(4)), y, batch_size=4, epochs=1, verbose=0)
    trainer.save(str(tmp_path / "m.keras"))
    loaded = load_checkpoint(str(tmp_path / "m.keras"))
    assert loaded.name == "anatomy_attn_mask" and len(loaded.inputs) == 2 and am.model_arch(loaded) == "attn_mask"
    zeros = np.zeros((4, N_CLASSES), np.float32)
    np.testing.assert_array_equal(loaded.predict_on_batch([x, zeros]), net.predict_on_batch([x, zeros]))
    np.testing.assert_array_equal(trainer.predict_on_batch([x, zeros]), net.predict_on_batch([x, zeros]))
    assert trainer.count_params() == net.count_params()
    saved = trainer.get_weights()
    trainer.set_weights([w * 0 for w in saved])
    assert all((w == 0).all() for w in net.get_weights())
    trainer.set_weights(loaded.get_weights())
    for got, want in zip(net.get_weights(), loaded.get_weights()):
        np.testing.assert_array_equal(got, want)


# ------------------------------------------------------------------ 03 end to end

TREE = {"train": {"1141": 4, "1142": 4, "7000": 4}, "validation": {"1141": 1, "1142": 1, "7000": 1},
        "test": {"1141": 1, "1142": 1, "7000": 1}}


def mask_for(k):
    """A 64 x 64 label map with an eye blob that moves with k (all six classes present)."""
    lab = np.zeros((64, 64), np.uint8)
    r, c = 16 + 4 * (k % 5), 12 + 6 * (k % 4)
    lab[r:r + 24, c:c + 36] = 2                         # sclera
    lab[r + 6:r + 18, c + 12:c + 24] = 1                # iris
    lab[r - 4:r, c:c + 36] = 3                          # lid
    lab[r + 8:r + 14, c - 3:c] = 4                      # caruncle
    lab[2:8, 10:50] = 5                                 # brow
    return lab


@pytest.fixture
def split_tree(tmp_path):
    """02-style class-folder tree (tiny PNGs) and a masks64 file for every image."""
    rng = np.random.default_rng(0)
    rels, labels, k = [], [], 0
    for split, classes in TREE.items():
        for name, n in classes.items():
            d = tmp_path / "tree" / split / name
            d.mkdir(parents=True)
            for i in range(n):
                frame = 1001 + i + 100 * list(TREE).index(split)
                Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
                    d / f"cond__enrollment__{name}_{frame}_l_c.png")
                rels.append(f"cond/enrollment_left_eye/{name}/{name}_{frame}_l_c.png")
                labels.append(mask_for(k))
                k += 1
    masks = tmp_path / "masks64_t.npz"
    mg.save_masks64(str(masks), rels, np.stack(labels), sm.CLASS_NAMES, {"source": "test"})
    cfg_path = tmp_path / "tiny_anatomy.yml"
    cfg = tiny_cfg()
    cfg["training"].update({"epochs": 3})
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return tmp_path / "tree", masks, cfg_path


def _train(monkeypatch, tree, out, *extra):
    monkeypatch.setattr(sys, "argv", ["03_train_baseline.py", "--split-dir", str(tree), "--n-classes", "3",
                                      "--batch-size", "4", "--verbose", "0", "--tag", "t", "--out-dir", str(out),
                                      *map(str, extra)])
    train_script.main()
    path = out / "run_t.json"
    return json.loads(path.read_text()) if path.exists() else None


class Killed(RuntimeError):
    pass


class StopAfterLogging(tf.keras.callbacks.Callback):
    """Kill the run at the end of epoch 2, after its log row and callback state, before its backup."""

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            raise Killed()


@pytest.mark.parametrize("arch", NEW_ARCHS)
def test_03_trains_every_arch_and_resumes(split_tree, tmp_path, monkeypatch, capsys, arch):
    tree, masks, cfg_path = split_tree
    kill = [True]

    def callbacks(*args, f=train_script.make_callbacks, **kwargs):
        cbs = f(*args, **kwargs)
        return cbs[:-1] + [StopAfterLogging()] + cbs[-1:] if kill[0] else cbs

    monkeypatch.setattr(train_script, "make_callbacks", callbacks)
    out = tmp_path / "models"
    args = ["--cfg", cfg_path, "--arch", arch, "--resume", "--save-weights"]
    if arch in am.MASK_ARCHS:
        args += ["--masks", masks]
    with pytest.raises(Killed):
        _train(monkeypatch, tree, out, *args)
    assert train_script.backup_exists(str(out / "backup_t")) and (out / "Xception_ArcFace_t.keras").exists()

    kill[0] = False
    capsys.readouterr()
    run = _train(monkeypatch, tree, out, *args)
    printed = capsys.readouterr().out
    assert "resuming at epoch 2" in printed
    assert run["resumed"] and run["resumed_at_epoch"] == 2 and run["epochs_run"] == 3
    assert run["arch"] == arch and run["args"]["arch"] == arch and run["cfg"]["anatomy"] == run["anatomy"]
    assert run["anatomy"]["part_groups"] == am.DEFAULT_PART_GROUPS and run["anatomy"]["hidden"] == 8
    if arch in am.MASK_ARCHS:
        assert run["masks_sha256"] == hashlib.sha256(masks.read_bytes()).hexdigest()
        assert run["mask_coverage"] == {"train": {"n_images": 12, "n_with_mask": 12, "share": 1.0},
                                        "validation": {"n_images": 3, "n_with_mask": 3, "share": 1.0},
                                        "test": {"n_images": 3, "n_with_mask": 3, "share": 1.0}}
        assert run["mask_classes"] == list(sm.CLASS_NAMES)
    else:
        assert run["masks"] is None and run["mask_coverage"] is None
    log = pd.read_csv(out / "log_t.csv")
    assert log.epoch.tolist() == [0, 1, 2]
    if arch in am.MASK_LOSS_ARCHS:
        assert {"ce", "mask_loss", "val_ce", "val_mask_loss", "val_loss", "accuracy"} <= set(log.columns)
        np.testing.assert_allclose(log.val_loss, log.val_ce + log.val_mask_loss, rtol=1e-4)
        assert {"ce", "mask_loss"} <= set(pd.read_csv(out / "history_t.csv").columns)
    # every arm is selected on the validation CE: val_ce for the mask-loss archs, val_loss (= CE) for the others
    monitor = "val_ce" if arch in am.MASK_LOSS_ARCHS else "val_loss"
    assert run["monitor"] == monitor and train_script.monitor_for(arch) == monitor
    assert run["best_epoch"] == int(log.epoch[log[monitor].idxmin()]) + 1
    assert run["best_monitor"] == pytest.approx(log[monitor].min())
    assert run["best_val_loss"] == pytest.approx(float(log.val_loss[log[monitor].idxmin()]))
    if arch in am.MASK_LOSS_ARCHS:
        assert set(run["mask_area_below_min"]) == {"train", "validation", "test"}
    else:
        assert "mask_area_below_min" not in run
    assert not (out / "backup_t").exists()

    ckpt = out / "Xception_ArcFace_t.keras"              # the plain inference model, not the trainer
    model = load_checkpoint(str(ckpt))
    assert model.name == f"anatomy_{arch}" and am.model_arch(model) == arch
    assert len(model.inputs) == (3 if arch == "attnet_gate" else 2)
    results = json.loads((out / "results_t.json").read_text())
    assert 0 <= results["test_acc_margin_free"] <= 1
    fresh = am.build_anatomy_model(yaml.safe_load(cfg_path.read_text()), arch)   # --save-weights: same weights
    fresh.load_weights(str(out / "Xception_ArcFace_t.weights.h5"))
    for got, want in zip(fresh.get_weights(), model.get_weights()):
        np.testing.assert_array_equal(got, want)

    capsys.readouterr()                                  # the same command again: nothing to do
    _train(monkeypatch, tree, out, *args)
    assert "nothing to resume" in capsys.readouterr().out


def test_03_resume_with_another_mask_loss_weight_stops(split_tree, tmp_path, monkeypatch):
    tree, masks, cfg_path = split_tree
    kill = [True]

    def callbacks(*args, f=train_script.make_callbacks, **kwargs):
        cbs = f(*args, **kwargs)
        return cbs[:-1] + [StopAfterLogging()] + cbs[-1:] if kill[0] else cbs

    monkeypatch.setattr(train_script, "make_callbacks", callbacks)
    out = tmp_path / "models"
    args = ["--cfg", cfg_path, "--arch", "attn_mask", "--masks", masks, "--resume"]
    with pytest.raises(Killed):
        _train(monkeypatch, tree, out, *args, "--mask-loss-weight", "0.5")
    kill[0] = False
    log = (out / "log_t.csv").read_text()
    with pytest.raises(SystemExit, match=r"cfg\.anatomy\.mask_loss_weight: 0\.5 -> 0\.7"):
        _train(monkeypatch, tree, out, *args, "--mask-loss-weight", "0.7")
    assert train_script.backup_exists(str(out / "backup_t")) and (out / "log_t.csv").read_text() == log
    assert not (out / "run_t.json").exists()


def test_03_baseline_record_and_flags_are_unchanged(split_tree, tmp_path, monkeypatch, capsys):
    tree, masks, _ = split_tree
    monkeypatch.setattr(train_script, "build_model", lambda cfg: _tiny_baseline(cfg))
    run = _train(monkeypatch, tree, tmp_path / "models", "--epochs", "1")
    assert not {"arch", "anatomy", "masks", "masks_sha256", "mask_coverage"} & set(run)
    assert not set(train_script.ANATOMY_ARGS) & set(run["args"])        # the args of records before --arch
    assert "anatomy" not in run["cfg"] and "monitor" not in run
    for extra, message in ((["--arch", "parts"], "--arch parts needs --masks"),
                           (["--masks", masks], "--masks is used only by"),
                           (["--arch", "gap", "--masks", masks], "--masks is used only by"),
                           (["--arch", "attnet_gate", "--masks", masks, "--mask-loss-weight", "2"],
                            "--mask-loss-weight is used only by"),
                           (["--allow-missing-masks"], "--allow-missing-masks needs --masks")):
        capsys.readouterr()
        with pytest.raises(SystemExit) as stop:
            _train(monkeypatch, tree, tmp_path / "bad", *extra)
        assert stop.value.code == 2                                     # argparse's error exit
        assert message in capsys.readouterr().err, extra


def test_callbacks_and_best_epoch_follow_the_monitor(tmp_path):
    """The mask-loss archs are stopped, scheduled and checkpointed on val_ce, not on CE + lambda x KL."""
    t = {"early_stop_patience": 5, "reduce_lr_factor": 0.2, "reduce_lr_patience": 2, "min_lr": 1e-6}
    for monitor in ("val_loss", "val_ce"):
        cbs = train_script.make_callbacks(t, str(tmp_path / "m.keras"), str(tmp_path / "log.csv"),
                                          str(tmp_path / "backup"), monitor=monitor)
        assert [c.monitor for c in cbs[:3]] == [monitor] * 3
        state = next(c for c in cbs if isinstance(c, train_script.ResumeState))
        assert state.monitor == monitor
    # val_ce is lowest at epoch 2 and rises after it, while the total keeps falling with the KL term
    log = pd.DataFrame({"epoch": [0, 1, 2, 3], "val_ce": [2.0, 1.5, 1.6, 1.7], "val_mask_loss": [3.0, 2.0, 1.0, 0.5]})
    log["val_loss"] = log.val_ce + log.val_mask_loss
    assert train_script.best_epoch(log) == (4, pytest.approx(2.2))
    assert train_script.best_epoch(log, "val_ce") == (2, pytest.approx(3.5))
    assert train_script.best_epoch(log.drop(columns="val_ce"), "val_ce") == (None, None)
    assert [train_script.monitor_for(a) for a in ("baseline", "gap", "attn", "attnet_gate", "attn_mask", "parts")]         == ["val_loss"] * 4 + ["val_ce"] * 2


def _tiny_baseline(cfg):
    from conftest import build_tiny_model
    model = build_tiny_model(cfg["model"]["n_classes"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["training"]["learning_rate"]),
                  loss=tf.keras.losses.CategoricalCrossentropy())
    return model


def test_03_stops_on_missing_masks_unless_allowed(split_tree, tmp_path, monkeypatch, capsys):
    tree, masks, cfg_path = split_tree
    source = mg.load_label_source(str(masks))
    keep = [i for i, r in enumerate(source["relpaths"]) if not r.endswith("1002_l_c.png")]   # 3 train images
    partial = tmp_path / "masks64_partial.npz"
    mg.save_masks64(str(partial), [source["relpaths"][i] for i in keep], source["labels"][keep], source["class_names"])
    args = ["--cfg", cfg_path, "--arch", "parts", "--masks", partial, "--epochs", "1"]
    with pytest.raises(SystemExit, match="3 of 12 images"):
        _train(monkeypatch, tree, tmp_path / "models", *args)
    run = _train(monkeypatch, tree, tmp_path / "models", *args, "--allow-missing-masks", "--mask-loss-weight", "0.5")
    assert run["mask_coverage"]["train"] == {"n_images": 12, "n_with_mask": 9, "share": 0.75}
    assert run["anatomy"]["mask_loss_weight"] == 0.5 and run["args"]["allow_missing_masks"]


# ------------------------------------------------------------------ 04 and 15 with attnet_gate

def test_04_and_15_feed_the_region_input(split_tree, tmp_path, monkeypatch, capsys):
    tree, masks, cfg_path = split_tree
    out = tmp_path / "models"
    _train(monkeypatch, tree, out, "--cfg", cfg_path, "--arch", "attnet_gate", "--masks", masks, "--epochs", "1")
    ckpt = out / "Xception_ArcFace_t.keras"
    data_dir = tree / "test"
    rows = [{"link_path": os.path.join(str(data_dir), c, f), "relpath": f"cond/enrollment_left_eye/{c}/{f.split('__')[2]}",
             "condition": "cond", "device": "d", "lighting": "l", "eye": "left", "phase": "enrollment", "idx": 1,
             "subject": int(c)} for c in sorted(os.listdir(data_dir)) for f in sorted(os.listdir(data_dir / c))]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    eval_script = _load("04_eval_clean")
    base = ["04_eval_clean.py", "--split-dir", str(tree), "--n-classes", "3", "--ckpt", str(ckpt), "--manifest",
            str(tmp_path / "manifest.csv"), "--class-index", str(out / "class_index_t.json"), "--batch-size", "2"]
    monkeypatch.setattr(sys, "argv", base + ["--out-dir", str(tmp_path / "e0")])
    with pytest.raises(SystemExit, match="region-masks"):
        eval_script.main()
    monkeypatch.setattr(sys, "argv", base + ["--out-dir", str(tmp_path / "e1"), "--region-masks", str(masks)])
    eval_script.main()
    pred = pd.read_csv(tmp_path / "e1" / "predictions_full.csv")
    model = load_checkpoint(str(ckpt))
    table = mg.MaskTable(str(masks))
    images = np.stack([np.asarray(tf.keras.utils.load_img(p, target_size=(256, 256)), np.float32) / 255
                       for p in pred.link_path])
    regions = mg.regions_for_rows(table, table.rows_for_links(pred.link_path), am.region_classes(model))
    probs = model.predict_on_batch([images, np.zeros((len(images), 3), np.float32), regions])
    np.testing.assert_array_equal(pred.pred_idx.to_numpy(), probs.argmax(1))
    np.testing.assert_allclose(pred.confidence.to_numpy(), probs.max(1), rtol=1e-5)
    assert json.loads((tmp_path / "e1" / "summary_full.json").read_text())["arch"] == "attnet_gate"

    verif = _load("15_eval_verification")
    root = tmp_path / "root"
    relpaths = [r["relpath"] for r in rows]
    for r, link in zip(relpaths, [row["link_path"] for row in rows]):         # the same images under a VISOB root
        os.makedirs(root / os.path.dirname(r), exist_ok=True)
        Image.open(link).save(root / r)
    region_of = verif.region_lookup(model, str(masks), relpaths)
    emb = verif.embed_images(model, relpaths, str(root), 2, 1, region_of)
    want = am.embed_regions(model, np.stack([verif.load_image(str(root / r)) for r in relpaths]),
                            mg.regions_for_rows(table, table.rows_for_relpaths(relpaths), am.region_classes(model)))
    np.testing.assert_allclose(emb, want, atol=1e-6)
    with pytest.raises(SystemExit, match="region-masks"):
        verif.region_lookup(model, None, relpaths)
    with pytest.raises(SystemExit, match="no mask"):
        verif.region_lookup(model, str(masks), relpaths + ["cond/enrollment_left_eye/1141/1141_9999_l_c.png"])
    # x1 (external images, no eye masks for them) is refused for attnet_gate: after loading the model, and
    # before loading it when --region-masks is given
    with pytest.raises(SystemExit, match="x1 is not available for attnet_gate"):
        verif.region_lookup(model, str(masks), relpaths, external=True)
    monkeypatch.setattr(sys, "argv", ["15_eval_verification.py", "--ckpt", str(tmp_path / "never_loaded.keras"),
                                      "--modes", "p3,x1", "--external-manifest", str(tmp_path / "ubipr.csv"),
                                      "--region-masks", str(masks)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        verif.main()
    assert stop.value.code == 2 and "x1 is not available for attnet_gate" in capsys.readouterr().err
    assert verif.region_lookup(_tiny_baseline({"model": {"n_classes": 3}, "training": {"learning_rate": 1e-3}}),
                               str(masks), relpaths) is None
