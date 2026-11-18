"""Training recipe of 03 (seed, resume over sessions, reshuffled batches, weights-only copy, run record,
configs/cross_session.yml) and 04's --class-index, on tiny class-folder trees with the tiny test model."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from conftest import REPO, build_tiny_model


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load("03_train_baseline")
eval_script = _load("04_eval_clean")

BASELINE = os.path.join(REPO, "configs", "baseline.yml")
CROSS_SESSION = os.path.join(REPO, "configs", "cross_session.yml")
RUN_FIELDS = {"git_commit", "git_dirty", "argv", "args", "seed", "cfg", "versions", "split_dir", "images_per_split",
              "images_by_phase", "class_list_sha256", "epochs_run", "best_epoch", "best_val_loss", "minutes",
              "minutes_all_sessions", "resumed", "checkpoint", "checkpoint_sha256", "finished"}


def _class_tree(root, layout):
    """layout: split -> {class folder: number of images}; 4 x 4 PNGs named like 02's links."""
    rng = np.random.default_rng(0)
    for split, classes in layout.items():
        for name, n in classes.items():
            d = root / split / name
            d.mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(d / f"cond__enrollment__{name}_{1001 + i}_l_c.png")
    return root


TREE = {"train": {"1141": 4, "1142": 4, "7000": 4}, "validation": {"1141": 1, "1142": 1, "7000": 1},
        "test": {"1141": 1, "1142": 1, "7000": 1}}


def _tiny_model(cfg):
    """The tiny test model, compiled like create_model() (Adam at the cfg's learning rate)."""
    model = build_tiny_model(cfg["model"]["n_classes"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["training"]["learning_rate"]),
                  loss=tf.keras.losses.CategoricalCrossentropy())
    return model


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(train_script, "build_model", _tiny_model)
    return _class_tree(tmp_path / "tree", TREE)


def _train(monkeypatch, tree, out, *extra):
    monkeypatch.setattr(sys, "argv", ["03_train_baseline.py", "--split-dir", str(tree), "--n-classes", "3",
                                      "--batch-size", "4", "--verbose", "0", "--tag", "t", "--out-dir", str(out),
                                      *map(str, extra)])
    train_script.main()
    return json.loads((out / "run_t.json").read_text()) if (out / "run_t.json").exists() else None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------- defaults

def test_default_callbacks_keep_the_baseline_settings(tmp_path):
    cfg = train_script.load_cfg(BASELINE)
    t = cfg["training"]
    assert (t["epochs"], t["early_stop_patience"], t["min_lr"]) == (10, 20, t["learning_rate"])
    es, rlr, mc, csv = train_script.make_callbacks(t, str(tmp_path / "m.keras"), str(tmp_path / "log.csv"))
    assert type(es).__name__ == "EarlyStopping" and es.patience == 20 and not es.restore_best_weights
    assert type(rlr).__name__ == "ReduceLROnPlateau" and rlr.min_lr == t["learning_rate"] == 1e-4
    assert (rlr.patience, rlr.factor) == (5, 0.2)
    assert type(mc).__name__ == "ModelCheckpoint" and mc.save_best_only and mc.monitor == "val_loss"
    assert type(csv).__name__ == "CSVLogger" and not csv.append

    resumable = train_script.make_callbacks(t, str(tmp_path / "m.keras"), str(tmp_path / "log.csv"),
                                            str(tmp_path / "backup"), resumed=True)
    assert [type(c).__name__ for c in resumable] == ["EarlyStopping", "ReduceLROnPlateau", "AtomicModelCheckpoint",
                                                     "CSVLogger", "ResumeState", "BackupAndRestore"]
    assert isinstance(resumable[2], tf.keras.callbacks.ModelCheckpoint) and resumable[2].save_best_only
    assert resumable[3].append and resumable[4].early_stop is resumable[0]
    assert not resumable[5].delete_checkpoint                # 03 removes the backup after the run record


def test_default_run_writes_the_run_record(tree, tmp_path, monkeypatch):
    seeds, made = [], []
    monkeypatch.setattr(tf.keras.utils, "set_random_seed",
                        lambda s, f=tf.keras.utils.set_random_seed: (seeds.append(s), f(s))[1])
    monkeypatch.setattr(train_script, "make_callbacks",
                        lambda *a, f=train_script.make_callbacks, **k: made.append(f(*a, **k)) or made[-1])
    out = tmp_path / "models"
    run = _train(monkeypatch, tree, out, "--epochs", "2")

    assert seeds == [42]
    assert [type(c).__name__ for c in made[0]] == ["EarlyStopping", "ReduceLROnPlateau", "ModelCheckpoint",
                                                   "CSVLogger"]
    assert made[0][0].patience == 20 and made[0][1].min_lr == 1e-4 and not made[0][3].append
    assert RUN_FIELDS <= set(run)
    assert run["finished"] and not run["resumed"] and run["seed"] == run["args"]["seed"] == 42
    assert run["argv"][0] == "03_train_baseline.py" and run["args"]["split_dir"] == str(tree)
    assert run["cfg"]["training"]["epochs"] == 2 and run["cfg"]["training"]["early_stop_patience"] == 20
    assert run["cfg"]["training"]["batch_size"] == 4 and run["cfg"]["model"]["n_classes"] == 3
    assert set(run["versions"]) == {"python", "numpy", "tensorflow", "tf_keras"}
    assert run["versions"]["tensorflow"] == tf.__version__
    assert run["split_dir"] == str(tree)
    assert run["images_per_split"] == {"train": 12, "validation": 3, "test": 3}
    assert run["images_by_phase"]["train"] == {"enrollment": 12}
    assert run["class_list_sha256"] == hashlib.sha256(b"1141\n1142\n7000").hexdigest()
    log = pd.read_csv(out / "log_t.csv")
    assert run["epochs_run"] == len(log) == 2
    assert run["best_val_loss"] == pytest.approx(log.val_loss.min())
    assert run["best_epoch"] == int(log.val_loss.idxmin()) + 1
    assert run["minutes"] >= 0 and run["minutes_all_sessions"] == run["minutes"]
    assert run["checkpoint"] == str(out / "Xception_ArcFace_t.keras")
    assert run["checkpoint_sha256"] == _sha256(out / "Xception_ArcFace_t.keras")
    assert run["weights"] is None and not list(out.glob("*.weights.h5")) and not (out / "backup_t").exists()
    history = pd.read_csv(out / "history_t.csv")              # as before: fit's history, no epoch column
    assert list(history.columns) == ["loss", "val_loss", "lr"] and len(history) == 2
    assert json.loads((out / "results_t.json").read_text())["epochs_run"] == 2


def test_seed_is_recorded_and_applied(tree, tmp_path, monkeypatch):
    seeds = []
    monkeypatch.setattr(tf.keras.utils, "set_random_seed",
                        lambda s, f=tf.keras.utils.set_random_seed: (seeds.append(s), f(s))[1])
    run = _train(monkeypatch, tree, tmp_path / "models", "--epochs", "1", "--seed", "7")
    assert seeds == [7] and run["seed"] == run["args"]["seed"] == 7


# ------------------------------------------------------------------- resume

class Killed(RuntimeError):
    """Stands in for Kaggle stopping the session."""


class SetLearningRate(tf.keras.callbacks.Callback):
    """At the end of the first epoch, set the rate as a ReduceLROnPlateau drop would."""

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, 7e-4)


class StopAfterLogging(tf.keras.callbacks.Callback):
    """Kill the run at the end of epoch 2, after its log row and callback state, before its backup."""

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            raise Killed()


def test_resume_continues_after_an_interruption(tree, tmp_path, monkeypatch, capsys):
    kill = [True]

    def callbacks(*args, f=train_script.make_callbacks, **kwargs):
        cbs = f(*args, **kwargs)
        return [SetLearningRate()] + cbs[:-1] + [StopAfterLogging()] + cbs[-1:] if kill[0] else cbs

    monkeypatch.setattr(train_script, "make_callbacks", callbacks)
    out = tmp_path / "models"
    args = ("--epochs", "3", "--cfg", CROSS_SESSION, "--resume")
    with pytest.raises(Killed):
        _train(monkeypatch, tree, out, *args)
    backup = out / "backup_t"
    assert train_script.backup_exists(str(backup)) and not (out / "run_t.json").exists()
    assert sorted(json.loads((backup / "callback_state.json").read_text())["epochs"]) == ["0", "1"]
    assert pd.read_csv(out / "log_t.csv").epoch.tolist() == [0, 1]   # epoch 2 logged, not backed up
    first_best = json.loads((backup / "callback_state.json").read_text())["epochs"]["0"]["checkpoint"]["best"]

    kill[0] = False                                           # the next session: the same command again
    capsys.readouterr()
    run = _train(monkeypatch, tree, out, *args)
    assert "resuming at epoch 2" in capsys.readouterr().out
    assert run["resumed"] and run["resumed_at_epoch"] == 2
    assert run["epochs_run"] == 3 and run["epochs_this_session"] == 2
    log = pd.read_csv(out / "log_t.csv")
    assert log.epoch.tolist() == [0, 1, 2]                    # appended; the re-run epoch 2 kept once
    assert log.lr.iloc[1:].tolist() == pytest.approx([7e-4, 7e-4])   # optimizer rate came back with the backup
    assert log.val_loss.iloc[0] == pytest.approx(first_best)
    assert run["best_val_loss"] == pytest.approx(log.val_loss.min())
    assert len(pd.read_csv(out / "history_t.csv")) == 3
    assert run["minutes_all_sessions"] >= run["minutes"]
    assert not backup.exists()                               # training ended: backup and state removed
    assert run["checkpoint_sha256"] == _sha256(out / "Xception_ArcFace_t.keras")

    assert not list(out.glob("*.partial.keras"))

    capsys.readouterr()                                       # once more: nothing left to resume
    before = (out / "run_t.json").read_text()
    _train(monkeypatch, tree, out, *args)
    assert "nothing to resume" in capsys.readouterr().out
    assert (out / "run_t.json").read_text() == before and len(pd.read_csv(out / "log_t.csv")) == 3

    ckpt = _sha256(out / "Xception_ArcFace_t.keras")         # another epochs cap: stops, the finished run is kept
    with pytest.raises(SystemExit, match=r"cfg\.training\.epochs: 3 -> 2"):
        _train(monkeypatch, tree, out, "--epochs", "2", "--cfg", CROSS_SESSION, "--resume")
    assert (out / "run_t.json").read_text() == before and _sha256(out / "Xception_ArcFace_t.keras") == ckpt


def test_a_finished_run_is_recognised_under_other_paths_and_verbosity(tree, tmp_path, monkeypatch, capsys):
    # the same data, config and models copied elsewhere (another Kaggle mount): only paths and --verbose differ
    out = tmp_path / "models"
    _train(monkeypatch, tree, out, "--epochs", "1", "--resume")
    tree2, out2, cfg2 = tmp_path / "tree_copy", tmp_path / "models_copy", tmp_path / "copy_of_baseline.yml"
    shutil.copytree(tree, tree2)
    shutil.copytree(out, out2)
    shutil.copy(BASELINE, cfg2)
    before, log = (out2 / "run_t.json").read_text(), (out2 / "log_t.csv").read_text()
    capsys.readouterr()
    _train(monkeypatch, tree, out2, "--epochs", "1", "--resume", "--verbose", "2", "--split-dir", tree2,
           "--cfg", cfg2)
    assert "nothing to resume" in capsys.readouterr().out
    assert (out2 / "run_t.json").read_text() == before and (out2 / "log_t.csv").read_text() == log


def test_a_finished_run_with_another_seed_is_not_trained_over(tree, tmp_path, monkeypatch):
    out = tmp_path / "models"
    _train(monkeypatch, tree, out, "--epochs", "1", "--resume")
    before, ckpt = (out / "run_t.json").read_text(), _sha256(out / "Xception_ArcFace_t.keras")
    with pytest.raises(SystemExit, match="seed: 42 -> 7") as stop:
        _train(monkeypatch, tree, out, "--epochs", "1", "--resume", "--seed", "7")
    assert "another --tag or --out-dir" in str(stop.value)
    assert (out / "run_t.json").read_text() == before and _sha256(out / "Xception_ArcFace_t.keras") == ckpt


class StopAfterBackup(tf.keras.callbacks.Callback):
    """Kill the run at the end of epoch *epoch* (0-based), after its backup (placed after BackupAndRestore)."""

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epoch:
            raise Killed()


class FakeEarlyStop(tf.keras.callbacks.Callback):
    """Make *early_stop* stop the run at the end of epoch 2, as if val_loss had stalled."""

    def __init__(self, early_stop):
        super().__init__()
        self.early_stop = early_stop

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1:
            self.early_stop.stopped_epoch = epoch
            self.model.stop_training = True


@pytest.mark.parametrize("stop", ["cap", "early_stop", "early_stop_not_backed_up"])
def test_resume_after_the_last_backup_trains_no_further_epoch(tree, tmp_path, monkeypatch, capsys, stop):
    # cap / early_stop: stopped after the backup of the last epoch (the cap, or the epoch at which
    # EarlyStopping fired), before the run record: the next session trains nothing and writes the record
    # of all epochs. early_stop_not_backed_up: stopped after EarlyStopping fired but before that epoch's
    # backup: the epoch is trained again, and the run goes on unless EarlyStopping fires again
    kill = [True]

    def callbacks(*args, f=train_script.make_callbacks, **kwargs):
        cbs = f(*args, **kwargs)
        if not kill[0]:
            return cbs
        if stop == "cap":
            return cbs + [StopAfterBackup(2)]
        if stop == "early_stop":
            return cbs[:4] + [FakeEarlyStop(cbs[0])] + cbs[4:] + [StopAfterBackup(1)]
        return cbs[:4] + [FakeEarlyStop(cbs[0])] + cbs[4:5] + [StopAfterLogging()] + cbs[5:]

    monkeypatch.setattr(train_script, "make_callbacks", callbacks)
    out = tmp_path / "models"
    args = ("--epochs", "3", "--cfg", CROSS_SESSION, "--resume", "--save-weights")
    with pytest.raises(Killed):
        _train(monkeypatch, tree, out, *args)
    logged = 3 if stop == "cap" else 2
    assert len(pd.read_csv(out / "log_t.csv")) == logged and not (out / "run_t.json").exists()
    assert train_script.backup_epoch(str(out / "backup_t")) == (0 if stop == "early_stop_not_backed_up" else logged - 1)

    kill[0] = False
    capsys.readouterr()
    run = _train(monkeypatch, tree, out, *args)
    printed = capsys.readouterr().out
    assert ("EarlyStopping had stopped the run after epoch 2" in printed) == (stop == "early_stop")
    if stop == "early_stop_not_backed_up":
        assert "resuming at epoch 2" in printed
        assert run["resumed_at_epoch"] == 2 and run["epochs_this_session"] == 2 and run["epochs_run"] == 3
        assert not run["stopped_early"]
        logged = 3
    else:
        assert f"no epoch left to train after epoch {logged}" in printed
        assert run["resumed_at_epoch"] is None and run["epochs_this_session"] == 0 and run["epochs_run"] == logged
        assert run["stopped_early"] == (stop == "early_stop")
    assert run["finished"] and run["resumed"]
    assert run["minutes_all_sessions"] >= run["minutes"]
    assert pd.read_csv(out / "log_t.csv").epoch.tolist() == list(range(logged))
    assert len(pd.read_csv(out / "history_t.csv")) == logged
    assert run["weights_sha256"] == _sha256(out / "Xception_ArcFace_t.weights.h5")
    assert not (out / "backup_t").exists()


def test_a_backup_left_after_the_run_record_is_removed(tree, tmp_path, monkeypatch, capsys):
    out = tmp_path / "models"
    args = ("--epochs", "1", "--resume")
    removed = []
    rmtree = train_script.shutil.rmtree
    monkeypatch.setattr(train_script.shutil, "rmtree", lambda p, **k: removed.append(p))   # stopped before it
    _train(monkeypatch, tree, out, *args)
    assert removed == [str(out / "backup_t")] and train_script.backup_exists(str(out / "backup_t"))
    monkeypatch.setattr(train_script.shutil, "rmtree", rmtree)
    capsys.readouterr()
    before = (out / "run_t.json").read_text()
    _train(monkeypatch, tree, out, *args)
    assert "nothing to resume" in capsys.readouterr().out
    assert (out / "run_t.json").read_text() == before and not (out / "backup_t").exists()


def test_a_stop_before_the_evaluation_is_finished_by_the_next_session(tree, tmp_path, monkeypatch, capsys):
    out = tmp_path / "models"
    args = ("--epochs", "2", "--resume", "--save-weights")
    _train(monkeypatch, tree, out, *args)
    log = (out / "log_t.csv").read_text()
    results = json.loads((out / "results_t.json").read_text())
    (out / "results_t.json").unlink()                  # as if the session stopped after the run record
    (out / "Xception_ArcFace_t.weights.h5").unlink()
    capsys.readouterr()
    run = _train(monkeypatch, tree, out, *args)
    printed = capsys.readouterr().out
    assert "writing the missing weights / test results" in printed and "nothing to resume" not in printed
    assert (out / "log_t.csv").read_text() == log      # no epoch trained again
    again = json.loads((out / "results_t.json").read_text())
    assert again["test_acc_margin_free"] == results["test_acc_margin_free"] and again["epochs_run"] == 2
    assert run["weights_sha256"] == _sha256(out / "Xception_ArcFace_t.weights.h5")
    _train(monkeypatch, tree, out, *args)
    assert "nothing to resume" in capsys.readouterr().out


def test_atomic_checkpoint_keeps_the_old_best_when_a_save_is_cut_off(tmp_path):
    cfg = train_script.load_cfg(CROSS_SESSION)
    cfg["model"]["n_classes"] = 3
    model = _tiny_model(cfg)
    ckpt = tmp_path / "m.keras"
    mc = train_script.make_callbacks(cfg["training"], str(ckpt), str(tmp_path / "log.csv"),
                                     str(tmp_path / "backup"))[2]
    mc.set_model(model)
    mc.on_epoch_end(0, {"val_loss": 1.0})
    assert ckpt.exists() and not (tmp_path / "m.partial.keras").exists()
    good = _sha256(ckpt)

    def cut_off(path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"PK half a zip")
        raise Killed()

    model.save = cut_off
    with pytest.raises(Killed):
        mc.on_epoch_end(1, {"val_loss": 0.5})
    assert _sha256(ckpt) == good                              # still the readable best of epoch 1
    tf.keras.models.load_model(str(ckpt), custom_objects={"ArcFace": train_script.ArcFace})


def test_resume_state_restores_the_callbacks(tmp_path):
    cfg = train_script.load_cfg(CROSS_SESSION)
    cfg["model"]["n_classes"] = 3
    tf.keras.utils.set_random_seed(0)
    model = _tiny_model(cfg)
    ckpt, log, backup = str(tmp_path / "m.keras"), str(tmp_path / "log.csv"), str(tmp_path / "backup")
    model.save(ckpt)
    es, rlr, mc, _, state, _ = train_script.make_callbacks(cfg["training"], ckpt, log, backup)
    for cb in (es, rlr, state):
        cb.set_model(model)
        cb.on_train_begin()
    es.wait, es.best, es.best_epoch = 3, 0.5, 1
    rlr.wait, rlr.best, rlr.cooldown_counter = 1, 0.5, 0
    mc.best = 0.5
    state.on_epoch_end(1)

    other = _tiny_model(cfg)                                  # other weights, as after a restart
    es2, rlr2, mc2, _, state2, _ = train_script.make_callbacks(cfg["training"], ckpt, log, backup, resumed=True)
    for cb in (es2, rlr2, state2):
        cb.set_model(other)
        cb.on_train_begin()
    assert (es2.wait, es2.best, mc2.best) == (0, np.inf, np.inf)
    state2.on_epoch_begin(2)
    assert (es2.wait, es2.best, es2.best_epoch, es2.stopped_epoch) == (3, 0.5, 1, 0)
    assert (rlr2.wait, rlr2.best, rlr2.cooldown_counter, mc2.best) == (1, 0.5, 0, 0.5)
    for got, want in zip(es2.best_weights, model.get_weights()):   # the checkpoint's, not the current ones
        np.testing.assert_array_equal(got, want)
    state2.on_epoch_begin(3)                                  # only the first epoch of a session restores
    assert state2.initial_epoch == 2

    with open(ckpt, "wb") as f:                               # a broken checkpoint: warn, keep training
        f.write(b"not a zip")
    es3, _, mc3, _, state3, _ = train_script.make_callbacks(cfg["training"], ckpt, log, backup, resumed=True)
    for cb in (es3, state3):
        cb.set_model(other)
        cb.on_train_begin()
    state3.on_epoch_begin(2)
    assert es3.best == 0.5 and es3.best_weights is None and mc3.best == 0.5

    os.remove(ckpt)                                           # no checkpoint: the next epoch must write one
    es4, _, mc4, _, state4, _ = train_script.make_callbacks(cfg["training"], ckpt, log, backup, resumed=True)
    for cb in (es4, state4):
        cb.set_model(other)
        cb.on_train_begin()
    state4.on_epoch_begin(2)
    assert es4.best == 0.5 and mc4.best == np.inf


# ---------------------------------------------------------------- reshuffle

def test_reshuffle_draws_new_batches(tmp_path):
    root = _class_tree(tmp_path, {"train": TREE["train"]})
    classes = train_script.class_list(str(root))
    plain = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    for reshuffle in (False, True):
        gen = train_script.ClassListDataGen(str(root / "train"), plain, classes, batch_size=4, target_size=(8, 8),
                                            reshuffle=reshuffle)
        gen[0]
        first = gen.generator.index_array.copy()
        gen.on_epoch_end()
        assert np.array_equal(gen.generator.index_array, first) != reshuffle
    vendored = train_script.CustomDataGen(str(root / "train"), plain, batch_size=4, target_size=(8, 8))
    vendored[0]
    first = vendored.generator.index_array.copy()
    vendored.on_epoch_end()                                   # the vendored class keeps its batches
    np.testing.assert_array_equal(vendored.generator.index_array, first)


@pytest.mark.parametrize("reshuffle", [False, True])
def test_reshuffle_in_training(tree, tmp_path, monkeypatch, reshuffle):
    orders = []

    class Recording(train_script.ClassListDataGen):
        def on_epoch_end(self):
            if self.generator.directory.endswith("train"):
                orders.append(self.generator.index_array.copy())   # the order this epoch trained on
            super().on_epoch_end()

    monkeypatch.setattr(train_script, "ClassListDataGen", Recording)
    _train(monkeypatch, tree, tmp_path / "models", "--epochs", "3", *(["--reshuffle"] if reshuffle else []))
    assert len(orders) == 3
    same = [np.array_equal(orders[0], o) for o in orders[1:]]
    assert same == ([False, False] if reshuffle else [True, True])


# ------------------------------------------------------------ save-weights

def test_save_weights_writes_a_loadable_weights_file(tree, tmp_path, monkeypatch):
    out = tmp_path / "models"
    run = _train(monkeypatch, tree, out, "--epochs", "1", "--save-weights")
    weights = out / "Xception_ArcFace_t.weights.h5"
    assert run["weights"] == str(weights) and run["weights_source"] == "best_checkpoint"
    assert run["weights_sha256"] == _sha256(weights)
    assert weights.stat().st_size < (out / "Xception_ArcFace_t.keras").stat().st_size   # no optimizer state
    fresh = build_tiny_model(3)
    fresh.load_weights(str(weights))
    best = tf.keras.models.load_model(str(out / "Xception_ArcFace_t.keras"),
                                      custom_objects={"ArcFace": train_script.ArcFace})
    for got, want in zip(fresh.get_weights(), best.get_weights()):
        np.testing.assert_array_equal(got, want)


# --------------------------------------------------------- cross-session cfg

def test_cross_session_config(tmp_path):
    base, cs = train_script.load_cfg(BASELINE), train_script.load_cfg(CROSS_SESSION)
    t = cs["training"]
    assert (t["epochs"], t["early_stop_patience"], t["early_stop_restore_best"]) == (30, 5, True)
    assert (t["reduce_lr_patience"], t["reduce_lr_factor"], t["min_lr"]) == (2, 0.2, 1e-6)
    for key in ("learning_rate", "batch_size"):
        assert t[key] == base["training"][key]
    for section in ("model", "arcface", "data", "paths"):     # fine_tune_at, input size, augmentation inputs
        assert cs[section] == base[section]
    changed = {k for k in base["training"] if base["training"][k] != t[k]} | (set(t) - set(base["training"]))
    assert changed == {"epochs", "early_stop_patience", "early_stop_restore_best", "reduce_lr_patience", "min_lr"}

    es, rlr, _, _ = train_script.make_callbacks(t, str(tmp_path / "m.keras"), str(tmp_path / "log.csv"))
    assert es.patience == 5 and es.restore_best_weights
    assert (rlr.patience, rlr.factor, rlr.min_lr) == (2, 0.2, 1e-6)

    cfg = dict(cs, model=dict(cs["model"], n_classes=3))      # EarlyStopping goes back to the best epoch
    tf.keras.utils.set_random_seed(0)
    model = _tiny_model(cfg)
    es.set_model(model)
    es.on_train_begin()
    best = model.get_weights()
    es.on_epoch_end(0, {"val_loss": 1.0})
    for epoch in range(1, 6):
        model.set_weights([w + epoch for w in best])
        es.on_epoch_end(epoch, {"val_loss": 1.0 + epoch})
    assert model.stop_training and es.stopped_epoch == 5
    es.on_train_end()
    for got, want in zip(model.get_weights(), best):
        np.testing.assert_array_equal(got, want)


# ------------------------------------------------------- 04 --class-index

@pytest.fixture
def eval_setup(tmp_path):
    root = _class_tree(tmp_path / "tree", {"test": {"1141": 2, "1142": 2, "7000": 2}})
    tf.keras.utils.set_random_seed(0)
    ckpt = tmp_path / "m.keras"
    build_tiny_model(3).save(str(ckpt))
    return root, ckpt


def _manifest(root, path):
    data_dir = os.path.join(str(root), "test")
    rows = [{"link_path": os.path.join(data_dir, c, f), "relpath": f"x/{c}/{f}", "condition": "cond",
             "device": "d", "lighting": "l", "eye": "left", "phase": "verification", "idx": 1, "subject": int(c)}
            for c in sorted(os.listdir(data_dir)) for f in sorted(os.listdir(os.path.join(data_dir, c)))]
    pd.DataFrame(rows).to_csv(path, index=False)


def _evaluate(monkeypatch, root, ckpt, out, *extra):
    _manifest(root, out.parent / "manifest.csv")
    monkeypatch.setattr(sys, "argv", ["04_eval_clean.py", "--split-dir", str(root), "--n-classes", "3",
                                      "--ckpt", str(ckpt), "--manifest", str(out.parent / "manifest.csv"),
                                      "--out-dir", str(out), "--tag", "t", *map(str, extra)])
    eval_script.main()
    return pd.read_csv(out / "predictions_t.csv"), json.loads((out / "summary_t.json").read_text())


def test_eval_default_class_order_is_unchanged(eval_setup, tmp_path, monkeypatch):
    root, ckpt = eval_setup
    pred, summary = _evaluate(monkeypatch, root, ckpt, tmp_path / "out")
    assert dict(zip(pred.true_subject, pred.true_idx)) == {1141: 0, 1142: 1, 7000: 2}
    assert summary["class_index"] is None and summary["n_images"] == 6


def test_eval_takes_the_class_order_from_the_file(eval_setup, tmp_path, monkeypatch, capsys):
    root, ckpt = eval_setup
    index = tmp_path / "class_index_t.json"
    index.write_text(json.dumps({"7000": 0, "1141": 1, "1142": 2}))
    pred, summary = _evaluate(monkeypatch, root, ckpt, tmp_path / "out", "--class-index", index)
    assert dict(zip(pred.true_subject, pred.true_idx)) == {7000: 0, 1141: 1, 1142: 2}
    assert (pred.true_subject == pred.link_path.map(lambda p: int(os.path.basename(os.path.dirname(p))))).all()
    assert summary["class_index"] == str(index) and "class order from" in capsys.readouterr().out

    (root / "test" / "1142").rename(tmp_path / "1142")      # a split without one training class still works
    pred, _ = _evaluate(monkeypatch, root, ckpt, tmp_path / "out2", "--class-index", index)
    assert dict(zip(pred.true_subject, pred.true_idx)) == {7000: 0, 1141: 1}


def test_eval_stops_on_a_folder_missing_from_the_class_index(eval_setup, tmp_path, monkeypatch):
    root, ckpt = eval_setup
    index = tmp_path / "class_index_t.json"
    index.write_text(json.dumps({"1141": 0, "1142": 1, "9999": 2}))
    with pytest.raises(SystemExit, match="not in"):
        _evaluate(monkeypatch, root, ckpt, tmp_path / "out", "--class-index", index)
    index.write_text(json.dumps({"1141": 0, "1142": 1, "7000": 3}))
    with pytest.raises(SystemExit, match="not 0..2"):
        _evaluate(monkeypatch, root, ckpt, tmp_path / "out", "--class-index", index)
    assert not (tmp_path / "out").exists()
