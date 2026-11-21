"""notebooks/train_cross_session.ipynb must stay in step with the scripts it runs: every code cell parses, every
command parses with its script's own argparse parser, and the epoch timer reads the line Keras really prints.

The notebook uses flags of the training-recipe PR (03 --seed/--resume/--reshuffle/--save-weights, 04
--class-index). Until that PR is merged a script may lack all of its recipe flags (they are then left out of the
parse); a script with only some of them fails."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex

import numpy as np
import pytest
import tensorflow as tf

from conftest import REPO

NOTEBOOK = os.path.join(REPO, "notebooks", "train_cross_session.ipynb")
# flags of the training-recipe PR, per script: (flag, takes a value)
RECIPE_FLAGS = {"scripts/03_train_baseline.py": [("--seed", True), ("--resume", False), ("--reshuffle", False),
                                                 ("--save-weights", False)],
                "scripts/04_eval_clean.py": [("--class-index", True)]}


def _cells():
    with open(NOTEBOOK, encoding="utf-8") as f:
        return [c for c in json.load(f)["cells"] if c["cell_type"] == "code"]


def _python(cell):
    """Cell source with IPython '!' and '%' lines replaced by 'pass' (indentation kept)."""
    lines = []
    for line in "".join(cell["source"]).splitlines():
        body = line.lstrip()
        lines.append(line[:len(line) - len(body)] + "pass" if body.startswith(("!", "%")) else line)
    return "\n".join(lines)


def _cell(number):
    """The code cell whose first line is '# <number>. ...'."""
    found = [c for c in _cells() if "".join(c["source"]).startswith(f"# {number}. ")]
    assert len(found) == 1, f"expected one cell numbered {number}"
    return _python(found[0])


def _namespace(**overrides):
    """Run the settings (2) and commands (3) cells as the notebook would after its setup cell."""
    ns = {"REPO": REPO, "COMMIT": "0" * 40}
    exec(_cell(2), ns)
    ns.update(overrides)
    exec(_cell(3), ns)
    return ns


def _helpers(tmp_path):
    """The settings, commands and helper cells (2-4), with the outputs under tmp_path."""
    ns = _namespace(LOGS=str(tmp_path / "logs"), MODELS=str(tmp_path / "models"),
                    SMOKE_MODELS=str(tmp_path / "smoke_models"))
    ns.update(os=os, subprocess=__import__("subprocess"), time=__import__("time"), SESSION_START=0.0)
    exec(_cell(4), ns)
    return ns


def _parser(script):
    """The argparse parser main() of *script* builds (captured when main() calls parse_args)."""
    spec = importlib.util.spec_from_file_location(os.path.basename(script)[:-3], os.path.join(REPO, script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Captured(Exception):
        pass

    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        raise Captured(self)

    argparse.ArgumentParser.parse_args = capture
    try:
        module.main()
    except Captured as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = original
    raise AssertionError(f"{script}: main() never called parse_args")


def _split(cmd):
    """(script, arguments after it) of a 'python -u scripts/NN_name.py ...' command."""
    tokens = shlex.split(cmd)
    i = next(i for i, t in enumerate(tokens) if t.startswith("scripts/"))
    assert tokens[:i] == ["python", "-u"], cmd
    return tokens[i], tokens[i + 1:]


def _without(argv, flags):
    """argv without the given flags (and the value of a flag that takes one)."""
    takes_value = dict(flags)
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
        elif token in takes_value:
            skip = takes_value[token]
        else:
            out.append(token)
    return out


def test_code_cells_parse():
    cells = _cells()
    assert len(cells) == 15
    for cell in cells:
        compile(_python(cell), "<cell>", "exec")


@pytest.mark.parametrize("run_full", [False, True])
def test_every_command_parses_with_its_script(run_full):
    ns = _namespace(RUN_FULL=run_full, PREVIOUS_OUTPUT="/kaggle/input/earlier")
    by_script = {}
    for name, cmd in ns["CMD"].items():
        script, argv = _split(cmd)
        by_script.setdefault(script, []).append((name, argv))
    assert set(by_script) == {"scripts/01_build_manifest.py", "scripts/02_make_splits.py",
                              "scripts/03_train_baseline.py", "scripts/04_eval_clean.py",
                              "scripts/15_eval_verification.py"}
    for script, commands in by_script.items():
        parser = _parser(script)
        known = set(parser._option_string_actions)
        recipe = RECIPE_FLAGS.get(script, [])
        present = [f for f, _ in recipe if f in known]
        assert len(present) in (0, len(recipe)), f"{script} has only some recipe flags: {present}"
        for name, argv in commands:
            if not present:
                argv = _without(argv, recipe)
            used = {t for t in argv if t.startswith("--")}
            assert used <= known, f"{name}: {script} has no {sorted(used - known)}"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    pytest.fail(f"{name}: {err.getvalue()}")


def test_commands_follow_the_protocol():
    ns = _namespace(RUN_FULL=True)
    cmd = ns["CMD"]
    assert "--protocol cross_session" in cmd["02_cross_session"] and "--write-split-csvs" in cmd["02_cross_session"]
    assert "--max-classes" in cmd["02_small"] and "--out-manifest" in cmd["02_small"]
    assert ns["MAX_FRAME"] == 1009 and all("--max-frame 1009" in cmd[k] for k in ("02_cross_session", "02_open_set",
                                                                                "02_small"))
    assert all("--max-frame 1009" in c for k, c in cmd.items() if k.startswith("15_"))
    for seed in ns["SEEDS"]:
        t = f"cs_s{seed}"
        assert f"--seed {seed} " in cmd[f"03_{t}"] and f"--tag {t}" in cmd[f"03_{t}"]
        assert "--epochs" not in cmd[f"03_{t}"]                 # the full runs use the config's cap
        assert "--cfg configs/cross_session.yml" in cmd[f"03_{t}"]
        assert f"--class-index /kaggle/working/trained_models/class_index_{t}.json" in cmd[f"04_{t}"]
        assert "--manifest /kaggle/working/manifest_split_cross_session.csv" in cmd[f"04_{t}"]
        assert "--modes p2,p3" in cmd[f"15_{t}"] and "--class-index" in cmd[f"15_{t}"]


def test_epoch_timer_reads_the_keras_verbose_2_line(tmp_path):
    ns = _namespace(LOGS=str(tmp_path))
    ns.update(os=os, subprocess=__import__("subprocess"), time=__import__("time"), SESSION_START=0.0)
    exec(_cell(4), ns)
    tf.keras.utils.set_random_seed(0)              # conftest enables op determinism
    model = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(2, activation="softmax")])
    model.compile("sgd", "categorical_crossentropy")
    x, y = np.zeros((32, 4), "float32"), np.eye(2, dtype="float32")[np.arange(32) % 2]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model.fit(x, y, validation_data=(x, y), epochs=2, batch_size=8, verbose=2)
    (tmp_path / "03_demo.log").write_text("TF 2.20 | GPUs: []\nEpoch 1/2\n" + out.getvalue())
    seconds = ns["epoch_seconds"]("03_demo")
    assert len(seconds) == 2 and all(isinstance(s, int) and s >= 0 for s in seconds), out.getvalue()
    assert re.search(r"val_loss", out.getvalue())


def test_backup_written_follows_backup_and_restore(tmp_path):
    # the --resume test (cell 10) stops the first run once backup_written() sees a complete backup; this relies on
    # the files tf.keras BackupAndRestore writes, so check it against the installed tf_keras
    ns = _helpers(tmp_path)
    backup = str(tmp_path / "backup_demo")
    seen = []

    class Probe(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            seen.append(("begin", epoch, ns["backup_written"](backup)))

        def on_epoch_end(self, epoch, logs=None):       # runs after BackupAndRestore (later in the list)
            seen.append(("end", epoch, ns["backup_written"](backup)))

    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(2, activation="softmax")])
    model.compile("sgd", "categorical_crossentropy")
    x, y = np.zeros((16, 4), "float32"), np.eye(2, dtype="float32")[np.arange(16) % 2]
    model.fit(x, y, epochs=2, batch_size=8, verbose=0,
              callbacks=[tf.keras.callbacks.BackupAndRestore(backup), Probe()])
    assert seen == [("begin", 0, False), ("end", 0, True), ("begin", 1, True), ("end", 1, True)]
    assert not ns["backup_written"](backup)            # a finished training deletes its backup


def test_settle_waits_until_no_file_is_written(tmp_path):
    # the 12-hour guard stops jobs only once no checkpoint has been written for `quiet` seconds
    ns = _helpers(tmp_path)
    models = tmp_path / "models"
    assert ns["settle"]([str(models)], quiet=0.5, limit=5, every=0.05) < 0.5    # no folder yet: no wait
    (models / "backup_cs_s1" / "chief").mkdir(parents=True)
    (models / "backup_cs_s1" / "chief" / "ckpt-1.index").write_bytes(b"x")
    waited = ns["settle"]([str(models)], quiet=0.5, limit=5, every=0.05)
    assert 0.3 < waited < 2
    (models / "Xception_ArcFace_cs_s1.keras").write_bytes(b"x")
    assert ns["settle"]([str(models)], quiet=5, limit=0.3, every=0.05) < 1          # gives up after limit


def test_fields_the_notebook_reads_are_written_by_the_scripts(tmp_path):
    # the checks and the time estimate read these keys from the scripts' JSON records; a renamed key would make a
    # check fail on Kaggle only. Keys of the training-recipe PR are checked once 03 has its flags.
    ns = _helpers(tmp_path)
    exec(_cell(5), ns)

    def source(script):
        with open(os.path.join(REPO, script), encoding="utf-8") as f:
            return f.read()

    def absent(script, keys):
        text = source(script)
        return [k for k in keys if f'"{k}"' not in text]

    assert not absent("scripts/04_eval_clean.py", ["n_images", "accuracy_margin_free", "true_subject"])
    assert not absent("scripts/15_eval_verification.py",
                      ["training_subjects", "training_images_by_phase", "git_commit", "per_mode", "timing",
                       "load_model_s", "embedding_s", "n_images_embedded", "subset", "mode", "eer", "n_subjects",
                       "rank1"])
    assert not absent("scripts/03_train_baseline.py", ["test_acc_margin_free", "images_by_phase"])
    if '"--resume"' in source("scripts/03_train_baseline.py"):
        assert not absent("scripts/03_train_baseline.py", ns["RUN_KEYS"] + ["minutes_all_sessions"])


def _fake_raw_manifest(path):
    """manifest_raw.csv (01's columns) of 3 paper classes and one open-set subject, with frames 1010+ in both phases."""
    rows = []
    for subject in (1141, 1142, 1143, 4000):
        for condition in ("iphone_day_light_short", "oppo_dim_light_short"):
            for eye in ("left", "right"):
                for phase, frames in (("enrollment", (1001, 1002, 1003, 1010, 1011)),
                                      ("verification", (1001, 1002, 1010))):
                    for idx in frames:
                        rows.append({"relpath": f"{condition}/{phase}_{eye}_eye/{subject}/{subject}_{idx}_{eye[0]}_c.png",
                                     "subject": subject, "idx": idx, "eye": eye, "phase": phase,
                                     "condition": condition})
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("max_frame", [1009, None])
def test_split_checks_accept_the_split_02_writes(tmp_path, monkeypatch, max_frame):
    """The notebook's split checks (cell 5) pass on the split that 02 writes with the notebook's own --max-frame:
    the images left out above MAX_FRAME must not count as missing from the split."""
    import sys
    spec = importlib.util.spec_from_file_location("split02", os.path.join(REPO, "scripts", "02_make_splits.py"))
    split02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split02)
    ns = _helpers(tmp_path)
    ns.update(W=str(tmp_path), SPLITS=str(tmp_path / "splits"), CS_MANIFEST=str(tmp_path / "cs.csv"),
              MAX_FRAME=max_frame)
    exec(_cell(5), ns)
    _fake_raw_manifest(tmp_path / "manifest_raw.csv")
    monkeypatch.setattr(os, "symlink", lambda src, dst: open(dst, "w").close())
    argv = ["02", "--manifest", str(tmp_path / "manifest_raw.csv"), "--root", str(tmp_path / "data"),
            "--protocol", "cross_session", "--out-dir", str(tmp_path / "tree"), "--out-manifest", ns["CS_MANIFEST"],
            "--write-split-csvs", ns["SPLITS"]] + ([] if max_frame is None else ["--max-frame", str(max_frame)])
    monkeypatch.setattr(sys, "argv", argv)
    split02.main()
    if max_frame is not None:
        import pandas as pd
        assert pd.read_csv(ns["CS_MANIFEST"]).idx.max() <= max_frame        # 02 really left the late frames out
    checks = ns["run_checks"]((ns["checks_split"], ns["CS_MANIFEST"], 3), (ns["checks_split_csvs"],))
    failed = [(name, detail) for name, ok, detail, hard in checks if hard and not ok]
    assert not failed, failed


def test_version_checks_accept_the_kaggle_image():
    """Kaggle's image moved to TensorFlow 2.21 (tf-keras 2.20.1 still installs and trained cs_s0 on it, 18 Nov):
    the pre-flight and the 03 checks accept TF 2.20 and 2.21 with tf-keras 2.20.1, nothing else."""
    for number in (5, 6):
        src = _cell(number)
        assert "('2.20.', '2.21.')" in src, f"cell {number} still demands TensorFlow 2.20 only"
