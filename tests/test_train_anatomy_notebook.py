"""notebooks/train_anatomy.ipynb must stay in step with the scripts it runs: every code cell parses, every command
parses with its script's own argparse parser and follows the protocol (masks where an arch needs them, the same
images for both 22 runs), the job plan puts the mask models after the masks, results.zip keeps no checkpoint, mask
or image, and the fields the notebook reads are the ones the scripts write."""
import contextlib
import io
import json
import os
import shlex

import numpy as np
import pytest
import tensorflow as tf

from conftest import REPO
from test_train_notebook import _parser, _split

NOTEBOOK = os.path.join(REPO, "notebooks", "train_anatomy.ipynb")
SCRIPTS = {"scripts/01_build_manifest.py", "scripts/02_make_splits.py", "scripts/03_train_baseline.py",
           "scripts/04_eval_clean.py", "scripts/15_eval_verification.py", "scripts/17_make_eye_masks.py",
           "scripts/21_make_train_masks.py", "scripts/22_anatomical_shapley.py", "scripts/23_analyse_asa.py"}


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
    found = [c for c in _cells() if "".join(c["source"]).startswith(f"# {number}. ")]
    assert len(found) == 1, f"expected one cell numbered {number}"
    return _python(found[0])


def _namespace(**overrides):
    """The settings (2) and commands (3) cells, with *overrides* applied to the settings."""
    ns = {"REPO": REPO, "COMMIT": "0" * 40, "os": os}
    exec(_cell(2), ns)
    ns.update(overrides)
    exec(_cell(3), ns)
    return ns


def _helpers(tmp_path, **overrides):
    ns = _namespace(LOGS=str(tmp_path / "logs"), **overrides)
    ns.update(subprocess=__import__("subprocess"), time=__import__("time"), SESSION_START=0.0)
    exec(_cell(4), ns)
    return ns


def _args(cmd):
    """{flag: value or True} of a command."""
    tokens = shlex.split(cmd)
    out = {}
    for i, t in enumerate(tokens):
        if t.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            out[t] = nxt if nxt is not None and not nxt.startswith("--") else True
    return out


SETTINGS = [{}, {"RUN_FULL": True, "PREVIOUS_OUTPUT": "/kaggle/input/earlier"},
            {"ARCHS": ["baseline", "gap", "parts"], "SEEDS": [1, 2], "MASK_LOSS_WEIGHT": 0.5, "ASA_MAX_IMAGES": 50},
            {"BASELINE_CKPT": "/kaggle/input/cs/trained_models/Xception_ArcFace_cs_s1.keras", "ASA_ARCH": "attn_mask"}]


def test_code_cells_parse():
    cells = _cells()
    assert len(cells) == 16
    for cell in cells:
        compile(_python(cell), "<cell>", "exec")


@pytest.mark.parametrize("settings", SETTINGS)
def test_every_command_parses_with_its_script(settings):
    ns = _namespace(**settings)
    by_script = {}
    for name, cmd in ns["CMD"].items():
        script, argv = _split(cmd)
        by_script.setdefault(script, []).append((name, argv))
    assert set(by_script) <= SCRIPTS and {"scripts/21_make_train_masks.py", "scripts/22_anatomical_shapley.py",
                                          "scripts/23_analyse_asa.py", "scripts/17_make_eye_masks.py"} <= set(by_script)
    for script, commands in by_script.items():
        parser = _parser(script)
        known = set(parser._option_string_actions)
        for name, argv in commands:
            used = {t for t in argv if t.startswith("--")}
            assert used <= known, f"{name}: {script} has no {sorted(used - known)}"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    pytest.fail(f"{name}: {err.getvalue()}")


@pytest.mark.parametrize("settings", SETTINGS)
def test_commands_follow_the_protocol(settings):
    ns = _namespace(**settings)
    cmd, mask_archs = ns["CMD"], ns["MASK_ARCHS"]
    assert set(mask_archs) == {"attn_mask", "parts", "attnet_gate"} and ns["TEST_MASK_ARCHS"] == ("attnet_gate",)
    from src.models import anatomy_model as am
    assert tuple(mask_archs) == am.MASK_ARCHS and tuple(ns["LOSS_ARCHS"]) == am.MASK_LOSS_ARCHS
    tags = list(ns["FULL_TAGS"].values()) + list(ns["SMOKE_TAGS"].values())
    assert len(set(tags)) == len(tags)
    for full, (key, t) in [(True, kv) for kv in ns["FULL_TAGS"].items()] + \
                          [(False, ((a, ns["SEEDS"][0]), st)) for a, st in ns["SMOKE_TAGS"].items()]:
        arch, seed = key
        a03, a04, a15 = (_args(cmd[f"{s}_{t}"]) for s in ("03", "04", "15"))
        masks = ns["MASKS64"] if full else ns["SMALL_MASKS64"]
        assert a03["--arch"] == arch and a03["--seed"] == str(seed) and a03["--tag"] == t
        assert a03["--cfg"] == "configs/anatomy.yml" and all(f in a03 for f in ("--resume", "--reshuffle",
                                                                                "--save-weights"))
        assert a03.get("--masks") == (masks if arch in mask_archs else None)
        assert ("--mask-loss-weight" in a03) == (arch in ns["LOSS_ARCHS"] and ns["MASK_LOSS_WEIGHT"] is not None)
        assert ("--epochs" in a03) == (not full)                        # the full runs use the config's cap
        for a in (a04, a15):
            assert a.get("--region-masks") == (masks if arch == "attnet_gate" else None)
            assert a["--ckpt"] == f"{a03['--out-dir']}/Xception_ArcFace_{t}.keras"
            assert a["--class-index"] == f"{a03['--out-dir']}/class_index_{t}.json"
        assert a15["--modes"] == ("p2" if not full and arch == "attnet_gate" else "p2,p3")
        assert a04["--split-dir"] == a03["--split-dir"] == (ns["CS_TREE"] if full else ns["SMALL_TREE"])
        mask_deps = ns["NEEDS"].get(f"03_{t}", [])
        assert mask_deps == ([("21_full" if full else "21_smoke")] if arch in mask_archs else [])
    a21 = _args(cmd["21_full"])
    assert a21["--csv"] == ns["MASK_LIST"] and a21["--splits"] == "train,validation,test,open_set"
    assert _args(cmd["21_smoke"])["--split-dir"] == ns["SMALL_TREE"]
    assert _args(cmd["17_pool"])["--csv"] == ns["ASA_POOL"]
    names = ["mask"] + (["base"] if ns["BASE"] else [])
    assert (ns["BASE"] is not None) == bool(settings.get("BASELINE_CKPT") or "baseline" in settings.get("ARCHS", []))
    full22 = {n: _args(cmd[f"22_{n}"]) for n in names}
    for n, a in full22.items():
        smoke = _args(cmd[f"22_smoke_{n}"])
        assert a["--masks"] == smoke["--masks"] == ns["POOL_MASKS"]
        assert a["--exclude-from"] == f"{smoke['--out-dir']}/images_asa_{smoke['--tag']}.csv"
        assert _args(cmd[f"23_{n}"])["--rows"] == f"{a['--out-dir']}/rows_asa_{a['--tag']}.csv"
        assert ns["NEEDS"][f"22_{n}"] == [f"22_smoke_{n}"]
    # both models draw the same images (same list, seeds and limits), so their ASA values are paired
    keys = ("--correct-csv", "--seed", "--per-class", "--max-images")
    assert len({tuple(a.get(k) for k in keys) for a in full22.values()}) == 1
    # the full runs continue a stopped run, and the two parallel runs (and the smoke runs that time them)
    # use half the CPU threads each for inpainting
    for n, a in full22.items():
        assert a["--resume"] is True and "--resume" not in _args(cmd[f"22_smoke_{n}"])
        assert a["--workers"] == _args(cmd[f"22_smoke_{n}"])["--workers"] == str(ns["ASA_WORKERS"])
    assert _args(cmd["22_smoke"])["--workers"] == str(ns["ASA_WORKERS"])
    if settings.get("BASELINE_CKPT"):
        a = _args(cmd["04_base_ext"])
        assert a["--ckpt"] == settings["BASELINE_CKPT"] and a["--class-index"].endswith("/class_index_cs_s1.json")
    assert ns["ASA_MASK_ARCH"] == settings.get("ASA_ARCH", "parts")
    # every step a step needs is a step of the session
    for name, needs in ns["NEEDS"].items():
        assert all(n in cmd for n in needs), name


def _resume_cell(tmp_path, **overrides):
    """Cells 2-4 and the resume cell (8) with PREVIOUS_OUTPUT unset, writing under tmp_path."""
    w = str(tmp_path)
    ns = _helpers(tmp_path, W=w, MODELS=f"{w}/trained_models", TRAIN_MASKS=f"{w}/train_masks", **overrides)
    ns.update(PREV=None, MASK_COUNTS={"train": 3, "validation": 1, "test": 2, "open_set": 2})
    os.makedirs(ns["MODELS"], exist_ok=True)
    os.makedirs(ns["TRAIN_MASKS"], exist_ok=True)
    exec(_cell(8), ns)
    return ns


def test_a_03_step_is_skipped_only_when_its_outputs_are_complete(tmp_path):
    """A run record that says finished is not enough: a stop during 03's test evaluation leaves no weights or
    results, and 03 --resume writes them only if the notebook launches the step again."""
    ns = _resume_cell(tmp_path)
    models, t = ns["MODELS"], "an_parts_s1"
    finished, done = ns["training_finished"], ns["training_done"]
    assert not finished(t) and not done(t)
    with open(f"{models}/run_{t}.json", "w") as f:
        json.dump({"finished": True, "weights_sha256": None}, f)
    os.makedirs(f"{models}/backup_{t}")
    assert not finished(t)                                    # the backup is removed after the record
    os.rmdir(f"{models}/backup_{t}")
    assert finished(t) and not done(t)                        # stopped before the weights and results
    open(f"{models}/Xception_ArcFace_{t}.weights.h5", "w").close()
    with open(f"{models}/run_{t}.json", "w") as f:
        json.dump({"finished": True, "weights_sha256": "ab" * 32}, f)
    assert not done(t)                                        # results_<tag>.json still missing
    open(f"{models}/results_{t}.json", "w").close()
    assert done(t)


def test_reused_masks_must_cover_this_sessions_mask_list(tmp_path):
    """masks64 copied from a session without attnet_gate has no open set: 21 must run again."""
    ns = _resume_cell(tmp_path)
    masks_done, folder = ns["masks_done"], ns["TRAIN_MASKS"]
    assert not masks_done("an")
    open(f"{folder}/masks64_an.npz", "w").close()
    with open(f"{folder}/masks64_summary_an.json", "w") as f:
        json.dump({"images_per_split": {"train": 3, "validation": 1, "test": 2}}, f)
    assert masks_done("an") and not masks_done("an", ns["MASK_COUNTS"])
    with open(f"{folder}/masks64_summary_an.json", "w") as f:
        json.dump({"images_per_split": {"train": 3, "validation": 1, "test": 2, "open_set": 2}}, f)
    assert masks_done("an", ns["MASK_COUNTS"])
    cell11 = _cell(11)
    assert "masks_done(TAG, MASK_COUNTS)" in cell11 and "not masks_done(TAG, MASK_COUNTS)" in cell11


@pytest.mark.parametrize("clock, mask_ready, runs", [(1.0, True, True), (11.1, True, False), (1.0, False, False)])
def test_external_baseline_04_waits_for_time_and_for_the_mask_model(tmp_path, clock, mask_ready, runs):
    """Cell 13 runs 04 on BASELINE_CKPT in the foreground only when it ends before HARD_STOP_H and 22 could use
    it (the mask model's checkpoint and correct-image list exist)."""
    w = str(tmp_path)
    ns = _helpers(tmp_path, RUN_FULL=True, ASA_ARCH="parts", W=w, MODELS=f"{w}/trained_models",
                  EVAL=f"{w}/eval_clean", ASA=f"{w}/asa", VERIF=f"{w}/verification",
                  BASELINE_CKPT="/kaggle/input/cs/trained_models/Xception_ArcFace_cs_s1.keras")
    calls = []
    ns.update(EST={"04_h": 0.5, "22_min_per_image": 0.1}, hours=lambda: clock,
              sh=lambda cmd, check=True, gpu=None: calls.append(cmd) or 1)
    os.makedirs(ns["MODELS"])
    os.makedirs(ns["EVAL"])
    if mask_ready:
        t = ns["tag"]("parts", 1)
        open(ns["ckpt"](ns["MODELS"], t), "w").close()
        open(f"{ns['EVAL']}/correct_test_{t}.csv", "w").close()
    exec(_cell(13), ns)
    assert calls == ([ns["CMD"]["04_base_ext"]] if runs else [])
    assert ("04_base_ext" in ns["A"]) == runs


def test_plan_puts_the_mask_models_after_the_masks(tmp_path):
    ns = _helpers(tmp_path, ARCHS=["attn_mask", "parts", "attn", "gap", "attnet_gate"])
    jobs = [(a, [f"03_{a}", f"04_{a}", f"15_{a}"], h)
            for a, h in (("attn_mask", 3.0), ("parts", 3.0), ("attn", 2.5), ("gap", 2.0), ("attnet_gate", 2.0))]
    chains, load = ns["plan_chains"](jobs, ["21_full"], 2.0)
    steps = chains[0] + chains[1]
    assert sorted(steps) == sorted(["21_full"] + [s for _, names, _ in jobs for s in names])
    assert chains[0][0] == "21_full"
    # mask-free models start right away on GPU 1; no mask model's hours are counted before the masks exist
    assert chains[1][:3] == ["03_attn", "04_attn", "15_attn"]
    total = 2.0 + sum(h for _, _, h in jobs)
    assert max(load.values()) <= total / 2 + 3.0 and min(load.values()) >= 2.0
    for g, names in chains.items():                                   # each model's steps stay in order on one GPU
        for a, model_steps, _ in jobs:
            if model_steps[0] in names:
                i = names.index(model_steps[0])
                assert names[i:i + 3] == model_steps
    chains, load = ns["plan_chains"]([("gap", ["03_gap"], 1.0)], [], 0.0)
    assert chains == {0: ["03_gap"], 1: []} and load == {0: 1.0, 1: 0.0}


def test_results_zip_keeps_no_checkpoint_mask_or_image(tmp_path):
    keep = _helpers(tmp_path)["zip_keep"]
    for rel in ("trained_models/Xception_ArcFace_an_parts_s1.keras", "trained_models/Xception_ArcFace_x.weights.h5",
                "train_masks/masks64_an.npz", "eye_masks/masks_asa_pool.npz", "eye_masks/qc/overlay_01.png",
                "trained_models/backup_an_parts_s1/chief/checkpoint", "eye_masks/combine.pth"):
        assert not keep(rel), rel
    for rel in ("trained_models/run_an_parts_s1.json", "train_masks/part_presence_an.csv", "asa/rows_asa_an_mask.csv",
                "analysis_asa/an_mask/fig_phi_shift.png", "verification/verification_roc_det_an_gap_s1.png",
                "eye_masks/mask_summary_asa_pool.json", "logs/03_an_parts_s1.log"):
        assert keep(rel), rel


def test_epoch_timer_reads_the_mask_trainer_line(tmp_path):
    # the estimate reads one Keras --verbose 2 line per epoch; the mask-loss trainer adds ce / mask_loss to it
    import yaml
    from conftest import make_images
    from src.models import anatomy_model as am
    ns = _helpers(tmp_path)
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"backbone": "tiny", "fine_tune_at": 0, "dense_units": 8, "n_classes": 2})
    cfg["anatomy"]["hidden"] = 4
    tf.keras.utils.set_random_seed(0)
    trainer = am.make_trainer(am.build_anatomy_model(cfg, "parts"), cfg)
    x = make_images(4)
    y = np.eye(2, dtype="float32")[[0, 1, 0, 1]]
    targets = np.full((4, 8, 8, 5), 1 / 64, np.float32)
    weights = np.ones((4, 5), np.float32)
    data = ((x, y, targets, weights), y)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        trainer.fit(*data, validation_data=data, epochs=2, batch_size=2, verbose=2)
    os.makedirs(tmp_path / "logs", exist_ok=True)
    (tmp_path / "logs" / "03_demo.log").write_text("Epoch 1/2\n" + out.getvalue())
    seconds = ns["epoch_seconds"]("03_demo")
    assert len(seconds) == 2 and all(s >= 0 for s in seconds), out.getvalue()
    assert "mask_loss" in out.getvalue()


def test_fields_the_notebook_reads_are_written_by_the_scripts(tmp_path):
    ns = _helpers(tmp_path)
    exec(_cell(5), ns)

    def absent(script, keys):
        with open(os.path.join(REPO, script), encoding="utf-8") as f:
            text = f.read()
        return [k for k in keys if f'"{k}"' not in text]

    assert not absent("scripts/03_train_baseline.py", ns["RUN_KEYS"] + ns["ANATOMY_KEYS"]
                      + ["test_acc_margin_free", "images_by_phase", "minutes_all_sessions"])
    assert not absent("src/data/masked_generator.py", ["share"])        # mask_coverage of run_<tag>.json
    assert not absent("scripts/04_eval_clean.py", ["n_images", "accuracy_margin_free", "arch", "region_masks"])
    assert not absent("scripts/15_eval_verification.py", ["per_mode", "timing", "load_model_s", "embedding_s",
                                                          "n_images_embedded", "subset", "mode", "eer", "git_commit"])
    assert not absent("scripts/21_make_train_masks.py", ["images_per_split", "timing", "segmenting_s", "total_s",
                                                         "n_images", "code_commit", "code_dirty"])
    assert not absent("scripts/22_anatomical_shapley.py", ["minutes", "n_images", "n_rows", "git_commit", "n_done"])
    assert not absent("scripts/03_train_baseline.py", ["finished", "weights_sha256"])      # training_done
    import importlib.util
    spec = importlib.util.spec_from_file_location("s21", os.path.join(REPO, "scripts", "21_make_train_masks.py"))
    s21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s21)
    assert s21.threshold_key(0.005) == "share_ge_0.5pct"            # the eyebrow line of checks_masks
