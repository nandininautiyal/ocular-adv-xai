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
           "scripts/21_make_train_masks.py", "scripts/22_anatomical_shapley.py", "scripts/23_analyse_asa.py",
           "scripts/24_gradient_masking.py"}


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
    assert not absent("scripts/22_anatomical_shapley.py", ["attacks", "rows_per_target_kind"])   # checks_asa
    assert not absent("scripts/22_anatomical_shapley.py", ["anatomy", "attention"])              # checks_asa
    assert not absent("scripts/24_gradient_masking.py", ["masking_flags", "checks_skipped", "seconds_per_image",
                                                         "git_commit", "n_images", "arch", "has_attention"])
    assert not absent("scripts/24_gradient_masking.py", ["seconds_per_batch", "batch_size"])     # masking_hours
    import importlib.util
    spec = importlib.util.spec_from_file_location("s21", os.path.join(REPO, "scripts", "21_make_train_masks.py"))
    s21 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s21)
    assert s21.threshold_key(0.005) == "share_ge_0.5pct"            # the eyebrow line of checks_masks


@pytest.mark.parametrize("settings", SETTINGS + [{"MASKING_ARCHS": ["parts"], "ASA_VERIF_PAIRS": False,
                                                  "ASA_BASELINE_HIDING": None,
                                                  "ASA_ATTACKS": "fgsm,pgd,aas_pgd,bpda_pgd"}])
def test_masking_check_and_the_asa_flags(settings):
    """24 runs on the smoke and the full model (seed SEEDS[0]) of every arch of MASKING_ARCHS that is trained, on
    that model's own correct-image list, after its 04, with the same grids in the smoke and the full run; every 22
    run of the session has the same 22 flags (the smoke runs time the full runs), except that aas_pgd runs only on
    models with an attention pool (on the baseline it is pgd itself)."""
    ns = _namespace(**settings)
    cmd, s0 = ns["CMD"], ns["SEEDS"][0]
    check = [a for a in ns["MASKING_ARCHS"] if a in ns["ARCHS"]]
    assert ns["MASK_CHECK"] == check and "attnet_gate" not in check
    for full, tags in ((False, ns["SMOKE_TAGS"]), (True, {a: ns["FULL_TAGS"][(a, s0)] for a in ns["ARCHS"]})):
        for arch, t in tags.items():
            assert (f"24_{t}" in cmd) == (arch in check)
            if arch not in check:
                continue
            a24, a03, a04 = _args(cmd[f"24_{t}"]), _args(cmd[f"03_{t}"]), _args(cmd[f"04_{t}"])
            assert a24["--ckpt"] == f"{a03['--out-dir']}/Xception_ArcFace_{t}.keras"
            assert a24["--correct-csv"] == f"{a04['--out-dir']}/correct_test_{t}.csv"
            assert a24["--tag"] == t and a24["--out-dir"] == ns["MASKING"]
            assert ns["NEEDS"][f"24_{t}"] == [f"04_{t}"]
            if full:
                assert a24["--per-class"] == "1" and "--max-images" not in a24
            else:
                assert a24["--max-images"] == str(ns["MASKING_SMOKE_IMAGES"]) and "--per-class" not in a24
            assert not [k for k in a24 if k.endswith(("-grid", "-levels", "steps", "restarts", "--taus", "--levels"))]
    for (arch, s), t in ns["FULL_TAGS"].items():
        assert (f"24_{t}" in cmd) == (arch in check and s == s0)
    runs = [k for k in cmd if k.startswith("22_")]
    assert runs
    arch_of = {"22_smoke": ns["SMOKE_ASA_ARCH"], "22_smoke_mask": ns["ASA_MASK_ARCH"], "22_mask": ns["ASA_MASK_ARCH"],
               "22_smoke_base": "baseline", "22_base": "baseline"}
    for name in runs:
        a = _args(cmd[name])
        attention = arch_of[name] in ("attn", "attn_mask", "parts")
        expected = [x for x in ns["ASA_ATTACKS"].split(",") if x != "aas_pgd" or attention]
        assert a["--attacks"] == ",".join(expected) and a["--pixel-conditions"] == ns["ASA_PIXEL_CONDITIONS"]
        assert a.get("--baseline-hiding") == ns["ASA_BASELINE_HIDING"]
        assert ("--verif-pairs" in a) == ns["ASA_VERIF_PAIRS"]
    defaults = _namespace()
    assert "aas_pgd" in defaults["ASA_ATTACKS"].split(",") and defaults["ASA_VERIF_PAIRS"] is True
    assert set(defaults["MASK_CHECK"]) == {"attn_mask", "parts", "gap"}
    # the masking check is one of a model's steps: planned and estimated with it, skipped once its summary exists
    assert "MODEL_STEPS" in _cell(11) and "EST['24_h']" in _cell(11)
    assert "'24': f'{MASKING}/summary_masking_{t}.json'" in _cell(12)


def test_aas_pgd_only_where_there_is_an_attention_pool(tmp_path):
    """ATTENTION_ARCHS are the archs built with an attention pool; with a baseline the 22 runs of the base model leave
    aas_pgd out (it would be pgd itself), and checks_asa expects the attacks of each model from its summary."""
    import yaml
    from src.attacks.masking import has_attention
    from src.models import anatomy_model as am
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"backbone": "tiny", "fine_tune_at": 0, "dense_units": 8, "n_classes": 3})
    cfg["anatomy"]["hidden"] = 4
    ns = _helpers(tmp_path, ASA=str(tmp_path / "asa"), ANALYSIS=str(tmp_path / "analysis"),
                  BASELINE_CKPT="/kaggle/input/cs/trained_models/Xception_ArcFace_cs_s1.keras")
    assert set(ns["ATTENTION_ARCHS"]) == {a for a in am.ARCHS if has_attention(am.build_anatomy_model(cfg, a))}
    assert "aas_pgd" not in _args(ns["CMD"]["22_base"])["--attacks"].split(",")
    assert "aas_pgd" in _args(ns["CMD"]["22_mask"])["--attacks"].split(",")
    exec(_cell(5), ns)
    os.makedirs(ns["ASA"])
    os.makedirs(f"{ns['ANALYSIS']}/an_base")
    open(f"{ns['ASA']}/rows_asa_an_base.csv", "w").close()
    open(f"{ns['ANALYSIS']}/an_base/report_asa.md", "w").close()

    def check(attacks, attention):
        with open(f"{ns['ASA']}/summary_asa_an_base.json", "w") as f:
            json.dump({"git_commit": "0" * 40, "attacks": attacks, "anatomy": {"attention": attention},
                       "rows_per_target_kind": {"true_class": 1, "genuine": 1}}, f)
        return all(ok for _, ok, _, _ in ns["checks_asa"]("an_base"))

    assert check(["fgsm", "pgd"], False) and check(["fgsm", "pgd", "aas_pgd"], True)
    assert not check(["fgsm", "pgd", "aas_pgd"], False) and not check(["fgsm", "pgd"], True)


def test_masking_estimate_per_batch():
    """Cell 11 estimates 24's full run per batch when the smoke run filled one (a batch of 4 images costs about as
    much as one of 16), else per image, which can only overstate it."""
    import ast
    tree = ast.parse(_cell(11))
    [fn] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "masking_hours"]
    ns = {"N_CLASSES": 300}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "cell11", "exec"), ns)
    full = {"seconds_per_batch": 360.0, "seconds_per_image": 22.5, "n_images": 16, "batch_size": 16}
    assert ns["masking_hours"](full) == pytest.approx((360.0 * 19 + 300) / 3600)
    partial = dict(full, n_images=4, seconds_per_batch=300.0, seconds_per_image=75.0)
    assert ns["masking_hours"](partial) == pytest.approx((75.0 * 300 + 300) / 3600)
    older = {"seconds_per_image": 10.0, "n_images": 4}                  # a summary written before seconds_per_batch
    assert ns["masking_hours"](older) == pytest.approx((10.0 * 300 + 300) / 3600)
    assert _namespace()["MASKING_SMOKE_IMAGES"] == 16                  # one full batch of 24's --batch-size 16


def test_verification_pairs_keep_classes_with_a_template(tmp_path):
    """22 --verif-pairs stops on an image without a second image of its class: cells 10 and 13 keep only classes
    with two or more images when ASA_VERIF_PAIRS is on."""
    import pandas as pd
    frame = pd.DataFrame({"relpath": list("abcde"), "true_idx": [0, 0, 1, 2, 2]})
    assert list(_helpers(tmp_path)["with_templates"](frame).relpath) == ["a", "b", "d", "e"]
    assert _helpers(tmp_path, ASA_VERIF_PAIRS=False)["with_templates"](frame) is frame
    assert "with_templates(" in _cell(10) and "with_templates(" in _cell(13)


def test_masking_flags_warn_but_a_missing_or_foreign_summary_fails(tmp_path):
    ns = _helpers(tmp_path, MASKING=str(tmp_path / "masking"))
    exec(_cell(5), ns)
    os.makedirs(ns["MASKING"])
    t = "an_parts_s1"
    [(name, ok, _, hard)] = ns["checks_masking"](t)
    assert not ok and hard
    summary = {"git_commit": "0" * 40, "masking_flags": [], "checks_skipped": [], "n_images": 3, "arch": "parts"}
    with open(f"{ns['MASKING']}/summary_masking_{t}.json", "w") as f:
        json.dump(summary, f)
    open(f"{ns['MASKING']}/rows_masking_{t}.csv", "w").close()
    assert all(ok for _, ok, _, _ in ns["checks_masking"](t))
    with open(f"{ns['MASKING']}/summary_masking_{t}.json", "w") as f:
        json.dump(dict(summary, masking_flags=["pgd_budget_not_monotone"]), f)
    (_, ok1, _, hard1), (_, ok2, detail, hard2) = ns["checks_masking"](t)
    assert ok1 and hard1 and not ok2 and not hard2 and "pgd_budget_not_monotone" in detail
    with open(f"{ns['MASKING']}/summary_masking_{t}.json", "w") as f:
        json.dump(dict(summary, git_commit="1234567"), f)
    assert not ns["checks_masking"](t)[0][1]
