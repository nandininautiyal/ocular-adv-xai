"""scripts/22_anatomical_shapley.py on the tiny conftest model with synthetic eye masks, and
scripts/23_analyse_asa.py on its output (the CPU end-to-end run of 22 -> 23)."""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from src.attacks.budget import quantise
from src.models.arcface_inference import predict_logits
from src.xai.seg_masks import save_mask_file
from src.xai.shapley import DEFAULT_GROUPS, normalise_groups

CLASSES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")
GROUPS = ("iris", "sclera", "lid", "brow", "skin")
LEVELS = (2, 8)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load("22_anatomical_shapley")


@pytest.fixture(scope="module")
def analyse():
    return _load("23_analyse_asa")


def eye_labels(n):
    """conftest.make_images anatomy: the disk is the iris, a ring the sclera, the top 16 rows the
    brow, rows 16-23 the lid on even images only (odd images have no lid), the bottom 16 rows unknown."""
    yy, xx = np.mgrid[0:256, 0:256]
    out = []
    for k in range(n):
        r2 = (yy - (48 + 40 * k)) ** 2 + (xx - (200 - 30 * k)) ** 2
        labels = np.zeros((256, 256), np.uint8)
        labels[r2 < 45 ** 2] = 2
        labels[r2 < 30 ** 2] = 1
        labels[:16] = 5
        if k % 2 == 0:
            labels[16:24] = 3
        labels[240:] = 255
        out.append(labels)
    return np.stack(out)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, model, images, labels):
    """Checkpoint, PNGs, a correct-image CSV with manifest relpaths (as 04 writes it) and a mask file."""
    root = tmp_path_factory.mktemp("asa")
    ckpt = root / "tiny.keras"
    model.save(ckpt)
    rows = []
    for i, (img, label) in enumerate(zip(images, labels)):
        path = root / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        rows.append({"link_path": str(path), "relpath": f"test_office/enrollment_left_eye/{1000 + label}/{path.name}",
                     "true_idx": int(label), "true_subject": 1000 + int(label), "eye": "left", "device": "test",
                     "lighting": "office", "phase": "enrollment"})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "correct_test_tiny.csv", index=False)
    masks = root / "masks_tiny.npz"
    save_mask_file(masks, frame.relpath, eye_labels(len(frame)), meta={"model": "synthetic"})
    clean = np.stack([np.asarray(Image.open(p), dtype=np.float32) / 255 for p in frame.link_path])
    return {"root": root, "ckpt": ckpt, "csv": root / "correct_test_tiny.csv", "masks": masks, "frame": frame,
            "clean": clean}


def common(inputs, out_dir):
    return ["--ckpt", str(inputs["ckpt"]), "--correct-csv", str(inputs["csv"]), "--masks", str(inputs["masks"]),
            "--out-dir", str(out_dir)]


@pytest.fixture(scope="module")
def run(script, inputs):
    out_dir = inputs["root"] / "out"
    script.main(common(inputs, out_dir) + ["--max-images", "4", "--levels", ",".join(map(str, LEVELS)),
                                           "--attacks", "fgsm,pgd,bpda_pgd", "--defences", "jpeg", "--steps", "2",
                                           "--batch-size", "3", "--workers", "2", "--seed", "3", "--tag", "tiny"])
    rows = pd.read_csv(out_dir / "rows_asa_tiny.csv")
    summary = json.loads((out_dir / "summary_asa_tiny.json").read_text())
    return {"dir": out_dir, "rows": rows, "summary": summary}


def test_rows_cover_every_condition(script, run):
    rows, summary = run["rows"], run["summary"]
    n_conditions = script.conditions_per_image(len(LEVELS), 1, ["fgsm", "pgd", "bpda_pgd"])
    assert n_conditions == 1 + 1 + 2 * 2 + 2 * 2 * 2 + 2 * 1 * 2
    true = rows[rows.target == "true"]
    assert rows.image_id.nunique() == 4 and len(true) == 4 * n_conditions * 3
    assert set(rows.condition) == {"clean", "noise", "defence_on_clean", "noise+defence", "attack", "attack+defence"}
    assert set(rows.baseline) == {"blur", "mean", "inpaint"} and set(rows.score) == {"margin"}
    bpda = rows[rows.attack == "bpda_pgd"]
    assert len(bpda) and (bpda.adapted_to == "jpeg").all() and rows[rows.attack == "pgd"].adapted_to.isna().all()
    assert {f"{p}_{g}" for p in ("phi", "area", "present", "mass_layercam", "mass_eigencam") for g in GROUPS} \
        <= set(rows.columns)
    assert {f"value_{k}" for k in range(32)} <= set(rows.columns)
    assert {"del_auc_asa", "del_auc_layercam", "del_auc_eigencam", "del_auc_random"} <= set(rows.columns)
    assert not any(c.startswith(("mass_attention", "receipt_", "del_auc_attention")) for c in rows.columns)
    assert summary["anatomy"]["attention"] is False and summary["anatomy"]["receipt"] is False
    assert summary["n_rows"] == len(rows) and summary["levels"] == list(LEVELS)
    assert summary["groups"] == {g: list(c) for g, c in normalise_groups(DEFAULT_GROUPS)}
    assert summary["git_commit"] and "tensorflow" in summary["versions"]
    assert summary["asa_forward_passes"]["upper_bound_per_image"] == n_conditions * 3 * 32
    assert summary["group_presence"] == {"iris": 4, "sclera": 4, "lid": 2, "brow": 4, "skin": 4}
    images = pd.read_csv(run["dir"] / "images_asa_tiny.csv")
    assert sorted(images.image_id) == sorted(rows.image_id.unique())


def test_rows_obey_the_game_and_the_image_contract(run):
    rows = run["rows"]
    phi = rows[[f"phi_{g}" for g in GROUPS]].to_numpy()
    np.testing.assert_allclose(phi.sum(axis=1), rows.v_full - rows.v_empty, atol=1e-9)
    np.testing.assert_array_equal(rows.v_full, rows.value_31)
    np.testing.assert_array_equal(rows.v_empty, rows.value_0)
    # the score with every part kept is the true-class margin of the condition image
    true = rows[rows.target == "true"]
    np.testing.assert_allclose(true.v_full, true.margin_true, atol=1e-3)
    # odd images have no lid: an exact null player, and 16 instead of 32 coalitions per baseline
    no_lid = rows[~rows.present_lid.astype(bool)]
    assert len(no_lid) and (no_lid.phi_lid == 0).all() and (no_lid.n_forward == 3 * 16).all()
    assert (rows[rows.present_lid.astype(bool)].n_forward == 3 * 32).all()
    assert (rows.image_id[~rows.present_lid.astype(bool)] % 2 == 1).all()
    # budgets: 8-bit images within whole levels
    for cond in ("noise", "attack"):
        sub = rows[rows.condition == cond]
        assert (sub.pert_linf <= sub.epsilon + 1e-6).all()
        np.testing.assert_array_equal(sub.budget_levels, np.round(sub.epsilon * 255))
    noise = rows[rows.condition == "noise"]
    np.testing.assert_allclose(noise.pert_linf, noise.epsilon, atol=1e-4)
    # adversarial-class rows explain the prediction, only where it is wrong
    pred = rows[rows.target == "pred"]
    assert len(pred) and (pred.target_idx == pred.pred_idx).all() and not pred.correct.any()
    assert (pred.condition != "clean").all()
    defended = rows[rows.condition == "attack+defence"]
    assert defended.attack_success.notna().all() and defended.defended_clean_correct.notna().all()
    finite = rows[["del_auc_asa", "del_auc_random"]].notna().all(axis=1)
    assert finite.mean() > 0.9


def test_adaptive_attack_success_is_judged_after_the_defence(run):
    """BPDA-PGD is computed through the defence to fool f(D(x')): its attack_success is the defended outcome
    (as 14's success_after_defence); fgsm / pgd keep the undefended outcome. Both are in the summary."""
    rows, summary = run["rows"], run["summary"]
    rows = rows.assign(adapted_to=rows.adapted_to.fillna(""))
    defended = rows[rows.condition == "attack+defence"]
    bpda, plain = defended[defended.attack == "bpda_pgd"], defended[defended.attack != "bpda_pgd"]
    assert len(bpda) and len(plain)
    assert (bpda.attack_success.astype(bool) == ~bpda.correct.astype(bool)).all()
    assert (plain.attack_success.astype(bool) == plain.attack_success_undefended.astype(bool)).all()
    keys = ["image_id", "attack", "adapted_to", "epsilon"]
    attacked = rows[(rows.condition == "attack") & (rows.target == "true") & (rows.baseline == "blur")]
    m = defended.merge(attacked[keys + ["correct"]], on=keys, suffixes=("", "_attacked"), validate="many_to_one")
    assert (m.attack_success_undefended.astype(bool) == ~m.correct_attacked.astype(bool)).all()
    after = summary["attack_success_after_defence_rate"]
    true_blur = bpda[(bpda.target == "true") & (bpda.baseline == "blur")]
    for eps, g in true_blur.groupby("epsilon"):
        r = after[f"bpda_pgd[jpeg]@{eps:.6g}+jpeg"]
        assert r["n"] == 4 and r["rate"] == pytest.approx((~g.correct.astype(bool)).mean())
    assert any(k.startswith("pgd@") for k in after)
    assert not any("bpda" in k for k in summary["restoration_rate_among_successful_attacks"])


def without_wall_times(rows):
    """The rows without the time_<method> columns: wall times are measurements and differ between two runs
    (every other column, including the forward-pass counts nfwd_<method>, must be equal)."""
    times = [c for c in rows.columns if c.startswith("time_")]
    assert times and (rows[times].fillna(0.0) >= 0).all().all()
    return rows.drop(columns=times)


def test_resume_continues_a_stopped_run(script, inputs, tmp_path, monkeypatch, capsys):
    """Rows are written after every batch: a run stopped after its first batch continues with --resume and
    gives the rows and rates of an uninterrupted run."""
    opts = ["--max-images", "4", "--levels", "2", "--attacks", "fgsm", "--defences", "jpeg", "--baselines", "mean",
            "--cams", "layercam", "--batch-size", "2", "--seed", "5", "--tag", "r"]
    script.main(common(inputs, tmp_path / "whole") + opts)
    want = pd.read_csv(tmp_path / "whole" / "rows_asa_r.csv")
    whole = json.loads((tmp_path / "whole" / "summary_asa_r.json").read_text())

    class Stopped(RuntimeError):
        pass

    real, calls = script.run_batch, []

    def stop_at_second_batch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Stopped()
        return real(*args, **kwargs)

    out = tmp_path / "resumed"
    monkeypatch.setattr(script, "run_batch", stop_at_second_batch)
    with pytest.raises(Stopped):
        script.main(common(inputs, out) + opts + ["--resume"])
    state = json.loads((out / "parts_asa_r" / "state.json").read_text())
    assert state["n_done"] == 2 and not (out / "rows_asa_r.csv").exists()
    assert sorted(os.listdir(out / "parts_asa_r")) == ["rows_000000.csv", "state.json"]
    monkeypatch.setattr(script, "run_batch", real)
    capsys.readouterr()
    script.main(common(inputs, out) + opts + ["--resume", "--workers", "1"])   # --workers does not change rows
    assert "resuming after 2 of 4 images" in capsys.readouterr().out
    pd.testing.assert_frame_equal(without_wall_times(pd.read_csv(out / "rows_asa_r.csv")), without_wall_times(want))
    summary = json.loads((out / "summary_asa_r.json").read_text())
    assert summary["resumed"] and summary["resumed_after_images"] == 2 and not (out / "parts_asa_r").exists()
    for section in ("attack_success_rate", "attack_success_after_defence_rate", "noise_control_flip_rate",
                    "restoration_rate_among_successful_attacks"):
        assert summary[section] == whole[section], section
    assert summary["asa_forward_passes"] == whole["asa_forward_passes"]
    capsys.readouterr()                                    # the same command again: nothing to do
    script.main(common(inputs, out) + opts + ["--resume"])
    assert "nothing to resume" in capsys.readouterr().out
    # a stopped run continues only with its own settings; without --resume it starts again from scratch
    calls.clear()
    monkeypatch.setattr(script, "run_batch", stop_at_second_batch)
    with pytest.raises(Stopped):
        script.main(common(inputs, out) + opts + ["--resume", "--tag", "r2"])
    monkeypatch.setattr(script, "run_batch", real)
    with pytest.raises(SystemExit, match="other settings"):
        script.main(common(inputs, out) + opts + ["--resume", "--tag", "r2", "--seed", "6"])
    script.main(common(inputs, out) + opts + ["--tag", "r2"])
    assert not json.loads((out / "summary_asa_r2.json").read_text())["resumed"]
    pd.testing.assert_frame_equal(without_wall_times(pd.read_csv(out / "rows_asa_r2.csv")), without_wall_times(want))


def test_noise_follows_the_05_convention(run, model, inputs):
    """Sign noise from default_rng(seed), drawn per batch and budget in order, rounded to 8 bits."""
    rows = run["rows"]
    images = pd.read_csv(run["dir"] / "images_asa_tiny.csv")
    rng = np.random.default_rng(3)
    first = images.image_id[:3].to_numpy()                     # batch size 3
    clean = inputs["clean"][first]
    labels = images.true_idx[:3].to_numpy()
    for level in LEVELS:
        eps = round(level / 255, 7)
        noise = eps * np.sign(rng.uniform(-1.0, 1.0, clean.shape)).astype(np.float32)
        noisy = quantise(np.clip(clean + noise, 0, 1), clean, eps)
        logits = predict_logits(model, noisy)
        margin = logits[np.arange(3), labels] - np.where(np.eye(logits.shape[1])[labels] > 0, -np.inf, logits).max(1)
        got = rows[(rows.condition == "noise") & np.isclose(rows.epsilon, eps) & (rows.target == "true")
                   & (rows.baseline == "blur")].set_index("image_id").margin_true
        np.testing.assert_allclose(got[first].to_numpy(), margin, atol=1e-3)


def test_analysis_end_to_end(run, analyse):
    out = run["dir"] / "analysis"
    checks = analyse.check_rows(run["rows"], run["summary"])
    assert any("summary JSON" in c for c in checks) and any("efficiency" in c for c in checks)
    analyse.main(["--rows", str(run["dir"] / "rows_asa_tiny.csv"), "--n-boot", "20", "--out-dir", str(out)])
    for name in ("phi_shift", "phi_shift_by_outcome", "baseline_sensitivity", "faithfulness",
                 "attribution_restoration"):
        assert len(pd.read_csv(out / f"{name}.csv")), name
    assert (out / "report_asa.md").read_text(encoding="utf-8").startswith("# Anatomical Shapley Attribution")
    assert (out / "fig_phi_shift.png").exists() and (out / "fig_deletion.png").exists()
    shift = pd.read_csv(out / "phi_shift.csv")
    assert set(shift.condition) == {"attack", "attack+defence"} and set(shift.measure) == {"phi", "share"}
    lid = shift[shift.group == "lid"]
    assert (lid.n == 2).all()                                  # only the images with a lid
    # a summary that disagrees with the rows stops the analysis
    tampered = json.loads(json.dumps(run["summary"]))
    name = next(iter(tampered["attack_success_rate"]))
    tampered["attack_success_rate"][name]["n"] += 1
    with pytest.raises(ValueError, match="attack_success_rate"):
        analyse.check_rows(run["rows"], tampered)


def test_inputs_are_checked_before_the_model_loads(script, inputs, tmp_path):
    frame = inputs["frame"]
    partial = tmp_path / "masks_partial.npz"
    save_mask_file(partial, frame.relpath[1:], eye_labels(len(frame))[1:])
    args = common(inputs, tmp_path / "x")
    args[1] = str(tmp_path / "no_such_model.keras")               # never reached
    args[5] = str(partial)
    with pytest.raises(SystemExit, match="no mask for 1 of the 5"):
        script.main(args + ["--per-class", "1"])
    with pytest.raises(SystemExit, match="pupil"):
        script.main(common(inputs, tmp_path / "x") + ["--groups", "iris=iris;pupil=pupil"])
    for bad in (["--attacks", "cw"], ["--baselines", "zero"], ["--levels", "0"], ["--levels", "1.5"],
                ["--score", "prob"], ["--workers", "0"], ["--attacks", "bpda_pgd", "--defences", "none"],
                ["--shift-part", "pupil:2"], ["--shift-part", "lid"], ["--shift-part", "lid:two"]):
        with pytest.raises(SystemExit):
            script.main(common(inputs, tmp_path / "x") + bad)
    assert not (tmp_path / "x").exists()


def test_shift_part_changes_only_the_masks(script, inputs, tmp_path):
    """--shift-part lid:0 gives exactly the rows of a run without it; lid:-8 removes the 8-row lid band, so the lid
    is absent everywhere; the setting is recorded in the summary (and so in the resume key)."""
    cheap = ["--max-images", "5", "--levels", "1", "--attacks", "fgsm", "--defences", "none", "--part-baselines",
             "none", "--pixel-baselines", "none", "--baselines", "blur", "--seed", "3", "--workers", "1"]
    out = {}
    for name, extra in (("plain", []), ("zero", ["--shift-part", "lid:0"]), ("gone", ["--shift-part", "lid:-8"])):
        script.main(common(inputs, tmp_path) + cheap + ["--tag", name] + extra)
        out[name] = (pd.read_csv(tmp_path / f"rows_asa_{name}.csv"),
                     json.loads((tmp_path / f"summary_asa_{name}.json").read_text()))
    plain, zero, gone = (out[k][0] for k in ("plain", "zero", "gone"))
    phi = [c for c in plain.columns if c.startswith("phi_")]
    assert phi and len(plain) == len(zero) == len(gone)
    pd.testing.assert_frame_equal(plain[phi], zero[phi])
    assert out["plain"][1]["group_presence"]["lid"] > 0 and out["gone"][1]["group_presence"]["lid"] == 0
    assert out["gone"][1]["args"]["shift_part"] == "lid:-8" and out["plain"][1]["args"]["shift_part"] is None


def test_groups_from_a_config_file(script, tmp_path):
    cfg = tmp_path / "anatomy.yml"
    cfg.write_text("anatomy:\n  part_groups:\n    eye: [iris, sclera, caruncle]\n    rest: [skin, lid, brow]\n")
    assert script.resolve_groups(str(cfg)) == (("eye", ("iris", "sclera", "caruncle")),
                                               ("rest", ("skin", "lid", "brow")))
    assert script.resolve_groups(None) == normalise_groups(DEFAULT_GROUPS)
    assert script.resolve_groups("iris=iris;skin=skin") == (("iris", ("iris",)), ("skin", ("skin",)))


@pytest.fixture(scope="module")
def parts_model():
    """A tiny parts model (src/models/anatomy_model.py) with a perturbed, non-trivial head; its parts
    are the default groups in a different order than 22's groups."""
    import tensorflow as tf
    import yaml
    from src.models import anatomy_model as am
    with open(os.path.join(REPO, "configs", "anatomy.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model"].update({"backbone": "tiny", "fine_tune_at": 0, "dense_units": 16, "n_classes": 5})
    cfg["anatomy"]["hidden"] = 8
    cfg["anatomy"]["part_groups"] = {g: cfg["anatomy"]["part_groups"][g] for g in ("skin", "iris", "sclera",
                                                                                   "brow", "lid")}
    tf.keras.utils.set_random_seed(4)
    model = am.build_anatomy_model(cfg, "parts")
    rng = np.random.default_rng(4)
    model.set_weights([w + rng.normal(0, 0.3, w.shape).astype(np.float32) for w in model.get_weights()])
    return model


@pytest.mark.parametrize("score", ["margin", "logit"])
def test_attention_and_receipt_columns(script, parts_model, images, labels, score):
    """A parts model gets attention part masses and the receipt in the units of the score, summing to
    v_full, as columns and deletion orders."""
    caps = script.anatomy_capabilities(parts_model)
    model_parts = ["skin", "iris", "sclera", "brow", "lid"]
    assert caps["arch"] == "parts" and caps["receipt"] and caps["attention"]
    assert caps["receipt_parts"] == model_parts
    groups = normalise_groups(DEFAULT_GROUPS)
    index = [model_parts.index(g) for g in GROUPS]
    base = [{"image_id": i, "true_idx": int(labels[i])} for i in range(2)]
    ctx = script.Context(parts_model, images[:2], labels[:2], base, eye_labels(2), CLASSES, groups, ["mean"], score,
                         ["layercam"], None, caps, index, 64, 0)
    rows, pred = script.condition_rows(ctx, images[:2], {"condition": "clean"})
    logits = predict_logits(parts_model, images[:2])
    for row, i in zip(rows, range(2)):
        receipt = np.array([row[f"receipt_{g}"] for g in GROUPS])
        np.testing.assert_allclose(receipt.sum(), row["receipt_sum"], atol=1e-4)
        np.testing.assert_allclose(row["receipt_sum"], row["v_full"], atol=1e-3)
        target = logits[i, row["target_idx"]]
        expected = target if score == "logit" else target - np.delete(logits[i], row["target_idx"]).max()
        np.testing.assert_allclose(row["v_full"], expected, atol=1e-3)
        masses = np.array([row[f"mass_attention_{g}"] for g in GROUPS])
        assert np.all(masses >= 0) and masses.sum() <= 1 + 1e-6
        assert np.isfinite(row["del_auc_attention"]) and np.isfinite(row["del_auc_receipt"])
    # groups the model has no part for get NaN receipts (and no receipt order)
    ctx.receipt_index = [1, 2, None, 3, 0]
    rows, _ = script.condition_rows(ctx, images[:2], {"condition": "clean"})
    assert np.isnan(rows[0]["receipt_lid"]) and np.isfinite(rows[0]["receipt_iris"])
    assert np.isnan(rows[0]["del_auc_receipt"])                    # image 0 has a lid with no receipt


def test_warm_up_covers_the_timed_regions(script, parts_model, images, labels, monkeypatch):
    """After warm_up, a condition builds no sub-model inside the timed regions and every coalition forward
    pass uses a batch size the warm-up already ran (one-time costs stay out of the time_ columns)."""
    from src.models import anatomy_model, arcface_inference
    from src.xai import shapley
    caps = script.anatomy_capabilities(parts_model)
    groups = normalise_groups(DEFAULT_GROUPS)
    index = [caps["receipt_parts"].index(g) for g in GROUPS]
    base = [{"image_id": i, "true_idx": int(labels[i])} for i in range(2)]
    ctx = script.Context(parts_model, images[:2], labels[:2], base, eye_labels(2), CLASSES, groups, ["mean"],
                         "margin", ["layercam", "eigencam"], None, caps, index, 10, 0,
                         part_estimators=("kernelshap", "lime"), pixel_estimators=("kernelshap", "lime"),
                         pixel_samples=24, patch=64, pixel_conditions=("clean",), pixel_methods=("mean",))
    arcface_inference._CACHE.pop(parts_model, None)          # start cold: no cached sub-model
    anatomy_model._SUB_MODELS.pop(parts_model, None)
    assert script._chunk_sizes(500, 96) == {96, 20} and script._chunk_sizes(16, 96) == {16}
    assert script._chunk_sizes(192, 96) == {96}
    sizes = script.warm_up_sizes(ctx)
    assert sizes == list(range(1, 11))                 # 2^5 games > 10 per pass; 24 pixel coalitions -> 10, 10, 4
    assert script.warm_up(ctx) > 0
    built = (set(arcface_inference._CACHE[parts_model]), set(anatomy_model._SUB_MODELS[parts_model]))
    assert {"attention", "parts"} <= built[1] and len(built[0]) >= 2         # logits and the CAM layer

    used, forward = [], shapley._forward

    def recording(model, batch, output, batch_size):
        used.extend(min(batch_size, len(batch) - i) for i in range(0, len(batch), batch_size))
        return forward(model, batch, output, batch_size)

    monkeypatch.setattr(shapley, "_forward", recording)
    rows, _ = script.condition_rows(ctx, images[:2], {"condition": "clean"})
    assert rows and used and set(used) <= set(sizes)
    assert (set(arcface_inference._CACHE[parts_model]), set(anatomy_model._SUB_MODELS[parts_model])) == built


def test_parts_model_end_to_end(script, analyse, parts_model, inputs, tmp_path):
    """22 on a saved parts checkpoint, then 23's checks, including the receipt summing to v_full."""
    ckpt = tmp_path / "parts.keras"
    parts_model.save(ckpt)
    out = tmp_path / "out"
    args = common(inputs, out)
    args[1] = str(ckpt)
    script.main(args + ["--max-images", "2", "--levels", "2", "--attacks", "fgsm", "--defences", "none",
                        "--baselines", "mean", "--cams", "layercam", "--tag", "parts"])
    rows = pd.read_csv(out / "rows_asa_parts.csv")
    summary = json.loads((out / "summary_asa_parts.json").read_text())
    assert summary["anatomy"]["arch"] == "parts" and summary["anatomy"]["attention"]
    assert {f"receipt_{g}" for g in GROUPS} | {f"mass_attention_{g}" for g in GROUPS} | {"receipt_sum"}         <= set(rows.columns)
    checks = analyse.check_rows(rows, summary)
    assert any("receipt sums to v_full" in c for c in checks)
    broken = rows.copy()
    broken.loc[0, "receipt_sum"] += 1.0
    with pytest.raises(ValueError, match="receipt"):
        analyse.check_rows(broken, summary)


# ---------------------------------------------------------------------------
# Sampling baselines, verification pairs, run times and agreement
# ---------------------------------------------------------------------------

FU_OPTS = ["--max-images", "4", "--levels", "2", "--attacks", "fgsm,pgd", "--defences", "jpeg", "--steps", "2",
           "--baselines", "blur,mean", "--verif-pairs", "--pixel-conditions", "clean,attack", "--pixel-samples", "24",
           "--batch-size", "2", "--seed", "3"]


@pytest.fixture(scope="module")
def pair_inputs(tmp_path_factory, model, images, labels):
    """Two correct images per subject, one per phase (the second with faint noise), so every image has a
    genuine template from the other phase; eye masks for all of them."""
    root = tmp_path_factory.mktemp("asa_pairs")
    ckpt = root / "tiny.keras"
    model.save(ckpt)
    rows, masks = [], eye_labels(len(images))
    for phase, offset in (("enrollment", 0), ("verification", 1)):
        for i, (img, label) in enumerate(zip(images, labels)):
            noisy = np.clip(img + offset * np.random.default_rng(i).normal(0, 0.01, img.shape), 0, 1)
            path = root / f"{label}_{i}_{phase}.png"
            Image.fromarray((noisy * 255).round().astype(np.uint8)).save(path)
            rows.append({"link_path": str(path), "relpath": f"test_office/{phase}_left_eye/{1000 + label}/{path.name}",
                         "true_idx": int(label), "true_subject": 1000 + int(label), "eye": "left", "device": "test",
                         "lighting": "office", "phase": phase})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "correct_pairs.csv", index=False)
    save_mask_file(root / "masks_pairs.npz", frame.relpath, np.concatenate([masks, masks]), meta={"model": "synthetic"})
    return {"root": root, "ckpt": ckpt, "csv": root / "correct_pairs.csv", "masks": root / "masks_pairs.npz",
            "frame": frame}


@pytest.fixture(scope="module")
def fu_run(script, pair_inputs):
    out_dir = pair_inputs["root"] / "out"
    script.main(common(pair_inputs, out_dir) + FU_OPTS + ["--tag", "fu"])
    return {"dir": out_dir, "rows": pd.read_csv(out_dir / "rows_asa_fu.csv"),
            "summary": json.loads((out_dir / "summary_asa_fu.json").read_text())}


def test_templates_prefer_the_other_phase(script, pair_inputs):
    pool = pair_inputs["frame"]
    sampled = pool.iloc[[0, 6]].reset_index().rename(columns={"index": "image_id"})
    templates = script.choose_templates(sampled, pool, seed=1)
    genuine, impostor = templates["genuine"], templates["impostor"]
    assert genuine.template_image_id.tolist() == [5, 1]             # the same subject, other phase
    assert genuine.other_phase.all() and (genuine.class_idx == sampled.true_idx).all()
    assert (impostor.class_idx != sampled.true_idx).all() and impostor.other_phase.all()
    assert script.choose_templates(sampled, pool, seed=1)["impostor"].equals(impostor)       # seeded
    with pytest.raises(ValueError, match="no other correct image"):
        script.choose_templates(sampled, pool.drop(index=5), seed=1)


def _two_eye_pool(n_classes=6, per=3, eye_column=True):
    """A correct-image pool with both eyes and both phases of every class (relpaths as in 01)."""
    rows = []
    for c in range(n_classes):
        for eye in ("left", "right"):
            for phase in ("enrollment", "verification"):
                for k in range(per):
                    name = f"{1000 + c}_{k}_{eye[0]}_c.png"
                    rows.append({"link_path": f"/split/{1000 + c}/{phase}_{eye}_{name}",
                                 "relpath": f"test_office/{phase}_{eye}_eye/{1000 + c}/{name}",
                                 "true_idx": c, "eye": eye, "phase": phase})
    frame = pd.DataFrame(rows)
    return frame if eye_column else frame.drop(columns="eye")


@pytest.mark.parametrize("eye_column", [True, False])
def test_templates_have_the_probes_eye(script, eye_column):
    """Left probes get left templates and right probes right ones (the eye column, else the relpath)."""
    pool = _two_eye_pool(eye_column=eye_column)
    eyes = script.image_eyes(pool)
    assert eyes.tolist() == [r.split("/")[1].split("_")[1] for r in pool.relpath]
    sampled = pool.iloc[::5].reset_index().rename(columns={"index": "image_id"})
    assert set(eyes[sampled.image_id]) == {"left", "right"}
    for seed in range(5):
        templates = script.choose_templates(sampled, pool, seed=seed)
        for kind, t in templates.items():
            assert (eyes[t.template_image_id].to_numpy() == eyes[sampled.image_id].to_numpy()).all(), kind
            assert t.same_eye.all() and t.other_phase.all()
        assert (templates["genuine"].class_idx == sampled.true_idx).all()
        assert (templates["impostor"].class_idx != sampled.true_idx).all()
        assert (templates["genuine"].template_image_id != sampled.image_id).all()
    again = script.choose_templates(sampled, pool, seed=4)
    assert all(again[k].equals(templates[k]) for k in templates)                         # seeded


def test_templates_fall_back_to_any_eye(script, capsys):
    """Without a same-eye candidate the template is of the other eye, flagged and warned about once; the
    impostor class is drawn among the classes that have the probe's eye."""
    pool = _two_eye_pool(n_classes=4, per=1)
    eyes = script.image_eyes(pool)
    probe = pool.index[(pool.true_idx == 0) & (eyes == "right") & (pool.phase == "enrollment")][0]
    # class 0 keeps only this right image; classes 1 and 2 lose theirs, so only class 3 has a right eye
    drop = pool.index[(eyes == "right") & (pool.index != probe) & pool.true_idx.isin([0, 1, 2])]
    reduced = pool.drop(index=drop)
    sampled = reduced.loc[[probe]].reset_index().rename(columns={"index": "image_id"})
    for seed in range(5):
        capsys.readouterr()
        templates = script.choose_templates(sampled, reduced, seed=seed)
        genuine, impostor = templates["genuine"].iloc[0], templates["impostor"].iloc[0]
        assert not genuine.same_eye and eyes[genuine.template_image_id] == "left" and genuine.class_idx == 0
        assert impostor.same_eye and impostor.class_idx == 3 and eyes[impostor.template_image_id] == "right"
        out = capsys.readouterr().out
        assert out.count("WARNING") == 1 and "1 verification templates" in out and "genuine 1" in out
    # no other class has the probe's eye: the impostor falls back too
    alone = reduced.drop(index=reduced.index[(eyes[reduced.index] == "right") & (reduced.true_idx == 3)])
    templates = script.choose_templates(sampled, alone, seed=0)
    assert not templates["impostor"].same_eye.any() and not templates["genuine"].same_eye.any()
    assert "impostor 1" in capsys.readouterr().out


def test_verification_rows(fu_run, model, pair_inputs):
    """--verif-pairs: genuine and impostor rows on clean, noise and attack, with the cosine of the probe's
    embedding and the clean template's as v_full (efficiency holds as for every game)."""
    from src.eval.verification import embed
    rows, summary = fu_run["rows"], fu_run["summary"]
    assert set(rows.target_kind) >= {"true_class", "genuine", "impostor"}
    assert (rows.target_kind == rows.target.map({"true": "true_class", "pred": "adv_class", "genuine": "genuine",
                                                 "impostor": "impostor"})).all()
    verif = rows[rows.target.isin(["genuine", "impostor"])]
    assert set(verif.condition) == {"clean", "noise", "attack"} and (verif.score == "verif").all()
    # one row per image, variant, baseline and template
    per = verif.groupby(["target", "condition", "attack", "baseline"], dropna=False).image_id.nunique()
    assert (per == 4).all() and len(per) == 2 * 4 * 2
    genuine, impostor = verif[verif.target == "genuine"], verif[verif.target == "impostor"]
    assert (genuine.target_idx == genuine.true_idx).all() and (impostor.target_idx != impostor.true_idx).all()
    frame = pair_inputs["frame"]
    template_phase = genuine.template_relpath.str.split("/").str[1].str.split("_").str[0]
    assert (template_phase != genuine.phase).all()                       # the other phase (session)
    assert (frame.loc[genuine.template_image_id, "true_idx"].to_numpy() == genuine.true_idx.to_numpy()).all()
    phi = verif[[f"phi_{g}" for g in GROUPS]].to_numpy()
    np.testing.assert_allclose(phi.sum(axis=1), verif.v_full - verif.v_empty, atol=1e-9)
    # v_full of the clean rows: the cosine of the clean probe with the template
    clean = verif[(verif.condition == "clean") & (verif.baseline == "blur")]
    load = lambda paths: np.stack([np.asarray(Image.open(p), np.float32) / 255 for p in paths])  # noqa: E731
    probes = embed(model, load(clean.link_path))
    templates = embed(model, load(frame.loc[clean.template_image_id, "link_path"]))
    np.testing.assert_allclose(clean.v_full, (probes * templates).sum(axis=1), atol=1e-4)
    assert (genuine.v_full.groupby(genuine.condition).mean()["clean"]
            > impostor.v_full.groupby(impostor.condition).mean()["clean"])
    assert verif[[c for c in verif.columns if c.startswith("mass_")]].isna().all().all()
    assert summary["verification"]["enabled"] and summary["verification"]["other_phase"] == {"genuine": 4,
                                                                                            "impostor": 4}
    assert summary["verification"]["same_eye"] == {"genuine": 4, "impostor": 4}
    assert summary["rows_per_target_kind"] == {k: int(v) for k, v in rows.target_kind.value_counts().items()}
    assert summary["forward_passes_by_method"]["templates"] == 8
    assert summary["forward_passes_by_method"]["asa_verif"] == verif[verif.target == "genuine"].groupby(
        ["image_id", "condition", "attack"], dropna=False).n_forward.first().sum()


def test_sampling_baseline_columns(fu_run):
    """The part baselines play the same game as ASA: their values equal the estimators run on the row's own
    coalition values (looked up), with the same seeded coalitions; the pixel baselines are on clean and attack
    rows with the blur baseline only."""
    from src.xai.part_baselines import from_game, part_budget
    rows = fu_run["rows"]
    cls = rows[rows.target.isin(["true", "pred"])]
    values = cls[[f"value_{k}" for k in range(32)]].to_numpy()
    present = cls[[f"present_{g}" for g in GROUPS]].to_numpy(bool)
    for estimator, code in (("kernelshap", 1), ("lime", 2)):
        got = cls[[f"{estimator}_parts_{g}" for g in GROUPS]].to_numpy()
        want = np.stack([from_game(estimator, v, part_budget(p.sum(), 16), seed=[3, int(i), code], present=p)
                         for v, i, p in zip(values, cls.image_id, present)])
        np.testing.assert_allclose(got, want, atol=2e-3)
    ks = cls[[f"kernelshap_parts_{g}" for g in GROUPS]].to_numpy()
    # efficiency, with the estimator's own (float32) evaluation of the full and the empty coalition
    np.testing.assert_allclose(ks.sum(axis=1), cls.v_full - cls.v_empty, rtol=1e-5, atol=1e-5)
    pixel = cls[[f"kernelshap_pixels_{g}" for g in GROUPS] + [f"lime_pixels_{g}" for g in GROUPS]]
    has = pixel.notna().all(axis=1)
    assert has.any() and (pixel.isna().all(axis=1) | has).all()
    assert set(cls.condition[has]) == {"clean", "attack"} and set(cls.baseline[has]) == {"blur"}
    assert has[(cls.baseline == "blur") & cls.condition.isin(["clean", "attack"])].all()
    summary = fu_run["summary"]
    assert summary["part_baselines"]["samples"] == 16 and summary["pixel_baselines"]["n_patches"] == 64
    assert summary["part_baselines"]["budget_by_present_parts"] == {"1": 2, "2": 2, "3": 4, "4": 8, "5": 16}
    assert summary["pixel_baselines"]["hiding"] == ["blur"]


def test_runtime_and_js_columns(fu_run):
    from src.xai.part_baselines import pairwise_js, player_map_from_masks, player_value_map
    from src.xai.shapley import group_masks
    rows = fu_run["rows"]
    true = rows[rows.target == "true"]
    lid = true.present_lid.astype(bool)
    assert (true.nfwd_asa[lid] == 32).all() and (true.nfwd_asa[~lid] == 16).all()
    # half of the present parts' coalitions: 16 of 32 with five parts, 8 of 16 with four (not all 16 of them)
    assert (true.nfwd_kernelshap_parts[lid] == 16).all() and (true.nfwd_kernelshap_parts[~lid] == 8).all()
    assert (true.nfwd_lime_parts[lid] <= 16).all() and (true.nfwd_lime_parts[~lid] <= 8).all()
    pix = true.time_kernelshap_pixels.notna()
    assert (true.nfwd_kernelshap_pixels[pix] == 24).all() and (true.nfwd_lime_pixels[pix] <= 24).all()
    assert (true.nfwd_layercam == 1).all() and (true.nfwd_eigencam == 1).all()
    for method in ("asa", "kernelshap_parts", "lime_parts", "layercam", "eigencam"):
        assert (true[f"time_{method}"] > 0).all(), method
    assert (true.time_kernelshap_pixels[pix] > 0).all()
    js = [c for c in rows.columns if c.startswith("js_")]
    sources = ["asa", "kernelshap_parts", "lime_parts", "kernelshap_pixels", "lime_pixels", "layercam", "eigencam"]
    assert js == [f"js_{a}__{b}" for k, a in enumerate(sources) for b in sources[k + 1:]]
    finite = rows[js].stack().dropna()
    assert ((finite >= 0) & (finite <= 1)).all()
    assert true.js_asa__layercam.notna().mean() > 0.5
    assert rows[rows.target.isin(["genuine", "impostor"])][js].isna().all().all()
    # every game with v_full != v_empty has a positive oriented phi (efficiency), so its ASA map exists and
    # js_asa__layercam is finite wherever LayerCAM has mass, also where the attack won (v_full < v_empty)
    cls = rows[rows.target.isin(["true", "pred"])]
    cam_mass = cls.js_layercam__eigencam.notna() & (cls.v_full != cls.v_empty)
    assert cls.js_asa__layercam[cam_mass].notna().all()
    # recompute one divergence from the row's own part values, oriented as 23's faithfulness orients phi
    from src.xai.shapley import oriented_phi
    row = true[(true.condition == "clean") & (true.baseline == "mean")].iloc[0]
    masks = group_masks(eye_labels(5)[int(row.image_id) % 5], CLASSES, normalise_groups(DEFAULT_GROUPS))
    player_map = player_map_from_masks(masks)
    maps = {s: player_value_map(oriented_phi(row[[f"{p}_{g}" for g in GROUPS]].to_numpy(float), row.v_full,
                                             row.v_empty), player_map)
            for s, p in (("a", "phi"), ("b", "lime_parts"))}
    if np.isfinite(row.js_asa__lime_parts):
        assert row.js_asa__lime_parts == pytest.approx(pairwise_js(maps)[("a", "b")], abs=1e-9)
    hardware = fu_run["summary"]["runtime"]["hardware"]
    assert hardware["cpu_threads"] and "gpus" in hardware
    assert fu_run["summary"]["runtime"]["warmup_seconds"] > 0


def test_agreement_map_of_a_won_attack(script):
    """Where the attack won (v_full < v_empty, every phi <= 0) the ASA map uses phi oriented as 23's
    faithfulness orients it: it has mass, so its divergence from a CAM is finite rather than NaN."""
    from src.xai.part_baselines import js_divergence, player_value_map
    player_map = np.repeat(np.repeat(np.arange(4).reshape(2, 2), 8, axis=0), 8, axis=1)    # 4 quadrants
    phi = np.array([-3.0, -1.0, -0.5, 0.0])
    v_full, v_empty = -4.5, 0.0                                                            # efficiency holds
    cam_on = {p: (player_map == p).astype(float) + 0.01 for p in range(4)}
    assert player_value_map(phi, player_map).sum() == 0                     # the unoriented map: no mass
    assert np.isnan(js_divergence(player_value_map(phi, player_map), cam_on[0]))
    oriented = script.game_map(phi, v_full, v_empty, player_map)
    np.testing.assert_allclose(oriented, player_value_map(-phi, player_map))
    js = {p: js_divergence(oriented, cam) for p, cam in cam_on.items()}
    assert all(np.isfinite(v) for v in js.values())
    assert js[0] < js[1] < js[3]               # the most negative phi is the part that explains the gap most
    up = np.array([3.0, 1.0, 0.5, 0.0])                                     # v_full >= v_empty: unchanged
    np.testing.assert_array_equal(script.game_map(up, 4.5, 0.0, player_map), player_value_map(up, player_map))


def test_fu_analysis_end_to_end(fu_run, analyse):
    out = fu_run["dir"] / "analysis"
    checks = analyse.check_rows(fu_run["rows"], fu_run["summary"])
    assert any("verification rows" in c for c in checks) and any("KernelSHAP part values" in c for c in checks)
    analyse.main(["--rows", str(fu_run["dir"] / "rows_asa_fu.csv"), "--n-boot", "20", "--out-dir", str(out)])
    for name in ("sampling_error", "verification_scores", "verification_shift", "runtime", "agreement"):
        assert len(pd.read_csv(out / f"{name}.csv")), name
    error = pd.read_csv(out / "sampling_error.csv")
    assert set(error.estimator) == {"kernelshap_parts", "lime_parts", "kernelshap_pixels", "lime_pixels"}
    assert {"clean", "attack", "all"} <= set(error.condition)
    runtime = pd.read_csv(out / "runtime.csv")
    assert {"asa", "kernelshap_parts", "lime_parts", "kernelshap_pixels", "lime_pixels", "layercam", "eigencam",
            "asa_verif"} == set(runtime.method)
    assert (runtime.q25_s <= runtime.median_s).all() and (runtime.median_s <= runtime.q75_s).all()
    asa_blur = runtime[(runtime.method == "asa") & (runtime.baseline == "blur")].iloc[0]
    assert asa_blur.n == len(fu_run["rows"].query("target == 'true' and baseline == 'blur'"))
    scores = pd.read_csv(out / "verification_scores.csv")
    assert set(scores.target) == {"genuine", "impostor"}
    assert scores.loc[scores.condition == "attack", "shift"].notna().all()
    shift = pd.read_csv(out / "verification_shift.csv")
    assert set(shift.target) == {"genuine", "impostor"} and set(shift.condition) == {"attack"}
    text = (out / "report_asa.md").read_text(encoding="utf-8")
    for section in ("Sampling baselines against exact ASA", "Verification pairs", "measured on our hardware",
                    "Agreement between attributions", "untimed warm-up"):
        assert section in text, section
    # verification rows that break the accounting stop the analysis
    rows = fu_run["rows"]
    wrong = rows.copy()
    wrong.loc[wrong.target == "impostor", "target_idx"] = wrong.loc[wrong.target == "impostor", "true_idx"]
    with pytest.raises(ValueError, match="impostor rows another one"):
        analyse.check_rows(wrong)
    with pytest.raises(ValueError, match="verification variant"):
        analyse.check_rows(rows.drop(index=rows.index[rows.target == "genuine"][0]))
    with pytest.raises(ValueError, match="target kind"):
        analyse.check_rows(rows, {**fu_run["summary"], "rows_per_target_kind": {"true_class": 1}})


def test_resume_keeps_the_new_rows_and_columns(script, pair_inputs, tmp_path, monkeypatch):
    """A run with verification pairs and sampling baselines stopped after its first batch continues with --resume
    to the rows of an uninterrupted run (wall times aside) and the same forward-pass counts."""
    opts = ["--max-images", "4", "--levels", "2", "--attacks", "fgsm", "--defences", "jpeg", "--baselines", "mean",
            "--cams", "layercam", "--verif-pairs", "--pixel-samples", "8", "--batch-size", "2", "--seed", "4",
            "--tag", "fr"]
    script.main(common(pair_inputs, tmp_path / "whole") + opts)
    want = pd.read_csv(tmp_path / "whole" / "rows_asa_fr.csv")
    whole = json.loads((tmp_path / "whole" / "summary_asa_fr.json").read_text())

    class Stopped(RuntimeError):
        pass

    real, calls = script.run_batch, []

    def stop_at_second_batch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise Stopped()
        return real(*args, **kwargs)

    out = tmp_path / "resumed"
    monkeypatch.setattr(script, "run_batch", stop_at_second_batch)
    with pytest.raises(Stopped):
        script.main(common(pair_inputs, out) + opts + ["--resume"])
    monkeypatch.setattr(script, "run_batch", real)
    script.main(common(pair_inputs, out) + opts + ["--resume"])
    got = pd.read_csv(out / "rows_asa_fr.csv")
    assert {"genuine", "impostor"} <= set(got.target) and "js_asa__kernelshap_parts" in got
    pd.testing.assert_frame_equal(without_wall_times(got), without_wall_times(want))
    summary = json.loads((out / "summary_asa_fr.json").read_text())
    assert summary["resumed"] and summary["forward_passes_by_method"] == whole["forward_passes_by_method"]
    assert summary["rows_per_target_kind"] == whole["rows_per_target_kind"]


# ---------------------------------------------------------------------------
# aas_pgd (src/attacks/masking.py) as an attack of 22
# ---------------------------------------------------------------------------

AAS_OPTS = ["--max-images", "2", "--levels", "2", "--defences", "jpeg", "--steps", "2", "--baselines", "mean",
            "--cams", "none", "--part-baselines", "none", "--pixel-baselines", "none", "--seed", "3"]


def test_aas_pgd_equals_pgd_without_attention(script, analyse, inputs, tmp_path):
    """conftest's model has no attention pool, so aas_pgd is pgd: its attack and attack+defence rows equal
    pgd's in everything but the attack name and the wall times, and 23's accounting accepts them."""
    assert script.conditions_per_image(1, 1, ["pgd", "aas_pgd", "bpda_pgd"]) == 1 + 1 + 1 + 1 + 3 + 3
    out = tmp_path / "out"
    script.main(common(inputs, out) + AAS_OPTS + ["--attacks", "pgd,aas_pgd", "--taus", "1,4", "--tag", "aas"])
    rows = pd.read_csv(out / "rows_asa_aas.csv")
    summary = json.loads((out / "summary_asa_aas.json").read_text())
    assert summary["attacks"] == ["pgd", "aas_pgd"] and summary["aas_taus"] == [1.0, 4.0]
    assert summary["n_rows"] == len(rows) and rows.image_id.nunique() == 2
    keys = ["image_id", "condition", "defence", "baseline", "target"]
    drop = ["row_id", "attack"] + [c for c in rows.columns if c.startswith("time_")]

    def runs(name):
        part = rows[rows.attack == name].sort_values(keys).reset_index(drop=True)
        return part.drop(columns=drop)

    pgd_rows, aas_rows = runs("pgd"), runs("aas_pgd")
    assert len(aas_rows) and set(aas_rows.condition) == {"attack", "attack+defence"}
    assert aas_rows.adapted_to.isna().all()
    pd.testing.assert_frame_equal(aas_rows, pgd_rows)
    assert "aas_pgd@0.0078431+jpeg" in summary["attack_success_after_defence_rate"]
    analyse.check_rows(rows, summary)
    for bad in (["--attacks", "aas_pgd", "--taus", "0,2"], ["--attacks", "aas_pgd", "--taus", "2,2"],
                ["--attacks", "aas_pgd", "--taus", "x"]):
        with pytest.raises(SystemExit):
            script.main(common(inputs, tmp_path / "x") + AAS_OPTS + bad)
    assert not (tmp_path / "x").exists()


def test_aas_pgd_on_an_attention_model(script, analyse, parts_model, inputs, tmp_path):
    """On the parts model (an attention pool) aas_pgd runs through the temperature-scaled copies, stays
    within the budget in whole 8-bit levels and its rows pass 23's accounting."""
    ckpt = tmp_path / "parts.keras"
    parts_model.save(ckpt)
    out = tmp_path / "out"
    args = common(inputs, out)
    args[1] = str(ckpt)
    script.main(args + AAS_OPTS + ["--attacks", "aas_pgd", "--tag", "aas_parts"])
    rows = pd.read_csv(out / "rows_asa_aas_parts.csv")
    summary = json.loads((out / "summary_asa_aas_parts.json").read_text())
    assert summary["anatomy"]["attention"] and summary["aas_taus"] == [1.0, 2.0, 4.0, 8.0, 16.0]
    attacked = rows[rows.condition == "attack"]
    assert len(attacked) and (attacked.attack == "aas_pgd").all()
    assert (attacked.pert_linf <= 2 / 255 + 1e-6).all()
    analyse.check_rows(rows, summary)


def test_parts_model_with_verification_pairs(script, analyse, parts_model, pair_inputs, tmp_path):
    """The receipt explains the classification score: genuine / impostor rows of a parts model have none, and
    23's receipt check covers the true / pred rows only (it used to fail on the NaN receipts of verification rows)."""
    ckpt = tmp_path / "parts.keras"
    parts_model.save(ckpt)
    out = tmp_path / "out"
    args = common(pair_inputs, out)
    args[1] = str(ckpt)
    script.main(args + ["--max-images", "2", "--levels", "2", "--attacks", "fgsm", "--defences", "none",
                        "--baselines", "mean", "--cams", "none", "--part-baselines", "none", "--pixel-baselines",
                        "none", "--verif-pairs", "--tag", "parts_verif"])
    rows = pd.read_csv(out / "rows_asa_parts_verif.csv")
    summary = json.loads((out / "summary_asa_parts_verif.json").read_text())
    verif = rows.target.isin(["genuine", "impostor"])
    assert verif.any() and rows.receipt_sum[verif].isna().all() and rows.receipt_sum[~verif].notna().all()
    checks = analyse.check_rows(rows, summary)
    assert any("receipt sums to v_full" in c for c in checks)
    broken = rows.copy()
    broken.loc[verif.idxmax(), "receipt_sum"] = 0.0
    with pytest.raises(ValueError, match="classification score only"):
        analyse.check_rows(broken, summary)
