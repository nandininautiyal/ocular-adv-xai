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
                ["--score", "prob"], ["--workers", "0"], ["--attacks", "bpda_pgd", "--defences", "none"]):
        with pytest.raises(SystemExit):
            script.main(common(inputs, tmp_path / "x") + bad)
    assert not (tmp_path / "x").exists()


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
