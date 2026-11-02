"""End-to-end: run scripts/05_run_experiment.py the way the docs say, on the tiny model."""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from src.compare.divergence import compare_heatmaps
from src.defences.inference import jpeg_compress
from src.compare.regions import (grids_from_labels, heuristic_masks, label_masks, normalised_entropy,
                                 occlusion_grids, region_mass_named)
from src.models.arcface_inference import predict_logits
from src.xai.attribution import layercam

SCRIPT = os.path.join(REPO, "scripts", "05_run_experiment.py")
HEURISTIC = ("pupil", "iris", "sclera", "periocular")
EYE_CLASSES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")
CONDITIONS = {"clean", "noise", "defence_on_clean", "noise+defence", "attack", "attack+defence"}
DEFENDED = ["attack+defence", "noise+defence"]
VALUES = 256 * 256 * 3


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, model, images, labels):
    """Checkpoint, PNGs and a correct-image CSV with manifest relpaths, as 04_eval_clean.py writes it."""
    root = tmp_path_factory.mktemp("run_experiment")
    ckpt = root / "tiny.keras"
    model.save(ckpt)
    rows = []
    for i, (img, label) in enumerate(zip(images, labels)):
        path = root / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        eye = "left" if i % 2 else "right"
        rows.append({"link_path": str(path),
                     "relpath": f"test_office/enrollment_{eye}_eye/{1000 + label}/{path.name}",
                     "true_idx": int(label), "true_subject": 1000 + int(label), "eye": eye, "device": "test",
                     "lighting": "office", "phase": "enrollment"})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "correct_test_tiny.csv", index=False)
    clean = np.stack([np.asarray(Image.open(p), dtype=np.float64) / 255 for p in frame.link_path])
    return {"root": root, "ckpt": ckpt, "csv": root / "correct_test_tiny.csv", "frame": frame, "clean": clean}


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(inputs, out_dir, *args):
    cmd = [sys.executable, SCRIPT, "--ckpt", str(inputs["ckpt"]), "--correct-csv", str(inputs["csv"]),
           "--per-class", "1", "--batch-size", "3", "--out-dir", str(out_dir), *args]
    return subprocess.run(cmd, cwd=str(inputs["root"]), capture_output=True, text=True, timeout=900)


def eye_labels(n):
    """Label maps for conftest.make_images: the coloured disk is the iris, a ring around it the
    sclera, the top 16 rows the brow, the bottom 16 rows unknown (255); no lid or caruncle."""
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


def save_masks(path, relpaths, labels, class_names=EYE_CLASSES):
    """An eye-mask file in the shared format (masks_<tag>.npz)."""
    np.savez_compressed(path, relpaths=np.array(list(relpaths), dtype=str), labels=labels,
                        class_names=np.array(class_names, dtype=str),
                        meta=np.array(json.dumps({"model": "synthetic"})))


def same_eps(column, text):
    """Rows whose epsilon is the one written in an examples key ('' or 'None': no epsilon)."""
    if text in ("", "None"):
        return column.isna()
    return np.isclose(column.fillna(-1.0), float(text), rtol=0, atol=1e-9)


def test_run_experiment_end_to_end(inputs, model):
    out_dir = inputs["root"] / "out"
    result = run(inputs, out_dir, "--attacks", "fgsm,pgd,cw,patch,monocle", "--epsilons", "0.0039216,0.0313725",
                 "--save-examples", "1", "--tag", "tiny", "--concept-masks", "--save-maps")
    assert result.returncode == 0, result.stderr[-3000:]

    table = pd.read_csv(out_dir / "rows_tiny.csv")
    frame, clean_px = inputs["frame"], inputs["clean"]
    assert table.image_id.nunique() == 5
    assert set(table.condition) == CONDITIONS
    assert set(table.xai) == {"layercam", "eigencam"}
    assert table.row_id.tolist() == list(range(len(table)))
    assert (table.relpath.to_numpy() == frame.relpath[table.image_id].to_numpy()).all()
    images = pd.read_csv(out_dir / "images_tiny.csv")
    assert (images.relpath.to_numpy() == frame.relpath[images.image_id].to_numpy()).all()
    clean = table[table.condition == "clean"]
    assert clean.correct.all() and (clean.spearman.round(6) == 1.0).all()

    # Defended maps (attacks and the noise control) are also compared with the defended clean map,
    # for true-class maps only (before this run only attack+defence rows had these columns).
    defended = table[table.condition.isin(DEFENDED)]
    own_class = defended.target != "pred"
    assert defended[own_class].vs_defended_clean_spearman.notna().all()
    assert defended[~own_class].vs_defended_clean_spearman.isna().all()
    assert table[~table.condition.isin(DEFENDED)].vs_defended_clean_spearman.isna().all()
    control = table[table.condition == "noise+defence"]
    assert (control.target != "pred").sum() == 5 * 2 * 2 * 2          # images x epsilons x defences x methods
    assert set(control.defence) == {"jpeg", "bit_depth"} and control.attack.isna().all()

    # Regions (heuristic masks partition the image): masses and areas sum to 1, enrichment = mass / area
    assert np.allclose(table[[f"mass_{r}" for r in HEURISTIC]].sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(table[[f"area_{r}" for r in HEURISTIC]].sum(axis=1), 1.0, atol=1e-5)
    assert (clean[[f"dmass_{r}" for r in HEURISTIC]].abs() < 1e-9).all().all()
    for r in HEURISTIC:
        np.testing.assert_allclose(table[f"enr_{r}"], table[f"mass_{r}"] / table[f"area_{r}"], rtol=1e-9)

    # Perturbation size is image space; "l2" stays the heatmap distance
    assert (table.pert_linf == table.linf).all()
    linf = table[(table.condition == "attack") & (table.attack == "fgsm")].groupby("epsilon").linf.max()
    assert (linf <= linf.index + 1e-6).all()
    noise = table[table.condition == "noise"]
    n_levels = np.round(noise.epsilon * 255)
    assert set(n_levels) == {1, 8}
    # no clean value is within 8 levels of 0 or 1, so the sign noise moves every value by the full budget
    np.testing.assert_allclose(noise.pert_l2, n_levels / 255 * np.sqrt(VALUES), rtol=1e-5)
    patch = table[(table.condition == "attack") & (table.attack == "patch")]
    expected = [np.sqrt((clean_px[i, 96:160, 96:160] ** 2).sum()) for i in patch.image_id]
    np.testing.assert_allclose(patch.pert_l2, expected, rtol=1e-5)
    assert (table.l2 <= 1.0).all() and (patch.pert_l2 > 10).all()

    # Every scored image of the first image is in the examples file (keyed by the position in
    # images_tiny.csv); stored as uint8, which is exact because all of them are on the 8-bit grid
    examples = np.load(out_dir / "examples_tiny.npz")
    first = images.image_id[0]
    reference = examples["0|clean||||layercam|true|image"].astype(np.float64) / 255
    np.testing.assert_array_equal(reference, clean_px[first])
    checked = set()
    for key in (k[:-len("|image")] for k in examples.files if k.endswith("|image")):
        _, condition, attack, eps, defence, xai, target = key.split("|")
        row = table[(table.image_id == first) & (table.condition == condition) & (table.xai == xai)
                    & (table.target == target) & (table.attack.fillna("") == attack)
                    & (table.defence.fillna("") == defence) & same_eps(table.epsilon, eps)]
        assert len(row) == 1, key
        diff = examples[f"{key}|image"].astype(np.float64) / 255 - reference
        assert row.pert_l2.iloc[0] == pytest.approx(np.sqrt((diff ** 2).sum()), rel=1e-5, abs=1e-6), key
        assert row.pert_linf.iloc[0] == pytest.approx(np.abs(diff).max(), abs=1e-6), key
        checked.add(condition)
    assert checked == CONDITIONS
    # CW: the scored 8-bit image is misclassified by more than kappa = 5 logits
    cw_image = examples["0|attack|cw|None||layercam|true|image"].astype(np.float32) / 255
    logits = predict_logits(model, cw_image)[0]
    label = frame.true_idx[first]
    assert np.delete(logits, label).max() - logits[label] > 5.0 - 1e-3

    # Entropy and raw magnitudes; the _clean columns are those of the clean row of the same map
    assert table.entropy.between(0, 1).all() and (table.raw_max >= 0).all()
    ref = clean.set_index(["image_id", "xai"])
    for col in ("entropy", "raw_sum", "raw_max"):
        expected = ref.loc[list(zip(table.image_id, table.xai)), col].to_numpy()
        np.testing.assert_allclose(table[f"{col}_clean"], expected, rtol=1e-12)

    # Occluders: share of the image changed and masses over the pixels left alone
    occluder = table.attack.isin(["patch", "monocle"])
    assert table[~occluder].occluded_frac.isna().all() and table[~occluder].massvis_iris.isna().all()
    assert np.allclose(table[table.attack == "patch"].occluded_frac, 64 * 64 / 256 ** 2)
    assert table[table.attack == "monocle"].occluded_frac.between(0.01, 0.5).all()
    vis = table[occluder][[f"massvis_{r}" for r in HEURISTIC]]
    assert np.allclose(vis.sum(axis=1), 1.0, atol=1e-5)          # over the regions left visible
    dvis = table[occluder][[f"dmassvis_{r}" for r in HEURISTIC]]
    assert (vis.isna().to_numpy() == dvis.isna().to_numpy()).all()
    # A region with no pixel left visible (hidden, or empty in the clean image) is not measured:
    # NaN, never a 0 that would read as "no shift". Exact for the patch, whose box is fixed.
    clean32 = np.stack([np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in frame.link_path])
    box = np.zeros((256, 256), bool)
    box[96:160, 96:160] = True
    hidden = ~(heuristic_masks(clean32) & ~box).any(axis=(-2, -1))       # (image, region)
    assert hidden.any() and not hidden.all(axis=0).any()
    for r in table[table.attack == "patch"].itertuples():
        assert [bool(np.isnan(getattr(r, f"massvis_{x}"))) for x in HEURISTIC] == hidden[r.image_id].tolist()

    # Monocle placement, one per image, also on the defended rows
    monocle = table[table.attack == "monocle"]
    assert set(monocle.monocle_placement) <= {"haar", "iris", "centre_fallback"}
    assert table[table.attack != "monocle"].monocle_placement.isna().all()
    placements = monocle[monocle.condition == "attack"].drop_duplicates("image_id").monocle_placement

    # Maps: every row's native map, float32, keyed by row_id; the stored maps reproduce the metrics
    stored = np.load(out_dir / "maps_tiny.npz")
    assert set(stored.files) == {str(i) for i in table.row_id}
    maps = {int(k): stored[k] for k in stored.files}
    assert all(m.dtype == np.float32 and m.shape == (16, 16) for m in maps.values())
    clean_id = clean.set_index(["image_id", "xai"]).row_id
    for r in table[table.image_id == first].itertuples():
        metrics = compare_heatmaps(maps[clean_id[(r.image_id, r.xai)]], maps[r.row_id])
        assert r.spearman == pytest.approx(metrics["spearman"], abs=1e-9)
        assert r.entropy == pytest.approx(normalised_entropy(maps[r.row_id]), abs=1e-12)
    defended_clean = table[table.condition == "defence_on_clean"]
    reference_id = defended_clean.set_index(["image_id", "xai", "defence", "target"]).row_id
    for r in control[control.target != "pred"].itertuples():
        partner = maps[reference_id[(r.image_id, r.xai, r.defence, r.target)]]
        expected = compare_heatmaps(partner, maps[r.row_id])["spearman"]
        assert r.vs_defended_clean_spearman == pytest.approx(expected, abs=1e-9)

    summary = json.loads((out_dir / "summary_tiny.json").read_text())
    assert summary["n_images"] == 5 and summary["concept_masks"] is True and summary["n_rows"] == len(table)
    assert summary["attack_success_rate"]["pgd@0.0313725"]["rate"] > 0
    assert summary["attack_success_rate"]["cw"]["rate"] > 0
    assert not any(k.startswith("cw@") for k in summary["attack_success_rate"])   # CW runs once
    assert summary["restoration_rate_among_successful_attacks"]
    assert "noise@0.0313725" in summary["noise_control_flip_rate"]
    assert set(summary["defence_on_clean_flip_rate"]) == {"clean+jpeg", "clean+bit_depth"}
    assert set(summary["noise_defence_flip_rate"]) == {f"noise@{e}+{d}" for e in ("0.0039216", "0.0313725")
                                                       for d in ("jpeg", "bit_depth")}
    assert summary["epsilons"] == [0.0039216, 0.0313725] and summary["levels"] == [1, 8]
    assert summary["cw"] == {"kappa": 5.0, "steps": 50, "binary_steps": 4, "success_after_8bit_rounding": True}
    assert summary["attack_settings"]["cw"]["quantise_fn"] == "src.attacks.budget.quantise"
    assert summary["defence_settings"] == {"jpeg": {"quality": 75}, "bit_depth": {"bits": 4}}
    assert summary["mask_source"] == "heuristic" and summary["regions"] == list(HEURISTIC)
    assert summary["mask_file_sha256"] is None and summary["mask_meta"] is None
    counts = summary["monocle_placement_counts"]
    assert set(counts) == {"haar", "iris", "centre_fallback"} and sum(counts.values()) == 5
    assert counts == {p: int((placements == p).sum()) for p in counts}
    assert summary["maps_file"] == "maps_tiny.npz" and summary["xai_target_layer"] == "last_act"
    assert summary["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", summary["git_commit"])
    assert {"python", "numpy", "tensorflow", "tf_keras"} <= set(summary["versions"])
    assert summary["perturbation_size"]["patch"]["pert_l2_median"] > 10
    assert (out_dir / "summary_tiny.csv").exists()


def test_run_experiment_with_an_eye_mask_file(inputs):
    frame, root = inputs["frame"], inputs["root"]
    labels = eye_labels(len(frame))
    masks = root / "masks_tiny.npz"
    # a mask file that lacks a sampled image stops the run before the model is loaded, naming the image
    save_masks(masks, frame.relpath[1:], labels[1:])
    result = run(inputs, root / "missing", "--mask-source", str(masks))
    assert result.returncode != 0 and frame.relpath[0] in result.stderr
    assert not (root / "missing").exists()

    save_masks(masks, frame.relpath[::-1], labels[::-1])        # any order: masks are looked up by relpath
    out_dir = root / "eye"
    result = run(inputs, out_dir, "--attacks", "patch,monocle", "--levels", "1", "--defences", "jpeg",
                 "--jpeg-quality", "50", "--xai", "layercam", "--save-examples", "1", "--save-maps",
                 "--mask-source", str(masks), "--tag", "eye")
    assert result.returncode == 0, result.stderr[-3000:]

    # --jpeg-quality reaches the defended images; rows keep the name "jpeg"
    examples = np.load(out_dir / "examples_eye.npz")
    first = inputs["clean"][pd.read_csv(out_dir / "images_eye.csv").image_id[0]].astype(np.float32)
    defended = examples["0|defence_on_clean|||jpeg|layercam|true|image"]
    as_uint8 = lambda x: (x * 255).round().astype(np.uint8)  # noqa: E731
    np.testing.assert_array_equal(defended, as_uint8(jpeg_compress(first, quality=50)))
    assert not np.array_equal(defended, as_uint8(jpeg_compress(first, quality=75)))

    table = pd.read_csv(out_dir / "rows_eye.csv")
    assert set(table.condition) == CONDITIONS
    assert np.allclose(table[table.condition == "noise"].epsilon, 0.0039216)         # --levels 1
    # regions are the file's classes (the file turns on --concept-masks)
    assert {f"{p}_{c}" for p in ("mass", "dmass", "area", "enr") for c in EYE_CLASSES} <= set(table.columns)
    assert "mass_pupil" not in table and "mass_periocular" not in table
    stored = np.load(out_dir / "maps_eye.npz")
    clean_id = table[table.condition == "clean"].set_index("image_id").row_id
    patch_mask = np.zeros((256, 256), bool)
    patch_mask[96:160, 96:160] = True
    for r in table.itertuples():
        lab = labels[r.image_id]
        assert r.area_iris == pytest.approx((lab == 1).mean(), abs=1e-6)
        assert r.area_brow == pytest.approx(16 / 256, abs=1e-6) and r.area_lid == 0 and np.isnan(r.enr_lid)
        assert sum(getattr(r, f"area_{c}") for c in EYE_CLASSES) == pytest.approx(1 - 16 / 256, abs=1e-6)
        grids = grids_from_labels(lab[np.newaxis], EYE_CLASSES, 16)[0][0]
        heatmap = stored[str(r.row_id)]
        for name, value in region_mass_named(heatmap, grids, EYE_CLASSES).items():
            assert getattr(r, name) == pytest.approx(value, abs=1e-9)
        if r.attack == "patch":
            assert r.occluded_frac == pytest.approx(64 * 64 / 256 ** 2)
            vis_grids, visible = occlusion_grids(label_masks(lab[np.newaxis], EYE_CLASSES)[0], patch_mask, 16)
            vis = region_mass_named(heatmap, vis_grids, EYE_CLASSES, visible=visible, prefix="massvis")
            before = region_mass_named(stored[str(clean_id[r.image_id])], vis_grids, EYE_CLASSES,
                                       visible=visible, prefix="massvis")
            for name, value in vis.items():
                assert getattr(r, name) == pytest.approx(value, abs=1e-9, nan_ok=True)
                assert getattr(r, f"d{name}") == pytest.approx(value - before[name], abs=1e-9, nan_ok=True)
            # lid and caruncle have no pixel in these label maps, so nothing of them is visible
            assert {k for k, v in vis.items() if np.isnan(v)} == {"massvis_lid", "massvis_caruncle"}
    # Haar finds no eye in these images, so every monocle is centred on the mask's iris
    assert (table[table.attack == "monocle"].monocle_placement == "iris").all()

    summary = json.loads((out_dir / "summary_eye.json").read_text())
    assert summary["mask_source"] == str(masks) and summary["mask_meta"] == {"model": "synthetic"}
    assert summary["mask_file_sha256"] == hashlib.sha256(masks.read_bytes()).hexdigest()
    assert summary["regions"] == list(EYE_CLASSES) and summary["concept_masks"] is True
    assert summary["monocle_placement_counts"] == {"haar": 0, "iris": 5, "centre_fallback": 0}
    assert summary["levels"] == [1] and summary["epsilons"] == [0.0039216]
    assert summary["defence_settings"] == {"jpeg": {"quality": 50}} and set(table.defence.dropna()) == {"jpeg"}


def test_run_experiment_stops_on_a_budget_that_is_not_whole_levels(inputs):
    out_dir = inputs["root"] / "bad_budget"
    result = run(inputs, out_dir, "--epsilons", "0.0039216,0.0078")
    assert result.returncode == 2
    assert "0.0078" in result.stderr and "levels" in result.stderr
    assert not out_dir.exists()


def test_budgets_must_be_whole_8bit_levels(runner):
    parse, ap = runner.parse_budgets, argparse.ArgumentParser()
    defaults = [0.0039216, 0.0078431, 0.0156863]
    assert parse(ap, runner.DEFAULT_EPSILONS, None, True) == defaults
    assert parse(ap, runner.DEFAULT_EPSILONS, "1,2,4", True) == defaults       # same budgets, same row keys
    assert parse(ap, "0.0078431", None, True) == [0.0078431]
    assert parse(ap, "0.001,0.0019608", None, False) == [0.001, 0.0019608]    # float budgets without rounding
    for epsilons, levels in [("0.0078", None), ("0.001", None), ("0.03", None), ("0", None), ("x", None),
                             (runner.DEFAULT_EPSILONS, "1.5"), (runner.DEFAULT_EPSILONS, "0"),
                             (runner.DEFAULT_EPSILONS, "256"), (runner.DEFAULT_EPSILONS, " ")]:
        with pytest.raises(SystemExit):
            parse(ap, epsilons, levels, True)
    with pytest.raises(SystemExit):
        parse(ap, "-0.0039216", None, False)


def test_regions_follow_each_methods_map_size(runner, model, images, labels, monkeypatch):
    # A method with 64 x 64 maps next to LayerCAM's 16 x 16 ones: each gets region grids of its own
    # size (before, the grids had the first method's size and the second method raised an error).
    first = next(layer.name for layer in model.layers
                 if type(layer).__name__ == "Activation" and layer.name != "last_act")
    monkeypatch.setitem(runner.XAI_REGISTRY, "layercam_fine", functools.partial(layercam, target_layer=first))
    clean = runner.quantise(images)
    ev = runner.Evaluator(model, ["layercam", "layercam_fine"], clean, labels,
                          [{"image_id": i} for i in range(len(clean))],
                          regions=(runner.REGIONS, runner.heuristic_masks(clean)), first_row_id=7, keep_maps=True)
    ev(clean, {"condition": "clean"})
    patched = runner.patch_occlusion(clean, **runner.PATCH)
    ev(patched, {"condition": "attack", "attack": "patch", "epsilon": None},
       occluded=runner.occlusion_mask(clean, patched))
    assert set(ev.grids) == {(16, 16), (64, 64)}
    rows = pd.DataFrame(ev.rows)
    assert rows.row_id.tolist() == list(range(7, 7 + len(rows)))
    assert set(ev.maps) == {str(i) for i in rows.row_id}
    assert {m.shape for m in ev.maps.values()} == {(16, 16), (64, 64)}
    assert np.allclose(rows[[f"mass_{r}" for r in HEURISTIC]].sum(axis=1), 1.0, atol=1e-5)
    spread = rows.groupby("image_id")[[f"area_{r}" for r in HEURISTIC]].agg(lambda v: v.max() - v.min())
    assert (spread < 1e-6).all().all()                     # areas are pixel shares on either grid
    patch = rows[rows.attack == "patch"]
    assert set(patch.xai) == {"layercam", "layercam_fine"} and np.allclose(patch.occluded_frac, 1 / 16)
    assert np.allclose(patch[[f"massvis_{r}" for r in HEURISTIC]].sum(axis=1), 1.0, atol=1e-5)


def test_target_layer_reaches_every_xai_method(runner, model, images, labels):
    # --target-layer is passed to each method (and to the clean maps), not only validated
    first = next(layer.name for layer in model.layers
                 if type(layer).__name__ == "Activation" and layer.name != "last_act")
    for xai in ("layercam", "eigencam"):
        maps, raw_sum, raw_max = runner.heatmaps(model, xai, images, labels, target_layer=first)
        assert maps.shape == (5, 64, 64) and raw_sum.shape == raw_max.shape == (5,) and np.isfinite(raw_max).all()
        assert runner.heatmaps(model, xai, images, labels)[0].shape == (5, 16, 16)       # default layer
    ev = runner.Evaluator(model, ["layercam", "eigencam"], images, labels, [{"image_id": i} for i in range(5)],
                          target_layer=first)
    ev(images, {"condition": "clean"})
    assert {m.shape for m in ev.clean_maps.values()} == {(5, 64, 64)}
    assert (pd.DataFrame(ev.rows).spearman.round(6) == 1.0).all()


def test_eye_mask_file_is_checked(runner, tmp_path):
    labels = eye_labels(2)
    path = tmp_path / "masks_two.npz"
    save_masks(path, ["a\\b.png", "c/d.png"], labels)                  # relpaths get "/" separators
    out, names, meta = runner.load_eye_masks(str(path), ["c/d.png", "a/b.png", "c/d.png"])
    assert names == EYE_CLASSES and meta == {"model": "synthetic"} and out.dtype == np.uint8
    np.testing.assert_array_equal(out, labels[[1, 0, 1]])
    with pytest.raises(SystemExit, match="e/f.png"):
        runner.load_eye_masks(str(path), ["a/b.png", "e/f.png"])
    bad = labels.copy()
    bad[0, 0, 0] = 9                                                    # no class 9; 255 would be fine
    save_masks(path, ["a/b.png", "c/d.png"], bad)
    with pytest.raises(SystemExit, match="class name"):
        runner.load_eye_masks(str(path), ["a/b.png"])
    np.savez(tmp_path / "not_masks.npz", labels=labels)
    with pytest.raises(SystemExit, match="not an eye-mask file"):
        runner.load_eye_masks(str(tmp_path / "not_masks.npz"), ["a/b.png"])
