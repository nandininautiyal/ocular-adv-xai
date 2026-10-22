import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from src.detect.consistency import (PercentileFusion, RobustZFusion, detection_features, rank_inconsistency,
                                   score_columns)
from src.detect.evaluate import attack_clustering, auroc, leave_one_attack_out, tpr_at_fpr


# --- scores -------------------------------------------------------------------

def test_rank_inconsistency():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert rank_inconsistency(a, a) == pytest.approx(0.0)
    assert rank_inconsistency(a, -a) == pytest.approx(2.0)
    assert rank_inconsistency(np.zeros((8, 8)), np.zeros((8, 8))) == 0.0
    assert rank_inconsistency(np.zeros((8, 8)), a) == 1.0


def test_detection_features_on_tiny_model(model, images):
    before = images.copy()
    feats = detection_features(model, images[:3])
    np.testing.assert_array_equal(images, before)
    expected = {"pred", "fs_l1_jpeg", "fs_l1_bit_depth", "pred_cos_jpeg", "pred_cos_bit_depth"}
    expected |= {f"xc_{m}_{x}_{t}" for m in ("spearman", "js") for x in ("layercam", "eigencam")
                 for t in ("jpeg", "bit_depth")}
    assert set(feats) == expected
    for name, values in feats.items():
        assert values.shape == (3,) and np.isfinite(values).all(), name
        if name != "pred":
            assert (values >= -1e-6).all(), name
    families = score_columns(list(feats))
    assert len(families["explanation"]) == 8 and len(families["feature_squeezing"]) == 2


def test_robust_z_fusion():
    fusion = RobustZFusion({"a": np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "b": np.array([10.0, 10.0, 10.0, 12.0, 8.0])})
    assert fusion.centre == {"a": 2.0, "b": 10.0}
    z = fusion({"a": np.array([2.0, 100.0]), "b": np.array([10.0, 10.0])}, ["a", "b"])
    assert z[0] == pytest.approx(0.0) and z[1] > 10          # no saturation above the calibration range
    assert np.isfinite(RobustZFusion({"c": np.zeros(5)}).z("c", [1.0])).all()


def test_percentile_fusion():
    fusion = PercentileFusion({"a": np.arange(100), "b": np.arange(100) * 10})
    out = fusion({"a": np.array([-1, 49, 1000]), "b": np.array([0, 0, 0])}, ["a", "b"])
    np.testing.assert_allclose(out, [0.01, 0.5, 1.0])


# --- metrics ------------------------------------------------------------------

def test_auroc():
    assert auroc([0, 1, 2], [3, 4]) == 1.0
    assert auroc([3, 4], [0, 1, 2]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert np.isnan(auroc([], [1]))


def test_tpr_at_fpr_uses_calibration_threshold():
    cal = np.arange(100)
    op = tpr_at_fpr(cal, negative=np.arange(100), positive=np.array([10, 96, 200]), fpr=0.05)
    assert op["threshold"] == 95
    assert op["tpr"] == pytest.approx(2 / 3) and op["fpr"] == pytest.approx(0.04)


def test_leave_one_attack_out_and_clustering():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, (60, 2))
    a = rng.normal([6, 0], 0.5, (30, 2))
    b = rng.normal([0, 6], 0.5, (30, 2))
    adv, attacks = np.vstack([a, b]), np.array(["a"] * 30 + ["b"] * 30)
    out = leave_one_attack_out(clean[:30], clean[30:], adv, attacks)
    assert set(out) == {"a", "b"} and all(0.0 <= v <= 1.0 for v in out.values())
    assert attack_clustering(adv, attacks) == pytest.approx(1.0)
    assert np.isnan(leave_one_attack_out(clean[:30], clean[30:], a, ["a"] * 30)["a"])


# --- end to end ---------------------------------------------------------------

def test_run_detection_end_to_end(tmp_path, model, images, labels):
    ckpt = tmp_path / "tiny.keras"
    model.save(ckpt)
    rows = []
    for copy in range(2):                       # two images per class: calibration + evaluation
        for i, (img, label) in enumerate(zip(images, labels)):
            path = tmp_path / f"{label}_{copy}_{i}_l_c.png"
            Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
            rows.append({"link_path": str(path), "true_idx": int(label), "eye": "left"})
    correct_csv = tmp_path / "correct_test_tiny.csv"
    pd.DataFrame(rows).to_csv(correct_csv, index=False)

    out_dir = tmp_path / "det"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "11_run_detection.py"),
           "--ckpt", str(ckpt), "--correct-csv", str(correct_csv),
           "--attacks", "fgsm,pgd,patch", "--epsilons", "0.03", "--batch-size", "3",
           "--out-dir", str(out_dir), "--tag", "tiny"]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]

    feats = pd.read_csv(out_dir / "features_tiny.csv")
    assert (feats.kind == "calibration").sum() == 5 and (feats.kind == "clean").sum() == 5
    assert set(feats.loc[feats.kind == "adversarial", "variant"]) == {"fgsm@0.03", "pgd@0.03", "patch"}
    calib = set(feats.loc[feats.kind == "calibration", "link_path"])
    assert calib.isdisjoint(feats.loc[feats.kind == "clean", "link_path"])
    assert {"fused_explanation", "fused_feature_squeezing", "fused_all"} <= set(feats.columns)

    table = pd.read_csv(out_dir / "detection_tiny.csv")
    assert {"group", "score", "auroc", "tpr_at_5fpr"} <= set(table.columns)

    summary = json.loads((out_dir / "summary_tiny.json").read_text())
    assert summary["n_calibration"] == 5 and summary["n_clean_eval"] == 5
    assert summary["attack_success_rate"]["pgd@0.03"] > 0
    assert set(summary["pooled_auroc"]) == {"fused_explanation", "fused_feature_squeezing", "fused_prediction",
                                            "fused_all", "fused_prediction_explanation"}
    assert "supervised_logistic" in summary["unseen_attack_auroc"]
