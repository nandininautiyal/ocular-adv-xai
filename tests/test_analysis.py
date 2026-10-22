import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from src.analysis.stats import (auroc_ci, cluster_bootstrap_ci, holm, paired_auroc_difference,
                                paired_subject_test)


def test_cluster_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 2)
    values = rng.normal(5.0, 1.0, 60)
    est, lo, hi = cluster_bootstrap_ci(values, groups, n_boot=500)
    assert lo <= est <= hi and lo < 5.0 < hi
    assert np.isnan(cluster_bootstrap_ci([np.nan], [0])[0])


def test_cluster_bootstrap_resamples_whole_subjects():
    # two subjects, each constant: every resample's median is 0, 1 or 0.5
    est, lo, hi = cluster_bootstrap_ci([0, 0, 1, 1], [0, 0, 1, 1], n_boot=300)
    assert est == 0.5 and lo >= 0.0 and hi <= 1.0


def test_paired_subject_test():
    groups = np.repeat(np.arange(20), 2)
    b = np.zeros(40)
    out = paired_subject_test(b + 1.0 + 0.01 * np.arange(40), b, groups)
    assert out["n_subjects"] == 20 and out["median_diff"] > 0.9
    assert out["rank_biserial"] == pytest.approx(1.0) and out["p"] < 0.001
    assert np.isnan(paired_subject_test(b, b, groups)["p"])          # no differences at all


def test_holm():
    adj = holm([0.01, 0.04, np.nan, 0.03])
    np.testing.assert_allclose(adj[[0, 1, 3]], [0.03, 0.06, 0.06])
    assert np.isnan(adj[2])


def test_auroc_ci_and_paired_difference():
    rng = np.random.default_rng(1)
    g_neg, g_pos = np.arange(40), np.arange(40)
    neg, pos = rng.normal(0, 1, 40), rng.normal(2, 1, 40)
    est, lo, hi = auroc_ci(neg, pos, g_neg, g_pos, n_boot=300)
    assert lo <= est <= hi and est > 0.8
    weak_neg, weak_pos = rng.normal(0, 1, 40), rng.normal(0.1, 1, 40)
    d = paired_auroc_difference(neg, pos, weak_neg, weak_pos, g_neg, g_pos, n_boot=300)
    assert d["diff"] > 0.2 and d["low"] > 0 and d["p"] < 0.05
    same = paired_auroc_difference(neg, pos, neg, pos, g_neg, g_pos, n_boot=100)
    assert same["diff"] == 0 and same["p"] == 1.0                    # ties count half to each side


def _experiment_rows():
    rng = np.random.default_rng(2)
    rows = []
    for image in range(24):           # 12 subjects: enough for Holm-corrected p < 0.05
        subject = image // 2
        base = {"image_id": image, "true_idx": subject}
        for xai, target in (("layercam", "true"), ("eigencam", "any")):
            rows.append({**base, "condition": "clean", "xai": xai, "target": target, "correct": True,
                         "spearman": 1.0})
            for eps in (0.001, 0.0078431):
                rows.append({**base, "condition": "noise", "epsilon": eps, "xai": xai, "target": target,
                             "correct": True, "spearman": 0.98})
                rows.append({**base, "condition": "attack", "attack": "pgd", "epsilon": eps, "xai": xai,
                             "target": target, "correct": False, "spearman": 0.4 + rng.normal(0, 0.05),
                             "mass_iris": 0.2, "dmass_iris": -0.1, "mass_pupil": 0.05, "dmass_pupil": 0.0,
                             "mass_sclera": 0.15, "dmass_sclera": 0.0, "mass_periocular": 0.6,
                             "dmass_periocular": 0.1})
                rows.append({**base, "condition": "attack+defence", "attack": "pgd", "epsilon": eps,
                             "defence": "jpeg", "xai": xai, "target": target, "correct": True,
                             "attack_success": True, "spearman": 0.6,
                             "vs_defended_clean_spearman": 0.8 + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def _detection_features():
    rng = np.random.default_rng(3)
    rows = []
    for subject in range(15):
        rows.append({"true_idx": subject, "kind": "calibration"})
        rows.append({"true_idx": subject, "kind": "clean", "fused_explanation": rng.uniform(0, 0.6),
                     "fused_feature_squeezing": rng.uniform(0, 1), "fused_all": rng.uniform(0, 0.7),
                     "xc_spearman_layercam_jpeg": rng.uniform(0, 0.3), "fs_l1_jpeg": rng.uniform(0, 1)})
        for attack in ("fgsm", "patch"):
            rows.append({"true_idx": subject, "kind": "adversarial", "attack": attack,
                         "variant": f"{attack}@0.001" if attack == "fgsm" else attack, "success": True,
                         "fused_explanation": rng.uniform(0.5, 1), "fused_feature_squeezing": rng.uniform(0, 1),
                         "fused_all": rng.uniform(0.5, 1), "xc_spearman_layercam_jpeg": rng.uniform(0.2, 1),
                         "fs_l1_jpeg": rng.uniform(0, 1)})
    return pd.DataFrame(rows)


def test_analyse_end_to_end(tmp_path):
    rows, feats = tmp_path / "rows.csv", tmp_path / "features.csv"
    _experiment_rows().to_csv(rows, index=False)
    _detection_features().to_csv(feats, index=False)
    out = tmp_path / "analysis"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "12_analyse.py"), "--experiment-rows", str(rows),
           "--detection", str(feats), "--n-boot", "100", "--out-dir", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]

    shift = pd.read_csv(out / "heatmap_shift.csv")
    assert len(shift) == 4 and (shift.spearman_median < shift.noise_spearman_median).all()
    assert (shift.p_holm < 0.05).all()
    restored = pd.read_csv(out / "restoration.csv")
    assert (restored.median_gain > 0).all() and (restored.prediction_restored_rate == 1.0).all()
    regions = pd.read_csv(out / "region_shift.csv")
    assert set(regions.region) == {"pupil", "iris", "sclera", "periocular"}
    aurocs = pd.read_csv(out / "detection_auroc.csv")
    assert {"all", "fgsm", "patch"} <= set(aurocs.group)
    versus = pd.read_csv(out / "detection_vs_fs.csv")
    assert set(versus.ours) == {"fused_explanation", "fused_all"}
    for fig in ("fig_heatmap_shift.png", "fig_detection.png", "fig_regions.png"):
        assert (out / fig).stat().st_size > 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Heatmap shift" in report and "Feature Squeezing" in report
