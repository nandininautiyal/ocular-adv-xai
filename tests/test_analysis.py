import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from src.analysis.region_null import HEURISTIC_REGIONS
from src.analysis.stats import (auroc_ci, cluster_bootstrap_ci, cluster_bootstrap_median, cluster_permutation_test,
                                finite_median, format_p, holm, paired_auroc_difference, paired_subject_ci,
                                paired_subject_test, parse_bool, stratified_difference)


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


def test_paired_subject_test_matches_scipy_on_small_tied_samples():
    """Few subjects with tied differences: the vectorised sign-flip p equals scipy's own."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(8)
    for n in (2, 4, 6, 9):
        d = rng.choice([0.1, 0.2, 0.5], n) * rng.choice([-1, 1], n)
        assert paired_subject_test(d, np.zeros(n), np.arange(n))["p"] == wilcoxon(d).pvalue


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
           "--detection", str(feats), "--n-boot", "100", "--n-perm", "99", "--out-dir", str(out)]
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
    assert set(versus.ours) == {"fused_explanation", "fused_all"}      # the third pair needs logit columns
    for fig in ("fig_heatmap_shift.png", "fig_detection.png", "fig_regions.png"):
        assert (out / fig).stat().st_size > 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Heatmap shift" in report and "fused_feature_squeezing" in report
    region_lines = [line for line in report.splitlines() if line.startswith("- Region shift")]
    assert region_lines and all(" @ " in line and "/255" in line for line in region_lines)   # budget printed

    for name in ("region_shift_v2", "region_null", "heatmap_shift_by_outcome", "restoration_by_outcome"):
        assert len(pd.read_csv(out / f"{name}.csv"))
    v2 = (out / "report_v2.md").read_text(encoding="utf-8")
    assert "## What we can and cannot claim" in v2 and "needs the new runner" in v2
    assert "Only the periocular region is analysed" in v2 and "@ 2/255" in v2


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def test_parse_bool():
    values = [True, False, "True", "False", " true ", "FALSE", 1, 0, 1.0, 0.0, "1", "0", np.True_, np.int64(0)]
    assert parse_bool(np.array(values, dtype=object)).tolist() == [True, False] * 7
    assert parse_bool(pd.Series(["True", "False"])).tolist() == [True, False]
    assert pd.Series(["False"]).astype(bool).iloc[0]                 # the trap: astype(bool) says True
    for bad in (np.nan, None, "yes", 2, ""):
        with pytest.raises(ValueError, match="flag"):
            parse_bool(np.array([True, bad], dtype=object), "flag")


def test_format_p_and_finite_median():
    assert format_p(6.08e-51) == "< 1e-10" and format_p(0.0123) == "0.012" and format_p(np.nan) == "n/a"
    assert finite_median([np.nan, 1.0, 3.0, np.inf]) == 2.0 and np.isnan(finite_median([np.nan]))


def test_cluster_bootstrap_median_matches_cluster_bootstrap_ci():
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 25, 90)                                 # unequal cluster sizes
    values = rng.normal(size=90)
    values[::17] = np.nan
    for n_boot, seed in ((300, 0), (257, 3)):
        assert (cluster_bootstrap_median(values, groups, n_boot=n_boot, seed=seed)
                == cluster_bootstrap_ci(values, groups, np.median, n_boot=n_boot, seed=seed))


def test_paired_subject_ci_describes_what_paired_subject_test_tests():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 2)
    a, b = rng.normal(0.5, 1, 60), rng.normal(0, 1, 60)
    per_subject = (a - b).reshape(30, 2).mean(1)
    est, lo, hi = paired_subject_ci(a, b, groups, n_boot=500)
    assert est == pytest.approx(per_subject.mean()) and lo < est < hi
    assert paired_subject_test(a, b, groups)["median_diff"] == pytest.approx(np.median(per_subject))
    assert paired_subject_ci(a, 0.0, groups, n_boot=50)[0] == pytest.approx(a.reshape(30, 2).mean(1).mean())
    a[0] = np.nan                                                    # an unpaired row is dropped
    expected = per_subject.copy()
    expected[0] = a[1] - b[1]
    assert paired_subject_ci(a, b, groups, n_boot=50)[0] == pytest.approx(expected.mean())


def test_cluster_permutation_test_finds_a_shift_and_keeps_its_level():
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(40), 2)
    labels = rng.random(80) < 0.5
    values = rng.normal(size=80) + 1.5 * labels
    out = cluster_permutation_test(values, labels, groups, n_perm=999)
    assert out["diff"] > 1 and out["p"] < 0.01
    assert np.isnan(cluster_permutation_test(values, np.ones(80, bool), groups)["p"])
    # Labels and values both clustered by subject, no real effect: exchanging subjects keeps the 5 % level;
    # shuffling images ignores the clustering and rejects far too often.
    subject_level = image_level = n = 0
    for s in range(120):
        r = np.random.default_rng(100 + s)
        g, lab = np.repeat(np.arange(20), 2), np.repeat(r.random(20) < 0.5, 2)
        if lab.all() or not lab.any():
            continue
        v = np.repeat(r.normal(size=20), 2) + 0.1 * r.normal(size=40)
        n += 1
        subject_level += cluster_permutation_test(v, lab, g, n_perm=199, seed=s)["p"] < 0.05
        observed = abs(np.median(v[lab]) - np.median(v[~lab]))
        null = [abs(np.median(v[q]) - np.median(v[~q])) for q in (r.permutation(lab) for _ in range(199))]
        image_level += (1 + np.sum(np.array(null) >= observed - 1e-12)) / 200 < 0.05
    assert subject_level / n <= 0.10 < image_level / n


def test_stratified_difference_removes_the_mix_of_strata():
    """Restored rows come mostly from the high stratum: pooled, that looks like a big effect;
    within strata the effect is +0.02."""
    rng = np.random.default_rng(7)
    rows = []
    for stratum, (rate, level) in enumerate(((0.8, 0.9), (0.2, 0.6))):
        for subject in range(60):
            for _ in range(2):
                restored = rng.random() < rate
                rows.append((stratum, subject, restored, level + 0.02 * restored + rng.normal(0, 0.01)))
    s, g, lab, v = map(np.array, zip(*rows))
    pooled = stratified_difference(v, lab, np.zeros(len(v)), g, n_boot=300)
    within = stratified_difference(v, lab, s, g, n_boot=300)
    assert pooled["diff"] > 0.15 and pooled["diff"] == pytest.approx(np.median(v[lab]) - np.median(v[~lab]))
    assert within["diff"] == pytest.approx(0.02, abs=0.006) and within["low"] < 0.02 < within["high"]
    assert within["n_strata"] == 2 and within["high"] < pooled["low"] and within["p"] < 0.01
    assert stratified_difference(v, lab, s, g, n_boot=50, min_group=10_000)["n_strata"] == 0


# ---------------------------------------------------------------------------
# v2 tables of scripts/12_analyse.py on a runner-shaped toy table with known effects
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyse():
    spec = importlib.util.spec_from_file_location("analyse", os.path.join(REPO, "scripts", "12_analyse.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EPS = (0.0039216, 0.0078431)
CLEAN_MASS = {"pupil": 0.15, "iris": 0.35, "sclera": 0.25, "periocular": 0.25}


def _v2_rows(noise_defence=False, seed=0):
    """90 images of 45 subjects; LayerCAM (true + adversarial class) and Eigen-CAM. FGSM at 1/255 and
    2/255 fools 2 of 3 images, patch half. A successful attack moves 0.10 of the true-class map to the
    periocular region, 0.01 of the adversarial-class map; failed attacks and noise move nothing, Eigen-CAM
    0.03 either way. FGSM 1/255 restores 3 of 4 fooled images at Spearman 0.85, FGSM 2/255 1 of 4 at 0.65:
    within each condition restored maps are 0.02 closer to the defended clean map."""
    rng = np.random.default_rng(seed)

    def masses(shift):
        shift += rng.normal(0, 0.002)
        m = {**CLEAN_MASS, "iris": CLEAN_MASS["iris"] - shift, "periocular": CLEAN_MASS["periocular"] + shift}
        return {**{f"mass_{k}": v for k, v in m.items()}, **{f"dmass_{k}": m[k] - CLEAN_MASS[k] for k in m}}

    rows = []
    for image in range(90):
        base = {"image_id": image, "true_idx": image // 2}
        for xai, target in (("layercam", "true"), ("eigencam", "any")):
            common = {**base, "xai": xai, "target": target}
            rows.append({**common, "condition": "clean", "correct": True, "spearman": 1.0,
                         "top10_intersection": 1.0, **masses(0.0)})
            for eps in EPS:
                rows.append({**common, "condition": "noise", "epsilon": eps, "correct": True,
                             "spearman": 0.98 + rng.normal(0, 0.005), "top10_intersection": 1.0, **masses(0.0)})
                if noise_defence:
                    rows.append({**common, "condition": "noise+defence", "epsilon": eps, "defence": "jpeg",
                                 "correct": True, "spearman": 0.95, "vs_defended_clean_spearman": 0.9})
            rows.append({**common, "condition": "defence_on_clean", "defence": "jpeg", "correct": True,
                         "spearman": 0.95 + rng.normal(0, 0.005), "top10_intersection": 1.0, **masses(0.0)})
            for attack, eps in (("fgsm", EPS[0]), ("fgsm", EPS[1]), ("patch", np.nan)):
                success = image % 3 != 0 if attack == "fgsm" else image % 2 == 0
                info = {**common, "attack": attack, "epsilon": eps}
                shift = (0.10 if success else 0.0) if xai == "layercam" else 0.03
                rows.append({**info, "condition": "attack", "correct": not success,
                             "spearman": (0.6 if success else 0.7) + rng.normal(0, 0.01),
                             "top10_intersection": 0.5, **masses(shift)})
                if success and xai == "layercam":
                    rows.append({**info, "target": "pred", "condition": "attack", "correct": False,
                                 "spearman": 0.65 + rng.normal(0, 0.01), "top10_intersection": 0.5,
                                 **masses(0.01)})
                if attack == "fgsm":
                    weak = eps == EPS[0]
                    restored = success and ((image % 4 != 0) if weak else (image % 4 == 0))
                    level = 0.85 if weak else 0.65
                else:
                    restored, level = success and image % 6 == 0, 0.5
                rows.append({**info, "condition": "attack+defence", "defence": "jpeg", "attack_success": success,
                             "correct": restored or not success, "spearman": 0.7,
                             "vs_defended_clean_spearman": (level + 0.02 * restored + rng.normal(0, 0.005))
                             if success else 0.95})
    return pd.DataFrame(rows)


def test_region_shift_v2_splits_by_target_and_outcome(analyse):
    rows = _v2_rows()
    table = analyse.region_shift_v2(rows, n_boot=100)
    peri = table[(table.region == "periocular") & (table.measure == "dmass")]
    pick = lambda **k: analyse._one(peri, **k)  # noqa: E731
    s = pick(attack="fgsm", epsilon=EPS[1], xai="layercam", target="true", outcome="success")
    assert s.section == "white-box" and s.n == 60 and s.n_subjects == 45
    assert s.dmass_median == pytest.approx(0.10, abs=0.005) and s.diff_vs_noise == pytest.approx(0.10, abs=0.005)
    assert s.subject_ci_low < s.subject_mean < s.subject_ci_high and s.p_noise_holm < 0.05
    p = pick(attack="fgsm", epsilon=EPS[1], xai="layercam", target="pred", outcome="success")
    assert p.dmass_median == pytest.approx(0.01, abs=0.005)
    assert p.true_minus_pred == pytest.approx(0.09, abs=0.005) and p.p_true_minus_pred_holm < 0.05
    f = pick(attack="fgsm", epsilon=EPS[1], xai="layercam", target="true", outcome="failed")
    assert abs(f.dmass_median) < 0.005 and f.n == 30
    patch = peri[peri.attack == "patch"]
    assert set(patch.section) == {"occlusion"} and patch.noise_n.eq(0).all() and patch.p_noise.isna().all()
    assert {"p_holm", "p_noise_holm", "p_true_minus_pred_holm"} <= set(table.columns)

    # v2 runner columns: occlusion rows also get the mass over non-occluded pixels
    vis = rows.copy()
    occluded = (vis.attack == "patch") & (vis.condition == "attack")
    for r in HEURISTIC_REGIONS:
        vis[f"massvis_{r}"] = np.where(occluded, vis[f"mass_{r}"], np.nan)
        vis[f"dmassvis_{r}"] = np.where(occluded, vis[f"dmass_{r}"] / 2, np.nan)
    measures = analyse.region_shift_v2(vis, n_boot=50).groupby("measure").attack.agg(set)
    assert measures["dmassvis"] == {"patch"} and measures["dmass"] == {"fgsm", "patch"}
    # regions come from the columns (a mask file's class names)
    renamed = rows.rename(columns=lambda c: c.replace("periocular", "skin"))
    assert "skin" in set(analyse.region_shift_v2(renamed, n_boot=50).region)


def test_heatmap_shift_by_outcome_with_controls(analyse):
    table = analyse.heatmap_shift_by_outcome(_v2_rows(), n_boot=100)
    noise = table[table.condition == "noise"]
    assert set(noise.outcome) == {"all"} and (noise.n == 90).all() and len(noise) == 4
    assert set(table[table.condition == "defence_on_clean"].defence) == {"jpeg"}
    s = analyse._one(table, condition="attack", attack="fgsm", epsilon=EPS[0], xai="layercam", target="true",
                     outcome="success")
    f = analyse._one(table, condition="attack", attack="fgsm", epsilon=EPS[0], xai="layercam", target="true",
                     outcome="failed")
    assert s.spearman_median == pytest.approx(0.6, abs=0.01) and f.spearman_median == pytest.approx(0.7, abs=0.01)
    assert s.diff_noise_minus_attack == pytest.approx(0.38, abs=0.01) and s.p_holm < 0.05
    assert s.spearman_ci_low < s.spearman_median < s.spearman_ci_high


def test_restoration_by_outcome_on_a_toy_table(analyse):
    table = analyse.restoration_by_outcome(_v2_rows(), n_boot=300, n_perm=499)
    weak = analyse._one(table, scope="condition", attack="fgsm", epsilon=EPS[0], xai="layercam")
    assert (weak.n, weak.n_restored, weak.n_not_restored) == (60, 45, 15)
    assert weak["diff"] == pytest.approx(0.02, abs=0.005) and weak.p < 0.05 and weak.diff_ci_low > 0
    naive = analyse._one(table, scope="pooled_naive", xai="layercam")
    within = analyse._one(table, scope="pooled_within", xai="layercam")
    assert naive["diff"] > 0.1                                       # mostly which condition gets restored
    assert within["diff"] == pytest.approx(0.02, abs=0.005)
    assert within.diff_ci_high < naive.diff_ci_low and within.n_strata == 3 and np.isnan(within.restored_median)
    assert "noise+defence" not in set(table.scope) and table.control_n.eq(0).all()

    controlled = analyse.restoration_by_outcome(_v2_rows(noise_defence=True), n_boot=200, n_perm=199)
    ctrl = controlled[controlled.scope == "noise+defence"]
    assert len(ctrl) == 4 and np.allclose(ctrl.control_median, 0.9)
    weak = analyse._one(controlled, scope="condition", attack="fgsm", epsilon=EPS[0], xai="layercam")
    assert weak.restored_minus_control == pytest.approx(0.87 - 0.9, abs=0.005) and weak.control_n == 45
    assert analyse._one(controlled, scope="condition", attack="patch", xai="layercam").control_n == 0  # no budget
    within = analyse._one(controlled, scope="pooled_within", xai="layercam")
    assert within.control_n == 45 + 15 and within.restored_minus_control < 0     # restored FGSM maps, both budgets


def test_restoration_by_outcome_with_controls_and_no_successful_attack(analyse):
    rows = _v2_rows(noise_defence=True)
    rows = rows[~((rows.condition == "attack+defence") & rows.attack_success.eq(True))]
    table = analyse.restoration_by_outcome(rows, n_boot=50, n_perm=49)
    assert set(table.scope) == {"noise+defence"} and len(table) == 4
    assert table.p_holm.isna().all() and table.p_vs_control_holm.isna().all()
    lines = []
    assert analyse._report_restoration(table, ["layercam", "eigencam"], lines) == {}
    assert any("Matched control" in line for line in lines)


def test_v2_tables_read_text_booleans_like_real_ones(analyse):
    rows = _v2_rows()
    text = rows.copy()
    text["correct"] = text.correct.map(str)
    text["attack_success"] = text.attack_success.map(lambda v: v if pd.isna(v) else str(v))
    pd.testing.assert_frame_equal(analyse.restoration(rows, 50), analyse.restoration(text, 50))
    pd.testing.assert_frame_equal(analyse.restoration_by_outcome(rows, 50, 49),
                                  analyse.restoration_by_outcome(text, 50, 49))
    pd.testing.assert_frame_equal(analyse.heatmap_shift(rows, 50), analyse.heatmap_shift(text, 50))


def test_check_rows(analyse):
    rows = _v2_rows()
    passed = analyse.check_rows(rows)
    assert any("one true-class or class-agnostic row for each of the 90 images" in p for p in passed)
    summary = json.loads(json.dumps(analyse.summary_rates(rows)))
    assert any("equal those in the summary JSON" in p for p in analyse.check_rows(rows, summary))
    with pytest.raises(ValueError, match="duplicate"):
        analyse.check_rows(pd.concat([rows, rows.iloc[:1]]))
    flipped = rows.copy()
    flipped.loc[flipped.condition == "attack+defence", "attack_success"] = True
    with pytest.raises(ValueError, match="attack_success"):
        analyse.check_rows(flipped)
    summary["attack_success_rate"]["fgsm@0.0039216"]["n"] += 1
    with pytest.raises(ValueError, match="fgsm@0.0039216"):
        analyse.check_rows(rows, summary)
    with pytest.raises(ValueError, match="correct"):
        analyse.check_rows(rows.assign(correct="maybe"))
    with pytest.raises(ValueError, match="exactly one"):                 # one noise map missing
        analyse.check_rows(rows.drop(index=rows.index[rows.condition == "noise"][0]))
    failed = rows[(rows.condition == "attack") & (rows.target == "true") & rows.correct].iloc[:1]
    with pytest.raises(ValueError, match="adversarial-class"):           # a pred map for an image not fooled
        analyse.check_rows(pd.concat([rows, failed.assign(target="pred", correct=False)]))


def _report_text(analyse, rows):
    v2 = {"region_shift_v2": analyse.region_shift_v2(rows, 50)}
    v2["region_null"], source = analyse.region_null(rows, 50)
    v2["heatmap_shift_by_outcome"] = analyse.heatmap_shift_by_outcome(rows, 50)
    v2["restoration_by_outcome"] = analyse.restoration_by_outcome(rows, 50, 49)
    args = argparse.Namespace(experiment_rows="rows_toy.csv", n_boot=50, n_perm=49)
    return analyse.report_v2(v2, source, ["toy rows"], rows, args)


def _claims(text):
    can, cannot = text.split("Can claim:")[1].split("Cannot claim:")
    return can, cannot


def test_report_p_value_wording(analyse):
    assert analyse._pv(6e-51) == "< 1e-10" and analyse._pv(0.0312) == "= 0.031"
    # a bootstrap p at its floor (no resample on the other side of 0) is an upper bound, not a value
    assert analyse._boot_pv(2 / 2001, 2000) == "<= 0.001 (the smallest value 2000 resamples can give)"
    assert analyse._boot_pv(0.004, 2000) == "= 0.004" and analyse._boot_pv(np.nan, 2000) == "= n/a"


def test_report_v2_shows_every_monocle_placement(analyse):
    """v2 runner rows record where the monocle went; the tables split monocle rows by it, and the
    report must show each placement, not whichever comes first."""
    rows = _v2_rows()
    monocle = rows.attack == "patch"
    rows.loc[monocle, "attack"] = "monocle"
    rows["monocle_placement"] = np.where(monocle, np.where(rows.image_id < 46, "haar", "centre_fallback"), None)
    still = monocle & (rows.monocle_placement == "centre_fallback") & (rows.condition == "attack")
    for r in HEURISTIC_REGIONS:                                       # the fallback placement moves nothing
        rows.loc[still, f"mass_{r}"] -= rows.loc[still, f"dmass_{r}"]
        rows.loc[still, f"dmass_{r}"] = 0.0
    lines = _report_text(analyse, rows).splitlines()
    haar = [line for line in lines if line.startswith("| monocle (haar placement) @ no L-inf budget |")]
    fallback = [line for line in lines if line.startswith("| monocle (centre_fallback placement) @ no L-inf budget |")]
    assert len(haar) == len(fallback) == 3                            # region shift, area null and heatmap tables
    shift, n = haar[0].split("|")[2].split()                          # LayerCAM true-class, success
    assert float(shift) == pytest.approx(0.10, abs=0.005) and n == "(23)"
    assert fallback[0].split("|")[2].strip() == "+0.000 (22)"
    assert any(line.startswith("- monocle (haar placement) @ no L-inf budget: layercam true-class, success +0.")
               for line in lines)


def test_report_v2_claims_follow_the_data(analyse):
    can, cannot = _claims(_report_text(analyse, _v2_rows()))
    assert "Successful attacks make the true-class evidence less centred on the eye" in can
    assert "Attacks move heatmaps more than same-budget noise (" in can and "including attacks that failed" in can

    # attacks that move maps no more than noise and shift no mass: neither claim may be made
    rows = _v2_rows()
    attacked = rows.condition == "attack"
    rows.loc[attacked, "spearman"] = rows.loc[rows.condition == "noise", "spearman"].sample(
        int(attacked.sum()), replace=True, random_state=0).to_numpy()
    for r in HEURISTIC_REGIONS:
        rows.loc[attacked, f"mass_{r}"] -= rows.loc[attacked, f"dmass_{r}"]
        rows.loc[attacked, f"dmass_{r}"] = 0.0
    can, cannot = _claims(_report_text(analyse, rows))
    assert "less centred on the eye" not in can and "less centred on the eye' in general" in cannot
    assert "Attacks move heatmaps more than same-budget noise" not in can
    assert "'Attacks move heatmaps more than same-budget noise' for every attack: only" in cannot

    # too few images for any test: no claim, and no 'nan %' or '0 of 0' sentences
    text = _report_text(analyse, _v2_rows()[lambda r: r.image_id < 8])
    can, _ = _claims(text)
    assert "Nothing from these rows" in can
    assert "nan %" not in text and "0 of 0" not in text and "n/a to n/a" not in text


def test_analyse_v2_end_to_end_with_area_csv(tmp_path):
    rows = _v2_rows()
    rows.to_csv(tmp_path / "rows_toy.csv", index=False)
    links = [f"/tmp/ocular_recognition/test/{i // 2}/img_{i}.png" for i in range(90)]
    pd.DataFrame({"image_id": range(90), "link_path": links}).to_csv(tmp_path / "images_toy.csv", index=False)
    areas = pd.DataFrame({"relpath": [f"x/img_{i}.png" for i in range(90)], "link_path": links,
                          "area_pupil": 0.07, "area_iris": 0.14, "area_sclera": 0.09, "area_periocular": 0.70})
    areas.to_csv(tmp_path / "areas.csv", index=False)
    out = tmp_path / "analysis"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "12_analyse.py"), "--experiment-rows",
           str(tmp_path / "rows_toy.csv"), "--areas", str(tmp_path / "areas.csv"), "--n-boot", "50",
           "--n-perm", "49", "--out-dir", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    null = pd.read_csv(out / "region_null.csv")
    assert set(null.area_source) == {"areas_csv"} and set(null.region) == set(HEURISTIC_REGIONS)
    assert null.r2.notna().all() and np.allclose(null.r2_chance, 1 / 3)
    report = (out / "report_v2.md").read_text(encoding="utf-8")
    assert "Areas from the --areas CSV" in report and "R^2" in report
