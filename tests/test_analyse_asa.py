"""scripts/23_analyse_asa.py on synthetic rows with known answers."""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import REPO

GROUPS = ("iris", "sclera", "skin")
N = 12                                      # subjects, one image each
EPS = 0.0039216
SHIFT = lambda i: np.array([-2.0 - 0.1 * i, 0.0, 0.5])          # noqa: E731  attack shift of phi
RESIDUE = np.array([-0.5, 0.0, 0.0])                           # what the defence leaves of it


@pytest.fixture(scope="module")
def analyse():
    spec = importlib.util.spec_from_file_location("analyse_asa", os.path.join(REPO, "scripts", "23_analyse_asa.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rows():
    """Rows in the format of 22: clean, noise, defence_on_clean, noise+defence, attack (pgd) and
    attack+defence (jpeg), baselines blur and mean (mean = 2 x blur on clean images). Even subjects
    are fooled by the attack and restored by the defence; image 0 has no sclera."""
    rows = []

    def add(i, baseline, condition, phi, correct, target="true", **info):
        present = np.array([True, i != 0, True])
        phi = np.where(present, phi, 0.0)
        row = {"image_id": i, "relpath": f"x/{i}.png", "true_idx": i, "condition": condition, "attack": None,
               "adapted_to": None, "epsilon": None, "defence": None, "baseline": baseline, "score": "margin",
               "target": target, "target_idx": i if target == "true" else (i + 1) % N,
               "pred_idx": i if correct else (i + 1) % N, "correct": correct, "v_empty": 0.0,
               "v_full": float(phi.sum()), **info}
        row.update({f"phi_{g}": float(v) for g, v in zip(GROUPS, phi)})
        row.update({f"present_{g}": bool(p) for g, p in zip(GROUPS, present)})
        clean_phi = np.array([3.0, 2.0, 1.0]) * (1 + 0.05 * i)
        row.update({f"mass_layercam_{g}": float(v) for g, v in zip(GROUPS, clean_phi / clean_phi.sum())})
        row.update({"del_auc_asa": 0.3 + 0.01 * i, "del_auc_layercam": 0.35 + 0.01 * i,
                    "del_auc_random": 0.5 + 0.02 * (-1) ** i})
        rows.append(row)

    for i in range(N):
        for scale, baseline in ((1.0, "blur"), (2.0, "mean")):
            clean = scale * np.array([3.0, 2.0, 1.0]) * (1 + 0.05 * i)
            defended_clean = 0.9 * clean
            fooled = i % 2 == 0
            add(i, baseline, "clean", clean, True)
            add(i, baseline, "noise", clean + 0.01 * (-1) ** i, True, epsilon=EPS)
            add(i, baseline, "defence_on_clean", defended_clean, True, defence="jpeg")
            add(i, baseline, "noise+defence", defended_clean, True, epsilon=EPS, defence="jpeg",
                defended_clean_correct=True)
            add(i, baseline, "attack", clean + SHIFT(i), not fooled, attack="pgd", epsilon=EPS)
            if fooled:
                add(i, baseline, "attack", -clean, False, target="pred", attack="pgd", epsilon=EPS)
            add(i, baseline, "attack+defence", defended_clean + RESIDUE, True, attack="pgd", epsilon=EPS,
                defence="jpeg", attack_success=fooled, defended_clean_correct=True)
    return pd.DataFrame(rows)


def summary_of(analyse, rows):
    r = analyse.prepare(rows)
    return {"n_rows": len(rows), "n_images": int(rows.image_id.nunique()),
            "rows_per_condition": {k: int(v) for k, v in rows.condition.value_counts().items()},
            **analyse.summary_rates(r)}


def test_row_accounting(analyse):
    rows = make_rows()
    summary = summary_of(analyse, rows)
    assert summary["attack_success_rate"] == {f"pgd@{EPS:.6g}": {"rate": 0.5, "n": N}}
    assert summary["restoration_rate_among_successful_attacks"] == {f"pgd@{EPS:.6g}+jpeg": {"rate": 1.0, "n": 6}}
    assert summary["attack_success_after_defence_rate"] == {f"pgd@{EPS:.6g}+jpeg": {"rate": 0.0, "n": N}}
    assert summary["noise_defence_flip_rate"] == {f"noise@{EPS:.6g}+jpeg": {"rate": 0.0, "n": N}}
    checks = analyse.check_rows(rows, summary)
    assert len(checks) == 7

    broken = {
        "duplicate": pd.concat([rows, rows.iloc[[0]]], ignore_index=True),
        "adversarial-class": rows.drop(index=rows.index[rows.target == "pred"][0]),
        "attack_success": rows.assign(attack_success=rows.attack_success.where(
            ~((rows.condition == "attack+defence") & (rows.image_id == 3)), True)),
        "efficiency": rows.assign(phi_iris=rows.phi_iris + (rows.index == 5)),
        "null player": rows.assign(phi_sclera=rows.phi_sclera.where(rows.image_id != 0, 0.1),
                                   v_full=rows.v_full + np.where(rows.image_id == 0, 0.1, 0.0)),
        "one true-class row": rows.drop(index=rows.index[(rows.target == "true") & (rows.condition == "noise")][0]),
    }
    for message, frame in broken.items():
        with pytest.raises(ValueError, match=message):
            analyse.check_rows(frame)
    for change in ({"n_rows": len(rows) + 1}, {"n_images": N - 1}):
        with pytest.raises(ValueError):
            analyse.check_rows(rows, {**summary, **change})
    wrong = json.loads(json.dumps(summary))
    wrong["restoration_rate_among_successful_attacks"][f"pgd@{EPS:.6g}+jpeg"]["rate"] = 0.5
    with pytest.raises(ValueError, match="restoration"):
        analyse.check_rows(rows, wrong)
    with pytest.raises(ValueError, match="only in the summary"):
        analyse.check_rows(rows, {**summary, "attack_success_rate": {**summary["attack_success_rate"],
                                                                     "fgsm@0.0039216": {"rate": 0.0, "n": N}}})


def test_adaptive_rows_take_the_outcome_after_the_defence(analyse):
    """A BPDA-PGD image succeeds when the defended image is wrong: its attack and attack+defence rows get that
    outcome, it has no restoration rate, and attack_success that disagrees stops the analysis."""
    rows = make_rows()
    rows["attack_success_undefended"] = rows.attack_success
    bpda = rows[rows.condition.isin(["attack", "attack+defence"])].copy()
    bpda["attack"], bpda["adapted_to"] = "bpda_pgd", "jpeg"
    beaten = bpda.image_id % 3 == 0                            # the defended image is wrong for every 3rd subject
    d = bpda.condition == "attack+defence"
    bpda.loc[d, "correct"] = ~beaten[d]
    bpda.loc[d, "pred_idx"] = np.where(beaten[d], (bpda.image_id[d] + 1) % N, bpda.image_id[d])
    bpda.loc[d, "attack_success_undefended"] = bpda.loc[d, "attack_success"]
    bpda.loc[d, "attack_success"] = beaten[d]
    extra = bpda[d & beaten].assign(target="pred", target_idx=lambda f: (f.image_id + 1) % N)
    rows = pd.concat([rows, bpda, extra], ignore_index=True)
    r = analyse.prepare(rows)
    adaptive = r[r.attack == "bpda_pgd"]
    assert ((adaptive.outcome == "success") == (adaptive.image_id % 3 == 0)).all()
    plain = r[(r.attack == "pgd") & (r.condition == "attack")]
    assert ((plain.outcome == "success") == ~plain.correct).all()
    summary = summary_of(analyse, rows)
    name = f"bpda_pgd[jpeg]@{EPS:.6g}+jpeg"
    assert summary["attack_success_after_defence_rate"][name] == {"rate": 4 / N, "n": N}
    assert name not in summary["restoration_rate_among_successful_attacks"]
    assert any("24 BPDA-PGD rows after the defence" in c for c in analyse.check_rows(rows, summary))
    wrong = rows.copy()
    wrong.loc[(wrong.attack == "bpda_pgd") & (wrong.condition == "attack+defence"), "attack_success"] = \
        wrong.loc[(wrong.attack == "bpda_pgd") & (wrong.condition == "attack+defence"), "attack_success_undefended"]
    with pytest.raises(ValueError, match="attack_success"):
        analyse.check_rows(wrong)


def test_phi_shift_against_the_noise_control(analyse):
    r = analyse.prepare(make_rows())
    table = analyse.phi_shift(r, n_boot=200)
    one = table[(table.condition == "attack") & (table.baseline == "blur")
                & (table.measure == "phi")].set_index("group")
    iris = one.loc["iris"]
    assert iris.n == N and iris.n_subjects == N
    assert iris.mean_shift == pytest.approx(np.mean([SHIFT(i)[0] for i in range(N)]))
    assert iris.control_shift == pytest.approx(0.0, abs=1e-12)
    assert iris.diff_vs_control == pytest.approx(iris.mean_shift)
    assert iris.diff_ci_low <= iris.diff_vs_control <= iris.diff_ci_high
    assert iris.p == pytest.approx(2 / 2 ** N) and iris.p_holm <= 1.0
    assert one.loc["sclera"].n == N - 1                       # image 0 has no sclera
    noise = np.array([0.01 * (-1) ** i for i in range(N)])         # the noise control's shift of every phi
    assert one.loc["sclera"].diff_vs_control == pytest.approx(-noise[1:].mean())
    assert one.loc["skin"].diff_vs_control == pytest.approx(0.5)
    # attack+defence rows are compared with noise+defence (same budget and defence)
    defended = table[(table.condition == "attack+defence") & (table.baseline == "blur")
                     & (table.measure == "phi")].set_index("group")
    clean_iris = np.array([3.0 * (1 + 0.05 * i) for i in range(N)])
    assert defended.loc["iris"].mean_shift == pytest.approx(np.mean(-0.1 * clean_iris - 0.5))
    assert defended.loc["iris"].control_shift == pytest.approx(np.mean(-0.1 * clean_iris))
    assert defended.loc["iris"].diff_vs_control == pytest.approx(-0.5)
    np.testing.assert_allclose(table.p_holm[table.p.notna()] >= table.p[table.p.notna()], True)
    # shares are bounded and shift in the same direction for the iris
    shares = table[table.measure == "share"]
    assert (shares.mean_shift.abs() <= 2).all()
    assert shares[(shares.condition == "attack") & (shares.group == "iris")].mean_shift.lt(0).all()

    by = analyse.phi_shift(r, n_boot=200, by_outcome=True)
    success = by[(by.condition == "attack") & (by.baseline == "blur") & (by.measure == "phi") & (by.group == "iris")
                 & (by.outcome == "success")].iloc[0]
    assert success.n == N // 2 and np.isnan(success.p) and np.isnan(success.shift_ci_low)      # below MIN_N
    assert success.mean_shift == pytest.approx(np.mean([SHIFT(i)[0] for i in range(0, N, 2)]))


def test_baseline_sensitivity(analyse):
    table = analyse.baseline_sensitivity(analyse.prepare(make_rows()), n_boot=100)
    clean = table[(table.scope == "variant") & (table.condition == "clean")].iloc[0]
    assert clean.pair == "blur~mean" and clean.n == N
    assert clean.spearman_median == pytest.approx(1.0) and clean.sign_agreement_median == pytest.approx(1.0)
    assert set(table.scope) == {"variant", "pooled"}
    assert (table.same_empty_share == 1.0).all()               # v_empty = 0 in every synthetic game
    rows = make_rows()
    rows.loc[rows.baseline == "mean", "v_empty"] = 0.5
    rows.loc[rows.baseline == "mean", "v_full"] += 0.5
    assert (analyse.baseline_sensitivity(analyse.prepare(rows), n_boot=20).same_empty_share == 0.0).all()


def test_report_flags_the_shared_empty_coalition_of_inpaint_and_mean(analyse):
    """With full-coverage masks the inpaint game's empty coalition is the mean image: the report says how
    often the two games share v_empty, so their agreement is not read as two independent baselines."""
    import argparse
    rows = make_rows()
    rows = pd.concat([rows, rows[rows.baseline == "mean"].assign(baseline="inpaint")], ignore_index=True)
    tables = {"baseline_sensitivity": analyse.baseline_sensitivity(analyse.prepare(rows), n_boot=20)}
    assert set(tables["baseline_sensitivity"].pair) == {"blur~inpaint", "blur~mean", "inpaint~mean"}
    text = analyse.report(tables, [], None, argparse.Namespace(n_boot=20))
    assert "The inpaint and mean games share v_empty in 100.0% of the rows" in text
    assert "mean-colour image" in text and "same v_empty" in text
    text = analyse.report({"baseline_sensitivity": analyse.baseline_sensitivity(analyse.prepare(make_rows()), 20)},
                          [], None, argparse.Namespace(n_boot=20))
    assert "share v_empty in" not in text                      # no inpaint~mean pair, no note


def test_faithfulness(analyse):
    table = analyse.faithfulness(analyse.prepare(make_rows()), n_boot=100)
    clean = table[(table.condition == "clean") & (table.baseline == "blur") & (table.target == "true")]
    cam = clean.set_index("source").loc["layercam"]
    assert cam.rank_agreement_median == pytest.approx(1.0)      # image 0 (two groups) is NaN, not counted
    assert cam.deletion_auc_median == pytest.approx(0.35 + 0.01 * np.median(np.arange(N)))
    assert cam.auc_minus_random == pytest.approx(np.mean([0.35 + 0.01 * i - 0.5 - 0.02 * (-1) ** i for i in range(N)]))
    assert cam.auc_minus_asa == pytest.approx(0.05)
    assert cam.p_vs_asa < 0.01 and cam.p_vs_asa_holm >= cam.p_vs_asa
    assert set(clean.source) == {"asa", "layercam", "random"}
    assert set(table.condition) == {"clean", "attack"}


def test_rank_agreement_uses_the_asa_order_convention(analyse):
    """Where v_full < v_empty (the true class of an attacked image) the ASA deletion order deletes the most
    negative phi first; rank agreement orients phi the same way, so a source whose masses rank the parts in
    that order agrees with +1 (not -1)."""
    rows = make_rows()
    clean = rows.condition == "clean"
    for g in GROUPS:                                           # every clean game flipped: phi -> -phi
        rows.loc[clean, f"phi_{g}"] = -rows.loc[clean, f"phi_{g}"]
    rows.loc[clean, "v_full"] = -rows.loc[clean, "v_full"]     # v_full < v_empty = 0, efficiency holds
    table = analyse.faithfulness(analyse.prepare(rows), n_boot=50)
    cam = table[(table.condition == "clean") & (table.baseline == "blur") & (table.target == "true")]
    assert cam.set_index("source").loc["layercam"].rank_agreement_median == pytest.approx(1.0)
    from src.xai.shapley import asa_order, oriented_phi, order_from_scores
    phi, present = np.array([-3.0, -1.5, -1.0, -0.5]), np.ones(4, bool)
    np.testing.assert_array_equal(oriented_phi(phi, -9.0, -3.0), -phi)
    np.testing.assert_array_equal(oriented_phi(phi, 3.0, -3.0), phi)
    assert asa_order(phi, -9.0, -3.0, present) == order_from_scores(-phi, present)


def test_attribution_restoration(analyse):
    table = analyse.attribution_restoration(analyse.prepare(make_rows()), n_boot=100)
    phi = table[(table.baseline == "blur") & (table.measure == "phi")].set_index("split")
    assert phi.loc["all"].defended_distance == pytest.approx(0.5)
    assert phi.loc["all"].control_distance == pytest.approx(0.0)
    assert phi.loc["all"].diff_vs_control == pytest.approx(0.5) and phi.loc["all"].p < 0.01
    assert phi.loc["restored"].n == N // 2 and "not_restored" not in phi.index      # every fooled image restored
    assert phi.loc["all"].attack_distance > phi.loc["all"].defended_distance


def test_main_writes_tables_figures_and_report(analyse, tmp_path):
    rows = make_rows()
    path = tmp_path / "rows_asa_syn.csv"
    rows.to_csv(path, index=False)
    (tmp_path / "summary_asa_syn.json").write_text(json.dumps(summary_of(analyse, rows)))
    out = tmp_path / "analysis"
    analyse.main(["--rows", str(path), "--n-boot", "50", "--out-dir", str(out)])
    text = (out / "report_asa.md").read_text(encoding="utf-8")
    assert "summary JSON" in text and "Part shift under attack" in text and "Faithfulness" in text
    assert (out / "fig_phi_shift.png").exists() and (out / "fig_deletion.png").exists()
    # a summary JSON that disagrees stops the run before any table is written
    (tmp_path / "summary_asa_syn.json").write_text(json.dumps({**summary_of(analyse, rows), "n_rows": 1}))
    with pytest.raises(SystemExit, match="row accounting failed"):
        analyse.main(["--rows", str(path), "--out-dir", str(tmp_path / "again")])
    assert not (tmp_path / "again").exists()
