"""
scripts/23_analyse_asa.py

Tables, figures and report_asa.md from the rows of 22_anatomical_shapley.py
(laptop, no model, no TensorFlow).

Inputs: --rows rows_asa_<tag>.csv and --summary summary_asa_<tag>.json (default:
next to the rows file). The row accounting is checked first, as 12 does for 05,
and any mismatch stops the analysis: no duplicate rows; one true-class row per
image for every condition variant and baseline; adversarial-class rows exactly
where the prediction is wrong; attack_success of defended rows equal to the
attack's own outcome (fgsm / pgd: the attacked image fooled the model; BPDA-PGD:
the defended image did, as 22 defines it), and attack_success_undefended equal
to the outcome of the attacked image; efficiency (sum of phi = v_full - v_empty) in
every row; for a parts model, the receipt summing to v_full (receipt_sum); and
the counts and rates of the summary JSON.

Statistics never treat images as independent: every CI is a subject-cluster
bootstrap and every test a Wilcoxon signed-rank test on per-subject means
(src/analysis/stats.py); p-values are Holm-corrected within each table.
Subjects are true_idx. A part group enters a comparison only on images where
it has pixels (present_<g>). Groups of fewer than MIN_N pairs get estimates
only (no CI, no p).

Measures: phi (ASA value, in units of the score) and share (phi / sum over
groups of |phi|, which removes the overall drop of the score under attack and
keeps how the evidence is spread over the parts).

Tables (CSV in --out-dir):
  phi_shift.csv           per condition (attack, attack+defence) x baseline x group x measure:
                          mean over subjects of the shift adv - clean (true class) with its CI,
                          the same for the same-budget control (noise for attack, noise+defence
                          with the same defence for attack+defence), and their difference with
                          CI and Wilcoxon p (Holm)
  phi_shift_by_outcome.csv  the same, split by whether the attack succeeded: fgsm / pgd fooled
                          the undefended model; BPDA-PGD fooled the model behind the defence it
                          was computed through (its attack and attack+defence rows both take
                          that outcome)
  baseline_sensitivity.csv  agreement of phi between hiding baselines (blur, mean, inpaint):
                          Spearman over present groups and the share of groups with the same sign,
                          per condition variant and pooled, and same_empty_share: the share of
                          rows in which the two games have the same v_empty. With masks whose
                          groups cover every pixel (17's masks and the default groups), the
                          inpaint game's empty coalition is the mean-colour image, so mean and
                          inpaint share v_empty in every row and part of their agreement is
                          built in (the report says so)
  faithfulness.csv        LayerCAM, Eigen-CAM, attention and the parts receipt against ASA, on
                          clean and attacked images: rank agreement with phi (median, CI; phi is
                          oriented as the ASA deletion order orients it, i.e. sign-flipped where
                          v_full < v_empty, so a source that ranks the parts in that order
                          agrees with +1, and both columns use one convention) and
                          deletion AUC (median, CI) with the paired difference from the random
                          order and from the ASA order (Wilcoxon p, Holm). The ASA order is
                          deleted on its own game, which favours it: compare the other sources
                          with random, and with each other
  attribution_restoration.csv  per attack x budget x defence x baseline: distance of the defended
                          adversarial phi from the defended clean phi, against the distance of
                          the defended noise phi (noise+defence control), for all images and,
                          among successful attacks, by whether the label was restored (for
                          BPDA-PGD a success means the defended label is wrong, so it has no
                          restored split)
Figures: fig_phi_shift.png, fig_deletion.png. Report: report_asa.md.

Usage:
  python scripts/23_analyse_asa.py --rows results/asa/rows_asa_full.csv --out-dir analysis_asa
  (--n-boot 200 for a quick look; the default 2000 for the paper)
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.analysis.stats import (cluster_bootstrap_median, finite_median, format_p, holm,  # noqa: E402
                                paired_subject_ci, paired_subject_test, parse_bool)

RECEIPT_TOLERANCE = 1e-3       # receipt_sum vs v_full: two float32 computations of the same score
MIN_N = 10                     # fewer finite pairs: estimates only, no CI and no p-value
VARIANT = ["condition", "attack", "adapted_to", "eps", "defence"]
MEASURES = ("phi", "share")
CAM_SOURCES = ("layercam", "eigencam", "attention", "receipt")
CONTROL = {"attack": "noise", "attack+defence": "noise+defence"}
RATE_SECTIONS = {"clean": "clean_flip_rate", "noise": "noise_control_flip_rate",
                 "defence_on_clean": "defence_on_clean_flip_rate", "noise+defence": "noise_defence_flip_rate",
                 "attack": "attack_success_rate"}
DEFENDED_SECTIONS = ("attack_success_after_defence_rate", "restoration_rate_among_successful_attacks")
SAME_EMPTY_ATOL = 1e-5         # v_empty of two games on the same image: one fill, two forward passes


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def groups_in(columns):
    """Part groups in player order, from the phi_<g> columns."""
    return [c[len("phi_"):] for c in columns if c.startswith("phi_")]


def variant_name(condition, attack="", adapted_to="", eps="", defence=""):
    """Summary name of a variant, as 22 writes it ('pgd@0.0039216+jpeg', 'bpda_pgd[jpeg]@...')."""
    name = attack or ("noise" if str(condition).startswith("noise") else "clean")
    return name + (f"[{adapted_to}]" if adapted_to else "") + (f"@{eps}" if eps else "") + \
        (f"+{defence}" if defence else "")


def prepare(rows):
    """Copy of the rows with string variant keys ('' where a key does not apply), parsed flags,
    share_<g> columns and the attack outcome ('success' / 'failed'; 'all' for controls)."""
    r = rows.copy()
    for col in ("attack", "adapted_to", "defence"):
        r[col] = r[col].astype(object).where(r[col].notna(), "") if col in r else ""
    r["eps"] = r.epsilon.map(lambda e: "" if pd.isna(e) else f"{float(e):.6g}") if "epsilon" in r else ""
    r["correct"] = parse_bool(r.correct, "correct")
    groups = groups_in(r.columns)
    for g in groups:
        r[f"present_{g}"] = parse_bool(r[f"present_{g}"], f"present_{g}")
    phi = r[[f"phi_{g}" for g in groups]].to_numpy(np.float64)
    total = np.abs(phi).sum(axis=1, keepdims=True)
    share = np.where(total > 0, phi / np.where(total > 0, total, 1.0), np.nan)
    for k, g in enumerate(groups):
        r[f"share_{g}"] = share[:, k]
    outcome = np.full(len(r), "all", dtype=object)
    attacked = (r.condition == "attack").to_numpy()
    outcome[attacked] = np.where(r.correct.to_numpy()[attacked], "failed", "success")
    defended = (r.condition == "attack+defence").to_numpy()
    if defended.any():
        success = parse_bool(r.attack_success[defended], "attack_success (attack+defence rows)")
        outcome[defended] = np.where(success, "success", "failed")
        # a BPDA-PGD image succeeds when it fools the model behind the defence it was computed through:
        # its undefended (attack) rows take the outcome of that defended row
        adaptive = defended & (r.adapted_to != "").to_numpy() & (r.defence == r.adapted_to).to_numpy()
        keys = ["image_id", "attack", "adapted_to", "eps"]
        beat = dict(zip(r.loc[adaptive, keys].itertuples(index=False, name=None), outcome[adaptive]))
        idx = np.flatnonzero(attacked & (r.adapted_to != "").to_numpy())
        for i, k in zip(idx, r.iloc[idx][keys].itertuples(index=False, name=None)):
            outcome[i] = beat.get(k, outcome[i])
    r["outcome"] = outcome
    return r


def summary_rates(r):
    """The rates of summary_asa_<tag>.json, recomputed from the rows (true-class rows of one
    baseline: one per image and variant)."""
    per = r[(r.target == "true") & (r.baseline == r.baseline.iloc[0])]
    out = {s: {} for s in list(RATE_SECTIONS.values()) + list(DEFENDED_SECTIONS)}
    for key, g in per.groupby(VARIANT, sort=True):
        info = dict(zip(VARIANT, key))
        name = variant_name(**info)
        if info["condition"] in RATE_SECTIONS:
            out[RATE_SECTIONS[info["condition"]]][name] = {"rate": float((~g.correct).mean()), "n": int(len(g))}
        elif info["condition"] == "attack+defence":
            out["attack_success_after_defence_rate"][name] = {"rate": float((~g.correct).mean()), "n": int(len(g))}
            g = g[g.outcome == "success"]
            if len(g) and not info["adapted_to"]:      # restoration among the attacks that fooled the model
                out["restoration_rate_among_successful_attacks"][name] = {"rate": float(g.correct.mean()),
                                                                          "n": int(len(g))}
    return out


def check_rows(rows, summary=None, tolerance=1e-6):
    """Check the row accounting of 22 before any statistics; raise ValueError on a failure.

    Returns
    -------
    list of str — the checks that passed (they go into report_asa.md)
    """
    for col in ("image_id", "true_idx", "condition", "baseline", "target", "correct", "v_full", "v_empty"):
        if col not in rows:
            raise ValueError(f"the rows have no '{col}' column")
    groups = groups_in(rows.columns)
    if not groups:
        raise ValueError("the rows have no phi_<group> columns")
    r = prepare(rows)
    passed = [f"'correct' and present_<group> are True/False in all {len(r):,} rows; groups {groups}"]
    key = ["image_id", *VARIANT, "baseline", "target"]
    dup = int(r.duplicated(key).sum())
    if dup:
        raise ValueError(f"{dup} duplicate rows (same image, variant, baseline and target)")
    passed.append("no duplicate rows (image x condition x attack x adapted_to x epsilon x defence x baseline x target)")

    n_images, true = r.image_id.nunique(), r[r.target == "true"]
    per = true.groupby(VARIANT + ["baseline"]).image_id.agg(["size", "nunique"])
    bad = per[(per["size"] != n_images) | (per["nunique"] != n_images)]
    if len(bad):
        raise ValueError(f"{len(bad)} of {len(per)} variant x baseline groups do not have exactly one true-class row "
                         f"per image ({n_images} images): {bad.head(3).to_dict('index')}")
    passed.append(f"each of the {len(per)} variant x baseline groups has one true-class row for each of the "
                  f"{n_images} images ({r.true_idx.nunique()} subjects)")

    pred = r[r.target == "pred"]
    flipped = set(map(tuple, true.loc[~true.correct & (true.condition != "clean"),
                                      ["image_id", *VARIANT, "baseline"]].to_numpy()))
    if set(map(tuple, pred[["image_id", *VARIANT, "baseline"]].to_numpy())) != flipped or pred.correct.any():
        raise ValueError("adversarial-class rows do not match the true-class rows whose prediction is wrong")
    if len(pred) and (pred.target_idx != pred.pred_idx).any():
        raise ValueError("adversarial-class rows explain a class other than the prediction")
    passed.append(f"adversarial-class rows exist for exactly the {len(pred):,} non-clean true-class rows whose "
                  f"prediction is wrong")

    defended = true[true.condition == "attack+defence"]
    if len(defended):
        on = ["image_id", "attack", "adapted_to", "eps", "baseline"]
        attacked = true.loc[true.condition == "attack", on + ["correct"]].rename(
            columns={"correct": "attacked_correct"})
        cols = on + ["outcome", "correct"] + (["attack_success_undefended"] if "attack_success_undefended" in r else [])
        m = defended[cols].merge(attacked, on=on, how="left", validate="many_to_one")
        if m.attacked_correct.isna().any():
            raise ValueError("attack_success: some attack+defence rows have no attacked image")
        fooled = ~m.attacked_correct.astype(bool).to_numpy()
        adaptive = (m.adapted_to != "").to_numpy()
        # the attack's own goal: fgsm / pgd fool the undefended model, BPDA-PGD the defended one (this row)
        goal = np.where(adaptive, ~m.correct.astype(bool).to_numpy(), fooled)
        if ((m.outcome == "success").to_numpy() != goal).any():
            raise ValueError("attack_success of attack+defence rows differs from the outcome of the attack "
                             "(fgsm / pgd: the attacked image; BPDA-PGD: the defended image)")
        if "attack_success_undefended" in m and \
                (np.asarray(parse_bool(m.attack_success_undefended, "attack_success_undefended")) != fooled).any():
            raise ValueError("attack_success_undefended of attack+defence rows differs from the outcome of the "
                             "attacked image")
        passed.append(f"attack_success of all {len(defended):,} defended true-class rows equals the attack's outcome "
                      f"(fgsm / pgd on the undefended model, {int(adaptive.sum())} BPDA-PGD rows after the defence)")

    phi = r[[f"phi_{g}" for g in groups]].to_numpy(np.float64)
    span = (r.v_full - r.v_empty).to_numpy(np.float64)
    gap = np.abs(phi.sum(axis=1) - span)
    if not np.all(gap <= tolerance * np.maximum(1.0, np.abs(span))):
        raise ValueError(f"efficiency fails in {(gap > tolerance * np.maximum(1.0, np.abs(span))).sum()} rows "
                         f"(sum of phi != v_full - v_empty)")
    absent = np.stack([(~r[f"present_{g}"]).to_numpy() & (r[f"phi_{g}"] != 0).to_numpy() for g in groups])
    if absent.any():
        raise ValueError(f"{absent.sum()} absent groups have a non-zero phi (null player)")
    passed.append(f"efficiency (sum of phi = v_full - v_empty, max gap {gap.max():.2g}) and phi = 0 for absent groups "
                  f"in every row")

    if "receipt_sum" in r:          # parts models: the receipt splits the score of the full image exactly
        receipt = r.receipt_sum.to_numpy(np.float64)
        v_full = r.v_full.to_numpy(np.float64)
        off = np.abs(receipt - v_full)
        bad = ~(off <= RECEIPT_TOLERANCE * np.maximum(1.0, np.abs(v_full)))
        if bad.any():
            raise ValueError(f"the parts receipt does not sum to v_full in {int(bad.sum())} rows "
                             f"(max gap {np.nanmax(off):.3g})")
        passed.append(f"the parts receipt sums to v_full in every row (max gap {off.max():.2g}, float32 forward passes)")

    if summary is not None:
        if summary.get("n_rows") is not None and summary["n_rows"] != len(r):
            raise ValueError(f"the summary JSON counts {summary['n_rows']} rows, the file has {len(r)}")
        if summary.get("n_images") is not None and summary["n_images"] != n_images:
            raise ValueError(f"the summary JSON counts {summary['n_images']} images, the rows {n_images}")
        counts = r.condition.value_counts().to_dict()
        theirs = summary.get("rows_per_condition") or {}
        if theirs and {k: int(v) for k, v in theirs.items()} != {k: int(v) for k, v in counts.items()}:
            raise ValueError(f"rows per condition differ: summary {theirs}, rows {counts}")
        mine, checked = summary_rates(r), 0
        for section, values in mine.items():
            stated = summary.get(section)
            if stated is None:
                continue
            for name in sorted(set(values) | set(stated)):
                if name not in values or name not in stated:
                    raise ValueError(f"{section}[{name}] is only in the {'rows' if name in values else 'summary JSON'}")
                a, b = values[name], stated[name]
                if a["n"] != b["n"] or abs(a["rate"] - b["rate"]) > 1e-9:
                    raise ValueError(f"{section}[{name}]: rows give rate {a['rate']:.6f} on n={a['n']}, the summary "
                                     f"JSON {b['rate']:.6f} on n={b['n']}")
                checked += 1
        passed.append(f"row, image and per-condition counts and all {checked} success, restoration and flip rates "
                      f"(and their n) equal those in the summary JSON")
    return passed


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean_ci(values, subjects, n_boot):
    """Mean over subjects of the per-subject mean, with its cluster-bootstrap CI (CI from MIN_N values)."""
    values = np.asarray(values, np.float64)
    est, lo, hi = paired_subject_ci(values, 0.0, subjects, n_boot)
    if np.isfinite(values).sum() < MIN_N:
        return est, np.nan, np.nan
    return est, lo, hi


def _paired(a, b, subjects, n_boot):
    """Mean over subjects of the per-subject mean of a - b, its CI and the Wilcoxon p on the same
    per-subject means (CI and p from MIN_N finite pairs on)."""
    a = np.asarray(a, np.float64)
    b = np.broadcast_to(np.asarray(b, np.float64), a.shape)
    est, lo, hi = paired_subject_ci(a, b, subjects, n_boot)
    if (np.isfinite(a) & np.isfinite(b)).sum() < MIN_N:
        return est, np.nan, np.nan, np.nan
    return est, lo, hi, paired_subject_test(a, b, subjects)["p"]


def _median_ci(values, subjects, n_boot):
    values = np.asarray(values, np.float64)
    if np.isfinite(values).sum() < MIN_N:
        return finite_median(values), np.nan, np.nan
    return cluster_bootstrap_median(values, subjects, n_boot)


def _add_holm(table, columns):
    """Holm over all p-values of *columns* together (one family per table)."""
    if not len(table):
        return table
    adjusted = holm(np.concatenate([table[c].to_numpy(np.float64) for c in columns]))
    for i, c in enumerate(columns):
        table[f"{c}_holm"] = adjusted[i * len(table):(i + 1) * len(table)]
    return table


def _attach_controls(r, groups):
    """True-class condition rows (attack, attack+defence) joined with the clean row and the
    same-budget control row of the same image and baseline (columns get _clean / _ctrl)."""
    true = r[r.target == "true"]
    cols = [f"{m}_{g}" for m in MEASURES for g in groups]
    clean = true[true.condition == "clean"][["image_id", "baseline"] + cols]
    out = []
    for condition, control in CONTROL.items():
        cond = true[true.condition == condition]
        if not len(cond):
            continue
        ctrl = true[true.condition == control]
        on = ["image_id", "baseline", "eps"] + (["defence"] if condition == "attack+defence" else [])
        m = cond.merge(clean, on=["image_id", "baseline"], suffixes=("", "_clean"), validate="many_to_one")
        m = m.merge(_suffixed(ctrl, on, cols, "_ctrl"), on=on, how="left", validate="many_to_one")
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def phi_shift(r, n_boot, by_outcome=False):
    """Shift of phi (and share) from the clean image, against the same-budget control."""
    groups = groups_in(r.columns)
    m = _attach_controls(r, groups)
    if not len(m):
        return pd.DataFrame()
    keys = VARIANT + ["baseline"] + (["outcome"] if by_outcome else [])
    out = []
    for key, g in m.groupby(keys, sort=True):
        info = dict(zip(keys, key))
        for grp in groups:
            present = g[f"present_{grp}"].to_numpy(bool)
            if not present.any():
                continue
            h = g[present]
            subjects = h.true_idx.to_numpy()
            for measure in MEASURES:
                col = f"{measure}_{grp}"
                a = (h[col] - h[f"{col}_clean"]).to_numpy(np.float64)
                b = (h[f"{col}_ctrl"] - h[f"{col}_clean"]).to_numpy(np.float64)
                est, lo, hi = _mean_ci(a, subjects, n_boot)
                c_est, c_lo, c_hi = _mean_ci(b, subjects, n_boot)
                d_est, d_lo, d_hi, p = _paired(a, b, subjects, n_boot)
                out.append({**info, "group": grp, "measure": measure, "n": int(np.isfinite(a).sum()),
                            "n_subjects": int(len(np.unique(subjects))),
                            "clean_mean": float(np.nanmean(h[f"{col}_clean"])),
                            "mean_shift": est, "shift_ci_low": lo, "shift_ci_high": hi,
                            "control_shift": c_est, "control_ci_low": c_lo, "control_ci_high": c_hi,
                            "diff_vs_control": d_est, "diff_ci_low": d_lo, "diff_ci_high": d_hi, "p": p})
    return _add_holm(pd.DataFrame(out), ["p"])


def baseline_sensitivity(r, n_boot):
    """Agreement of phi between each pair of hiding baselines, per variant and target, and pooled."""
    from src.xai.shapley import rank_agreement
    groups = groups_in(r.columns)
    baselines = sorted(r.baseline.unique())
    if len(baselines) < 2:
        return pd.DataFrame()
    phi_cols = [f"phi_{g}" for g in groups]
    present_cols = [f"present_{g}" for g in groups]
    key = ["image_id", *VARIANT, "target"]
    per_pair = []
    for a, b in itertools.combinations(baselines, 2):
        m = r[r.baseline == a][key + ["true_idx", "v_empty"] + phi_cols + present_cols].merge(
            r[r.baseline == b][key + ["v_empty"] + phi_cols], on=key, suffixes=("", "_b"), validate="one_to_one")
        m["same_v_empty"] = np.isclose(m.v_empty.to_numpy(np.float64), m.v_empty_b.to_numpy(np.float64),
                                       rtol=0.0, atol=SAME_EMPTY_ATOL)
        pa, pb = m[phi_cols].to_numpy(np.float64), m[[f"{c}_b" for c in phi_cols]].to_numpy(np.float64)
        present = m[present_cols].to_numpy(bool)
        m["spearman"] = [rank_agreement(x, y, p) for x, y, p in zip(pa, pb, present)]
        same = (np.sign(pa) == np.sign(pb)) & present
        m["sign_agreement"] = np.where(present.any(axis=1), same.sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                                       np.nan)
        m["pair"] = f"{a}~{b}"
        per_pair.append(m)
    m = pd.concat(per_pair, ignore_index=True)
    out = []
    for scope, keys in (("variant", VARIANT + ["target", "pair"]), ("pooled", ["target", "pair"])):
        for k, g in m.groupby(keys, sort=True):
            info = dict(zip(keys, k))
            row = {"scope": scope, **{c: info.get(c, "") for c in VARIANT}, "target": info["target"],
                   "pair": info["pair"], "n": int(len(g)), "same_empty_share": float(g.same_v_empty.mean())}
            for metric in ("spearman", "sign_agreement"):
                est, lo, hi = _median_ci(g[metric], g.true_idx, n_boot)
                row.update({f"{metric}_median": est, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi,
                            f"{metric}_mean": float(np.nanmean(g[metric])) if g[metric].notna().any() else np.nan})
            out.append(row)
    return pd.DataFrame(out)


def faithfulness(r, n_boot):
    """CAMs, attention and receipt against ASA on clean and attacked images.

    Rank agreement correlates the source's masses with phi oriented as asa_order orients it
    (oriented_phi: sign-flipped where v_full < v_empty, e.g. the true class of an attacked image),
    the convention of the ASA deletion order next to it."""
    from src.xai.shapley import oriented_phi, rank_agreement
    groups = groups_in(r.columns)
    present_cols = [f"present_{g}" for g in groups]
    sources = [s for s in CAM_SOURCES
               if all((f"mass_{s}_{g}" if s != "receipt" else f"receipt_{g}") in r for g in groups)]
    auc_sources = [s for s in ("asa", *CAM_SOURCES, "random") if f"del_auc_{s}" in r]
    sub = r[r.condition.isin(["clean", "attack"])]
    out = []
    for key, g in sub.groupby(VARIANT + ["baseline", "target"], sort=True):
        info = dict(zip(VARIANT + ["baseline", "target"], key))
        subjects = g.true_idx.to_numpy()
        phi = np.stack([oriented_phi(p, full, empty) for p, full, empty in
                        zip(g[[f"phi_{x}" for x in groups]].to_numpy(np.float64), g.v_full, g.v_empty)])
        present = g[present_cols].to_numpy(bool)
        for source in auc_sources:
            row = {**info, "source": source, "n": int(len(g))}
            if source in sources:
                cols = [f"mass_{source}_{x}" if source != "receipt" else f"receipt_{x}" for x in groups]
                other = g[cols].to_numpy(np.float64)
                rho = np.array([rank_agreement(a, b, p) for a, b, p in zip(phi, other, present)])
                est, lo, hi = _median_ci(rho, subjects, n_boot)
                row.update({"rank_agreement_median": est, "rank_ci_low": lo, "rank_ci_high": hi})
            auc = g[f"del_auc_{source}"].to_numpy(np.float64)
            est, lo, hi = _median_ci(auc, subjects, n_boot)
            row.update({"deletion_auc_median": est, "auc_ci_low": lo, "auc_ci_high": hi})
            for ref in ("random", "asa"):
                if source != ref and f"del_auc_{ref}" in g:
                    d, d_lo, d_hi, p = _paired(auc, g[f"del_auc_{ref}"], subjects, n_boot)
                    row.update({f"auc_minus_{ref}": d, f"auc_minus_{ref}_ci_low": d_lo,
                                f"auc_minus_{ref}_ci_high": d_hi, f"p_vs_{ref}": p})
            out.append(row)
    table = pd.DataFrame(out)
    for c in ("p_vs_random", "p_vs_asa"):
        if c not in table:
            table[c] = np.nan
    return _add_holm(table, ["p_vs_random", "p_vs_asa"])


def _suffixed(frame, keys, cols, suffix):
    """*frame* with *cols* renamed to <col><suffix>, keeping the join *keys*."""
    return frame[keys + cols].rename(columns={c: f"{c}{suffix}" for c in cols})


def attribution_restoration(r, n_boot):
    """Distance of defended adversarial phi to defended clean phi vs the noise+defence control."""
    groups = groups_in(r.columns)
    true = r[r.target == "true"]
    defended = true[true.condition == "attack+defence"]
    if not len(defended):
        return pd.DataFrame()
    cols = [f"{m}_{g}" for m in MEASURES for g in groups]
    by = {"defence_on_clean": ["image_id", "baseline", "defence"], "noise+defence": ["image_id", "baseline", "eps",
                                                                                   "defence"],
          "attack": ["image_id", "baseline", "attack", "adapted_to", "eps"], "clean": ["image_id", "baseline"]}
    m = defended
    for condition, suffix in (("defence_on_clean", "_ref"), ("noise+defence", "_ctrl"), ("attack", "_adv"),
                              ("clean", "_clean")):
        other = _suffixed(true[true.condition == condition], by[condition], cols, suffix)
        m = m.merge(other, on=by[condition], how="left", validate="many_to_one")
    present = m[[f"present_{g}" for g in groups]].to_numpy(bool)
    for measure in MEASURES:
        def dist(suffix_a, suffix_b):
            a = m[[f"{measure}_{g}{suffix_a}" for g in groups]].to_numpy(np.float64)
            b = m[[f"{measure}_{g}{suffix_b}" for g in groups]].to_numpy(np.float64)
            return np.sqrt(np.where(present, (a - b) ** 2, 0.0).sum(axis=1))       # over the parts in the image
        m[f"d_def_{measure}"] = dist("", "_ref")
        m[f"d_ctrl_{measure}"] = dist("_ctrl", "_ref")
        m[f"d_adv_{measure}"] = dist("_adv", "_clean")
    out = []
    keys = ["attack", "adapted_to", "eps", "defence", "baseline"]
    for key, g in m.groupby(keys, sort=True):
        info = dict(zip(keys, key))
        splits = [("all", g)]
        success = g[g.outcome == "success"]
        splits += [("restored", success[success.correct]), ("not_restored", success[~success.correct])]
        for split, h in splits:
            if not len(h):
                continue
            subjects = h.true_idx.to_numpy()
            for measure in MEASURES:
                a, b = h[f"d_def_{measure}"].to_numpy(), h[f"d_ctrl_{measure}"].to_numpy()
                est, lo, hi = _mean_ci(a, subjects, n_boot)
                c_est, c_lo, c_hi = _mean_ci(b, subjects, n_boot)
                d_est, d_lo, d_hi, p = _paired(a, b, subjects, n_boot)
                out.append({**info, "split": split, "measure": measure, "n": int(len(h)),
                            "n_subjects": int(len(np.unique(subjects))),
                            "attack_distance": float(np.nanmean(h[f"d_adv_{measure}"])),
                            "defended_distance": est, "defended_ci_low": lo, "defended_ci_high": hi,
                            "control_distance": c_est, "control_ci_low": c_lo, "control_ci_high": c_hi,
                            "diff_vs_control": d_est, "diff_ci_low": d_lo, "diff_ci_high": d_hi, "p": p})
    return _add_holm(pd.DataFrame(out), ["p"])


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

def plot_phi_shift(table, path, baseline, measure="share"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[(table.condition == "attack") & (table.baseline == baseline) & (table.measure == measure)] \
        if len(table) else table
    if not len(t):
        return False
    groups = list(dict.fromkeys(t.group))
    variants = list(dict.fromkeys(zip(t.attack, t.adapted_to, t.eps)))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups) + 0.3 * len(variants)), 4))
    for k, (attack, adapted, eps) in enumerate(variants):
        v = t[(t.attack == attack) & (t.adapted_to == adapted) & (t.eps == eps)].set_index("group").reindex(groups)
        x = np.arange(len(groups)) + (k - (len(variants) - 1) / 2) * width
        err = np.vstack([v.diff_vs_control - v.diff_ci_low, v.diff_ci_high - v.diff_vs_control])
        ax.bar(x, v.diff_vs_control, width, yerr=np.nan_to_num(err), capsize=2,
               label=variant_name("attack", attack, adapted, eps))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(f"{measure} shift minus noise control\n(subject mean, 95% CI)")
    ax.set_title(f"ASA, true class, baseline {baseline}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_deletion(table, path, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[(table.baseline == baseline) & (table.target == "true")] if len(table) else table
    if not len(t):
        return False
    t = t.assign(variant=[variant_name(c, a, d, e) for c, a, d, e in zip(t.condition, t.attack, t.adapted_to, t.eps)])
    variants = list(dict.fromkeys(t.variant))
    sources = list(dict.fromkeys(t.source))
    width = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(variants) * len(sources) / 3), 4))
    for k, source in enumerate(sources):
        v = t[t.source == source].set_index("variant").reindex(variants)
        x = np.arange(len(variants)) + (k - (len(sources) - 1) / 2) * width
        err = np.vstack([v.deletion_auc_median - v.auc_ci_low, v.auc_ci_high - v.deletion_auc_median])
        ax.bar(x, v.deletion_auc_median, width, yerr=np.nan_to_num(err), capsize=2, label=source)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("deletion AUC (median, 95% CI; lower = more faithful)")
    ax.set_title(f"Part deletion, baseline {baseline}")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _num(x, spec="+.3f"):
    return "n/a" if x is None or not np.isfinite(x) else format(x, spec)


def _ci(est, lo, hi, spec="+.3f"):
    if est is None or not np.isfinite(est):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{format(est, spec)} (no CI)"
    return f"{format(est, spec)} [{format(lo, spec)}, {format(hi, spec)}]"


def _markdown(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
        ["| " + " | ".join(str(c) for c in row) + " |" for row in body]


def report(tables, checks, summary, args):
    lines = ["# Anatomical Shapley Attribution: results", ""]
    if summary:
        lines += [f"Run `{summary.get('tag')}`: {summary.get('n_images')} images, {summary.get('n_rows')} rows, "
                  f"commit {summary.get('git_commit')}{' (dirty)' if summary.get('git_dirty') else ''}; groups "
                  f"`{summary.get('groups_spec')}`; score {summary.get('score')}; baselines "
                  f"{summary.get('baselines')}; "
                  f"levels {summary.get('levels')}; attacks {summary.get('attacks')}; defences "
                  f"{summary.get('defences')}; model arch {summary.get('anatomy', {}).get('arch')}.", ""]
    lines += ["## Row accounting", ""] + [f"- {c}" for c in checks] + [""]
    lines += ["## Method notes", "",
              "- Subject-cluster bootstrap CIs and Wilcoxon tests on per-subject means; Holm within each table; "
              f"groups with fewer than {MIN_N} pairs get estimates only (n = {args.n_boot} resamples).",
              "- `share` = phi / sum of |phi| over the groups: how the evidence is spread over the parts, without the "
              "overall drop of the score under attack.",
              "- A group enters a comparison only on images where it has pixels; the masks are zero-shot until "
              "validated against hand labels (18).",
              "- The deletion AUC of the ASA order is computed on the same game as ASA itself, which favours it; "
              "compare the other sources with the random order and with each other.",
              "- Hidden parts use the clean image's eye mask in every condition; defended conditions explain the model "
              "on the defended image.",
              "- Attack outcome: fgsm / pgd succeed when they fool the undefended model; BPDA-PGD succeeds when it "
              "fools the model behind the defence it was computed through (`attack_success_after_defence_rate`).",
              "- Rank agreement uses phi oriented like the ASA deletion order (sign-flipped where v_full < v_empty), "
              "so it and the deletion AUC share one convention.", ""]
    sens = tables.get("baseline_sensitivity", pd.DataFrame())
    shared = sens[(sens.scope == "pooled") & sens.pair.isin(["inpaint~mean", "mean~inpaint"])] if len(sens) else sens
    if len(shared):
        share = float(np.average(shared.same_empty_share, weights=shared.n))
        lines[-1:-1] = [f"- The inpaint and mean games share v_empty in {share:.1%} of the rows. With no pixel kept, "
                        "inpainting has nothing to fill from and uses the mean colour; 17's masks map the background "
                        "to skin, so with the default groups every pixel is in a part and the inpaint game's empty "
                        "coalition is the mean-colour image. Its v_empty and deletion end point are then those of the "
                        "mean game, its phi measure the drop to the mean image (with the other parts inpainted), and "
                        "part of the mean ~ inpaint agreement below is built in."]

    shift = tables.get("phi_shift", pd.DataFrame())
    if len(shift):
        t = shift[(shift.condition == "attack") & (shift.measure == "share")]
        lines += ["## Part shift under attack (true class, share, minus the same-budget noise control)", ""]
        groups = list(dict.fromkeys(t.group))
        body = []
        for key, g in t.groupby(["baseline", "attack", "adapted_to", "eps"], sort=True):
            v = g.set_index("group")
            body.append([variant_name("attack", key[1], key[2], key[3]), key[0]] +
                        [f"{_ci(v.diff_vs_control[x], v.diff_ci_low[x], v.diff_ci_high[x])} (p_Holm "
                         f"{format_p(v.p_holm[x])})" if x in v.index else "n/a" for x in groups])
        lines += _markdown(["attack", "baseline"] + groups, body) + [""]

    faith = tables.get("faithfulness", pd.DataFrame())
    if len(faith):
        t = faith[faith.target == "true"]
        lines += ["## Faithfulness (true class)", "",
                  "Rank agreement = Spearman between phi (oriented like the ASA deletion order: sign-flipped where "
                  "v_full < v_empty) and the source's part masses; deletion AUC: lower = the score falls sooner when "
                  "parts are hidden in the source's order.", ""]
        body = []
        for r in t.itertuples():
            body.append([variant_name(r.condition, r.attack, r.adapted_to, r.eps), r.baseline, r.source,
                         _ci(getattr(r, "rank_agreement_median", np.nan), getattr(r, "rank_ci_low", np.nan),
                             getattr(r, "rank_ci_high", np.nan), ".2f"),
                         _ci(r.deletion_auc_median, r.auc_ci_low, r.auc_ci_high, ".3f"),
                         _ci(getattr(r, "auc_minus_random", np.nan), getattr(r, "auc_minus_random_ci_low", np.nan),
                             getattr(r, "auc_minus_random_ci_high", np.nan)) +
                         f" (p_Holm {format_p(r.p_vs_random_holm)})"])
        lines += _markdown(["condition", "baseline", "source", "rank agreement", "deletion AUC", "AUC - random"],
                           body) + [""]

    if len(sens):
        t = sens[sens.scope == "pooled"]
        lines += ["## Baseline sensitivity (all conditions pooled)", ""]
        lines += _markdown(["target", "pair", "n", "Spearman (median)", "same sign (median)", "same v_empty"],
                           [[r.target, r.pair, r.n,
                             _ci(r.spearman_median, r.spearman_ci_low, r.spearman_ci_high, ".2f"),
                             _ci(r.sign_agreement_median, r.sign_agreement_ci_low, r.sign_agreement_ci_high, ".2f"),
                             f"{r.same_empty_share:.0%}"]
                            for r in t.itertuples()]) + [""]

    rest = tables.get("attribution_restoration", pd.DataFrame())
    if len(rest):
        t = rest[(rest.split == "all") & (rest.measure == "share")]
        lines += ["## Restoration of the attribution (share; distance to the defended clean phi)", "",
                  "Defended adversarial vs noise+defence control (positive difference = the defence leaves the "
                  "attribution further from its defended clean version than it leaves a noisy image).", ""]
        lines += _markdown(["attack", "defence", "baseline", "n", "attack distance", "defended", "control",
                            "difference"],
                           [[variant_name("attack", r.attack, r.adapted_to, r.eps), r.defence, r.baseline, r.n,
                             _num(r.attack_distance, ".3f"),
                             _ci(r.defended_distance, r.defended_ci_low, r.defended_ci_high, ".3f"),
                             _ci(r.control_distance, r.control_ci_low, r.control_ci_high, ".3f"),
                             _ci(r.diff_vs_control, r.diff_ci_low, r.diff_ci_high) +
                             f" (p_Holm {format_p(r.p_holm)})"] for r in t.itertuples()]) + [""]
    return "\n".join(lines)


def _sibling(rows_path, prefix, ext):
    """summary_asa_<tag>.json next to rows_asa_<tag>.csv, if it exists."""
    folder, name = os.path.split(rows_path)
    if not (name.startswith("rows_asa_") and name.endswith(".csv")):
        return None
    path = os.path.join(folder, f"{prefix}{name[len('rows_asa_'):-len('.csv')]}{ext}")
    return path if os.path.exists(path) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="rows_asa_<tag>.csv from 22_anatomical_shapley.py")
    ap.add_argument("--summary", default=None,
                    help="summary_asa_<tag>.json of the run (default: next to the rows file; skipped if there is none)")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for every CI")
    ap.add_argument("--baseline", default=None, help="baseline shown in the figures (default: blur, else the first)")
    ap.add_argument("--out-dir", default="analysis_asa")
    args = ap.parse_args(argv)
    if args.n_boot < 1:
        ap.error("--n-boot must be at least 1")

    rows = pd.read_csv(args.rows)
    summary_path = args.summary or _sibling(args.rows, "summary_asa_", ".json")
    summary = None
    if summary_path:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    try:
        checks = check_rows(rows, summary)
    except ValueError as err:
        raise SystemExit(f"row accounting failed: {err}")
    print(f"{len(rows):,} rows; summary JSON: {summary_path or 'none'}")
    for c in checks:
        print("  check passed:", c)

    r = prepare(rows)
    tables = {"phi_shift": phi_shift(r, args.n_boot),
              "phi_shift_by_outcome": phi_shift(r, args.n_boot, by_outcome=True),
              "baseline_sensitivity": baseline_sensitivity(r, args.n_boot),
              "faithfulness": faithfulness(r, args.n_boot),
              "attribution_restoration": attribution_restoration(r, args.n_boot)}
    os.makedirs(args.out_dir, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    baselines = list(r.baseline.unique())
    baseline = args.baseline or ("blur" if "blur" in baselines else baselines[0])
    plot_phi_shift(tables["phi_shift"], os.path.join(args.out_dir, "fig_phi_shift.png"), baseline)
    plot_deletion(tables["faithfulness"], os.path.join(args.out_dir, "fig_deletion.png"), baseline)
    text = report(tables, checks, summary, args)
    with open(os.path.join(args.out_dir, "report_asa.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()
