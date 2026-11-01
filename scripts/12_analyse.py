"""
scripts/12_analyse.py

Turn the Kaggle outputs into the paper's statistics and figures. Runs on a laptop
(CPU, no model needed) from the downloaded CSVs; either input is optional.

  --experiment-rows   rows_<tag>.csv from 05_run_experiment.py
  --detection         features_<tag>.csv from 11_run_detection.py
  --areas             per-image region areas (relpath, link_path, area_<region>), e.g. for a
                      run whose rows have no area_<region> columns
  --images, --summary images_<tag>.csv and summary_<tag>.json of the run (default: next to
                      the rows file); used to join --areas and to check the row counts

Every CI is a 95 % subject-cluster bootstrap; paired tests are Wilcoxon signed-rank
tests on per-subject means, Holm-corrected within each table.

Writes to --out-dir:
  heatmap_shift.csv      attacked vs clean heatmap (Spearman), vs the same-budget noise control
  restoration.csv        defended-attacked vs defended-clean map, vs the attacked map
  region_shift.csv       change of heatmap mass per eye region (if --concept-masks was used)
  detection_auroc.csv    AUROC with CI per score and attack
  detection_vs_fs.csv    paired AUROC difference: ours vs Feature Squeezing
  fig_heatmap_shift.png, fig_detection.png, fig_regions.png
  report.md              the headline numbers in sentences
and, from the experiment rows, the v2 tables, which split every result by target
(true-class / class-agnostic map vs adversarial-class map) and outcome (attack
succeeded or failed):
  region_shift_v2.csv           region shift per attack x epsilon x XAI x target x outcome, white-box
                                and occlusion (patch, monocle) in separate sections, vs same-budget noise
  region_null.csv               area-baseline null: how much of each shift is the map spreading out
  heatmap_shift_by_outcome.csv  Spearman and top-10 % intersection, with the noise and defence controls
  restoration_by_outcome.csv    label restored vs not restored within each condition, and pooled
  report_v2.md                  every v2 result in sentences, and what we can and cannot claim

Image-level medians come with the CI of that median; tests run on per-subject means,
so the v2 tables also give the CI of the per-subject mean difference (the quantity
tested).

Usage:
  python scripts/12_analyse.py --experiment-rows rows_full.csv --detection features_full.csv --out-dir analysis
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.analysis.region_null import (HEURISTIC_PERIOCULAR_AREA, HEURISTIC_REGIONS,  # noqa: E402
                                      add_null_columns, attach_areas, null_table, regions_in)
from src.analysis.stats import (auroc_ci, cluster_bootstrap_ci, cluster_bootstrap_median,  # noqa: E402
                                cluster_permutation_test, finite_median, format_p, holm,
                                paired_auroc_difference, paired_subject_ci, paired_subject_test, parse_bool,
                                stratified_difference)

FUSED = ("fused_explanation", "fused_feature_squeezing", "fused_prediction", "fused_prediction_explanation",
         "fused_all")
# (ours, baseline): does explanation consistency beat Feature Squeezing, and does it add to logits?
COMPARISONS = (("fused_explanation", "fused_feature_squeezing"), ("fused_all", "fused_feature_squeezing"),
               ("fused_prediction_explanation", "fused_prediction"))


def _same_eps(series, eps):
    if pd.isna(eps):
        return series.isna()
    return np.isclose(series.astype(float), float(eps))


def heatmap_shift(rows, n_boot):
    r = rows[rows.target.isin(["true", "any"])]
    attacked, noise = r[r.condition == "attack"], r[r.condition == "noise"]
    out = []
    for (attack, eps, xai), g in attacked.groupby(["attack", "epsilon", "xai"], dropna=False):
        est, lo, hi = cluster_bootstrap_ci(g.spearman, g.true_idx, n_boot=n_boot)
        row = {"attack": attack, "epsilon": eps, "xai": xai, "n": len(g),
               "attack_success_rate": float((~parse_bool(g.correct, "correct")).mean()),
               "spearman_median": est, "ci_low": lo, "ci_high": hi,
               "noise_spearman_median": np.nan, "median_diff_noise_minus_attack": np.nan,
               "effect_rank_biserial": np.nan, "p": np.nan}
        if not pd.isna(eps):
            ctrl = noise[(noise.xai == xai) & _same_eps(noise.epsilon, eps)]
            m = g.merge(ctrl[["image_id", "target", "spearman"]], on=["image_id", "target"], suffixes=("", "_noise"))
            if len(m):
                test = paired_subject_test(m.spearman_noise, m.spearman, m.true_idx)
                row.update({"noise_spearman_median": float(m.spearman_noise.median()),
                            "median_diff_noise_minus_attack": test["median_diff"],
                            "effect_rank_biserial": test["rank_biserial"], "p": test["p"]})
        out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        table["p_holm"] = holm(table.p)
    return table


def restoration(rows, n_boot):
    r = rows[rows.target.isin(["true", "any"])]
    attacked, defended = r[r.condition == "attack"], r[r.condition == "attack+defence"]
    out = []
    for (attack, eps, defence, xai), g in defended.groupby(["attack", "epsilon", "defence", "xai"], dropna=False):
        g = g[parse_bool(g.attack_success, "attack_success")] if "attack_success" in g else g
        base = attacked[(attacked.attack == attack) & (attacked.xai == xai) & _same_eps(attacked.epsilon, eps)]
        m = g.merge(base[["image_id", "target", "spearman"]], on=["image_id", "target"], suffixes=("", "_attacked"))
        if not len(m):
            continue
        est, lo, hi = cluster_bootstrap_ci(m.vs_defended_clean_spearman, m.true_idx, n_boot=n_boot)
        test = paired_subject_test(m.vs_defended_clean_spearman, m.spearman_attacked, m.true_idx)
        out.append({"attack": attack, "epsilon": eps, "defence": defence, "xai": xai, "n_successful": len(m),
                    "prediction_restored_rate": float(parse_bool(m.correct, "correct").mean()),
                    "restored_spearman_median": est, "ci_low": lo, "ci_high": hi,
                    "attacked_spearman_median": float(m.spearman_attacked.median()),
                    "median_gain": test["median_diff"], "effect_rank_biserial": test["rank_biserial"],
                    "p": test["p"]})
    table = pd.DataFrame(out)
    if len(table):
        table["p_holm"] = holm(table.p)
    return table


def region_shift(rows, n_boot):
    regions = regions_in(rows.columns)            # pupil, iris, sclera, periocular for the heuristic masks
    if not regions:
        return pd.DataFrame()
    attacked = rows[(rows.condition == "attack") & rows.target.isin(["true", "any"])]
    out = []
    for (attack, eps, xai), g in attacked.groupby(["attack", "epsilon", "xai"], dropna=False):
        for region in regions:
            est, lo, hi = cluster_bootstrap_ci(g[f"dmass_{region}"], g.true_idx, n_boot=n_boot)
            test = paired_subject_test(g[f"dmass_{region}"], np.zeros(len(g)), g.true_idx)
            out.append({"attack": attack, "epsilon": eps, "xai": xai, "region": region,
                        "clean_mass_median": float((g[f"mass_{region}"] - g[f"dmass_{region}"]).median()),
                        "dmass_median": est, "ci_low": lo, "ci_high": hi, "p": test["p"]})
    table = pd.DataFrame(out)
    if len(table):
        table["p_holm"] = holm(table.p)
    return table


def detection(features, n_boot):
    clean = features[features.kind == "clean"]
    adv = features[features.kind == "adversarial"]
    adv = adv[parse_bool(adv.success, "success")]
    scores = [c for c in features.columns if c.startswith(("fused_", "xc_", "fs_l1_", "pred_cos_"))]
    groups = [("all", adv)] + [(a, g) for a, g in adv.groupby("attack")] + \
             [(v, g) for v, g in adv.groupby("variant") if v not in set(adv.attack)]
    aurocs, versus = [], []
    for name, g in groups:
        for s in scores:
            est, lo, hi = auroc_ci(clean[s], g[s], clean.true_idx, g.true_idx, n_boot=n_boot)
            aurocs.append({"group": name, "score": s, "n_adv": len(g), "auroc": est, "ci_low": lo, "ci_high": hi})
        for ours, base in COMPARISONS:
            if ours in g and base in g:
                d = paired_auroc_difference(clean[ours], g[ours], clean[base], g[base],
                                            clean.true_idx, g.true_idx, n_boot=n_boot)
                versus.append({"group": name, "ours": ours, "baseline": base, "n_adv": len(g), **d})
    versus = pd.DataFrame(versus)
    if len(versus):
        versus["p_holm"] = holm(versus.p)
    return pd.DataFrame(aurocs), versus


def plot_heatmap_shift(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    wb = table[table.epsilon.notna()]
    if not len(wb):
        return False
    xais = sorted(wb.xai.unique())
    fig, axes = plt.subplots(1, len(xais), figsize=(5 * len(xais), 4), squeeze=False, sharey=True)
    for ax, xai in zip(axes[0], xais):
        t = wb[wb.xai == xai]
        for attack, g in t.groupby("attack"):
            g = g.sort_values("epsilon")
            x = g.epsilon * 255
            ax.errorbar(x, g.spearman_median, yerr=[g.spearman_median - g.ci_low, g.ci_high - g.spearman_median],
                        marker="o", capsize=3, label=attack)
        noise = t.groupby("epsilon").noise_spearman_median.median()
        ax.plot(noise.index * 255, noise.values, "k--", label="noise control")
        ax.set_xscale("log")
        ticks = sorted(t.epsilon.unique() * 255)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.2g}" for v in ticks])
        ax.set_title(xai)
        ax.set_xlabel("epsilon (x/255)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Spearman vs clean heatmap (median, 95% CI)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_detection(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table.score.isin(FUSED)]
    families = [g for g in t.group.unique() if "@" not in str(g)]
    t = t[t.group.isin(families)]
    if not len(t):
        return False
    labels = {"fused_explanation": "explanation consistency (ours)", "fused_feature_squeezing": "Feature Squeezing",
              "fused_prediction": "logit consistency", "fused_prediction_explanation": "logit + explanation",
              "fused_all": "combined (ours)"}
    present = [s for s in FUSED if s in set(t.score)]
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(families)), 4))
    for i, s in enumerate(present):
        g = t[t.score == s].set_index("group").reindex(families)
        x = np.arange(len(families)) + i * width
        ax.bar(x, g.auroc, width, yerr=[g.auroc - g.ci_low, g.ci_high - g.auroc], capsize=2, label=labels[s])
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(families)) + 0.4 - width / 2)
    ax.set_xticklabels(families)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_regions(table, path):
    if not len(table):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table.copy()
    t["variant"] = t.apply(lambda r: r.attack if pd.isna(r.epsilon) else f"{r.attack}@{r.epsilon * 255:.2g}/255", axis=1)
    regions = list(dict.fromkeys(t.region))
    xais = sorted(t.xai.unique())
    fig, axes = plt.subplots(1, len(xais), figsize=(6 * len(xais), 0.35 * t.variant.nunique() + 1.5), squeeze=False)
    for ax, xai in zip(axes[0], xais):
        grid = t[t.xai == xai].pivot_table(index="variant", columns="region", values="dmass_median")[regions]
        lim = max(1e-6, np.nanmax(np.abs(grid.values)))
        im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels(grid.index, fontsize=7)
        ax.set_title(f"{xai}: change in heatmap mass")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def eps_label(eps):
    """The budget for a report line: '4/255', or 'no L-inf budget' (CW, patch, monocle)."""
    return "no L-inf budget" if eps is None or pd.isna(eps) else f"{float(eps) * 255:.3g}/255"


def report(shift, restored, regions, aurocs, versus, n_boot=None):
    lines = ["# Results summary", ""]
    if len(shift):
        s = shift[shift.epsilon.notna()]
        sig = s[s.p_holm < 0.05]
        lines.append(f"- Heatmap shift: {len(sig)} of {len(s)} white-box attack x epsilon x XAI conditions "
                     f"move the heatmap significantly more than same-budget noise (Holm p < 0.05).")
        worst = s.sort_values("spearman_median").head(1)
        for r in worst.itertuples():
            lines.append(f"- Largest shift: {r.attack} @ {r.epsilon * 255:.2g}/255 ({r.xai}): Spearman "
                         f"{r.spearman_median:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] vs noise {r.noise_spearman_median:.2f}.")
    if len(restored):
        sig = restored[(restored.p_holm < 0.05) & (restored.median_gain > 0)]
        lines.append(f"- Restoration: in {len(sig)} of {len(restored)} attack x defence x XAI conditions the defended "
                     f"heatmap is significantly closer to the defended-clean one than the attacked heatmap was.")
    if len(regions):
        top = regions.reindex(regions.dmass_median.abs().sort_values(ascending=False).index).head(3)
        for r in top.itertuples():
            lines.append(f"- Region shift: {r.attack} @ {eps_label(r.epsilon)} ({r.xai}) changes {r.region} mass by "
                         f"{r.dmass_median:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}].")
    if len(aurocs):
        pooled = aurocs[(aurocs.group == "all") & aurocs.score.isin(FUSED)]
        for r in pooled.itertuples():
            lines.append(f"- Detection AUROC, all successful attacks, {r.score}: {r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}].")
    if len(versus):
        for r in versus[versus.group == "all"].itertuples():
            verdict = "better than" if r.low > 0 else ("worse than" if r.high < 0 else "not significantly different from")
            # a bootstrap p cannot resolve values below 1 / n_boot
            p = f"p < {1 / n_boot:.1g}" if n_boot and r.p == 0 else f"p = {format_p(r.p)}"
            lines.append(f"- {r.ours} vs {r.baseline}: {r.diff:+.3f} AUROC [{r.low:+.3f}, {r.high:+.3f}], "
                         f"{p}: {verdict} the baseline.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# v2: every result split by target and outcome, the area-baseline null for region
# shifts, and restoration compared within conditions
# ---------------------------------------------------------------------------

OCCLUSION = ("patch", "monocle")    # they paint over part of the eye: their region shift is partly mechanical
MAIN = ("true", "any")              # the true-class map (LayerCAM) or the class-agnostic map (Eigen-CAM)
CONTROLS = ("noise", "defence_on_clean")
SHIFT_METRICS = ("spearman", "top10_intersection")
RESTORE_METRIC = "vs_defended_clean_spearman"
MIN_N = 10                          # groups with fewer rows get point estimates only: no CI, no p-value


def label_rows(rows):
    """Copy of the rows with the grouping columns of the v2 tables.

    section    occlusion (patch, monocle), white-box (any other attack) or control (no attack)
    outcome    success / failed for attacked rows (prediction flipped or not), "all" otherwise
    placement  where the monocle went (haar, iris, centre_fallback), when the rows record it
    """
    r = rows.copy()
    for col in ("attack", "epsilon", "defence"):
        if col not in r:
            r[col] = np.nan
    correct = parse_bool(r["correct"], "correct")
    attacked = (r.condition == "attack").to_numpy()
    r["section"] = np.where(r.attack.isin(OCCLUSION), "occlusion", np.where(r.attack.notna(), "white-box", "control"))
    r["outcome"] = np.where(attacked, np.where(correct, "failed", "success"), "all")
    if "monocle_placement" in r:
        r["placement"] = r.monocle_placement.where(r.attack == "monocle")
    return r


def _keys(frame, *names):
    """Group keys in table order; placement only when the rows record it."""
    return [k for k in names if k != "placement" or "placement" in frame]


def _with_noise(g, noise, eps, xai, cols):
    """*g* joined, image by image, with the same-budget noise row (columns get '_noise').
    The noise control explains the true class (or no class), so adversarial-class rows
    are paired with the true-class noise map of the same image."""
    ctrl = noise[(noise.xai == xai) & _same_eps(noise.epsilon, eps)]
    return g.merge(ctrl[["image_id"] + list(cols)], on="image_id", suffixes=("", "_noise"))


def _median_ci(values, subjects, n_boot):
    """Median with its subject-cluster bootstrap CI (the CI only from MIN_N finite values on)."""
    values = np.asarray(values, np.float64)
    if np.isfinite(values).sum() < MIN_N:
        return finite_median(values), np.nan, np.nan
    return cluster_bootstrap_median(values, subjects, n_boot)


def _paired(a, b, subjects, n_boot):
    """Mean over subjects of the per-subject mean of a - b, its CI and the Wilcoxon p on the
    same per-subject means (CI and p only from MIN_N finite pairs on)."""
    a = np.asarray(a, np.float64)
    b = np.broadcast_to(np.asarray(b, np.float64), a.shape)
    est, lo, hi = paired_subject_ci(a, b, subjects, n_boot)
    if (np.isfinite(a) & np.isfinite(b)).sum() < MIN_N:
        return est, np.nan, np.nan, np.nan
    return est, lo, hi, paired_subject_test(a, b, subjects)["p"]


def region_shift_v2(rows, n_boot):
    """Region shift per attack x epsilon x XAI x target x outcome x region, white-box and
    occlusion attacks in separate sections.

    Per row: the image-level median shift with its CI; the mean over subjects of the
    per-subject mean shift with its CI and Wilcoxon p (the test runs on these per-subject
    means); for attacks with a budget, the same for the shift minus that of the same-budget
    noise control on the same image; for adversarial-class rows, the true-class shift minus
    the adversarial-class shift on the same images (true_minus_pred). With the v2 runner's
    dmassvis_<region> columns, occlusion attacks also get measure "dmassvis" (mass over
    pixels the occluder left alone). Holm over all p-values of the table (p, p_noise and
    p_true_minus_pred together). Groups with fewer than MIN_N rows get point estimates only.
    """
    regions = regions_in(rows.columns)
    if not regions:
        return pd.DataFrame()
    r = label_rows(rows)
    noise = r[(r.condition == "noise") & r.target.isin(MAIN)]
    attacked = r[r.condition == "attack"]
    visible = [x for x in regions if f"massvis_{x}" in r and f"dmassvis_{x}" in r]
    keys = _keys(r, "section", "attack", "placement", "epsilon", "xai", "target", "outcome")
    dcols = [f"dmass_{x}" for x in regions]
    out = []
    for key, g in attacked.groupby(keys, dropna=False, sort=True):
        info = dict(zip(keys, key))
        subjects = g.true_idx.to_numpy()
        measures = [("dmass", "mass", regions)]
        if info["section"] == "occlusion" and visible:
            measures.append(("dmassvis", "massvis", visible))
        pairs = None if pd.isna(info["epsilon"]) else _with_noise(g, noise, info["epsilon"], info["xai"], dcols)
        partner = None
        if info["target"] == "pred":            # the true-class map of the same attacked images
            same = attacked[(attacked.attack == info["attack"]) & (attacked.xai == info["xai"])
                            & (attacked.target == "true") & _same_eps(attacked.epsilon, info["epsilon"])]
            partner = g.merge(same[["image_id"] + dcols], on="image_id", suffixes=("", "_true"))
        for measure, base, names in measures:
            for region in names:
                col = f"{measure}_{region}"
                values = g[col].to_numpy(np.float64)
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                est, lo, hi = _median_ci(values, subjects, n_boot)
                s_est, s_lo, s_hi, p = _paired(values, 0.0, subjects, n_boot)
                row = {**info, "measure": measure, "region": region, "n": int(finite.sum()),
                       "n_subjects": int(len(np.unique(subjects[finite]))),
                       "clean_mass_median": finite_median(g[f"{base}_{region}"] - g[col]),
                       "dmass_median": est, "ci_low": lo, "ci_high": hi,
                       "subject_mean": s_est, "subject_ci_low": s_lo, "subject_ci_high": s_hi, "p": p,
                       "noise_n": 0, "noise_dmass_median": np.nan, "diff_vs_noise": np.nan,
                       "diff_ci_low": np.nan, "diff_ci_high": np.nan, "p_noise": np.nan,
                       "true_minus_pred": np.nan, "true_minus_pred_ci_low": np.nan,
                       "true_minus_pred_ci_high": np.nan, "p_true_minus_pred": np.nan}
                if info["section"] == "occlusion" and "occluded_frac" in g:
                    row["occluded_frac_median"] = finite_median(g.occluded_frac)
                if measure == "dmass" and pairs is not None and len(pairs):
                    a, b = pairs[col].to_numpy(np.float64), pairs[f"{col}_noise"].to_numpy(np.float64)
                    d_est, d_lo, d_hi, d_p = _paired(a, b, pairs.true_idx.to_numpy(), n_boot)
                    row.update({"noise_n": len(pairs), "noise_dmass_median": finite_median(b),
                                "diff_vs_noise": d_est, "diff_ci_low": d_lo, "diff_ci_high": d_hi, "p_noise": d_p})
                if measure == "dmass" and partner is not None and len(partner):
                    t_est, t_lo, t_hi, t_p = _paired(partner[f"{col}_true"], partner[col],
                                                     partner.true_idx.to_numpy(), n_boot)
                    row.update({"true_minus_pred": t_est, "true_minus_pred_ci_low": t_lo,
                                "true_minus_pred_ci_high": t_hi, "p_true_minus_pred": t_p})
                out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        families = ("p", "p_noise", "p_true_minus_pred")
        adjusted = holm(np.concatenate([table[c].to_numpy() for c in families]))
        for i, c in enumerate(families):
            table[f"{c}_holm"] = adjusted[i * len(table):(i + 1) * len(table)]
    return table


def region_null(rows, n_boot, areas=None, images=None):
    """Area-baseline null (src/analysis/region_null.py) per attack x epsilon x XAI x target x
    outcome, plus the noise and defence-on-clean controls.

    Returns
    -------
    (table, area_source) — area_source None means only the periocular region was analysed
    (heuristic masks, no per-image areas)
    """
    regions = regions_in(rows.columns)
    if not regions:
        return pd.DataFrame(), None
    r = label_rows(rows)
    r = r[(r.condition == "attack") | (r.condition.isin(CONTROLS) & r.target.isin(MAIN))]
    r, source = attach_areas(r, regions, areas, images)
    r = add_null_columns(r, regions, source)
    if r is None:
        print(f"region null: regions {regions} have no per-image areas; pass --areas")
        return pd.DataFrame(), source
    keys = _keys(r, "section", "condition", "attack", "placement", "epsilon", "defence", "xai", "target", "outcome")
    return null_table(r, keys, regions, source, n_boot, min_n=MIN_N), source


def heatmap_shift_by_outcome(rows, n_boot):
    """Spearman and top-10 % intersection with the clean map, split by target and outcome.

    Attacked rows per attack x epsilon x XAI x target x outcome; the controls (same-budget
    noise; defence on the clean image) per epsilon or defence x XAI, true-class or class-
    agnostic maps, not split by outcome ("all": noise flips almost no prediction). Medians
    with CIs. For attacks with a budget: the per-subject mean Spearman gap to the noise
    control on the same images (noise minus attack), its CI and Wilcoxon p, Holm within
    the table. With the v2 runner's columns, also the median entropy change and the ratio
    of raw (pre-normalisation) map magnitudes to the clean map's. Groups with fewer than
    MIN_N rows get point estimates only.
    """
    r = label_rows(rows)
    noise = r[(r.condition == "noise") & r.target.isin(MAIN)]
    keep = (r.condition == "attack") | (r.condition.isin(CONTROLS) & r.target.isin(MAIN))
    keys = _keys(r, "section", "condition", "attack", "placement", "epsilon", "defence", "xai", "target", "outcome")
    metrics = [m for m in SHIFT_METRICS if m in r]
    out = []
    for key, g in r[keep].groupby(keys, dropna=False, sort=True):
        info = dict(zip(keys, key))
        subjects = g.true_idx.to_numpy()
        row = {**info, "n": len(g), "n_subjects": int(g.true_idx.nunique())}
        for m in metrics:
            est, lo, hi = _median_ci(g[m], subjects, n_boot)
            row.update({f"{m}_median": est, f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        if {"entropy", "entropy_clean"} <= set(g.columns):
            row["dentropy_median"] = finite_median(g.entropy - g.entropy_clean)
        for raw in ("raw_sum", "raw_max"):
            if {raw, f"{raw}_clean"} <= set(g.columns):
                row[f"{raw}_ratio_median"] = finite_median(g[raw] / g[f"{raw}_clean"])
        row.update({"noise_n": 0, "noise_spearman_median": np.nan, "noise_top10_median": np.nan,
                    "diff_noise_minus_attack": np.nan, "diff_ci_low": np.nan, "diff_ci_high": np.nan, "p": np.nan})
        if info["condition"] == "attack" and not pd.isna(info["epsilon"]) and "spearman" in metrics:
            m = _with_noise(g, noise, info["epsilon"], info["xai"], metrics)
            if len(m):
                d_est, d_lo, d_hi, d_p = _paired(m.spearman_noise, m.spearman, m.true_idx.to_numpy(), n_boot)
                row.update({"noise_n": len(m), "noise_spearman_median": float(m.spearman_noise.median()),
                            "noise_top10_median": (float(m.top10_intersection_noise.median())
                                                   if "top10_intersection" in metrics else np.nan),
                            "diff_noise_minus_attack": d_est, "diff_ci_low": d_lo, "diff_ci_high": d_hi, "p": d_p})
        out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        table["p_holm"] = holm(table.p)
    return table


def restoration_by_outcome(rows, n_boot, n_perm):
    """'Right answer, wrong reasons?' tested within conditions.

    Among successful attacks, maps of images whose label the defence restored vs maps of
    images it did not, by vs_defended_clean_spearman (true-class or class-agnostic maps):
      scope=condition      per attack x epsilon x defence x XAI: medians with CIs, the
                           difference (restored - not) with a subject-cluster bootstrap CI and
                           a subject-level permutation p (Holm over these rows); CI and p only
                           when both groups have MIN_N rows
      scope=pooled_within  per XAI: weighted mean of the within-condition differences (weights
                           n_restored * n_not / (n_restored + n_not), conditions with both
                           groups of MIN_N rows), bootstrap CI and p
      scope=pooled_naive   per XAI: restored vs not pooled over all conditions; this mixes in
                           which attacks get restored at all, and is shown only for contrast
      scope=noise+defence  the matched control, when the rows have it: same-budget noise, same
                           defence, vs the defended-clean map, per epsilon x defence x XAI.
                           Condition rows then compare their restored maps with it on the same
                           images (restored_minus_control: per-subject mean, CI, Wilcoxon p),
                           and the pooled_within row does so over all conditions with a budget
    """
    if RESTORE_METRIC not in rows:
        return pd.DataFrame()
    r = label_rows(rows)
    d = r[(r.condition == "attack+defence") & r.target.isin(MAIN)]
    if "attack_success" in d:
        d = d[parse_bool(d.attack_success, "attack_success")]
    d = d.assign(restored=parse_bool(d.correct, "correct"))
    controls = r[(r.condition == "noise+defence") & r.target.isin(MAIN)]
    empty = {"control_n": 0, "control_median": np.nan, "control_ci_low": np.nan, "control_ci_high": np.nan,
             "restored_minus_control": np.nan, "restored_minus_control_ci_low": np.nan,
             "restored_minus_control_ci_high": np.nan, "p_vs_control": np.nan}

    def compare(g, strata, scope, info):
        v, lab, subj = g[RESTORE_METRIC].to_numpy(np.float64), g.restored.to_numpy(), g.true_idx.to_numpy()
        n1 = int(lab.sum())
        r1, r0 = _median_ci(v[lab], subj[lab], n_boot), _median_ci(v[~lab], subj[~lab], n_boot)
        if scope == "pooled_within":            # pooled medians are not what this row estimates
            r1 = r0 = (np.nan, np.nan, np.nan)
        diff = stratified_difference(v, lab, strata, subj, n_boot, min_group=MIN_N)
        p, method = diff["p"], "cluster bootstrap"
        if scope == "condition":
            method = "subject permutation"
            if min(n1, len(g) - n1) >= MIN_N:
                p = cluster_permutation_test(v, lab, subj, n_perm)["p"]
            else:                               # too few rows in a group: point estimate only
                p = np.nan
                diff["diff"] = float(np.median(v[lab]) - np.median(v[~lab])) if 0 < n1 < len(g) else np.nan
        return {"scope": scope, **info, "n": len(g), "n_restored": n1, "n_not_restored": len(g) - n1,
                "restored_rate": n1 / len(g), "median_all": finite_median(v) if scope == "condition" else np.nan,
                "restored_median": r1[0], "restored_ci_low": r1[1],
                "restored_ci_high": r1[2], "not_restored_median": r0[0], "not_restored_ci_low": r0[1],
                "not_restored_ci_high": r0[2], "diff": diff["diff"], "diff_ci_low": diff["low"],
                "diff_ci_high": diff["high"], "n_strata": diff["n_strata"], "p": p, "p_method": method, **empty}

    def versus_control(pairs, row):
        if not len(pairs):
            return
        a, b = pairs[RESTORE_METRIC].to_numpy(np.float64), pairs[f"{RESTORE_METRIC}_control"].to_numpy(np.float64)
        est, lo, hi, p = _paired(a, b, pairs.true_idx.to_numpy(), n_boot)
        row.update({"control_n": len(pairs), "restored_minus_control": est, "restored_minus_control_ci_low": lo,
                    "restored_minus_control_ci_high": hi, "p_vs_control": p})

    out, all_pairs = [], {}
    keys = ["section", "attack", "epsilon", "defence", "xai"]
    for key, g in d.groupby(keys, dropna=False, sort=True):
        info = dict(zip(keys, key))
        row = compare(g, np.zeros(len(g)), "condition", info)
        if len(controls) and not pd.isna(info["epsilon"]):
            ctrl = controls[(controls.xai == info["xai"]) & (controls.defence == info["defence"])
                            & _same_eps(controls.epsilon, info["epsilon"])]
            row["control_median"] = float(ctrl[RESTORE_METRIC].median()) if len(ctrl) else np.nan
            pairs = g[g.restored].merge(ctrl[["image_id", RESTORE_METRIC]], on="image_id", suffixes=("", "_control"))
            versus_control(pairs, row)
            all_pairs.setdefault(info["xai"], []).append(pairs)
        out.append(row)
    for xai, g in d.groupby("xai", sort=True):
        info = {"section": "all", "attack": "all", "epsilon": np.nan, "defence": "all", "xai": xai}
        strata = g.groupby(["attack", "epsilon", "defence"], dropna=False).ngroup().to_numpy()
        within = compare(g, strata, "pooled_within", info)
        if all_pairs.get(xai):
            versus_control(pd.concat(all_pairs[xai], ignore_index=True), within)
        out += [within, compare(g, np.zeros(len(g)), "pooled_naive", info)]
    for (eps, defence, xai), g in controls.groupby(["epsilon", "defence", "xai"], sort=True):
        est, lo, hi = _median_ci(g[RESTORE_METRIC], g.true_idx.to_numpy(), n_boot)
        out.append({"scope": "noise+defence", "section": "control", "attack": np.nan, "epsilon": eps,
                    "defence": defence, "xai": xai, "n": len(g), **empty, "control_n": len(g),
                    "control_median": est, "control_ci_low": lo, "control_ci_high": hi})
    table = pd.DataFrame(out)
    if len(table):
        for col in ("p", "p_vs_control"):       # absent when only noise+defence control rows exist
            if col not in table:
                table[col] = np.nan
        cond = (table.scope == "condition").to_numpy()
        table["p_holm"] = np.nan
        table.loc[cond, "p_holm"] = holm(table.p[cond])
        table["p_vs_control_holm"] = np.nan
        table.loc[cond, "p_vs_control_holm"] = holm(table.p_vs_control[cond])
    return table


def summary_rates(rows):
    """The rates of summary_<tag>.json, recomputed from the rows the way 05's summarise() does
    (one row per image and variant; names like 'fgsm@0.0039216+jpeg')."""
    r = rows.copy()
    for col in ("attack", "epsilon", "defence"):
        if col not in r:
            r[col] = np.nan
    r["correct"] = parse_bool(r.correct, "correct")
    r["eps"] = r.epsilon.map(lambda e: "" if pd.isna(e) else f"{float(e):.6g}")
    r["attack"] = r.attack.astype(object).where(r.attack.notna(), "")
    r["defence"] = r.defence.astype(object).where(r.defence.notna(), "")
    per = r.drop_duplicates(subset=["image_id", "condition", "attack", "eps", "defence"])

    def rate(frame, value, prefix=""):
        out = {}
        for (attack, eps, defence), g in frame.groupby(["attack", "eps", "defence"]):
            name = (attack or prefix) + (f"@{eps}" if eps else "") + (f"+{defence}" if defence else "")
            out[name] = {"rate": float((g.correct == value).mean()), "n": int(len(g))}
        return out

    defended = per[per.condition == "attack+defence"]
    if len(defended) and "attack_success" in defended:
        defended = defended[parse_bool(defended.attack_success, "attack_success")]
    return {"attack_success_rate": rate(per[per.condition == "attack"], False),
            "restoration_rate_among_successful_attacks": rate(defended, True),
            "noise_control_flip_rate": rate(per[per.condition == "noise"], False, "noise"),
            "defence_on_clean_flip_rate": rate(per[per.condition == "defence_on_clean"], False, "clean")}


def check_rows(rows, summary=None):
    """Check the row accounting of 05 before any statistics; raise ValueError on a failure.

    Returns
    -------
    list of str — the checks that passed (they go into report_v2.md)
    """
    for col in ("image_id", "true_idx", "condition", "xai", "target", "correct"):
        if col not in rows:
            raise ValueError(f"the rows have no '{col}' column")
    r = label_rows(rows)
    r["correct"] = parse_bool(r.correct, "correct")
    passed = [f"'correct' is True/False (or 1/0) in all {len(r):,} rows"]
    r["eps"] = r.epsilon.map(lambda e: "" if pd.isna(e) else f"{float(e):.6g}")
    r["attack"] = r.attack.astype(object).where(r.attack.notna(), "")
    r["defence"] = r.defence.astype(object).where(r.defence.notna(), "")
    variant = ["condition", "attack", "eps", "defence"]
    dup = int(r.duplicated(["image_id", *variant, "xai", "target"]).sum())
    if dup:
        raise ValueError(f"{dup} duplicate rows (same image, condition, attack, epsilon, defence, XAI and target)")
    passed.append("no duplicate rows (image x condition x attack x epsilon x defence x XAI x target)")

    n_images, main = r.image_id.nunique(), r[r.target.isin(MAIN)]
    per = main.groupby(variant + ["xai"]).image_id.agg(["size", "nunique"])
    bad = per[(per["size"] != n_images) | (per["nunique"] != n_images)]
    if len(bad):
        raise ValueError(f"{len(bad)} of {len(per)} condition x XAI variants do not have exactly one true-class or "
                         f"class-agnostic row per image ({n_images} images): {bad.head(3).to_dict('index')}")
    passed.append(f"each of the {len(per)} condition x XAI variants has exactly one true-class or class-agnostic "
                  f"row for each of the {n_images} images ({r.true_idx.nunique()} subjects)")

    att = r[r.condition == "attack"]
    key = ["image_id", "attack", "eps", "xai"]
    pred = att[att.target == "pred"]
    if len(pred):
        flipped = set(map(tuple, att.loc[(att.target == "true") & ~att.correct, key].to_numpy()))
        if set(map(tuple, pred[key].to_numpy())) != flipped or pred.correct.any():
            raise ValueError("adversarial-class rows do not match the attacked true-class rows whose "
                             "prediction flipped")
        passed.append(f"adversarial-class (pred) rows exist for exactly the {len(pred):,} attacked true-class maps "
                      f"whose prediction flipped")

    defended = main[main.condition == "attack+defence"]
    if len(defended) and "attack_success" in rows:
        success = parse_bool(defended.attack_success, "attack_success (attack+defence rows)")
        ref = main.loc[main.condition == "attack", key + ["correct"]]
        m = defended[key].assign(success=success).merge(ref, on=key, how="left", validate="many_to_one")
        if m.correct.isna().any() or (m.success.to_numpy() != ~m.correct.astype(bool).to_numpy()).any():
            raise ValueError("attack_success of attack+defence rows differs from the outcome of the attacked image")
        passed.append(f"attack_success is True/False on all {len(defended):,} defended rows and equals the outcome "
                      f"of the attacked image (success = prediction flipped)")

    if summary is not None:
        mine, checked = summary_rates(rows), 0
        for section, values in mine.items():
            theirs = summary.get(section)
            if theirs is None:
                continue
            for name in sorted(set(values) | set(theirs)):
                if name not in values or name not in theirs:
                    print(f"note: {section}[{name}] is only in the {'rows' if name in values else 'summary JSON'}")
                    continue
                a, b = values[name], theirs[name]
                if a["n"] != b["n"] or abs(a["rate"] - b["rate"]) > 1e-9:
                    raise ValueError(f"{section}[{name}]: rows give rate {a['rate']:.6f} on n={a['n']}, the summary "
                                     f"JSON {b['rate']:.6f} on n={b['n']}")
                checked += 1
        passed.append(f"all {checked} attack-success, restoration and flip rates (and their n) equal those in the "
                      f"summary JSON")
    return passed


def _num(x, spec="+.3f"):
    """*x* formatted with *spec*; a value that rounds to zero never shows a minus sign."""
    if x is None or not np.isfinite(x):
        return "n/a"
    text = format(x, spec)
    return format(0.0, spec) if float(text.rstrip("%")) == 0 else text


def _pv(p):
    """'< 1e-10' or '= 0.03', to follow a 'p' in a sentence."""
    text = format_p(p)
    return text if text.startswith("<") else f"= {text}"


def _boot_pv(p, n_boot):
    """Like :func:`_pv` for a bootstrap p; at its floor 2 / (n_boot + 1) (no resample on the other side
    of 0) the p-value is only an upper bound, and says so."""
    floor = 2 / (n_boot + 1) if n_boot else 0.0
    if np.isfinite(p) and p <= floor * (1 + 1e-9):
        return f"<= {floor:.2g} (the smallest value {n_boot} resamples can give)"
    return _pv(p)


def _ci(est, lo, hi, spec="+.3f"):
    """'+0.163 [+0.136, +0.184]', or the estimate alone when the group was too small for a CI."""
    if est is None or not np.isfinite(est):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{format(est, spec)} (no CI: fewer than {MIN_N} rows)"
    return f"{format(est, spec)} [{format(lo, spec)}, {format(hi, spec)}]"


def _pick(table, **match):
    """Rows of a table with these column values (epsilon compared numerically; None or NaN = missing)."""
    if not len(table):
        return table
    sel = np.ones(len(table), bool)
    for col, value in match.items():
        if col not in table:
            return table.iloc[0:0]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            sel &= table[col].isna().to_numpy()
        elif col == "epsilon":
            sel &= np.isclose(table[col].astype(float).fillna(-1.0).to_numpy(), float(value))
        else:
            sel &= (table[col] == value).to_numpy()
    return table[sel]


def _one(table, **match):
    found = _pick(table, **match)
    return found.iloc[0] if len(found) else None


def _variants(table, section, order=()):
    """(attack, epsilon, placement) of a section, attacks in *order* (e.g. as 05 ran them; others after,
    alphabetically), budgets ascending. placement is where the monocle went when the rows record it (v2
    runner; the tables split monocle rows by it), else None."""
    t = table[table.section == section]
    rank = {a: i for i, a in enumerate(order)}
    places = t.placement if "placement" in t else [None] * len(t)
    found = {(a, None if pd.isna(e) else float(e), None if p is None or pd.isna(p) else str(p))
             for a, e, p in zip(t.attack, t.epsilon, places)}
    return sorted(found, key=lambda v: (rank.get(v[0], len(rank)), v[0], np.inf if v[1] is None else v[1], v[2] or ""))


def _variant(table, attack, eps, placement):
    """Column values that pick one attack variant of a v2 table (placement only when the table has it)."""
    match = {"attack": attack, "epsilon": eps}
    if "placement" in table:
        match["placement"] = placement
    return match


def _variant_label(attack, eps, placement=None):
    """'fgsm @ 4/255', or 'monocle (haar placement) @ no L-inf budget' when the placement is recorded."""
    place = "" if placement is None or pd.isna(placement) else f" ({placement} placement)"
    return f"{attack}{place} @ {eps_label(eps)}"


def _focus_region(regions):
    """The region the report is about: the one outside the central eye opening."""
    for name in ("periocular", "skin"):
        if name in regions:
            return name
    return regions[-1] if regions else None


def _map_columns(main_target):
    """(xai, target, outcome, header) for the summary tables: true-class / adversarial-class maps of
    class-specific methods, success / failed for class-agnostic ones."""
    cols = []
    for xai, target in main_target.items():
        if target == "true":
            cols += [(xai, "true", "success", f"{xai} true-class, success"),
                     (xai, "true", "failed", f"{xai} true-class, failed"),
                     (xai, "pred", "success", f"{xai} adversarial-class")]
        else:
            cols += [(xai, target, "success", f"{xai}, success"), (xai, target, "failed", f"{xai}, failed")]
    return cols


def _markdown(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in body]


def _report_region_shift(table, rows, region, main_target, order, lines):
    lines += ["## 1. Region shift by target and outcome (region_shift_v2.csv)", ""]
    if not len(table) or region is None:
        lines += ["No region masses in these rows (run 05 with --concept-masks).", ""]
        return {}
    where = ("outside the fixed central ellipse of the heuristic masks (69.25 % of every image; not a "
             "segmentation of skin)" if region == "periocular" else f"the '{region}' class of the mask file")
    lines += [f"Change of the heatmap share on **{region}**, {where}, relative to the clean map of the same image. "
              f"Cells: median (n images). CIs, per-subject means and tests for every region are in the CSV.", ""]
    t = table[(table.measure == "dmass") & (table.region == region)]
    cols = _map_columns(main_target)
    noise = rows[(rows.condition == "noise") & rows.target.isin(MAIN)]
    facts = {}
    for section, title in (("white-box", "White-box attacks"), ("occlusion", "Occlusion attacks (patch, monocle)")):
        variants = _variants(t, section, order)
        if not variants:
            continue
        header = ["attack @ budget"] + [c[3] for c in cols]
        if section == "white-box":
            header += [f"noise ({xai})" for xai in main_target]
        body = []
        for attack, eps, placement in variants:
            cells = [_variant_label(attack, eps, placement)]
            for xai, target, outcome, _ in cols:
                r = _one(t, section=section, **_variant(t, attack, eps, placement), xai=xai, target=target,
                         outcome=outcome)
                cells.append("-" if r is None else f"{_num(r.dmass_median)} ({r.n})")
            if section == "white-box":
                for xai in main_target:
                    q = noise.iloc[0:0] if eps is None else noise[(noise.xai == xai) & _same_eps(noise.epsilon, eps)]
                    cells.append(_num(q[f"dmass_{region}"].median()) if len(q) else "-")
            body.append(cells)
        lines += [f"### {title}", ""] + _markdown(header, body) + [""]

    class_xai = [x for x, tg in main_target.items() if tg == "true"]
    agnostic = [x for x, tg in main_target.items() if tg != "true"]
    wb = t[(t.section == "white-box") & t.epsilon.notna()]
    for xai in class_xai:
        succ, pred = _pick(wb, xai=xai, target="true", outcome="success"), _pick(wb, xai=xai, target="pred")
        fail = _pick(wb, xai=xai, target="true", outcome="failed")
        fail = fail[fail.n >= MIN_N]
        sig = lambda q: int(((q.p_noise_holm < 0.05) & (q.diff_vs_noise > 0)).sum())  # noqa: E731
        tested = lambda q: int(q.p_noise_holm.notna().sum())  # noqa: E731
        paired = pred[pred.p_true_minus_pred_holm.notna()]
        more = int(((paired.p_true_minus_pred_holm < 0.05) & (paired.true_minus_pred > 0)).sum())
        facts.update({"succ_range": (succ.dmass_median.min(), succ.dmass_median.max()),
                      "pred_range": (pred.dmass_median.min(), pred.dmass_median.max()),
                      "succ_sig": (sig(succ), tested(succ)), "pred_sig": (sig(pred), tested(pred)),
                      "fail_sig": (sig(fail), tested(fail)), "true_vs_pred": (more, len(paired)), "class_xai": xai})
        attacks_wb = [v[0] for v in _variants(wb, "white-box", order)]
        key = "fgsm" if "fgsm" in attacks_wb else (attacks_wb[0] if attacks_wb else None)
        if key is not None:
            eps = float(wb[wb.attack == key].epsilon.max())
            s = _one(wb, attack=key, epsilon=eps, xai=xai, target="true", outcome="success")
            p = _one(wb, attack=key, epsilon=eps, xai=xai, target="pred")
            f = _one(wb, attack=key, epsilon=eps, xai=xai, target="true", outcome="failed")
            text = f"- {key.upper()} @ {eps_label(eps)}, {xai}, {region}:"
            if s is not None:
                text += (f" the true-class map of successfully attacked images shifts by "
                         f"{_ci(s.dmass_median, s.ci_low, s.ci_high)} (median, n = {s.n}; per-subject mean "
                         f"{_ci(s.subject_mean, s.subject_ci_low, s.subject_ci_high)});")
                facts["key"] = (key, eps, s.dmass_median, p.dmass_median if p is not None else np.nan,
                                f.dmass_median if f is not None else np.nan)
            if p is not None:
                text += (f" the adversarial-class map of the same images by "
                         f"{_ci(p.dmass_median, p.ci_low, p.ci_high)} (n = {p.n});")
            if f is not None:
                text += (f" when the attack failed, the true-class map shifts by "
                         f"{_ci(f.dmass_median, f.ci_low, f.ci_high)} (n = {f.n}; per-subject mean "
                         f"{_ci(f.subject_mean, f.subject_ci_low, f.subject_ci_high)}, Holm p {_pv(f.p_holm)}).")
            lines.append(text)
        if not len(succ):
            continue
        text = (f"- White-box attacks with a budget, {xai}: successful attacks shift the true-class map by "
                f"{_num(succ.dmass_median.min())} to {_num(succ.dmass_median.max())} (medians over "
                f"{len(succ)} attack x budget conditions)")
        if len(pred):
            text += f", the adversarial-class map by {_num(pred.dmass_median.min())} to {_num(pred.dmass_median.max())}"
        if not tested(succ):
            lines.append(text + f". No condition has {MIN_N} images, so none is compared with same-budget noise.")
            continue
        counts = [f"{sig(q)} of {tested(q)} {what} conditions" for q, what in
                  ((succ, "true-class"), (pred, "adversarial-class"), (fail, "failed-attack")) if tested(q)]
        text += (f". Against same-budget noise on the same images (per-subject means, Holm over the whole table; "
                 f"conditions with at least {MIN_N} images), the shift is larger in "
                 + (", ".join(counts[:-1]) + " and " + counts[-1] if len(counts) > 1 else counts[0]) + ".")
        if len(paired):
            text += (f" On the same images, the true-class map shifts more than the adversarial-class map in {more} "
                     f"of {len(paired)} conditions (per-subject mean difference, Holm p < 0.05).")
        lines.append(text)
    for xai in agnostic:
        a = _pick(wb, xai=xai, outcome="success")
        f = _pick(wb, xai=xai, outcome="failed")
        f = f[f.n >= MIN_N]
        if len(a):
            text = (f"- {xai} (class-agnostic) shifts by {_num(a.dmass_median.min())} to "
                    f"{_num(a.dmass_median.max())} after successful white-box attacks")
            if len(f):
                text += (f", and by {_num(f.dmass_median.min())} to {_num(f.dmass_median.max())} after failed ones "
                         f"({len(f)} conditions with at least {MIN_N} images)")
                pairs = [(_one(a, attack=x.attack, epsilon=x.epsilon), x) for x in f.itertuples()]
                pairs = [(ok, fl) for ok, fl in pairs if ok is not None]
                if pairs and all(fl.dmass_median >= ok.dmass_median for ok, fl in pairs):
                    text += (": failed attacks shift it at least as much as successful ones, so its shift is not "
                             "tied to success")
            lines.append(text + ".")
    occ = t[t.section == "occlusion"]
    for attack, eps, placement in _variants(occ, "occlusion", order):
        parts = []
        for xai, target, outcome, header in cols:
            r = _one(occ, **_variant(occ, attack, eps, placement), xai=xai, target=target, outcome=outcome)
            if r is not None:
                parts.append(f"{header} {_ci(r.dmass_median, r.ci_low, r.ci_high)} (n = {r.n})")
        lines.append(f"- {_variant_label(attack, eps, placement)}: " + "; ".join(parts) + ".")
    if len(occ):
        vis = table[(table.measure == "dmassvis") & (table.region == region)]
        if len(vis):
            lines.append(f"- Over the pixels the occluder left alone (dmassvis), {region}: " + "; ".join(
                f"{_variant_label(r.attack, r.epsilon, getattr(r, 'placement', None))} {r.xai} {r.target} "
                f"{r.outcome} {_ci(r.dmass_median, r.ci_low, r.ci_high)}" for r in vis.itertuples()) + ".")
        else:
            lines.append("- Patch and monocle cover part of the eye, and the masks come from the clean image, so part "
                         "of their region shift is mechanical (covered cells lose their mass). Separating it needs the "
                         "v2 runner's massvis/dmassvis columns (mass over non-occluded pixels); these rows have none.")
    lines.append("")
    return facts


def _report_null(table, area_source, region, main_target, order, lines):
    lines += ["## 2. Area-baseline null (region_null.csv)", ""]
    if not len(table):
        lines += ["Not computed: no region masses, or regions without per-image areas (pass --areas).", ""]
        return {}
    lines += ["A map that only spreads out drifts towards a uniform map, which puts on each region exactly its "
              "area share. Enrichment = mass / area share (1 = as much as a uniform map); 'gap closed' (lam) = "
              "the share of the distance to the uniform map's mass that the shift covers.", ""]
    facts = {}
    if area_source is None:
        lines += [f"**Only the periocular region is analysed, exactly.** Its area is the same for every image "
                  f"({HEURISTIC_PERIOCULAR_AREA:.4f}, the outside of the fixed ellipse), but these rows store no "
                  f"per-image areas for pupil, iris and sclera. The 4-region null (one lam per row, residual per "
                  f"region, R^2) needs per-image areas: rerun with --areas <per-image area CSV> (one row per "
                  f"image: relpath/link_path and area_<region>) or on rows from the v2 runner, which writes "
                  f"area_<region> columns. With two regions (inside and outside the ellipse) one lam always "
                  f"fits perfectly, so the null itself cannot be tested here.", ""]
    else:
        lines += [f"Areas from {'the rows (area_<region> columns)' if area_source == 'columns' else 'the --areas CSV'}."
                  f" R^2 = share of the shift (pooled over rows and regions) explained by lam alone; chance for one "
                  f"free direction is in r2_chance.", ""]
    t = _pick(table, region="periocular" if area_source is None else region)
    for xai, target in main_target.items():
        # controls cover every image, so their clean-map columns describe all clean maps
        clean = _one(t, condition="noise", xai=xai, target=target)
        clean = clean if clean is not None else _one(t, condition="defence_on_clean", xai=xai, target=target)
        if clean is not None and np.isfinite(clean.enr_clean_median):
            facts.setdefault("clean_enr", {})[xai] = (clean.enr_clean_median, clean.frac_clean_above_area)
            lines.append(f"- Clean maps, {xai}: {region if area_source else 'periocular'} mass "
                         f"{clean.clean_mass_median:.3f} (median), enrichment {clean.enr_clean_median:.3f}; "
                         f"{100 * clean.frac_clean_above_area:.1f} % of clean maps put more than the area share there.")

    def cell(r):
        return "-" if r is None else f"{_num(r.enr_median, '.2f')} / {_num(100 * r.lam_median, '.0f')} % ({r.n})"

    cols = _map_columns(main_target)
    header = ["attack @ budget"] + [f"{c[3]}: enrichment / gap closed" for c in cols]
    body = []
    for section in ("white-box", "occlusion"):
        for attack, eps, placement in _variants(t, section, order):
            body.append([_variant_label(attack, eps, placement)] + [
                cell(_one(t, section=section, **_variant(t, attack, eps, placement), xai=xai, target=target,
                          outcome=outcome))
                for xai, target, outcome, _ in cols])
    for eps in sorted(set(_pick(t, condition="noise").epsilon.dropna())):      # noise: in the success columns
        body.append([f"noise @ {eps_label(eps)}"] + [
            cell(_one(t, condition="noise", epsilon=eps, xai=xai, target=target)
                 if outcome == "success" and target != "pred" else None)
            for xai, target, outcome, _ in cols])
    lines += [""] + _markdown(header, body) + [""]
    class_xai = [x for x, tg in main_target.items() if tg == "true"]
    for xai in class_xai:
        wb = _pick(t, section="white-box", xai=xai, target="true", outcome="success")
        wb = wb[wb.epsilon.notna()]
        if len(wb):
            facts["wb_lam"] = (wb.lam_median.min(), wb.lam_median.max())
            facts["wb_enr"] = (wb.enr_median.min(), wb.enr_median.max())
            facts["wb_above"] = (wb.frac_above_area.min(), wb.frac_above_area.max())
            lines.append(f"- {xai}, true-class map, successful white-box attacks with a budget: enrichment "
                         f"{wb.enr_median.min():.2f} to {wb.enr_median.max():.2f} after the attack; the shift closes "
                         f"{100 * wb.lam_median.min():.0f} % to {100 * wb.lam_median.max():.0f} % of the gap to a "
                         f"uniform map (median lam); {100 * wb.frac_above_area.min():.1f} % to "
                         f"{100 * wb.frac_above_area.max():.1f} % of attacked maps put more than the area share "
                         f"outside the ellipse.")
            fgsm = wb[wb.attack == "fgsm"]
            key = _one(fgsm, epsilon=float(fgsm.epsilon.max())) if len(fgsm) else None
            if key is not None:
                facts["key_null"] = (key.enr_median, key.lam_median)
                lam = (100 * key.lam_median, 100 * key.lam_ci_low, 100 * key.lam_ci_high)
                lines.append(f"- FGSM @ {eps_label(key.epsilon)}, {xai}, true-class map, successful: enrichment "
                             f"{_ci(key.enr_median, key.enr_ci_low, key.enr_ci_high, '.3f')} (clean maps of the same "
                             f"images {key.enr_clean_median:.3f}); gap closed {_ci(*lam, '.1f')} % "
                             f"(n = {key.n_fit} maps whose clean mass is at least 0.05 from the area share).")
        nz = _pick(t, condition="noise", xai=xai)
        if len(nz):
            budgets = ", ".join(eps_label(e) for e in sorted(nz.epsilon.dropna()))
            lines.append(f"- Same-budget noise at {budgets}, {xai}: gap closed "
                         f"{_num(100 * nz.lam_median.min(), '+.1f')} % to {_num(100 * nz.lam_median.max(), '+.1f')} %.")
    if area_source is not None:
        fits = t[(t.section != "control") & t.r2.notna()]       # one row per group: t holds one region
        if len(fits):
            lines.append(f"- The null explains R^2 = {fits.r2.min():.2f} to {fits.r2.max():.2f} of the attack-induced "
                         f"shift across {len(fits)} attack groups (chance {fits.r2_chance.iloc[0]:.2f}).")
    lines.append("")
    return facts


def _report_heatmap(table, main_target, order, lines):
    lines += ["## 3. Heatmap shift by target and outcome (heatmap_shift_by_outcome.csv)", ""]
    if not len(table):
        lines += ["No rows.", ""]
        return {}
    lines += ["Spearman correlation of each map with the clean map of the same image (1 = same ranking). Cells: "
              "median (n). Top-10 % intersections and CIs are in the CSV.", ""]
    cols = _map_columns(main_target)
    header = ["attack @ budget"] + [c[3] for c in cols] + [f"noise ({xai})" for xai in main_target]
    body = []
    for section in ("white-box", "occlusion"):
        for attack, eps, placement in _variants(table, section, order):
            cells = [_variant_label(attack, eps, placement)]
            for xai, target, outcome, _ in cols:
                r = _one(table, condition="attack", **_variant(table, attack, eps, placement), xai=xai, target=target,
                         outcome=outcome)
                cells.append("-" if r is None else f"{r.spearman_median:.3f} ({r.n})")
            for xai, target in main_target.items():
                r = _one(table, condition="noise", epsilon=eps, xai=xai, target=target) if eps is not None else None
                cells.append("-" if r is None else f"{r.spearman_median:.3f}")
            body.append(cells)
    lines += _markdown(header, body) + [""]
    facts = {}
    wb = table[(table.condition == "attack") & table.epsilon.notna()]
    tested = wb[wb.p_holm.notna()]
    sig = tested[(tested.p_holm < 0.05) & (tested.diff_noise_minus_attack > 0)]
    facts["sig"] = (len(sig), len(tested))
    facts["fail_sig"] = (int((sig.outcome == "failed").sum()), int((tested.outcome == "failed").sum()))
    if len(tested):
        lines.append(f"- In {len(sig)} of {len(tested)} white-box attack x budget x XAI x target x outcome groups with "
                     f"at least {MIN_N} images, the map moves more than under same-budget noise on the same images "
                     f"(per-subject mean Spearman gap, Wilcoxon, Holm p < 0.05).")
    else:
        lines.append(f"- No white-box attack x budget x XAI x target x outcome group has {MIN_N} images, so none is "
                     f"compared with same-budget noise.")
    for xai in [x for x, tg in main_target.items() if tg == "true"]:
        fail = _pick(wb, xai=xai, target="true", outcome="failed")
        fail = fail[fail.n >= MIN_N]
        for r in fail.itertuples():
            s = _one(wb, attack=r.attack, epsilon=r.epsilon, xai=xai, target="true", outcome="success")
            lines.append(f"- {r.attack} @ {eps_label(r.epsilon)}, {xai}: failed attacks still move the true-class "
                         f"map (Spearman {_ci(r.spearman_median, r.spearman_ci_low, r.spearman_ci_high, '.3f')}, "
                         f"n = {r.n}) against {_num(s.spearman_median if s is not None else np.nan, '.3f')} for "
                         f"successful ones and {_num(r.noise_spearman_median, '.3f')} for noise on the same images "
                         f"(gap to noise {_ci(r.diff_noise_minus_attack, r.diff_ci_low, r.diff_ci_high)}, Holm p "
                         f"{_pv(r.p_holm)}).")
        t_s, t_p = _pick(wb, xai=xai, target="true", outcome="success"), _pick(wb, xai=xai, target="pred")
        if len(t_s) and len(t_p):
            facts["true_vs_pred"] = ((t_s.spearman_median.min(), t_s.spearman_median.max()),
                                     (t_p.spearman_median.min(), t_p.spearman_median.max()))
            lines.append(f"- {xai}, successful white-box attacks with a budget: Spearman of the true-class map "
                         f"{t_s.spearman_median.min():.3f} to {t_s.spearman_median.max():.3f}; of the "
                         f"adversarial-class map (vs the clean true-class map) {t_p.spearman_median.min():.3f} to "
                         f"{t_p.spearman_median.max():.3f}.")
    dc = table[table.condition == "defence_on_clean"]
    if len(dc):
        lines.append("- A defence alone moves the clean maps: " + "; ".join(
            f"{r.defence} {r.xai} {_ci(r.spearman_median, r.spearman_ci_low, r.spearman_ci_high, '.3f')}"
            for r in dc.itertuples()) + " (Spearman with the undefended clean map).")
    lines.append("")
    return facts


def _report_restoration(table, xais, lines, n_boot=None):
    lines += ["## 4. Restoration within conditions (restoration_by_outcome.csv)", ""]
    if not len(table):
        lines += ["No attack+defence rows with vs_defended_clean_spearman.", ""]
        return {}
    lines += ["Among successful attacks: Spearman of the defended map with the defended clean map, for images whose "
              "label the defence restored vs images it did not.", ""]
    facts = {}
    for xai in [x for x in xais if x in set(table.xai)]:
        naive = _one(table, scope="pooled_naive", xai=xai)
        within = _one(table, scope="pooled_within", xai=xai)
        if naive is None or within is None:
            continue
        facts[xai] = (naive["diff"], within["diff"], within.diff_ci_low, within.diff_ci_high)
        lines.append(f"- {xai}, pooled over all attacks, budgets and both defences (how the claim was made so far): "
                     f"restored {_ci(naive.restored_median, naive.restored_ci_low, naive.restored_ci_high, '.3f')} "
                     f"(n = {naive.n_restored}) vs not restored "
                     f"{_ci(naive.not_restored_median, naive.not_restored_ci_low, naive.not_restored_ci_high, '.3f')} "
                     f"(n = {naive.n_not_restored}): difference "
                     f"{_ci(naive['diff'], naive.diff_ci_low, naive.diff_ci_high)}.")
        cond = _pick(table, scope="condition", xai=xai)
        share = within["diff"] / naive["diff"] if naive["diff"] else np.nan
        mix = cond[["restored_rate", "median_all"]].corr(method="spearman").iloc[0, 1] if len(cond) > 2 else np.nan
        mixing = (f"the {len(cond)} conditions, the restoration rate and the condition's median Spearman have a rank "
                  f"correlation of {_num(mix, '+.2f')}"
                  + (" (conditions that are restored more often also keep maps closer to the defended clean map)."
                     if np.isfinite(mix) and mix > 0.3 else "."))
        if within.n_strata:
            text = (f"- {xai}, within conditions (weighted mean of the restored-minus-not differences of "
                    f"{within.n_strata} attack x budget x defence conditions with at least {MIN_N} images in each "
                    f"group): {_ci(within['diff'], within.diff_ci_low, within.diff_ci_high)}, bootstrap p "
                    f"{_boot_pv(within.p, n_boot)}.")
        else:
            text = (f"- {xai}, within conditions: no condition has at least {MIN_N} restored and {MIN_N} not-restored "
                    f"images, so there is no within-condition estimate.")
        if np.isfinite(share):
            text += (f" That is {100 * share:.0f} % of the pooled difference; the rest comes from pooling conditions "
                     f"with different restoration rates: across {mixing}")
        else:
            text += f" Across {mixing}"
        lines.append(text)
        tested = cond[cond.p.notna()]
        sig = tested[tested.p_holm < 0.05]
        if len(tested):
            text = (f"- {xai}, per condition: differences {_num(tested['diff'].min())} to "
                    f"{_num(tested['diff'].max())} in the {len(tested)} conditions with at least {MIN_N} images per "
                    f"group; {len(sig)} of them with Holm p < 0.05 (subject-level permutation test)")
            if len(sig):
                text += ": " + "; ".join(f"{r.attack} @ {eps_label(r.epsilon)} + {r.defence} {_num(r.diff)}"
                                         for r in sig.itertuples())
            lines.append(text + f"; {len(cond) - len(tested)} conditions have too few restored or not-restored "
                                f"images to test.")
        else:
            lines.append(f"- {xai}, per condition: none of the {len(cond)} conditions has at least {MIN_N} restored "
                         f"and {MIN_N} not-restored images, so none is tested.")
        if within.control_n:
            gap = (within.restored_minus_control, within.restored_minus_control_ci_low,
                   within.restored_minus_control_ci_high)
            lines.append(f"- {xai}, matched control (same-budget noise, same defence, vs the defended clean map): "
                         f"restored maps minus control on the same images {_ci(*gap)} (per-subject mean over "
                         f"{within.control_n} pairs, Wilcoxon p {_pv(within.p_vs_control)}).")
    controls = table[table.scope == "noise+defence"]
    if len(controls):
        lines.append("- Matched control, noise+defence vs defended clean (median): " + "; ".join(
            f"{r.defence} @ {eps_label(r.epsilon)} {r.xai} "
            f"{_ci(r.control_median, r.control_ci_low, r.control_ci_high, '.3f')}"
            for r in controls.itertuples()) + ".")
    else:
        facts["no_control"] = True
        lines.append("- **The matched control needs the new runner.** These rows have no 'noise+defence' condition "
                     "(the same defence applied to the same-budget noise image), so we cannot say how close to the "
                     "defended-clean map a harmless perturbation of the same size would bring the map after the same "
                     "defence. The noise reference used so far is noise without any defence, so it is not matched.")
    lines.append("")
    return facts


def report_v2(v2, area_source, checks, rows, args):
    """report_v2.md: every v2 result in sentences with its numbers, and what we can and cannot claim."""
    regions = regions_in(rows.columns)
    region = _focus_region(regions)
    main = rows[rows.target.isin(MAIN)]
    main_target = {x: main[main.xai == x].target.iloc[0] for x in dict.fromkeys(main.xai)}
    lines = ["# Results v2: region shift by outcome, area-baseline null, restoration within conditions", "",
             f"Rows: {os.path.basename(args.experiment_rows)} ({len(rows):,} rows, {rows.image_id.nunique()} images of "
             f"{rows.true_idx.nunique()} subjects). CIs: 95 % subject-cluster bootstrap ({args.n_boot} resamples). "
             f"Tests: Wilcoxon signed-rank on per-subject means unless stated (restoration: subject-level "
             f"permutation, {args.n_perm} permutations), Holm within each table; p below 1e-10 is printed as "
             f"'< 1e-10' (the normal approximation bottoms out far below any meaningful value). Groups with fewer "
             f"than {MIN_N} images get point estimates only. True-class map = the class-specific method explaining "
             f"the true identity; adversarial-class map = the same method explaining the class the attacked image "
             f"is assigned to (only for successful attacks); success = the attack flipped the prediction.", "",
             "## Checks that passed", ""] + [f"- {c}" for c in checks] + [""]
    order = list(dict.fromkeys(rows.attack.dropna())) if "attack" in rows else []
    shift = _report_region_shift(v2["region_shift_v2"], rows, region, main_target, order, lines)
    null = _report_null(v2["region_null"], area_source, region, main_target, order, lines)
    heat = _report_heatmap(v2["heatmap_shift_by_outcome"], main_target, order, lines)
    restore = _report_restoration(v2["restoration_by_outcome"], list(main_target), lines, args.n_boot)

    # A claim goes under "Can claim" only when the numbers support it in every tested condition;
    # otherwise the same numbers go under "Cannot claim".
    can, cannot = [], []
    where = "outside the central ellipse" if region == "periocular" else f"to {region}"
    if "key" in shift and shift["succ_sig"][1]:
        key, eps, s, p, f = shift["key"]
        (n_sig, n_cond), (p_sig, p_cond) = shift["succ_sig"], shift["pred_sig"]
        n_more, n_pairs = shift["true_vs_pred"]
        text = (f"{key.upper()} @ {eps_label(eps)} moves {_num(s)} of the true-class map {where} (median), while the "
                f"adversarial-class map of the same images moves {_num(p)} and the true-class map after a failed "
                f"attack {_num(f)}. Across white-box attacks with a budget, the true-class shift exceeds same-budget "
                f"noise in {n_sig} of {n_cond} conditions")
        if p_cond:
            text += (f", the adversarial-class shift in {p_sig} of {p_cond}, and on the same images the true-class map "
                     f"shifts more than the adversarial-class map in {n_more} of {n_pairs} conditions")
        text += "."
        if np.isfinite(s) and s > 0 and n_cond and n_sig == n_cond:
            (s_lo, _), (_, p_hi) = shift["succ_range"], shift["pred_range"]
            if np.isfinite(p_hi) and np.isfinite(s_lo) and p_hi < 0.5 * s_lo:
                text += (f" The adversarial-class shift (at most {p_hi:+.3f}) stays below half the smallest "
                         f"true-class shift ({s_lo:+.3f}): the evidence for the wrong identity stays about where the "
                         f"clean evidence was.")
            can.append("Successful attacks make the true-class evidence less centred on the eye: " + text)
        else:
            cannot.append("'Successful attacks make the true-class evidence less centred on the eye' in general: "
                          + text)
    if "sig" in heat and heat["sig"][1]:
        (n_sig, n_tested), (f_sig, f_tested) = heat["sig"], heat["fail_sig"]
        if n_sig == n_tested:
            can.append(f"Attacks move heatmaps more than same-budget noise ({n_sig} of {n_tested} tested white-box "
                       f"groups, by Spearman with the clean map)"
                       + (", including attacks that failed to flip the prediction." if f_tested and f_sig == f_tested
                          else "."))
        else:
            cannot.append(f"'Attacks move heatmaps more than same-budget noise' for every attack: only {n_sig} of "
                          f"{n_tested} tested white-box groups do (by Spearman with the clean map).")
    shares = {}
    for xai, value in restore.items():
        if xai == "no_control" or not np.isfinite(value).all():
            continue
        naive, within, lo, hi = value
        shares[xai] = within / naive if naive else np.nan
        verdict = ("closer to the defended-clean map than not-restored ones" if lo > 0 else
                   "not detectably closer to the defended-clean map than not-restored ones" if hi >= 0 else
                   "further from the defended-clean map than not-restored ones")
        can.append(f"Restoration, {xai}: within conditions, label-restored maps are {verdict} (Spearman difference "
                   f"{within:+.3f} [{lo:+.3f}, {hi:+.3f}]); pooled over conditions the difference is {naive:+.3f}, so "
                   f"{100 * shares[xai]:.0f} % of it survives the within-condition comparison.")
    if region == "periocular":
        detail = ""
        if "wb_lam" in null:
            detail = (f" After successful white-box attacks the true-class map still puts only "
                      f"{null['wb_enr'][0]:.2f} to {null['wb_enr'][1]:.2f} times the uniform map's mass outside the "
                      f"ellipse, and the shift closes {100 * null['wb_lam'][0]:.0f} % to "
                      f"{100 * null['wb_lam'][1]:.0f} % of the gap to a uniform map.")
        cannot.append("'Attention moves from the eye to the periocular skin.' The periocular region is the outside of "
                      "one fixed ellipse (69.25 % of every image), not segmented skin, and a map that only spreads out "
                      "gains mass there without attending to anything." + detail + " Say instead: the true-class "
                      "evidence loses its eye-centred structure, while the adversarial-class evidence stays on the "
                      "eye.")
    if regions and area_source is None:
        if set(regions) == set(HEURISTIC_REGIONS):
            cannot.append("Pupil, iris or sclera statements, or 'more than diffusion' for any region: the masks are "
                          "unvalidated heuristics, and the 4-region area null needs per-image areas (the data-check "
                          "CSV via --areas, or rows from the v2 runner).")
        else:
            cannot.append(f"'More than diffusion' for any region ({', '.join(regions)}): the area-baseline null needs "
                          f"per-image areas (area_<region> columns from the v2 runner, or --areas).")
    shifts = v2["region_shift_v2"]
    if len(shifts) and (shifts.section == "occlusion").any():
        separated = (shifts.measure == "dmassvis").any()
        cannot.append("Occlusion (patch, monocle) region shifts as attention effects: the occluder covers part of the "
                      "eye, so part of the shift is mechanical. Report them separately; "
                      + ("dmassvis in region_shift_v2.csv gives the shift over the pixels the occluder left alone."
                         if separated else
                         "separating the two needs dmassvis (mass over non-occluded pixels) from the v2 runner."))
    finite = [s for s in shares.values() if np.isfinite(s)]
    if finite:
        text = ("'Right answer for the wrong reasons' as stated so far (pooled restored maps against the noise "
                "reference): " + ("most" if max(finite) < 0.5 else "part") + " of the pooled restored-vs-not gap "
                "comes from which attacks get restored")
        text += (", and the matched noise+defence control does not exist yet (needs the v2 runner), so the remaining "
                 "distance to the defended-clean map cannot be compared with what a harmless perturbation of the same "
                 "size causes." if restore.get("no_control") else ".")
        cannot.append(text)
    none = ["Nothing from these rows (too few images per group, or no columns for it)."]
    lines += ["## What we can and cannot claim", "", "Can claim:", ""] + [f"- {c}" for c in can or none] + \
             ["", "Cannot claim:", ""] + [f"- {c}" for c in cannot or none] + [""]
    return "\n".join(lines)


def _sibling(rows_path, prefix, ext):
    """images_<tag>.csv or summary_<tag>.json next to rows_<tag>.csv, if it exists."""
    folder, name = os.path.split(rows_path)
    if not (name.startswith("rows_") and name.endswith(".csv")):
        return None
    path = os.path.join(folder, f"{prefix}{name[len('rows_'):-len('.csv')]}{ext}")
    return path if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-rows", default=None, help="rows_<tag>.csv from 05_run_experiment.py")
    ap.add_argument("--detection", default=None, help="features_<tag>.csv from 11_run_detection.py")
    ap.add_argument("--areas", default=None,
                    help="CSV of per-image region areas (relpath and/or link_path, area_<region> for each region) "
                         "for rows without area_<region> columns; makes the area-baseline null exact for every "
                         "region instead of the periocular region only")
    ap.add_argument("--images", default=None,
                    help="images_<tag>.csv of the run, to join --areas by link_path when the rows have no relpath "
                         "(default: next to the rows file)")
    ap.add_argument("--summary", default=None,
                    help="summary_<tag>.json of the run; its attack-success and restoration counts must match the "
                         "rows (default: next to the rows file; skipped if there is none)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for every CI (use 100, with --n-perm 99, for a quick trial run)")
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="permutations of the subject-level permutation test (its smallest p is 1 / (n-perm + 1), "
                         "which must stay well below 0.05 / the number of tests for Holm to resolve them)")
    ap.add_argument("--out-dir", default="analysis")
    args = ap.parse_args()
    if not (args.experiment_rows or args.detection):
        ap.error("give --experiment-rows and/or --detection")
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    t0 = time.time()

    def done(what):
        print(f"  {what} ({time.time() - t0:.0f} s)", flush=True)

    shift = restored = regions = aurocs = versus = pd.DataFrame()
    if args.experiment_rows:
        rows = pd.read_csv(args.experiment_rows)
        summary_path = args.summary or _sibling(args.experiment_rows, "summary_", ".json")
        summary = None
        if summary_path:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        checks = check_rows(rows, summary)
        print(f"{len(rows):,} rows; summary JSON: {summary_path or 'none'}")
        for c in checks:
            print("  check passed:", c)
        shift, restored, regions = (heatmap_shift(rows, args.n_boot), restoration(rows, args.n_boot),
                                    region_shift(rows, args.n_boot))
        shift.to_csv(out("heatmap_shift.csv"), index=False)
        restored.to_csv(out("restoration.csv"), index=False)
        if len(regions):
            regions.to_csv(out("region_shift.csv"), index=False)
        plot_heatmap_shift(shift, out("fig_heatmap_shift.png"))
        plot_regions(regions, out("fig_regions.png"))
        done("heatmap_shift, restoration, region_shift")

        areas = images = None
        if args.areas:
            areas = pd.read_csv(args.areas)
            images_path = args.images or _sibling(args.experiment_rows, "images_", ".csv")
            images = pd.read_csv(images_path) if images_path else None
        v2 = {"region_shift_v2": region_shift_v2(rows, args.n_boot)}
        done("region_shift_v2")
        v2["region_null"], area_source = region_null(rows, args.n_boot, areas, images)
        done(f"region_null (areas: {area_source or 'periocular only, exact'})")
        v2["heatmap_shift_by_outcome"] = heatmap_shift_by_outcome(rows, args.n_boot)
        done("heatmap_shift_by_outcome")
        v2["restoration_by_outcome"] = restoration_by_outcome(rows, args.n_boot, args.n_perm)
        done("restoration_by_outcome")
        for name, table in v2.items():
            if len(table):
                table.to_csv(out(f"{name}.csv"), index=False)
        text = report_v2(v2, area_source, checks, rows, args)
        with open(out("report_v2.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
    if args.detection:
        aurocs, versus = detection(pd.read_csv(args.detection), args.n_boot)
        aurocs.to_csv(out("detection_auroc.csv"), index=False)
        versus.to_csv(out("detection_vs_fs.csv"), index=False)
        plot_detection(aurocs, out("fig_detection.png"))

    text = report(shift, restored, regions, aurocs, versus, args.n_boot)
    with open(out("report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
