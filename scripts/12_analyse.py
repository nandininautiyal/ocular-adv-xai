"""
scripts/12_analyse.py

Turn the Kaggle outputs into the paper's statistics and figures. Runs on a laptop
(CPU, no model needed) from the downloaded CSVs; either input is optional.

  --experiment-rows   rows_<tag>.csv from 05_run_experiment.py
  --detection         features_<tag>.csv from 11_run_detection.py

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

Usage:
  python scripts/12_analyse.py --experiment-rows rows_full.csv --detection features_full.csv --out-dir analysis
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.analysis.stats import (auroc_ci, cluster_bootstrap_ci, holm,  # noqa: E402
                                paired_auroc_difference, paired_subject_test)

REGIONS = ("pupil", "iris", "sclera", "periocular")
FUSED = ("fused_explanation", "fused_feature_squeezing", "fused_prediction", "fused_all")


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
               "attack_success_rate": float((~g.correct.astype(bool)).mean()),
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
        g = g[g.attack_success.astype(bool)] if "attack_success" in g else g
        base = attacked[(attacked.attack == attack) & (attacked.xai == xai) & _same_eps(attacked.epsilon, eps)]
        m = g.merge(base[["image_id", "target", "spearman"]], on=["image_id", "target"], suffixes=("", "_attacked"))
        if not len(m):
            continue
        est, lo, hi = cluster_bootstrap_ci(m.vs_defended_clean_spearman, m.true_idx, n_boot=n_boot)
        test = paired_subject_test(m.vs_defended_clean_spearman, m.spearman_attacked, m.true_idx)
        out.append({"attack": attack, "epsilon": eps, "defence": defence, "xai": xai, "n_successful": len(m),
                    "prediction_restored_rate": float(m.correct.astype(bool).mean()),
                    "restored_spearman_median": est, "ci_low": lo, "ci_high": hi,
                    "attacked_spearman_median": float(m.spearman_attacked.median()),
                    "median_gain": test["median_diff"], "effect_rank_biserial": test["rank_biserial"],
                    "p": test["p"]})
    table = pd.DataFrame(out)
    if len(table):
        table["p_holm"] = holm(table.p)
    return table


def region_shift(rows, n_boot):
    if f"dmass_{REGIONS[0]}" not in rows:
        return pd.DataFrame()
    attacked = rows[(rows.condition == "attack") & rows.target.isin(["true", "any"])]
    out = []
    for (attack, eps, xai), g in attacked.groupby(["attack", "epsilon", "xai"], dropna=False):
        for region in REGIONS:
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
    adv = features[(features.kind == "adversarial") & features.success.astype(bool)]
    scores = [c for c in features.columns if c.startswith(("fused_", "xc_", "fs_l1_", "pred_cos_"))]
    groups = [("all", adv)] + [(a, g) for a, g in adv.groupby("attack")] + \
             [(v, g) for v, g in adv.groupby("variant") if v not in set(adv.attack)]
    aurocs, versus = [], []
    for name, g in groups:
        for s in scores:
            est, lo, hi = auroc_ci(clean[s], g[s], clean.true_idx, g.true_idx, n_boot=n_boot)
            aurocs.append({"group": name, "score": s, "n_adv": len(g), "auroc": est, "ci_low": lo, "ci_high": hi})
        for ours in ("fused_explanation", "fused_all"):
            if ours in g and "fused_feature_squeezing" in g:
                d = paired_auroc_difference(clean[ours], g[ours], clean.fused_feature_squeezing,
                                            g.fused_feature_squeezing, clean.true_idx, g.true_idx, n_boot=n_boot)
                versus.append({"group": name, "ours": ours, "baseline": "fused_feature_squeezing",
                               "n_adv": len(g), **d})
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
              "fused_prediction": "logit consistency", "fused_all": "combined (ours)"}
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
    xais = sorted(t.xai.unique())
    fig, axes = plt.subplots(1, len(xais), figsize=(6 * len(xais), 0.35 * t.variant.nunique() + 1.5), squeeze=False)
    for ax, xai in zip(axes[0], xais):
        grid = t[t.xai == xai].pivot_table(index="variant", columns="region", values="dmass_median")[list(REGIONS)]
        lim = max(1e-6, np.nanmax(np.abs(grid.values)))
        im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels(REGIONS)
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels(grid.index, fontsize=7)
        ax.set_title(f"{xai}: change in heatmap mass")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def report(shift, restored, regions, aurocs, versus):
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
            lines.append(f"- Region shift: {r.attack} ({r.xai}) changes {r.region} mass by {r.dmass_median:+.3f} "
                         f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}].")
    if len(aurocs):
        pooled = aurocs[(aurocs.group == "all") & aurocs.score.isin(FUSED)]
        for r in pooled.itertuples():
            lines.append(f"- Detection AUROC, all successful attacks, {r.score}: {r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}].")
    if len(versus):
        for r in versus[versus.group == "all"].itertuples():
            verdict = "better than" if r.low > 0 else ("worse than" if r.high < 0 else "not significantly different from")
            lines.append(f"- {r.ours} vs Feature Squeezing: {r.diff:+.3f} AUROC [{r.low:+.3f}, {r.high:+.3f}], "
                         f"p = {r.p:.3g}: {verdict} the baseline.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-rows", default=None, help="rows_<tag>.csv from 05_run_experiment.py")
    ap.add_argument("--detection", default=None, help="features_<tag>.csv from 11_run_detection.py")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default="analysis")
    args = ap.parse_args()
    if not (args.experiment_rows or args.detection):
        ap.error("give --experiment-rows and/or --detection")
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)  # noqa: E731

    shift = restored = regions = aurocs = versus = pd.DataFrame()
    if args.experiment_rows:
        rows = pd.read_csv(args.experiment_rows)
        shift, restored, regions = (heatmap_shift(rows, args.n_boot), restoration(rows, args.n_boot),
                                    region_shift(rows, args.n_boot))
        shift.to_csv(out("heatmap_shift.csv"), index=False)
        restored.to_csv(out("restoration.csv"), index=False)
        if len(regions):
            regions.to_csv(out("region_shift.csv"), index=False)
        plot_heatmap_shift(shift, out("fig_heatmap_shift.png"))
        plot_regions(regions, out("fig_regions.png"))
    if args.detection:
        aurocs, versus = detection(pd.read_csv(args.detection), args.n_boot)
        aurocs.to_csv(out("detection_auroc.csv"), index=False)
        versus.to_csv(out("detection_vs_fs.csv"), index=False)
        plot_detection(aurocs, out("fig_detection.png"))

    text = report(shift, restored, regions, aurocs, versus)
    with open(out("report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
