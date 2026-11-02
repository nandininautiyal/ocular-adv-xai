"""
scripts/24_gradient_masking.py

Gradient-masking check of a checkpoint: do gradient attacks fail because the
model is robust, or because its gradients are uninformative? A robustness
difference between an attention model and the baseline counts only if this
check passes on both (Athalye et al., ICML 2018; Carlini et al. 2019, "On
Evaluating Adversarial Robustness"; IEEE 10658195 for the attention softmax).

Run it on Kaggle next to the checkpoint, after 04_eval_clean.py. Images are
sampled exactly like 14_adaptive_attacks.py (same --correct-csv, --per-class,
--seed, --exclude-from, --max-images), so with the same flags the two scripts
attack the same images. Every attacked image is rounded to 8 bits within whole
levels of its budget (src.attacks.budget.quantise), and success means the
unscaled model (attention temperature 1) misclassifies it.

Tests, per image:
  (a) PGD success against the number of steps (--steps-grid, default 10, 50,
      200; at each budget of --steps-levels) and against the budget in levels
      (--budget-grid, default 1, 2, 4, 8, 16, 32, 255; --budget-steps steps).
      Success must not fall when steps or budget grow (beyond --monotone-tol)
      and must reach --unbounded-min (0.99) at 255 levels, the whole [0, 1]
      range.
  (b) pgd vs aas_pgd (attention-temperature scaling, src/attacks/masking.py,
      temperatures --taus) vs random-restart PGD (--restarts starts; restart 0
      is the pgd run, so it can only add successes) at each budget of
      --levels with --steps steps. aas_pgd or the restarts beating pgd by more
      than --gain-tol means pgd's gradient was misleading.
  (c) Input-gradient statistics of the margin loss on the clean images and on
      the pgd images of (b): the share of exactly zero and of non-finite
      gradient components, the L1 / L2 / L-inf norms and, for an attention
      model, the L2 norm at every temperature of --taus. Attention saturation
      on the same images: the largest weight and the normalised entropy (1 =
      uniform over the 64 cells, 0 = one cell) of every attention map.

masking_flags in the summary lists every test that failed:
  pgd_steps_not_monotone@<L>levels, pgd_budget_not_monotone,
  unbounded_pgd_below_<min>, aas_pgd_beats_pgd@<L>levels,
  random_restarts_beat_pgd@<L>levels, nonfinite_gradients@<condition>,
  all_zero_gradient_images@<condition>, zero_gradients@<condition> (median
  share of zero components above --zero-grad-max), attention_saturated@<condition>
  (median largest attention weight above --saturation-max).
An empty list means no sign of gradient masking was found; checks_skipped
names the tests the chosen grids could not run (e.g. no 255 in --budget-grid).

Writes to --out-dir:
  rows_masking_<tag>.csv      one row per image x test x setting (test = steps,
                              budget, compare or gradient)
  summary_masking_<tag>.json  success rates with 95 % Wilson intervals, the
                              monotonicity checks, gradient and attention
                              statistics, aas_pgd's temperature use,
                              masking_flags, timings, settings and the git commit

Refuses three-input models (attnet_gate: its region input is not an image).

Usage (Kaggle, with the checkpoint dataset attached and 01, 02, 04 already run):
  smoke (2-3 min; prints the ETA of the full run's per-image cost):
    python scripts/24_gradient_masking.py --ckpt <checkpoint.keras> \\
        --correct-csv /kaggle/working/eval_clean/correct_test_<tag>.csv --max-images 4 --tag smoke
  full (one image per class):
    python scripts/24_gradient_masking.py --ckpt <checkpoint.keras> \\
        --correct-csv /kaggle/working/eval_clean/correct_test_<tag>.csv --per-class 1 \\
        --exclude-from /kaggle/working/masking/rows_masking_smoke.csv --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.budget import levels_to_epsilon, quantise  # noqa: E402
from src.attacks.masking import (DEFAULT_TAUS, aas_pgd, attention_stats, check_taus, gradient_stats,  # noqa: E402
                                 has_attention)
from src.attacks.whitebox import pgd  # noqa: E402
from src.models.anatomy_model import model_arch  # noqa: E402
from src.models.arcface_inference import load_checkpoint, predict_logits  # noqa: E402


def _load_script(name):
    """A numbered script as a module (its main() does not run)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ADAPTIVE = _load_script("14_adaptive_attacks")      # sampling, image loading, margins, rates, JSON helpers

BASE_COLS = _ADAPTIVE.BASE_COLS
COMPARE_ATTACKS = ("pgd", "aas_pgd", "rr_pgd")
RESTART_SEED_STRIDE = 1_000_003      # restart r starts from seed + r * stride (r = 0 is the pgd run)
DEFAULTS = {"levels": (1, 2, 4), "steps": 20, "steps_grid": (10, 50, 200), "steps_levels": (1, 2, 4),
            "budget_grid": (1, 2, 4, 8, 16, 32, 255), "budget_steps": 50, "restarts": 5, "taus": DEFAULT_TAUS,
            "step_size": None, "monotone_tol": 0.01, "unbounded_min": 0.99, "gain_tol": 0.05,
            "zero_grad_max": 0.5, "saturation_max": 0.9}


def settings_from(**overrides):
    """The settings of a run: DEFAULTS updated with *overrides* (unknown keys raise)."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    return {**DEFAULTS, **overrides}


# ---------------------------------------------------------------------------
# Attacks and rows
# ---------------------------------------------------------------------------

def pgd_8bit(model, clean, labels, levels, steps, seed, step_size=None):
    """PGD at *levels* whole 8-bit levels, rounded to the 8-bit grid within the budget."""
    eps = levels_to_epsilon(levels)
    return quantise(pgd(clean, model, eps, label=labels, steps=steps, step_size=step_size, seed=seed), clean, eps)


def outcome(model, clean, adv, labels):
    """Prediction, margin (true logit minus best other; negative = fooled) and L-inf size in levels."""
    logits = predict_logits(model, adv)
    linf = np.abs(adv - clean).reshape(len(adv), -1).max(axis=1)
    return logits.argmax(1), _ADAPTIVE.margins(logits, labels), np.round(linf * 255.0, 3)


def attack_rows(model, clean, adv, labels, base_rows, info, extra=None):
    """One row per image: *info* + success, margin and size, judged on the unscaled model."""
    pred, margin, linf_levels = outcome(model, clean, adv, labels)
    rows = []
    for i in range(len(adv)):
        row = {**base_rows[i], **info, "pred": int(pred[i]), "success": bool(pred[i] != labels[i]),
               "margin": float(margin[i]), "pert_linf_levels": float(linf_levels[i])}
        for key, values in (extra or {}).items():
            row[key] = float(values[i])
        rows.append(row)
    return rows


def random_restart_pgd(model, clean, labels, levels, steps, seed, restarts, first=None, step_size=None):
    """PGD from *restarts* random starts; per image the 8-bit result with the lowest margin.

    Restart r uses seed + r * RESTART_SEED_STRIDE, so restart 0 is the pgd run with *seed*
    (pass it as *first* to reuse it). Returns (images, restarts that fooled the model (N,)).
    """
    best, best_margin, fooled = None, None, np.zeros(len(clean), int)
    for r in range(restarts):
        adv = first if (r == 0 and first is not None) else pgd_8bit(
            model, clean, labels, levels, steps, seed + r * RESTART_SEED_STRIDE, step_size)
        logits = predict_logits(model, adv)
        margin = _ADAPTIVE.margins(logits, labels)
        fooled += logits.argmax(1) != labels
        if best is None:
            best, best_margin = adv.copy(), margin
        else:
            better = margin < best_margin
            best[better], best_margin = adv[better], np.where(better, margin, best_margin)
    return best, fooled


def stats_rows(model, images, labels, base_rows, condition, taus, map_names):
    """Gradient and attention statistics of *images* (the test = "gradient" rows)."""
    grads = gradient_stats(model, images, labels, taus=taus)
    attn = attention_stats(model, images)
    rows = []
    for i in range(len(images)):
        row = {**base_rows[i], "test": "gradient", "attack": "none" if condition == "clean" else "pgd",
               "condition": condition,
               "budget_levels": 0 if condition == "clean" else int(condition.split("@")[1])}
        row.update({f"grad_{k}": float(v[i]) for k, v in grads.items()})
        if attn is not None:
            max_weight, entropy = attn
            row["attn_max_weight"] = float(max_weight[i].max())
            row["attn_entropy_norm"] = float(entropy[i].mean())
            for k, name in enumerate(map_names):
                row[f"attn_max_{name}"] = float(max_weight[i, k])
                row[f"attn_entropy_{name}"] = float(entropy[i, k])
        rows.append(row)
    return rows


def _timed(timings, key, fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - start
    return out


def run_batch(model, clean, labels, base_rows, settings, seed=0, timings=None):
    """Tests (a), (b) and (c) on one batch of correctly classified 8-bit images.

    Parameters
    ----------
    model : tf.keras.Model with dual inputs [image, label]
    clean : ndarray (N, 256, 256, 3), float32, [0, 1], on the 8-bit grid
    labels : ndarray (N,) — true classes
    base_rows : list of N dicts — per-image columns copied into every row
    settings : dict from :func:`settings_from`
    seed : random-start seed of every PGD run (restarts add multiples of RESTART_SEED_STRIDE)
    timings : dict that collects seconds per test

    Returns
    -------
    rows : list of dicts
    """
    timings = {} if timings is None else timings
    s = settings
    attention = has_attention(model)
    taus = tuple(s["taus"])
    map_names = list(model.get_layer("attention_pool").map_names) if attention else []
    taus_text = ",".join(f"{t:g}" for t in taus)
    rows = []

    for levels in s["steps_levels"]:                                      # (a) steps
        for steps in s["steps_grid"]:
            adv = _timed(timings, "steps", pgd_8bit, model, clean, labels, levels, steps, seed, s["step_size"])
            rows += attack_rows(model, clean, adv, labels, base_rows,
                                {"test": "steps", "attack": "pgd", "budget_levels": levels, "steps": steps})
    for levels in s["budget_grid"]:                                       # (a) budget
        adv = _timed(timings, "budget", pgd_8bit, model, clean, labels, levels, s["budget_steps"], seed,
                     s["step_size"])
        rows += attack_rows(model, clean, adv, labels, base_rows,
                            {"test": "budget", "attack": "pgd", "budget_levels": levels, "steps": s["budget_steps"]})

    adversarial = {}
    for levels in s["levels"]:                                            # (b) compare
        eps = levels_to_epsilon(levels)
        common = {"test": "compare", "budget_levels": levels, "steps": s["steps"]}
        adv = _timed(timings, "pgd", pgd_8bit, model, clean, labels, levels, s["steps"], seed, s["step_size"])
        adversarial[levels] = adv
        rows += attack_rows(model, clean, adv, labels, base_rows, {**common, "attack": "pgd"})

        scaled, info = _timed(timings, "aas_pgd", aas_pgd, clean, model, eps, label=labels, steps=s["steps"],
                              step_size=s["step_size"], seed=seed, taus=taus, return_info=True)
        chosen = info["taus"]                                              # (steps, N)
        rows += attack_rows(model, clean, quantise(scaled, clean, eps), labels, base_rows,
                            {**common, "attack": "aas_pgd", "taus": taus_text},
                            {"tau_mean": chosen.mean(axis=0), "tau_not1_share": (chosen != 1.0).mean(axis=0)})

        best, fooled = _timed(timings, "rr_pgd", random_restart_pgd, model, clean, labels, levels, s["steps"], seed,
                              s["restarts"], first=adv, step_size=s["step_size"])
        rows += attack_rows(model, clean, best, labels, base_rows,
                            {**common, "attack": "rr_pgd", "restarts": s["restarts"]}, {"restarts_fooled": fooled})

    start = time.perf_counter()                                           # (c) gradients and attention
    rows += stats_rows(model, clean, labels, base_rows, "clean", taus if attention else None, map_names)
    for levels, adv in adversarial.items():
        rows += stats_rows(model, adv, labels, base_rows, f"pgd@{levels}", taus if attention else None, map_names)
    timings["gradient"] = timings.get("gradient", 0.0) + time.perf_counter() - start
    return rows


# ---------------------------------------------------------------------------
# Summary and flags
# ---------------------------------------------------------------------------

def drops(rates, tol):
    """Consecutive pairs (a, b) of an ordered {setting: rate} where the rate falls by more than *tol*."""
    keys = list(rates)
    return [(a, b) for a, b in zip(keys, keys[1:]) if rates[b] < rates[a] - tol]


def _median(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def summarise(rows, settings):
    """Rates, monotonicity, gradient / attention statistics and the masking flags of a run."""
    s = settings
    rate = _ADAPTIVE._rate
    out = {"pgd_success_vs_steps": {}, "pgd_success_vs_budget": {}, "attack_comparison": {},
           "aas_tau_use": {}, "gradient_stats": {}, "attention_stats": {}}
    steps = rows[rows.test == "steps"]
    for levels in s["steps_levels"]:
        g = steps[steps.budget_levels == levels]
        out["pgd_success_vs_steps"][f"{levels}levels"] = {
            str(n): rate(g.success[g.steps == n]) for n in s["steps_grid"]}
    budget = rows[rows.test == "budget"]
    out["pgd_success_vs_budget"] = {str(L): rate(budget.success[budget.budget_levels == L]) for L in s["budget_grid"]}
    compare = rows[rows.test == "compare"]
    for levels in s["levels"]:
        for attack in COMPARE_ATTACKS:
            g = compare[(compare.budget_levels == levels) & (compare.attack == attack)]
            out["attack_comparison"][f"{attack}@{levels}levels"] = rate(g.success)
        g = compare[(compare.budget_levels == levels) & (compare.attack == "aas_pgd")]
        out["aas_tau_use"][f"{levels}levels"] = {"median_mean_tau": _median(g.tau_mean),
                                                 "median_share_of_steps_tau_not_1": _median(g.tau_not1_share)}
    gradient = rows[rows.test == "gradient"]
    for condition, g in gradient.groupby("condition", sort=False):
        stats = {"n": int(len(g)), "median_zero_share": _median(g.grad_zero_share),
                 "max_nonfinite_share": float(g.grad_nonfinite_share.max()),
                 "images_all_zero_gradient": int((g.grad_l1 == 0).sum()),
                 "median_l1": _median(g.grad_l1), "median_l2": _median(g.grad_l2), "median_linf": _median(g.grad_linf)}
        stats.update({f"median_{c[len('grad_'):]}": _median(g[c]) for c in g.columns if c.startswith("grad_l2_tau")})
        out["gradient_stats"][condition] = stats
        if "attn_max_weight" in g.columns and g.attn_max_weight.notna().any():
            out["attention_stats"][condition] = {
                "median_max_weight": _median(g.attn_max_weight), "median_entropy_norm": _median(g.attn_entropy_norm),
                "share_max_weight_above_saturation": float((g.attn_max_weight > s["saturation_max"]).mean()),
                **{f"median_{c}": _median(g[c]) for c in g.columns
                   if c.startswith(("attn_max_", "attn_entropy_")) and c not in ("attn_max_weight", "attn_entropy_norm")}}
    flags, skipped = masking_flags(out, s)
    out["monotonicity"] = {
        "tolerance": s["monotone_tol"],
        "steps_drops": {k: drops({n: r["rate"] for n, r in v.items()}, s["monotone_tol"])
                        for k, v in out["pgd_success_vs_steps"].items()},
        "budget_drops": drops({L: r["rate"] for L, r in out["pgd_success_vs_budget"].items()}, s["monotone_tol"])}
    out["masking_flags"] = flags
    out["checks_skipped"] = skipped
    out["masking_check_passed"] = not flags
    return out


def masking_flags(summary, settings):
    """(flags, skipped checks) from the rates and statistics of :func:`summarise`."""
    s = settings
    flags, skipped = [], []
    for key, rates in summary["pgd_success_vs_steps"].items():
        if drops({n: r["rate"] for n, r in rates.items()}, s["monotone_tol"]):
            flags.append(f"pgd_steps_not_monotone@{key}")
    if len(summary["pgd_success_vs_steps"]) == 0 or len(s["steps_grid"]) < 2:
        skipped.append("steps_monotonicity")
    budget = {L: r["rate"] for L, r in summary["pgd_success_vs_budget"].items()}
    if len(budget) >= 2:
        if drops(budget, s["monotone_tol"]):
            flags.append("pgd_budget_not_monotone")
    else:
        skipped.append("budget_monotonicity")
    if "255" in budget:
        if not budget["255"] >= s["unbounded_min"]:
            flags.append(f"unbounded_pgd_below_{s['unbounded_min']:g}")
    else:
        skipped.append("unbounded_budget (add 255 to --budget-grid)")
    comparison = summary["attack_comparison"]
    for levels in s["levels"]:
        base = comparison.get(f"pgd@{levels}levels", {}).get("rate", float("nan"))
        for attack, flag in (("aas_pgd", "aas_pgd_beats_pgd"), ("rr_pgd", "random_restarts_beat_pgd")):
            other = comparison.get(f"{attack}@{levels}levels", {}).get("rate", float("nan"))
            if other > base + s["gain_tol"]:
                flags.append(f"{flag}@{levels}levels")
    for condition, g in summary["gradient_stats"].items():
        if g["max_nonfinite_share"] > 0:
            flags.append(f"nonfinite_gradients@{condition}")
        if g["images_all_zero_gradient"] > 0:
            flags.append(f"all_zero_gradient_images@{condition}")
        if g["median_zero_share"] > s["zero_grad_max"]:
            flags.append(f"zero_gradients@{condition}")
    for condition, a in summary["attention_stats"].items():
        if a["median_max_weight"] > s["saturation_max"]:
            flags.append(f"attention_saturated@{condition}")
    return flags, skipped


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_levels(ap, text, flag):
    """Whole numbers of 8-bit levels from 1 to 255, in the order given (errors via ap.error)."""
    try:
        values = tuple(int(v) for v in text.split(",") if v.strip())
    except ValueError:
        values = ()
    if not values or not all(1 <= v <= 255 for v in values) or len(set(values)) != len(values):
        ap.error(f"{flag} {text}: give distinct whole numbers of 8-bit levels from 1 to 255, e.g. 1,2,4")
    return values


def parse_positive_ints(ap, text, flag):
    try:
        values = tuple(int(v) for v in text.split(",") if v.strip())
    except ValueError:
        values = ()
    if not values or min(values) < 1 or len(set(values)) != len(values):
        ap.error(f"{flag} {text}: give distinct whole numbers of at least 1, e.g. 10,50,200")
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained .keras checkpoint")
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=1, help="images per class (as in 14)")
    ap.add_argument("--max-images", type=int, default=None, help="smoke run: cap the number of images (e.g. 4)")
    ap.add_argument("--exclude-from", default=None,
                    help="CSV with a link_path column (e.g. rows_masking_smoke.csv); those images are not used")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed (as in 14); PGD's random start of batch b uses seed + its first index")
    ap.add_argument("--levels", default="1,2,4", help="budgets in 8-bit levels of the attack comparison (b) and (c)")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"], help="steps of the attacks of (b), as 14's")
    ap.add_argument("--steps-grid", default="10,50,200", help="PGD steps of the steps test (a)")
    ap.add_argument("--steps-levels", default="1,2,4", help="budgets in levels at which the steps test runs")
    ap.add_argument("--budget-grid", default="1,2,4,8,16,32,255",
                    help="budgets in levels of the budget test (a); 255 is the unbounded attack")
    ap.add_argument("--budget-steps", type=int, default=DEFAULTS["budget_steps"], help="PGD steps of the budget test")
    ap.add_argument("--restarts", type=int, default=DEFAULTS["restarts"],
                    help="random starts of rr_pgd (restart 0 is the pgd run)")
    ap.add_argument("--taus", default=",".join(f"{t:g}" for t in DEFAULT_TAUS),
                    help="attention temperatures of aas_pgd and of the gradient norms")
    ap.add_argument("--step-size", type=float, default=None, help="default: 2.5 * epsilon / steps (as in 14)")
    ap.add_argument("--monotone-tol", type=float, default=DEFAULTS["monotone_tol"],
                    help="largest fall in success rate between neighbouring settings that is not flagged")
    ap.add_argument("--unbounded-min", type=float, default=DEFAULTS["unbounded_min"],
                    help="success that PGD must reach at 255 levels")
    ap.add_argument("--gain-tol", type=float, default=DEFAULTS["gain_tol"],
                    help="largest gain of aas_pgd or random restarts over pgd that is not flagged")
    ap.add_argument("--zero-grad-max", type=float, default=DEFAULTS["zero_grad_max"],
                    help="largest median share of exactly zero gradient components that is not flagged")
    ap.add_argument("--saturation-max", type=float, default=DEFAULTS["saturation_max"],
                    help="largest median top attention weight that is not flagged")
    ap.add_argument("--batch-size", type=int, default=16, help="images attacked together")
    ap.add_argument("--out-dir", default="/kaggle/working/masking",
                    help="folder for rows_masking_<tag>.csv and summary_masking_<tag>.json")
    ap.add_argument("--tag", default="full", help="name added to the output files")
    args = ap.parse_args(argv)

    if min(args.steps, args.budget_steps, args.restarts, args.batch_size, args.per_class) < 1:
        ap.error("--steps, --budget-steps, --restarts, --batch-size and --per-class must be at least 1")
    if args.step_size is not None and not args.step_size > 0:
        ap.error("--step-size must be positive")
    try:
        taus = check_taus([float(t) for t in args.taus.split(",") if t.strip()])
    except ValueError as e:
        ap.error(f"--taus: {e}")
    settings = settings_from(
        levels=parse_levels(ap, args.levels, "--levels"), steps=args.steps,
        steps_grid=parse_positive_ints(ap, args.steps_grid, "--steps-grid"),
        steps_levels=parse_levels(ap, args.steps_levels, "--steps-levels"),
        budget_grid=parse_levels(ap, args.budget_grid, "--budget-grid"), budget_steps=args.budget_steps,
        restarts=args.restarts, taus=taus, step_size=args.step_size, monotone_tol=args.monotone_tol,
        unbounded_min=args.unbounded_min, gain_tol=args.gain_tol, zero_grad_max=args.zero_grad_max,
        saturation_max=args.saturation_max)

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    df = pd.read_csv(args.correct_csv)
    if args.exclude_from:
        seen = set(pd.read_csv(args.exclude_from, usecols=["link_path"]).link_path)
        before = len(df)
        df = df[~df.link_path.isin(seen)]
        print(f"excluded {before - len(df)} images listed in {args.exclude_from}")
    if df.empty:
        raise SystemExit("no images left to sample (check --correct-csv and --exclude-from)")
    df = _ADAPTIVE.sample_images(df, args.per_class, args.seed, args.max_images)

    model = load_checkpoint(args.ckpt)          # after the CSV checks: the model takes minutes to load
    if len(model.inputs) != 2:
        raise SystemExit("24 needs a dual-input [image, label] model; attnet_gate (three inputs) is not supported")
    arch, attention = model_arch(model), has_attention(model)
    print(f"loaded: {args.ckpt} ({arch}, attention pool: {attention})")
    s = settings
    print(f"images: {len(df)} from {df.true_idx.nunique()} classes | steps test {s['steps_grid']} at "
          f"{s['steps_levels']} levels | budget test {s['budget_grid']} levels, {s['budget_steps']} steps | "
          f"compare pgd / aas_pgd (taus {taus}) / rr_pgd ({s['restarts']} starts) at {s['levels']} levels, "
          f"{s['steps']} steps | seed {args.seed}")
    if not attention:
        print("note: no attention pool, so aas_pgd is pgd and there are no attention statistics")

    base_cols = [c for c in BASE_COLS if c in df.columns]
    all_rows, timings = [], {}
    t0 = time.time()
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([_ADAPTIVE.load_image(_ADAPTIVE.resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        pred = predict_logits(model, clean).argmax(1)
        if (pred != labels).any():
            print(f"  WARNING: {(pred != labels).sum()} 'correct' images are misclassified now")
        all_rows += run_batch(model, clean, labels, batch[base_cols].to_dict("records"), settings,
                              args.seed + start, timings)
        done = start + len(batch)
        minutes = (time.time() - t0) / 60
        print(f"  {done}/{len(df)} images | {minutes:.1f} min | ETA {minutes / done * (len(df) - done):.1f} min "
              f"| {minutes * 60 / done:.1f} s per image", flush=True)

    rows = pd.DataFrame(all_rows)
    results = summarise(rows, settings)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "git_commit": _ADAPTIVE.git_commit(), "arch": arch,
               "has_attention": attention, "n_images": int(len(df)), "n_classes": int(df.true_idx.nunique()),
               "per_class": args.per_class, "max_images": args.max_images, "seed": args.seed,
               "exclude_from": args.exclude_from, "correct_csv": args.correct_csv,
               "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
               "step_size": args.step_size if args.step_size is not None else "2.5 * epsilon / steps",
               "attack_loss": "margin (best other logit - true logit, margin-free logits)",
               "success": "the unscaled model (attention temperature 1) misclassifies the 8-bit image",
               "random_start": "uniform in the epsilon ball, seed + first index of the batch; restart r adds "
                               f"r * {RESTART_SEED_STRIDE}",
               "aas_tau_selection": "per step and image, the proposal after which the unscaled model's margin "
                                    "loss is largest" if attention else "none (no attention pool: aas_pgd is pgd)",
               **results,
               "seconds": {k: round(v, 1) for k, v in timings.items()},
               "seconds_per_image": round((time.time() - t0) / max(1, len(df)), 2),
               "minutes": round((time.time() - t0) / 60, 1)}

    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, f"rows_masking_{args.tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"summary_masking_{args.tag}.json"), "w") as f:
        json.dump(_ADAPTIVE._json_safe(summary), f, indent=2)

    print("PGD success vs budget (levels):",
          {L: round(r["rate"], 3) for L, r in results["pgd_success_vs_budget"].items()})
    for key, rates in results["pgd_success_vs_steps"].items():
        print(f"PGD success vs steps at {key}:", {n: round(r["rate"], 3) for n, r in rates.items()})
    print("attack comparison:", {k: round(r["rate"], 3) for k, r in results["attack_comparison"].items()})
    print("masking flags:", results["masking_flags"] or "none", "| skipped:", results["checks_skipped"] or "none")
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()
