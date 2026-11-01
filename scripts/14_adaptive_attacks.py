"""
scripts/14_adaptive_attacks.py

Adaptive attacks against the inference-time defences: does JPEG or bit-depth
reduction still "restore" the label when the attacker knows the defence is
there? (Athalye et al., ICML 2018; Tramer et al., NeurIPS 2020.)

Run it on Kaggle next to the trained checkpoint, after 04_eval_clean.py has
written the correct-image list. Images are sampled exactly like
05_run_experiment.py: with the same --seed and --exclude-from, --per-class 1
takes, for every class, the first of the images that a 05 run with
--per-class 2 used.

For each image x budget x defence D (JPEG at --jpeg-quality, bit depth at --bits):
  pgd       non-adaptive: standard PGD on the undefended model, rounded to
            8 bits, then D. This is the attack behind 05's restoration rates.
  bpda_pgd  adaptive: PGD through D with BPDA (src/attacks/adaptive.py). Every
            step attacks D(8-bit image), and the attack keeps the iterate that
            is most adversarial after D.
  tr_pgd    transform-robust: maximises the margin loss on the image and on
            D(image) together, so it has to fool the model with and without D
            (the attack a detector comparing x with D(x) would face).
The three attacks use the same steps, step size and random start (seed + batch
start), so they differ only in what they optimise. Every attacked image is an
8-bit image within whole 8-bit levels of the budget (default 1, 2 and 4 of 255).

Each row (image x attack x budget x defence) records success after D (the
prediction for D(adv) is not the true label) and without D; the margins (true-
class logit minus the best other margin-free logit; negative = fooled) of D(adv)
and adv; the perturbation size in image space (pert_linf and pert_l2 on the
[0, 1] scale, pert_linf_levels in 8-bit levels, rounded to 0.001 so that
float32 round-off of ~1e-5 level never reads as more than the budget);
whether D(clean) is still classified correctly;
and how similar the true-class map of D(adv) is to the map of D(clean), for
LayerCAM and Eigen-CAM (Spearman and top-10 % intersection of the native 8 x 8
maps, src/compare/divergence.py).

Writes to --out-dir:
  rows_adaptive_<tag>.csv      one row per image x attack x budget x defence
  summary_adaptive_<tag>.json  success rates with 95 % Wilson intervals per
                               attack@budget+defence (after D, without D, both,
                               and the best attack per image), defence-on-clean
                               flip rates, median map similarities, timings and
                               the git commit

Usage (Kaggle, with the checkpoint dataset attached and 01, 02, 04 already run):
  smoke (2-3 min):
    python scripts/14_adaptive_attacks.py --ckpt <checkpoint.keras> \
        --correct-csv /kaggle/working/eval_clean/correct_test_full.csv --max-images 8 --tag smoke
  full (one image per class, 300 images):
    python scripts/14_adaptive_attacks.py --ckpt <checkpoint.keras> \
        --correct-csv /kaggle/working/eval_clean/correct_test_full.csv --per-class 1 --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import functools
import inspect
import json
import math
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.adaptive import bpda_pgd, identity, transform_robust_pgd  # noqa: E402
from src.attacks.whitebox import pgd  # noqa: E402
from src.compare.divergence import spearman, top_k_intersection  # noqa: E402
from src.defences.inference import bit_depth_reduction, jpeg_compress  # noqa: E402
from src.models.arcface_inference import load_checkpoint, predict_logits  # noqa: E402
from src.xai.attribution import XAI_REGISTRY  # noqa: E402

ATTACKS = ("pgd", "bpda_pgd", "tr_pgd")
BASE_COLS = ["image_id", "link_path", "relpath", "true_idx", "true_subject", "eye", "device", "lighting",
             "phase"]


def load_image(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def quantise(images, reference=None, epsilon=None):
    """Round to the 8-bit grid a saved image has: what a deployed system actually receives.

    With *reference* (the clean images, already on the grid) and *epsilon*, the result is
    also clipped to the largest whole number of 8-bit levels within the budget, so rounding
    never pushes a perturbation past epsilon (e.g. epsilon = 0.03 allows 7 levels, not 8).
    Private copy of scripts/05_run_experiment.py's quantise.
    """
    images = np.asarray(images, dtype=np.float32)
    out = np.round(np.clip(images, 0.0, 1.0) * 255.0) / 255.0
    if reference is not None and epsilon is not None:
        # budgets are written to ~5 decimals (0.0078431 * 255 = 1.99999), so allow for that
        levels = np.floor(epsilon * 255.0 + 1e-3) / 255.0
        reference = np.asarray(reference, dtype=np.float32)
        out = np.clip(out, reference - levels, reference + levels)
    return out.astype(np.float32)


def budget_levels(epsilon):
    """Whole 8-bit levels an 8-bit image can use within *epsilon* (as in quantise)."""
    return int(np.floor(epsilon * 255.0 + 1e-3))


def resolve_path(row, data_root):
    if isinstance(row.link_path, str) and os.path.exists(row.link_path):
        return row.link_path
    if data_root and isinstance(getattr(row, "relpath", None), str):
        path = os.path.join(data_root, row.relpath)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{row.link_path} not found; rebuild the split tree (01, 02) "
                            f"or pass --data-root to resolve the manifest relpath")


def wilson(successes, n, z=1.96):
    """95 % Wilson score interval, clipped to [0, 1] (round-off gives -1e-17 at k = 0)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def git_commit():
    """Commit of the repo this script runs from ("+dirty" if tracked files changed)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (commit + ("+dirty" if dirty else "")) if commit else "unknown"


def make_defences(names, jpeg_quality=75, bits=4):
    """name -> (defence with explicit parameters, parameter value) for each name in *names*."""
    available = {"jpeg": (functools.partial(jpeg_compress, quality=jpeg_quality), jpeg_quality),
                 "bit_depth": (functools.partial(bit_depth_reduction, bits=bits), bits)}
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown defences {unknown}; choose from {list(available)}")
    return {n: available[n] for n in names}


def sample_images(df, per_class, seed, max_images=None):
    """The images 05_run_experiment.py samples (same lines); image_id is the row in the correct CSV."""
    df = df.sample(frac=1.0, random_state=seed).groupby("true_idx").head(per_class)
    if max_images:
        df = df.head(max_images)
    return df.reset_index().rename(columns={"index": "image_id"})


def margins(logits, labels):
    """True-class logit minus the best other logit, per image (negative = misclassified)."""
    idx = np.arange(len(labels))
    true = logits[idx, labels]
    other = logits.copy()
    other[idx, labels] = -np.inf
    return true - other.max(axis=1)


def true_class_maps(model, xai, images, labels):
    """Native-resolution maps for the true class (Eigen-CAM is class-agnostic)."""
    fn = XAI_REGISTRY[xai]
    kwargs = {"class_idx": labels} if "class_idx" in inspect.signature(fn).parameters else {}
    return fn(model, images, size=None, **kwargs)


def _elapsed(timings, key, start):
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


def score(model, clean, labels, adv, defence, reference, xai_names, base_rows, info):
    """Rows for one batch of attacked images: success, margins and size, with and without the defence."""
    logits = predict_logits(model, adv)
    defended = defence(adv)
    logits_defended = predict_logits(model, defended)
    pred, pred_defended = logits.argmax(1), logits_defended.argmax(1)
    margin, margin_defended = margins(logits, labels), margins(logits_defended, labels)
    diff = (adv - clean).reshape(len(adv), -1)
    linf, l2 = np.abs(diff).max(axis=1), np.sqrt((diff ** 2).sum(axis=1))
    maps = {xai: true_class_maps(model, xai, defended, labels) for xai in xai_names}
    rows = []
    for i in range(len(adv)):
        row = {**base_rows[i], **info,
               "defended_clean_correct": bool(reference["correct"][i]),
               "pred_defended": int(pred_defended[i]),
               "success_defended": bool(pred_defended[i] != labels[i]),
               "margin_defended": float(margin_defended[i]),
               "pred_undefended": int(pred[i]),
               "success_undefended": bool(pred[i] != labels[i]),
               "margin_undefended": float(margin[i]),
               "pert_linf": float(linf[i]), "pert_linf_levels": round(float(linf[i]) * 255.0, 3),
               "pert_l2": float(l2[i])}
        for xai in xai_names:
            ref, new = reference["maps"][xai][i], maps[xai][i]
            row[f"{xai}_vs_defended_clean_spearman"] = spearman(ref, new)
            row[f"{xai}_vs_defended_clean_top10_intersection"] = top_k_intersection(ref, new, k=0.1)
        rows.append(row)
    return rows


def run_batch(model, clean, labels, base_rows, epsilons, defences, attacks=ATTACKS,
              xai_names=("layercam", "eigencam"), steps=20, step_size=None, seed=0, quantised=True,
              timings=None):
    """Every attack x budget x defence on one batch of clean images.

    Parameters
    ----------
    model : tf.keras.Model with dual inputs [image, label]
    clean : ndarray (N, 256, 256, 3), float32, [0, 1] — correctly classified 8-bit images
    labels : ndarray (N,) — true classes
    base_rows : list of N dicts — per-image columns copied into every row (image_id, ...)
    epsilons : list of float — L-inf budgets
    defences : dict name -> (defence, parameter value), as returned by make_defences
    attacks : subset of ATTACKS
    xai_names : names in XAI_REGISTRY; the maps explain the true class
    steps, step_size : for every attack (step_size None = 2.5 * epsilon / steps)
    seed : random-start seed, shared by the three attacks
    quantised : 8-bit attacked images (default) or float
    timings : dict that collects seconds per attack and for scoring ("evaluate")

    Returns
    -------
    rows : list of dicts, one per image x attack x budget x defence
    """
    timings = {} if timings is None else timings
    start = time.perf_counter()
    reference = {}
    for name, (defence, _) in defences.items():
        defended = defence(clean)
        reference[name] = {
            "correct": predict_logits(model, defended).argmax(1) == labels,
            "maps": {xai: true_class_maps(model, xai, defended, labels) for xai in xai_names}}
    _elapsed(timings, "evaluate", start)

    rows = []
    for eps in epsilons:
        common = {"label": labels, "steps": steps, "step_size": step_size, "seed": seed}
        adversarial = {}                                  # (attack, defence) -> attacked images
        if "pgd" in attacks:
            start = time.perf_counter()
            adv = pgd(clean, model, eps, **common)
            adv = quantise(adv, clean, eps) if quantised else adv
            _elapsed(timings, "pgd", start)
            adversarial.update({("pgd", name): adv for name in defences})
        for name, (defence, _) in defences.items():
            if "bpda_pgd" in attacks:
                start = time.perf_counter()
                adversarial[("bpda_pgd", name)] = bpda_pgd(clean, model, eps, defence,
                                                           quantise_8bit=quantised, **common)
                _elapsed(timings, "bpda_pgd", start)
            if "tr_pgd" in attacks:
                start = time.perf_counter()
                adversarial[("tr_pgd", name)] = transform_robust_pgd(clean, model, eps, [identity, defence],
                                                                     quantise_8bit=quantised, **common)
                _elapsed(timings, "tr_pgd", start)

        start = time.perf_counter()
        for (attack, name), adv in adversarial.items():
            defence, param = defences[name]
            info = {"attack": attack, "adaptive": attack != "pgd", "epsilon": eps,
                    "budget_levels": budget_levels(eps) if quantised else round(eps * 255.0, 3),
                    "defence": name, "defence_param": param, "steps": steps}
            rows += score(model, clean, labels, adv, defence, reference[name], xai_names, base_rows, info)
        _elapsed(timings, "evaluate", start)
    return rows


def _rate(flags):
    flags = np.asarray(flags, dtype=bool)
    k, n = int(flags.sum()), int(flags.size)
    low, high = wilson(k, n)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci_low": low, "ci_high": high}


def summarise(rows):
    """Success rates (95 % Wilson intervals), defence-on-clean flip rates, median map similarity.

    Keys are attack@epsilon+defence, ordered by budget, defence and attack. The rates
    "success_after_defence_any_attack" count an image as broken if any of the attacks run
    fooled the defended model on it (the per-image worst case).
    """
    attack_order = {a: i for i, a in enumerate(ATTACKS)}
    defence_order = {d: i for i, d in enumerate(pd.unique(rows.defence))}
    rows = rows.assign(_attack=rows.attack.map(attack_order), _defence=rows.defence.map(defence_order))
    rows = rows.sort_values(["epsilon", "_defence", "_attack"], kind="stable")
    rows["key"] = [f"{a}@{e:.6g}+{d}" for a, e, d in zip(rows.attack, rows.epsilon, rows.defence)]
    similarity = [c for c in rows.columns if "_vs_defended_clean_" in c]
    out = {name: {} for name in ("success_after_defence", "success_after_defence_where_defended_clean_correct",
                                 "success_without_defence", "success_with_and_without_defence",
                                 "median_map_similarity_to_defended_clean")}
    for key, g in rows.groupby("key", sort=False):
        out["success_after_defence"][key] = _rate(g.success_defended)
        out["success_after_defence_where_defended_clean_correct"][key] = _rate(
            g.success_defended[g.defended_clean_correct.astype(bool)])
        out["success_without_defence"][key] = _rate(g.success_undefended)
        out["success_with_and_without_defence"][key] = _rate(g.success_defended & g.success_undefended)
        out["median_map_similarity_to_defended_clean"][key] = {
            c: float(np.nanmedian(g[c])) if g[c].notna().any() else float("nan") for c in similarity}
    any_attack = rows.groupby(["epsilon", "defence", "image_id"], sort=False).success_defended.any()
    out["success_after_defence_any_attack"] = {
        f"{e:.6g}+{d}": _rate(g) for (e, d), g in any_attack.groupby(level=["epsilon", "defence"], sort=False)}
    clean = rows.drop_duplicates(["defence", "image_id"])
    out["defence_on_clean_flip_rate"] = {
        d: _rate(~g.defended_clean_correct.astype(bool)) for d, g in clean.groupby("defence", sort=False)}
    return out


def _json_safe(value):
    """NaN -> None, recursively, so the summary is valid JSON."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained .keras checkpoint")
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=1, help="images per class (1 = 300 images)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="smoke run: cap the number of images (e.g. 8 for a 2-3 minute trial)")
    ap.add_argument("--exclude-from", default=None,
                    help="CSV with a link_path column (images_<tag>.csv of 05, or rows_adaptive_<tag>.csv); "
                         "those images are not used")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed (as in 05); the random start of batch b uses seed + its first index")
    ap.add_argument("--epsilons", default="0.0039216,0.0078431,0.0156863",
                    help="L-inf budgets: 1/255, 2/255 and 4/255. With 8-bit images each budget is used as "
                         "the largest whole number of 8-bit levels within it")
    ap.add_argument("--defences", default="jpeg,bit_depth", help="jpeg and/or bit_depth")
    ap.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality of the jpeg defence")
    ap.add_argument("--bits", type=int, default=4, help="bit depth of the bit_depth defence")
    ap.add_argument("--attacks", default="pgd,bpda_pgd,tr_pgd",
                    help="pgd (non-adaptive), bpda_pgd (adaptive), tr_pgd (transform-robust)")
    ap.add_argument("--steps", type=int, default=20, help="gradient steps of every attack")
    ap.add_argument("--step-size", type=float, default=None, help="default: 2.5 * epsilon / steps")
    ap.add_argument("--xai", default="layercam,eigencam", help="maps to compare; 'none' skips them")
    ap.add_argument("--no-quantise", action="store_true",
                    help="keep attacked images in float32 (no 8-bit rounding anywhere)")
    ap.add_argument("--batch-size", type=int, default=16, help="images attacked together")
    ap.add_argument("--out-dir", default="/kaggle/working/adaptive",
                    help="folder for rows_adaptive_<tag>.csv and summary_adaptive_<tag>.json")
    ap.add_argument("--tag", default="full", help="name added to the output files")
    args = ap.parse_args(argv)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    defence_names = [d.strip() for d in args.defences.split(",") if d.strip()]
    xai_names = [x.strip() for x in args.xai.split(",") if x.strip() and x.strip() != "none"]
    epsilons = [float(e) for e in args.epsilons.split(",")]
    for name in attacks:
        if name not in ATTACKS:
            raise SystemExit(f"unknown attack {name}; choose from {list(ATTACKS)}")
    for name in xai_names:
        if name not in XAI_REGISTRY:
            raise SystemExit(f"unknown XAI method {name}; choose from {list(XAI_REGISTRY)}")
    if not attacks or not defence_names:
        raise SystemExit("need at least one attack and one defence")
    if not 1 <= args.jpeg_quality <= 100 or not 1 <= args.bits <= 8:
        raise SystemExit("--jpeg-quality must be in [1, 100] and --bits in [1, 8]")
    if min(args.steps, args.batch_size, args.per_class) < 1:
        raise SystemExit("--steps, --batch-size and --per-class must be at least 1")
    if not all(e > 0 for e in epsilons) or (args.step_size is not None and not args.step_size > 0):
        raise SystemExit("--epsilons and --step-size must be positive")
    try:
        defences = make_defences(defence_names, args.jpeg_quality, args.bits)
    except ValueError as e:
        raise SystemExit(str(e))
    quantised = not args.no_quantise
    if quantised:
        for eps in epsilons:
            if budget_levels(eps) < 1:
                raise SystemExit(f"epsilon {eps} is below one 8-bit level (1/255) and cannot change an "
                                 f"8-bit image; use budgets of at least 0.0039216 or --no-quantise")
            if abs(eps * 255.0 - round(eps * 255.0)) > 1e-3:
                print(f"note: epsilon {eps} = {eps * 255:.3f} levels; 8-bit images use {budget_levels(eps)}")

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
    df = sample_images(df, args.per_class, args.seed, args.max_images)

    model = load_checkpoint(args.ckpt)          # after the CSV checks: the model takes minutes to load
    print("loaded:", args.ckpt)
    base_cols = [c for c in BASE_COLS if c in df.columns]
    levels = [budget_levels(e) if quantised else round(e * 255.0, 3) for e in epsilons]
    print(f"images: {len(df)} from {df.true_idx.nunique()} classes | attacks: {attacks} | epsilons: {epsilons} "
          f"({levels} levels) | defences: jpeg q={args.jpeg_quality}, bit_depth {args.bits} bits -> "
          f"{defence_names} | steps: {args.steps} | xai: {xai_names} | 8-bit: {quantised} | seed: {args.seed}")

    all_rows, timings = [], {}
    t0 = time.time()
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([load_image(resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        pred = predict_logits(model, clean).argmax(1)
        if (pred != labels).any():
            print(f"  WARNING: {(pred != labels).sum()} 'correct' images are misclassified now")
        all_rows += run_batch(model, clean, labels, batch[base_cols].to_dict("records"), epsilons, defences,
                              attacks, xai_names, args.steps, args.step_size, args.seed + start, quantised,
                              timings)
        done = start + len(batch)
        minutes = (time.time() - t0) / 60
        print(f"  {done}/{len(df)} images | {minutes:.1f} min | ETA {minutes / done * (len(df) - done):.1f} min",
              flush=True)

    rows = pd.DataFrame(all_rows)
    n = len(df)
    runs_per_image = {"pgd": len(epsilons), "bpda_pgd": len(epsilons) * len(defences),
                      "tr_pgd": len(epsilons) * len(defences)}
    per_run = {a: timings[a] / (n * runs_per_image[a]) for a in attacks}
    summary = {"tag": args.tag, "ckpt": args.ckpt, "git_commit": git_commit(), "n_images": int(n),
               "n_classes": int(df.true_idx.nunique()), "per_class": args.per_class,
               "max_images": args.max_images, "seed": args.seed, "exclude_from": args.exclude_from,
               "epsilons": epsilons, "budget_levels": levels,
               "defences": {name: ({"quality": args.jpeg_quality} if name == "jpeg" else {"bits": args.bits})
                            for name in defence_names},
               "attacks": attacks, "steps": args.steps,
               "step_size": args.step_size if args.step_size is not None else "2.5 * epsilon / steps",
               "random_start": "uniform in the epsilon ball, seed + first index of the batch",
               "attack_loss": "margin (best other logit - true logit, margin-free logits)",
               "quantised_8bit": quantised, "xai": xai_names, "maps_explain": "true class",
               **summarise(rows),
               "seconds": {k: round(v, 1) for k, v in timings.items()},
               "seconds_per_image_per_run": {a: round(v, 3) for a, v in per_run.items()},
               "minutes": round((time.time() - t0) / 60, 1)}
    if "pgd" in per_run and per_run["pgd"] > 0:
        summary["cost_relative_to_pgd"] = {a: round(v / per_run["pgd"], 2) for a, v in per_run.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, f"rows_adaptive_{args.tag}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"summary_adaptive_{args.tag}.json"), "w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    print("success after the defence (95 % Wilson interval):")
    lines = list(summary["success_after_defence"].items())
    lines += [(f"any attack@{k}", r) for k, r in summary["success_after_defence_any_attack"].items()]
    width = max(len(k) for k, _ in lines) + 2
    for key, r in lines:
        print(f"  {key:<{width}}{r['rate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n={r['n']}")
    if "cost_relative_to_pgd" in summary:
        print("time per run relative to PGD:", summary["cost_relative_to_pgd"])
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()
