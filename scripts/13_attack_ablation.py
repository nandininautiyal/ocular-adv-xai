"""
scripts/13_attack_ablation.py

Which flaw makes the vendored ABRobOcular attacks weak? Runs FGSM, BIM and MIM
from the vendored settings to the standard ones, fixing one flaw at a time
(src/attacks/ablation.py), plus vendored vs standard PGD and CW, on correctly
classified clean test images.

Writes --out-dir/ablation_<tag>.csv with, per config x epsilon: attack success
rate (prediction != true label) with a 95 % Wilson interval, and mean / max L-inf
and L2 of the perturbation.

Usage (Kaggle):
  python scripts/13_attack_ablation.py --ckpt <checkpoint.keras> \
      --correct-csv /kaggle/working/eval_clean/correct_test_full.csv --n 300 --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.attacks.ablation import ablation_configs, iterative_attack  # noqa: E402
from src.attacks.whitebox import cw, paper_cw, paper_pgd, pgd  # noqa: E402
from src.models.arcface_inference import load_checkpoint, predict_labels  # noqa: E402


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def resolve_path(row, data_root):
    if isinstance(row.link_path, str) and os.path.exists(row.link_path):
        return row.link_path
    if data_root and isinstance(getattr(row, "relpath", None), str):
        path = os.path.join(data_root, row.relpath)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{row.link_path} not found; pass --data-root to resolve the manifest relpath")


def wilson(successes, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def runs(epsilon, seed):
    """config name -> function(images, model, labels) for one epsilon."""
    out = {name: (lambda kw: lambda x, m, y: iterative_attack(x, m, epsilon, label=y, **kw))(kwargs)
           for name, (_, kwargs) in ablation_configs(epsilon).items()}
    out["pgd/vendored"] = lambda x, m, y: paper_pgd(x, m, epsilon=epsilon, num_steps=10, step_size=epsilon / 10)
    out["pgd/standard_margin"] = lambda x, m, y: pgd(x, m, epsilon, label=y, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n", type=int, default=300, help="images (one per class first, then more)")
    ap.add_argument("--epsilons", default="0.001,0.0019608,0.0039216,0.0078431")
    ap.add_argument("--no-cw", action="store_true", help="skip vendored and standard CW (slow)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/ablation")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    model = load_checkpoint(args.ckpt)
    df = pd.read_csv(args.correct_csv).sample(frac=1.0, random_state=args.seed)
    df = df.assign(rank=df.groupby("true_idx").cumcount()).sort_values(["rank", "true_idx"]).head(args.n)
    epsilons = [float(e) for e in args.epsilons.split(",")]
    print(f"images: {len(df)} | epsilons: {epsilons}")

    records = {}
    t0 = time.time()
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        x = np.stack([load_image(resolve_path(r, args.data_root)) for r in batch.itertuples()])
        y = batch.true_idx.to_numpy()
        jobs = [(eps, name, fn) for eps in epsilons for name, fn in runs(eps, args.seed + start).items()]
        if not args.no_cw:
            jobs += [(np.nan, "cw/vendored", lambda x, m, y: paper_cw(x, m)),
                     (np.nan, "cw/standard_l2", lambda x, m, y: cw(x, m, label=y))]
        for eps, name, fn in jobs:
            adv = np.asarray(fn(x, model, y))
            diff = (adv - x).reshape(len(x), -1)
            rec = records.setdefault((name, eps), {"success": [], "linf": [], "l2": []})
            rec["success"] += list(predict_labels(model, adv) != y)
            rec["linf"] += list(np.abs(diff).max(axis=1))
            rec["l2"] += list(np.sqrt((diff ** 2).sum(axis=1)))
        print(f"  {start + len(batch)}/{len(df)} images | {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    for (name, eps), rec in records.items():
        k, n = int(np.sum(rec["success"])), len(rec["success"])
        lo, hi = wilson(k, n)
        rows.append({"config": name, "family": name.split("/")[0], "epsilon": eps, "n": n, "asr": k / n,
                     "asr_ci_low": lo, "asr_ci_high": hi, "mean_linf": float(np.mean(rec["linf"])),
                     "max_linf": float(np.max(rec["linf"])), "mean_l2": float(np.mean(rec["l2"]))})
    table = pd.DataFrame(rows).sort_values(["family", "epsilon", "config"])
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, f"ablation_{args.tag}.csv"), index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(table[["config", "epsilon", "n", "asr", "asr_ci_low", "asr_ci_high", "max_linf"]].to_string(index=False))
    print(f"minutes: {(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
