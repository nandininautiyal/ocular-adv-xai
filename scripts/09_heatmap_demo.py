"""
scripts/07_heatmap_demo.py

Small end-to-end visual check: clean -> attacked -> defended, with LayerCAM and Eigen-CAM
heatmaps and divergence metrics, on a handful of correctly classified test images.
Uses the project modules (src.attacks, src.defences, src.xai, src.compare) unchanged.

Heatmaps explain the TRUE class at every stage and are compared at native 8x8 resolution;
they are upsampled only for display. Controls: random noise with the same L-inf size as the
attack, and the defence applied to the clean image.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import argparse
import inspect
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.models.arcface_inference import load_checkpoint, predict_labels  # noqa: E402
from src.attacks import whitebox as wb  # noqa: E402
from src.defences.inference import bit_depth_reduction, jpeg_compress  # noqa: E402
from src.xai.attribution import eigencam, layercam  # noqa: E402
from src.compare.divergence import compare_heatmaps  # noqa: E402


def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
    return (tf.keras.preprocessing.image.img_to_array(img) / 255.0).astype(np.float32)


def overlay(ax, img, heat, title):
    ax.imshow(np.clip(img, 0, 1))
    ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1, interpolation="bilinear",
              extent=(0, img.shape[1], img.shape[0], 0))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", default="/kaggle/working/eval_clean/correct_test_full.csv")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--eps", type=float, default=4 / 255)
    ap.add_argument("--attack", default="pgd", choices=["fgsm", "bim", "pgd", "mim"])
    ap.add_argument("--loss", default="margin", choices=["ce", "margin"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/heatmap_demo")
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    model = load_checkpoint(args.ckpt)
    df = pd.read_csv(args.correct_csv).sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    X = np.stack([load_image(p) for p in df.link_path])
    y = df.true_idx.to_numpy().astype(int)

    fn = getattr(wb, args.attack)
    kw = {"loss": args.loss} if "loss" in inspect.signature(fn).parameters else {}
    adv = fn(X, model, epsilon=args.eps, label=y, **kw)
    print(f"{args.attack} eps={args.eps:.5f} {kw} | max L-inf actually used: {np.abs(adv - X).max():.5f}")

    rng = np.random.default_rng(args.seed)
    noise = np.clip(X + rng.uniform(-args.eps, args.eps, X.shape).astype(np.float32), 0, 1)
    stages = {
        "clean": X,
        "attacked": adv,
        "attacked+jpeg": jpeg_compress(adv, quality=75),
        "attacked+bit4": bit_depth_reduction(adv, bits=4),
        "noise (control)": noise,
        "clean+jpeg (control)": jpeg_compress(X, quality=75),
    }
    preds = {k: predict_labels(model, v) for k, v in stages.items()}
    print("\naccuracy on the true class per stage:")
    for k in stages:
        print(f"  {k:22s} {(preds[k] == y).mean():.2f}")

    maps = {
        "layercam": {k: layercam(model, v, class_idx=y, size=None) for k, v in stages.items()},
        "eigencam": {k: eigencam(model, v, size=None) for k, v in stages.items()},
    }
    print("\nnative map shape:", maps["layercam"]["clean"].shape)

    rows = []
    for cam in maps:
        for k in stages:
            if k == "clean":
                continue
            vals = [compare_heatmaps(maps[cam]["clean"][i], maps[cam][k][i]) for i in range(len(X))]
            row = {"cam": cam, "stage": k}
            for m in ("spearman", "top10_intersection", "top25_intersection", "js", "cosine"):
                if m in vals[0]:
                    row[m] = float(np.nanmean([v[m] for v in vals]))
            rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nclean vs stage (mean over images; spearman and top-k: 1 = same, js: 0 = same)\n" + table.round(3).to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "demo_metrics.csv"), index=False)
    arrays = {"true_idx": y}
    for cam in maps:
        for k, v in maps[cam].items():
            arrays[f"{cam}_{re.sub(r'[^a-z0-9]+', '_', k)}"] = v
    np.savez_compressed(os.path.join(args.out_dir, "demo_maps.npz"), **arrays)

    show = ["clean", "attacked", "attacked+jpeg", "attacked+bit4"]
    for cam in maps:
        fig, axes = plt.subplots(len(X), len(show), figsize=(2.6 * len(show), 2.6 * len(X)), squeeze=False)
        for i in range(len(X)):
            for j, k in enumerate(show):
                ok = "ok" if preds[k][i] == y[i] else "WRONG"
                overlay(axes[i][j], stages[k][i], maps[cam][k][i], f"{k} | {ok}")
        fig.suptitle(f"{cam} for the true class, native 8x8 maps")
        plt.tight_layout()
        path = os.path.join(args.out_dir, f"demo_{cam}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()