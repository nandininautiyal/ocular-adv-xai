"""
scripts/check_attacks.py

Validation run for the white-box attack modules on the trained 300-class model.
  1. margin-free logits reproduce model([image, zeros]); prints the top-1/top-2 logit gaps
  2. vendored reference attacks (paper_*) at eps = 0.001
     (earlier smoke test on 100 images: FGSM 37%, BIM 6%, MIM 22%, PGD 72%)
  3. standard attacks (projected every step) at several eps, CE loss vs margin loss
All images are correctly classified clean test images, so ASR = fraction now misclassified.
Nothing is saved as PNG: eps = 0.001 would be lost to 8-bit quantisation.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import argparse
import inspect
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.models.arcface_inference import get_n_classes, load_checkpoint, predict_logits  # noqa: E402
from src.attacks import whitebox as wb  # noqa: E402


def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))   # same as training / eval
    return (tf.keras.preprocessing.image.img_to_array(img) / 255.0).astype(np.float32)


def predict_chunked(model, X, bs=50):
    return np.concatenate([predict_logits(model, X[i:i + bs]) for i in range(0, len(X), bs)]).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", default="/kaggle/working/eval_clean/correct_test_full.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--out-dir", default="/kaggle/working/check_attacks")
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    model = load_checkpoint(args.ckpt)
    n_classes = get_n_classes(model)
    df = pd.read_csv(args.correct_csv).sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    X = np.stack([load_image(p) for p in df.link_path])
    y = df.true_idx.to_numpy().astype(int)
    print(f"{len(X)} correctly classified test images")

    # 1. margin-free path vs the model's own output
    logits = np.concatenate([predict_logits(model, X[i:i + 50]) for i in range(0, len(X), 50)])
    ref = np.concatenate([
        model.predict([X[i:i + 50], np.zeros((len(X[i:i + 50]), n_classes), "float32")], verbose=0)
        for i in range(0, len(X), 50)])
    srt = np.sort(logits, axis=1)
    gaps = srt[:, -1] - srt[:, -2]
    print("\n[1] margin-free logits vs model([image, zeros])")
    print("    argmax agreement:", float((logits.argmax(1) == ref.argmax(1)).mean()))
    print("    max |softmax(logits) - model output|:", float(np.abs(softmax(logits, axis=1) - ref).max()))
    print("    clean accuracy on this sample (should be 1.0):", float((logits.argmax(1) == y).mean()))
    print("    top-1 minus top-2 logit gap: median %.1f | 10th pct %.1f | 90th pct %.1f"
          % (np.median(gaps), np.percentile(gaps, 10), np.percentile(gaps, 90)))

    rows = []

    def record(family, name, loss, eps, adv, secs):
        pred = predict_chunked(model, adv)
        d = np.abs(adv - X)
        rows.append({"family": family, "attack": name, "loss": loss, "eps": round(eps, 5),
                     "ASR": float((pred != y).mean()),
                     "mean_linf": float(d.max(axis=(1, 2, 3)).mean()), "max_linf": float(d.max()),
                     "secs_per_image": secs / len(X)})
        print(rows[-1], flush=True)

    if not args.skip_paper:
        print("\n[2] vendored reference attacks, eps = 0.001")
        for name, fn in [("fgsm", wb.paper_fgsm), ("bim", wb.paper_bim),
                         ("mim", wb.paper_mim), ("pgd", wb.paper_pgd)]:
            t0 = time.time()
            adv = fn(X, model, epsilon=0.001)
            record("paper", name, "reference", 0.001, adv, time.time() - t0)

    print("\n[3] standard attacks, projected every step")
    has_loss = "loss" in inspect.signature(wb.fgsm).parameters
    losses = ["ce", "margin"] if has_loss else [None]
    for eps in [0.001, 2 / 255, 8 / 255]:
        for name, fn in [("fgsm", wb.fgsm), ("bim", wb.bim), ("pgd", wb.pgd), ("mim", wb.mim)]:
            for loss in losses:
                kw = {} if loss is None else {"loss": loss}
                t0 = time.time()
                adv = np.concatenate([fn(X[i:i + 25], model, epsilon=eps, label=y[i:i + 25], **kw)
                                      for i in range(0, len(X), 25)])
                record("standard", name, loss or "default", eps, adv, time.time() - t0)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + res.round(4).to_string(index=False))
    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "check_attacks.csv"), index=False)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()