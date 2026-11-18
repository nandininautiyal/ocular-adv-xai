"""
scripts/06_attack_smoke.py

Smoke test of the ABRobOcular reference white-box attacks (FGSM/BIM/MIM/PGD, optional CW)
on the trained 300-class model, using a random sample of CORRECTLY classified clean test
images (from 04_eval_clean.py). Everything stays in memory as float32: nothing is saved as
PNG, because eps = 0.001 would be rounded away by 8-bit quantisation.

ASR follows the paper's code: the adversarial prediction differs from the clean prediction
(for correctly classified images this equals misclassification).

Usage:
  python scripts/05_attack_smoke.py --ckpt <path>.keras \
      --correct-csv /kaggle/working/eval_clean/correct_test_full.csv --n 100
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # must be set before importing tensorflow
import argparse
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.models.recognition_model import ArcFace  # noqa: E402,F401  (also puts third_party/ABRobOcular on sys.path)
from src.models.anatomy_model import CUSTOM_OBJECTS  # noqa: E402  (ArcFace and the anatomy-head layers)
from AB_Rob_Ocular.AB_Rob_Ocular_Tensor import OcularAdversarialAttacks as OAA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", default="/kaggle/working/eval_clean/correct_test_full.csv")
    ap.add_argument("--n-classes", type=int, default=300)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.001])
    ap.add_argument("--attacks", nargs="+", default=["fgsm", "bim", "mim", "pgd"])
    ap.add_argument("--cw-n", type=int, default=0, help="also run CW on the first N images (slow, not eps-bounded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/attack_smoke")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print("TF", tf.__version__, "| tf.keras comes from:", tf.keras.Model.__module__, "| GPUs:", len(gpus))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    model = tf.keras.models.load_model(args.ckpt, custom_objects=CUSTOM_OBJECTS)
    zeros = np.zeros((1, args.n_classes), dtype="float32")   # margin-free inference, as in the reference

    def predict(xb):
        return int(np.asarray(model.predict_on_batch([xb, zeros])).argmax(1)[0])

    df = pd.read_csv(args.correct_csv)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"{len(df)} correctly classified test images | attacks {args.attacks} | eps {args.eps}")

    def run_attack(name, x, eps):
        if name == "fgsm":
            return OAA.fgsm_attack_face(x, zeros, model, epsilon=eps)
        if name == "bim":
            return OAA.bim_attack_face(x, zeros, model, epsilon=eps)
        if name == "mim":
            return OAA.mim_attack_face(x, zeros, model, epsilon=eps)
        if name == "pgd":
            # the reference pgd_attack_face ignores epsilon (eps = num_steps * step_size);
            # step_size = eps / 10 reproduces its default at eps = 0.001 and scales correctly otherwise
            return OAA.pgd_attack_face(x, zeros, model, epsilon=eps, num_steps=10, step_size=eps / 10)
        raise ValueError(name)

    rows = []
    t_start = time.time()
    for i, r in df.iterrows():
        x = OAA.preprocess_image(r.link_path)            # (256,256,3) float32 in [0,1], same as the loader
        xb = x[None]
        p0 = predict(xb)
        timings = {}
        for name in args.attacks:
            for eps in args.eps:
                t0 = time.time()
                adv = np.asarray(run_attack(name, x, eps))
                secs = time.time() - t0
                timings[f"{name}@{eps}"] = round(secs, 2)
                p1 = predict(adv)
                d = adv - xb
                rows.append({"attack": name, "eps": eps, "idx": i, "clean_pred": p0, "true_idx": int(r.true_idx),
                             "adv_pred": p1, "changed": p1 != p0, "correct_after": p1 == int(r.true_idx),
                             "linf": float(np.abs(d).max()), "l2": float(np.linalg.norm(d)), "secs": secs,
                             "zero_pert": float(np.abs(d).max()) == 0.0})
        if i == 0:
            print("first image: clean_pred == true:", p0 == int(r.true_idx), "| seconds per attack:", timings, flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)} images | {(time.time() - t_start) / 60:.1f} min elapsed", flush=True)

    if args.cw_n > 0:
        print("\nrunning CW (reference version, not eps-bounded)")
        for i, r in df.head(args.cw_n).iterrows():
            try:
                x = OAA.preprocess_image(r.link_path)
                xb = x[None]
                p0 = predict(xb)
                t0 = time.time()
                adv = np.asarray(OAA.cw_attack_face(x, zeros, model, learning_rate=0.001, max_iter=10, num_starts=5))
                secs = time.time() - t0
                p1 = predict(adv)
                d = adv - xb
                rows.append({"attack": "cw", "eps": float("nan"), "idx": i, "clean_pred": p0, "true_idx": int(r.true_idx),
                             "adv_pred": p1, "changed": p1 != p0, "correct_after": p1 == int(r.true_idx),
                             "linf": float(np.abs(d).max()), "l2": float(np.linalg.norm(d)), "secs": secs,
                             "zero_pert": float(np.abs(d).max()) == 0.0})
            except Exception:
                print("CW failed on image", i)
                traceback.print_exc()
                break

    res = pd.DataFrame(rows)
    summ = res.groupby(["attack", "eps"], dropna=False).agg(
        n=("changed", "size"), ASR=("changed", "mean"), acc_under_attack=("correct_after", "mean"),
        mean_linf=("linf", "mean"), max_linf=("linf", "max"), zero_perturbation=("zero_pert", "mean"),
        sec_per_image=("secs", "mean")).round(4)
    print("\n" + summ.to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, f"attack_smoke_{args.tag}.csv"), index=False)
    summ.reset_index().to_csv(os.path.join(args.out_dir, f"attack_smoke_summary_{args.tag}.csv"), index=False)
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()