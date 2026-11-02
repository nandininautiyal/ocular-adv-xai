"""
scripts/04_eval_clean.py

Clean-image evaluation of a trained ABRobOcular recognition model (Xception + ArcFace).

Uses the margin-free path (zero label vector), the same one the reference attacks use.
Reports accuracy overall and by device / lighting / eye / phase / condition, per-class
accuracy, confidence statistics, and writes:
  <out-dir>/predictions_<tag>.csv          one row per test image
  <out-dir>/correct_<split>_<tag>.csv      only the correctly classified images
                                           (the starting set for the attack experiments)
  <out-dir>/breakdown_<tag>.csv            accuracy by group
  <out-dir>/summary_<tag>.json

Class indices: by default flow_from_directory numbers the class folders of the evaluated
split in sorted order, which matches training only when that split has every class.
--class-index class_index_<tag>.json (written by 03_train_baseline.py next to the
checkpoint) gives the generator exactly the training order instead (classes=); the run
stops if a folder of the split is missing from the file. Use it for every model trained
after 03 started writing the file, and always for a cross-session model (a split can lack
subjects there).

Usage (after training, with the manifest from 02_make_splits.py):
  python scripts/04_eval_clean.py --split-dir /tmp/ocular_recognition --n-classes 300 \
      --ckpt /kaggle/working/trained_models/Xception_ArcFace_full.keras \
      --manifest /kaggle/working/manifest_split.csv --tag full
Cross-session model, with the training class order:
  python scripts/04_eval_clean.py --split-dir /tmp/ocular_cross_session --n-classes 300 \
      --ckpt /kaggle/working/trained_models/Xception_ArcFace_cs.keras \
      --class-index /kaggle/working/trained_models/class_index_cs.json \
      --manifest /kaggle/working/manifest_split_cross_session.csv --tag cs
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.models.recognition_model import ArcFace  # noqa: E402

MANIFEST_COLS = ["link_path", "relpath", "condition", "device", "lighting", "eye", "phase", "idx", "subject"]


def breakdown(df, col):
    g = df.groupby(col).correct.agg(accuracy="mean", n_correct="sum", n="count")
    g["accuracy"] = g["accuracy"].round(4)
    return g


def training_classes(class_index_path, data_dir):
    """Class names in training index order, from 03's class_index_<tag>.json.

    Stops if the indices are not 0..n-1 or if a class folder of *data_dir* is missing from the
    file (its images would get no index, or the wrong one).
    """
    with open(class_index_path) as f:
        index = json.load(f)
    classes = sorted(index, key=index.get)
    if sorted(index.values()) != list(range(len(index))):
        raise SystemExit(f"{class_index_path}: class indices are not 0..{len(index) - 1}")
    folders = sorted(n for n in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, n)))
    unknown = [n for n in folders if n not in index]
    if unknown:
        raise SystemExit(f"{len(unknown)} class folders of {data_dir} are not in {class_index_path}, "
                         f"e.g. {unknown[:10]}; was the model trained on this split's classes?")
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--ckpt", default="/kaggle/working/trained_models/Xception_ArcFace_full.keras")
    ap.add_argument("--manifest", default="/kaggle/working/manifest_split.csv")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="/kaggle/working/eval_clean")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--max-batches", type=int, default=None, help="debug: evaluate only the first N batches")
    ap.add_argument("--class-index", default=None,
                    help="class_index_<tag>.json from 03: use exactly its class order (default: the split's "
                         "sorted folders)")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print("TF", tf.__version__, "| tf.keras comes from:", tf.keras.Model.__module__, "| GPUs:", len(gpus))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    data_dir = os.path.join(args.split_dir, args.split)
    classes = training_classes(args.class_index, data_dir) if args.class_index else None   # before the slow load

    model = tf.keras.models.load_model(args.ckpt, custom_objects={"ArcFace": ArcFace})
    print("loaded:", args.ckpt)

    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        data_dir, target_size=(256, 256), batch_size=args.batch_size,
        class_mode="categorical", shuffle=False, classes=classes)   # shuffle=False keeps file order fixed
    if classes:
        print("class order from:", args.class_index)
    assert len(gen.class_indices) == args.n_classes, \
        f"--n-classes {args.n_classes} but found {len(gen.class_indices)} class folders"
    idx_to_class = {v: k for k, v in gen.class_indices.items()}

    n_batches = len(gen) if args.max_batches is None else min(len(gen), args.max_batches)
    probs = []
    for i in range(n_batches):
        x, _ = gen[i]
        zeros = np.zeros((len(x), args.n_classes), dtype="float32")   # margin-free inference
        probs.append(np.asarray(model.predict_on_batch([x, zeros])))
        if (i + 1) % 20 == 0 or i + 1 == n_batches:
            print(f"  batch {i + 1}/{n_batches}", flush=True)
    probs = np.concatenate(probs)
    n = len(probs)

    y_true = gen.classes[:n]
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    df = pd.DataFrame({
        "link_path": [os.path.join(data_dir, f) for f in gen.filenames[:n]],
        "true_idx": y_true,
        "pred_idx": pred,
        "true_subject": [int(idx_to_class[i]) for i in y_true],
        "pred_subject": [int(idx_to_class[i]) for i in pred],
        "confidence": probs.max(1),
        "correct": pred == y_true,
        "top5_correct": (top5 == y_true[:, None]).any(1),
    })

    man = pd.read_csv(args.manifest)[MANIFEST_COLS]
    df = df.merge(man, on="link_path", how="left")
    missing = int(df.device.isna().sum())
    if missing:
        print(f"WARNING: {missing} images were not found in the manifest "
              f"(check that --manifest matches --split-dir)")
    else:
        print("subject in manifest == subject in folder name:", bool((df.subject == df.true_subject).all()))

    acc = float(df.correct.mean())
    se = float(np.sqrt(acc * (1 - acc) / n))
    print(f"\nimages evaluated: {n}")
    print(f"CLEAN {args.split} accuracy, margin-free: {acc:.4f}  (approx. 95% CI {acc - 1.96 * se:.4f} to {acc + 1.96 * se:.4f})")
    print(f"top-5 accuracy: {df.top5_correct.mean():.4f}")

    frames = []
    for col in ["device", "lighting", "eye", "phase", "condition"]:
        if df[col].notna().any():
            b = breakdown(df, col)
            print(f"\naccuracy by {col}:\n{b.to_string()}")
            b = b.copy()
            b.insert(0, "group", col)
            b.index.name = "value"
            frames.append(b.reset_index())

    per_class = df.groupby("true_subject").correct.mean()
    print("\nper-class accuracy: min %.3f | 10th pct %.3f | median %.3f | max %.3f" % (
        per_class.min(), per_class.quantile(0.1), per_class.median(), per_class.max()))
    print("10 lowest classes:\n", per_class.sort_values().head(10).round(3).to_string())
    print("\nmean confidence: correct %.3f | wrong %.3f" % (
        df[df.correct].confidence.mean(), df[~df.correct].confidence.mean() if (~df.correct).any() else float("nan")))

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, f"predictions_{args.tag}.csv"), index=False)
    df[df.correct].to_csv(os.path.join(args.out_dir, f"correct_{args.split}_{args.tag}.csv"), index=False)
    if frames:
        pd.concat(frames).to_csv(os.path.join(args.out_dir, f"breakdown_{args.tag}.csv"), index=False)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "class_index": args.class_index, "split": args.split,
               "n_images": n,
               "accuracy_margin_free": acc, "ci95_halfwidth": 1.96 * se,
               "top5_accuracy": float(df.top5_correct.mean()),
               "n_correct": int(df.correct.sum()),
               "per_class_accuracy_median": float(per_class.median()),
               "per_class_accuracy_min": float(per_class.min())}
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote results to", args.out_dir)


if __name__ == "__main__":
    main()