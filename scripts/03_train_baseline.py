"""
scripts/03_train_baseline.py

Train the ABRobOcular recognition model (Xception + ArcFace) with THEIR create_model()
and THEIR CustomDataGen, on the class-folder tree built by 02_make_splits.py.
Augmentation and callbacks follow the reference loader/train.py.

Reports two test accuracies:
  - margin-free (zero label vector): real inference, as the attacks and predict.py use
  - with-labels: what model.evaluate() reports (ArcFace margin applied to the true class)

Usage:
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_small --n-classes 30 --epochs 2 --tag small
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_recognition --n-classes 300 --epochs 10 --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; must be set before importing tensorflow
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.models.recognition_model import ArcFace, CustomDataGen, build_model, load_cfg  # noqa: E402


def accuracies(model, gen, n_classes):
    """Return (margin-free accuracy, with-labels accuracy) over a CustomDataGen."""
    ok_free = ok_lab = total = 0
    for i in range(len(gen)):
        (x, y), _ = gen[i]
        y_true = y.argmax(1)
        p_free = model.predict_on_batch([x, np.zeros((len(x), n_classes), dtype="float32")])
        p_lab = model.predict_on_batch([x, y])
        ok_free += int((np.asarray(p_free).argmax(1) == y_true).sum())
        ok_lab += int((np.asarray(p_lab).argmax(1) == y_true).sum())
        total += len(x)
    return ok_free / total, ok_lab / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="/kaggle/working/trained_models")
    ap.add_argument("--cfg", default=os.path.join(REPO, "configs", "baseline.yml"))
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print("TF", tf.__version__, "| tf.keras:", tf.keras.__version__, "| GPUs:", gpus)
    if not gpus:
        print("WARNING: no GPU found, this will be very slow")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    cfg = load_cfg(args.cfg)
    cfg["model"]["n_classes"] = args.n_classes
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    tf.keras.utils.set_random_seed(42)

    bs = cfg["training"]["batch_size"]
    size = tuple(cfg["data"]["target_size"])
    aug = ImageDataGenerator(rescale=1.0 / 255, rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True, brightness_range=[0.8, 1.2])
    plain = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = CustomDataGen(os.path.join(args.split_dir, "train"), aug, batch_size=bs, target_size=size)
    val_gen = CustomDataGen(os.path.join(args.split_dir, "validation"), plain, batch_size=bs, target_size=size)
    test_gen = CustomDataGen(os.path.join(args.split_dir, "test"), plain, batch_size=bs, target_size=size)

    found = len(train_gen.generator.class_indices)
    print(f"classes found: {found} | train/val/test images: "
          f"{train_gen.generator.samples}/{val_gen.generator.samples}/{test_gen.generator.samples}")
    assert found == args.n_classes, f"--n-classes {args.n_classes} but found {found} class folders"

    model = build_model(cfg)
    print("model parameters:", f"{model.count_params():,}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, f"Xception_ArcFace_{args.tag}.keras")
    t = cfg["training"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=t["early_stop_patience"]),
        ReduceLROnPlateau(monitor="val_loss", factor=t["reduce_lr_factor"],
                          patience=t["reduce_lr_patience"], min_lr=t["min_lr"]),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss"),
    ]

    t0 = time.time()
    hist = model.fit(train_gen, validation_data=val_gen, epochs=t["epochs"], callbacks=callbacks)
    minutes = (time.time() - t0) / 60
    pd.DataFrame(hist.history).to_csv(os.path.join(args.out_dir, f"history_{args.tag}.csv"), index=False)
    print(f"training took {minutes:.1f} min for {len(hist.history['loss'])} epochs")

    try:
        model = tf.keras.models.load_model(ckpt, custom_objects={"ArcFace": ArcFace})
        print("evaluating the best checkpoint")
    except Exception as e:
        print("could not reload checkpoint, evaluating last-epoch weights:", repr(e)[:200])

    acc_free, acc_lab = accuracies(model, test_gen, args.n_classes)
    print(f"TEST accuracy, margin-free (real inference): {acc_free:.4f}")
    print(f"TEST accuracy, with labels (model.evaluate-style): {acc_lab:.4f}")

    res = {"tag": args.tag, "n_classes": args.n_classes, "epochs_run": len(hist.history["loss"]),
           "minutes": round(minutes, 1), "test_acc_margin_free": acc_free, "test_acc_with_labels": acc_lab,
           "checkpoint": ckpt}
    with open(os.path.join(args.out_dir, f"results_{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()