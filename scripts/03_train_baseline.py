"""
scripts/03_train_baseline.py

Train the ABRobOcular recognition model (Xception + ArcFace) with THEIR create_model()
and THEIR CustomDataGen, on the class-folder tree built by 02_make_splits.py.
Augmentation and callbacks follow the reference loader/train.py.

Reports two test accuracies:
  - margin-free (zero label vector): real inference, as the attacks and predict.py use
  - with-labels: what model.evaluate() reports (ArcFace margin applied to the true class)

Class indices: every split's generator gets the same sorted list of class folders, so a
subject has the same index in train, validation and test even when a split lacks some
classes (in the cross-session protocol a class with one enrollment bucket has no
validation images). Without the list, flow_from_directory numbers the folders of each
split separately, and every class after a missing one would get a different index in
that split. When every split has every class (the random protocol), the indices are
exactly the ones flow_from_directory picks by default. The run stops if a class has no
training images and warns if one has no validation or test images. The subject -> index
map is saved as class_index_<tag>.json next to the checkpoint.

Usage:
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_small --n-classes 30 --epochs 2 --tag small
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_recognition --n-classes 300 --epochs 10 --tag full
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_cross_session --n-classes 300 --epochs 10 --tag cs
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; must be set before importing tensorflow
import argparse
import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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


class ClassListDataGen(CustomDataGen):
    """The vendored CustomDataGen with an explicit class list (the vendored one takes none)."""

    def __init__(self, directory, datagen, classes, batch_size=32, target_size=(256, 256),
                 class_mode="categorical"):
        self.generator = datagen.flow_from_directory(directory, target_size=target_size, batch_size=batch_size,
                                                     class_mode=class_mode, classes=classes)


def class_list(split_dir, splits=("train", "validation", "test")):
    """Class folder names found in any split, sorted as strings like flow_from_directory sorts them."""
    names = set()
    for split in splits:
        d = os.path.join(split_dir, split)
        if os.path.isdir(d):
            names.update(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
    return sorted(names)


def missing_classes(iterator, classes):
    """Classes without images in the split behind *iterator* (a DirectoryIterator built with *classes*)."""
    counts = np.bincount(iterator.classes, minlength=len(classes))
    return [c for c, n in zip(classes, counts) if n == 0]


def check_coverage(missing):
    """Stop if a class has no training images; return warnings for classes missing elsewhere.

    missing : dict split -> class names without images in that split
    """
    if missing.get("train"):
        raise SystemExit(f"{len(missing['train'])} classes have no training images, e.g. "
                         f"{missing['train'][:10]}; rebuild the split with 02_make_splits.py")
    return [f"WARNING: {len(names)} classes have no {split} images, e.g. {names[:10]}"
            for split, names in missing.items() if split != "train" and names]


def images_by_phase(filenames):
    """Images per capture phase, read from 02's link names <condition>__<phase>__<file>."""
    phases = Counter(os.path.basename(f).split("__")[1] if os.path.basename(f).count("__") >= 2 else "unknown"
                     for f in filenames)
    return dict(sorted(phases.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-dir", default="/kaggle/working/trained_models")
    ap.add_argument("--cfg", default=os.path.join(REPO, "configs", "baseline.yml"))
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    # FIX: legacy tf.keras has no __version__, so report which Keras is really in use
    try:
        import tf_keras
        keras_info = f"tf_keras {tf_keras.__version__}"
    except ImportError:
        keras_info = "tf_keras NOT installed"
    print("TF", tf.__version__, "|", keras_info,
          "| tf.keras comes from:", tf.keras.Model.__module__, "| GPUs:", gpus)
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
    for split in ("train", "validation", "test"):  # with a class list Keras would find 0 images, not fail
        if not os.path.isdir(os.path.join(args.split_dir, split)):
            raise SystemExit(f"{os.path.join(args.split_dir, split)} not found; build it with 02_make_splits.py")
    classes = class_list(args.split_dir)          # one index per subject, the same in every split
    train_gen = ClassListDataGen(os.path.join(args.split_dir, "train"), aug, classes, batch_size=bs,
                                 target_size=size)
    val_gen = ClassListDataGen(os.path.join(args.split_dir, "validation"), plain, classes, batch_size=bs,
                               target_size=size)
    test_gen = ClassListDataGen(os.path.join(args.split_dir, "test"), plain, classes, batch_size=bs,
                                target_size=size)

    found = len(classes)
    print(f"classes found: {found} | train/val/test images: "
          f"{train_gen.generator.samples}/{val_gen.generator.samples}/{test_gen.generator.samples}")
    assert found == args.n_classes, f"--n-classes {args.n_classes} but found {found} class folders"
    gens = {"train": train_gen, "validation": val_gen, "test": test_gen}
    for warning in check_coverage({s: missing_classes(g.generator, classes) for s, g in gens.items()}):
        print(warning)
    phases = {s: images_by_phase(g.generator.filenames) for s, g in gens.items()}
    print("images per phase:", phases)

    model = build_model(cfg)
    print("model parameters:", f"{model.count_params():,}")

    os.makedirs(args.out_dir, exist_ok=True)
    class_index = os.path.join(args.out_dir, f"class_index_{args.tag}.json")
    with open(class_index, "w") as f:
        json.dump(train_gen.generator.class_indices, f, indent=1)
    print("class index written to:", class_index)
    ckpt = os.path.join(args.out_dir, f"Xception_ArcFace_{args.tag}.keras")
    t = cfg["training"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=t["early_stop_patience"]),
        ReduceLROnPlateau(monitor="val_loss", factor=t["reduce_lr_factor"],
                          patience=t["reduce_lr_patience"], min_lr=t["min_lr"]),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss"),
        CSVLogger(os.path.join(args.out_dir, f"log_{args.tag}.csv")),
    ]

    t0 = time.time()
    hist = model.fit(train_gen, validation_data=val_gen, epochs=t["epochs"], callbacks=callbacks,
                    verbose=args.verbose)
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
           "checkpoint": ckpt, "class_index": class_index, "split_dir": args.split_dir,
           "images_by_phase": phases}
    with open(os.path.join(args.out_dir, f"results_{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()