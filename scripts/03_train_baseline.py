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

Training recipe (all optional; without them a run trains exactly as before):
  --cfg          configs/baseline.yml (default: the base paper's 10 epochs, with EarlyStopping
                 and ReduceLROnPlateau settings that never act) or configs/cross_session.yml
                 (a real schedule: up to 30 epochs, early stopping with the best weights
                 restored, learning-rate drops; the file's comments say why)
  --seed         seed of tf.keras.utils.set_random_seed (default 42, the seed of every earlier
                 run); recorded in run_<tag>.json
  --resume       back up the training state after every epoch (tf.keras BackupAndRestore in
                 <out-dir>/backup_<tag>), so that a run Kaggle stops at its 12-hour limit
                 continues from the last finished epoch when the SAME command is run again
                 with the same --out-dir and --tag (copy backup_<tag>/, log_<tag>.csv and the
                 checkpoint back into --out-dir first). BackupAndRestore keeps the model, the
                 optimizer (incl. the learning rate) and the epoch; the state of EarlyStopping,
                 ReduceLROnPlateau and ModelCheckpoint (best monitored loss, waits, best epoch) is
                 kept next to it in backup_<tag>/callback_state.json, and EarlyStopping's best
                 weights are reloaded from the checkpoint. The checkpoint is written to
                 <name>.partial.keras and then renamed, so a stop while it is saved leaves the
                 old best intact. log_<tag>.csv is appended; an epoch that was logged but not
                 backed up before the stop keeps only the row of its second run. The backup is
                 removed right after run_<tag>.json is written; running the same command again
                 after that trains nothing (the record says finished, with the same settings)
                 instead of starting from scratch over the checkpoint, and a run that early
                 stopping had ended trains no further epoch. A --resume re-run whose settings
                 differ from those of the finished run (run_<tag>.json) or of the unfinished one
                 (backup_<tag>/settings.json) stops with the differences: effective cfg (content,
                 not path), seed, arch, reshuffle, allow-missing-masks, masks sha256, class list
                 and images per split (result_settings()). Without --resume a left-over backup
                 is ignored, with a warning. A resumed run is not bit-identical to an
                 uninterrupted one: the random streams (augmentation, batch order) restart
                 from the seed.
  --reshuffle    draw new training batches after every epoch. Keras calls on_epoch_end of the
                 training Sequence after each epoch, but the vendored CustomDataGen does not
                 pass it on to its DirectoryIterator, so without this flag every epoch reuses
                 the batches drawn at the start of the run (fit only shuffles their order).
  --save-weights also write Xception_ArcFace_<tag>.weights.h5: the best checkpoint's weights
                 without the optimizer state (smaller; load them with model.load_weights into
                 a model built from the same cfg)

Architecture (--arch, src/models/anatomy_model.py; default baseline = the vendored create_model,
whose behaviour and outputs are unchanged):
  gap / attn     other pooling heads on the same backbone (no masks)
  attn_mask      attention head trained with CE + lambda x KL(eye-region target || attention map);
                 masks are used only in training
  parts          one attention map per part group, per-part linear embeddings summed (the
                 margin-free cosine is exactly the sum of per-part terms), per-part mask loss
  attnet_gate    AttNet's fixed gate (x5 inside the eye region, x1/5 outside) before the gap head;
                 its model has a third input (the eye-region fraction per 8 x 8 cell) and needs
                 masks at test time as well (04 / 15 --region-masks)
  --masks        masks64_<tag>.npz (21_make_train_masks.py) or a 256 masks_<tag>.npz (17) holding
                 masks for the train, validation and test images; needed by attn_mask, parts and
                 attnet_gate. Images are matched to masks like 17 / 18 match them (condition, phase,
                 file name); a missing mask stops the run with a count unless --allow-missing-masks
                 (mask-loss weight 0 for that image, attnet_gate region 0). The masks go through
                 the same augmentation as the image (src/data/masked_generator.py).
  --mask-loss-weight  lambda (default anatomy.mask_loss_weight of --cfg; configs/anatomy.yml is
                 configs/cross_session.yml plus the anatomy section)
  The training loop, callbacks, --resume, --reshuffle and --save-weights work as for the baseline.
  The checkpoint is always the plain inference model (dual input; triple for attnet_gate) under the
  same name Xception_ArcFace_<tag>.keras (every arch is Xception + ArcFace): put the arch in --tag.
  For attn_mask and parts, val_loss is CE + lambda x mask loss, and the logs add ce, mask_loss,
  val_ce and val_mask_loss. They still optimise CE + lambda x mask loss, but early stopping, the
  learning-rate drops, the best checkpoint and best_epoch follow val_ce (the validation CE, the
  same quantity as the val_loss of every other arch), so every arm is selected and scheduled by
  the same rule and only the head (and its training loss) differs. run_<tag>.json adds arch,
  monitor (val_ce or val_loss) and best_monitor (its value at the best epoch), anatomy (the
  effective settings, also under cfg), masks, masks_sha256, mask_classes, mask_coverage per split
  and, for attn_mask and parts, mask_area_below_min (images whose eye region, or each part,
  covers less than anatomy.min_part_area of the image and so get mask-loss weight 0); a baseline
  record has exactly the fields it had before.

Writes to --out-dir:
  Xception_ArcFace_<tag>.keras       best checkpoint (lowest val_loss; val_ce for attn_mask, parts)
  class_index_<tag>.json             subject -> class index
  log_<tag>.csv                      CSVLogger, one row per epoch (all sessions with --resume)
  history_<tag>.csv                  per-epoch metrics (all sessions with --resume)
  results_<tag>.json                 test accuracies, margin-free and with labels
  run_<tag>.json                     run record, written right after training (before the test
                                     evaluation): git commit and dirty flag, argv and parsed
                                     args, the effective cfg, package versions, split dir,
                                     images per split and per phase, sha256 of the sorted class
                                     list (UTF-8, names joined with "\n"), epochs run (all
                                     sessions), best epoch (1-based, by the monitored loss) and
                                     its val_loss, training minutes (this session and all
                                     sessions), resumed yes/no,
                                     checkpoint path and sha256, and the weights file with
                                     --save-weights
  Xception_ArcFace_<tag>.weights.h5  with --save-weights
  backup_<tag>/                      with --resume, while training is unfinished

Usage:
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_small --n-classes 30 --epochs 2 --tag small
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_recognition --n-classes 300 --epochs 10 --tag full
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_cross_session --n-classes 300 --epochs 10 --tag cs
Cross-session model with the real schedule, resumable over Kaggle sessions (run the same
command again in the next session):
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_cross_session --n-classes 300 \
      --cfg configs/cross_session.yml --reshuffle --resume --save-weights --tag cs
Mask-supervised attention model (masks from 21_make_train_masks.py), same recipe:
  python scripts/03_train_baseline.py --split-dir /tmp/ocular_cross_session --n-classes 300 \
      --cfg configs/anatomy.yml --arch attn_mask --masks /kaggle/working/train_masks/masks64_cs.npz \
      --reshuffle --resume --save-weights --tag cs_attn_mask_s1
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; must be set before importing tensorflow
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (BackupAndRestore, Callback, CSVLogger, EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.data.masked_generator import MODES, MaskedDataGen, MaskTable, MissingMasksError  # noqa: E402
from src.models.anatomy_model import (ARCHS, CUSTOM_OBJECTS, MASK_ARCHS, MASK_LOSS_ARCHS,  # noqa: E402
                                      anatomy_settings, build_anatomy_model, feature_grid, make_trainer)
from src.models.recognition_model import ArcFace, CustomDataGen, build_model, load_cfg  # noqa: E402,F401

MASK_GRID = 8                     # feature cells per side that the mask targets are computed on
ANATOMY_ARGS = ("arch", "masks", "mask_loss_weight", "allow_missing_masks")


def monitor_for(arch):
    """The validation loss that EarlyStopping, ReduceLROnPlateau, ModelCheckpoint and best_epoch() follow.

    val_loss for every arch but attn_mask and parts. Their MaskTrainer logs val_loss = CE + lambda x
    mask loss, and val_ce = the compiled CE alone (with the model's regularisation losses, exactly what
    val_loss is for the other archs). Selecting and scheduling them on val_ce keeps the rule the same
    for every arm, so a difference between, e.g., attn and attn_mask comes from the head and its
    training loss, not from a different stopping epoch.
    """
    return "val_ce" if arch in MASK_LOSS_ARCHS else "val_loss"


def accuracies(model, gen, n_classes):
    """Return (margin-free accuracy, with-labels accuracy) over a CustomDataGen (or a MaskedDataGen:
    the targets it adds are ignored, the region input of attnet_gate is fed to the model)."""
    ok_free = ok_lab = total = 0
    for i in range(len(gen)):
        (x, y, *rest), _ = gen[i]
        extra = list(rest[:len(model.inputs) - 2])
        y_true = y.argmax(1)
        p_free = model.predict_on_batch([x, np.zeros((len(x), n_classes), dtype="float32")] + extra)
        p_lab = model.predict_on_batch([x, y] + extra)
        ok_free += int((np.asarray(p_free).argmax(1) == y_true).sum())
        ok_lab += int((np.asarray(p_lab).argmax(1) == y_true).sum())
        total += len(x)
    return ok_free / total, ok_lab / total


class ClassListDataGen(CustomDataGen):
    """The vendored CustomDataGen with an explicit class list (the vendored one takes none).

    reshuffle : pass Keras' on_epoch_end on to the DirectoryIterator, which then draws a new
                image order, so that every epoch has new batches. The vendored class drops the
                call, and its batches keep the composition drawn at the start of the run.
    """

    def __init__(self, directory, datagen, classes, batch_size=32, target_size=(256, 256),
                 class_mode="categorical", reshuffle=False):
        self.generator = datagen.flow_from_directory(directory, target_size=target_size, batch_size=batch_size,
                                                     class_mode=class_mode, classes=classes)
        self.reshuffle = reshuffle

    def on_epoch_end(self):
        if self.reshuffle:
            self.generator.on_epoch_end()


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


def class_list_sha256(classes):
    """sha256 of the sorted class list: UTF-8, the names joined with "\\n"."""
    return hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()


def file_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    """(commit, has uncommitted changes) of the repository this script runs from; (None, None) without git."""
    def git(*cmd):
        return subprocess.run(["git", "--no-optional-locks", *cmd], cwd=REPO, capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD") or None, bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def package_versions():
    out = {"python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__}
    try:
        out["tf_keras"] = importlib.metadata.version("tf-keras")
    except importlib.metadata.PackageNotFoundError:
        out["tf_keras"] = None
    return out


def backup_exists(backup_dir):
    """True if BackupAndRestore left a training state in *backup_dir* (it writes to <backup_dir>/chief)."""
    return tf.train.latest_checkpoint(os.path.join(backup_dir, "chief")) is not None


def backup_epoch(backup_dir):
    """Last epoch (0-based) held by the backup in *backup_dir*; None without a backup or if it cannot be read."""
    try:
        latest = tf.train.latest_checkpoint(os.path.join(backup_dir, "chief"))
        return int(tf.train.load_checkpoint(latest).get_tensor("ckpt_saved_epoch/.ATTRIBUTES/VARIABLE_VALUE"))
    except Exception:
        return None


class ResumeState(Callback):
    """What BackupAndRestore does not back up: the state of EarlyStopping, ReduceLROnPlateau and
    ModelCheckpoint, and the training minutes of earlier sessions.

    Without it a resumed run restarts those callbacks from scratch: ModelCheckpoint overwrites
    the best checkpoint with the first resumed epoch, EarlyStopping and ReduceLROnPlateau forget
    how long val_loss has not improved, and EarlyStopping with restore_best_weights "restores"
    the weights of the first resumed epoch.

    The state after every epoch is kept by epoch in *path* (JSON, replaced atomically). It is
    written before BackupAndRestore backs up the same epoch (this callback comes first in the
    list), so the state of the epoch a backup holds is always there; on resume, the first
    on_epoch_begin restores the state of the epoch before it. A session with no epoch left to
    train (stopped after the last backup, before the run record) restores the state of the last
    epoch at on_train_end instead, so that the run record still covers every session.
    """

    def __init__(self, path, early_stop, reduce_lr, checkpoint, ckpt_path):
        super().__init__()
        self.path, self.ckpt_path = path, ckpt_path
        self.early_stop, self.reduce_lr, self.checkpoint = early_stop, reduce_lr, checkpoint
        self.monitor = early_stop.monitor
        self.epochs = {}
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            # a state without "monitor" was written when every arch monitored val_loss
            if saved.get("monitor", "val_loss") == self.monitor:
                self.epochs = saved["epochs"]
            else:
                print(f"WARNING: {path} holds the callback state of a run that monitored "
                      f"{saved.get('monitor', 'val_loss')}, this run monitors {self.monitor}: EarlyStopping, "
                      "ReduceLROnPlateau and ModelCheckpoint start afresh")
        self.initial_epoch = None
        self.minutes_before = 0.0
        self.t0 = time.time()

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_epoch is not None:
            return
        self.initial_epoch = epoch
        if epoch > 0:
            self.restore(epoch - 1)

    def on_train_end(self, logs=None):
        if self.initial_epoch is None and self.epochs:
            self.restore(max(map(int, self.epochs)), resuming=False)

    def early_stopped_at(self):
        """Last epoch (0-based) at which EarlyStopping had stopped the run in the saved state, else None."""
        stopped = [int(e) for e, s in self.epochs.items() if s["early_stopping"]["stopped_epoch"] > 0]
        return max(stopped) if stopped else None

    def restore(self, epoch, resuming=True):
        """Put back the callback state saved after *epoch*; *resuming*: also EarlyStopping's best weights."""
        state = self.epochs.get(str(epoch))
        if state is None:
            print(f"WARNING: no callback state for epoch {epoch + 1} in {self.path}; EarlyStopping, "
                  "ReduceLROnPlateau and ModelCheckpoint start afresh")
            return
        es, rlr = state["early_stopping"], state["reduce_lr"]
        self.early_stop.wait, self.early_stop.best = es["wait"], es["best"]
        self.early_stop.best_epoch, self.early_stop.stopped_epoch = es["best_epoch"], es["stopped_epoch"]
        self.reduce_lr.wait, self.reduce_lr.best = rlr["wait"], rlr["best"]
        self.reduce_lr.cooldown_counter = rlr["cooldown_counter"]
        self.minutes_before = state["minutes"]
        if os.path.exists(self.ckpt_path):
            self.checkpoint.best = state["checkpoint"]["best"]
        else:   # without the old best checkpoint, the next epoch must write one whatever its loss
            print(f"WARNING: {self.ckpt_path} not found (copy it back into --out-dir before resuming); the next "
                  "epoch writes a new checkpoint, and the best epoch of the run record may not be the one saved")
        if resuming and self.early_stop.restore_best_weights and os.path.exists(self.ckpt_path):
            # the best epoch's weights are the ones ModelCheckpoint saved (same monitor, same comparison);
            # loaded on the CPU so that the GPU does not hold a second copy of the model
            try:
                with tf.device("/cpu:0"):
                    best = tf.keras.models.load_model(self.ckpt_path, custom_objects=CUSTOM_OBJECTS,
                                                      compile=False)
                self.early_stop.best_weights = best.get_weights()
                del best
            except Exception as e:
                print(f"WARNING: could not read {self.ckpt_path}, EarlyStopping cannot restore the best weights:",
                      repr(e)[:200])
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))   # restored by BackupAndRestore
        if not np.isclose(lr, state["lr"], rtol=1e-6):
            print(f"WARNING: learning rate after the restore {lr:g} differs from the saved {state['lr']:g}")
        where = f"resuming at epoch {epoch + 2}" if resuming else f"no epoch left to train after epoch {epoch + 1}"
        print(f"{where}: best {self.monitor} {es['best']:.4f} (epoch {es['best_epoch'] + 1}), lr {lr:g}, "
              f"{self.minutes_before:.1f} min trained before")

    def on_epoch_end(self, epoch, logs=None):
        self.epochs[str(epoch)] = {
            "minutes": self.minutes(),
            "lr": float(tf.keras.backend.get_value(self.model.optimizer.lr)),
            "early_stopping": {"wait": int(self.early_stop.wait), "best": float(self.early_stop.best),
                               "best_epoch": int(self.early_stop.best_epoch),
                               "stopped_epoch": int(self.early_stop.stopped_epoch)},
            "reduce_lr": {"wait": int(self.reduce_lr.wait), "best": float(self.reduce_lr.best),
                          "cooldown_counter": int(self.reduce_lr.cooldown_counter)},
            "checkpoint": {"best": float(self.checkpoint.best)},
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".tmp", "w") as f:
            json.dump({"monitor": self.monitor, "epochs": self.epochs}, f, indent=1)
        os.replace(self.path + ".tmp", self.path)

    def minutes(self):
        """Training minutes of the epochs kept: earlier sessions up to the restored epoch, plus this session."""
        return self.minutes_before + (time.time() - self.t0) / 60


class AtomicModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that writes the new best model to <name>.partial.keras and then renames it.

    Keras writes the .keras zip in place, so a process killed while saving (Kaggle's time limit)
    would leave a broken best checkpoint, the file a resumed run reloads its best weights from.
    """

    def _save_handler(self, filepath):
        root, ext = os.path.splitext(filepath)
        partial = f"{root}.partial{ext}"
        super()._save_handler(partial)
        os.replace(partial, filepath)


def make_callbacks(t, ckpt, log_path, backup_dir=None, resumed=False, monitor="val_loss"):
    """Callbacks for cfg["training"] *t*; with *backup_dir*, the training state is backed up after every epoch.

    EarlyStopping, ReduceLROnPlateau and ModelCheckpoint follow *monitor* (monitor_for(arch)).

    The order matters: ReduceLROnPlateau adds "lr" to the logs before CSVLogger writes them, and
    BackupAndRestore comes last, so a backed-up epoch has its checkpoint, log row and callback
    state written. CSVLogger appends only when *resumed* (a fresh start replaces an old log).
    With *backup_dir* the checkpoint is written atomically, and BackupAndRestore keeps its backup
    when training ends: main() removes it after the run record, so that a session stopped in
    between resumes with nothing left to train instead of training again from scratch.
    """
    checkpoint = AtomicModelCheckpoint if backup_dir else ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor=monitor, patience=t["early_stop_patience"],
                      restore_best_weights=bool(t.get("early_stop_restore_best", False))),
        ReduceLROnPlateau(monitor=monitor, factor=t["reduce_lr_factor"],
                          patience=t["reduce_lr_patience"], min_lr=t["min_lr"]),
        checkpoint(ckpt, save_best_only=True, monitor=monitor),
        CSVLogger(log_path, append=resumed),
    ]
    if backup_dir:
        callbacks += [ResumeState(os.path.join(backup_dir, "callback_state.json"), *callbacks[:3], ckpt),
                      BackupAndRestore(backup_dir, delete_checkpoint=False)]
    return callbacks


def merged_log(path):
    """Rows of a CSVLogger file written over several sessions, one per epoch: the last row of each
    epoch (an epoch logged but not backed up before a stop is run and logged again)."""
    log = pd.read_csv(path)
    return log.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def best_epoch(log, monitor="val_loss"):
    """(1-based epoch with the lowest *monitor*, the val_loss of that epoch) of a CSVLogger table;
    (None, None) without the monitor column. With the default monitor: the epoch of the lowest
    val_loss and that val_loss, the epoch ModelCheckpoint keeps."""
    if monitor not in log or not log[monitor].notna().any():
        return None, None
    row = log.loc[log[monitor].idxmin()]
    return int(row.epoch) + 1, (float(row.val_loss) if "val_loss" in log else None)


def save_best_weights(ckpt, path, fallback_model):
    """Weights-only copy of the best checkpoint, loaded without its optimizer (save_weights of a compiled
    model also writes the Adam slots); the in-memory weights if the checkpoint cannot be read.

    Returns where the weights came from: "best_checkpoint" or "in_memory_weights".
    """
    try:
        with tf.device("/cpu:0"):       # no second copy of the model on the GPU
            best = tf.keras.models.load_model(ckpt, custom_objects=CUSTOM_OBJECTS, compile=False)
        source = "best_checkpoint"
    except Exception as e:
        print("could not reload checkpoint, saving the in-memory weights:", repr(e)[:200])
        best, source = fallback_model, "in_memory_weights"
    best.save_weights(path)
    print("weights written to:", path)
    return source


def write_json(path, obj):
    """Write obj as JSON atomically (a stop mid-write leaves the old file)."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(path + ".tmp", path)


def finish_run(args, record, run_path, ckpt, model, test_gen, epochs_run, minutes, class_index, phases):
    """After training: evaluate the best checkpoint, write the weights-only copy (--save-weights) and
    results_<tag>.json. Also run by a --resume re-run whose earlier session stopped after the run record
    but before these files (model is then None and the checkpoint must load)."""
    try:
        model = tf.keras.models.load_model(ckpt, custom_objects=CUSTOM_OBJECTS)
        print("evaluating the best checkpoint")
    except Exception as e:
        if model is None:
            raise
        print("could not reload checkpoint, evaluating last-epoch weights:", repr(e)[:200])
        model = getattr(model, "net", model)          # a MaskTrainer: its inference model

    if args.save_weights:
        weights_path = os.path.join(args.out_dir, f"Xception_ArcFace_{args.tag}.weights.h5")
        record["weights_source"] = save_best_weights(ckpt, weights_path, model)
        record["weights"], record["weights_sha256"] = weights_path, file_sha256(weights_path)
        write_json(run_path, record)

    acc_free, acc_lab = accuracies(model, test_gen, args.n_classes)
    print(f"TEST accuracy, margin-free (real inference): {acc_free:.4f}")
    print(f"TEST accuracy, with labels (model.evaluate-style): {acc_lab:.4f}")

    res = {"tag": args.tag, "n_classes": args.n_classes, "epochs_run": epochs_run,
           "minutes": round(minutes, 1), "test_acc_margin_free": acc_free, "test_acc_with_labels": acc_lab,
           "checkpoint": ckpt, "class_index": class_index, "split_dir": args.split_dir,
           "images_by_phase": phases}
    with open(os.path.join(args.out_dir, f"results_{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def run_args(args):
    """The arguments a run record keeps. A baseline run keeps exactly the arguments of the records written
    before --arch existed. A --resume re-run compares result_settings(), not these."""
    recorded = dict(vars(args))
    if args.arch == "baseline" and not args.masks and args.mask_loss_weight is None and not args.allow_missing_masks:
        for key in ANATOMY_ARGS:
            recorded.pop(key, None)
    return recorded


def jsonable(obj):
    """*obj* as it reads back from a JSON file written by write_json (tuples become lists, numpy scalars numbers)."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def result_settings(args, cfg, classes, images_per_split, masks_sha256):
    """The settings the trained model depends on, which a --resume re-run must share with the run it continues
    or finds finished.

    cfg is the effective config (the --cfg file's content with --epochs, --batch-size, --n-classes and, for the
    anatomy archs, the anatomy settings incl. --mask-loss-weight applied), so the path of --cfg does not matter,
    only what it sets. The split is identified by its class list and images per split, the masks by their
    sha256, not by their paths, so that the same data under another directory (another Kaggle mount) counts as
    the same. Not compared: --verbose, --tag, --out-dir, --resume, --save-weights (the weights-only copy is
    written from the best checkpoint at the end) and the git commit (a resumed session may run fixed code).
    """
    return jsonable({"cfg": cfg, "seed": args.seed, "arch": args.arch, "reshuffle": bool(args.reshuffle),
                     "allow_missing_masks": bool(args.allow_missing_masks), "masks_sha256": masks_sha256,
                     "class_list_sha256": class_list_sha256(classes), "images_per_split": images_per_split})


def recorded_settings(record):
    """result_settings() of the run behind *record* (run_<tag>.json). A baseline record has no arch, masks_sha256
    or anatomy args (the fields of the records before --arch), which then take their baseline values; cfg, the
    class list and the images per split are compared only when the record has them."""
    args = record.get("args") or {}
    settings = {"seed": record.get("seed", args.get("seed", 42)),
                "arch": record.get("arch", args.get("arch", "baseline")),
                "reshuffle": bool(args.get("reshuffle", False)),
                "allow_missing_masks": bool(args.get("allow_missing_masks", False)),
                "masks_sha256": record.get("masks_sha256")}
    settings.update({k: record[k] for k in ("cfg", "class_list_sha256", "images_per_split") if k in record})
    return settings


def settings_differences(saved, current, prefix=""):
    """'<setting>: <saved> -> <current>' for every setting of *saved* that *current* differs in (cfg key by key)."""
    out = []
    for key in saved:
        old, new = saved[key], current.get(key) if isinstance(current, dict) else None
        if isinstance(old, dict) and isinstance(new, dict):
            out += settings_differences(old, new, f"{prefix}{key}.")
            out += [f"{prefix}{key}.{k}: (none) -> {new[k]!r}" for k in new if k not in old]
        elif old != new:
            out.append(f"{prefix}{key}: {old!r} -> {new!r}")
    return out


def check_anatomy_args(ap, args):
    """Stop on --arch / --masks / --mask-loss-weight combinations that would silently do nothing."""
    if args.arch in MASK_ARCHS and not args.masks:
        ap.error(f"--arch {args.arch} needs --masks masks64_<tag>.npz (scripts/21_make_train_masks.py)")
    if args.masks and args.arch not in MASK_ARCHS:
        ap.error(f"--masks is used only by --arch {', '.join(MASK_ARCHS)}, not {args.arch}")
    if args.mask_loss_weight is not None and args.arch not in MASK_LOSS_ARCHS:
        ap.error(f"--mask-loss-weight is used only by --arch {', '.join(MASK_LOSS_ARCHS)}")
    if args.allow_missing_masks and not args.masks:
        ap.error("--allow-missing-masks needs --masks")


def masked_generators(args, table, settings, classes, aug, plain, bs, size):
    """Train / validation / test MaskedDataGen of --arch, like 03's ClassListDataGen with masks added."""
    gens = []
    for split, datagen, reshuffle in (("train", aug, args.reshuffle), ("validation", plain, False),
                                      ("test", plain, False)):
        try:
            gens.append(MaskedDataGen(os.path.join(args.split_dir, split), datagen, classes, table, MODES[args.arch],
                                      settings["part_groups"], settings["eps"], settings["min_part_area"],
                                      batch_size=bs, target_size=size, reshuffle=reshuffle,
                                      allow_missing=args.allow_missing_masks, grid=MASK_GRID))
        except MissingMasksError as e:
            raise SystemExit(str(e))
    return gens


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
    ap.add_argument("--seed", type=int, default=42, help="seed of tf.keras.utils.set_random_seed (default 42)")
    ap.add_argument("--resume", action="store_true",
                    help="back up the training state after every epoch in <out-dir>/backup_<tag>, and continue "
                         "from it when the same command is run again")
    ap.add_argument("--reshuffle", action="store_true",
                    help="new training batches every epoch (the vendored generator keeps those of the first)")
    ap.add_argument("--save-weights", action="store_true",
                    help="also write Xception_ArcFace_<tag>.weights.h5 (best weights, no optimizer state)")
    ap.add_argument("--arch", choices=ARCHS, default="baseline",
                    help="baseline (the vendored model, default), gap, attn, attn_mask, parts or attnet_gate "
                         "(src/models/anatomy_model.py); the checkpoint keeps the name Xception_ArcFace_<tag>.keras "
                         "and run_<tag>.json records the arch, so put the arch in --tag")
    ap.add_argument("--masks", default=None,
                    help="masks64_<tag>.npz from 21_make_train_masks.py (or a 256 masks_<tag>.npz from 17) covering "
                         "the train, validation and test images; needed by attn_mask, parts and attnet_gate")
    ap.add_argument("--mask-loss-weight", type=float, default=None,
                    help="lambda of the mask loss (attn_mask, parts); default anatomy.mask_loss_weight of --cfg")
    ap.add_argument("--allow-missing-masks", action="store_true",
                    help="train on images without a mask with mask-loss weight 0 (attnet_gate: region 0) "
                         "instead of stopping")
    args = ap.parse_args()
    check_anatomy_args(ap, args)

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
    tf.keras.utils.set_random_seed(args.seed)

    bs = cfg["training"]["batch_size"]
    size = tuple(cfg["data"]["target_size"])
    aug = ImageDataGenerator(rescale=1.0 / 255, rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True, brightness_range=[0.8, 1.2])
    plain = ImageDataGenerator(rescale=1.0 / 255)
    for split in ("train", "validation", "test"):  # with a class list Keras would find 0 images, not fail
        if not os.path.isdir(os.path.join(args.split_dir, split)):
            raise SystemExit(f"{os.path.join(args.split_dir, split)} not found; build it with 02_make_splits.py")
    classes = class_list(args.split_dir)          # one index per subject, the same in every split
    settings = table = None
    if args.arch != "baseline":                   # the effective anatomy settings go into the run record's cfg
        cfg["anatomy"] = anatomy_settings(cfg)
        if args.mask_loss_weight is not None:
            cfg["anatomy"]["mask_loss_weight"] = args.mask_loss_weight
        settings = cfg["anatomy"]
    if args.arch in MODES:
        table = MaskTable(args.masks)
        print(f"masks: {len(table)} maps from {args.masks} | classes {table.class_names} | part groups "
              f"{settings['part_groups']}")
        train_gen, val_gen, test_gen = masked_generators(args, table, settings, classes, aug, plain, bs, size)
        print("mask coverage:", {s: g.coverage() for s, g in
                                 (("train", train_gen), ("validation", val_gen), ("test", test_gen))})
        if args.arch in MASK_LOSS_ARCHS:
            print(f"images whose mask target gets weight 0 (eye region / part below min_part_area "
                  f"{settings['min_part_area']}):",
                  {s: g.below_min_area() for s, g in (("train", train_gen), ("validation", val_gen))})
    else:
        train_gen = ClassListDataGen(os.path.join(args.split_dir, "train"), aug, classes, batch_size=bs,
                                     target_size=size, reshuffle=args.reshuffle)
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
    images_per_split = {s: int(g.generator.samples) for s, g in gens.items()}
    masks_sha256 = file_sha256(args.masks) if args.masks else None

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, f"Xception_ArcFace_{args.tag}.keras")
    log_path = os.path.join(args.out_dir, f"log_{args.tag}.csv")
    run_path = os.path.join(args.out_dir, f"run_{args.tag}.json")
    backup_dir = os.path.join(args.out_dir, f"backup_{args.tag}")
    settings_path = os.path.join(backup_dir, "settings.json")
    current = result_settings(args, cfg, classes, images_per_split, masks_sha256)
    if args.resume and os.path.exists(run_path) and os.path.exists(ckpt):
        try:
            with open(run_path) as f:
                previous = json.load(f)
        except ValueError:
            previous = {}
        changed = settings_differences(recorded_settings(previous), current) if previous.get("finished") else []
        if changed:
            # training again would start from scratch and overwrite the finished run's checkpoint
            raise SystemExit(f"{run_path} records a finished run '{args.tag}' with other settings than this command "
                             f"({'; '.join(changed)}). Nothing was trained: use another --tag or --out-dir to train "
                             "with these settings, or run the recorded command to complete the finished run.")
        if previous.get("finished"):
            # the same command has trained to the end; a backup still there was left by a session that
            # stopped between the run record and the removal of the backup
            shutil.rmtree(backup_dir, ignore_errors=True)
            weights = os.path.join(args.out_dir, f"Xception_ArcFace_{args.tag}.weights.h5")
            done = os.path.exists(os.path.join(args.out_dir, f"results_{args.tag}.json")) and (
                not args.save_weights or (os.path.exists(weights) and previous.get("weights_sha256")))
            if done:
                print(f"{run_path} says this command has finished run '{args.tag}': nothing to resume "
                      "(use another --tag or --out-dir to train again)")
            else:                              # stopped after the run record, before the evaluation
                print(f"{run_path} says training finished; writing the missing weights / test results")
                finish_run(args, previous, run_path, ckpt, None, test_gen, previous.get("epochs_run"),
                           previous.get("minutes_all_sessions") or 0.0,
                           os.path.join(args.out_dir, f"class_index_{args.tag}.json"), phases)
            return
    resumed = args.resume and backup_exists(backup_dir)
    if not args.resume and backup_exists(backup_dir):
        print(f"WARNING: {backup_dir} holds the backup of an unfinished run; without --resume it is ignored "
              "and this run starts from scratch")
    if resumed and os.path.exists(settings_path):
        with open(settings_path) as f:
            changed = settings_differences(json.load(f), current)
        if changed:                            # continuing would mix two settings in one model and one record
            raise SystemExit(f"{backup_dir} holds an unfinished run '{args.tag}' with other settings than this "
                             f"command ({'; '.join(changed)}). Nothing was trained: run the original command again "
                             "to continue it, or use another --tag or --out-dir (or remove the backup) to start anew.")
    elif resumed:
        print(f"WARNING: {backup_dir} has no settings.json (written by an older 03), so this command's settings "
              "cannot be checked against those of the run it continues; they are recorded for the next sessions")
    stale_state = os.path.join(backup_dir, "callback_state.json")
    if args.resume and not resumed and os.path.exists(stale_state):
        os.remove(stale_state)                 # left by an attempt killed before its first backup
    if args.resume and not (resumed and os.path.exists(settings_path)):
        os.makedirs(backup_dir, exist_ok=True)  # the settings the backups of this run are made with
        write_json(settings_path, current)
    partial = ckpt[:-len(".keras")] + ".partial.keras"
    if args.resume and os.path.exists(partial):
        os.remove(partial)                     # a checkpoint save cut off by a stop; the old best is intact
    if args.resume:
        print("resume:", f"continuing from the backup in {backup_dir}" if resumed else
              f"no backup yet, starting from scratch; backups go to {backup_dir}")

    if args.arch == "baseline":
        model = build_model(cfg)
    else:
        model = build_anatomy_model(cfg, args.arch)
        if args.arch in MODES and feature_grid(model) != (MASK_GRID, MASK_GRID):
            raise SystemExit(f"the mask targets are {MASK_GRID} x {MASK_GRID} cells, but the feature map is "
                             f"{feature_grid(model)}: use a 256 x 256 input")
        if args.arch in MASK_LOSS_ARCHS:          # trains with CE + mask loss, saves the inference model
            model = make_trainer(model, cfg)
        print(f"arch: {args.arch} | anatomy settings: {settings}")
    print("model parameters:", f"{model.count_params():,}")

    class_index = os.path.join(args.out_dir, f"class_index_{args.tag}.json")
    with open(class_index, "w") as f:
        json.dump(train_gen.generator.class_indices, f, indent=1)
    print("class index written to:", class_index)
    t = cfg["training"]
    monitor = monitor_for(args.arch)
    if monitor != "val_loss":
        print(f"early stopping, learning-rate drops and the best checkpoint follow {monitor} (the validation CE), "
              "as val_loss does for the archs without a mask loss")
    callbacks = make_callbacks(t, ckpt, log_path, backup_dir if args.resume else None, resumed, monitor)
    state = next((c for c in callbacks if isinstance(c, ResumeState)), None)
    epochs = t["epochs"]
    stopped = state.early_stopped_at() if resumed else None
    saved = backup_epoch(backup_dir) if stopped is not None else None
    if saved is not None and stopped <= saved:  # EarlyStopping had ended the run and that epoch is backed up;
        epochs = min(epochs, stopped + 1)      # the session stopped before the run record: train no further
        print(f"EarlyStopping had stopped the run after epoch {stopped + 1}")

    t0 = time.time()
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks,
                    verbose=args.verbose)
    minutes = (time.time() - t0) / 60
    epochs_session = len(hist.history.get("loss", []))
    if args.resume:
        log = merged_log(log_path)             # every session's epochs, one row each
        log.to_csv(log_path, index=False)
        history = log.drop(columns="epoch")
        history = history[[k for k in hist.history if k in history]] if hist.history else history
    else:
        log = pd.read_csv(log_path)
        history = pd.DataFrame(hist.history)
    history.to_csv(os.path.join(args.out_dir, f"history_{args.tag}.csv"), index=False)
    epochs_run = len(log)
    print(f"training took {minutes:.1f} min for {epochs_session} epochs"
          + (f" ({epochs_run} over all sessions)" if args.resume else ""))

    # the run record goes out first, and with --resume the backup is removed only after it: a session
    # stopped in between resumes with nothing left to train, or finds the record and trains nothing
    commit, dirty = git_state()
    best, best_val_loss = best_epoch(log, monitor)
    record = {
        "tag": args.tag, "finished": True, "git_commit": commit, "git_dirty": dirty,
        "argv": sys.argv, "args": run_args(args), "seed": args.seed, "cfg": cfg, "versions": package_versions(),
        "split_dir": args.split_dir,
        "images_per_split": images_per_split,
        "images_by_phase": phases,
        "n_classes": len(classes), "class_list_sha256": class_list_sha256(classes),
        "epochs_run": epochs_run, "epochs_this_session": epochs_session,
        "best_epoch": best, "best_val_loss": best_val_loss,
        "stopped_early": bool(callbacks[0].stopped_epoch > 0),
        "minutes": round(minutes, 2), "minutes_all_sessions": round(state.minutes() if state else minutes, 2),
        "resumed": bool(resumed),
        "resumed_at_epoch": state.initial_epoch + 1 if resumed and state.initial_epoch is not None else None,
        "checkpoint": ckpt, "checkpoint_sha256": file_sha256(ckpt) if os.path.exists(ckpt) else None,
        "weights": None, "weights_source": None, "weights_sha256": None,
        "class_index": class_index, "log": log_path,
    }
    if args.arch != "baseline":            # a baseline record keeps exactly the fields it had before --arch
        best_monitor = (float(log.loc[log.epoch == best - 1, monitor].iloc[0])
                        if best is not None and monitor in log else None)
        record.update({"arch": args.arch, "monitor": monitor, "best_monitor": best_monitor, "anatomy": settings,
                       "masks": os.path.abspath(args.masks) if args.masks else None,
                       "masks_sha256": masks_sha256,
                       "mask_classes": table.class_names if table else None,
                       "mask_coverage": {s: g.coverage() for s, g in gens.items()} if table else None})
        if args.arch in MASK_LOSS_ARCHS:
            record["mask_area_below_min"] = {s: g.below_min_area() for s, g in gens.items()}

    write_json(run_path, record)
    print("run record written to:", run_path)
    if args.resume:
        shutil.rmtree(backup_dir, ignore_errors=True)   # training has ended: backup and callback state

    finish_run(args, record, run_path, ckpt, model, test_gen, epochs_run, minutes, class_index, phases)


if __name__ == "__main__":
    main()