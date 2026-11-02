"""
scripts/02_make_splits.py

Splits for three protocols (--protocol).

random (default): the paper-style protocol. Keep subjects with ID in [1141, 3279] plus
7000 (300 classes), pool all images of a subject (all devices, lighting, eyes, phases),
and split each class randomly 60/20/20 into train/validation/test.

The test split is pinned to splits/visob_test.csv: the 13,868 test images of the run that
trained the full baseline checkpoint. The remaining images of each class are split 75/25
into train/validation (60/20 of the total). Before this file existed, the split depended on
the order the disk listed files in, so a new Kaggle session could put training images into
"test" (on 2026-10-22 only 4,166 of the original test images stayed in test). Pass
--test-list none for a fresh random 60/20/20 split (only for a newly trained model).

Consecutive frame indices in a (subject, condition, eye, phase) bucket are near-duplicate
frames, and 98.2 % of the random split's test images share a bucket with training images,
so its accuracy mostly measures matching a frame to its neighbour. Keep it only to
reproduce the base paper with the existing checkpoint.

cross_session: the same 300 classes. Test = every verification-phase image; train and
validation come from the enrollment-phase images, with validation made of whole
(subject, condition, eye) buckets, so no bucket is split between train and validation and
test images come from the other capture session. Each class gives --val-frac of its
enrollment buckets to validation (at least one, never all); a class with a single
enrollment bucket keeps it in train and is reported. Use it for every newly trained model.

open_set: no training and no tree. Lists every image (both phases) of the subjects outside
the 300 classes, for open-set verification in 15_eval_verification.py.

Builds the class-folder tree the ABRobOcular loader expects, using symlinks (no copying):
  <out-dir>/{train,validation,test}/<subject>/<condition>__<phase>__<filename>.png
and writes a manifest with the split and link path for every image. --write-split-csvs DIR
also writes DIR/<protocol>_{train,validation,test}.csv (relpath, subject) so the lists can
be committed (open_set writes only open_set_test.csv).

Usage:
  python scripts/02_make_splits.py                       # all 300 classes
  python scripts/02_make_splits.py --max-classes 30 --out-dir /tmp/ocular_small \
         --out-manifest /kaggle/working/manifest_small.csv   # quick debug subset
  python scripts/02_make_splits.py --protocol cross_session --write-split-csvs /kaggle/working/splits
         # tree /tmp/ocular_cross_session, manifest /kaggle/working/manifest_split_cross_session.csv
  python scripts/02_make_splits.py --protocol open_set --write-split-csvs /kaggle/working/splits
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd


def find_root(base="/kaggle/input"):
    for r, dirs, _ in os.walk(base):
        if "VISOB_Augmented_Recent" in dirs:
            return os.path.join(r, "VISOB_Augmented_Recent")
    raise FileNotFoundError("VISOB_Augmented_Recent not found under " + base)


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
# default (tree, manifest) per protocol; the random ones are what 03-05 expect
OUTPUTS = {
    "random": ("/tmp/ocular_recognition", "/kaggle/working/manifest_split.csv"),
    "cross_session": ("/tmp/ocular_cross_session", "/kaggle/working/manifest_split_cross_session.csv"),
    "open_set": (None, "/kaggle/working/manifest_open_set.csv"),
}
BUCKET = ["subject", "condition", "eye", "phase"]


def in_paper_classes(subject):
    """True for the 300 classes of the paper protocol: subject IDs 1141..3279 plus 7000."""
    return subject.between(1141, 3279) | (subject == 7000)


def assign_splits(m, seed=42, test_relpaths=None):
    """Return a copy of manifest *m* with a ``split`` column.

    Rows are sorted by relpath first, so the result does not depend on row order.
    With *test_relpaths*, exactly those images are the test split and the rest of each
    class is split 75/25 into train/validation; otherwise each class is split 60/20/20.
    """
    m = m.sort_values("relpath").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    m["split"] = ""
    if test_relpaths is not None:
        test_relpaths = set(test_relpaths)
        missing = test_relpaths - set(m.relpath)
        if missing:
            raise ValueError(f"{len(missing)} pinned test images are not in the manifest, "
                             f"e.g. {sorted(missing)[:3]}")
        m.loc[m.relpath.isin(test_relpaths), "split"] = "test"
    for _, labels in m.groupby("subject").groups.items():
        labels = np.array(list(labels))
        if test_relpaths is not None:
            labels = rng.permutation(labels[(m.loc[labels, "split"] != "test").to_numpy()])
            n_tr = int(round(0.75 * len(labels)))
            m.loc[labels[:n_tr], "split"] = "train"
            m.loc[labels[n_tr:], "split"] = "validation"
        else:
            labels = rng.permutation(labels)
            n = len(labels)
            n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
            m.loc[labels[:n_tr], "split"] = "train"
            m.loc[labels[n_tr:n_tr + n_va], "split"] = "validation"
            m.loc[labels[n_tr + n_va:], "split"] = "test"
    return m


def assign_cross_session(m, seed=42, val_frac=0.25):
    """Return a copy of manifest *m* with a cross-session ``split`` column.

    test = every verification-phase image. Each class's enrollment images are split
    into train and validation by whole (condition, eye) buckets: round(val_frac x
    buckets) of them, at least 1 and at most all but one, go to validation. A class
    with a single enrollment bucket keeps it in train (no validation images).
    Rows are sorted by relpath first, so the result does not depend on row order.
    """
    m = m.sort_values("relpath").reset_index(drop=True)
    m["split"] = ""
    m.loc[m.phase == "verification", "split"] = "test"
    rng = np.random.default_rng(seed)
    for _, enrol in m[m.phase == "enrollment"].groupby("subject"):
        buckets = enrol.groupby(["condition", "eye"]).groups
        keys = sorted(buckets)
        n_val = min(max(1, int(round(val_frac * len(keys)))), len(keys) - 1)
        val = {keys[i] for i in rng.permutation(len(keys))[:n_val]}
        for key in keys:
            m.loc[buckets[key], "split"] = "validation" if key in val else "train"
    unknown = m.split == ""
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} images have a phase other than enrollment/verification, "
                         f"e.g. {m.relpath[unknown].iloc[0]}")
    return m


def select_open_set(m):
    """Every image of the subjects outside the 300 classes (both phases), sorted, with split 'test'."""
    m = m[~in_paper_classes(m.subject)].sort_values("relpath").reset_index(drop=True)
    m["split"] = "test"
    return m


def write_split_csvs(m, protocol, out_dir):
    """Write <out_dir>/<protocol>_<split>.csv (relpath, subject; sorted) for every split present.

    Committed lists keep a split fixed even if the Kaggle copy or NumPy's random streams change.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for split in ("train", "validation", "test"):
        part = m.loc[m.split == split, ["relpath", "subject"]].sort_values("relpath")
        if len(part):
            paths.append(os.path.join(out_dir, f"{protocol}_{split}.csv"))
            part.to_csv(paths[-1], index=False)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/kaggle/working/manifest_raw.csv")
    ap.add_argument("--root", default=None)
    ap.add_argument("--protocol", choices=list(OUTPUTS), default="random",
                    help="random: the paper's per-class split with the pinned test list (reproduces the "
                         "existing checkpoint); cross_session: train/validation from enrollment images, "
                         "test = verification images; open_set: only the evaluation list of the subjects "
                         "outside the 300 classes")
    ap.add_argument("--out-dir", default=None,
                    help="symlink tree; default /tmp/ocular_recognition (random) or "
                         "/tmp/ocular_cross_session (cross_session); open_set builds none")
    ap.add_argument("--out-manifest", default=None,
                    help="manifest with the split of every image; default /kaggle/working/manifest_split.csv "
                         "(random), manifest_split_cross_session.csv or manifest_open_set.csv")
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST,
                    help="random protocol: CSV with a relpath column fixing the test split; 'none' for a "
                         "random split")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="cross_session: share of each class's enrollment buckets used for validation")
    ap.add_argument("--write-split-csvs", default=None, metavar="DIR",
                    help="also write DIR/<protocol>_{train,validation,test}.csv (relpath, subject), e.g. "
                         "into splits/ to commit them")
    args = ap.parse_args()
    if not 0 < args.val_frac < 1:                # e.g. 25 meant as a percentage would leave one train bucket
        ap.error(f"--val-frac must be a fraction between 0 and 1, got {args.val_frac}")
    out_dir = args.out_dir or OUTPUTS[args.protocol][0]
    out_manifest = args.out_manifest or OUTPUTS[args.protocol][1]

    root = args.root or find_root()
    m = pd.read_csv(args.manifest)

    if args.protocol == "open_set":
        m = select_open_set(m)
        m["link_path"] = [os.path.join(root, r) for r in m.relpath]   # no tree: the original files
        m.to_csv(out_manifest, index=False)
        phases = m.groupby("subject").phase.nunique()
        print(f"open set: {m.subject.nunique()} subjects outside the 300 classes | images: {len(m)} "
              f"({int((m.phase == 'enrollment').sum())} enrollment, "
              f"{int((m.phase == 'verification').sum())} verification)")
        print(f"subjects with images in both phases (enrollment templates vs verification probes): "
              f"{int((phases == 2).sum())}")
        print("no tree built (nothing is trained); link_path points at the original files")
        print("manifest written to:", out_manifest)
        for path in write_split_csvs(m, args.protocol, args.write_split_csvs) if args.write_split_csvs else []:
            print("split list written to:", path)
        return

    # paper protocol: IDs 1141..3279 plus 7000
    m = m[in_paper_classes(m.subject)].copy()
    classes = sorted(m.subject.unique())
    if args.max_classes:
        classes = classes[: args.max_classes]
        m = m[m.subject.isin(classes)].copy()
    print(f"classes: {len(classes)} | images: {len(m)}")

    if args.protocol == "cross_session":
        m = assign_cross_session(m, args.seed, args.val_frac)
        print(f"cross-session split: test = every verification image; validation = whole enrollment "
              f"buckets ({args.val_frac:.0%} of each class's buckets)")
    else:
        test_relpaths = None
        if args.test_list and args.test_list.lower() != "none":
            test_relpaths = pd.read_csv(args.test_list).relpath
            if args.max_classes:                       # debug subset: keep its classes' test images
                test_relpaths = test_relpaths[test_relpaths.isin(m.relpath)]
            print(f"test split pinned to {len(test_relpaths)} images from {args.test_list}")
        else:
            print("WARNING: random 60/20/20 split; do not evaluate the full baseline checkpoint on it",
                  file=sys.stderr)
        m = assign_splits(m, args.seed, test_relpaths)

    # symlink tree
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    links = []
    for r in m.itertuples():
        d = os.path.join(out_dir, r.split, str(r.subject))
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, f"{r.condition}__{r.phase}__{os.path.basename(r.relpath)}")
        os.symlink(os.path.join(root, r.relpath), link)
        links.append(link)
    m["link_path"] = links
    m.to_csv(out_manifest, index=False)

    # report
    print("\nimages per split:\n", m.split.value_counts().to_string())
    print("\nimages per class per split (min / median / max):")
    print(m.groupby(["subject", "split"]).size().unstack().describe().loc[["min", "50%", "max"]].round(0).to_string())

    train_buckets = set(map(tuple, m[m.split == "train"][BUCKET].values))
    test = m[m.split == "test"]
    frac = np.mean([tuple(x) in train_buckets for x in test[BUCKET].values])
    print(f"\nfraction of test images whose (subject, condition, eye, phase) bucket also has training images: {frac:.1%}")
    if args.protocol == "cross_session":
        shared = train_buckets & set(map(tuple, m[m.split == "validation"][BUCKET].values))
        print(f"buckets shared by train and validation: {len(shared)} | phases in test: "
              f"{sorted(test.phase.unique())}")
        for split, why in (("train", "no enrollment images; 03_train_baseline.py will refuse to train"),
                           ("validation", "only one enrollment bucket, which stays in train"),
                           ("test", "no verification images")):
            missing = sorted(int(c) for c in set(classes) - set(m.subject[m.split == split]))
            if missing:
                print(f"WARNING: {len(missing)} classes have no {split} images ({why}): {missing[:20]}"
                      + (" ..." if len(missing) > 20 else ""))
    print("tree written to:", out_dir)
    print("manifest written to:", out_manifest)
    for path in write_split_csvs(m, args.protocol, args.write_split_csvs) if args.write_split_csvs else []:
        print("split list written to:", path)


if __name__ == "__main__":
    main()
