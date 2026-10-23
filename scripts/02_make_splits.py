"""
scripts/02_make_splits.py

Paper-style protocol: keep subjects with ID in [1141, 3279] plus 7000 (300 classes),
pool all images of a subject (all devices, lighting, eyes, phases), and split each class
randomly 60/20/20 into train/validation/test.

The test split is pinned to splits/visob_test.csv: the 13,868 test images of the run that
trained the full baseline checkpoint. The remaining images of each class are split 75/25
into train/validation (60/20 of the total). Before this file existed, the split depended on
the order the disk listed files in, so a new Kaggle session could put training images into
"test" (on 2026-10-22 only 4,166 of the original test images stayed in test). Pass
--test-list none for a fresh random 60/20/20 split (only for a newly trained model).

Builds the class-folder tree the ABRobOcular loader expects, using symlinks (no copying):
  <out-dir>/{train,validation,test}/<subject>/<condition>__<phase>__<filename>.png
and writes a manifest with the split and link path for every image.

Usage:
  python scripts/02_make_splits.py                       # all 300 classes
  python scripts/02_make_splits.py --max-classes 30 --out-dir /tmp/ocular_small \
         --out-manifest /kaggle/working/manifest_small.csv   # quick debug subset
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/kaggle/working/manifest_raw.csv")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-dir", default="/tmp/ocular_recognition")
    ap.add_argument("--out-manifest", default="/kaggle/working/manifest_split.csv")
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST,
                    help="CSV with a relpath column fixing the test split; 'none' for a random split")
    args = ap.parse_args()

    root = args.root or find_root()
    m = pd.read_csv(args.manifest)

    # paper protocol: IDs 1141..3279 plus 7000
    m = m[m.subject.between(1141, 3279) | (m.subject == 7000)].copy()
    classes = sorted(m.subject.unique())
    if args.max_classes:
        classes = classes[: args.max_classes]
        m = m[m.subject.isin(classes)].copy()
    print(f"classes: {len(classes)} | images: {len(m)}")

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
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    links = []
    for r in m.itertuples():
        d = os.path.join(args.out_dir, r.split, str(r.subject))
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, f"{r.condition}__{r.phase}__{os.path.basename(r.relpath)}")
        os.symlink(os.path.join(root, r.relpath), link)
        links.append(link)
    m["link_path"] = links
    m.to_csv(args.out_manifest, index=False)

    # report
    print("\nimages per split:\n", m.split.value_counts().to_string())
    print("\nimages per class per split (min / median / max):")
    print(m.groupby(["subject", "split"]).size().unstack().describe().loc[["min", "50%", "max"]].round(0).to_string())

    bucket = ["subject", "condition", "eye", "phase"]
    train_buckets = set(map(tuple, m[m.split == "train"][bucket].values))
    test = m[m.split == "test"]
    frac = np.mean([tuple(x) in train_buckets for x in test[bucket].values])
    print(f"\nfraction of test images whose (subject, condition, eye, phase) bucket also has training images: {frac:.1%}")
    print("tree written to:", args.out_dir)
    print("manifest written to:", args.out_manifest)


if __name__ == "__main__":
    main()