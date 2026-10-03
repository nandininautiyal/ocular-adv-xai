"""
scripts/02_make_splits.py

Paper-style protocol: keep subjects with ID in [1141, 3279] plus 7000 (300 classes),
pool all images of a subject (all devices, lighting, eyes, phases), and split each class
randomly 60/20/20 into train/validation/test.

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

import numpy as np
import pandas as pd


def find_root(base="/kaggle/input"):
    for r, dirs, _ in os.walk(base):
        if "VISOB_Augmented_Recent" in dirs:
            return os.path.join(r, "VISOB_Augmented_Recent")
    raise FileNotFoundError("VISOB_Augmented_Recent not found under " + base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/kaggle/working/manifest_raw.csv")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-dir", default="/tmp/ocular_recognition")
    ap.add_argument("--out-manifest", default="/kaggle/working/manifest_split.csv")
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
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

    # random 60/20/20 within each class
    rng = np.random.default_rng(args.seed)
    m["split"] = ""
    for _, labels in m.groupby("subject").groups.items():
        labels = rng.permutation(np.array(list(labels)))
        n = len(labels)
        n_tr, n_va = int(round(0.6 * n)), int(round(0.2 * n))
        m.loc[labels[:n_tr], "split"] = "train"
        m.loc[labels[n_tr:n_tr + n_va], "split"] = "validation"
        m.loc[labels[n_tr + n_va:], "split"] = "test"

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