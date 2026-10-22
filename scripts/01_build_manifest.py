"""
scripts/01_build_manifest.py

Walk the Kaggle VISOB copy and write a manifest CSV (one row per image).
Everything is parsed from the path, so no image is opened and it runs fast.

Layout: <root>/<device>_<lighting>_short/<phase>_<eye>_eye/<subject>/<subject>_<idx>_<l|r>_c.png
Usage:  python scripts/01_build_manifest.py [--root PATH] [--out manifest_raw.csv]
"""
import argparse
import os
import re

import pandas as pd

FNAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<idx>\d+)_(?P<side>[lr])_c\.png$", re.IGNORECASE)
PHASE_RE = re.compile(r"^(?P<phase>enrollment|verification)_(?P<eye>left|right)_eye$")


def find_root(base="/kaggle/input"):
    for r, dirs, _ in os.walk(base):
        if "VISOB_Augmented_Recent" in dirs:
            return os.path.join(r, "VISOB_Augmented_Recent")
    raise FileNotFoundError("VISOB_Augmented_Recent not found under " + base)


def parse_condition(folder):
    name = folder.lower()
    device = name.split("_")[0]
    if "office" in name:
        lighting = "office"
    elif "day" in name:
        lighting = "day"
    elif "dim" in name:
        lighting = "dim"
    else:
        lighting = "unknown"
    return device, lighting


def build(root):
    rows, skipped = [], []
    for cond in sorted(os.listdir(root)):
        cond_dir = os.path.join(root, cond)
        if not os.path.isdir(cond_dir):
            continue
        device, lighting = parse_condition(cond)
        for pdir in sorted(os.listdir(cond_dir)):
            m = PHASE_RE.match(pdir)
            if not m:  # Results_*, results_recent_*, etc.
                continue
            phase, eye = m.group("phase"), m.group("eye")
            pdir_path = os.path.join(cond_dir, pdir)
            for subj in sorted(os.listdir(pdir_path)):
                sdir = os.path.join(pdir_path, subj)
                if not os.path.isdir(sdir):
                    continue
                for fname in sorted(os.listdir(sdir)):   # sorted: row order must not depend on the disk
                    rel = os.path.relpath(os.path.join(sdir, fname), root)
                    fm = FNAME_RE.match(fname)
                    if not fm:
                        skipped.append(rel)
                        continue
                    rows.append({
                        "relpath": rel,
                        "subject": int(fm.group("subject")),
                        "folder_subject": int(subj) if subj.isdigit() else -1,
                        "idx": int(fm.group("idx")),
                        "side": fm.group("side").lower(),
                        "eye": eye,
                        "phase": phase,
                        "device": device,
                        "lighting": lighting,
                        "condition": cond,
                    })
    return pd.DataFrame(rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="manifest_raw.csv")
    args = ap.parse_args()

    root = args.root or find_root()
    print("Data root:", root)
    df, skipped = build(root)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Skipped {len(skipped)} files that did not match the filename pattern")
    for s in skipped[:10]:
        print("  e.g.", s)


if __name__ == "__main__":
    main()