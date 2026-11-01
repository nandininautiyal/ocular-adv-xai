"""
scripts/21_make_train_masks.py

Compact eye masks for training the mask-supervised models (03 --arch attn_mask | parts |
attnet_gate): masks64_<tag>.npz, the 256 x 256 model-geometry labels of
src/xai/seg_masks.py reduced to 64 x 64 with PIL nearest (8 x 8 label pixels per
feature cell; see src/data/masked_generator.py for the contract). Also reports how
often each part group is in the crop at all (part presence), per split: how often is
the eyebrow in a VISOB crop?

Two modes:
  --from-mask-file masks_<tag>.npz   convert a mask file of 17 (CPU, seconds); the
                                     segmenter weights sha256 is taken from its meta
  segment images directly with 17's code (GPU with PyTorch; CPU works, slowly):
    --split-dir DIR --splits train,validation,test   every image of those splits of a
                                     02 tree (relpaths read off the symlinks with
                                     --data-root, or from the pinned test list)
    --csv PATH                       relpath and/or link_path columns, e.g. 02's
                                     manifest_split_cross_session.csv (its split column
                                     is used for the report and filtered by --splits)
  Images are segmented in batches and only their 64 x 64 maps are kept (4 KB per image,
  about 400 MB for 100,000 images), so memory stays low.

Splits for the presence report: from --split-dir, or a split column of --csv; with
--from-mask-file from --report-split-dir (a 02 tree) or --report-csv (relpath or link_path
plus split), images matched like 17 / 18 match them; otherwise one split "all".

Writes to --out-dir:
  masks64_<tag>.npz                relpaths, labels (N x 64 x 64, uint8), class_names, meta (JSON:
                                   source, code commit, created, segmenter weights sha256, ...)
  part_presence_<tag>.csv / .json  per split and part group: images, mean area (share of the image),
                                   share of images where the group covers >= 0.5 % / 2 % / 5 %
  masks64_summary_<tag>.json       settings, counts per split, timing, git commit, outputs

Usage (Kaggle; segmentation needs Internet for the weights, see 17):
  smoke:   python scripts/21_make_train_masks.py --split-dir /tmp/ocular_cross_session --max-images 40 --tag smoke
  full:    python scripts/21_make_train_masks.py --split-dir /tmp/ocular_cross_session \
               --splits train,validation,test --tag cs
  convert: python scripts/21_make_train_masks.py --from-mask-file /kaggle/working/eye_masks/masks_test.npz \
               --tag test
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.data.masked_generator import labels_to_64, load_label_source, save_masks64  # noqa: E402
from src.models.anatomy_model import anatomy_settings  # noqa: E402
from src.xai import seg_masks as sm  # noqa: E402

DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
DEFAULT_CFG = os.path.join(REPO, "configs", "anatomy.yml")
THRESHOLDS = (0.005, 0.02, 0.05)


def _load_17():
    """scripts/17_make_eye_masks.py as a module (its image loading, data-root search and git helpers)."""
    spec = importlib.util.spec_from_file_location("make_eye_masks_17", os.path.join(REPO, "scripts",
                                                                                     "17_make_eye_masks.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s17 = _load_17()


def threshold_key(t):
    """Column name of a coverage threshold: 0.005 -> 'share_ge_0.5pct'."""
    return f"share_ge_{100 * t:g}pct"


def group_areas(labels, class_names, groups):
    """Share of each image's pixels in each part group: DataFrame (N rows, one column per group).

    labels: uint8 (N, h, w); unknown (255) pixels are in no group but count in the image.
    """
    names = list(class_names)
    out = {}
    for group, classes in groups.items():
        missing = [c for c in classes if c not in names]
        if missing:
            raise ValueError(f"classes {missing} of group {group!r} are not in the mask classes {names}")
        member = np.isin(labels, [names.index(c) for c in classes])
        out[group] = member.reshape(len(labels), -1).mean(axis=1)
    return pd.DataFrame(out)


def part_presence(labels, class_names, groups, splits, thresholds=THRESHOLDS):
    """Per split (and "all") and part group: n_images, mean_area and the share of images whose group
    area is >= each threshold. labels uint8 (N, h, w); splits: N split names."""
    areas = group_areas(labels, class_names, groups)
    areas["split"] = np.asarray(splits, dtype=object)
    rows = []
    names = sorted(areas.split.unique())
    for split in names + (["all"] if len(names) > 1 else []):
        part = areas if split == "all" else areas[areas.split == split]
        for group in groups:
            row = {"split": split, "group": group, "n_images": len(part), "mean_area": float(part[group].mean())}
            for t in thresholds:
                row[threshold_key(t)] = float((part[group] >= t).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def split_lookup(split_dir=None, splits=None, frame=None):
    """(condition, phase, file) key -> split, from a 02 tree and/or a CSV with a split column."""
    keys = {}
    if split_dir:
        for split in splits:
            d = os.path.join(split_dir, split)
            for subject in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                for name in sorted(os.listdir(os.path.join(d, subject))):
                    try:
                        keys[sm.link_key(name)] = split
                    except ValueError:
                        pass
    if frame is not None and "split" in frame:
        for col, key_fn in (("relpath", sm.relpath_key), ("link_path", sm.link_key)):
            if col in frame:
                for value, split in zip(frame[col], frame.split):
                    if isinstance(value, str) and value:
                        try:
                            keys[key_fn(value)] = split
                        except ValueError:
                            pass
                break
    return keys


def splits_of(relpaths, keys):
    """Split of each relpath from :func:`split_lookup` ("all" without a lookup, "unknown" if not found)."""
    if not keys:
        return ["all"] * len(relpaths)
    return [keys.get(sm.relpath_key(r), "unknown") for r in relpaths]


def segment_table(table, predict_fn, batch_size=16, factor=sm.DEFAULT_PAD_FACTOR, canvas="stretch",
                  flip_tta=False, n_input=None):
    """64 x 64 masks (N, 64, 64) of every image of *table* (``seg_masks.image_table``), batch by batch.

    The pipeline is 17's: native crop -> sm.segment -> to_model_geometry (256) -> labels_to_64; only the
    64 x 64 maps are kept. Prints progress with a time estimate (and one for *n_input* images).
    """
    n = len(table)
    out = np.zeros((n,) + (64, 64), np.uint8)
    n_batches = (n + batch_size - 1) // batch_size
    every = max(1, n_batches // 20)
    t0 = time.time()
    t_first, n_first = None, 0
    for b, start in enumerate(range(0, n, batch_size)):
        paths = table.path.iloc[start:start + batch_size].tolist()
        natives = [s17.load_native(p) for p in paths]
        native_labels = sm.segment(natives, predict_fn, batch_size=len(natives), factor=factor, canvas=canvas,
                                   flip_tta=flip_tta)
        out[start:start + len(paths)] = labels_to_64(np.stack([sm.to_model_geometry(lab) for lab in native_labels]))
        done = start + len(paths)
        if b == 0:                                  # includes the GPU warm-up
            t_first, n_first = time.time(), done
            print(f"  first batch: {done} images in {t_first - t0:.1f} s (includes warm-up)", flush=True)
        if (b + 1) % every == 0 or b + 1 == n_batches or b == 1:
            rate = (time.time() - t_first) / (done - n_first) if done > n_first else (t_first - t0) / done
            print(f"  {done}/{n} images | {time.time() - t0:.0f} s | ~{rate:.3f} s/image | "
                  f"~{rate * (n - done) / 60:.1f} min left", flush=True)
            if b == 1 and n_input and n_input > n:
                print(f"  estimate for all {n_input} input images: ~{rate * n_input / 60:.0f} min", flush=True)
    seconds = time.time() - t0
    if n_batches == 1 and n_input and n_input > n:
        print(f"  estimate for all {n_input} input images: at most ~{seconds / n * n_input / 60:.0f} min "
              f"(one batch measured, warm-up included)", flush=True)
    return out, seconds


def image_list(args, known):
    """Image table (seg_masks.image_table) with a split column, from --split-dir or --csv."""
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.split_dir:
        tables = []
        for split in splits:
            d = os.path.join(args.split_dir, split)
            if not os.path.isdir(d):
                raise SystemExit(f"{d} not found; build the tree with 02_make_splits.py")
            links = [os.path.join(d, s, f) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))
                     for f in sorted(os.listdir(os.path.join(d, s)))]
            t = sm.image_table(pd.DataFrame({"link_path": links}), data_root=args.data_root, known_relpaths=known)
            tables.append(t.assign(split=split))
        return pd.concat(tables, ignore_index=True)
    frame = pd.read_csv(args.csv)
    if "split" in frame:
        frame = frame[frame.split.isin(splits)].reset_index(drop=True)
    table = sm.image_table(frame, data_root=args.data_root, known_relpaths=known)
    keys = split_lookup(frame=frame)
    return table.assign(split=splits_of(table.relpath, keys))


def write_outputs(args, relpaths, labels, class_names, splits, meta, settings, timing):
    """masks64 file, part presence CSV / JSON and the summary JSON; prints the presence table."""
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {"masks64": os.path.join(args.out_dir, f"masks64_{args.tag}.npz"),
             "presence_csv": os.path.join(args.out_dir, f"part_presence_{args.tag}.csv"),
             "presence_json": os.path.join(args.out_dir, f"part_presence_{args.tag}.json"),
             "summary": os.path.join(args.out_dir, f"masks64_summary_{args.tag}.json")}
    save_masks64(paths["masks64"], relpaths, labels, class_names, meta)
    groups = settings["part_groups"]
    presence = part_presence(labels, class_names, groups, splits)
    presence.to_csv(paths["presence_csv"], index=False)
    with open(paths["presence_json"], "w") as f:
        json.dump({"tag": args.tag, "groups": groups, "thresholds": list(THRESHOLDS),
                   "area": "share of the image's pixels (64 x 64 map; unknown pixels count in the image)",
                   "rows": presence.to_dict(orient="records")}, f, indent=2)
    pd.set_option("display.width", 200)
    print("\npart presence (share of images where the group covers at least 0.5 % / 2 % / 5 % of the image):")
    print(presence.round(4).to_string(index=False))
    for _, row in presence[presence.group == "brow"].iterrows():
        print(f"brow covers >= 0.5 % of the image in {row[threshold_key(0.005)]:.1%} of the {row.split} images "
              f"(mean area {row.mean_area:.2%})")
    counts = pd.Series(splits).value_counts().to_dict()
    summary = {"tag": args.tag, "n_images": len(relpaths), "images_per_split": counts,
               "class_names": list(class_names), "part_groups": groups, "settings": vars(args),
               "code_commit": meta.get("code_commit"), "code_dirty": meta.get("code_dirty"),
               "weights_sha256": meta.get("weights_sha256"), "source": meta.get("source"),
               "timing": timing, "outputs": paths}
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for name, path in paths.items():
        print(f"saved {name}: {path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="64 x 64 training masks and the part-presence report.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-mask-file", help="convert masks_<tag>.npz from 17 (CPU)")
    src.add_argument("--split-dir", help="segment every image of --splits of this 02 tree")
    src.add_argument("--csv", help="segment the images of this CSV (relpath and/or link_path; optional split)")
    ap.add_argument("--splits", default="train,validation,test",
                    help="splits of --split-dir / of the CSV's split column")
    ap.add_argument("--report-split-dir", default=None,
                    help="--from-mask-file: a 02 tree whose split of each image the presence report uses")
    ap.add_argument("--report-csv", default=None,
                    help="--from-mask-file: a CSV with relpath (or link_path) and split columns for the report")
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="config whose anatomy.part_groups the report uses")
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST, help="pinned test list (recovers relpaths of links)")
    ap.add_argument("--data-root", default=None, help="VISOB root; found under /kaggle/input if not given")
    ap.add_argument("--weights", default=None, help="combine.pth; downloaded and sha256-checked if not given")
    ap.add_argument("--weights-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "periorbital"))
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pad-factor", type=float, default=sm.DEFAULT_PAD_FACTOR)
    ap.add_argument("--canvas", choices=sm.CANVASES, default="stretch")
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="smoke run: this many images (stratified)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/train_masks")
    ap.add_argument("--tag", default="cs")
    args = ap.parse_args()

    import yaml
    with open(args.cfg) as f:
        settings = anatomy_settings(yaml.safe_load(f))
    commit, dirty = s17.git_commit()
    t0 = time.time()
    timing = {}

    if args.from_mask_file:
        source = load_label_source(args.from_mask_file)
        relpaths, labels = source["relpaths"], source["labels"]
        if args.max_images:
            relpaths, labels = relpaths[:args.max_images], labels[:args.max_images]
        report_frame = pd.read_csv(args.report_csv) if args.report_csv else None
        keys = split_lookup(args.report_split_dir, [s.strip() for s in args.splits.split(",") if s.strip()],
                            report_frame)
        splits = splits_of(relpaths, keys)
        src_meta = source["meta"]
        meta = {"source": f"converted from {os.path.abspath(args.from_mask_file)} ({source['source_size']} x "
                          f"{source['source_size']})",
                "source_meta": src_meta, "weights_sha256": src_meta.get("weights_sha256"),
                "geometry": "64 x 64 = the 256 x 256 model-geometry labels resized with PIL nearest",
                "code_commit": commit, "code_dirty": dirty, "n_images": len(relpaths),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"converted {len(relpaths)} masks from {args.from_mask_file}")
        class_names = source["class_names"]
    else:
        data_root = args.data_root
        if data_root is None and os.path.isdir("/kaggle/input"):
            try:
                data_root = s17.find_root()
            except FileNotFoundError:
                data_root = None
        args.data_root = data_root
        print("data root:", data_root)
        known = pd.read_csv(args.test_list).relpath.tolist() if os.path.exists(args.test_list) else []
        table = image_list(args, known)
        n_input = len(table)
        if n_input == 0:
            raise SystemExit("no images to segment")
        if args.max_images:
            table = sm.stratified_sample(table, args.max_images, keys=("split", "device", "lighting", "eye"),
                                         seed=args.seed).reset_index(drop=True)
        missing = table[table.path.isna()]
        if len(missing):
            raise SystemExit(f"{len(missing)} of {len(table)} images not found (pass --data-root or rebuild the "
                             f"tree with 01 and 02), e.g. {missing.relpath.head(5).tolist()}")
        print(f"images: {len(table)} of {n_input} | per split: {table.split.value_counts().to_dict()}")
        torch = sm.require_torch()
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        t = time.time()
        weights = args.weights or sm.fetch_weights(args.weights_dir)
        model = sm.load_periorbital_model(weights, device=device)
        timing["model_load_s"] = round(time.time() - t, 1)
        print(f"model: periorbital DeepLabV3-ResNet101 ({weights}, sha256 ok) on {device}")
        predict = sm.torch_predictor(model, device=device, amp=args.amp)
        labels, seconds = segment_table(table, predict, args.batch_size, args.pad_factor, args.canvas,
                                        args.flip_tta, n_input=n_input)
        timing["segmenting_s"] = round(seconds, 1)
        relpaths, splits, class_names = table.relpath.tolist(), table.split.tolist(), list(sm.CLASS_NAMES)
        meta = {"source": "segmented by 21_make_train_masks.py with 17's pipeline",
                "model": "periorbital DeepLabV3-ResNet101 (Nahass et al., Ophthalmology Science 2025)",
                "weights_file": os.path.basename(weights), "weights_sha256": sm.WEIGHTS_SHA256,
                "padding": {"factor": args.pad_factor, "mode": "edge", "canvas": args.canvas},
                "flip_tta": args.flip_tta, "amp": args.amp, "device": device, "data_root": data_root,
                "geometry": "64 x 64 = the 256 x 256 model-geometry labels resized with PIL nearest",
                "code_commit": commit, "code_dirty": dirty, "n_images": len(relpaths),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    timing["total_s"] = round(time.time() - t0, 1)
    write_outputs(args, relpaths, labels, class_names, splits, meta, settings, timing)


if __name__ == "__main__":
    main()
