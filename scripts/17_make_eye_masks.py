"""
scripts/17_make_eye_masks.py

Anatomical eye masks for VISOB images: skin, iris (pupil included), sclera, upper
lid, caruncle and brow, from the periorbital segmentation model (Nahass et al.
2025; src/xai/seg_masks.py). Each mask is computed on the native 240 x 160 crop
and then resized to 256 x 256 with nearest-neighbour exactly like the runner
resizes the image, so mask pixel (i, j) belongs to model-input pixel (i, j).

Inputs (one of):
  --csv PATH     a CSV with relpath and/or link_path columns: correct_test_<tag>.csv
                 (04), images_<tag>.csv (05; link_path only, relpaths are recovered
                 from the pinned test list) or manifest_split.csv (02)
  --all-test     the 13,868 images of splits/visob_test.csv (needs the VISOB data)

Writes to --out-dir:
  masks_<tag>.npz            relpaths, labels (N x 256 x 256, uint8), class_names, meta (JSON):
                             the mask file the runner reads (see src/xai/seg_masks.py)
  mask_areas_<tag>.csv       one row per image: relpath, link_path, mask_source, area_<class>
                             (share of the image on the 8 x 8 heatmap grid, averaged like
                             src/compare/regions.py) and empty_iris
  mask_qc_<tag>.png          overlays for a stratified sample (phone x lighting x eye)
  mask_qc_low_iris_<tag>.png the images with the smallest iris area: look here for failures
  mask_summary_<tag>.json    class-area quantiles, empty-iris rate, timing, weights sha256,
                             git commit and settings

The weights (combine.pth, 199 MB, MIT licence) are checked against a pinned sha256.
Without --weights they are downloaded from Hugging Face into --weights-dir, so turn
"Internet" on in the Kaggle notebook settings. PyTorch is preinstalled on Kaggle;
the script also runs on a CPU (about 1-3 s per image).

The masks are zero-shot: check the QC overlays, and validate them against hand
labels with scripts/18_validate_masks.py before reporting region-level results.

Usage (Kaggle, GPU, after 01 and 02, or with --data-root):
  smoke (2-3 min): python scripts/17_make_eye_masks.py --all-test --max-images 40 --tag smoke
  full test split: python scripts/17_make_eye_masks.py --all-test --tag test
  the 600 images of the 27 Oct run:
                   python scripts/17_make_eye_masks.py --csv images_full.csv --tag full
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.xai import seg_masks as sm  # noqa: E402

DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
STRATA = ("device", "lighting", "eye")


def find_root(base="/kaggle/input"):
    for r, dirs, _ in os.walk(base):
        if "VISOB_Augmented_Recent" in dirs:
            return os.path.join(r, "VISOB_Augmented_Recent")
    raise FileNotFoundError("VISOB_Augmented_Recent not found under " + base)


def git_commit(repo=REPO):
    """(commit hash, whether src/ or scripts/ differ from it, new files included) of the code that
    made the masks; ("unknown", None) outside git."""
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "src", "scripts"],
                               capture_output=True, text=True, timeout=20, check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def load_native(path):
    """The stored crop as RGB uint8, decoded and converted like tf.keras.utils.load_img does (no resize)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def quantiles(values):
    v = np.asarray(values, np.float64)
    return {"mean": float(v.mean()), **{f"p{q:02d}": float(np.quantile(v, q / 100)) for q in (5, 25, 50, 75, 95)}}


def overlay_grid(tiles, path, cols=6):
    """Save (image, title) tiles as one figure."""
    if not tiles:
        return None
    rows = (len(tiles) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, title) in zip(axes.ravel(), tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=7)
    fig.suptitle("iris blue, sclera green, lid magenta, caruncle yellow, brow red (model-input geometry)",
                 fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def make_masks(table, predict_fn, out_dir, tag, batch_size=16, factor=sm.DEFAULT_PAD_FACTOR, canvas="stretch",
               flip_tta=False, qc_per_stratum=2, n_low_iris=12, seed=0, meta=None, n_input=None):
    """Segment every image of *table* (from ``seg_masks.image_table``) and write the outputs.

    *predict_fn* is the network (``seg_masks.torch_predictor``) or, in tests, any function with
    the same contract. Returns the summary dict that is also written as JSON.
    """
    table = table.reset_index(drop=True)
    if table.empty:
        raise ValueError("no images to segment")
    if table.path.isna().any():
        raise ValueError("every image needs a readable path")
    os.makedirs(out_dir, exist_ok=True)
    n = len(table)
    n_input = n_input or n
    labels = np.zeros((n,) + sm.MODEL_SIZE, np.uint8)
    areas, sizes = [], {}
    n_qc = qc_per_stratum * max(table.groupby(list(STRATA)).ngroups, 1)
    qc_rows = set(sm.stratified_sample(table, n_qc, keys=STRATA, seed=seed).index) if qc_per_stratum > 0 else set()
    qc_images = {}

    n_batches = (n + batch_size - 1) // batch_size
    every = max(1, n_batches // 20)
    t0 = time.time()
    t_first, n_first = None, 0
    for b, start in enumerate(range(0, n, batch_size)):
        part = table.iloc[start:start + batch_size]
        natives = [load_native(p) for p in part.path]
        native_labels = sm.segment(natives, predict_fn, batch_size=len(natives), factor=factor, canvas=canvas,
                                   flip_tta=flip_tta)
        for k, (img, lab) in enumerate(zip(natives, native_labels)):
            i = start + k
            size = f"{img.shape[1]}x{img.shape[0]}"
            sizes[size] = sizes.get(size, 0) + 1
            labels[i] = sm.to_model_geometry(lab)
            areas.append(sm.class_areas(labels[i]))
            if i in qc_rows:
                qc_images[i] = sm.model_input_image(img)
        done = start + len(part)
        if b == 0:                                   # the first batch includes the GPU warm-up
            t_first, n_first = time.time(), done
            print(f"  first batch: {done} images in {t_first - t0:.1f} s (includes warm-up)", flush=True)
        if (b + 1) % every == 0 or b + 1 == n_batches or b == 1:
            rate = (time.time() - t_first) / (done - n_first) if done > n_first else (t_first - t0) / done
            print(f"  {done}/{n} images | {time.time() - t0:.0f} s | ~{rate:.3f} s/image | "
                  f"~{rate * (n - done) / 60:.1f} min left", flush=True)
            if b == 1 and n_input > n:
                print(f"  estimate for all {n_input} input images: ~{rate * n_input / 60:.0f} min "
                      f"(+ model loading)", flush=True)
    seconds = time.time() - t0
    if n_batches == 1 and n_input > n:               # a smoke run in one batch: estimate from it, warm-up included
        print(f"  estimate for all {n_input} input images: at most ~{seconds / n * n_input / 60:.0f} min "
              f"(+ model loading; one batch measured, warm-up included)", flush=True)

    area_df = pd.DataFrame(areas)
    out = pd.concat([table[["relpath", "link_path"]].reset_index(drop=True),
                     pd.Series(["periorbital"] * n, name="mask_source"), area_df], axis=1)
    out["empty_iris"] = out.area_iris == 0
    paths = {
        "masks": os.path.join(out_dir, f"masks_{tag}.npz"),
        "areas": os.path.join(out_dir, f"mask_areas_{tag}.csv"),
        "qc": os.path.join(out_dir, f"mask_qc_{tag}.png"),
        "qc_low_iris": os.path.join(out_dir, f"mask_qc_low_iris_{tag}.png"),
        "summary": os.path.join(out_dir, f"mask_summary_{tag}.json"),
    }
    meta = dict(meta or {})
    meta.update({"n_images": n, "native_sizes": sizes,
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    sm.save_mask_file(paths["masks"], table.relpath.tolist(), labels, sm.CLASS_NAMES, meta)
    out.to_csv(paths["areas"], index=False)

    strata = table[list(STRATA)].reset_index(drop=True)
    qc_tiles = []
    for i in sorted(qc_images, key=lambda j: tuple(strata.iloc[j])):
        s = strata.iloc[i]
        qc_tiles.append((sm.overlay(qc_images[i], labels[i]),
                         f"{s.device} | {s.lighting} | {s.eye}\niris {out.area_iris[i]:.0%}"))
    if overlay_grid(qc_tiles, paths["qc"]) is None:     # --qc-per-stratum 0: no stratified grid
        del paths["qc"]
    low = np.argsort(out.area_iris.to_numpy(), kind="stable")[:n_low_iris]
    low_tiles = []
    for i in low:
        img = sm.model_input_image(load_native(table.path.iloc[i]))
        low_tiles.append((sm.overlay(img, labels[i]), f"{table.relpath.iloc[i].split('/')[-1]}\n"
                                                       f"iris {out.area_iris[i]:.1%}"))
    overlay_grid(low_tiles, paths["qc_low_iris"])

    summary = {
        "tag": tag, "n_images": n, "mask_source": "periorbital", "class_names": list(sm.CLASS_NAMES),
        "area_quantiles": {c: quantiles(out[f"area_{c}"]) for c in sm.CLASS_NAMES},
        "n_empty_iris": int(out.empty_iris.sum()), "empty_iris_rate": float(out.empty_iris.mean()),
        "low_iris_rate_below_2pct": float((out.area_iris < 0.02).mean()),
        "empty_iris_rate_by_stratum": {" | ".join(k): float(v) for k, v in
                                       out.join(strata).groupby(list(STRATA)).empty_iris.mean().items()},
        "native_sizes": sizes,
        "seconds_segmenting": round(seconds, 1), "seconds_per_image": round(seconds / max(n, 1), 4),
        "settings": {"pad_factor": factor, "canvas": canvas, "flip_tta": flip_tta, "batch_size": batch_size},
        **{k: meta[k] for k in ("weights_file", "weights_sha256", "code_commit", "code_dirty", "device",
                                "torch", "torchvision") if k in meta},
        "outputs": paths,
    }
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 200)
    cols = [f"area_{c}" for c in sm.CLASS_NAMES] + ["empty_iris"]
    joined = out.join(strata)
    for key in STRATA:
        print(f"\nmean area share by {key} (8 x 8 grid):")
        print(joined.groupby(key)[cols].mean().round(3).to_string())
    print(f"\nimages: {n} | native sizes: {sizes} | empty iris: {summary['n_empty_iris']} "
          f"({summary['empty_iris_rate']:.1%}) | iris below 2 % of the image: "
          f"{summary['low_iris_rate_below_2pct']:.1%}")
    print(f"segmenting took {seconds / 60:.1f} min ({summary['seconds_per_image']:.3f} s/image)")
    for name, path in paths.items():
        print(f"saved {name}: {path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Anatomical eye masks (periorbital model) in model-input geometry.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV with relpath and/or link_path columns (04's correct_test_<tag>.csv, "
                                   "05's images_<tag>.csv, 02's manifest)")
    src.add_argument("--all-test", action="store_true", help="all images of the pinned test list")
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST,
                    help="pinned test list; used by --all-test and to recover relpaths from link paths")
    ap.add_argument("--data-root", default=None,
                    help="VISOB root (VISOB_Augmented_Recent); found under /kaggle/input if not given")
    ap.add_argument("--weights", default=None, help="combine.pth; downloaded and sha256-checked if not given")
    ap.add_argument("--weights-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "periorbital"),
                    help="where to download the weights (kept out of the Kaggle outputs)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pad-factor", type=float, default=sm.DEFAULT_PAD_FACTOR,
                    help="canvas side / crop side; the model was trained on crops with much more skin")
    ap.add_argument("--canvas", choices=sm.CANVASES, default="stretch",
                    help="stretch: pad each side by the factor (keeps the model-input stretch, as in the pilot); "
                         "square: pad to a square (keeps the native aspect ratio)")
    ap.add_argument("--flip-tta", action="store_true", help="average the logits of the image and its mirror")
    ap.add_argument("--amp", action="store_true", help="float16 autocast on the GPU (faster, tiny differences)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="smoke run: a stratified subset of this many images")
    ap.add_argument("--qc-per-stratum", type=int, default=2, help="overlays per phone x lighting x eye cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/kaggle/working/eye_masks")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()

    data_root = args.data_root
    if data_root is None and os.path.isdir("/kaggle/input"):
        try:
            data_root = find_root()
        except FileNotFoundError:
            data_root = None
    print("data root:", data_root)

    known = pd.read_csv(args.test_list).relpath.tolist() if os.path.exists(args.test_list) else []
    frame = pd.DataFrame({"relpath": known}) if args.all_test else pd.read_csv(args.csv)
    table = sm.image_table(frame, data_root=data_root, known_relpaths=known)
    n_input = len(table)
    if n_input == 0:
        raise SystemExit(f"no images to segment: {args.test_list if args.all_test else args.csv} is missing "
                         f"or lists no images")
    if args.max_images:
        table = sm.stratified_sample(table, args.max_images, keys=STRATA, seed=args.seed).reset_index(drop=True)
    missing = table[table.path.isna()]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} of {len(table)} images not found (rebuild the split tree with "
                                f"01 and 02, or pass --data-root); e.g. {missing.relpath.head(5).tolist()}")
    print(f"images: {len(table)} of {n_input} in the input | strata: "
          f"{table.groupby(list(STRATA)).ngroups} (phone x lighting x eye)")

    torch = sm.require_torch()
    import torchvision
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    t0 = time.time()
    weights = args.weights or sm.fetch_weights(args.weights_dir)
    model = sm.load_periorbital_model(weights, device=device)
    load_seconds = time.time() - t0
    print(f"model: periorbital DeepLabV3-ResNet101 ({weights}, sha256 ok) on {device} | "
          f"torch {torch.__version__} | loaded in {load_seconds:.0f} s")

    commit, dirty = git_commit()
    meta = {
        "model": "periorbital DeepLabV3-ResNet101 (Nahass et al., Ophthalmology Science 2025, arXiv:2409.20407)",
        "weights_file": os.path.basename(weights), "weights_sha256": sm.WEIGHTS_SHA256,
        "weights_source": f"https://huggingface.co/{sm.WEIGHTS_REPO} (revision {sm.WEIGHTS_REVISION})",
        "model_classes": list(sm.PERIORBITAL_CLASSES),
        "class_map": {m: sm.CLASS_NAMES[c] for m, c in zip(sm.PERIORBITAL_CLASSES, sm.PERIORBITAL_TO_CANONICAL)},
        "padding": {"factor": args.pad_factor, "mode": "edge", "canvas": args.canvas},
        "network_input": "padded crop resized to 256 x 256 (PIL bilinear), (x / 255 - 0.5) / 0.5",
        "geometry": "labels in model-input geometry: native label map resized to 256 x 256 with PIL nearest, "
                    "as tf.keras.utils.load_img(path, target_size=(256, 256)) resizes the image",
        "flip_tta": args.flip_tta, "amp": args.amp,
        "code_commit": commit, "code_dirty": dirty,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__,
        "device": device, "source": "all-test" if args.all_test else os.path.abspath(args.csv),
        "data_root": data_root,
    }
    predict = sm.torch_predictor(model, device=device, amp=args.amp)
    summary = make_masks(table, predict, args.out_dir, args.tag, batch_size=args.batch_size,
                         factor=args.pad_factor, canvas=args.canvas, flip_tta=args.flip_tta,
                         qc_per_stratum=args.qc_per_stratum, seed=args.seed, meta=meta, n_input=n_input)
    summary["seconds_model_load"] = round(load_seconds, 1)
    with open(summary["outputs"]["summary"], "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
