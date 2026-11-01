"""
scripts/18_validate_masks.py

Validate eye masks against hand labels, and prepare the images to hand-label.
The labelling guide is docs/mask_labelling.md.

1) --export-sample: choose the images to label and copy them out. Run it where the
   VISOB data is (Kaggle), then download the zip:
     python scripts/18_validate_masks.py --export-sample \
         --csv /kaggle/working/eval_clean/correct_test_full.csv \
         --masks /kaggle/working/eye_masks/masks_test.npz --out-dir /kaggle/working/label_pack --zip
   A stratified sample of --n images (default 60; phone x lighting x eye, one image per
   subject where possible) is written to --out-dir:
     images/<name>           the native PNG files, byte for byte (<name> = relpath, "/" -> "__")
     prefill/<name>          the model's mask as a colour picture: start labelling from it
     overlays/<name>         image and prefill blended, to look at
     labels/                 empty: save the finished labels here, named <name>
     cvat_prefill.zip        the prefill in CVAT's "Segmentation mask 1.1" format
     model_masks_sample.npz  the model's masks of every sampled image (mask-file format); the
                             validation reads it, annotators should not open it
     sample.csv, labelmap.txt, README.txt
   --n-blind of the images (default 12) get no prefill and must be labelled from scratch,
   so we can measure how much starting from the model's answer biases the labels.

2) validate (the default): score the masks against the hand labels. Runs on a laptop
   from the downloaded folder; no model and no VISOB copy needed:
     python scripts/18_validate_masks.py --labels-dir label_pack/labels \
         --images-dir label_pack/images --masks label_pack/model_masks_sample.npz \
         --sample-csv label_pack/sample.csv --out-dir validation
   Hand labels are PNG files named <name>: ids 0 skin, 1 iris, 2 sclera, 3 lid, 4 caruncle,
   5 brow, 255 ignore, or the same classes in the guide's colours; native size (240 x 160)
   or model-input size (256 x 256). Scoring happens in model-input geometry (where region
   mass is computed):
     learned masks (--masks) on the 6 classes;
     learned and heuristic masks (src/xai/masks.py on the model-input image, exactly as the
     runner computes them) on iris (pupil included), sclera and periocular (the rest);
     with --labels-dir-b, a second annotator against the first: the human ceiling.
   Writes to --out-dir:
     label_qc_<tag>.csv            one row per label file (size, off-palette pixels, ignore share)
     validation_<tag>.csv          IoU and Dice per image, method and class
     validation_summary_<tag>.csv  mean IoU and Dice with 95 % subject-cluster bootstrap CIs,
                                   pooled IoU, by subset (all / prefilled / blind)
     validation_<tag>.md           the tables and the paired learned-vs-heuristic difference
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.analysis.stats import cluster_bootstrap_ci, paired_subject_test  # noqa: E402
from src.xai import seg_masks as sm  # noqa: E402

DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
STRATA = ("device", "lighting", "eye")
MAX_IGNORE_SHARE = 0.05       # QC: a label with more "ignore" than this is flagged


def find_root(base="/kaggle/input"):
    for r, dirs, _ in os.walk(base):
        if "VISOB_Augmented_Recent" in dirs:
            return os.path.join(r, "VISOB_Augmented_Recent")
    raise FileNotFoundError("VISOB_Augmented_Recent not found under " + base)


def load_native(path):
    """The stored crop as RGB uint8, decoded like tf.keras.utils.load_img does (no resize)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def cvat_labelmap():
    """CVAT / Datumaro label map: "background" is our skin (everything not labelled)."""
    lines = ["# label:color_rgb:parts:actions"]
    for name, colour in zip(sm.CLASS_NAMES, sm.PALETTE):
        lines.append(f"{'background' if name == 'skin' else name}:{','.join(str(int(c)) for c in colour)}::")
    lines.append(f"ignore:{','.join(str(c) for c in sm.IGNORE_COLOUR)}::")
    return "\n".join(lines) + "\n"


README = """Eye-mask labelling pack ({n} images, made {created}).

Read docs/mask_labelling.md in the repository first. In short:
  images/<name>    the image to label (native 240 x 160 crop)
  prefill/<name>   the model's guess, as a colour picture: correct it, do not trust it
                   (images listed with prefilled = False in sample.csv have no prefill:
                   label those from scratch, and do not look at the model's masks)
  labels/<name>    save your finished label here, same name, PNG, same size as the image
Colours (exact values, no anti-aliasing):
{colours}
Do not open model_masks_sample.npz: it is for the validation script only.
Keep this folder off cloud drives and share it only inside the team.
"""


def export_sample(table, out_dir, n=60, n_blind=12, masks=None, seed=0, make_zip=False, masks_path=None):
    """Copy a stratified sample of images (and the model's masks as prefill) for hand labelling.

    *table* comes from ``seg_masks.image_table``; *masks* is a loaded mask file or None
    (then nothing is prefilled). Returns the sample as a DataFrame (also written as sample.csv).
    """
    if masks is not None:
        has_mask = table.relpath.isin(set(masks["relpaths"]))
        if not has_mask.all():
            print(f"{int((~has_mask).sum())} of {len(table)} listed images have no mask in the mask file "
                  f"and are left out of the sample")
        table = table[has_mask]
        if table.empty:
            raise ValueError("none of the images in the list has a mask in the mask file")
    if table.path.isna().any():
        missing = table[table.path.isna()].relpath.head(5).tolist()
        raise FileNotFoundError(f"{int(table.path.isna().sum())} images not found, e.g. {missing}")
    sample = sm.stratified_sample(table, n, keys=STRATA, seed=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed + 1)
    n_blind = min(n_blind, len(sample)) if masks is not None else len(sample)
    blind = set(rng.choice(len(sample), n_blind, replace=False).tolist()) if n_blind else set()
    sample["name"] = [sm.label_filename(r) for r in sample.relpath]
    sample["prefilled"] = [i not in blind for i in range(len(sample))]

    for sub in ("images", "prefill", "overlays", "labels"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    sizes, prefilled_names, sample_labels = [], [], []
    for _, r in sample.iterrows():
        shutil.copyfile(r.path, os.path.join(out_dir, "images", r["name"]))
        native = load_native(r.path)
        sizes.append(native.shape[:2])
        if masks is None:
            continue
        model_labels = masks["labels"][masks["index"][r.relpath]]
        sample_labels.append(model_labels)
        if r.prefilled:                                  # the model's mask at native size, as a colour picture
            native_labels = sm.from_model_geometry(model_labels, native.shape[:2])
            sm.write_label_png(os.path.join(out_dir, "prefill", r["name"]), native_labels, colour=True)
            Image.fromarray(sm.overlay(native, native_labels)).save(os.path.join(out_dir, "overlays", r["name"]))
            prefilled_names.append(r["name"])
    sample["native_h"] = [s[0] for s in sizes]
    sample["native_w"] = [s[1] for s in sizes]

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if masks is not None:
        meta = dict(masks["meta"], n_images=len(sample), subset_of=os.path.basename(masks_path) if masks_path else "",
                    subset_created=created)
        meta.pop("native_sizes", None)                   # counted over the full mask file, not this subset
        sm.save_mask_file(os.path.join(out_dir, "model_masks_sample.npz"), sample.relpath.tolist(),
                          np.stack(sample_labels), masks["class_names"], meta)
    with open(os.path.join(out_dir, "labelmap.txt"), "w") as f:
        f.write(cvat_labelmap())
    with zipfile.ZipFile(os.path.join(out_dir, "cvat_prefill.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labelmap.txt", cvat_labelmap())
        stems = [os.path.splitext(name)[0] for name in prefilled_names]
        zf.writestr("ImageSets/Segmentation/default.txt", "\n".join(stems) + ("\n" if stems else ""))
        for name, stem in zip(prefilled_names, stems):
            zf.write(os.path.join(out_dir, "prefill", name), f"SegmentationClass/{stem}.png")
    colours = "\n".join(f"  {name:9s} id {k:3d}  RGB {tuple(int(c) for c in colour)}"
                        for k, (name, colour) in enumerate(zip(sm.CLASS_NAMES, sm.PALETTE)))
    colours += f"\n  {'ignore':9s} id {sm.IGNORE:3d}  RGB {sm.IGNORE_COLOUR}"
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(README.format(n=len(sample), created=created, colours=colours))
    cols = ["name", "relpath", "link_path", "subject", "device", "lighting", "eye", "phase", "prefilled",
            "native_w", "native_h"]
    sample[cols].to_csv(os.path.join(out_dir, "sample.csv"), index=False)
    if make_zip:
        archive = shutil.make_archive(os.path.normpath(out_dir), "zip", out_dir)
        print("zip:", archive)
    return sample[cols]


# --- validation -----------------------------------------------------------------------

def label_to_model_geometry(labels, native_hw=None):
    """Hand label at native size (240 x 160) or already 256 x 256 -> model-input geometry."""
    if labels.shape == sm.MODEL_SIZE:
        return labels
    if native_hw is not None and labels.shape != tuple(native_hw):
        raise ValueError(f"label is {labels.shape[1]} x {labels.shape[0]} but the image is "
                         f"{native_hw[1]} x {native_hw[0]}")
    return sm.to_model_geometry(labels)


def score_image(gt, learned=None, heuristic=None):
    """IoU / Dice rows for one image, all maps in model-input geometry.

    gt, learned : label maps over ``CLASS_NAMES`` (gt may contain 255)
    heuristic : region dict from ``get_concept_masks``
    """
    rows = []
    gt3 = sm.comparison_labels(sm.to_heuristic_regions(gt))

    def add(method, space, names, result):
        for k, name in enumerate(names):
            rows.append({"method": method, "space": space, "cls": name, "iou": result["iou"][k],
                         "dice": result["dice"][k], "intersection": result["intersection"][k],
                         "union": result["union"][k], "gt_px": result["gt_px"][k], "pred_px": result["pred_px"][k]})

    if learned is not None:
        add("learned", "6 classes", sm.CLASS_NAMES, sm.iou_dice(learned, gt, len(sm.CLASS_NAMES)))
        learned3 = sm.comparison_labels(sm.to_heuristic_regions(learned))
        add("learned", "3 regions", sm.COMPARISON_REGIONS, sm.iou_dice(learned3, gt3, len(sm.COMPARISON_REGIONS)))
    if heuristic is not None:
        add("heuristic", "3 regions", sm.COMPARISON_REGIONS,
            sm.iou_dice(sm.comparison_labels(heuristic), gt3, len(sm.COMPARISON_REGIONS)))
    return rows


def score_annotators(labels_a, labels_b):
    """Second annotator (B) scored against the first (A): the agreement a model can hope for."""
    rows = []
    for space, names, a, b in (("6 classes", sm.CLASS_NAMES, labels_a, labels_b),
                               ("3 regions", sm.COMPARISON_REGIONS,
                                sm.comparison_labels(sm.to_heuristic_regions(labels_a)),
                                sm.comparison_labels(sm.to_heuristic_regions(labels_b)))):
        # pixels either annotator marked "ignore" are skipped
        b = np.where(a == sm.IGNORE, sm.IGNORE, b)
        a = np.where(b == sm.IGNORE, sm.IGNORE, a)
        result = sm.iou_dice(b, a, len(names))
        for k, name in enumerate(names):
            rows.append({"method": "annotator B vs A", "space": space, "cls": name, "iou": result["iou"][k],
                         "dice": result["dice"][k], "intersection": result["intersection"][k],
                         "union": result["union"][k], "gt_px": result["gt_px"][k], "pred_px": result["pred_px"][k]})
    return rows


def summarise(per_image, n_boot=2000, seed=0):
    """Mean IoU / Dice per subset, method, space and class, with subject-cluster bootstrap CIs."""
    out = []
    keys = ["subset", "method", "space", "cls"]
    for key, g in per_image.groupby(keys, sort=False):
        iou, iou_lo, iou_hi = cluster_bootstrap_ci(g.iou, g.subject, stat=np.mean, n_boot=n_boot, seed=seed)
        dice, dice_lo, dice_hi = cluster_bootstrap_ci(g.dice, g.subject, stat=np.mean, n_boot=n_boot, seed=seed)
        union, both = g.union.sum(), g.gt_px.sum() + g.pred_px.sum()
        finite = g[np.isfinite(g.iou)]
        out.append(dict(zip(keys, key), n_images=len(finite), n_subjects=finite.subject.nunique(),
                        iou_mean=iou, iou_low=iou_lo, iou_high=iou_hi,
                        dice_mean=dice, dice_low=dice_lo, dice_high=dice_hi,
                        iou_pooled=g.intersection.sum() / union if union else np.nan,
                        dice_pooled=2 * g.intersection.sum() / both if both else np.nan))
    return pd.DataFrame(out)


def paired_difference(per_image, n_boot=2000, seed=0):
    """Learned minus heuristic IoU on the 3 regions, image by image (same images, same labels)."""
    r = per_image[(per_image.subset == "all") & (per_image.space == "3 regions")
                  & per_image.method.isin(["learned", "heuristic"])]
    wide = r.pivot_table(index=["relpath", "subject", "cls"], columns="method", values="iou").reset_index()
    out = []
    if not {"learned", "heuristic"} <= set(wide.columns):
        return pd.DataFrame(out)
    for cls, g in wide.groupby("cls", sort=False):
        g = g[np.isfinite(g.learned) & np.isfinite(g.heuristic)]
        est, lo, hi = cluster_bootstrap_ci(g.learned - g.heuristic, g.subject, stat=np.mean, n_boot=n_boot,
                                           seed=seed)
        test = paired_subject_test(g.learned, g.heuristic, g.subject)
        out.append({"cls": cls, "n_images": len(g), "n_subjects": test["n_subjects"], "iou_diff_mean": est,
                    "ci_low": lo, "ci_high": hi, "p_wilcoxon_subjects": test["p"]})
    return pd.DataFrame(out)


def md_table(df, floats=3):
    if df.empty:
        return "(none)\n"
    cells = [[f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    return "\n".join(lines + ["| " + " | ".join(c) + " |" for c in cells]) + "\n"


def report(summary, diff, label_qc, args, path):
    show = summary.copy()
    show["IoU [95% CI]"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in
                            zip(show.iou_mean, show.iou_low, show.iou_high)]
    show["Dice [95% CI]"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in
                             zip(show.dice_mean, show.dice_low, show.dice_high)]
    cols = ["method", "cls", "n_images", "n_subjects", "IoU [95% CI]", "Dice [95% CI]", "iou_pooled"]
    everything = show[show.subset == "all"]
    models = everything.method.isin(["learned", "heuristic"])
    lines = [f"# Eye-mask validation ({args.tag})", "",
             f"Made {datetime.now(timezone.utc).isoformat(timespec='seconds')} from {len(label_qc)} hand labels "
             f"in `{args.labels_dir}`; learned masks: `{args.masks}`.", "",
             "All maps are compared in model-input geometry (256 x 256, nearest-neighbour from the 240 x 160 "
             "crop), where region mass is computed. Pixels labelled 255 (ignore) are skipped. Means are over "
             "images; CIs are 95 % subject-cluster bootstraps; pooled IoU sums intersections and unions over "
             "all images. A class missing from both maps of an image is left out of that image's mean.", "",
             "## Iris (pupil included), sclera, periocular: learned vs heuristic", "",
             md_table(everything[models & (everything.space == "3 regions")][cols]),
             "## Learned masks, all six classes", "",
             md_table(everything[(everything.method == "learned") & (everything.space == "6 classes")][cols]),
             "## Paired difference, learned minus heuristic IoU (same images)", "",
             md_table(diff)]
    if set(show.subset) >= {"prefilled", "blind"}:
        split = show[show.subset.isin(["prefilled", "blind"]) & (show.method == "learned")]
        lines += ["## Anchoring check: learned masks on prefilled vs blind (from scratch) labels", "",
                  "If the prefilled scores are clearly higher, the prefill biased the labels; quote the blind "
                  "numbers.", "", md_table(split[["subset", "space"] + cols])]
    human = everything[everything.method == "annotator B vs A"]
    if len(human):
        lines += ["## Two annotators (B scored against A): the human ceiling", "",
                  md_table(human[["space"] + cols])]
    flagged = label_qc[~label_qc.ok]
    lines += ["## Label QC", "",
              f"{len(flagged)} of {len(label_qc)} label files flagged (wrong size, colours outside the palette, "
              f"or more than {MAX_IGNORE_SHARE:.0%} ignore).", "",
              md_table(flagged[["file", "width", "height", "off_palette_px", "ignore_share"]]) if len(flagged) else "",
              "## Caveats", "",
              "- Prefilled labels start from the model's answer, which favours the learned masks (anchoring); "
              "compare the 'prefilled' and 'blind' subsets before quoting a number.",
              "- The model's 'lid' is the upper lid (crease to lid margin, upper lashes included); the lower lid "
              "is skin, in the labels and in the model.",
              "- Heuristic masks have no lids, caruncle or brow, so they are scored on the three regions only.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def validate(args):
    files = sorted(f for f in os.listdir(args.labels_dir) if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(f"no PNG labels in {args.labels_dir}")
    masks = sm.load_mask_file(args.masks) if args.masks and not args.check_only else None
    if masks is not None and list(masks["class_names"]) != list(sm.CLASS_NAMES):
        raise ValueError(f"the mask file's classes {masks['class_names']} are not {list(sm.CLASS_NAMES)}")
    prefilled = {}
    if args.sample_csv:
        s = pd.read_csv(args.sample_csv)
        prefilled = dict(zip(s.relpath, s.prefilled.astype(bool)))
    get_concept_masks = None
    if not (args.no_heuristic or args.check_only):
        from src.xai.masks import get_concept_masks

    qc_rows, rows, n_missing_mask, n_missing_image = [], [], 0, 0
    for f in files:
        relpath = sm.relpath_from_filename(f)
        image_path = None
        if args.images_dir and os.path.exists(os.path.join(args.images_dir, f)):
            image_path = os.path.join(args.images_dir, f)
        elif args.data_root and os.path.exists(os.path.join(args.data_root, relpath)):
            image_path = os.path.join(args.data_root, relpath)
        native = load_native(image_path) if image_path else None
        try:
            raw, off = sm.read_label_png(os.path.join(args.labels_dir, f))
        except (OSError, ValueError) as err:             # flag the file instead of stopping the whole check
            print(f"WARNING: {f} is not a readable label map ({err}); skipped")
            qc_rows.append({"file": f, "relpath": relpath, "width": np.nan, "height": np.nan,
                            "off_palette_px": np.nan, "ignore_share": np.nan, "classes": "", "ok": False})
            continue
        size_ok = native is None or raw.shape in (native.shape[:2], sm.MODEL_SIZE)
        ignore_share = float((raw == sm.IGNORE).mean())
        qc_rows.append({"file": f, "relpath": relpath, "width": raw.shape[1], "height": raw.shape[0],
                        "off_palette_px": off, "ignore_share": ignore_share,
                        "classes": " ".join(sm.CLASS_NAMES[k] for k in np.unique(raw) if k < len(sm.CLASS_NAMES)),
                        "ok": bool(size_ok and off == 0 and ignore_share <= MAX_IGNORE_SHARE)})
        if not size_ok:
            print(f"WARNING: {f} is {raw.shape[1]} x {raw.shape[0]}, the image is "
                  f"{native.shape[1]} x {native.shape[0]}; skipped")
            continue
        if args.check_only:
            continue
        gt = label_to_model_geometry(raw, None if native is None else native.shape[:2])
        learned = None
        if masks is not None:
            if relpath in masks["index"]:
                learned = masks["labels"][masks["index"][relpath]]
            else:
                n_missing_mask += 1
        heuristic = None
        if get_concept_masks is not None:
            if native is not None:
                heuristic = get_concept_masks(sm.model_input_image(native))
            else:
                n_missing_image += 1
        subject = sm.strata_from_relpath(relpath)["subject"]
        for row in score_image(gt, learned, heuristic):
            rows.append({"relpath": relpath, "subject": subject, "prefilled": prefilled.get(relpath), **row})
        if args.labels_dir_b and os.path.exists(os.path.join(args.labels_dir_b, f)):
            try:
                raw_b, _ = sm.read_label_png(os.path.join(args.labels_dir_b, f))
                gt_b = label_to_model_geometry(raw_b, None if native is None else native.shape[:2])
            except (OSError, ValueError) as err:
                print(f"WARNING: second label of {f} skipped: {err}")
                continue
            for row in score_annotators(gt, gt_b):
                rows.append({"relpath": relpath, "subject": subject, "prefilled": prefilled.get(relpath), **row})

    os.makedirs(args.out_dir, exist_ok=True)
    label_qc = pd.DataFrame(qc_rows)
    label_qc.to_csv(os.path.join(args.out_dir, f"label_qc_{args.tag}.csv"), index=False)
    print(f"label files: {len(label_qc)} | flagged: {int((~label_qc.ok).sum())} "
          f"(off-palette pixels in {int((label_qc.off_palette_px > 0).sum())}, "
          f"ignore above {MAX_IGNORE_SHARE:.0%} in {int((label_qc.ignore_share > MAX_IGNORE_SHARE).sum())})")
    if n_missing_mask:
        print(f"WARNING: {n_missing_mask} labelled images have no learned mask in {args.masks}")
    if n_missing_image:
        print(f"WARNING: {n_missing_image} labelled images were not found (--images-dir / --data-root): "
              f"no heuristic masks for them")
    if args.check_only:
        print("label check only: no scores computed")
        return
    if not rows:
        print("no scores computed: pass --masks, and --images-dir or --data-root for the heuristic masks")
        return

    per_image = pd.DataFrame(rows)
    subsets = [per_image.assign(subset="all")]
    if per_image.prefilled.notna().any():
        subsets.append(per_image[per_image.prefilled == True].assign(subset="prefilled"))  # noqa: E712
        subsets.append(per_image[per_image.prefilled == False].assign(subset="blind"))     # noqa: E712
    per_image = pd.concat(subsets, ignore_index=True)
    per_image.to_csv(os.path.join(args.out_dir, f"validation_{args.tag}.csv"), index=False)
    summary = summarise(per_image, n_boot=args.n_boot, seed=args.seed)
    summary.to_csv(os.path.join(args.out_dir, f"validation_summary_{args.tag}.csv"), index=False)
    diff = paired_difference(per_image, n_boot=args.n_boot, seed=args.seed)
    report(summary, diff, label_qc, args, os.path.join(args.out_dir, f"validation_{args.tag}.md"))

    pd.set_option("display.width", 200)
    show = summary[summary.subset == "all"][["method", "space", "cls", "n_images", "iou_mean", "iou_low",
                                              "iou_high", "dice_mean", "iou_pooled"]]
    print("\n" + show.round(3).to_string(index=False))
    if len(diff):
        print("\nlearned minus heuristic IoU:\n" + diff.round(3).to_string(index=False))
    print("\nsaved to", args.out_dir)


def main():
    ap = argparse.ArgumentParser(description="Validate eye masks against hand labels, or export images to label.")
    ap.add_argument("--export-sample", action="store_true", help="export images to label instead of validating")
    ap.add_argument("--csv", help="export: image list with relpath and/or link_path columns")
    ap.add_argument("--all-test", action="store_true", help="export: sample from the whole pinned test list")
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    ap.add_argument("--n", type=int, default=60, help="export: number of images")
    ap.add_argument("--n-blind", type=int, default=12, help="export: images without prefill")
    ap.add_argument("--zip", action="store_true", help="export: also write <out-dir>.zip")
    ap.add_argument("--masks", default=None, help="mask file from 17 (prefill for export; learned masks to score)")
    ap.add_argument("--data-root", default=None, help="VISOB root; found under /kaggle/input if not given")
    ap.add_argument("--labels-dir", help="validate: folder of hand-label PNGs")
    ap.add_argument("--labels-dir-b", default=None, help="validate: a second annotator's labels of some images")
    ap.add_argument("--images-dir", default=None, help="validate: folder of native images named like the labels")
    ap.add_argument("--sample-csv", default=None, help="validate: sample.csv of the export (prefilled column)")
    ap.add_argument("--no-heuristic", action="store_true", help="validate: skip the heuristic masks")
    ap.add_argument("--check-only", action="store_true", help="validate: only check the label files")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    if args.data_root is None and os.path.isdir("/kaggle/input"):
        try:
            args.data_root = find_root()
        except FileNotFoundError:
            pass

    if args.export_sample:
        if not (args.csv or args.all_test):
            ap.error("--export-sample needs --csv or --all-test")
        out_dir = args.out_dir or "/kaggle/working/label_pack"
        known = pd.read_csv(args.test_list).relpath.tolist() if os.path.exists(args.test_list) else []
        frame = pd.DataFrame({"relpath": known}) if args.all_test else pd.read_csv(args.csv)
        table = sm.image_table(frame, data_root=args.data_root, known_relpaths=known)
        masks = sm.load_mask_file(args.masks) if args.masks else None
        if masks is None:
            print("WARNING: no --masks: nothing to prefill, every image will be labelled from scratch")
        sample = export_sample(table, out_dir, n=args.n, n_blind=args.n_blind, masks=masks, seed=args.seed,
                               make_zip=args.zip, masks_path=args.masks)
        print(f"exported {len(sample)} images ({int(sample.prefilled.sum())} prefilled) to {out_dir}")
        print(sample.groupby(list(STRATA)).size().rename("images").to_string())
        print(f"distinct subjects: {sample.subject.nunique()}")
        with open(os.path.join(out_dir, "export.json"), "w") as f:
            json.dump({"n": len(sample), "n_prefilled": int(sample.prefilled.sum()), "seed": args.seed,
                       "source": args.csv or "all-test", "masks": args.masks}, f, indent=2)
        return

    if not args.labels_dir:
        ap.error("validation needs --labels-dir (or use --export-sample)")
    args.out_dir = args.out_dir or "validation"
    validate(args)


if __name__ == "__main__":
    main()
