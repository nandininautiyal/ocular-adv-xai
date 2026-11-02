"""
scripts/05_run_experiment.py

Heatmap-divergence experiment for the Ocular Adversarial XAI project.
Run it on Kaggle next to the trained checkpoint, after 04_eval_clean.py has
written the correct-image list (attacks only start from correctly classified
images).

For each sampled image (stratified by class):
  clean            prediction and heatmaps
  noise            control: random +-epsilon sign noise, one per attack epsilon
  defence_on_clean control: each defence applied to the clean image
  noise+defence    control: each defence applied to each noise image
  attack           each white-box attack x epsilon (CW has no epsilon) and the
                   two black-box attacks (patch, monocle)
  attack+defence   each defence applied to every attacked image
Heatmaps are always computed for the TRUE class (does the evidence for the true
identity survive the attack, and come back after the defence?) and, when an
attack flips the prediction, also for the adversarial class. Every map is
compared with the clean map of the same method, at native resolution (8 x 8);
defended maps (attack+defence, noise+defence) are also compared with the
defended clean map (``vs_defended_clean_*`` columns, true-class maps only),
because a defence moves heatmaps on its own. The controls say how much a
heatmap moves for reasons other than the attack; noise+defence is the matched
control for restoration (how close a defended map comes to the defended clean
map when the input was only noisy).

8-bit realism: by default every attacked and noisy image is rounded to the 8-bit
grid a saved PNG or camera frame has, before it is scored (src/attacks/budget.py).
Without this, budgets below 1/255 are only float noise that JPEG or bit-depth
reduction removes for free, which says nothing about the defence. Every budget
must then be a whole number of 8-bit levels: the run stops at start-up on e.g.
0.0078 (1.989 levels, which the rounding would silently run as 1 level).
``--levels 1,2,4`` gives the budgets as levels. Use ``--no-quantise`` only to
reproduce the paper's float-domain setting (``--impl paper`` does so automatically).

Attack and defence settings (all recorded in summary_<tag>.json):
  CW        confidence margin --cw-kappa (default 5 logits, against a clean top-1 /
            top-2 gap of ~30-40). With 8-bit rounding every CW candidate is rounded
            before its success check, so CW returns 8-bit adversarial images;
            --cw-float-success judges success in float as the 27 Oct run did, where
            rounding afterwards undid 37.5 % of the CW successes.
  monocle   on the Haar eye box (paper, Eq. 7); where the cascade finds no eye, on
            the iris (the eye-mask label map with --mask-source FILE, else the Hough
            circle); else at the image centre. Column monocle_placement.
  defences  JPEG (--jpeg-quality, default 75) and bit-depth reduction (--bits,
            default 4). Rows keep the names "jpeg" and "bit_depth".

Columns of rows_<tag>.csv besides the heatmap metrics of src/compare/divergence.py
(which all compare maps: "l2" is the RMS distance between two maps):
  row_id, relpath            row key (into maps_<tag>.npz) and the manifest relpath
  linf, pert_linf, pert_l2   L-inf (linf = pert_linf) and L2 distance of the scored
                             image from the clean image, on the [0, 1] scale
  entropy, entropy_clean     normalised Shannon entropy (0..1) of the native map as
                             scored (min-max scaled) and of the clean map
  raw_sum, raw_max (+_clean) sum and maximum of the native map before min-max scaling
  occluded_frac              patch and monocle rows: share of the image the occluder changed
  monocle_placement          monocle rows: haar, iris or centre_fallback
With --concept-masks, per region (heuristic masks: pupil, iris, sclera, periocular;
an eye-mask file: its class_names, e.g. skin, iris, sclera, lid, caruncle, brow;
label 255 belongs to no region), using the regions of the clean image on a grid of
each method's native map size:
  mass_, dmass_              share of the map on the region, and its change from the clean map
  area_, enr_                the region's share of the image, and mass / area (1 = no preference)
  massvis_, dmassvis_        patch and monocle rows: the same over the pixels the occluder
                             left unchanged (the occlusion mask is every pixel the attack changed;
                             NaN for a region the occluder covers completely)

Writes to --out-dir:
  rows_<tag>.csv       one row per image x condition x XAI method x target
  summary_<tag>.csv    mean metrics per condition / method / target / success
  summary_<tag>.json   attack success, restoration and flip rates, every setting, git
                       commit, package versions and monocle placement counts
  images_<tag>.csv     the sampled images (pass it to --exclude-from of a later run)
  examples_<tag>.npz   images and maps of the first --save-examples images
  maps_<tag>.npz       with --save-maps: every native map (float32) keyed by str(row_id);
                       the clean maps are those of the rows with condition "clean"

Pilot and final run must not share images or a seed:
  pilot : --seed 1 --max-images 8 --tag pilot
  final : --seed 2 --exclude-from /kaggle/working/experiment/images_pilot.csv --tag full

Usage (Kaggle, with the checkpoint dataset attached and 01, 02, 04 already run):
  python scripts/05_run_experiment.py \
      --ckpt /kaggle/input/<checkpoint-dataset>/Xception_ArcFace_full.keras \
      --correct-csv /kaggle/working/eval_clean/correct_test_full.csv \
      --per-class 2 --concept-masks --save-maps --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.blackbox import BOX_SHARE, monocle_blending, patch_occlusion  # noqa: E402
from src.attacks.budget import levels, levels_to_epsilon, quantise  # noqa: E402
from src.attacks.whitebox import ATTACK_REGISTRY, PAPER_ATTACK_REGISTRY  # noqa: E402
from src.compare.divergence import METRICS, compare_heatmaps  # noqa: E402
from src.compare.regions import (IGNORE_LABEL, REGIONS, enrichment, heuristic_masks,  # noqa: E402
                                 label_masks, masks_to_grids, normalised_entropy, occlusion_grids,
                                 occlusion_mask, region_areas, region_mass_named)
from src.defences.inference import DEFENCE_REGISTRY, make_defence  # noqa: E402
from src.models.arcface_inference import find_feature_layer, load_checkpoint, predict_logits  # noqa: E402
from src.xai.attribution import XAI_REGISTRY  # noqa: E402

BLACKBOX = ("patch", "monocle")
PATCH = {"patch_size": 64, "position": (96, 96)}
KEYS = ["condition", "attack", "epsilon", "defence"]
DEFENDED = ("attack+defence", "noise+defence")    # conditions also compared with the defended clean map
PLACEMENTS = ("haar", "iris", "centre_fallback")
DEFAULT_EPSILONS = "0.0039216,0.0078431,0.0156863"
CW_KAPPA = 5.0            # logits; see --cw-kappa
CW_DEFAULTS = {k: p.default for k, p in inspect.signature(ATTACK_REGISTRY["cw"]).parameters.items()}
# numeric columns averaged in summary_<tag>.csv besides the divergence METRICS
SUMMARY_PREFIXES = ("vs_defended_clean_", "mass_", "dmass_", "massvis_", "dmassvis_", "area_", "enr_", "raw_")
SUMMARY_COLUMNS = ("pert_linf", "pert_l2", "entropy", "entropy_clean", "occluded_frac")


def load_image(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def resolve_path(row, data_root):
    if isinstance(row.link_path, str) and os.path.exists(row.link_path):
        return row.link_path
    if data_root and isinstance(getattr(row, "relpath", None), str):
        path = os.path.join(data_root, row.relpath)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{row.link_path} not found; rebuild the split tree (01, 02) "
                            f"or pass --data-root to resolve the manifest relpath")


def takes(fn, name):
    return name in inspect.signature(fn).parameters


def has_budget(fn):
    """True if the attack uses an L-inf epsilon. CW accepts ``epsilon=None`` only so that
    registry calls are uniform; it has no budget and must run once, not once per epsilon."""
    params = inspect.signature(fn).parameters
    return "epsilon" in params and params["epsilon"].default is not None


def split_list(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_budgets(ap, epsilons, levels_text, quantised):
    """The L-inf budgets of the run, from --epsilons or --levels.

    With 8-bit rounding every budget must be a whole number (1 to 255) of 8-bit
    levels (src.attacks.budget.levels): 0.0078 is 1.989 levels, a typo for 2/255
    that the rounding would run as 1 level without a word. ``--levels n`` is
    written as n / 255 to 7 decimals, like the defaults, so ``--levels 1,2,4``
    gives the same rows and summary keys as the default --epsilons. Errors go
    through ``ap.error`` (exit status 2) before anything is loaded.
    """
    if levels_text is not None:
        usage = f"--levels {levels_text}: give whole numbers of 8-bit levels from 1 to 255, e.g. 1,2,4"
        try:
            values = [levels_to_epsilon(int(v)) for v in split_list(levels_text)]
        except ValueError as err:
            ap.error(f"{usage} ({err})")
        if not values or min(values) <= 0:
            ap.error(usage)
        return [round(v, 7) for v in values]
    try:
        values = [float(v) for v in split_list(epsilons)]
    except ValueError as err:
        ap.error(f"--epsilons {epsilons}: give numbers, e.g. {DEFAULT_EPSILONS} ({err})")
    if not values or not all(v > 0 for v in values):
        ap.error(f"--epsilons {epsilons}: budgets must be positive")
    if quantised:
        for eps in values:
            try:
                n = levels(eps)
            except ValueError as err:
                ap.error(f"--epsilons: {err}. With 8-bit rounding (the default) every budget must be a whole "
                         f"number of levels; use --levels 1,2,4, or --no-quantise for float budgets")
            if n < 1:
                ap.error(f"--epsilons: {eps!r} is 0 levels, which leaves the image unchanged")
    return values


def heatmaps(model, xai, images, class_idx, target_layer=None):
    """Native maps of *images* and the sum and maximum of each map before min-max scaling
    (NaN for a method that cannot return them)."""
    fn = XAI_REGISTRY[xai]
    kwargs = {"class_idx": class_idx} if takes(fn, "class_idx") else {}
    if target_layer is not None:
        kwargs["target_layer"] = target_layer
    if not takes(fn, "return_raw"):
        nan = np.full(len(images), np.nan)
        return fn(model, images, size=None, **kwargs), nan, nan
    maps, raw_sum, raw_max = fn(model, images, size=None, return_raw=True, **kwargs)
    return maps, np.asarray(raw_sum, dtype=np.float64), np.asarray(raw_max, dtype=np.float64)


def blackbox_attack(name, images, sides, eye_labels=None, iris_label=None):
    """Patch or monocle images, and where each monocle went (None for the patch).

    With *eye_labels* (the eye-mask label maps of *images*), a monocle the Haar
    cascade cannot place is centred on the pixels labelled *iris_label*;
    without them on the Hough iris circle, and on the image centre if neither
    finds the iris.
    """
    if name == "patch":
        return patch_occlusion(images, **PATCH), None
    out, placements = [], []
    for k, (img, side) in enumerate(zip(images, sides)):
        labels = {} if eye_labels is None else {"eye_labels": eye_labels[k], "iris_label": iris_label}
        adv, placement = monocle_blending(img, side=side, return_placement=True, **labels)
        out.append(adv)
        placements.append(placement)
    return np.stack(out), placements


def load_eye_masks(path, relpaths):
    """Label maps of the images *relpaths* (in that order) from an eye-mask file.

    The file (masks_<tag>.npz) holds ``relpaths``, ``labels`` (N, 256, 256) uint8 in
    model-input geometry (255 = unknown), ``class_names`` (index = label id) and
    ``meta`` (JSON). Masks are looked up by relpath; the run stops, listing them,
    if any sampled image has none.

    Returns
    -------
    labels : ndarray (len(relpaths), 256, 256), uint8
    class_names : tuple of str
    meta : dict
    """
    with np.load(path, allow_pickle=False) as z:
        missing = {"relpaths", "labels", "class_names", "meta"} - set(z.files)
        if missing:
            raise SystemExit(f"--mask-source {path} is not an eye-mask file: it has no {sorted(missing)}")
        index = {str(r).replace("\\", "/"): k for k, r in enumerate(z["relpaths"])}
        absent = sorted(set(relpaths) - set(index))
        if absent:
            raise SystemExit(f"--mask-source {path} has no mask for {len(absent)} of the {len(set(relpaths))} "
                             f"sampled images (make masks for them, or sample from images that have one):\n  "
                             + "\n  ".join(absent))
        class_names = tuple(str(c) for c in z["class_names"])
        labels = z["labels"]
        if labels.ndim != 3 or labels.shape[1:] != (256, 256) or labels.dtype != np.uint8:
            raise SystemExit(f"--mask-source {path}: labels must be uint8 (N, 256, 256) in model-input "
                             f"geometry, got {labels.dtype} {labels.shape}")
        labels = labels[[index[r] for r in relpaths]]
        raw_meta = str(z["meta"][()])
    bad = (labels >= len(class_names)) & (labels != IGNORE_LABEL)
    if bad.any():
        raise SystemExit(f"--mask-source {path}: label ids {np.unique(labels[bad])[:5].tolist()} have no "
                         f"class name (class_names: {list(class_names)}; {IGNORE_LABEL} = unknown)")
    try:
        meta = json.loads(raw_meta)
    except json.JSONDecodeError:
        meta = {"unparsed": raw_meta}
    return labels, class_names, meta


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
    import cv2
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "tensorflow": tf.__version__, "opencv": cv2.__version__}
    try:
        out["tf_keras"] = importlib.metadata.version("tf-keras")
    except importlib.metadata.PackageNotFoundError:
        out["tf_keras"] = None
    return out


def attack_settings(registry, attacks, overrides):
    """Keyword settings each attack ran with: its defaults, updated with *overrides*.

    Budgets of the standard attacks come from --epsilons (they have no default);
    the paper wrappers keep their default epsilon of 0.001.
    """
    out = {}
    for name in attacks:
        if name in BLACKBOX:
            out[name] = (dict(PATCH) if name == "patch"
                         else {"box_share": BOX_SHARE, "placement_order": list(PLACEMENTS)})
            continue
        params = inspect.signature(registry[name]).parameters
        out[name] = {k: p.default for k, p in params.items()
                     if k != "label" and isinstance(p.default, (bool, int, float, str, type(None)))}
        if "seed" in params:
            out[name]["seed"] = "index of the batch's first image"
        out[name].update(overrides.get(name, {}))
    return out


class Evaluator:
    """Scores one batch of image variants against the clean batch and collects rows.

    regions : None, or (names, masks): region names and pixel masks of the clean
        images, (N, R, 256, 256) bool (heuristic masks or an eye-mask file); they are
        area-averaged to the native map size of each XAI method
    first_row_id : row_id of the batch's first row (row ids are unique within a run)
    keep_maps : keep every native map (float32) in ``self.maps``, keyed by str(row_id)
    target_layer : layer the XAI methods explain (None: their default)
    """

    def __init__(self, model, xai_names, clean, labels, base_rows, regions=None, first_row_id=0,
                 keep_maps=False, target_layer=None):
        self.model, self.xai_names, self.target_layer = model, xai_names, target_layer
        self.clean, self.labels, self.base_rows = clean, labels, base_rows
        self.clean_maps, self.clean_raw, self.clean_entropy = {}, {}, {}
        for xai in xai_names:
            uses_class = takes(XAI_REGISTRY[xai], "class_idx")
            maps, raw_sum, raw_max = heatmaps(model, xai, clean, labels if uses_class else None, target_layer)
            self.clean_maps[xai], self.clean_raw[xai] = maps, (raw_sum, raw_max)
            self.clean_entropy[xai] = [normalised_entropy(m) for m in maps]
        # Region masses use the anatomy of the clean image for every variant, area-averaged to
        # the native map size of each method (methods may differ, e.g. 8 x 8 and 16 x 16).
        self.region_names, self.region_masks = regions if regions is not None else (None, None)
        self.grids, self.areas, self.clean_mass = {}, {}, {}
        if regions is not None:
            for xai, maps in self.clean_maps.items():
                shape = maps.shape[1:]
                if shape not in self.grids:
                    self.grids[shape] = masks_to_grids(self.region_masks, shape)
                    self.areas[shape] = [region_areas(g, self.region_names) for g in self.grids[shape]]
                self.clean_mass[xai] = [region_mass_named(m, g, self.region_names)
                                        for m, g in zip(maps, self.grids[shape])]
        self.defended_clean_maps = {}      # (xai, defence, target) -> maps of the defended clean batch
        self.rows, self.maps, self.examples = [], {}, {}
        self.next_row_id, self.keep_maps = first_row_id, keep_maps

    def __call__(self, images, info, extra=None, occluded=None):
        """Score *images* (same order as the clean batch); return the predictions.

        Maps of defended images (attack+defence, noise+defence) are also compared
        with the map of the defended clean image (columns ``vs_defended_clean_*``):
        a defence changes the heatmap of a clean image too, so that map, not the
        clean one, is the reference for whether the defence restored the
        explanation. The reference exists for the true-class maps (and
        class-agnostic methods) only; rows for the adversarial-class target carry
        no such columns.

        *occluded* (N, H, W) bool marks the pixels an occluder (patch, monocle)
        changed: those rows get occluded_frac and, with regions, massvis_<region>
        and dmassvis_<region>, the region masses over the pixels left unchanged.
        """
        pred = predict_logits(self.model, images).argmax(1)
        diff = (images - self.clean).reshape(len(images), -1)
        linf, l2 = np.abs(diff).max(1), np.sqrt((diff ** 2).sum(1))
        flipped = pred != self.labels
        visible = {}
        if occluded is not None and self.grids:
            visible = {shape: occlusion_grids(self.region_masks, occluded, shape) for shape in self.grids}

        for xai in self.xai_names:
            uses_class = takes(XAI_REGISTRY[xai], "class_idx")
            targets = {"true": self.labels} if uses_class else {"any": None}
            if uses_class and flipped.any() and info["condition"] != "clean":
                targets["pred"] = pred
            clean_maps, (clean_sum, clean_max) = self.clean_maps[xai], self.clean_raw[xai]
            for target, class_idx in targets.items():
                maps, raw_sum, raw_max = heatmaps(self.model, xai, images, class_idx, self.target_layer)
                shape = maps.shape[1:]
                # Keyed by target as well: the adversarial-class maps of a defended clean
                # image must never replace its true-class maps as the reference.
                if info["condition"] == "defence_on_clean" and target != "pred":
                    self.defended_clean_maps[(xai, info["defence"], target)] = maps
                defended_ref = None
                if info["condition"] in DEFENDED and target != "pred":
                    defended_ref = self.defended_clean_maps.get((xai, info["defence"], target))
                for i in range(len(images)):
                    if target == "pred" and not flipped[i]:
                        continue
                    row = {"row_id": self.next_row_id, **self.base_rows[i], **info, "xai": xai,
                           "target": target, "pred_idx": int(pred[i]), "correct": bool(not flipped[i]),
                           "linf": float(linf[i]), "pert_linf": float(linf[i]), "pert_l2": float(l2[i])}
                    if extra:
                        row.update({k: v[i] for k, v in extra.items()})
                    row.update(compare_heatmaps(clean_maps[i], maps[i]))
                    row.update({"entropy": normalised_entropy(maps[i]),
                                "entropy_clean": self.clean_entropy[xai][i],
                                "raw_sum": float(raw_sum[i]), "raw_max": float(raw_max[i]),
                                "raw_sum_clean": float(clean_sum[i]), "raw_max_clean": float(clean_max[i])})
                    if self.grids:
                        mass = region_mass_named(maps[i], self.grids[shape][i], self.region_names)
                        clean_mass = self.clean_mass[xai][i]
                        row.update(mass)
                        row.update({f"d{k}": v - clean_mass[k] for k, v in mass.items()})
                        row.update(self.areas[shape][i])
                        row.update(enrichment(mass, self.areas[shape][i]))
                    if occluded is not None:
                        row["occluded_frac"] = float(occluded[i].mean())
                        if self.grids:
                            grids, weight = visible[shape][0][i], visible[shape][1][i]
                            vis = region_mass_named(maps[i], grids, self.region_names, visible=weight,
                                                    prefix="massvis")
                            clean_vis = region_mass_named(clean_maps[i], grids, self.region_names,
                                                          visible=weight, prefix="massvis")
                            row.update(vis)
                            row.update({f"d{k}": v - clean_vis[k] for k, v in vis.items()})
                    if defended_ref is not None:
                        row.update({f"vs_defended_clean_{k}": v
                                    for k, v in compare_heatmaps(defended_ref[i], maps[i]).items()})
                    self.rows.append(row)
                    if self.keep_maps:
                        self.maps[str(self.next_row_id)] = np.array(maps[i], dtype=np.float32)
                    self.next_row_id += 1
                    if i in self.examples:
                        key = "|".join(str(info.get(k, "")) for k in KEYS) + f"|{xai}|{target}"
                        self.examples[i][key] = (images[i], maps[i])
        return pred


def summarise(rows):
    """Per-condition rates, NaN-aware mean metrics, and per-metric NaN counts."""
    rows = rows.copy()
    for col in ("attack", "epsilon", "defence"):
        if col not in rows:
            rows[col] = np.nan
    rows["epsilon"] = rows["epsilon"].map(lambda e: "" if pd.isna(e) else f"{e:.6g}")
    rows[["attack", "defence"]] = rows[["attack", "defence"]].fillna("")
    per_variant = rows.drop_duplicates(subset=["image_id"] + KEYS)

    def rate(frame, col, value, prefix=""):
        out = {}
        for (attack, eps, defence), g in frame.groupby(["attack", "epsilon", "defence"]):
            name = (attack or prefix) + (f"@{eps}" if eps else "") + (f"+{defence}" if defence else "")
            out[name] = {"rate": float((g[col] == value).mean()), "n": int(len(g))}
        return out

    def condition(name):
        return per_variant[per_variant.condition == name]

    defended = condition("attack+defence")
    if len(defended):
        defended = defended[defended.attack_success.astype(bool)]
    rates = {
        "attack_success_rate": rate(condition("attack"), "correct", False),
        "restoration_rate_among_successful_attacks": rate(defended, "correct", True),
        "noise_control_flip_rate": rate(condition("noise"), "correct", False, prefix="noise"),
        "defence_on_clean_flip_rate": rate(condition("defence_on_clean"), "correct", False, prefix="clean"),
        "noise_defence_flip_rate": rate(condition("noise+defence"), "correct", False, prefix="noise"),
    }
    sizes = {}       # image-space perturbation of each attack, per image
    for (attack, eps), g in condition("attack").groupby(["attack", "epsilon"]):
        sizes[attack + (f"@{eps}" if eps else "")] = {
            "pert_l2_median": float(g.pert_l2.median()), "pert_linf_median": float(g.pert_linf.median()),
            "pert_linf_max": float(g.pert_linf.max()), "n": int(len(g))}
    rates["perturbation_size"] = sizes

    rows["success"] = ~rows["correct"]
    group = KEYS + ["xai", "target", "success"]
    metrics = [c for c in rows.columns
               if c in METRICS or c in SUMMARY_COLUMNS or c.startswith(SUMMARY_PREFIXES)]
    grouped = rows.groupby(group, dropna=False)
    columns = {"n": grouped.size()}    # built once: inserting column by column fragments the frame
    for metric in metrics:
        columns[metric] = grouped[metric].agg(
            lambda values: float(np.nanmean(values)) if values.notna().any() else float("nan"))
        columns[f"{metric}_nan_count"] = grouped[metric].agg(lambda values: int(values.isna().sum()))
    table = pd.concat(columns, axis=1)
    return rates, table.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None,
                    help="smoke run: cap the number of images (8 images take a few minutes on a Kaggle T4)")
    ap.add_argument("--exclude-from", default=None,
                    help="CSV with a link_path column (images_<tag>.csv or features_<tag>.csv of an "
                         "earlier run); those images are not used, e.g. to keep a final run disjoint "
                         "from its pilot")
    ap.add_argument("--impl", choices=["standard", "paper"], default="standard",
                    help="white-box implementation: standard (ours) or paper (vendored, eps=0.001)")
    ap.add_argument("--attacks", default="fgsm,bim,pgd,mim,cw,patch,monocle")
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument("--epsilons", default=DEFAULT_EPSILONS,
                        help="L-inf budgets: 1/255, 2/255 and 4/255. Standard BIM/PGD/MIM already reach 100%% "
                             "success at 2/255 on the full baseline (scripts/08_check_attacks.py); 4/255 shows "
                             "how heatmap shift and restoration keep changing once success saturates. With 8-bit "
                             "rounding every budget must be a whole number of 8-bit levels (0.0078431 = 2 "
                             "levels; 0.0078 stops the run); other budgets are only meaningful with --no-quantise")
    budget.add_argument("--levels", default=None,
                        help="L-inf budgets as whole 8-bit levels instead of --epsilons, e.g. 1,2,4 (= 1/255, "
                             "2/255, 4/255, written to 7 decimals like the --epsilons defaults)")
    ap.add_argument("--no-quantise", action="store_true",
                    help="keep attacked images in float32 (the paper's setting). Default: round every "
                         "attacked or noisy image to the 8-bit grid before scoring it")
    ap.add_argument("--cw-kappa", type=float, default=CW_KAPPA,
                    help="CW confidence margin in margin-free logits (s * cos, s = 64): a CW image counts as "
                         "adversarial only when some other class beats the true class by more than this. "
                         "Default 5, against a clean top-1 / top-2 gap of ~30-40; 0 gives the minimal attack, "
                         "whose images sit on the decision boundary")
    ap.add_argument("--cw-steps", type=int, default=CW_DEFAULTS["steps"],
                    help="CW Adam steps per binary-search step")
    ap.add_argument("--cw-binary-steps", type=int, default=CW_DEFAULTS["binary_steps"],
                    help="CW binary-search steps over its constant c")
    ap.add_argument("--cw-float-success", action="store_true",
                    help="judge CW success on the float image and round it afterwards (the 27 Oct protocol, "
                         "where rounding undid 37.5 %% of the CW successes). Default with 8-bit rounding: "
                         "every CW candidate is rounded before its success check")
    ap.add_argument("--defences", default="jpeg,bit_depth")
    ap.add_argument("--jpeg-quality", type=int, default=75,
                    help="JPEG quality of the jpeg defence (1-100; the base paper does not report one)")
    ap.add_argument("--bits", type=int, default=4,
                    help="bit depth of the bit_depth defence (1-8; the base paper does not report one)")
    ap.add_argument("--xai", default="layercam,eigencam")
    ap.add_argument("--target-layer", default=None,
                    help="layer the XAI methods explain, e.g. block13_sepconv2_act (16 x 16 maps). Default: "
                         "the last 4-D activation before the head (block14_sepconv2_act, 8 x 8)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-examples", type=int, default=4)
    ap.add_argument("--save-maps", action="store_true",
                    help="write every native map (float32) to maps_<tag>.npz, keyed by the row_id column")
    ap.add_argument("--concept-masks", action="store_true",
                    help="add region columns (see --mask-source): mass_<region> (heatmap fraction on the "
                         "region, masks from the clean image), dmass_<region> (change vs the clean map), "
                         "area_<region> and enr_<region>, and massvis_/dmassvis_<region> for occluders")
    ap.add_argument("--mask-source", default="heuristic",
                    help="regions for --concept-masks: 'heuristic' (src/xai/masks.py: pupil, iris, sclera, "
                         "periocular) or the path of an eye-mask file masks_<tag>.npz (its class_names, looked "
                         "up by relpath; the monocle is then centred on its iris). A file turns on "
                         "--concept-masks")
    ap.add_argument("--out-dir", default="/kaggle/working/experiment")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    # --- settings, checked before anything slow is loaded --------------------------------
    attacks, defences, xai_names = split_list(args.attacks), split_list(args.defences), split_list(args.xai)
    registry = ATTACK_REGISTRY if args.impl == "standard" else PAPER_ATTACK_REGISTRY
    for name in attacks:
        if name not in registry and name not in BLACKBOX:
            ap.error(f"unknown attack {name!r}; choose from {sorted(registry) + list(BLACKBOX)}")
    for name in xai_names:
        if name not in XAI_REGISTRY:
            ap.error(f"unknown XAI method {name!r}; choose from {sorted(XAI_REGISTRY)}")
        if args.target_layer is not None and not takes(XAI_REGISTRY[name], "target_layer"):
            ap.error(f"--target-layer: {name} does not take a target layer")
    if not (np.isfinite(args.cw_kappa) and args.cw_kappa >= 0):
        ap.error(f"--cw-kappa must be a margin >= 0 logits, got {args.cw_kappa}")
    if args.cw_steps < 1 or args.cw_binary_steps < 1:
        ap.error("--cw-steps and --cw-binary-steps must be at least 1")
    parameters = {"jpeg": {"quality": args.jpeg_quality}, "bit_depth": {"bits": args.bits}}
    defence_fns = {}
    for name in defences:
        try:
            defence_fns[name] = make_defence(name, **parameters.get(name, {}))
        except KeyError:
            ap.error(f"unknown defence {name!r}; choose from {sorted(DEFENCE_REGISTRY)}")
        except (TypeError, ValueError) as err:
            ap.error(str(err))

    quantised = not args.no_quantise
    if args.impl == "paper" and quantised:
        print("note: --impl paper reproduces the vendored float-domain setting, so 8-bit rounding is off")
        quantised = False
    epsilons = parse_budgets(ap, args.epsilons, args.levels, quantised)
    q = quantise if quantised else (lambda images, *budget: np.asarray(images, dtype=np.float32))
    # CW: confidence margin, and success judged on the rounded image when images are rounded
    cw_kwargs = {"kappa": args.cw_kappa, "steps": args.cw_steps, "binary_steps": args.cw_binary_steps}
    cw_quantise_aware = quantised and not args.cw_float_success
    if cw_quantise_aware:
        cw_kwargs["quantise_fn"] = quantise
    mask_file = None if args.mask_source == "heuristic" else args.mask_source
    if mask_file is not None and not os.path.isfile(mask_file):
        ap.error(f"--mask-source {mask_file}: no such file (or use 'heuristic')")
    concept_masks = args.concept_masks or mask_file is not None
    if mask_file is not None and not args.concept_masks:
        print("note: --mask-source with a file turns on --concept-masks")

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    # --- images ----------------------------------------------------------------------
    df = pd.read_csv(args.correct_csv)
    if args.exclude_from:
        seen = set(pd.read_csv(args.exclude_from, usecols=["link_path"]).link_path)
        before = len(df)
        df = df[~df.link_path.isin(seen)]
        print(f"excluded {before - len(df)} images listed in {args.exclude_from}")
    if df.empty:
        raise SystemExit("no images left to sample (check --correct-csv and --exclude-from)")
    df = df.sample(frac=1.0, random_state=args.seed).groupby("true_idx").head(args.per_class)
    if args.max_images:
        df = df.head(args.max_images)
    df = df.reset_index().rename(columns={"index": "image_id"})
    if "relpath" in df.columns:          # manifest relpaths, data-root relative with "/" (mask files use them)
        df["relpath"] = df["relpath"].map(lambda r: r.replace("\\", "/") if isinstance(r, str) else r)

    mask_labels, class_names, mask_meta, iris_label = None, None, None, None
    if mask_file is not None:
        if "relpath" not in df.columns or not df.relpath.map(lambda r: isinstance(r, str)).all():
            raise SystemExit("--mask-source looks masks up by relpath, but --correct-csv has no relpath for "
                             "some images (04_eval_clean.py writes a relpath column)")
        mask_labels, class_names, mask_meta = load_eye_masks(mask_file, df.relpath.tolist())
        if "iris" in class_names:
            iris_label = class_names.index("iris")
        else:
            print(f"note: {mask_file} has no 'iris' class, so the monocle uses the Hough iris circle")
        print(f"eye masks: {mask_file} | regions {list(class_names)} | {len(df)} images found")
    region_names = (class_names if mask_file is not None else REGIONS) if concept_masks else None

    model = load_checkpoint(args.ckpt)
    print("loaded:", args.ckpt)
    if args.target_layer is not None:
        try:
            model.get_layer(args.target_layer)
        except ValueError:
            raise SystemExit(f"--target-layer {args.target_layer!r} is not a layer of {args.ckpt}") from None
    target_layer = args.target_layer or find_feature_layer(model)

    print(f"images: {len(df)} from {df.true_idx.nunique()} classes | attacks: {attacks} | "
          f"epsilons: {epsilons} | defences: {[defence_fns[d].label for d in defences]} | xai: {xai_names} "
          f"at {target_layer} | impl: {args.impl} | 8-bit: {quantised} | seed: {args.seed}")
    if "cw" in attacks and args.impl == "standard":
        print(f"CW: kappa {args.cw_kappa:g} logits | {args.cw_binary_steps} x {args.cw_steps} steps | "
              f"success judged {'after 8-bit rounding' if cw_quantise_aware else 'in float'}")
    if concept_masks:
        print(f"regions: {list(region_names)} from {mask_file or 'heuristic masks (src/xai/masks.py)'}")

    # --- experiment ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    base_cols = [c for c in ["image_id", "relpath", "true_idx", "true_subject", "eye", "device", "lighting",
                             "phase"] if c in df.columns]
    all_rows, examples, maps = [], {}, {}
    placement_counts = Counter()
    t0 = time.time()
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([load_image(resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        sides = [e if e in ("left", "right") else None for e in batch.get("eye", [None] * len(batch))]
        eye_labels = None if mask_labels is None else mask_labels[start:start + len(batch)]
        regions = None
        if concept_masks:
            regions = ((REGIONS, heuristic_masks(clean)) if eye_labels is None
                       else (class_names, label_masks(eye_labels, class_names)))

        ev = Evaluator(model, xai_names, clean, labels, batch[base_cols].to_dict("records"), regions=regions,
                       first_row_id=len(all_rows), keep_maps=args.save_maps, target_layer=args.target_layer)
        ev.examples = {i: {} for i in range(len(batch)) if start + i < args.save_examples}

        pred = ev(clean, {"condition": "clean"})
        if (pred != labels).any():
            print(f"  WARNING: {(pred != labels).sum()} 'correct' images are misclassified now")

        noisy = {}
        for eps in epsilons:
            noise = eps * np.sign(rng.uniform(-1.0, 1.0, clean.shape)).astype(np.float32)
            noisy[eps] = q(np.clip(clean + noise, 0.0, 1.0), clean, eps)
            ev(noisy[eps], {"condition": "noise", "epsilon": eps})
        for d, defend in defence_fns.items():
            ev(defend(clean), {"condition": "defence_on_clean", "defence": d})
        for eps, images in noisy.items():          # after defence_on_clean: its maps are the reference
            for d, defend in defence_fns.items():
                ev(defend(images), {"condition": "noise+defence", "epsilon": eps, "defence": d})

        for name in attacks:
            extra, occluded = {}, None
            if name in BLACKBOX:
                adv, placements = blackbox_attack(name, clean, sides,
                                                  eye_labels if iris_label is not None else None, iris_label)
                adv = q(adv)
                occluded = occlusion_mask(clean, adv)          # every pixel the occluder changed
                if placements is not None:
                    extra["monocle_placement"] = placements
                    placement_counts.update(placements)
                runs = [(None, adv)]
            else:
                fn = registry[name]
                kwargs = {"label": labels} if takes(fn, "label") else {}
                if takes(fn, "seed"):
                    kwargs["seed"] = start
                if name == "cw" and args.impl == "standard":
                    kwargs.update(cw_kwargs)
                if not has_budget(fn):
                    budgets = [None]                                   # CW: no epsilon
                elif args.impl == "standard":
                    budgets = epsilons
                else:
                    budgets = [inspect.signature(fn).parameters["epsilon"].default]  # paper: 0.001
                runs = [(eps, q(fn(clean, model, **({"epsilon": eps} if eps is not None else {}), **kwargs),
                                *((clean, eps) if eps is not None else ())))
                        for eps in budgets]
            for eps, adv in runs:
                info = {"condition": "attack", "attack": name, "epsilon": eps}
                adv_pred = ev(adv, info, extra=extra, occluded=occluded)
                success = {**extra, "attack_success": adv_pred != labels}
                for d, defend in defence_fns.items():
                    ev(defend(adv), {**info, "condition": "attack+defence", "defence": d},
                       extra=success, occluded=occluded)

        all_rows += ev.rows
        maps.update(ev.maps)
        examples.update({start + i: v for i, v in ev.examples.items()})
        done = start + len(batch)
        minutes = (time.time() - t0) / 60
        print(f"  {done}/{len(df)} images | {minutes:.1f} min | about {minutes / done * (len(df) - done):.0f} "
              f"min left", flush=True)

    # --- outputs ---------------------------------------------------------------------
    rows = pd.DataFrame(all_rows)
    rates, table = summarise(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, f"rows_{args.tag}.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, f"summary_{args.tag}.csv"), index=False)
    df[[c for c in ["image_id", "link_path", "relpath", "true_idx", "true_subject", "eye", "device", "lighting",
                    "phase"] if c in df.columns]].to_csv(os.path.join(args.out_dir, f"images_{args.tag}.csv"),
                                                         index=False)
    maps_file = None
    if args.save_maps:
        maps_file = f"maps_{args.tag}.npz"
        np.savez_compressed(os.path.join(args.out_dir, maps_file), **maps)

    commit, dirty = git_state()
    overrides = {"cw": {**cw_kwargs, "quantise_fn": "src.attacks.budget.quantise" if cw_quantise_aware else None}
                 if args.impl == "standard" else {}}
    summary = {"tag": args.tag, "ckpt": args.ckpt, "impl": args.impl,
               "attack_loss": "margin (standard attacks)" if args.impl == "standard" else "vendored paper loss",
               "n_images": int(len(df)), "n_rows": int(len(rows)), "concept_masks": bool(concept_masks),
               "mask_source": mask_file or "heuristic", "mask_meta": mask_meta,
               "mask_file_sha256": file_sha256(mask_file) if mask_file is not None else None,
               "regions": list(region_names) if concept_masks else None,
               "quantised_8bit": bool(quantised), "seed": args.seed, "exclude_from": args.exclude_from,
               "epsilons": epsilons, "levels": [levels(e) for e in epsilons] if quantised else None,
               "attacks": attacks, "attack_settings": attack_settings(registry, attacks, overrides),
               "cw": ({"kappa": args.cw_kappa, "steps": args.cw_steps, "binary_steps": args.cw_binary_steps,
                       "success_after_8bit_rounding": bool(cw_quantise_aware)}
                      if args.impl == "standard" else None),
               "monocle_placement_counts": ({**{p: 0 for p in PLACEMENTS},
                                             **{k: int(v) for k, v in placement_counts.items()}}
                                            if "monocle" in attacks else None),
               "monocle_eye_labels": bool(mask_file is not None and iris_label is not None),
               "defences": defences,
               "defence_settings": {d: dict(defence_fns[d].keywords) for d in defences},
               "xai": xai_names, "xai_target_layer": target_layer,
               "maps": "native resolution, min-max scaled; entropy is computed on these maps, raw_sum and "
                       "raw_max before the scaling",
               "maps_file": maps_file, "git_commit": commit, "git_dirty": dirty, "versions": package_versions(),
               "args": vars(args), "minutes": round((time.time() - t0) / 60, 1), **rates}
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if examples:
        arrays = {}
        for image_id, entries in examples.items():
            for key, (img, heat) in entries.items():
                arrays[f"{image_id}|{key}|image"] = (img * 255).round().astype(np.uint8)
                arrays[f"{image_id}|{key}|map"] = heat.astype(np.float16)
        np.savez_compressed(os.path.join(args.out_dir, f"examples_{args.tag}.npz"), **arrays)

    print(json.dumps({k: rates[k] for k in rates if k != "perturbation_size"}, indent=2))
    if summary["monocle_placement_counts"]:
        print("monocle placement:", summary["monocle_placement_counts"])
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()
