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
  attack           each white-box attack x epsilon (CW has no epsilon) and the
                   two black-box attacks (patch, monocle)
  attack+defence   each defence applied to every attacked image
Heatmaps are always computed for the TRUE class (does the evidence for the true
identity survive the attack, and come back after the defence?) and, when an
attack flips the prediction, also for the adversarial class. Every map is
compared with the clean map of the same method, at native resolution (8 x 8);
attacked-then-defended maps are also compared with the defended clean map
(``vs_defended_clean_*`` columns, true-class maps only), because a defence moves
heatmaps on its own. The controls say how much a heatmap moves for reasons other
than the attack.

8-bit realism: by default every attacked and noisy image is rounded to the 8-bit
grid a saved PNG or camera frame has, before it is scored. Without this, budgets
below 1/255 are only float noise that JPEG or bit-depth reduction removes for
free, which says nothing about the defence. Budgets below 1/255 are therefore
not meaningful here; use ``--no-quantise`` only to reproduce the paper's float-
domain setting (``--impl paper`` does so automatically).

Writes to --out-dir:
  rows_<tag>.csv       one row per image x condition x XAI method x target
  summary_<tag>.csv    mean metrics per condition / method / target / success
  summary_<tag>.json   attack success rate, restoration rate, noise flip rate
  images_<tag>.csv     the sampled images (pass it to --exclude-from of a later run)
  examples_<tag>.npz   images and maps of the first --save-examples images

Pilot and final run must not share images or a seed:
  pilot : --seed 1 --max-images 8 --tag pilot
  final : --seed 2 --exclude-from /kaggle/working/experiment/images_pilot.csv --tag full

Usage (Kaggle, with the checkpoint dataset attached and 01, 02, 04 already run):
  python scripts/05_run_experiment.py \
      --ckpt /kaggle/input/<checkpoint-dataset>/Xception_ArcFace_full.keras \
      --correct-csv /kaggle/working/eval_clean/correct_test_full.csv \
      --per-class 2 --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import inspect
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.blackbox import monocle_blending, patch_occlusion  # noqa: E402
from src.attacks.whitebox import ATTACK_REGISTRY, PAPER_ATTACK_REGISTRY  # noqa: E402
from src.compare.divergence import METRICS, compare_heatmaps  # noqa: E402
from src.compare.regions import region_grids, region_mass  # noqa: E402
from src.defences.inference import DEFENCE_REGISTRY  # noqa: E402
from src.models.arcface_inference import load_checkpoint, predict_logits  # noqa: E402
from src.xai.attribution import XAI_REGISTRY  # noqa: E402

BLACKBOX = ("patch", "monocle")
KEYS = ["condition", "attack", "epsilon", "defence"]
ONE_LEVEL = 1.0 / 255.0       # the smallest 8-bit step


def load_image(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


def quantise(images, reference=None, epsilon=None):
    """Round to the 8-bit grid a saved image has: what a deployed system actually receives.

    With *reference* (the clean images, already on the grid) and *epsilon*, the result is
    also clipped to the largest whole number of 8-bit levels within the budget, so rounding
    never pushes a perturbation past epsilon (e.g. epsilon = 0.03 allows 7 levels, not 8).
    """
    images = np.asarray(images, dtype=np.float32)
    out = np.round(np.clip(images, 0.0, 1.0) * 255.0) / 255.0
    if reference is not None and epsilon is not None:
        levels = np.floor(epsilon * 255.0 + 1e-6) / 255.0
        reference = np.asarray(reference, dtype=np.float32)
        out = np.clip(out, reference - levels, reference + levels)
    return out.astype(np.float32)


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


def heatmaps(model, xai, images, class_idx):
    fn = XAI_REGISTRY[xai]
    kwargs = {"class_idx": class_idx} if takes(fn, "class_idx") else {}
    return fn(model, images, size=None, **kwargs)


def blackbox_attack(name, images, sides):
    if name == "patch":
        return patch_occlusion(images, patch_size=64, position=(96, 96))
    return np.stack([monocle_blending(img, side=side) for img, side in zip(images, sides)])


class Evaluator:
    """Scores one batch of image variants against the clean batch and collects rows."""

    def __init__(self, model, xai_names, clean, labels, base_rows, concept_masks=False):
        self.model, self.xai_names = model, xai_names
        self.clean, self.labels, self.base_rows = clean, labels, base_rows
        self.clean_maps = {}
        for xai in xai_names:
            uses_class = takes(XAI_REGISTRY[xai], "class_idx")
            self.clean_maps[xai] = heatmaps(model, xai, clean, labels if uses_class else None)
        # Region masses use the anatomy of the clean image for every variant.
        self.grids = (region_grids(clean, next(iter(self.clean_maps.values())).shape[-1])
                      if concept_masks else None)
        self.defended_clean_maps = {}      # (xai, defence, target) -> maps of the defended clean batch
        self.rows = []
        self.examples = {}

    def __call__(self, images, info, extra=None):
        """Score *images* (same order as the clean batch); return the predictions.

        Maps of attacked-then-defended images are also compared with the map of
        the defended clean image (columns ``vs_defended_clean_*``): a defence
        changes the heatmap of a clean image too, so that map, not the clean
        one, is the reference for whether the defence restored the explanation.
        The reference exists for the true-class maps (and class-agnostic methods)
        only; rows for the adversarial-class target carry no such columns.
        """
        pred = predict_logits(self.model, images).argmax(1)
        diff = (images - self.clean).reshape(len(images), -1)
        linf, l2 = np.abs(diff).max(1), np.sqrt((diff ** 2).sum(1))
        flipped = pred != self.labels

        for xai in self.xai_names:
            uses_class = takes(XAI_REGISTRY[xai], "class_idx")
            targets = {"true": self.labels} if uses_class else {"any": None}
            if uses_class and flipped.any() and info["condition"] != "clean":
                targets["pred"] = pred
            for target, class_idx in targets.items():
                maps = heatmaps(self.model, xai, images, class_idx)
                # Keyed by target as well: the adversarial-class maps of a defended clean
                # image must never replace its true-class maps as the reference.
                if info["condition"] == "defence_on_clean" and target != "pred":
                    self.defended_clean_maps[(xai, info["defence"], target)] = maps
                defended_ref = None
                if info["condition"] == "attack+defence" and target != "pred":
                    defended_ref = self.defended_clean_maps.get((xai, info["defence"], target))
                for i in range(len(images)):
                    if target == "pred" and not flipped[i]:
                        continue
                    row = {**self.base_rows[i], **info, "xai": xai, "target": target,
                           "pred_idx": int(pred[i]), "correct": bool(not flipped[i]),
                           "linf": float(linf[i]), "l2": float(l2[i])}
                    if extra:
                        row.update({k: v[i] for k, v in extra.items()})
                    row.update(compare_heatmaps(self.clean_maps[xai][i], maps[i]))
                    if self.grids is not None:
                        mass = region_mass(maps[i], self.grids[i])
                        clean_mass = region_mass(self.clean_maps[xai][i], self.grids[i])
                        row.update(mass)
                        row.update({f"d{k}": v - clean_mass[k] for k, v in mass.items()})
                    if defended_ref is not None:
                        row.update({f"vs_defended_clean_{k}": v
                                    for k, v in compare_heatmaps(defended_ref[i], maps[i]).items()})
                    self.rows.append(row)
                    if i in self.examples:
                        key = "|".join(str(info.get(k, "")) for k in KEYS) + f"|{xai}|{target}"
                        self.examples[i][key] = (images[i], maps[i])
        return pred


def summarise(rows):
    """Per-condition rates, NaN-aware mean metrics, and per-metric NaN counts."""
    rows = rows.copy()
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
    }

    rows["success"] = ~rows["correct"]
    group = KEYS + ["xai", "target", "success"]
    metrics = [c for c in rows.columns
               if c in METRICS or c.startswith(("vs_defended_clean_", "mass_", "dmass_"))]
    grouped = rows.groupby(group, dropna=False)
    table = pd.DataFrame(index=grouped.size().index)
    for metric in metrics:
        table[metric] = grouped[metric].agg(
            lambda values: float(np.nanmean(values)) if values.notna().any() else float("nan"))
        table[f"{metric}_nan_count"] = grouped[metric].agg(lambda values: int(values.isna().sum()))
    table.insert(0, "n", grouped.size())
    return rates, table.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None, help="debug: cap the number of images")
    ap.add_argument("--exclude-from", default=None,
                    help="CSV with a link_path column (images_<tag>.csv or features_<tag>.csv of an "
                         "earlier run); those images are not used, e.g. to keep a final run disjoint "
                         "from its pilot")
    ap.add_argument("--impl", choices=["standard", "paper"], default="standard",
                    help="white-box implementation: standard (ours) or paper (vendored, eps=0.001)")
    ap.add_argument("--attacks", default="fgsm,bim,pgd,mim,cw,patch,monocle")
    ap.add_argument("--epsilons", default="0.0039216,0.0078431",
                    help="L-inf budgets: 1/255 and 2/255. Standard BIM/PGD/MIM already reach 100%% "
                         "success at 2/255 on the full baseline (scripts/08_check_attacks.py). Budgets "
                         "below 1/255 are not representable in 8 bits and are only meaningful with "
                         "--no-quantise")
    ap.add_argument("--no-quantise", action="store_true",
                    help="keep attacked images in float32 (the paper's setting). Default: round every "
                         "attacked or noisy image to the 8-bit grid before scoring it")
    ap.add_argument("--defences", default="jpeg,bit_depth")
    ap.add_argument("--xai", default="layercam,eigencam")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-examples", type=int, default=4)
    ap.add_argument("--concept-masks", action="store_true",
                    help="add mass_<region> (heatmap fraction on pupil/iris/sclera/periocular, "
                         "masks from the clean image) and dmass_<region> (change vs the clean map)")
    ap.add_argument("--out-dir", default="/kaggle/working/experiment")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    epsilons = [float(e) for e in args.epsilons.split(",")]
    defences = [d.strip() for d in args.defences.split(",") if d.strip()]
    xai_names = [x.strip() for x in args.xai.split(",") if x.strip()]
    registry = ATTACK_REGISTRY if args.impl == "standard" else PAPER_ATTACK_REGISTRY
    for name in attacks:
        assert name in registry or name in BLACKBOX, f"unknown attack {name}"

    quantised = not args.no_quantise
    if args.impl == "paper" and quantised:
        print("note: --impl paper reproduces the vendored float-domain setting, so 8-bit rounding is off")
        quantised = False
    q = quantise if quantised else (lambda images, *budget: np.asarray(images, dtype=np.float32))
    if quantised and any(e < ONE_LEVEL - 1e-7 for e in epsilons):
        print("WARNING: budgets below 1/255 cannot be represented in 8 bits; their results are "
              "quantisation artefacts. Use them only with --no-quantise.")

    model = load_checkpoint(args.ckpt)
    print("loaded:", args.ckpt)

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
    print(f"images: {len(df)} from {df.true_idx.nunique()} classes | attacks: {attacks} | "
          f"epsilons: {epsilons} | defences: {defences} | xai: {xai_names} | impl: {args.impl} | "
          f"8-bit: {quantised} | seed: {args.seed}")

    rng = np.random.default_rng(args.seed)
    base_cols = [c for c in ["image_id", "true_idx", "true_subject", "eye", "device", "lighting", "phase"]
                 if c in df.columns]
    all_rows, examples = [], {}
    t0 = time.time()
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([load_image(resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        sides = [e if e in ("left", "right") else None for e in batch.get("eye", [None] * len(batch))]

        ev = Evaluator(model, xai_names, clean, labels, batch[base_cols].to_dict("records"),
                       concept_masks=args.concept_masks)
        ev.examples = {i: {} for i in range(len(batch)) if start + i < args.save_examples}

        pred = ev(clean, {"condition": "clean"})
        if (pred != labels).any():
            print(f"  WARNING: {(pred != labels).sum()} 'correct' images are misclassified now")

        for eps in epsilons:
            noise = eps * np.sign(rng.uniform(-1.0, 1.0, clean.shape)).astype(np.float32)
            ev(q(np.clip(clean + noise, 0.0, 1.0), clean, eps), {"condition": "noise", "epsilon": eps})
        for d in defences:
            ev(DEFENCE_REGISTRY[d](clean), {"condition": "defence_on_clean", "defence": d})

        for name in attacks:
            if name in BLACKBOX:
                runs = [(None, q(blackbox_attack(name, clean, sides)))]
            else:
                fn = registry[name]
                kwargs = {"label": labels} if takes(fn, "label") else {}
                if takes(fn, "seed"):
                    kwargs["seed"] = start
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
                adv_pred = ev(adv, info)
                success = {"attack_success": adv_pred != labels}
                for d in defences:
                    ev(DEFENCE_REGISTRY[d](adv), {**info, "condition": "attack+defence", "defence": d},
                       extra=success)

        all_rows += ev.rows
        examples.update({start + i: v for i, v in ev.examples.items()})
        done = start + len(batch)
        print(f"  {done}/{len(df)} images | {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = pd.DataFrame(all_rows)
    rates, table = summarise(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    rows.to_csv(os.path.join(args.out_dir, f"rows_{args.tag}.csv"), index=False)
    table.to_csv(os.path.join(args.out_dir, f"summary_{args.tag}.csv"), index=False)
    df[[c for c in ["image_id", "link_path", "true_idx", "true_subject", "eye", "device", "lighting", "phase"]
        if c in df.columns]].to_csv(os.path.join(args.out_dir, f"images_{args.tag}.csv"), index=False)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "impl": args.impl,
               "attack_loss": "margin (standard attacks)" if args.impl == "standard" else "vendored paper loss",
               "n_images": int(len(df)), "concept_masks": bool(args.concept_masks),
               "quantised_8bit": bool(quantised), "seed": args.seed, "exclude_from": args.exclude_from,
               "epsilons": epsilons, "attacks": attacks, "defences": defences, "xai": xai_names,
               "minutes": round((time.time() - t0) / 60, 1), **rates}
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if examples:
        arrays = {}
        for image_id, entries in examples.items():
            for key, (img, heat) in entries.items():
                arrays[f"{image_id}|{key}|image"] = (img * 255).round().astype(np.uint8)
                arrays[f"{image_id}|{key}|map"] = heat.astype(np.float16)
        np.savez_compressed(os.path.join(args.out_dir, f"examples_{args.tag}.npz"), **arrays)

    print(json.dumps(rates, indent=2))
    print("wrote results to", args.out_dir)


if __name__ == "__main__":
    main()