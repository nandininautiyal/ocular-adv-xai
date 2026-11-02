"""
scripts/22_anatomical_shapley.py

Anatomical Shapley Attribution (ASA, src/xai/shapley.py) of clean, noisy,
attacked and defended images: how much does each eye part (iris, sclera, lid,
brow, skin) contribute to the recognition score, and how do these
contributions move under attack and come back after a defence?

Run it on Kaggle next to the trained checkpoint, after 04_eval_clean.py has
written the correct-image list and 17_make_eye_masks.py the eye masks of the
same images (``--masks masks_<tag>.npz``; the run stops if a sampled image has
no mask). Images are sampled exactly like 05 and 14 (same --seed,
--exclude-from and --per-class give the same images).

Conditions per image (the conventions of 05 and 14):
  clean             the correctly classified 8-bit image
  noise             control: +-epsilon sign noise per budget, rounded to 8 bits
                    (the generator of 05: one numpy Generator seeded with --seed)
  attack            fgsm and pgd (src/attacks/whitebox.py, margin loss, the true
                    class), rounded to 8 bits within whole levels of each budget
                    (src/attacks/budget.py); with --attacks ...,bpda_pgd also
                    BPDA-PGD through each defence (src/attacks/adaptive.py, column
                    adapted_to), which returns 8-bit images
  defence_on_clean  each defence on the clean image
  noise+defence     each defence on each noise image (the matched control for
                    restoration)
  attack+defence    each defence on each fgsm / pgd image; BPDA-PGD images only
                    with the defence they were adapted to
Budgets are whole 8-bit levels (--levels 1,2,4 = 1/255, 2/255, 4/255, written to
7 decimals like 05). PGD and BPDA-PGD use --steps steps of 2.5 * epsilon / steps
and a random start seeded with --seed + the batch's first index (as in 14).

For every condition image the true class is explained, and when the prediction
is not the true class (any condition but clean) also the predicted, adversarial
class, as in 05. Per baseline (--baselines blur,mean,inpaint) the 2^P coalition
images go through the model once (absent parts are skipped: exact null players),
and the score (--score margin: the class's margin-free logit minus the best
other logit; or logit) gives ASA values, pairwise Shapley interactions and the
deletion AUC of several part orders, all from the same coalition values.
Hidden parts use the anatomy of the clean image (its eye mask) in every
condition: the perturbations are a few 8-bit levels and move no anatomy. For
defended conditions ASA explains the model on the defended image (the defence
is not applied again to the coalition images).

The inpaint game and full-coverage masks: with no pixel kept, inpainting has
nothing to fill from and the mean colour is used (src/xai/shapley.py). 17's masks
map the background to skin, so the default groups (iris, sclera + caruncle, lid,
brow, skin) cover every pixel of every image, and the empty coalition of the
inpaint game is then the mean-colour image in EVERY image: v_empty and the end of
every deletion curve of inpaint rows equal those of mean rows, and each inpaint
phi carries the jump from the mean image to an inpainted one. Read inpaint phi as
"the part's share of the drop from the image to the mean image, with the other
parts inpainted", and do not count mean ~ inpaint agreement as two independent
baselines (23 reports how often the two games share v_empty). Groups that leave a
class out (--groups) keep those pixels in every coalition and avoid this.

Other explanations of the same image, in part masses (share of each map's mass
on each part group, src/xai/shapley.part_masses):
  --cams layercam,eigencam   LayerCAM (the row's target class) and Eigen-CAM
                             (class-agnostic) at --target-layer (default: the last
                             4-D activation, block14_sepconv2_act for Xception)
  attention                  attention maps of an anatomy model (averaged over its
                             maps), when the model has an attention_pool layer
                             (03 --arch attn, attn_mask, parts)
  receipt                    the parts model's exact per-part decomposition of the
                             score of the full image, in the units of --score: for
                             logit, s x the part's term of the target's margin-free
                             cosine (src.models.anatomy_model.part_receipt); for
                             margin, the target's terms minus those of the best other
                             class. Matched to the groups by part name. The receipt
                             sums to v_full (receipt_sum; 23 checks it), while phi
                             sums to v_full - v_empty: the receipt splits the score
                             itself, ASA the score's drop when the parts are hidden.
Attention and receipt come from src.models.anatomy_model (attention_maps,
part_receipt, model_arch, part_names); models without an attention_pool layer
(baseline, gap) have no attention columns, models other than parts no receipt.

Columns of rows_asa_<tag>.csv (one row per image x condition x baseline x target):
  row_id, image_id, relpath, link_path, true_idx, true_subject, eye, device, lighting, phase
  condition, attack, adapted_to, epsilon, budget_levels, defence, defence_param
                       the variant (empty where it does not apply); adapted_to = the
                       defence a BPDA-PGD image was computed through
  baseline, score      hiding method and score function of the game
  target, target_idx   "true" (true class) or "pred" (the predicted class, rows only
                       where the prediction is wrong), and its class index
  pred_idx, correct    prediction for the condition image and whether it is the true class
  margin_true          true-class margin of the condition image (negative = fooled)
  attack_success       attack+defence rows: the attack reached its own goal. fgsm / pgd: it
                       fooled the undefended model (pred of the attack row != true);
                       bpda_pgd[D]: it fooled the model behind the defence D it was computed
                       through (pred of this row != true), as 14's success_after_defence
  attack_success_undefended  attack+defence rows: the attack fooled the undefended model
                       (equal to attack_success for fgsm / pgd)
  defended_clean_correct  rows with a defence: D(clean) is classified correctly
  pert_linf, pert_l2   distance of the condition image from the clean image ([0, 1] scale)
  area_<g>, present_<g>   share of the image in part group g, and whether it has a pixel
  v_full, v_empty      score with every part kept / every part hidden
  phi_<g>              ASA value of group g (sum over g = v_full - v_empty)
  int_<g>__<h>         Shapley interaction index of the pair (g, h)
  value_<k>            the coalition values (k = bitmask of kept groups, bit p = group p)
  mass_<src>_<g>       part masses of src = layercam, eigencam, attention
  receipt_<g>          the parts model's receipt (units of --score; NaN for a group the
                       model has no part for)
  receipt_sum          sum of the receipt over all of the model's parts (= v_full)
  del_auc_<src>        normalised deletion AUC (src/xai/shapley.deletion_from_values: the score
                       rescaled from 1 with every part kept to 0 with every present part
                       hidden; lower = it gets there sooner) when the present groups are hidden
                       in the order of src = asa (this row's phi, largest share of
                       v_full - v_empty first; it is computed on the same game, which favours
                       it), layercam, eigencam, attention, receipt (highest first), or random
                       (one seeded permutation per image, the same in every condition)
  n_forward            coalition images of this image and condition that went through the
                       model (all baselines)

Writes to --out-dir:
  rows_asa_<tag>.csv      the rows above
  images_asa_<tag>.csv    the sampled images (for --exclude-from of a later run)
  summary_asa_<tag>.json  settings, git commit and git_dirty, package versions, timing,
                          forward passes, row counts per condition, and the rates per variant
                          (23 checks the rows against them): attack_success_rate (on the
                          undefended model, every attack), attack_success_after_defence_rate
                          (every attack x defence; the headline number of a BPDA-PGD attack),
                          restoration_rate_among_successful_attacks (fgsm / pgd only: among the
                          attacks that fooled the undefended model, the share the defence
                          classifies correctly; for BPDA-PGD it is 1 - success after defence)
                          and the flip rates of the controls
  parts_asa_<tag>/        while the run is unfinished: the rows of every finished batch and
                          the state to continue from (--resume); removed at the end

Resume (--resume): rows are written after every batch of --batch-size images, with the noise
generator, rates, timings and counts, so a run stopped (e.g. at Kaggle's time limit) keeps its
finished batches; the SAME command with --resume continues after the last finished batch and
gives the rows of an uninterrupted run (the random starts are seeded per batch, the noise
generator state is restored). It stops if the settings or the sampled images differ from the
stopped run's, and does nothing if this command has already finished (its summary is there).
Without --resume a left-over parts_asa_<tag>/ is removed and the run starts from scratch.

Cost: with the defaults (levels 1,2,4, fgsm and pgd, jpeg and bit_depth) every image has
30 condition images, so ASA needs up to 30 x 3 baselines x 32 = 2,880 forward passes per
image (printed at start-up with the inpainting estimate). Inpainting runs on the CPU, about
60 ms per coalition image and thread (30 x 31 per image; --workers threads); on the laptop
CPU with the tiny test model it was most of the run time. --baselines blur,mean skips it.

Usage (Kaggle, with the checkpoint dataset attached and 01, 02, 04, 17 already run):
  smoke (a few minutes):
    python scripts/22_anatomical_shapley.py --ckpt <checkpoint.keras> \
        --correct-csv /kaggle/working/eval_clean/correct_test_full.csv \
        --masks /kaggle/working/eye_masks/masks_full.npz --max-images 4 --tag smoke
  full (one image per class, 300 images; check the time estimate of the smoke run first;
  --resume continues a stopped run when the same command is run again):
    python scripts/22_anatomical_shapley.py --ckpt <checkpoint.keras> \
        --correct-csv /kaggle/working/eval_clean/correct_test_full.csv \
        --masks /kaggle/working/eye_masks/masks_full.npz --per-class 1 --seed 2 \
        --exclude-from /kaggle/working/asa/images_asa_smoke.csv --resume --tag full
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.attacks.adaptive import bpda_pgd  # noqa: E402
from src.attacks.budget import levels, quantise  # noqa: E402
from src.attacks.whitebox import fgsm, pgd  # noqa: E402
from src.defences.inference import DEFENCE_REGISTRY, make_defence  # noqa: E402
from src.models.arcface_inference import find_feature_layer, load_checkpoint, predict_logits  # noqa: E402
from src.xai.attribution import XAI_REGISTRY  # noqa: E402
from src.xai.seg_masks import load_mask_file  # noqa: E402
from src.xai.shapley import (DEFAULT_GROUPS, DEFAULT_WORKERS, METHODS, asa_order, coalition_outputs,  # noqa: E402
                             deletion_from_values, exact_shapley, format_groups, group_masks, groups_from_config,
                             normalise_groups, order_from_scores, parse_groups, part_masses, score_values,
                             shapley_interactions)

from src.models.anatomy_model import attention_maps, model_arch, part_names, part_receipt  # noqa: E402


def _load_script(name):
    """A numbered script as a module (its main() does not run)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_script("05_run_experiment")          # budgets, image loading, git state, versions
_ADAPTIVE = _load_script("14_adaptive_attacks")      # sampling, margins, JSON helpers

ATTACKS = ("fgsm", "pgd", "bpda_pgd")
SCORES = ("margin", "logit")
SOURCES = ("asa", "layercam", "eigencam", "attention", "receipt", "random")
BASE_COLS = ["image_id", "relpath", "link_path", "true_idx", "true_subject", "eye", "device", "lighting", "phase"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def resolve_groups(text):
    """Part groups from --groups: None (the default groups), a YAML config with
    anatomy.part_groups, or a spec such as ``iris=iris;sclera=sclera+caruncle;skin=skin``."""
    if text is None:
        return normalise_groups(DEFAULT_GROUPS)
    if os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            return groups_from_config(yaml.safe_load(f))
    return parse_groups(text)


def variant_name(condition, attack=None, adapted_to=None, epsilon=None, defence=None):
    """Name of a condition variant in the summary rates, like 05's: 'pgd@0.0039216+jpeg',
    'bpda_pgd[jpeg]@0.0078431+jpeg', 'noise@0.0039216', 'clean+jpeg'."""
    def given(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x)) and x != ""
    name = attack if given(attack) else ("noise" if str(condition).startswith("noise") else "clean")
    if given(adapted_to):
        name += f"[{adapted_to}]"
    if given(epsilon):
        name += f"@{float(epsilon):.6g}"
    if given(defence):
        name += f"+{defence}"
    return name


RATE_SECTIONS = {"clean": "clean_flip_rate", "noise": "noise_control_flip_rate",
                 "defence_on_clean": "defence_on_clean_flip_rate", "noise+defence": "noise_defence_flip_rate",
                 "attack": "attack_success_rate"}
DEFENDED_SECTIONS = ("attack_success_after_defence_rate", "restoration_rate_among_successful_attacks")
RESUME_FREE = ("resume", "workers")      # arguments that do not change the rows (a resumed run may change them)


class Rates:
    """Flip and success counts per variant, collected while the rows are made (23 recomputes
    them from the rows and compares)."""

    def __init__(self):
        self.counts = defaultdict(lambda: [0, 0])          # (section, name) -> [hits, n]

    def add(self, section, name, hits):
        hits = np.asarray(hits, bool)
        if not hits.size:            # e.g. no successful attack to restore in this batch
            return
        self.counts[(section, name)][0] += int(hits.sum())
        self.counts[(section, name)][1] += int(hits.size)

    def to_list(self):
        """[[section, name, hits, n], ...]: the counts as JSON (the resume state)."""
        return [[s, name, k, n] for (s, name), (k, n) in sorted(self.counts.items())]

    @classmethod
    def from_list(cls, items):
        rates = cls()
        for section, name, k, n in items:
            rates.counts[(section, name)] = [int(k), int(n)]
        return rates

    def as_dict(self):
        out = {s: {} for s in list(RATE_SECTIONS.values()) + list(DEFENDED_SECTIONS)}
        for (section, name), (k, n) in sorted(self.counts.items()):
            low, high = _ADAPTIVE.wilson(k, n)
            out[section][name] = {"rate": k / n if n else float("nan"), "n": n, "ci_low": low, "ci_high": high}
        return out


# ---------------------------------------------------------------------------
# Explanations other than ASA
# ---------------------------------------------------------------------------

def anatomy_capabilities(model):
    """What the model offers beyond CAMs: its arch, attention maps, the parts receipt."""
    arch = model_arch(model)
    attention = any(layer.name == "attention_pool" for layer in model.layers)
    receipt = arch == "parts"
    return {"arch": arch, "attention": attention, "receipt": receipt,
            "receipt_parts": [str(p) for p in part_names(model)] if receipt else None,
            "receipt_units": "score (--score): sums to v_full" if receipt else None}


def receipt_terms(model, images, class_idx, logits, score):
    """Parts receipt (N, P_model) in the units of the ASA score, summing to the score of each image.

    logit: s x the per-part terms of the target's margin-free cosine (they sum to the target's
    margin-free logit). margin: the target's terms minus those of the best other class of
    *logits* (the margin is a difference of two logits, each additive over the parts with the
    same ||e||, so the difference is additive too).
    """
    class_idx = np.asarray(class_idx, np.int64)
    terms = np.asarray(part_receipt(model, images, class_idx, units="logit"), np.float64)
    if score == "logit":
        return terms
    others = np.array(logits, np.float64, copy=True)
    others[np.arange(len(others)), class_idx] = -np.inf
    return terms - np.asarray(part_receipt(model, images, others.argmax(axis=1), units="logit"), np.float64)


def cam_maps(model, xai, images, class_idx, target_layer):
    """Native maps of one XAI method; Eigen-CAM ignores the class."""
    fn = XAI_REGISTRY[xai]
    kwargs = {"class_idx": class_idx} if "class_idx" in inspect.signature(fn).parameters else {}
    if target_layer is not None:
        kwargs["target_layer"] = target_layer
    return fn(model, images, size=None, **kwargs)


def class_specific(xai):
    return "class_idx" in inspect.signature(XAI_REGISTRY[xai]).parameters


def explanations(model, images, targets, cams, target_layer, capabilities, receipt_index, logits=None,
                 score="margin"):
    """Per target ("true", "pred"): source -> per-image maps (CAMs, attention) or part vectors
    (receipt, already in group order, plus "receipt_sum" over all of the model's parts).

    targets : dict target -> class indices (N,)
    receipt_index : for each group, its column in the model's receipt (None if the model has
        no such part)
    logits, score : margin-free logits of *images* and the ASA score (units of the receipt)
    """
    out = {t: {} for t in targets}
    shared = {}
    for xai in cams:
        if not class_specific(xai):
            shared[xai] = cam_maps(model, xai, images, None, target_layer)
    if capabilities["attention"]:
        maps = np.asarray(attention_maps(model, images), np.float32)       # (N, 8, 8, K)
        shared["attention"] = maps.mean(axis=-1)
    for target, class_idx in targets.items():
        out[target].update(shared)
        for xai in cams:
            if class_specific(xai):
                out[target][xai] = cam_maps(model, xai, images, class_idx, target_layer)
        if capabilities["receipt"]:
            if logits is None:
                logits = predict_logits(model, images)
            receipt = receipt_terms(model, images, class_idx, logits, score)                # (N, P_model)
            out[target]["receipt_sum"] = receipt.sum(axis=1)
            out[target]["receipt"] = np.stack(
                [receipt[:, j] if j is not None else np.full(len(images), np.nan) for j in receipt_index], axis=1)
    return out


# ---------------------------------------------------------------------------
# Rows of one condition
# ---------------------------------------------------------------------------

class Context:
    """Everything about a batch of images that stays the same across conditions."""

    def __init__(self, model, clean, labels, base_rows, mask_labels, class_names, groups, baselines, score,
                 cams, target_layer, capabilities, receipt_index, asa_batch_size, seed, workers=1):
        self.model, self.clean, self.labels, self.base_rows = model, clean, labels, base_rows
        self.groups, self.names = groups, [name for name, _ in groups]
        self.baselines, self.score, self.cams, self.target_layer = baselines, score, cams, target_layer
        self.capabilities, self.receipt_index, self.asa_batch_size = capabilities, receipt_index, asa_batch_size
        self.workers = workers
        self.masks = [group_masks(lab, class_names, groups) for lab in mask_labels]
        self.mask_labels, self.class_names = mask_labels, class_names
        self.present = [m.reshape(len(m), -1).any(axis=1) for m in self.masks]
        self.areas = [m.reshape(len(m), -1).mean(axis=1) for m in self.masks]
        self.random_order = []
        for row, present in zip(base_rows, self.present):        # one seeded order per image, every condition
            rng = np.random.default_rng([seed, int(row["image_id"])])
            self.random_order.append([int(p) for p in rng.permutation(np.flatnonzero(present))])
        self.n_forward = 0


def condition_rows(ctx, images, info, extra=None, timings=None):
    """Rows of one condition: every image x baseline x target. Returns (rows, predictions).

    extra : dict column -> per-image values, or a function of the predictions (N,) returning one
    """
    timings = {} if timings is None else timings
    start = time.perf_counter()
    logits = predict_logits(ctx.model, images)
    pred = logits.argmax(1)
    extra = (extra(pred) if callable(extra) else extra) or {}
    margin_true = _ADAPTIVE.margins(logits, ctx.labels)
    flipped = pred != ctx.labels
    targets = {"true": ctx.labels}
    if info["condition"] != "clean" and flipped.any():
        targets["pred"] = pred
    expl = explanations(ctx.model, images, targets, ctx.cams, ctx.target_layer, ctx.capabilities, ctx.receipt_index,
                        logits, ctx.score)
    timings["explanations"] = timings.get("explanations", 0.0) + time.perf_counter() - start

    diff = (images - ctx.clean).reshape(len(images), -1)
    linf, l2 = np.abs(diff).max(axis=1), np.sqrt((diff ** 2).sum(axis=1))
    names, rows = ctx.names, []
    for i in range(len(images)):
        start = time.perf_counter()
        outputs, present, n_forward = coalition_outputs(ctx.model, images[i], ctx.masks[i], ctx.baselines,
                                                        "logits", ctx.asa_batch_size, workers=ctx.workers)
        timings["asa"] = timings.get("asa", 0.0) + time.perf_counter() - start
        ctx.n_forward += n_forward
        common = {**ctx.base_rows[i], **info, "pred_idx": int(pred[i]), "correct": bool(not flipped[i]),
                  "margin_true": float(margin_true[i]), "pert_linf": float(linf[i]), "pert_l2": float(l2[i])}
        common.update({k: v[i] for k, v in extra.items()})
        common.update({f"area_{g}": float(a) for g, a in zip(names, ctx.areas[i])})
        common.update({f"present_{g}": bool(p) for g, p in zip(names, present)})
        for target, class_idx in targets.items():
            if target == "pred" and not flipped[i]:
                continue
            masses, scores = {}, {}
            for src, values in expl[target].items():
                if src == "receipt_sum":
                    masses["receipt_sum"] = float(values[i])
                elif src == "receipt":
                    scores[src] = values[i]
                    masses.update({f"receipt_{g}": float(v) for g, v in zip(names, values[i])})
                else:
                    m = part_masses(values[i], None, ctx.class_names, ctx.groups, masks=ctx.masks[i])
                    scores[src] = m
                    masses.update({f"mass_{src}_{g}": float(v) for g, v in zip(names, m)})
            for baseline in ctx.baselines:
                values = score_values(outputs[baseline], ctx.score, target=int(class_idx[i]))
                phi, inter = exact_shapley(values), shapley_interactions(values)
                row = {**common, "baseline": baseline, "score": ctx.score, "target": target,
                       "target_idx": int(class_idx[i]), "v_full": float(values[-1]), "v_empty": float(values[0])}
                row.update({f"phi_{g}": float(v) for g, v in zip(names, phi)})
                row.update({f"int_{names[p]}__{names[q]}": float(inter[p, q])
                            for p, q in itertools.combinations(range(len(names)), 2)})
                row.update(masses)
                orders = {"asa": asa_order(phi, values[-1], values[0], present), "random": ctx.random_order[i]}
                orders.update({src: order_from_scores(s, present) for src, s in scores.items()})
                for src in SOURCES:
                    if src in orders:
                        order = orders[src]
                        row[f"del_auc_{src}"] = (float("nan") if order is None
                                                 else deletion_from_values(values, order)["auc"])
                row.update({f"value_{k}": float(v) for k, v in enumerate(values)})
                row["n_forward"] = n_forward
                rows.append(row)
    return rows, pred


# ---------------------------------------------------------------------------
# One batch: every condition
# ---------------------------------------------------------------------------

def run_batch(ctx, epsilons, defences, attacks, steps, seed, rng, rates, timings):
    """All conditions of one batch of clean images; returns the rows.

    defences : dict name -> (defence function, parameter value)
    rng : numpy Generator for the noise control (one per run, as in 05)
    """
    clean, labels = ctx.clean, ctx.labels
    rows = []

    def add(images, info, extra=None):
        new, pred = condition_rows(ctx, images, info, extra, timings)
        rows.extend(new)
        return pred

    def info(condition, attack=None, adapted_to=None, eps=None, defence=None):
        param = defences[defence][1] if defence is not None else None
        return {"condition": condition, "attack": attack, "adapted_to": adapted_to, "epsilon": eps,
                "budget_levels": levels(eps) if eps is not None else None, "defence": defence,
                "defence_param": param}

    pred = add(clean, info("clean"))
    rates.add(RATE_SECTIONS["clean"], "clean", pred != labels)
    defended_clean_correct = {}
    for d, (defend, _) in defences.items():
        pred = add(defend(clean), info("defence_on_clean", defence=d))
        defended_clean_correct[d] = pred == labels
        rates.add(RATE_SECTIONS["defence_on_clean"], variant_name("defence_on_clean", defence=d), pred != labels)

    noisy = {}
    for eps in epsilons:            # the noise of 05: sign noise from one Generator, rounded within the budget
        noise = eps * np.sign(rng.uniform(-1.0, 1.0, clean.shape)).astype(np.float32)
        noisy[eps] = quantise(np.clip(clean + noise, 0.0, 1.0), clean, eps)
        pred = add(noisy[eps], info("noise", eps=eps))
        rates.add(RATE_SECTIONS["noise"], variant_name("noise", epsilon=eps), pred != labels)
        for d, (defend, _) in defences.items():
            pred = add(defend(noisy[eps]), info("noise+defence", eps=eps, defence=d),
                       {"defended_clean_correct": defended_clean_correct[d]})
            rates.add(RATE_SECTIONS["noise+defence"], variant_name("noise+defence", epsilon=eps, defence=d),
                      pred != labels)

    for eps in epsilons:
        runs = []                                   # (attack, adapted_to, images, defences to apply)
        common = {"label": labels}
        for name in ("fgsm", "pgd"):
            if name not in attacks:
                continue
            start = time.perf_counter()
            if name == "fgsm":
                adv = fgsm(clean, ctx.model, eps, **common)
            else:
                adv = pgd(clean, ctx.model, eps, steps=steps, seed=seed, **common)
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
            runs.append((name, None, quantise(adv, clean, eps), list(defences)))
        if "bpda_pgd" in attacks:
            for d, (defend, _) in defences.items():
                start = time.perf_counter()
                adv = bpda_pgd(clean, ctx.model, eps, defend, steps=steps, seed=seed, quantise_8bit=True, **common)
                timings["bpda_pgd"] = timings.get("bpda_pgd", 0.0) + time.perf_counter() - start
                runs.append(("bpda_pgd", d, adv, [d]))
        for name, adapted_to, adv, apply in runs:
            pred = add(adv, info("attack", name, adapted_to, eps))
            undefended = pred != labels
            rates.add(RATE_SECTIONS["attack"], variant_name("attack", name, adapted_to, eps), undefended)
            adaptive = adapted_to is not None
            for d in apply:
                def extra(defended_pred, undefended=undefended, d=d, adaptive=adaptive):
                    # BPDA-PGD is computed through d to fool f(d(x)): its success is judged after d (as in 14)
                    return {"attack_success": (defended_pred != labels) if adaptive else undefended,
                            "attack_success_undefended": undefended,
                            "defended_clean_correct": defended_clean_correct[d]}
                pred = add(defences[d][0](adv), info("attack+defence", name, adapted_to, eps, d), extra)
                variant = variant_name("attack+defence", name, adapted_to, eps, d)
                rates.add("attack_success_after_defence_rate", variant, pred != labels)
                if not adaptive:            # restoration of the label among the attacks that fooled the model
                    rates.add("restoration_rate_among_successful_attacks", variant, (pred == labels)[undefended])
    return rows


def conditions_per_image(n_eps, n_def, attacks):
    """Condition images per clean image (each explained with every baseline)."""
    n = 1 + n_def + n_eps * (1 + n_def)                       # clean, defence_on_clean, noise, noise+defence
    plain = sum(a in attacks for a in ("fgsm", "pgd"))
    n += n_eps * plain * (1 + n_def)                          # attack, attack+defence
    if "bpda_pgd" in attacks:
        n += n_eps * n_def * 2
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split_list(text):
    return [part.strip() for part in str(text).split(",") if part.strip() and part.strip() != "none"]


def write_state(path, state):
    """The resume state as JSON, replaced atomically."""
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(path + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained .keras checkpoint (dual input [image, label])")
    ap.add_argument("--correct-csv", required=True, help="correct_<split>_<tag>.csv from 04_eval_clean.py")
    ap.add_argument("--masks", required=True,
                    help="masks_<tag>.npz from 17_make_eye_masks.py with a mask for every sampled image")
    ap.add_argument("--groups", default=None,
                    help="part groups: a YAML config with anatomy.part_groups, or a spec such as "
                         f"'{format_groups(DEFAULT_GROUPS)}' (the default)")
    ap.add_argument("--data-root", default=None, help="VISOB root, used if link_path no longer exists")
    ap.add_argument("--per-class", type=int, default=1, help="images per class (1 = 300 images)")
    ap.add_argument("--max-images", type=int, default=None, help="smoke run: cap the number of images (e.g. 4)")
    ap.add_argument("--exclude-from", default=None,
                    help="CSV with a link_path column (images_<tag>.csv of 05, images_asa_<tag>.csv); "
                         "those images are not used")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling and noise seed (as in 05); PGD random starts use seed + the batch's first index")
    ap.add_argument("--levels", default="1,2,4", help="L-inf budgets in whole 8-bit levels (1,2,4 = 1/255, ...)")
    ap.add_argument("--attacks", default="fgsm,pgd",
                    help="fgsm, pgd (non-adaptive, 8-bit) and optionally bpda_pgd (adaptive, through each defence)")
    ap.add_argument("--steps", type=int, default=10,
                    help="steps of pgd and bpda_pgd (10 = whitebox.pgd's default, as in 05; 14 used 20)")
    ap.add_argument("--defences", default="jpeg,bit_depth", help="jpeg and/or bit_depth ('none': no defence)")
    ap.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality of the jpeg defence")
    ap.add_argument("--bits", type=int, default=4, help="bit depth of the bit_depth defence")
    ap.add_argument("--baselines", default="blur,mean,inpaint", help="hiding methods (src/xai/shapley.py)")
    ap.add_argument("--score", default="margin", choices=SCORES,
                    help="margin (target logit minus the best other margin-free logit) or logit")
    ap.add_argument("--cams", default="layercam,eigencam", help="CAMs for part masses ('none' skips them)")
    ap.add_argument("--target-layer", default=None,
                    help="layer the CAMs explain (default: the last 4-D activation, block14_sepconv2_act)")
    ap.add_argument("--batch-size", type=int, default=16, help="images attacked together")
    ap.add_argument("--asa-batch-size", type=int, default=96, help="coalition images per forward pass")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="CPU threads for inpainting, the slowest part on CPU (about 60 ms per coalition image "
                         "and core); results do not depend on it")
    ap.add_argument("--out-dir", default="/kaggle/working/asa")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--resume", action="store_true",
                    help="continue a stopped run of the same command after its last finished batch "
                         "(parts_asa_<tag>/ in --out-dir); nothing to do if it has finished")
    args = ap.parse_args(argv)

    # --- settings, checked before anything slow is loaded --------------------------------
    attacks, defence_names = split_list(args.attacks), split_list(args.defences)
    baselines, cams = split_list(args.baselines), split_list(args.cams)
    for name in attacks:
        if name not in ATTACKS:
            ap.error(f"unknown attack {name!r}; choose from {list(ATTACKS)}")
    for name in baselines:
        if name not in METHODS:
            ap.error(f"unknown baseline {name!r}; choose from {list(METHODS)}")
    for name in cams:
        if name not in XAI_REGISTRY:
            ap.error(f"unknown CAM {name!r}; choose from {sorted(XAI_REGISTRY)}")
    if not attacks or not baselines:
        ap.error("need at least one attack and one baseline")
    if "bpda_pgd" in attacks and not defence_names:
        ap.error("bpda_pgd attacks through a defence: give --defences")
    if min(args.steps, args.batch_size, args.asa_batch_size, args.per_class, args.workers) < 1:
        ap.error("--steps, --batch-size, --asa-batch-size, --per-class and --workers must be at least 1")
    parameters = {"jpeg": {"quality": args.jpeg_quality}, "bit_depth": {"bits": args.bits}}
    defences = {}
    for name in defence_names:
        try:
            fn = make_defence(name, **parameters.get(name, {}))
        except KeyError:
            ap.error(f"unknown defence {name!r}; choose from {sorted(DEFENCE_REGISTRY)}")
        except (TypeError, ValueError) as err:
            ap.error(str(err))
        defences[name] = (fn, next(iter(parameters[name].values())))
    epsilons = _RUNNER.parse_budgets(ap, None, args.levels, True)
    try:
        groups = resolve_groups(args.groups)
    except (ValueError, OSError) as err:
        ap.error(f"--groups: {err}")
    names = [name for name, _ in groups]
    if not os.path.isfile(args.masks):
        ap.error(f"--masks {args.masks}: no such file")

    # --- images and masks ------------------------------------------------------------
    df = pd.read_csv(args.correct_csv)
    if args.exclude_from:
        seen = set(pd.read_csv(args.exclude_from, usecols=["link_path"]).link_path)
        before = len(df)
        df = df[~df.link_path.isin(seen)]
        print(f"excluded {before - len(df)} images listed in {args.exclude_from}")
    if df.empty:
        raise SystemExit("no images left to sample (check --correct-csv and --exclude-from)")
    df = _ADAPTIVE.sample_images(df, args.per_class, args.seed, args.max_images)
    if "relpath" not in df.columns or not df.relpath.map(lambda r: isinstance(r, str)).all():
        raise SystemExit("--masks are looked up by relpath, but --correct-csv has no relpath for some images "
                         "(04_eval_clean.py writes a relpath column)")
    df["relpath"] = df.relpath.map(lambda r: r.replace("\\", "/"))
    try:
        mask_file = load_mask_file(args.masks)
    except ValueError as err:
        raise SystemExit(f"--masks: {err}")
    missing = [r for r in df.relpath if r not in mask_file["index"]]
    if missing:
        raise SystemExit(f"--masks {args.masks} has no mask for {len(missing)} of the {len(df)} sampled images "
                         f"(make masks for them with 17):\n  " + "\n  ".join(missing[:20]))
    class_names = mask_file["class_names"]
    try:
        group_masks(np.zeros((2, 2), np.uint8), class_names, groups)       # every group class is in the file
    except ValueError as err:
        raise SystemExit(f"--groups do not fit the mask file: {err}")
    mask_labels = mask_file["labels"][[mask_file["index"][r] for r in df.relpath]]

    # --- resume: the same settings and images as the stopped run, checked before the model loads ---------
    identity = {"args": {k: v for k, v in vars(args).items() if k not in RESUME_FREE},
                "images": df.link_path.astype(str).tolist()}
    resume_key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    parts_dir = os.path.join(args.out_dir, f"parts_asa_{args.tag}")
    state_path = os.path.join(parts_dir, "state.json")
    summary_path = os.path.join(args.out_dir, f"summary_asa_{args.tag}.json")
    state = None
    if args.resume and os.path.exists(summary_path) and not os.path.isdir(parts_dir):
        try:
            with open(summary_path, encoding="utf-8") as f:
                finished = json.load(f).get("resume_key") == resume_key
        except ValueError:
            finished = False
        if finished:
            print(f"{summary_path} says this command has finished run '{args.tag}': nothing to resume "
                  "(use another --tag or --out-dir to run it again)")
            return
    if os.path.isdir(parts_dir):
        if args.resume and os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("resume_key") != resume_key:
                raise SystemExit(f"{parts_dir} holds a stopped run with other settings or other sampled images; "
                                 f"run its own command again, or use another --tag (or remove that folder)")
        else:
            print(f"{'no resume state in' if args.resume else 'WARNING: without --resume,'} {parts_dir}: "
                  "removed, starting from scratch")
            shutil.rmtree(parts_dir)
    start_at = int(state["n_done"]) if state else 0

    n_conditions = conditions_per_image(len(epsilons), len(defences), attacks)
    upper = n_conditions * len(baselines) * 2 ** len(names)
    print(f"images: {len(df)} from {df.true_idx.nunique()} classes | attacks: {attacks} | levels: "
          f"{[levels(e) for e in epsilons]} | defences: {[defences[d][0].label for d in defences]} | baselines: "
          f"{baselines} | score: {args.score} | groups: {format_groups(groups)} | cams: {cams} | seed: {args.seed}")
    print(f"forward passes per image for ASA: at most {n_conditions} condition images x {len(baselines)} baselines "
          f"x 2^{len(names)} coalitions = {upper} (absent parts need fewer), plus the attacks and CAMs")
    if "inpaint" in baselines:
        seconds = n_conditions * (2 ** len(names) - 1) * 0.06 / args.workers
        print(f"inpainting: about {seconds:.0f} s of CPU per image with {args.workers} threads "
              f"(--baselines blur,mean skips it)")

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)
    model = load_checkpoint(args.ckpt)          # after the CSV and mask checks: the model takes minutes to load
    print("loaded:", args.ckpt)
    if len(model.inputs) != 2:
        raise SystemExit(f"{args.ckpt} has {len(model.inputs)} inputs; 22 explains dual-input [image, label] "
                         f"models (an attnet_gate model also needs its region input)")
    if args.target_layer is not None:
        try:
            model.get_layer(args.target_layer)
        except ValueError:
            raise SystemExit(f"--target-layer {args.target_layer!r} is not a layer of {args.ckpt}") from None
    target_layer = args.target_layer or (find_feature_layer(model) if cams else None)
    capabilities = anatomy_capabilities(model)
    receipt_index = [None] * len(names)
    if capabilities["receipt"]:
        parts = capabilities["receipt_parts"]
        receipt_index = [parts.index(g) if g in parts else None for g in names]
        if set(parts) != set(names):
            print(f"note: the model's parts {parts} differ from the groups {names}; receipt columns of groups "
                  f"without a part are NaN")
    print(f"model: arch {capabilities['arch']} | attention maps: {capabilities['attention']} | parts receipt: "
          f"{capabilities['receipt']} | CAM layer: {target_layer}")

    # --- experiment ------------------------------------------------------------------
    base_cols = [c for c in BASE_COLS if c in df.columns]
    rng = np.random.default_rng(args.seed)
    rates, timings, n_forward, minutes_before = Rates(), {}, 0, 0.0
    os.makedirs(parts_dir, exist_ok=True)
    if state:                       # continue after the last finished batch, with its noise stream and counts
        rng.bit_generator.state = state["rng"]
        rates = Rates.from_list(state["rates"])
        timings, n_forward, minutes_before = dict(state["seconds"]), int(state["n_forward"]), float(state["minutes"])
        for name in os.listdir(parts_dir):          # a batch written after the last state was saved runs again
            if name.startswith("rows_") and name.endswith(".csv") and int(name[5:-4]) >= start_at:
                os.remove(os.path.join(parts_dir, name))
        print(f"resuming after {start_at} of {len(df)} images ({minutes_before:.1f} min before)")
    t0 = time.time()
    for start in range(start_at, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([_RUNNER.load_image(_RUNNER.resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        ctx = Context(model, clean, labels, batch[base_cols].to_dict("records"),
                      mask_labels[start:start + len(batch)], class_names, groups, baselines, args.score, cams,
                      args.target_layer, capabilities, receipt_index, args.asa_batch_size, args.seed,
                      args.workers)
        rows = run_batch(ctx, epsilons, defences, attacks, args.steps, args.seed + start, rng, rates, timings)
        n_forward += ctx.n_forward
        done = start + len(batch)
        minutes = (time.time() - t0) / 60
        # the batch's rows, then the state that points past them (both replaced atomically): a stop in
        # between only makes a resumed run compute this batch again
        part = os.path.join(parts_dir, f"rows_{start:06d}.csv")
        pd.DataFrame(rows).to_csv(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        write_state(state_path, {"resume_key": resume_key, "n_done": done, "n_images": int(len(df)),
                                 "rng": rng.bit_generator.state, "rates": rates.to_list(),
                                 "seconds": timings, "n_forward": int(n_forward),
                                 "minutes": minutes_before + minutes})
        print(f"  {done}/{len(df)} images | {minutes:.1f} min | ETA "
              f"{minutes / (done - start_at) * (len(df) - done):.1f} min | "
              f"{ctx.n_forward / len(batch):.0f} ASA forward passes per image", flush=True)

    # --- outputs ---------------------------------------------------------------------
    # the rows of every batch, read back from the parts (the same path for a resumed and an uninterrupted run)
    parts = sorted(name for name in os.listdir(parts_dir) if name.startswith("rows_") and name.endswith(".csv"))
    rows = pd.concat([pd.read_csv(os.path.join(parts_dir, name)) for name in parts], ignore_index=True, sort=False)
    rows.insert(0, "row_id", np.arange(len(rows)))
    first = ["row_id"] + base_cols
    rows = rows[first + [c for c in rows.columns if c not in first]]
    rows.to_csv(os.path.join(args.out_dir, f"rows_asa_{args.tag}.csv"), index=False)
    df[[c for c in BASE_COLS if c in df.columns]].to_csv(os.path.join(args.out_dir, f"images_asa_{args.tag}.csv"),
                                                         index=False)
    commit, dirty = _RUNNER.git_state()
    present = np.stack([m.reshape(len(m), -1).any(axis=1)
                        for m in (group_masks(lab, class_names, groups) for lab in mask_labels)])
    summary = {"tag": args.tag, "ckpt": args.ckpt, "git_commit": commit, "git_dirty": dirty,
               "versions": _RUNNER.package_versions(), "args": vars(args),
               "n_images": int(len(df)), "n_classes": int(df.true_idx.nunique()), "n_rows": int(len(rows)),
               "rows_per_condition": {k: int(v) for k, v in Counter(rows.condition).items()},
               "masks": args.masks, "mask_file_sha256": _RUNNER.file_sha256(args.masks), "mask_meta": mask_file["meta"],
               "class_names": class_names, "groups": {name: list(c) for name, c in groups},
               "groups_spec": format_groups(groups),
               "group_presence": {g: int(present[:, p].sum()) for p, g in enumerate(names)},
               "epsilons": epsilons, "levels": [levels(e) for e in epsilons], "quantised_8bit": True,
               "attacks": attacks, "attack_loss": "margin (best other logit - true logit, margin-free logits)",
               "steps": args.steps, "step_size": "2.5 * epsilon / steps (pgd, bpda_pgd)",
               "random_start": "uniform in the epsilon ball, seed + first index of the batch",
               "noise": "epsilon * sign(uniform(-1, 1)) from numpy default_rng(seed), rounded to 8 bits (as in 05)",
               "defences": {d: {"quality": p} if d == "jpeg" else {"bits": p} for d, (_, p) in defences.items()},
               "baselines": baselines, "score": args.score, "cams": cams, "cam_layer": target_layer,
               "anatomy": capabilities, "targets": "true class; predicted class where it differs (not clean)",
               "conditions_per_image": n_conditions,
               "asa_forward_passes": {"total": int(n_forward), "per_image": n_forward / max(len(df), 1),
                                      "upper_bound_per_image": upper},
               "seconds": {k: round(v, 1) for k, v in timings.items()},
               "minutes": round(minutes_before + (time.time() - t0) / 60, 2),
               "resumed": state is not None, "resumed_after_images": start_at if state else None,
               "resume_key": resume_key,
               **rates.as_dict()}
    with open(summary_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(_ADAPTIVE._json_safe(summary), f, indent=2)
    os.replace(summary_path + ".tmp", summary_path)
    shutil.rmtree(parts_dir, ignore_errors=True)         # finished: the summary says so from now on

    for section, title in (("attack_success_rate", "attack success on the undefended model"),
                           ("attack_success_after_defence_rate", "attack success after the defence (BPDA-PGD: "
                                                                 "the defence it was computed through)")):
        print(f"{title} (95 % Wilson interval):")
        for name, r in summary[section].items():
            print(f"  {name:<40}{r['rate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n={r['n']}")
    print(f"{len(rows):,} rows | {n_forward / max(len(df), 1):.0f} ASA forward passes per image | "
          f"{summary['minutes']} min | wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
