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
                    (src/attacks/budget.py); with --attacks ...,aas_pgd also the
                    attention-scaled PGD of src/attacks/masking.py (non-adaptive
                    like pgd; --taus; on a model without attention it is pgd); with
                    --attacks ...,bpda_pgd also BPDA-PGD through each defence
                    (src/attacks/adaptive.py, column adapted_to), which returns
                    8-bit images
  defence_on_clean  each defence on the clean image
  noise+defence     each defence on each noise image (the matched control for
                    restoration)
  attack+defence    each defence on each fgsm / pgd / aas_pgd image; BPDA-PGD images only
                    with the defence they were adapted to
Budgets are whole 8-bit levels (--levels 1,2,4 = 1/255, 2/255, 4/255, written to
7 decimals like 05). PGD, AAS-PGD and BPDA-PGD use --steps steps of 2.5 * epsilon / steps
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

Sampling baselines (src/xai/part_baselines.py; numpy only, no shap or lime package):
  --part-baselines kernelshap,lime   KernelSHAP (Shapley kernel, shap's enumeration of complete
                             sizes, the rest sampled without replacement, efficiency enforced as in
                             shap) and LIME (uniform samples, kernel sqrt(exp(-d^2 / 0.25^2)) of the
                             cosine distance to the full coalition, ridge alpha 1) on the SAME part
                             players, hiding baselines (--baseline-hiding, default all of --baselines)
                             and score as ASA, with half of the 2^P coalitions of the image's present
                             parts, at most --baseline-samples (default 16 = half of the 32 when all five
                             parts are present; src/xai/part_baselines.part_budget). A fixed 16 would be
                             all 2^4 coalitions of the images without a brow, where KernelSHAP is exact
                             and its error 0 by construction. KernelSHAP's budget includes the empty and
                             the full coalition.
                             They run stand-alone (their own coalition images and forward passes), so
                             their run time is measured, and their error against exact ASA on the same
                             game is a result (23: sampling_error.csv).
  --pixel-baselines kernelshap,lime  the same estimators on square patches (--patch-size 32 = 64
                             players, --pixel-samples 500), hidden with --pixel-hiding (default blur),
                             on the conditions of --pixel-conditions (default clean; each condition costs
                             2 x --pixel-samples forward passes per image), aggregated to the parts:
                             part p = sum over patches j of w_j x (share of patch j's pixels in part p).
  The coalitions are seeded per image (default_rng([--seed, image_id, 1..4])): every condition,
  target and hiding baseline of an image uses the same ones.

Verification pairs (--verif-pairs): the conditions of --verif-conditions (default clean, noise,
attack; noise is the same-budget control of the shifts) are also explained with score "verif",
the cosine of the probe's embedding (the condition image) with a clean template embedding: a
GENUINE template (another correct image of the same class from --correct-csv, preferring the other
phase, i.e. the other session) and an IMPOSTOR template (an image of a class drawn with
default_rng([--seed, image_id, 6]), also preferring the other phase). As in 15, both are of the
probe's eye (left with left, right with right; a class holds both eyes of a subject); where no
same-eye image exists any eye is used, with a warning and the count in the summary (same_eye). The
coalition images are the same as for the classification score (built once, embedded separately). An
image without another correct image of its class stops the run before the model loads.

Run times: per image, condition and method, the wall time in seconds (time_<method>) and the forward
passes (nfwd_<method>) of exact ASA (per hiding baseline: building the coalition images and their
forward passes), the part and pixel baselines (stand-alone, the regression included), LayerCAM,
Eigen-CAM, attention and the receipt (computed per batch: the batch's time divided by its images;
LayerCAM also needs one backward pass). 23 summarises them; they are measured on the run's hardware
(summary: runtime.hardware). Before the first batch of a run (and again for a last batch of another
size) an untimed warm-up builds the lazily cached CAM, attention and parts sub-models and runs one
forward pass at every coalition batch size, so one-time costs stay out of the time_ columns
(summary: runtime.warmup_seconds).

Agreement: every attribution of a row becomes one 256 x 256 non-negative map (ASA, the part
baselines and the receipt: positive part values spread uniformly over the part's pixels; pixel
baselines: positive patch values over the patch's pixels; CAMs and attention: the native map upsampled
bilinearly, clipped at 0), normalised to a distribution, and js_<a>__<b> is the Jensen-Shannon
divergence (base 2) of every pair. The values of ASA and the part / pixel baselines are first
oriented as asa_order and 23's faithfulness orient phi (sign-flipped where v_full < v_empty, e.g. the
true class once the attack won), so such rows keep a map instead of dropping out as NaN. Deletion
curves stay the main faithfulness test.

Columns of rows_asa_<tag>.csv (one row per image x condition x baseline x target):
  row_id, image_id, relpath, link_path, true_idx, true_subject, eye, device, lighting, phase
  condition, attack, adapted_to, epsilon, budget_levels, defence, defence_param
                       the variant (empty where it does not apply); adapted_to = the
                       defence a BPDA-PGD image was computed through
  baseline, score      hiding method and score function of the game
  target, target_idx   "true" (true class) or "pred" (the predicted class, rows only
                       where the prediction is wrong), and its class index; with --verif-pairs
                       also "genuine" / "impostor" (score verif; target_idx = the template's class)
  target_kind          true_class, adv_class, genuine or impostor (the target, spelled out)
  template_image_id, template_relpath   the template of genuine / impostor rows (image_id = its
                       row in --correct-csv); only with --verif-pairs
  pred_idx, correct    prediction for the condition image and whether it is the true class
  margin_true          true-class margin of the condition image (negative = fooled)
  attack_success       attack+defence rows: the attack reached its own goal. fgsm / pgd / aas_pgd: it
                       fooled the undefended model (pred of the attack row != true);
                       bpda_pgd[D]: it fooled the model behind the defence D it was computed
                       through (pred of this row != true), as 14's success_after_defence
  attack_success_undefended  attack+defence rows: the attack fooled the undefended model
                       (equal to attack_success for fgsm / pgd / aas_pgd)
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
  receipt_sum          sum of the receipt over all of the model's parts (= v_full); the masses and the
                       receipt explain the classification score: NaN on genuine / impostor rows
  del_auc_<src>        normalised deletion AUC (src/xai/shapley.deletion_from_values: the score
                       rescaled from 1 with every part kept to 0 with every present part
                       hidden; lower = it gets there sooner) when the present groups are hidden
                       in the order of src = asa (this row's phi, largest share of
                       v_full - v_empty first; it is computed on the same game, which favours
                       it), layercam, eigencam, attention, receipt (highest first), or random
                       (one seeded permutation per image, the same in every condition)
  n_forward            coalition images of this image and condition that went through the
                       model for ASA (all baselines)
  <est>_parts_<g>      part baseline est = kernelshap, lime on the row's game (NaN for the hiding
                       baselines outside --baseline-hiding and on genuine / impostor rows)
  <est>_pixels_<g>     pixel baseline aggregated to group g (NaN outside --pixel-conditions and
                       --pixel-hiding)
  time_<m>, nfwd_<m>   wall time (s) and forward passes of method m for this image and condition:
                       asa (this row's baseline; genuine / impostor rows: the embedding passes),
                       kernelshap_parts, lime_parts, kernelshap_pixels, lime_pixels, layercam,
                       eigencam, attention, receipt
  js_<a>__<b>          Jensen-Shannon divergence (base 2) of the 256 x 256 maps of a and b (NaN where a
                       map is missing or has no positive mass)

Writes to --out-dir:
  rows_asa_<tag>.csv      the rows above
  images_asa_<tag>.csv    the sampled images (for --exclude-from of a later run)
  summary_asa_<tag>.json  settings, git commit and git_dirty, package versions, timing,
                          forward passes, row counts per condition, and the rates per variant
                          (23 checks the rows against them): attack_success_rate (on the
                          undefended model, every attack), attack_success_after_defence_rate
                          (every attack x defence; the headline number of a BPDA-PGD attack),
                          restoration_rate_among_successful_attacks (non-adaptive attacks only: among the
                          attacks that fooled the undefended model, the share the defence
                          classifies correctly; for BPDA-PGD it is 1 - success after defence)
                          and the flip rates of the controls; rows per target kind; the settings of
                          the part and pixel baselines, the verification templates (how many from the
                          other phase), the agreement maps; forward passes per method and the hardware
                          the run times were measured on
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
image (printed at start-up with the inpainting estimate, and with the forward passes of the
sampling baselines and the verification pairs). The part baselines add up to 2 x 16 per condition
image and hiding baseline (about as much as ASA, inpainting included: --baseline-hiding blur,mean
leaves the inpaint game out); the pixel baselines 2 x 500 per condition image of --pixel-conditions.
Inpainting runs on the CPU, about 60 ms per coalition image and thread (30 x 31 per image for ASA;
--workers threads); on the laptop CPU with the tiny test model it was most of the run time.
--baselines blur,mean skips it.

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
  with verification pairs and the pixel baselines on the attacked images as well, add:
        --verif-pairs --pixel-conditions clean,attack
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import re
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
from src.attacks.masking import DEFAULT_TAUS, aas_pgd, check_taus  # noqa: E402
from src.attacks.whitebox import fgsm, pgd  # noqa: E402
from src.defences.inference import DEFENCE_REGISTRY, make_defence  # noqa: E402
from src.models.arcface_inference import find_feature_layer, load_checkpoint, predict_logits  # noqa: E402
from src.xai.attribution import XAI_REGISTRY  # noqa: E402
from src.xai.seg_masks import load_mask_file  # noqa: E402
from src.xai.part_baselines import (DEFAULT_PART_SAMPLES, DEFAULT_PATCH, DEFAULT_PIXEL_SAMPLES,  # noqa: E402
                                    ESTIMATORS, LIME_ALPHA, LIME_KERNEL_WIDTH, aggregate_to_parts, design,
                                    evaluated, fit, pairwise_js, part_budget, patch_map, patch_part_shares,
                                    player_map_from_masks, player_value_map, sampled_outputs, upsample_map)
from src.xai.shapley import (DEFAULT_GROUPS, DEFAULT_WORKERS, METHODS, asa_order, canonical_indices,  # noqa: E402
                             coalition_images, deletion_from_values, exact_shapley, format_groups, group_masks,
                             groups_from_config, model_outputs, normalise_groups, order_from_scores,
                             oriented_phi, parse_groups, part_masses, score_values, shapley_interactions)

from src.models.anatomy_model import attention_maps, model_arch, part_names, part_receipt  # noqa: E402


def _load_script(name):
    """A numbered script as a module (its main() does not run)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_script("05_run_experiment")          # budgets, image loading, git state, versions
_ADAPTIVE = _load_script("14_adaptive_attacks")      # sampling, margins, JSON helpers

ATTACKS = ("fgsm", "pgd", "aas_pgd", "bpda_pgd")
NON_ADAPTIVE = ("fgsm", "pgd", "aas_pgd")      # computed on the undefended model; the image meets every defence
SCORES = ("margin", "logit")
SOURCES = ("asa", "layercam", "eigencam", "attention", "receipt", "random")
CONDITIONS = ("clean", "noise", "attack", "defence_on_clean", "noise+defence", "attack+defence")
TARGET_KINDS = {"true": "true_class", "pred": "adv_class", "genuine": "genuine", "impostor": "impostor"}
VERIF_TARGETS = ("genuine", "impostor")
PLAN_SEEDS = {("kernelshap", "parts"): 1, ("lime", "parts"): 2, ("kernelshap", "pixels"): 3, ("lime", "pixels"): 4}
TEMPLATE_SEEDS = {"genuine": 5, "impostor": 6}
AGREEMENT = ("asa", "kernelshap_parts", "lime_parts", "kernelshap_pixels", "lime_pixels", "receipt", "layercam",
             "eigencam", "attention")
CAM_FORWARDS = {"layercam": 1, "eigencam": 1, "attention": 1}     # receipt: 1 (logit) or 2 (margin) forward passes
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
                 score="margin", times=None):
    """Per target ("true", "pred"): source -> per-image maps (CAMs, attention) or part vectors
    (receipt, already in group order, plus "receipt_sum" over all of the model's parts).

    targets : dict target -> class indices (N,)
    receipt_index : for each group, its column in the model's receipt (None if the model has
        no such part)
    logits, score : margin-free logits of *images* and the ASA score (units of the receipt)
    times : dict, filled with target -> source -> wall time per image in seconds (the batch's
        time divided by its images; class-agnostic maps are computed once for all targets)
    """
    out = {t: {} for t in targets}
    times = {} if times is None else times
    shared, shared_time = {}, {}
    per_image = 1.0 / max(len(images), 1)
    for xai in cams:
        if not class_specific(xai):
            start = time.perf_counter()
            shared[xai] = cam_maps(model, xai, images, None, target_layer)
            shared_time[xai] = (time.perf_counter() - start) * per_image
    if capabilities["attention"]:
        start = time.perf_counter()
        maps = np.asarray(attention_maps(model, images), np.float32)       # (N, 8, 8, K)
        shared["attention"] = maps.mean(axis=-1)
        shared_time["attention"] = (time.perf_counter() - start) * per_image
    for target, class_idx in targets.items():
        out[target].update(shared)
        times[target] = dict(shared_time)
        for xai in cams:
            if class_specific(xai):
                start = time.perf_counter()
                out[target][xai] = cam_maps(model, xai, images, class_idx, target_layer)
                times[target][xai] = (time.perf_counter() - start) * per_image
        if capabilities["receipt"]:
            if logits is None:
                logits = predict_logits(model, images)
            start = time.perf_counter()
            receipt = receipt_terms(model, images, class_idx, logits, score)                # (N, P_model)
            times[target]["receipt"] = (time.perf_counter() - start) * per_image
            out[target]["receipt_sum"] = receipt.sum(axis=1)
            out[target]["receipt"] = np.stack(
                [receipt[:, j] if j is not None else np.full(len(images), np.nan) for j in receipt_index], axis=1)
    return out


# ---------------------------------------------------------------------------
# Rows of one condition
# ---------------------------------------------------------------------------

def agreement_sources(part_estimators, pixel_estimators, cams, capabilities):
    """The attributions of a run that get a 256 x 256 map and pairwise Jensen-Shannon columns, in column order."""
    maps = (["asa"] + [f"{e}_parts" for e in part_estimators] + [f"{e}_pixels" for e in pixel_estimators]
            + (["receipt"] if capabilities["receipt"] else []) + list(cams)
            + (["attention"] if capabilities["attention"] else []))
    return [s for s in AGREEMENT if s in maps] + [s for s in maps if s not in AGREEMENT]


class Context:
    """Everything about a batch of images that stays the same across conditions.

    part_estimators : sampling baselines on the part players ("kernelshap", "lime"), each with
        *part_samples* coalitions per image, condition and hiding baseline in *part_methods* (default: all)
    pixel_estimators : the same estimators on square patches of *patch* pixels with *pixel_samples*
        coalitions, on the conditions in *pixel_conditions* and the hiding baselines in *pixel_methods*
    templates : for --verif-pairs, dict kind ("genuine", "impostor") -> {"images" (N, 256, 256, 3),
        "image_id", "relpath", "class_idx"}, one template per image of the batch; the conditions in
        *verif_conditions* are also explained with the verification score against them
    The sampled coalitions are seeded per image ([seed, image_id, estimator and player kind]), so every
    condition, target and hiding baseline of an image uses the same coalitions.
    """

    def __init__(self, model, clean, labels, base_rows, mask_labels, class_names, groups, baselines, score,
                 cams, target_layer, capabilities, receipt_index, asa_batch_size, seed, workers=1,
                 part_estimators=(), part_samples=DEFAULT_PART_SAMPLES, part_methods=None, pixel_estimators=(),
                 pixel_samples=DEFAULT_PIXEL_SAMPLES, patch=DEFAULT_PATCH, pixel_conditions=(), pixel_methods=(),
                 templates=None, verif_conditions=()):
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
        self.n_forward_by = defaultdict(int)                      # method -> forward passes of this batch

        # sampling baselines on the part players (only the present groups play, as in ASA)
        self.part_estimators = list(part_estimators)
        self.part_methods = set(baselines if part_methods is None else part_methods)
        self.player_maps = [player_map_from_masks(m) for m in self.masks]
        self.part_plans = []
        for row, present in zip(base_rows, self.present):
            players, plans = np.flatnonzero(present), {}
            for est in self.part_estimators:
                if not len(players):
                    continue
                plan = design(est, len(players), part_budget(len(players), part_samples),
                              [seed, int(row["image_id"]), PLAN_SEEDS[(est, "parts")]])
                rows = evaluated(est, plan)
                full = np.zeros((len(rows), len(present)), bool)
                full[:, players] = rows
                plans[est] = (plan, players, full)
            self.part_plans.append(plans)
        # and on square patches, aggregated to the parts by pixel share
        self.pixel_conditions, self.pixel_methods = set(pixel_conditions), set(pixel_methods)
        self.pixel_estimators = list(pixel_estimators) if self.pixel_conditions and self.pixel_methods else []
        self.patches, self.patch_shares, self.pixel_plans = None, [], []
        if self.pixel_estimators:
            self.patches = patch_map(clean.shape[1:3], patch)
            n_patches = int(self.patches.max()) + 1
            for row, masks in zip(base_rows, self.masks):
                self.patch_shares.append(patch_part_shares(self.patches, masks))
                plans = {}
                for est in self.pixel_estimators:
                    plan = design(est, n_patches, pixel_samples,
                                  [seed, int(row["image_id"]), PLAN_SEEDS[(est, "pixels")]])
                    plans[est] = (plan, evaluated(est, plan))
                self.pixel_plans.append(plans)
        # verification templates: clean images, embedded once
        self.templates = {}
        for kind, t in (templates or {}).items():
            self.templates[kind] = {**{k: v for k, v in t.items() if k != "images"},
                                    "embedding": model_outputs(model, t["images"], "embedding", asa_batch_size)}
            self.n_forward_by["templates"] += len(t["images"])
        self.verif_conditions = set(verif_conditions) if self.templates else set()

        # attributions compared by Jensen-Shannon divergence, methods with a run time, optional columns
        self.sources = agreement_sources(self.part_estimators, self.pixel_estimators, cams, capabilities)
        maps = set(self.sources)
        self.methods = (["asa"] + [f"{e}_parts" for e in self.part_estimators]
                        + [f"{e}_pixels" for e in self.pixel_estimators] + list(cams)
                        + (["attention"] if capabilities["attention"] else [])
                        + (["receipt"] if capabilities["receipt"] else []))
        n = self.names
        optional = [f"mass_{s}_{g}" for s in list(cams) + (["attention"] if capabilities["attention"] else [])
                    for g in n]
        if capabilities["receipt"]:
            optional += [f"receipt_{g}" for g in n] + ["receipt_sum"]
        optional += [f"del_auc_{s}" for s in SOURCES if s in ("asa", "random") or s in maps]
        optional += [f"{e}_{kind}_{g}" for kind, ests in (("parts", self.part_estimators),
                                                          ("pixels", self.pixel_estimators))
                     for e in ests for g in n]
        optional += [f"{c}_{m}" for m in self.methods for c in ("time", "nfwd")]
        optional += [f"js_{a}__{b}" for a, b in itertools.combinations(self.sources, 2)]
        if self.templates:
            optional += ["template_image_id", "template_relpath"]
        self.optional = optional


def game_outputs(ctx, image, masks, method, cache, kinds):
    """Model outputs of the 2^P coalition images of one hiding method, built once, for each output kind
    ("logits", "embedding"). Absent groups are exact null players (src/xai/shapley.coalition_outputs).

    Returns (outputs: kind -> (2^P, C or D), present (P,), coalition images that went through the model
    per kind, seconds: kind -> building the images plus that kind's forward passes).
    """
    present = masks.reshape(len(masks), -1).any(axis=1)
    canonical = canonical_indices(len(masks), present)
    needed = np.unique(canonical)
    start = time.perf_counter()
    images = coalition_images(image, masks, method, cache, needed, ctx.workers)
    build = time.perf_counter() - start
    position = {k: i for i, k in enumerate(needed)}
    rows = np.array([position[k] for k in canonical])
    outputs, seconds = {}, {}
    for kind in kinds:
        start = time.perf_counter()
        outputs[kind] = model_outputs(ctx.model, images, kind, ctx.asa_batch_size)[rows]
        seconds[kind] = build + time.perf_counter() - start
    return outputs, present, int(len(needed)), seconds


def game_map(phi, v_full, v_empty, player_map):
    """Agreement map of attributions of one game (ASA, the part and pixel baselines): phi oriented as
    asa_order and 23's faithfulness orient it (oriented_phi: sign-flipped where v_full < v_empty, e.g. the
    true class once the attack won), then its positive values spread over the players' pixels. Without the
    orientation a game whose phi are all <= 0 would have no map and its js_ columns would be NaN."""
    return player_value_map(oriented_phi(phi, v_full, v_empty), player_map)


def js_columns(ctx, maps):
    """js_<a>__<b> for every pair of the run's agreement sources (NaN where a map is missing or has no
    positive mass), from 256 x 256 maps (src/xai/part_baselines.pairwise_js)."""
    js = pairwise_js({s: maps.get(s) for s in ctx.sources})
    return {f"js_{a}__{b}": v for (a, b), v in js.items()}


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
    expl_times = {}
    expl = explanations(ctx.model, images, targets, ctx.cams, ctx.target_layer, ctx.capabilities, ctx.receipt_index,
                        logits, ctx.score, expl_times)
    timings["explanations"] = timings.get("explanations", 0.0) + time.perf_counter() - start

    diff = (images - ctx.clean).reshape(len(images), -1)
    linf, l2 = np.abs(diff).max(axis=1), np.sqrt((diff ** 2).sum(axis=1))
    verif_here = info["condition"] in ctx.verif_conditions
    pixels_here = info["condition"] in ctx.pixel_conditions
    kinds = ["logits"] + (["embedding"] if verif_here else [])
    names, rows = ctx.names, []
    for i in range(len(images)):
        start = time.perf_counter()
        cache = {}                       # blur / mean fills of this image, shared by ASA and the sampling baselines
        games = {b: game_outputs(ctx, images[i], ctx.masks[i], b, cache, kinds) for b in ctx.baselines}
        present = games[ctx.baselines[0]][1]
        n_forward = sum(g[2] for g in games.values())
        timings["asa"] = timings.get("asa", 0.0) + time.perf_counter() - start
        ctx.n_forward += n_forward
        ctx.n_forward_by["asa"] += n_forward
        if verif_here:
            ctx.n_forward_by["asa_verif"] += n_forward

        start = time.perf_counter()
        sampled = {}                     # (baseline, "parts" | "pixels", estimator) -> (outputs, n_forward, seconds)
        for b in ctx.baselines:
            for est, (_, _, coalitions) in (ctx.part_plans[i].items() if b in ctx.part_methods else ()):
                t = time.perf_counter()
                out, nf = sampled_outputs(ctx.model, images[i], ctx.player_maps[i], coalitions, b, "logits",
                                          ctx.asa_batch_size, cache, ctx.workers)
                sampled[(b, "parts", est)] = (out, nf, time.perf_counter() - t)
                ctx.n_forward_by[f"{est}_parts"] += nf
            if pixels_here and b in ctx.pixel_methods:
                for est, (_, coalitions) in ctx.pixel_plans[i].items():
                    t = time.perf_counter()
                    out, nf = sampled_outputs(ctx.model, images[i], ctx.patches, coalitions, b, "logits",
                                              ctx.asa_batch_size, cache, ctx.workers)
                    sampled[(b, "pixels", est)] = (out, nf, time.perf_counter() - t)
                    ctx.n_forward_by[f"{est}_pixels"] += nf
        timings["sampling_baselines"] = timings.get("sampling_baselines", 0.0) + time.perf_counter() - start

        common = {**ctx.base_rows[i], **info, "pred_idx": int(pred[i]), "correct": bool(not flipped[i]),
                  "margin_true": float(margin_true[i]), "pert_linf": float(linf[i]), "pert_l2": float(l2[i])}
        common.update({k: v[i] for k, v in extra.items()})
        common.update({f"area_{g}": float(a) for g, a in zip(names, ctx.areas[i])})
        common.update({f"present_{g}": bool(p) for g, p in zip(names, present)})
        for target, class_idx in targets.items():
            if target == "pred" and not flipped[i]:
                continue
            masses, scores, native = {}, {}, {}
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
                    native[src] = upsample_map(values[i], images.shape[1:3])
            run_times = {}
            for src, seconds in expl_times.get(target, {}).items():
                run_times[f"time_{src}"] = seconds
                run_times[f"nfwd_{src}"] = CAM_FORWARDS.get(src, 1 if ctx.score == "logit" else 2)
            for baseline in ctx.baselines:
                outputs, _, n_game, seconds = games[baseline]
                values = score_values(outputs["logits"], ctx.score, target=int(class_idx[i]))
                phi, inter = exact_shapley(values), shapley_interactions(values)
                row = {**common, "baseline": baseline, "score": ctx.score, "target": target,
                       "target_kind": TARGET_KINDS[target], "target_idx": int(class_idx[i]),
                       "v_full": float(values[-1]), "v_empty": float(values[0])}
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
                maps = {"asa": game_map(phi, values[-1], values[0], ctx.player_maps[i]), **native}
                if "receipt" in scores:
                    maps["receipt"] = player_value_map(scores["receipt"], ctx.player_maps[i])
                row.update({"time_asa": seconds["logits"], "nfwd_asa": n_game})
                for (b, kind, est), (out, nf, sec) in sampled.items():
                    if b != baseline:
                        continue
                    t = time.perf_counter()
                    est_values = score_values(out, ctx.score, target=int(class_idx[i]))
                    if kind == "parts":
                        plan, players, _ = ctx.part_plans[i][est]
                        weights = np.zeros(len(names))
                        weights[players] = fit(est, plan, est_values)
                        parts, player_map = weights, ctx.player_maps[i]
                    else:
                        weights = fit(est, ctx.pixel_plans[i][est][0], est_values)
                        parts, player_map = aggregate_to_parts(weights, ctx.patch_shares[i]), ctx.patches
                    row.update({f"{est}_{kind}_{g}": float(v) for g, v in zip(names, parts)})
                    row.update({f"time_{est}_{kind}": sec + time.perf_counter() - t, f"nfwd_{est}_{kind}": nf})
                    maps[f"{est}_{kind}"] = game_map(weights, values[-1], values[0], player_map)
                row.update({f"value_{k}": float(v) for k, v in enumerate(values)})
                row["n_forward"] = n_forward
                row.update(run_times)
                row.update(js_columns(ctx, maps))
                for column in ctx.optional:
                    row.setdefault(column, np.nan)
                rows.append(row)
        if verif_here:              # the probe (this condition image) against clean genuine / impostor templates
            for kind in VERIF_TARGETS:
                template = ctx.templates[kind]
                for baseline in ctx.baselines:
                    outputs, _, n_game, seconds = games[baseline]
                    values = score_values(outputs["embedding"], "verif", template=template["embedding"][i])
                    phi, inter = exact_shapley(values), shapley_interactions(values)
                    row = {**common, "baseline": baseline, "score": "verif", "target": kind, "target_kind": kind,
                           "target_idx": int(template["class_idx"][i]),
                           "v_full": float(values[-1]), "v_empty": float(values[0])}
                    row.update({f"phi_{g}": float(v) for g, v in zip(names, phi)})
                    row.update({f"int_{names[p]}__{names[q]}": float(inter[p, q])
                                for p, q in itertools.combinations(range(len(names)), 2)})
                    for src, order in (("asa", asa_order(phi, values[-1], values[0], present)),
                                       ("random", ctx.random_order[i])):
                        row[f"del_auc_{src}"] = (float("nan") if order is None
                                                 else deletion_from_values(values, order)["auc"])
                    row.update({f"value_{k}": float(v) for k, v in enumerate(values)})
                    row["n_forward"] = n_forward
                    row.update({"time_asa": seconds["embedding"], "nfwd_asa": n_game,
                                "template_image_id": int(template["image_id"][i]),
                                "template_relpath": template["relpath"][i]})
                    row.update(js_columns(ctx, {"asa": game_map(phi, values[-1], values[0], ctx.player_maps[i])}))
                    for column in ctx.optional:
                        row.setdefault(column, np.nan)
                    rows.append(row)
    return rows, pred


# ---------------------------------------------------------------------------
# Warm-up before the timed regions
# ---------------------------------------------------------------------------

def _chunk_sizes(n, cap):
    """Batch sizes of the forward passes over *n* images, *cap* per pass."""
    return {min(n, cap)} | ({n % cap} if n > cap and n % cap else set())


def warm_up_sizes(ctx):
    """Every batch size the timed coalition forward passes can use: 1 up to 2^P or --asa-batch-size (an ASA
    game and the distinct coalitions of a part baseline never exceed 2^P images, and LIME's distinct rows
    vary from image to image), and the chunks of the pixel baselines' coalition sets of *ctx*."""
    cap = ctx.asa_batch_size
    sizes = set(range(1, min(2 ** len(ctx.names), cap) + 1))
    for plans in ctx.pixel_plans:
        for _, coalitions in plans.values():
            sizes |= _chunk_sizes(len(np.unique(coalitions, axis=0)), cap)
    return sorted(sizes)


def warm_up(ctx, coalitions=True):
    """One untimed pass over what the timed regions would otherwise pay for on first use: the CAM, attention
    and parts sub-models (built lazily and cached per model) at this batch's size and, with *coalitions*, one
    forward pass at every coalition batch size of :func:`warm_up_sizes` (GPU kernels are tuned per input
    shape). Embeddings run through the same sub-model as the logits. Nothing is counted or kept.
    Returns the seconds it took."""
    start = time.perf_counter()
    logits = predict_logits(ctx.model, ctx.clean)
    explanations(ctx.model, ctx.clean, {"true": ctx.labels}, ctx.cams, ctx.target_layer, ctx.capabilities,
                 ctx.receipt_index, logits, ctx.score)
    if coalitions:
        for size in warm_up_sizes(ctx):
            model_outputs(ctx.model, np.repeat(ctx.clean[:1], size, axis=0), "logits", ctx.asa_batch_size)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# One batch: every condition
# ---------------------------------------------------------------------------

def run_batch(ctx, epsilons, defences, attacks, steps, seed, rng, rates, timings, taus=DEFAULT_TAUS):
    """All conditions of one batch of clean images; returns the rows.

    defences : dict name -> (defence function, parameter value)
    rng : numpy Generator for the noise control (one per run, as in 05)
    taus : attention temperatures of aas_pgd
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
        for name in NON_ADAPTIVE:
            if name not in attacks:
                continue
            start = time.perf_counter()
            if name == "fgsm":
                adv = fgsm(clean, ctx.model, eps, **common)
            elif name == "pgd":
                adv = pgd(clean, ctx.model, eps, steps=steps, seed=seed, **common)
            else:                                   # same start, step size and steps as pgd
                adv = aas_pgd(clean, ctx.model, eps, steps=steps, seed=seed, taus=taus, **common)
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


def conditions_per_image(n_eps, n_def, attacks, conditions=None):
    """Condition images per clean image (each explained with every baseline); with *conditions*, only
    those of the listed conditions (e.g. the conditions of --pixel-conditions)."""
    plain = sum(a in attacks for a in NON_ADAPTIVE)
    adaptive = n_def if "bpda_pgd" in attacks else 0          # one BPDA-PGD image per defence, with that defence
    counts = {"clean": 1, "defence_on_clean": n_def, "noise": n_eps, "noise+defence": n_eps * n_def,
              "attack": n_eps * (plain + adaptive), "attack+defence": n_eps * (plain * n_def + adaptive)}
    return sum(n for c, n in counts.items() if conditions is None or c in conditions)


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


EYE_RE = re.compile(r"(?:^|[/\\])(?:enrollment|verification)_(left|right)_eye(?:[/\\]|$)")


def image_eyes(frame):
    """The eye ('left' / 'right', None where unknown) of every row of *frame*: its eye column where that
    holds left / right, else the <phase>_<eye>_eye folder of its relpath or link_path."""
    out = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    if "eye" in frame.columns:
        eye = frame["eye"].map(lambda e: str(e).strip().lower() if isinstance(e, str) else None)
        out = eye.where(eye.isin(["left", "right"]), None).astype(object)
    for column in ("relpath", "link_path"):
        if column in frame.columns and out.isna().any():
            found = frame[column].map(lambda p: (m.group(1) if isinstance(p, str) and (m := EYE_RE.search(p))
                                                 else None))
            out = out.where(out.notna(), found)
    return out


def choose_templates(sampled, pool, seed):
    """Verification templates of every sampled image, from the correct-image list *pool* (its index is
    the image_id): a GENUINE template (another image of the same class) and an IMPOSTOR template (an
    image of another class, the class drawn with numpy default_rng([seed, image_id, 6])). As in 15, a probe
    is compared only with the same eye (a VISOB class holds both eyes of a subject): the genuine template is
    an image of the probe's eye, and the impostor class is drawn among the other classes with an image of
    the probe's eye, its template one of those images. Where no same-eye image exists (or the eye is unknown)
    any image of the class is used, counted in one warning. Within those candidates both prefer an image of
    the other phase (enrollment / verification, i.e. the other session); the image is drawn with
    default_rng([seed, image_id, 5 or 6]). The eye comes from the eye column, else the relpath / link_path.

    Returns
    -------
    dict kind -> DataFrame aligned with *sampled*: template_image_id, relpath, link_path, class_idx,
    other_phase (True where the template is from the other phase), same_eye (True where the template is
    of the probe's eye)
    Raises ValueError if an image's class has no other correct image.
    """
    has_phase = "phase" in pool.columns
    pool = pool.assign(eye_=image_eyes(pool))
    probe_eyes = image_eyes(sampled).tolist()
    by_class = {c: g for c, g in pool.groupby("true_idx", sort=True)}
    classes = np.array(sorted(by_class))
    eyes_of = {c: set(g["eye_"].dropna()) for c, g in by_class.items()}
    out = {kind: [] for kind in VERIF_TARGETS}
    missing, fallback = [], {kind: 0 for kind in VERIF_TARGETS}
    for row, eye in zip(sampled.itertuples(), probe_eyes):
        image_id, label = int(row.image_id), int(row.true_idx)
        phase = getattr(row, "phase", None) if has_phase else None
        for kind in VERIF_TARGETS:
            rng = np.random.default_rng([seed, image_id, TEMPLATE_SEEDS[kind]])
            if kind == "genuine":
                candidates = by_class[label][by_class[label].index != image_id]
            else:
                others = classes[classes != label]
                if not len(others):
                    raise ValueError("an impostor template needs a second class in --correct-csv")
                same = others[[eye in eyes_of[c] for c in others]] if eye is not None else others[:0]
                others = same if len(same) else others
                candidates = by_class[int(others[rng.integers(len(others))])]
            if not len(candidates):
                missing.append(str(row.link_path))
                continue
            same_eye = candidates[candidates["eye_"] == eye] if eye is not None else candidates[:0]
            if len(same_eye):
                candidates = same_eye
            else:
                fallback[kind] += 1
            other = candidates[candidates.phase != phase] if has_phase and isinstance(phase, str) else candidates[:0]
            pick = other if len(other) else candidates
            chosen = pick.iloc[int(rng.integers(len(pick)))]
            out[kind].append({"template_image_id": int(chosen.name), "relpath": chosen.get("relpath"),
                              "link_path": chosen.link_path, "class_idx": int(chosen.true_idx),
                              "other_phase": bool(len(other)), "same_eye": bool(len(same_eye))})
    if missing:
        raise ValueError(f"{len(missing)} sampled images have no other correct image of their class for a genuine "
                         f"template:\n  " + "\n  ".join(missing[:20]))
    if any(fallback.values()):
        print(f"WARNING: {sum(fallback.values())} verification templates have no candidate of the probe's eye (or "
              f"the eye is unknown) and use any eye: " + ", ".join(f"{k} {n}" for k, n in fallback.items()))
    return {kind: pd.DataFrame(rows) for kind, rows in out.items()}


def batch_templates(templates, start, n, data_root):
    """The templates of images start .. start + n - 1 for Context: kind -> images (clean, loaded like the
    probes), image_id, relpath, class_idx; None without --verif-pairs."""
    if templates is None:
        return None
    out = {}
    for kind, frame in templates.items():
        part = frame.iloc[start:start + n]
        out[kind] = {"images": np.stack([_RUNNER.load_image(_RUNNER.resolve_path(r, data_root))
                                         for r in part.itertuples()]),
                     "image_id": part.template_image_id.to_numpy(), "relpath": part.relpath.tolist(),
                     "class_idx": part.class_idx.to_numpy()}
    return out


def hardware():
    """Where the run times were measured: platform, CPU threads and GPU names."""
    import platform
    gpus = []
    for device in tf.config.list_physical_devices("GPU"):
        try:
            gpus.append(tf.config.experimental.get_device_details(device).get("device_name", device.name))
        except Exception:                                    # details are not available on every platform
            gpus.append(device.name)
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_threads": os.cpu_count(),
            "gpus": gpus}


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
                    help="fgsm, pgd (non-adaptive, 8-bit), optionally aas_pgd (non-adaptive, attention-temperature "
                         "scaling, src/attacks/masking.py) and bpda_pgd (adaptive, through each defence)")
    ap.add_argument("--taus", default=",".join(f"{t:g}" for t in DEFAULT_TAUS),
                    help="attention temperatures aas_pgd tries at every step (as in 14 and 24)")
    ap.add_argument("--steps", type=int, default=10,
                    help="steps of pgd, aas_pgd and bpda_pgd (10 = whitebox.pgd's default, as in 05; 14 used 20)")
    ap.add_argument("--defences", default="jpeg,bit_depth", help="jpeg and/or bit_depth ('none': no defence)")
    ap.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality of the jpeg defence")
    ap.add_argument("--bits", type=int, default=4, help="bit depth of the bit_depth defence")
    ap.add_argument("--baselines", default="blur,mean,inpaint", help="hiding methods (src/xai/shapley.py)")
    ap.add_argument("--score", default="margin", choices=SCORES,
                    help="margin (target logit minus the best other margin-free logit) or logit")
    ap.add_argument("--cams", default="layercam,eigencam", help="CAMs for part masses ('none' skips them)")
    ap.add_argument("--target-layer", default=None,
                    help="layer the CAMs explain (default: the last 4-D activation, block14_sepconv2_act)")
    ap.add_argument("--part-baselines", default="kernelshap,lime",
                    help="sampling baselines on the same part players, hiding methods and score as ASA "
                         "(src/xai/part_baselines.py; 'none' skips them)")
    ap.add_argument("--baseline-hiding", default=None,
                    help="hiding baselines (of --baselines) used by the part baselines (default: all of them; "
                         "each runs stand-alone, so with inpaint it adds the inpainting of its coalitions)")
    ap.add_argument("--baseline-samples", type=int, default=DEFAULT_PART_SAMPLES,
                    help="cap of the coalitions per part baseline, image, condition and hiding baseline: half "
                         "of the 2^P coalitions of the image's present parts, at most this (KernelSHAP: the "
                         "empty and the full one included; 16 = half of the 2^5 coalitions)")
    ap.add_argument("--pixel-baselines", default="kernelshap,lime",
                    help="the same estimators on square patches, aggregated to the parts by pixel share "
                         "('none' skips them)")
    ap.add_argument("--pixel-samples", type=int, default=DEFAULT_PIXEL_SAMPLES,
                    help="coalitions per pixel baseline, image, condition and hiding baseline")
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH,
                    help="patch side in pixels for the pixel baselines (32 = 8 x 8 = 64 players)")
    ap.add_argument("--pixel-conditions", default="clean",
                    help=f"conditions that get the pixel baselines (of {', '.join(CONDITIONS)}): each costs "
                         "2 x --pixel-samples forward passes per image and hiding method; e.g. clean,attack")
    ap.add_argument("--pixel-hiding", default=None,
                    help="hiding baselines (of --baselines) used by the pixel baselines (default: blur, else the "
                         "first of --baselines)")
    ap.add_argument("--verif-pairs", action="store_true",
                    help="also explain the verification score (cosine with a template embedding) against a "
                         "genuine template (another correct image of the subject and eye, preferably of the other "
                         "phase) and a seeded same-eye impostor template; the probe is the condition image, "
                         "templates stay clean")
    ap.add_argument("--verif-conditions", default="clean,noise,attack",
                    help="conditions explained with --verif-pairs (noise: the same-budget control of the shifts)")
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
    try:
        taus = check_taus([float(t) for t in args.taus.split(",") if t.strip()])
    except ValueError as e:
        ap.error(f"--taus: {e}")
    if min(args.steps, args.batch_size, args.asa_batch_size, args.per_class, args.workers) < 1:
        ap.error("--steps, --batch-size, --asa-batch-size, --per-class and --workers must be at least 1")
    part_estimators, pixel_estimators = split_list(args.part_baselines), split_list(args.pixel_baselines)
    for name in part_estimators + pixel_estimators:
        if name not in ESTIMATORS:
            ap.error(f"unknown sampling baseline {name!r}; choose from {list(ESTIMATORS)}")
    if min(args.baseline_samples, args.pixel_samples) < 2:
        ap.error("--baseline-samples and --pixel-samples must be at least 2")
    if not 1 <= args.patch_size <= 256:
        ap.error("--patch-size must be between 1 and 256 pixels")
    pixel_conditions, verif_conditions = split_list(args.pixel_conditions), split_list(args.verif_conditions)
    for flag, values in (("--pixel-conditions", pixel_conditions), ("--verif-conditions", verif_conditions)):
        for name in values:
            if name not in CONDITIONS:
                ap.error(f"{flag}: unknown condition {name!r}; choose from {list(CONDITIONS)}")
    part_hiding = baselines if args.baseline_hiding is None else split_list(args.baseline_hiding)
    for name in part_hiding:
        if name not in baselines:
            ap.error(f"--baseline-hiding {name!r} is not one of --baselines {baselines}")
    if not part_hiding:
        part_estimators = []
    if args.pixel_hiding is None:
        pixel_hiding = ["blur" if "blur" in baselines else baselines[0]] if baselines else []
    else:
        pixel_hiding = split_list(args.pixel_hiding)
    for name in pixel_hiding:
        if name not in baselines:
            ap.error(f"--pixel-hiding {name!r} is not one of --baselines {baselines}")
    if not (pixel_estimators and pixel_conditions and pixel_hiding):
        pixel_estimators, pixel_conditions, pixel_hiding = [], [], []
    if args.verif_pairs and not verif_conditions:
        ap.error("--verif-pairs needs at least one --verif-conditions")
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
    pool = df.copy()                       # every correct image (index = image_id): verification templates
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
    templates = None
    if args.verif_pairs:
        try:
            templates = choose_templates(df, pool, args.seed)
        except ValueError as err:
            raise SystemExit(f"--verif-pairs: {err}")
        print("verification templates: " + ", ".join(
            f"{kind} {int(t.other_phase.sum())} of {len(t)} from the other phase, {int(t.same_eye.sum())} of the "
            f"probe's eye" for kind, t in templates.items()))

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
    if part_estimators:
        n_part = n_conditions * len(part_hiding) * len(part_estimators)
        print(f"part baselines {part_estimators} with {part_hiding}: at most {n_part} x {args.baseline_samples} = "
              f"{n_part * args.baseline_samples} forward passes per image (run stand-alone, inpainting included)")
    if pixel_estimators:
        n_pixel = conditions_per_image(len(epsilons), len(defences), attacks, pixel_conditions)
        print(f"pixel baselines {pixel_estimators} ({args.patch_size} px patches) on {pixel_conditions} with "
              f"{pixel_hiding}: {n_pixel} condition images x {len(pixel_hiding) * len(pixel_estimators)} x "
              f"{args.pixel_samples} = {n_pixel * len(pixel_hiding) * len(pixel_estimators) * args.pixel_samples} "
              "forward passes per image")
    if args.verif_pairs:
        n_verif = conditions_per_image(len(epsilons), len(defences), attacks, verif_conditions)
        print(f"verification pairs on {verif_conditions}: {n_verif} condition images x {len(baselines)} x "
              f"2^{len(names)} more forward passes per image (embeddings)")

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
    forwards_by = Counter()                 # method -> forward passes (ASA, the sampling baselines, templates)
    os.makedirs(parts_dir, exist_ok=True)
    if state:                       # continue after the last finished batch, with its noise stream and counts
        rng.bit_generator.state = state["rng"]
        rates = Rates.from_list(state["rates"])
        timings, n_forward, minutes_before = dict(state["seconds"]), int(state["n_forward"]), float(state["minutes"])
        forwards_by = Counter(state.get("forwards_by", {}))
        for name in os.listdir(parts_dir):          # a batch written after the last state was saved runs again
            if name.startswith("rows_") and name.endswith(".csv") and int(name[5:-4]) >= start_at:
                os.remove(os.path.join(parts_dir, name))
        print(f"resuming after {start_at} of {len(df)} images ({minutes_before:.1f} min before)")
    t0 = time.time()
    warmed, warmup_seconds = set(), 0.0     # batch sizes already warmed up (the last batch may be smaller)
    for start in range(start_at, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        clean = np.stack([_RUNNER.load_image(_RUNNER.resolve_path(r, args.data_root)) for r in batch.itertuples()])
        labels = batch.true_idx.to_numpy()
        ctx = Context(model, clean, labels, batch[base_cols].to_dict("records"),
                      mask_labels[start:start + len(batch)], class_names, groups, baselines, args.score, cams,
                      args.target_layer, capabilities, receipt_index, args.asa_batch_size, args.seed,
                      args.workers, part_estimators=part_estimators, part_samples=args.baseline_samples,
                      part_methods=part_hiding,
                      pixel_estimators=pixel_estimators, pixel_samples=args.pixel_samples, patch=args.patch_size,
                      pixel_conditions=pixel_conditions, pixel_methods=pixel_hiding,
                      templates=batch_templates(templates, start, len(batch), args.data_root),
                      verif_conditions=verif_conditions)
        if len(batch) not in warmed:
            warmup_seconds += warm_up(ctx, coalitions=not warmed)
            warmed.add(len(batch))
        rows = run_batch(ctx, epsilons, defences, attacks, args.steps, args.seed + start, rng, rates, timings,
                         taus)
        n_forward += ctx.n_forward
        forwards_by.update(ctx.n_forward_by)
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
                                 "forwards_by": {k: int(v) for k, v in forwards_by.items()},
                                 "minutes": minutes_before + minutes})
        print(f"  {done}/{len(df)} images | {minutes:.1f} min | ETA "
              f"{minutes / (done - start_at) * (len(df) - done):.1f} min | "
              f"{ctx.n_forward / len(batch):.0f} ASA forward passes per image", flush=True)

    # --- outputs ---------------------------------------------------------------------
    # the rows of every batch, read back from the parts (the same path for a resumed and an uninterrupted run)
    parts = sorted(name for name in os.listdir(parts_dir) if name.startswith("rows_") and name.endswith(".csv"))
    rows = pd.concat([pd.read_csv(os.path.join(parts_dir, name)) for name in parts], ignore_index=True, sort=False)
    rows = pd.concat([pd.DataFrame({"row_id": np.arange(len(rows))}), rows], axis=1)
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
               "steps": args.steps, "step_size": "2.5 * epsilon / steps (pgd, aas_pgd, bpda_pgd)",
               **({"aas_taus": list(taus), "aas_tau_selection": "per step and image, the proposal after which "
                   "the unscaled model's margin loss is largest"} if "aas_pgd" in attacks else {}),
               "random_start": "uniform in the epsilon ball, seed + first index of the batch",
               "noise": "epsilon * sign(uniform(-1, 1)) from numpy default_rng(seed), rounded to 8 bits (as in 05)",
               "defences": {d: {"quality": p} if d == "jpeg" else {"bits": p} for d, (_, p) in defences.items()},
               "baselines": baselines, "score": args.score, "cams": cams, "cam_layer": target_layer,
               "anatomy": capabilities, "targets": "true class; predicted class where it differs (not clean)" + (
                   "; genuine and impostor templates (score verif) on " + ", ".join(verif_conditions)
                   if args.verif_pairs else ""),
               "rows_per_target_kind": {k: int(v) for k, v in Counter(rows.target_kind).items()},
               "part_baselines": {
                   "estimators": part_estimators, "samples": args.baseline_samples, "hiding": part_hiding,
                   "budget": "half of the 2^P coalitions of the image's present parts, at most samples, at "
                             "least 2 (src/xai/part_baselines.part_budget)",
                   "budget_by_present_parts": {str(p): part_budget(p, args.baseline_samples)
                                               for p in range(1, len(groups) + 1)},
                   "kernelshap": "Shapley kernel; sizes enumerated as in shap's KernelExplainer while the budget "
                                 "covers them, the rest sampled by kernel mass with complements, without "
                                 "replacement; efficiency enforced by eliminating one player (as shap); the "
                                 "budget counts the empty and the full coalition",
                   "lime": {"sampling": "uniform (each part kept with probability 1/2), first sample the full image",
                            "kernel": "sqrt(exp(-d^2 / w^2)), d = cosine distance to the full coalition",
                            "kernel_width": LIME_KERNEL_WIDTH, "ridge_alpha": LIME_ALPHA},
                   "seed": "default_rng([seed, image_id, 1 (kernelshap) | 2 (lime)]): the same coalitions in every "
                           "condition, target and hiding baseline of an image"},
               "pixel_baselines": {
                   "estimators": pixel_estimators, "samples": args.pixel_samples, "patch_size": args.patch_size,
                   "n_patches": int(patch_map((256, 256), args.patch_size).max()) + 1,
                   "conditions": pixel_conditions, "hiding": pixel_hiding,
                   "aggregation": "part p = sum over patches j of w_j x (share of patch j's pixels in part p)",
                   "seed": "default_rng([seed, image_id, 3 (kernelshap) | 4 (lime)])"},
               "verification": {
                   "enabled": bool(args.verif_pairs), "conditions": verif_conditions if args.verif_pairs else [],
                   "templates": "genuine: another correct image of the class and the probe's eye; impostor: an "
                                "image of the probe's eye of a class drawn with default_rng([seed, image_id, 6]) "
                                "among the classes with one (any eye where none exists, counted in same_eye); "
                                "both prefer the other phase; the probe is the condition image, the template "
                                "stays clean",
                   "other_phase": ({k: int(t.other_phase.sum()) for k, t in templates.items()}
                                   if templates else None),
                   "same_eye": ({k: int(t.same_eye.sum()) for k, t in templates.items()}
                                if templates else None)},
               "agreement": {"sources": agreement_sources(part_estimators, pixel_estimators, cams, capabilities),
                             "maps": "256 x 256, non-negative: positive part (or patch) values spread uniformly "
                                     "over its pixels (ASA and the part / pixel baselines: phi oriented as in "
                                     "23's faithfulness, sign-flipped where v_full < v_empty); CAMs and "
                                     "attention upsampled bilinearly and clipped at 0",
                             "divergence": "Jensen-Shannon, base 2, of the maps normalised to distributions"},
               "forward_passes_by_method": {k: int(v) for k, v in sorted(forwards_by.items())},
               "runtime": {"hardware": hardware(),
                           "note": "time_<method> / nfwd_<method> columns: wall time in seconds and forward passes "
                                   "per image and condition, measured on this hardware; batched methods (CAMs, "
                                   "attention, receipt) get their batch's time divided by its images",
                           "warmup": "untimed, before the first batch (and a smaller last batch): sub-models "
                                     "built and one forward pass at every coalition batch size",
                           "warmup_seconds": round(warmup_seconds, 2)},
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
