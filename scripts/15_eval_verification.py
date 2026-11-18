"""
scripts/15_eval_verification.py

Biometric verification and identification with a trained Xception + ArcFace checkpoint,
reported as ISO/IEC 19795-1 describes: ROC and DET curves, EER, TAR at FAR 1e-2 and
1e-3, and rank-1 identification, overall and per device and lighting, with 95 %
subject-cluster bootstrap CIs. The score of two images is the cosine of their embeddings
at the input of the ArcFace head (src/eval/verification.py); FAR and FRR are the false
match and false non-match rates of single comparisons.

Protocols (--modes, any combination):
  p1  closed-set verification on the pinned random-split test set (splits/visob_test.csv):
      all same-eye pairs of its images. INFLATED, for reproduction only: consecutive
      frames of one (subject, condition, eye, phase) bucket are near-duplicates that sit
      on both sides of a pair and, for the existing checkpoint, in its training set.
      Rank-1 is leave-one-out nearest neighbour.
  p3  open-set verification on subjects never used in training (IDs outside 1141-3279
      and 7000): each subject's enrollment-phase images of one eye form a template (their
      mean embedding, renormalised), its verification-phase images are the probes.
      Cross-session and unseen identities; works with the existing checkpoint.
  p2  the same template / probe protocol on the training classes. Only meaningful for a
      model trained with 02_make_splits.py --protocol cross_session, which never saw a
      verification image; the existing checkpoint trained on about 60 % of them.
  x1  zero-shot cross-dataset open-set verification on an external dataset prepared by
      20_prepare_ubipr.py (--external-manifest): UBIPr, Canon EOS 5D at 4-8 m against the
      smartphone images the model was trained on. Templates are the session-1 images of each
      class (one eye of one person), probes its session-2 images; rows marked unusable in the
      manifest are left out. Genuine: probe and template of the same class; impostor: a
      template of another person (a probe is never scored against another class of its own
      person). Bootstrap clusters and n_subjects are persons. Subsets: --external-subsets
      (default camera distance and glasses), by the probe.
      --x1-protocol alonso2025 replaces this pairing (not the crops) with that of
      Alonso-Fernandez et al. 2025 (arXiv 2510.26282): one frontal image per eye, session and
      distance (--x1-frontal) of the persons with two sessions; both eyes are one identity, so
      crops with the manifest's eye 'L' (nose on the crop's right) are flipped horizontally
      before embedding; single images are compared (no templates): session 1 at Di against
      session 2 at Di (genuine, same person) or both sessions at Dj, i < j (genuine,
      cross-distance), and session 1 at Di against session 2 at Dj of every other person
      (impostor), with D1 = 8 m ... D5 = 4 m. The paper reports 8,600 genuine and 438,600
      impostor scores (86 persons, 1,718 images); the summary JSON gives ours next to theirs.
      Subsets: all, intra, cross and each of the 15 distance combinations. No rank-1 (no
      gallery). Their crops (a 7.6 Rs square around the sclera centre, from their own manual
      sclera annotation, which UBIPr's files do not hold) are NOT reproduced: both protocols
      use 20's corner-based VISOB-like crops.

Probes are compared only with templates (p1: images) of the same eye; a probe whose
subject has no enrollment template for that eye is left out and counted. Per-device and
per-lighting rows use the probe's device / lighting (p1: both images in the group).
Rank-1 ranks the probe's own template among all templates of that eye.

Writes to --out-dir:
  verification_metrics_<tag>.csv   one row per mode x subset (all, device=..., lighting=...)
  verification_scores_<tag>.csv    a random sample of genuine and impostor scores per mode
  verification_roc_det_<tag>.png   ROC and DET curves of every mode
  verification_summary_<tag>.json  headline numbers, CMC, counts, caveats, git commit, timing
  embeddings_<tag>.npz             with --save-embeddings: float16 embeddings and relpaths
  embeddings_<name>_<tag>.npz      x1 with --save-embeddings: the same for the external images

Usage (Kaggle, checkpoint dataset and VISOB attached, 01 already run):
  python scripts/15_eval_verification.py \
      --ckpt /kaggle/input/<checkpoint-dataset>/Xception_ArcFace_full.keras \
      --manifest /kaggle/working/manifest_raw.csv --modes p1,p3 --tag full
  smoke run (2-3 min): add --max-subjects 10 --n-boot 100 --tag smoke
Cross-dataset (VISOB not needed; 20_prepare_ubipr.py already run):
  python scripts/15_eval_verification.py --ckpt <checkpoint>.keras --modes x1 \
      --external-manifest /kaggle/working/ubipr/ubipr_manifest.csv --tag ubipr
  the pairing of Alonso-Fernandez et al. 2025: add --x1-protocol alonso2025 and another --tag
Any arch of 03 --arch loads. The attnet_gate model needs the eye masks at test time: add
--region-masks masks64_<tag>.npz (21) or masks_<tag>.npz (17) covering every embedded image
(matched by condition, phase and file name; a missing mask stops the run). Other archs ignore it.
attnet_gate runs p1, p2 and p3 only: the masks are keyed by VISOB relpaths (condition / phase_eye /
file), and nothing makes masks for the external images of x1, so --modes x1 with --region-masks
stops before the model loads (and an attnet_gate checkpoint with x1 stops right after it loads).
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # reference code targets Keras 2; set before importing tensorflow
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.eval.verification import (ALONSO2025_DISTANCES, ALONSO2025_PAPER, BINS_PER_UNIT, FARS,  # noqa: E402
                                   alonso2025_nominal_counts, alonso2025_pairs, alonso2025_select,
                                   bootstrap_metrics, build_templates, cmc, embed, error_rates_from_counts,
                                   exclude_same_person, far_key, metrics_from_counts, pair_counts,
                                   pair_list_counts, rank1_ci, ranks, roc_from_counts, template_counts,
                                   template_scores)
from src.models.arcface_inference import load_checkpoint  # noqa: E402
from src.models.anatomy_model import embed_regions, feature_grid, model_arch, region_classes  # noqa: E402

DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
RELPATH_RE = re.compile(r"^(?P<condition>[^/]+)/(?P<phase>enrollment|verification)_(?P<eye>left|right)_eye/"
                        r"[^/]+/(?P<subject>\d+)_\d+_[lr]_c\.png$", re.IGNORECASE)
PROTOCOLS = {
    "p1": "closed-set verification: all same-eye pairs of the pinned random-split test set",
    "p2": "closed-set cross-session verification: enrollment templates vs verification probes, training subjects",
    "p3": "open-set cross-session verification: enrollment templates vs verification probes, unseen subjects",
    "x1": "zero-shot cross-dataset open-set verification: session-1 templates vs session-2 probes of an external "
          "dataset",
}
X1_CAVEAT = ("zero-shot cross-dataset and cross-sensor: a VISOB (smartphone) model on {name} images, eye crops "
             "re-framed from landmarks by 20_prepare_ubipr.py; n_subjects counts persons")
X1_PROTOCOLS = ("ours", "alonso2025")
X1_ALONSO_PROTOCOL = ("zero-shot cross-dataset verification, pairing of Alonso-Fernandez et al. 2025: single frontal "
                      "images, session 1 vs session 2 within and across distances, both eyes one identity")
X1_ALONSO_CAVEAT = ("pairing of Alonso-Fernandez et al. 2025 (frontal image per eye, session and distance; left-eye "
                    "crops flipped; both eyes one identity), but on 20_prepare_ubipr.py's corner-based VISOB-like "
                    "crops, not their 7.6 Rs squares around the sclera centre, and on this archive's images and "
                    "persons, so the numbers of comparisons differ from theirs (see the summary JSON)")
VISOB_MODES = ("p1", "p2", "p3")
P1_CAVEAT = ("inflated by near-duplicate frames: consecutive frames of one bucket sit on both sides of a pair "
             "and in training; reproduces the base-paper protocol only")
COLORS = {"p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a", "x1": "#8a5cd6"}   # fixed per protocol, never by rank
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}


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


def image_table(relpaths):
    """Subject, phase, eye, device, lighting and condition of each relpath (01's layout), sorted."""
    rows = []
    for rel in sorted({str(r).replace("\\", "/") for r in relpaths}):
        m = RELPATH_RE.match(rel)
        if not m:
            raise ValueError(f"unexpected relpath {rel!r}: expected "
                             "<condition>/<phase>_<eye>_eye/<subject>/<subject>_<idx>_<l|r>_c.png")
        device, lighting = parse_condition(m["condition"])
        rows.append({"relpath": rel, "subject": int(m["subject"]), "phase": m["phase"].lower(),
                     "eye": m["eye"].lower(), "device": device, "lighting": lighting,
                     "condition": m["condition"]})
    return pd.DataFrame(rows, columns=["relpath", "subject", "phase", "eye", "device", "lighting", "condition"])


def external_table(manifest, subset_cols=()):
    """Image table of an external manifest (20_prepare_ubipr.py), sorted by relpath.

    subject = class_id (one eye of one person), person = person_id (class_id if absent), phase =
    enrollment for session 1 and verification for session 2, eye as given; rows with usable = 0 are
    dropped. Returns (table, number of rows dropped as unusable).
    """
    df = pd.read_csv(manifest)
    missing = ({"relpath", "class_id", "eye", "session"} | set(subset_cols)) - set(df.columns)
    if missing:
        raise SystemExit(f"{manifest} lacks the columns {sorted(missing)} (write it with 20_prepare_ubipr.py)")
    unusable = (df.usable == 0).to_numpy() if "usable" in df.columns else np.zeros(len(df), bool)
    df = df[~unusable]
    if not df.session.isin([1, 2]).all():
        raise SystemExit(f"{manifest}: sessions must be 1 (templates) or 2 (probes), found "
                         f"{sorted(df.session.unique().tolist())}")
    table = pd.DataFrame({"relpath": df.relpath.astype(str).str.replace("\\", "/", regex=False),
                          "subject": df.class_id.astype(int).to_numpy(),
                          "person": (df.person_id if "person_id" in df.columns else df.class_id).astype(int).to_numpy(),
                          "phase": np.where(df.session == 1, "enrollment", "verification"),
                          "eye": df.eye.astype(str).to_numpy()})
    for col in subset_cols:
        table[col] = df[col].to_numpy()
    if table.relpath.duplicated().any():
        raise SystemExit(f"{manifest}: duplicate relpaths")
    return table.sort_values("relpath").reset_index(drop=True), int(unusable.sum())


def in_paper_classes(subjects):
    """True for subject IDs 1141..3279 plus 7000, the 300 classes of 02_make_splits.py."""
    s = np.asarray(subjects)
    return ((s >= 1141) & (s <= 3279)) | (s == 7000)


def load_image(path, flip=False):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest); flip: mirrored
    left-right (x1 with --x1-protocol alonso2025: the crops of the eye labelled 'L')."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    img = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(img[:, ::-1]) if flip else img


def git_commit():
    """HEAD of the repo, with '-dirty' if tracked files have uncommitted changes; 'unknown' without git."""
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        return (commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")) or "unknown"
    except Exception:
        return "unknown"


def training_info(ckpt, class_index=None):
    """(training subjects or None, where they came from, training images per phase).

    Reads class_index_<tag>.json and results_<tag>.json, which 03_train_baseline.py writes
    next to Xception_ArcFace_<tag>.keras. Without a class index the training subjects are
    the 300 classes of the paper protocol. Images per phase is None without a results file
    and {} for a results file written before 03 recorded phases (a random-split model).
    """
    folder = os.path.dirname(os.path.abspath(ckpt))
    m = re.match(r"Xception_ArcFace_(.+)\.(keras|h5)$", os.path.basename(ckpt))
    tag = m.group(1) if m else None
    if class_index and not os.path.exists(class_index):
        raise FileNotFoundError(class_index)
    candidates = [class_index] if class_index else \
        [os.path.join(folder, f"class_index_{tag}.json") if tag else None, os.path.join(folder, "class_index.json")]
    seen, source = None, "subject IDs 1141-3279 plus 7000 (no class index found)"
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                seen, source = {int(k) for k in json.load(f)}, path
            break
    phases = None
    results = os.path.join(folder, f"results_{tag}.json") if tag else None
    if results and os.path.exists(results):
        with open(results) as f:
            phases = json.load(f).get("images_by_phase", {}).get("train", {})
    return seen, source, phases


def caveats(mode, train_phases, external_name="ubipr", x1_protocol="ours"):
    """What a reader must know before quoting this mode's numbers for this checkpoint."""
    if mode == "p1":
        notes = [P1_CAVEAT]
        if train_phases and not train_phases.get("verification"):
            notes.append("this model was trained without verification images (cross-session), so the pinned "
                         "test set holds its training images; use p2 / p3 for it")
        return notes
    if mode == "p2":
        if train_phases is None:
            return ["how this checkpoint was trained is unknown (no results_<tag>.json next to it); p2 is valid "
                    "only for a model trained with 02 --protocol cross_session"]
        if not train_phases:
            return ["this checkpoint predates the cross-session protocol (its results_<tag>.json has no "
                    "images_by_phase), so it was trained on the random split, which holds about 60 % of these "
                    "verification images: p2 is inflated for it"]
        if train_phases.get("verification"):
            return [f"this checkpoint was trained on {train_phases['verification']} verification images, so p2 "
                    "is not cross-session for it and is inflated"]
    if mode == "x1":
        return [X1_CAVEAT.format(name=external_name)] + ([X1_ALONSO_CAVEAT] if x1_protocol == "alonso2025" else [])
    return []


def protocol_text(mode, args):
    """The protocol description of a mode (x1: of the --x1-protocol in use)."""
    if mode == "x1" and getattr(args, "x1_protocol", "ours") == "alonso2025":
        return X1_ALONSO_PROTOCOL
    return PROTOCOLS[mode]


ALONSO_COLS = ("cam_distance", "pose", "gaze", "image_idx")


def alonso2025_table(manifest, rule="closest"):
    """x1 image table of --x1-protocol alonso2025: the usable images of external_table, then one frontal image
    per class, session and distance of the persons with both sessions (alonso2025_select), with the distance index
    D1 = 8 m ... D5 = 4 m and flip = True for the eye 'L'. Returns (table, number of unusable rows dropped)."""
    table, unusable = external_table(manifest, ALONSO_COLS)
    other = sorted(set(table.cam_distance.unique().tolist()) - set(ALONSO2025_DISTANCES))
    if other:
        raise SystemExit(f"{manifest}: camera distances {other} are not UBIPr's {list(ALONSO2025_DISTANCES)} m")
    keep = alonso2025_select(table.subject, table.person, np.where(table.phase == "enrollment", 1, 2),
                             table.cam_distance, table.pose, table.gaze, table.image_idx, rule)
    table = table[keep].reset_index(drop=True)
    table["distance_index"] = table.cam_distance.map({d: k + 1 for k, d in enumerate(ALONSO2025_DISTANCES)})
    table["flip"] = (table.eye == "L").to_numpy()
    return table, unusable


def select_images(modes, args, seen):
    """Image table of every requested mode (subjects, phases and eyes parsed from relpaths)."""
    selected = {}
    if "p1" in modes:
        selected["p1"] = image_table(pd.read_csv(args.test_list).relpath)
    if "p2" in modes or ("p3" in modes and not args.p3_list):
        if not os.path.exists(args.manifest):
            raise SystemExit(f"{args.manifest} not found: run scripts/01_build_manifest.py first or pass --manifest")
        table = image_table(pd.read_csv(args.manifest).relpath)
        if "p2" in modes:
            selected["p2"] = table[table.subject.isin(seen) if seen else in_paper_classes(table.subject)]
    if "p3" in modes:
        table = image_table(pd.read_csv(args.p3_list).relpath) if args.p3_list else table
        unseen = ~in_paper_classes(table.subject) & ~table.subject.isin(seen or set())
        if args.p3_list and not unseen.all():
            raise SystemExit(f"{args.p3_list} holds {int((~unseen).sum())} images of training subjects; "
                             "p3 must be open-set (use 02 --protocol open_set)")
        selected["p3"] = table[unseen]
    if "x1" in modes and getattr(args, "x1_protocol", "ours") == "alonso2025":
        selected["x1"], args.external_unusable = alonso2025_table(args.external_manifest,
                                                                  getattr(args, "x1_frontal", "closest"))
    elif "x1" in modes:                   # the external dataset: its own manifest and root
        selected["x1"], args.external_unusable = external_table(args.external_manifest,
                                                                external_subset_cols(args))
    for mode, df in selected.items():
        if args.max_subjects:
            col = "person" if mode == "x1" else "subject"     # x1: keep both eyes of each person
            df = df[df[col].isin(sorted(df[col].unique())[:args.max_subjects])]
        selected[mode] = df.reset_index(drop=True)
    return selected


def external_subset_cols(args):
    """Manifest columns that split x1 into subsets (--external-subsets)."""
    return tuple(c.strip() for c in (getattr(args, "external_subsets", None) or "").split(",") if c.strip())


def embed_images(model, relpaths, root, batch_size, workers, region_of=None, flips=None):
    """Embeddings (N, D) of root/<relpath>, in order, with progress and a time estimate.

    region_of: for a model with a region input (attnet_gate), a function relpaths -> (n, 8, 8, 1) regions.
    flips: optional (N,) bool, True for the images to mirror left-right before embedding.
    """
    emb, t0 = None, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(relpaths), batch_size):
            batch = relpaths[start:start + batch_size]
            paths = [os.path.join(root, r) for r in batch]
            if flips is None:
                images = np.stack(list(pool.map(load_image, paths)))
            else:
                images = np.stack(list(pool.map(load_image, paths, flips[start:start + batch_size])))
            if region_of is None:
                e = embed(model, images, batch_size=batch_size)
            else:
                e = embed_regions(model, images, region_of(batch), batch_size=batch_size)
            if emb is None:
                emb = np.zeros((len(relpaths), e.shape[1]), np.float32)
            emb[start:start + len(e)] = e
            done = start + len(e)
            if (start // batch_size) % 50 == 0 or done == len(relpaths):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  embedded {done}/{len(relpaths)} images | {(time.time() - t0) / 60:.1f} min | "
                      f"about {(len(relpaths) - done) / rate / 60:.1f} min left", flush=True)
    return emb


X1_NO_REGION = ("x1 is not available for attnet_gate: its region input needs eye masks, and the masks of 17 / 21 "
                "are keyed by VISOB relpaths; nothing makes them for the external (UBIPr) images")


def region_lookup(model, region_masks, relpaths, external=False):
    """None for a dual-input model; for attnet_gate a function relpaths -> region inputs (n, 8, 8, 1)
    from --region-masks, after checking that every image to embed has a mask. *external*: x1 is
    requested, which a model with a region input cannot run (no masks for the external images)."""
    if len(model.inputs) == 2:
        if region_masks:
            print(f"NOTE: --region-masks is ignored: this {model_arch(model)} model has no region input")
        return None
    if external:
        raise SystemExit(X1_NO_REGION)
    if not region_masks:
        raise SystemExit(f"this {model_arch(model)} model has a region input: pass --region-masks")
    from src.data.masked_generator import MaskTable, regions_for_rows
    table = MaskTable(region_masks)
    missing = [r for r, row in zip(relpaths, table.rows_for_relpaths(relpaths)) if row < 0]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(relpaths)} images have no mask in {region_masks}, "
                         f"e.g. {missing[:3]}")
    eye, grid = region_classes(model), feature_grid(model)[0]
    print(f"region input: eye-region fraction of {eye} per cell, from {region_masks}")
    return lambda batch: regions_for_rows(table, table.rows_for_relpaths(batch), eye, grid)


def subsets(images, cols=("device", "lighting")):
    """('all', mask), then one (name, mask) per value of each column (device and lighting for VISOB)."""
    out = [("all", np.ones(len(images), bool))]
    for col in cols:
        out += [(f"{col}={v}", (images[col] == v).to_numpy()) for v in sorted(images[col].unique())]
    return out


def score_row(genuine, impostor, args, pairs_twice=False):
    """Point metrics and bootstrap CIs of per-subject grid counts."""
    row = metrics_from_counts(genuine.sum(0), impostor.sum(0))
    if pairs_twice:                       # pair_counts counts every unordered pair once per image
        row["n_genuine"] //= 2
        row["n_impostor"] //= 2
    for name, (low, high) in bootstrap_metrics(genuine, impostor, n_boot=args.n_boot, seed=args.seed).items():
        row[f"{name}_ci_low"], row[f"{name}_ci_high"] = low, high
    return row


def operating_points(row, genuine, impostor, global_thr, threshold):
    """FAR / FRR of this subset at the thresholds set on the whole mode (and at --threshold)."""
    g, i = genuine.sum(0), impostor.sum(0)
    for key, thr in global_thr.items():
        row[f"frr_at_global_thr_{key}"], row[f"far_at_global_thr_{key}"] = error_rates_from_counts(g, i, thr)
    if threshold is not None:
        row["threshold"] = threshold
        row["frr_at_threshold"], row["far_at_threshold"] = error_rates_from_counts(g, i, threshold)


def curve(genuine, impostor, row):
    far, tar, _ = roc_from_counts(genuine.sum(0), impostor.sum(0))
    return {"far": far, "frr": 1.0 - tar, "eer": row["eer"], "tar_at_far_1e-3": row["tar_at_far_1e-3"]}


def evaluate_pairs(images, emb, args):
    """p1: every same-eye pair within each subset. Returns (rows, curve, score sample, extra)."""
    rows, global_thr, out_curve = [], {}, None
    for name, mask in subsets(images):
        sub = images[mask]
        counts = pair_counts(emb[sub.row.to_numpy()], sub.subject.to_numpy(), sub.eye.to_numpy())
        row = {"subset": name, "subset_rule": "both images", "n_subjects": len(counts["subjects"]),
               "n_probes": len(sub), **score_row(counts["genuine"], counts["impostor"], args, pairs_twice=True)}
        row["rank1"], row["rank1_ci_low"], row["rank1_ci_high"] = rank1_ci(
            counts["nn_correct"], sub.subject.to_numpy(), n_boot=args.n_boot, seed=args.seed)
        row["rank1_method"] = "leave-one-out nearest neighbour"
        if name == "all":
            global_thr = {far_key(f): row[f"thr_at_far_{far_key(f)}"] for f in FARS}
            out_curve = curve(counts["genuine"], counts["impostor"], row)
        operating_points(row, counts["genuine"], counts["impostor"], global_thr, args.threshold)
        rows.append(row)
        print(f"  p1 {name}: {len(sub)} images | EER {row['eer']:.4f} | TAR@FAR=1e-3 {row['tar_at_far_1e-3']:.4f}",
              flush=True)
    return rows, out_curve, sample_pair_scores(images, emb, args.sample_scores, args.seed), {}


def evaluate_templates(images, emb, args, mode, external=False):
    """p2 / p3 / x1: enrollment templates vs verification probes. Returns (rows, curve, score sample, extra).

    external (x1): a probe's comparisons with the other classes of its person are left out, persons are the
    bootstrap clusters and n_subjects, and the subsets come from --external-subsets.
    """
    enrol, probes = images[images.phase == "enrollment"], images[images.phase == "verification"]
    templates, t_subjects, t_eyes = build_templates(emb[enrol.row.to_numpy()], enrol.subject.to_numpy(),
                                                    enrol.eye.to_numpy())
    have = set(zip(t_subjects.tolist(), t_eyes.tolist()))
    keep = np.array([(s, e) in have for s, e in zip(probes.subject, probes.eye)], bool)
    probes = probes[keep].reset_index(drop=True)
    if len(probes) == 0 or len(templates) == 0:
        raise SystemExit(f"{mode}: no verification probe has an enrollment template of its eye")
    scores, own = template_scores(templates, t_subjects, emb[probes.row.to_numpy()], probes.subject.to_numpy(),
                                  t_eyes, probes.eye.to_numpy())
    clusters, subset_cols, n_same_person = probes.subject.to_numpy(), ("device", "lighting"), 0
    if external:
        person_of = dict(zip(enrol.subject, enrol.person))
        scores, n_same_person = exclude_same_person(scores, own, probes.person.to_numpy(),
                                                    np.array([person_of[s] for s in t_subjects]))
        clusters, subset_cols = probes.person.to_numpy(), external_subset_cols(args)
    probe_ranks = ranks(scores, own)
    rows, global_thr, out_curve = [], {}, None
    for name, mask in subsets(probes, subset_cols):
        subjects = clusters[mask]
        counts = template_counts(scores[mask], own[mask], subjects)
        row = {"subset": name, "subset_rule": "probe", "n_subjects": len(counts["subjects"]),
               "n_probes": int(mask.sum()), "n_templates": len(templates),
               **score_row(counts["genuine"], counts["impostor"], args)}
        row["rank1"], row["rank1_ci_low"], row["rank1_ci_high"] = rank1_ci(
            (probe_ranks[mask] == 1).astype(np.float64), subjects, n_boot=args.n_boot, seed=args.seed)
        row["rank1_method"] = "own template among the templates of the probe's eye"
        if external:
            row["rank1_method"] += " (other classes of the probe's person left out)"
        if name == "all":
            global_thr = {far_key(f): row[f"thr_at_far_{far_key(f)}"] for f in FARS}
            out_curve = curve(counts["genuine"], counts["impostor"], row)
        operating_points(row, counts["genuine"], counts["impostor"], global_thr, args.threshold)
        rows.append(row)
        print(f"  {mode} {name}: {int(mask.sum())} probes | EER {row['eer']:.4f} | "
              f"TAR@FAR=1e-3 {row['tar_at_far_1e-3']:.4f} | rank-1 {row['rank1']:.4f}", flush=True)
    extra = {"n_enrollment_images": len(enrol), "n_probes_without_template": int((~keep).sum()),
             "cmc_rank_1_to_20": [round(float(v), 6) for v in cmc(probe_ranks, 20)]}
    if external:
        extra.update({"n_classes_with_template": len(templates), "n_probe_classes": int(probes.subject.nunique()),
                      "n_persons": int(probes.person.nunique()),
                      "n_same_person_comparisons_left_out": n_same_person,
                      "n_unusable_images_left_out": int(getattr(args, "external_unusable", 0)),
                      "subsets": list(subset_cols), "bootstrap_clusters": "persons"})
    sample_cols = ("relpath", "subject", "person", "eye") + tuple(subset_cols) if external else None
    sample = sample_template_scores(probes, templates, t_subjects, t_eyes, scores, own, args.sample_scores,
                                    args.seed, mode, cols=sample_cols)
    return rows, out_curve, sample, extra


def distance_label(i):
    """'D1(8m)' for the distance index i of the protocol of Alonso-Fernandez et al. 2025."""
    return f"D{i}({ALONSO2025_DISTANCES[i - 1]}m)"


def alonso2025_counts(images, pairs):
    """Images and comparisons of --x1-protocol alonso2025 next to the paper's (for the summary JSON)."""
    n_persons = int(images.person.nunique())
    genuine, intra = pairs["genuine"], pairs["di"] == pairs["dj"]
    nominal_g, nominal_i = alonso2025_nominal_counts(n_persons)
    combos = {}
    for i, j in sorted(set(zip(pairs["di"].tolist(), pairs["dj"].tolist()))):
        at = (pairs["di"] == i) & (pairs["dj"] == j)
        combos[f"{distance_label(i)}-{distance_label(j)}"] = {"genuine": int((at & genuine).sum()),
                                                              "impostor": int((at & ~genuine).sum())}
    cells = images.groupby(["subject", "phase", "cam_distance"]).size()
    return {"persons": n_persons, "classes": int(images.subject.nunique()), "images": len(images),
            "images_by_session": {"1": int((images.phase == "enrollment").sum()),
                                  "2": int((images.phase == "verification").sum())},
            "images_by_cam_distance": {str(k): int(v)
                                       for k, v in images.cam_distance.value_counts().sort_index().items()},
            "images_annotated_frontal": int(((images.pose == 0) & (images.gaze == 0)).sum()),
            "images_flipped": int(images.flip.sum()),
            "eye_class_session_distance_cells_missing": 4 * len(ALONSO2025_DISTANCES) * n_persons - len(cells),
            "genuine": int(genuine.sum()), "genuine_intra": int((genuine & intra).sum()),
            "genuine_cross": int((genuine & ~intra).sum()), "impostor": int((~genuine).sum()),
            "by_combination": combos,
            "nominal_for_these_persons": {"genuine": nominal_g, "impostor": nominal_i,
                                          "rule": "every person with all 20 images: 4 genuine per person and "
                                                  "distance, 8 per person and cross-distance pair, 4 impostor "
                                                  "per ordered pair of persons and combination"},
            "paper": dict(ALONSO2025_PAPER),
            "matches_paper": bool(int(genuine.sum()) == ALONSO2025_PAPER["genuine"]
                                  and int((~genuine).sum()) == ALONSO2025_PAPER["impostor"])}


def evaluate_alonso2025(images, emb, args, mode="x1"):
    """x1 with --x1-protocol alonso2025: single-image comparisons of Alonso-Fernandez et al. 2025.

    Every comparison is counted for the person of its session-1 image (the bootstrap clusters). Subsets: all,
    intra (Di = Dj), cross (Di != Dj) and each distance combination present. Returns (rows, curve, score sample,
    extra).
    """
    pairs = alonso2025_pairs(images.person.to_numpy(), np.where(images.phase == "enrollment", 1, 2),
                             images.distance_index.to_numpy())
    if not pairs["genuine"].any() or pairs["genuine"].all():
        raise SystemExit(f"{mode} (alonso2025): no genuine or no impostor comparison; check the manifest")
    rows_of = images.row.to_numpy()
    similarity = emb[rows_of] @ emb[rows_of].T             # (n_images, n_images): a few MB, unlike per-pair rows
    scores = similarity[pairs["first"], pairs["second"]].astype(np.float32)
    clusters = images.person.to_numpy()[pairs["first"]]
    intra = pairs["di"] == pairs["dj"]
    groups = [("all", np.ones(len(scores), bool)), ("intra", intra), ("cross", ~intra)]
    for i, j in sorted(set(zip(pairs["di"].tolist(), pairs["dj"].tolist()))):
        groups.append((f"{distance_label(i)}-{distance_label(j)}", (pairs["di"] == i) & (pairs["dj"] == j)))
    rows, global_thr, out_curve = [], {}, None
    for name, mask in groups:
        counts = pair_list_counts(scores[mask], pairs["genuine"][mask], clusters[mask])
        row = {"subset": name, "subset_rule": "distance combination (session-1 image at Di, other image at Dj)",
               "n_subjects": len(counts["subjects"]), "n_probes": int(np.unique(pairs["first"][mask]).size),
               **score_row(counts["genuine"], counts["impostor"], args)}
        row["rank1"] = row["rank1_ci_low"] = row["rank1_ci_high"] = float("nan")
        row["rank1_method"] = "not defined: single-image pairs, no gallery"
        if name == "all":
            global_thr = {far_key(f): row[f"thr_at_far_{far_key(f)}"] for f in FARS}
            out_curve = curve(counts["genuine"], counts["impostor"], row)
        operating_points(row, counts["genuine"], counts["impostor"], global_thr, args.threshold)
        rows.append(row)
        print(f"  {mode} alonso2025 {name}: {row['n_genuine']} genuine / {row['n_impostor']} impostor | "
              f"EER {row['eer']:.4f} | TAR@FAR=1e-3 {row['tar_at_far_1e-3']:.4f}", flush=True)
    pair_info = alonso2025_counts(images, pairs)
    extra = {"x1_protocol": "alonso2025", "frontal_rule": getattr(args, "x1_frontal", "closest"),
             "n_persons": pair_info["persons"],
             "n_unusable_images_left_out": int(getattr(args, "external_unusable", 0)),
             "bootstrap_clusters": "persons (of the session-1 image)", "pair_counts": pair_info}
    print(f"  {mode} alonso2025: {pair_info['persons']} persons, {pair_info['images']} images, "
          f"{pair_info['genuine']} genuine / {pair_info['impostor']} impostor comparisons (paper: "
          f"{ALONSO2025_PAPER['persons']} persons, {ALONSO2025_PAPER['images']} images, "
          f"{ALONSO2025_PAPER['genuine']} / {ALONSO2025_PAPER['impostor']})", flush=True)
    sample = sample_pair_list_scores(images, pairs, scores, args.sample_scores, args.seed, mode)
    return rows, out_curve, sample, extra


def sample_pair_list_scores(images, pairs, scores, n, seed, mode):
    """Uniform samples of the genuine and impostor comparisons of an explicit pair list, with their scores."""
    rng = np.random.default_rng(seed)
    relpath, subject = images.relpath.to_numpy(), images.subject.to_numpy()
    cols = ("relpath", "subject", "person", "eye", "cam_distance")
    frames = []
    for kind, is_kind in (("genuine", pairs["genuine"]), ("impostor", ~pairs["genuine"])):
        idx = np.flatnonzero(is_kind)
        pick = np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
        a, b = pairs["first"][pick], pairs["second"][pick]
        frame = _sample_frame(mode, kind, scores[pick], images, a, relpath[b], subject[b], cols)
        frame["combination"] = [f"{distance_label(i)}-{distance_label(j)}"
                                for i, j in zip(pairs["di"][pick], pairs["dj"][pick])]
        frames.append(frame)
    return pd.concat(frames)


def _sample_frame(mode, kind, score, images, first, other, other_subject, cols=None):
    """Sampled comparisons: the first image's metadata, what it was compared with, and the score."""
    cols = {c: images[c].to_numpy()[first] for c in (cols or ("relpath", "subject", "eye", "device", "lighting"))}
    return pd.DataFrame({"mode": mode, "kind": kind, "score": np.round(score, 6), **cols, "other": other,
                         "other_subject": other_subject})


def _impostor_draws(n, size_a, size_b, valid, rng):
    """Up to n uniform (a, b) index pairs for which valid(a, b) holds (rejection sampling)."""
    a_all, b_all = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for attempt in range(100):
        if sum(map(len, a_all)) >= n:
            break
        a, b = rng.integers(0, size_a, 4 * n + 16), rng.integers(0, size_b, 4 * n + 16)
        ok = valid(a, b)
        if not ok.any() and attempt >= 2:          # e.g. a single subject: no impostor pair exists
            break
        a_all.append(a[ok])
        b_all.append(b[ok])
    return np.concatenate(a_all)[:n], np.concatenate(b_all)[:n]


def sample_pair_scores(images, emb, n, seed):
    """Uniform samples of genuine and impostor same-eye pairs of p1 with their scores."""
    rng = np.random.default_rng(seed)
    subject, eye, rows = images.subject.to_numpy(), images.eye.to_numpy(), images.row.to_numpy()
    first, second = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for idx in images.groupby(["subject", "eye"]).indices.values():
        i, j = np.triu_indices(len(idx), k=1)
        first.append(idx[i])
        second.append(idx[j])
    first, second = np.concatenate(first), np.concatenate(second)
    pick = rng.choice(len(first), min(n, len(first)), replace=False)
    gi, gj = first[pick], second[pick]
    ii, ij = _impostor_draws(n, len(images), len(images),
                             lambda a, b: (eye[a] == eye[b]) & (subject[a] != subject[b]), rng)
    relpath = images.relpath.to_numpy()
    frames = []
    for kind, a, b in (("genuine", gi, gj), ("impostor", ii, ij)):
        score = np.sum(emb[rows[a]] * emb[rows[b]], axis=1)
        frames.append(_sample_frame("p1", kind, score, images, a, relpath[b], subject[b]))
    return pd.concat(frames)


def sample_template_scores(probes, templates, t_subjects, t_eyes, scores, own, n, seed, mode, cols=None):
    """Uniform samples of genuine and impostor probe-template scores of p2 / p3 / x1 (cols: probe metadata)."""
    rng = np.random.default_rng(seed)
    gp = rng.choice(len(probes), min(n, len(probes)), replace=False)
    gt = own.argmax(axis=1)[gp]                                  # the probe's own template
    ip, it = _impostor_draws(n, len(probes), len(templates),
                             lambda a, b: ~np.isnan(scores[a, b]) & ~own[a, b], rng)
    frames = []
    for kind, p, t in (("genuine", gp, gt), ("impostor", ip, it)):
        names = [f"template {s}/{e}" for s, e in zip(t_subjects[t], t_eyes[t])]
        frames.append(_sample_frame(mode, kind, scores[p, t], probes, p, names, t_subjects[t], cols))
    return pd.concat(frames)


def plot_curves(curves, labels, path):
    """ROC (TAR against FAR, log scale) and DET (FRR against FAR, normal-deviate scale) of every mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, (roc_ax, det_ax) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=INK["surface"])
    ticks = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5])
    tick_labels = ["0.01%", "0.1%", "1%", "5%", "20%", "50%"]
    lo, hi = norm.ppf(ticks[0]), norm.ppf(ticks[-1])
    for mode, c in curves.items():
        far, frr = c["far"], c["frr"]
        color = COLORS[mode]
        roc_ax.plot(far, 1 - frr, color=color, lw=1.6, solid_joinstyle="round", label=labels[mode])
        k = int(np.searchsorted(far, 1e-3 * (1 + 1e-9), side="right")) - 1
        roc_ax.plot([max(far[k], 1e-5)], [1 - frr[k]], "o", ms=7, color=color, mec=INK["surface"], mew=1.5)
        inside = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
        det_ax.plot(norm.ppf(far[inside]), norm.ppf(frr[inside]), color=color, lw=1.6, solid_joinstyle="round")
        if np.isfinite(c["eer"]) and c["eer"] > 0:
            z = float(np.clip(norm.ppf(c["eer"]), lo, hi))
            det_ax.plot([z], [z], "o", ms=7, color=color, mec=INK["surface"], mew=1.5)
            # to the right of the point: the curve and the diagonal both cross it at about +-45 degrees
            det_ax.annotate(f"{mode.upper()} EER {100 * c['eer']:.2f}%", (z, z), xytext=(9, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color=INK["primary"],
                            bbox={"boxstyle": "square,pad=0.15", "fc": INK["surface"], "ec": "none"})
    roc_ax.set_xscale("log", nonpositive="mask")
    roc_ax.set_xlim(1e-5, 1)
    roc_ax.set_ylim(0, 1.01)
    roc_ax.set_xlabel("false accept rate (FAR)")
    roc_ax.set_ylabel("true accept rate (TAR)")
    roc_ax.set_title("ROC (dots: TAR at FAR = 0.1%)", loc="left", fontsize=10.5, color=INK["primary"])
    det_ax.plot([lo, hi], [lo, hi], color=INK["axis"], lw=0.8, zorder=0)                # FAR = FRR
    det_ax.set_xlim(lo, hi)
    det_ax.set_ylim(lo, hi)
    det_ax.set_xticks(norm.ppf(ticks), tick_labels)
    det_ax.set_yticks(norm.ppf(ticks), tick_labels)
    det_ax.set_xlabel("false accept rate (FAR)")
    det_ax.set_ylabel("false reject rate (FRR)")
    det_ax.set_title("DET (normal-deviate axes; diagonal: FAR = FRR)", loc="left", fontsize=10.5,
                     color=INK["primary"])
    for ax in (roc_ax, det_ax):
        ax.set_facecolor(INK["surface"])
        ax.grid(True, which="major", color=INK["grid"], lw=0.6, ls="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK["axis"])
        ax.tick_params(colors=INK["muted"], labelsize=8.5)
        ax.xaxis.label.set_color(INK["secondary"])
        ax.yaxis.label.set_color(INK["secondary"])
    fig.legend(*roc_ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 1.0),
               ncol=1, frameon=False, fontsize=8.5, labelcolor=INK["primary"])
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.048 * len(curves)])
    fig.savefig(path, dpi=200, facecolor=INK["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modes", default="p1,p3",
                    help="comma-separated: p1 (all pairs of the pinned test set; inflated, reproduction only), "
                         "p2 (templates vs probes, training subjects; for cross-session models), p3 (templates vs "
                         "probes, unseen subjects), x1 (zero-shot cross-dataset: session-1 templates vs session-2 "
                         "probes of --external-manifest)")
    ap.add_argument("--manifest", default="/kaggle/working/manifest_raw.csv",
                    help="manifest_raw.csv from 01_build_manifest.py (needed by p2, and by p3 without --p3-list)")
    ap.add_argument("--data-root", default=None, help="VISOB root of the relpaths (default: found under /kaggle/input)")
    ap.add_argument("--test-list", default=DEFAULT_TEST_LIST, help="p1 images: CSV with a relpath column")
    ap.add_argument("--p3-list", default=None,
                    help="p3 images: CSV with a relpath column, e.g. open_set_test.csv from 02 --protocol open_set "
                         "(default: every manifest image of a subject outside the 300 classes)")
    ap.add_argument("--class-index", default=None,
                    help="class_index_<tag>.json from 03 (default: next to the checkpoint if present, else subject "
                         "IDs 1141-3279 plus 7000)")
    ap.add_argument("--external-manifest", default=None,
                    help="x1 images: ubipr_manifest.csv from 20_prepare_ubipr.py (relpath, class_id, person_id, eye, "
                         "session, usable, subset columns)")
    ap.add_argument("--external-root", default=None,
                    help="root of the external relpaths (default: the folder of --external-manifest)")
    ap.add_argument("--external-name", default="ubipr", help="name of the external dataset in outputs and notes")
    ap.add_argument("--external-subsets", default="cam_distance,glasses",
                    help="comma-separated manifest columns that split x1 into subsets (by the probe; "
                         "--x1-protocol ours only)")
    ap.add_argument("--x1-protocol", choices=X1_PROTOCOLS, default="ours",
                    help="x1 pairing: ours (session-1 class templates vs session-2 probes, same eye label; the "
                         "default) or alonso2025 (Alonso-Fernandez et al. 2025: single frontal images, session 1 vs "
                         "session 2 within and across distances, both eyes one identity, 'L' crops flipped)")
    ap.add_argument("--x1-frontal", choices=("closest", "strict"), default="closest",
                    help="alonso2025: the frontal image per eye, session and distance: closest (smallest |pose| + "
                         "|gaze|) or strict (annotated pose 0 and gaze 0 only)")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="smoke run: keep the first N subjects of each mode (x1: the first N persons)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="threads that read images")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the CIs (0: no CIs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="also report FAR and FRR at this score threshold (e.g. one set on another protocol)")
    ap.add_argument("--sample-scores", type=int, default=20000, help="genuine and impostor scores saved per mode")
    ap.add_argument("--save-embeddings", action="store_true", help="also write embeddings_<tag>.npz (float16)")
    ap.add_argument("--region-masks", default=None,
                    help="attnet_gate only: masks64_<tag>.npz (21) or masks_<tag>.npz (17) for the region input")
    ap.add_argument("--out-dir", default="/kaggle/working/verification")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    t0 = time.time()
    requested = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    unknown = requested - set(PROTOCOLS)
    if unknown or not requested:
        raise SystemExit(f"unknown modes {sorted(unknown)}; choose from {sorted(PROTOCOLS)}")
    modes = [m for m in PROTOCOLS if m in requested]
    if "x1" in modes and args.region_masks:
        ap.error(X1_NO_REGION + " (--region-masks is for attnet_gate, which runs p1, p2 and p3 only)")
    if "x1" in modes and not args.external_manifest:
        raise SystemExit("x1 needs --external-manifest (ubipr_manifest.csv from 20_prepare_ubipr.py)")
    if "x1" in modes and not os.path.exists(args.external_manifest):
        raise SystemExit(f"{args.external_manifest} not found: run scripts/20_prepare_ubipr.py first")
    external_root = args.external_root or os.path.dirname(os.path.abspath(args.external_manifest or "."))

    gpus = tf.config.list_physical_devices("GPU")
    print("TF", tf.__version__, "| tf.keras comes from:", tf.keras.Model.__module__, "| GPUs:", len(gpus))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("memory growth:", e)

    seen, seen_source, train_phases = training_info(args.ckpt, args.class_index)
    described = train_phases or ("unknown: no results_<tag>.json" if train_phases is None
                                 else "not recorded: trained before the cross-session protocol existed")
    print(f"training subjects: {len(seen) if seen else 300} from {seen_source} | training images by phase: "
          f"{described}")
    visob = [m for m in modes if m in VISOB_MODES]
    root = (args.data_root or find_root()) if visob else args.data_root      # x1 alone needs no VISOB
    selected = select_images(modes, args, seen)
    for mode, df in selected.items():
        if df.empty:
            raise SystemExit(f"{mode}: no images selected (check --manifest, --test-list, --p3-list"
                             f"{', --external-manifest' if mode == 'x1' else ''})")
        print(f"{mode}: {len(df)} images of {df.subject.nunique()} subjects "
              f"({int((df.phase == 'enrollment').sum())} enrollment, {int((df.phase == 'verification').sum())} "
              f"verification){' | smoke run' if args.max_subjects else ''}")
        if mode == "x1":
            print(f"  x1: {args.external_name} from {external_root}: {df.person.nunique()} persons, subjects are "
                  f"classes (one eye each); {getattr(args, 'external_unusable', 0)} unusable images left out")
        if mode == "x1" and args.x1_protocol == "alonso2025":
            print(f"  x1 protocol: alonso2025 (frontal rule {args.x1_frontal}): {len(df)} images, "
                  f"{int(df.flip.sum())} flipped ('L'); --external-subsets is not used")
        for note in caveats(mode, train_phases, args.external_name, args.x1_protocol):
            print(f"  NOTE ({mode}): {note}")

    timing, t = {}, time.time()
    model = load_checkpoint(args.ckpt)
    timing["load_model_s"] = round(time.time() - t, 1)
    print(f"loaded: {args.ckpt} ({timing['load_model_s']} s) | arch: {model_arch(model)}")
    relpaths = sorted(set().union(*(set(selected[m].relpath) for m in visob)))
    region_of = region_lookup(model, args.region_masks, relpaths, external="x1" in selected)
    emb = emb_x = None
    if relpaths:
        print(f"embedding {len(relpaths)} images (batch {args.batch_size}) from {root}", flush=True)
        t = time.time()
        emb = embed_images(model, relpaths, root, args.batch_size, args.workers, region_of)
        timing["embedding_s"] = round(time.time() - t, 1)
        row_of = {r: i for i, r in enumerate(relpaths)}
        for mode in visob:
            selected[mode]["row"] = selected[mode].relpath.map(row_of).to_numpy()
    if "x1" in selected:                  # external images: their own root and embedding matrix
        ext_relpaths = selected["x1"].relpath.tolist()           # unique and sorted (external_table)
        print(f"embedding {len(ext_relpaths)} {args.external_name} images (batch {args.batch_size}) from "
              f"{external_root}", flush=True)
        t = time.time()
        flips = selected["x1"].flip.to_numpy() if args.x1_protocol == "alonso2025" else None
        emb_x = embed_images(model, ext_relpaths, external_root, args.batch_size, args.workers, region_of, flips)
        timing["embedding_x1_s"] = round(time.time() - t, 1)
        selected["x1"]["row"] = np.arange(len(ext_relpaths))

    all_rows, curves, samples, per_mode = [], {}, [], {}
    for mode in modes:
        t = time.time()
        if mode == "p1":
            rows, c, sample, extra = evaluate_pairs(selected[mode], emb, args)
        elif mode == "x1" and args.x1_protocol == "alonso2025":
            rows, c, sample, extra = evaluate_alonso2025(selected[mode], emb_x, args, mode)
        elif mode == "x1":
            rows, c, sample, extra = evaluate_templates(selected[mode], emb_x, args, mode, external=True)
        else:
            rows, c, sample, extra = evaluate_templates(selected[mode], emb, args, mode)
        timing[f"{mode}_s"] = round(time.time() - t, 1)
        notes = caveats(mode, train_phases, args.external_name, args.x1_protocol)
        overall = rows[0]
        for target in FARS:
            key = far_key(target)
            if overall[f"far_achieved_{key}"] * overall["n_impostor"] < 30:
                notes.append(f"TAR at FAR {key} rests on fewer than 30 accepted impostor comparisons (rule of 30)")
        for row in rows:
            row.update({"mode": mode, "protocol": protocol_text(mode, args), "caveat": "; ".join(notes)})
        all_rows += rows
        curves[mode] = c
        samples.append(sample)
        per_mode[mode] = {"protocol": protocol_text(mode, args), "caveats": notes, "n_images": len(selected[mode]),
                          **{k: overall[k] for k in overall if k not in ("mode", "protocol", "caveat", "subset")},
                          **extra, "seconds": timing[f"{mode}_s"]}

    keys = [far_key(f) for f in FARS]
    columns = (["mode", "subset", "subset_rule", "n_subjects", "n_probes", "n_templates", "n_genuine", "n_impostor",
                "eer", "eer_ci_low", "eer_ci_high", "eer_threshold"]
               + [f"{p}{k}{s}" for k in keys for p, s in (("tar_at_far_", ""), ("tar_at_far_", "_ci_low"),
                                                        ("tar_at_far_", "_ci_high"), ("thr_at_far_", ""),
                                                        ("far_achieved_", ""))]
               + ["rank1", "rank1_ci_low", "rank1_ci_high", "rank1_method"]
               + [f"{p}{k}" for k in keys for p in ("far_at_global_thr_", "frr_at_global_thr_")]
               + (["threshold", "far_at_threshold", "frr_at_threshold"] if args.threshold is not None else [])
               + ["protocol", "caveat"])
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {name: os.path.join(args.out_dir, f"{name}_{args.tag}.{ext}") for name, ext in
             (("verification_metrics", "csv"), ("verification_scores", "csv"), ("verification_roc_det", "png"),
              ("verification_summary", "json"), ("embeddings", "npz"))}
    pd.DataFrame(all_rows).reindex(columns=columns).to_csv(paths["verification_metrics"], index=False)
    pd.concat(samples).to_csv(paths["verification_scores"], index=False)
    labels = {m: f"{m.upper()} {'(inflated) ' if m == 'p1' else ''}{protocol_text(m, args).split(':')[0]}: "
                 f"EER {100 * per_mode[m]['eer']:.2f}%, TAR@FAR=0.1% {100 * per_mode[m]['tar_at_far_1e-3']:.1f}%, "
                 f"rank-1 {100 * per_mode[m]['rank1']:.1f}%" for m in modes}
    if "x1" in labels:
        labels["x1"] = labels["x1"].replace("X1 ", f"X1 ({args.external_name}) ", 1)
    plot_curves(curves, labels, paths["verification_roc_det"])
    if args.save_embeddings and emb is not None:
        np.savez_compressed(paths["embeddings"], embeddings=emb.astype(np.float16), relpaths=np.array(relpaths))
    if args.save_embeddings and emb_x is not None:
        np.savez_compressed(os.path.join(args.out_dir, f"embeddings_{args.external_name}_{args.tag}.npz"),
                            embeddings=emb_x.astype(np.float16), relpaths=np.array(selected["x1"].relpath.tolist()))

    timing["total_minutes"] = round((time.time() - t0) / 60, 1)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "git_commit": git_commit(), "modes": modes,
               "data_root": root, "manifest": args.manifest, "test_list": args.test_list, "p3_list": args.p3_list,
               "training_subjects": seen_source, "training_images_by_phase": train_phases,
               "max_subjects": args.max_subjects, "n_boot": args.n_boot, "seed": args.seed,
               "batch_size": args.batch_size, "score_grid_step": 1.0 / BINS_PER_UNIT,
               "n_images_embedded": len(relpaths), "embedding_dim": int((emb if emb is not None else emb_x).shape[1]),
               "comparisons": "same eye only; FAR = false match rate, FRR = false non-match rate",
               "per_mode": per_mode, "timing": timing}
    if "x1" in modes:                     # only with x1, so that p1 / p2 / p3 summaries keep their keys
        summary["external"] = {"name": args.external_name, "manifest": os.path.abspath(args.external_manifest),
                               "root": external_root, "subsets": list(external_subset_cols(args)),
                               "n_images_embedded": len(selected["x1"]),
                               "n_unusable_images_left_out": int(getattr(args, "external_unusable", 0)),
                               "genuine": "probe and template of the same class (one eye of one person)",
                               "impostor": "template of another person, same eye label; a probe's comparisons with "
                                           "other classes of its own person are left out"}
        if args.x1_protocol == "alonso2025":  # ours keeps exactly the keys and values above
            summary["external"].update({
                "protocol": "alonso2025", "frontal_rule": args.x1_frontal, "subsets": ["intra", "cross", "Di-Dj"],
                "reference": "Alonso-Fernandez et al. 2025, arXiv 2510.26282, Sect. III-IV",
                "identity": "person (both eyes); crops with eye 'L' (nose on the crop's right) flipped left-right",
                "genuine": "single images of the same person: session 1 at Di vs session 2 at Di, or vs both sessions "
                           "at Dj (i < j; D1 = 8 m ... D5 = 4 m)",
                "impostor": "session 1 at Di vs session 2 at Dj (i <= j) of every other person",
                "crops": "20_prepare_ubipr.py's corner-based VISOB-like crops, not the paper's 7.6 Rs squares "
                         "around the sclera centre (UBIPr's annotation holds no sclera boundary)",
                "pair_counts": per_mode["x1"]["pair_counts"]})
    with open(paths["verification_summary"], "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m in modes:
        r = per_mode[m]
        if m == "x1" and args.x1_protocol == "alonso2025":
            c = r["pair_counts"]
            print(f"X1 alonso2025 comparisons: {c['genuine']} genuine / {c['impostor']} impostor of {c['persons']} "
                  f"persons, {c['images']} images | paper: {c['paper']['genuine']} / {c['paper']['impostor']} of "
                  f"{c['paper']['persons']} persons, {c['paper']['images']} images | nominal for {c['persons']} "
                  f"complete persons: {c['nominal_for_these_persons']['genuine']} / "
                  f"{c['nominal_for_these_persons']['impostor']}")
        print(f"{m.upper()}: EER {r['eer']:.4f} [{r['eer_ci_low']:.4f}, {r['eer_ci_high']:.4f}] | "
              f"TAR@FAR=1e-2 {r['tar_at_far_1e-2']:.4f} | TAR@FAR=1e-3 {r['tar_at_far_1e-3']:.4f} | "
              f"rank-1 {r['rank1']:.4f} | {r['n_subjects']} subjects, {r['n_genuine']} genuine / "
              f"{r['n_impostor']} impostor comparisons")
    print(f"wrote results to {args.out_dir} ({timing['total_minutes']} min)")


if __name__ == "__main__":
    main()
