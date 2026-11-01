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

Usage (Kaggle, checkpoint dataset and VISOB attached, 01 already run):
  python scripts/15_eval_verification.py \
      --ckpt /kaggle/input/<checkpoint-dataset>/Xception_ArcFace_full.keras \
      --manifest /kaggle/working/manifest_raw.csv --modes p1,p3 --tag full
  smoke run (2-3 min): add --max-subjects 10 --n-boot 100 --tag smoke
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
from src.eval.verification import (BINS_PER_UNIT, FARS, bootstrap_metrics, build_templates, cmc,  # noqa: E402
                                   embed, error_rates_from_counts, far_key, metrics_from_counts, pair_counts,
                                   rank1_ci, ranks, roc_from_counts, template_counts, template_scores)
from src.models.arcface_inference import load_checkpoint  # noqa: E402

DEFAULT_TEST_LIST = os.path.join(REPO, "splits", "visob_test.csv")
RELPATH_RE = re.compile(r"^(?P<condition>[^/]+)/(?P<phase>enrollment|verification)_(?P<eye>left|right)_eye/"
                        r"[^/]+/(?P<subject>\d+)_\d+_[lr]_c\.png$", re.IGNORECASE)
PROTOCOLS = {
    "p1": "closed-set verification: all same-eye pairs of the pinned random-split test set",
    "p2": "closed-set cross-session verification: enrollment templates vs verification probes, training subjects",
    "p3": "open-set cross-session verification: enrollment templates vs verification probes, unseen subjects",
}
P1_CAVEAT = ("inflated by near-duplicate frames: consecutive frames of one bucket sit on both sides of a pair "
             "and in training; reproduces the base-paper protocol only")
COLORS = {"p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a"}      # fixed per protocol, never by rank
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


def in_paper_classes(subjects):
    """True for subject IDs 1141..3279 plus 7000, the 300 classes of 02_make_splits.py."""
    s = np.asarray(subjects)
    return ((s >= 1141) & (s <= 3279)) | (s == 7000)


def load_image(path):
    """(256, 256, 3) float32 in [0, 1], loaded like flow_from_directory does (nearest)."""
    img = tf.keras.utils.load_img(path, target_size=(256, 256))
    return np.asarray(img, dtype=np.float32) / 255.0


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


def caveats(mode, train_phases):
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
    return []


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
    for mode, df in selected.items():
        if args.max_subjects:
            df = df[df.subject.isin(sorted(df.subject.unique())[:args.max_subjects])]
        selected[mode] = df.reset_index(drop=True)
    return selected


def embed_images(model, relpaths, root, batch_size, workers):
    """Embeddings (N, D) of root/<relpath>, in order, with progress and a time estimate."""
    emb, t0 = None, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(relpaths), batch_size):
            paths = [os.path.join(root, r) for r in relpaths[start:start + batch_size]]
            e = embed(model, np.stack(list(pool.map(load_image, paths))), batch_size=batch_size)
            if emb is None:
                emb = np.zeros((len(relpaths), e.shape[1]), np.float32)
            emb[start:start + len(e)] = e
            done = start + len(e)
            if (start // batch_size) % 50 == 0 or done == len(relpaths):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  embedded {done}/{len(relpaths)} images | {(time.time() - t0) / 60:.1f} min | "
                      f"about {(len(relpaths) - done) / rate / 60:.1f} min left", flush=True)
    return emb


def subsets(images):
    """('all', mask), then one (name, mask) per device and per lighting."""
    out = [("all", np.ones(len(images), bool))]
    for col in ("device", "lighting"):
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


def evaluate_templates(images, emb, args, mode):
    """p2 / p3: enrollment templates vs verification probes. Returns (rows, curve, score sample, extra)."""
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
    probe_ranks = ranks(scores, own)
    rows, global_thr, out_curve = [], {}, None
    for name, mask in subsets(probes):
        subjects = probes.subject.to_numpy()[mask]
        counts = template_counts(scores[mask], own[mask], subjects)
        row = {"subset": name, "subset_rule": "probe", "n_subjects": len(counts["subjects"]),
               "n_probes": int(mask.sum()), "n_templates": len(templates),
               **score_row(counts["genuine"], counts["impostor"], args)}
        row["rank1"], row["rank1_ci_low"], row["rank1_ci_high"] = rank1_ci(
            (probe_ranks[mask] == 1).astype(np.float64), subjects, n_boot=args.n_boot, seed=args.seed)
        row["rank1_method"] = "own template among the templates of the probe's eye"
        if name == "all":
            global_thr = {far_key(f): row[f"thr_at_far_{far_key(f)}"] for f in FARS}
            out_curve = curve(counts["genuine"], counts["impostor"], row)
        operating_points(row, counts["genuine"], counts["impostor"], global_thr, args.threshold)
        rows.append(row)
        print(f"  {mode} {name}: {int(mask.sum())} probes | EER {row['eer']:.4f} | "
              f"TAR@FAR=1e-3 {row['tar_at_far_1e-3']:.4f} | rank-1 {row['rank1']:.4f}", flush=True)
    extra = {"n_enrollment_images": len(enrol), "n_probes_without_template": int((~keep).sum()),
             "cmc_rank_1_to_20": [round(float(v), 6) for v in cmc(probe_ranks, 20)]}
    sample = sample_template_scores(probes, templates, t_subjects, t_eyes, scores, own, args.sample_scores,
                                    args.seed, mode)
    return rows, out_curve, sample, extra


def _sample_frame(mode, kind, score, images, first, other, other_subject):
    """Sampled comparisons: the first image's metadata, what it was compared with, and the score."""
    cols = {c: images[c].to_numpy()[first] for c in ("relpath", "subject", "eye", "device", "lighting")}
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


def sample_template_scores(probes, templates, t_subjects, t_eyes, scores, own, n, seed, mode):
    """Uniform samples of genuine and impostor probe-template scores of p2 / p3."""
    rng = np.random.default_rng(seed)
    gp = rng.choice(len(probes), min(n, len(probes)), replace=False)
    gt = own.argmax(axis=1)[gp]                                  # the probe's own template
    ip, it = _impostor_draws(n, len(probes), len(templates),
                             lambda a, b: ~np.isnan(scores[a, b]) & ~own[a, b], rng)
    frames = []
    for kind, p, t in (("genuine", gp, gt), ("impostor", ip, it)):
        names = [f"template {s}/{e}" for s, e in zip(t_subjects[t], t_eyes[t])]
        frames.append(_sample_frame(mode, kind, scores[p, t], probes, p, names, t_subjects[t]))
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
                         "probes, unseen subjects)")
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
    ap.add_argument("--max-subjects", type=int, default=None, help="smoke run: keep the first N subjects of each mode")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="threads that read images")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the CIs (0: no CIs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="also report FAR and FRR at this score threshold (e.g. one set on another protocol)")
    ap.add_argument("--sample-scores", type=int, default=20000, help="genuine and impostor scores saved per mode")
    ap.add_argument("--save-embeddings", action="store_true", help="also write embeddings_<tag>.npz (float16)")
    ap.add_argument("--out-dir", default="/kaggle/working/verification")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    t0 = time.time()
    requested = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    unknown = requested - set(PROTOCOLS)
    if unknown or not requested:
        raise SystemExit(f"unknown modes {sorted(unknown)}; choose from {sorted(PROTOCOLS)}")
    modes = [m for m in PROTOCOLS if m in requested]

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
    root = args.data_root or find_root()
    selected = select_images(modes, args, seen)
    for mode, df in selected.items():
        if df.empty:
            raise SystemExit(f"{mode}: no images selected (check --manifest, --test-list, --p3-list)")
        print(f"{mode}: {len(df)} images of {df.subject.nunique()} subjects "
              f"({int((df.phase == 'enrollment').sum())} enrollment, {int((df.phase == 'verification').sum())} "
              f"verification){' | smoke run' if args.max_subjects else ''}")
        for note in caveats(mode, train_phases):
            print(f"  NOTE ({mode}): {note}")

    timing, t = {}, time.time()
    model = load_checkpoint(args.ckpt)
    timing["load_model_s"] = round(time.time() - t, 1)
    print(f"loaded: {args.ckpt} ({timing['load_model_s']} s)")
    relpaths = sorted(set().union(*(set(df.relpath) for df in selected.values())))
    print(f"embedding {len(relpaths)} images (batch {args.batch_size}) from {root}", flush=True)
    t = time.time()
    emb = embed_images(model, relpaths, root, args.batch_size, args.workers)
    timing["embedding_s"] = round(time.time() - t, 1)
    row_of = {r: i for i, r in enumerate(relpaths)}
    for mode in selected:
        selected[mode]["row"] = selected[mode].relpath.map(row_of).to_numpy()

    all_rows, curves, samples, per_mode = [], {}, [], {}
    for mode in modes:
        t = time.time()
        if mode == "p1":
            rows, c, sample, extra = evaluate_pairs(selected[mode], emb, args)
        else:
            rows, c, sample, extra = evaluate_templates(selected[mode], emb, args, mode)
        timing[f"{mode}_s"] = round(time.time() - t, 1)
        notes = caveats(mode, train_phases)
        overall = rows[0]
        for target in FARS:
            key = far_key(target)
            if overall[f"far_achieved_{key}"] * overall["n_impostor"] < 30:
                notes.append(f"TAR at FAR {key} rests on fewer than 30 accepted impostor comparisons (rule of 30)")
        for row in rows:
            row.update({"mode": mode, "protocol": PROTOCOLS[mode], "caveat": "; ".join(notes)})
        all_rows += rows
        curves[mode] = c
        samples.append(sample)
        per_mode[mode] = {"protocol": PROTOCOLS[mode], "caveats": notes, "n_images": len(selected[mode]),
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
    labels = {m: f"{m.upper()} {'(inflated) ' if m == 'p1' else ''}{PROTOCOLS[m].split(':')[0]}: "
                 f"EER {100 * per_mode[m]['eer']:.2f}%, TAR@FAR=0.1% {100 * per_mode[m]['tar_at_far_1e-3']:.1f}%, "
                 f"rank-1 {100 * per_mode[m]['rank1']:.1f}%" for m in modes}
    plot_curves(curves, labels, paths["verification_roc_det"])
    if args.save_embeddings:
        np.savez_compressed(paths["embeddings"], embeddings=emb.astype(np.float16), relpaths=np.array(relpaths))

    timing["total_minutes"] = round((time.time() - t0) / 60, 1)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "git_commit": git_commit(), "modes": modes,
               "data_root": root, "manifest": args.manifest, "test_list": args.test_list, "p3_list": args.p3_list,
               "training_subjects": seen_source, "training_images_by_phase": train_phases,
               "max_subjects": args.max_subjects, "n_boot": args.n_boot, "seed": args.seed,
               "batch_size": args.batch_size, "score_grid_step": 1.0 / BINS_PER_UNIT,
               "n_images_embedded": len(relpaths), "embedding_dim": int(emb.shape[1]),
               "comparisons": "same eye only; FAR = false match rate, FRR = false non-match rate",
               "per_mode": per_mode, "timing": timing}
    with open(paths["verification_summary"], "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m in modes:
        r = per_mode[m]
        print(f"{m.upper()}: EER {r['eer']:.4f} [{r['eer_ci_low']:.4f}, {r['eer_ci_high']:.4f}] | "
              f"TAR@FAR=1e-2 {r['tar_at_far_1e-2']:.4f} | TAR@FAR=1e-3 {r['tar_at_far_1e-3']:.4f} | "
              f"rank-1 {r['rank1']:.4f} | {r['n_subjects']} subjects, {r['n_genuine']} genuine / "
              f"{r['n_impostor']} impostor comparisons")
    print(f"wrote results to {args.out_dir} ({timing['total_minutes']} min)")


if __name__ == "__main__":
    main()
