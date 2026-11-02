"""
scripts/16_check_dataset.py

Is our Kaggle VISOB copy (mubashirulhassan00/visob-augmented-recent v1) what we think it is?
Four checks on CPU; no model or GPU is needed.

  counts     unique subjects, images per phase / device / lighting, images outside the 300
             classes and how the frames are numbered, against the official VISOB Visit-1 counts
             (48,250 enrollment and 46,797 verification images; the copy has 88,552 and 45,706).
  synthetic  are the enrollment frames numbered 1010 and up synthetic copies of the frames
             numbered up to 1009? Each such frame is matched against the frames <= 1009 of its
             own (subject, condition, eye) bucket on 64 x 96 grey thumbnails: NCC and SSIM as
             they are, and the best NCC over a horizontal flip, shifts of up to 2 pixels and a
             grid of rotations and zooms (NCC ignores brightness and contrast; SSIM is taken
             after a brightness/contrast match). Genuine frames (verification frames, and
             enrollment frames <= 1009) are scored the same way against the other genuine
             frames of their bucket; their 99th percentiles are the thresholds. A frame is a
             near copy if it beats them as it is, a transformed copy if it beats them only after
             the search, and distinct otherwise. How often a flip, rotation or zoom is needed to
             match (frames >= 1010 vs genuine frames), the time structure of consecutive frames,
             image statistics (brightness, contrast, sharpness, noise, file facts), exact pixel
             duplicates and contact sheets complete the picture.
  overlap    share of test (and validation) images with an adjacent frame (idx +-1, same
             bucket and phase) or any image of their bucket in train, from 02's manifest_split.csv
  areas      per-image area share of each heuristic eye region for the images of an existing
             run (images_full.csv or correct_test_full.csv), computed exactly as 05 computes
             its region masks (load like 05, get_concept_masks, masks_to_grid)

Writes to --out-dir: summary.json (git commit, timing, headline numbers), report.md, and
  counts_by_folder.csv, counts_by_index.csv, buckets.csv                    (counts)
  frames.csv, pairs.csv, image_stats.csv, synthetic_summary.csv, image_stats_summary.csv,
  image_stats_paired.csv, image_sizes.csv, duplicates.csv, fig_similarity.png,
  contact_sheets/*.png                                                      (synthetic)
  overlap.csv                                                               (overlap)
  areas_<images-csv name>.csv, areas_<images-csv name>_grids.npz            (areas)

Usage (Kaggle CPU session, repo cloned, 01 and 02 already run):
  smoke: python scripts/16_check_dataset.py --max-buckets 20 --max-images 20 \
             --images-csv /kaggle/input/<run>/images_full.csv --out-dir /kaggle/working/data_check_smoke
  full:  python scripts/16_check_dataset.py --images-csv /kaggle/input/<run>/images_full.csv
"""
import os
if __name__ == "__main__":
    # One BLAS thread per process: the matching is many small matrix products, and --workers
    # already runs one process per core. Set before numpy is imported; workers inherit it.
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.compare.regions import REGIONS  # noqa: E402
from src.xai.masks import get_concept_masks, masks_to_grid  # noqa: E402


def _load_script(name):
    """Import a numbered script (e.g. 01_build_manifest.py) as a module."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest_script = _load_script("01_build_manifest")     # same walk, parsing and row order as 01

PARTS = ("counts", "synthetic", "overlap", "areas")
FIRST_IDX = 1001                    # genuine sequences start at frame 1001 (1000 never occurs)
LOW_MAX, HIGH_MIN = 1009, 1010      # frames <= 1009 match the official layout; >= 1010 is the extra block
OFFICIAL_IMAGES = {"enrollment": 48250, "verification": 46797}   # VISOB Visit 1 (arXiv:1811.00846)
OFFICIAL_SUBJECTS = 550             # VISOB 1.0; the base paper's Table 2 lists 585 classes
BUCKET = ["subject", "condition", "eye"]     # one capture sequence per phase
RECORD_COLS = ["relpath", "subject", "condition", "device", "lighting", "eye", "phase", "idx"]
THUMB = (64, 96)                    # (h, w): the native 160 x 240 image shrunk 2.5x
MARGIN = 6                          # template border, leaves room for shifts and rotated corners
ID_VARIANT = (False, 0.0, 1.0)      # no flip, no rotation, no zoom
SCORES = ("ncc_id", "ssim_id", "ncc_best", "ssim_best")
MATCH_FIELDS = SCORES + ("ncc_shift", "flip", "angle", "scale", "dy", "dx", "brightness_change", "contrast_change")
TRANSFORM_GAIN = 0.1                # NCC a flip/rotation/zoom must add over shifts alone to count as needed; on
                                    # real eye crops flips add 0.3-0.6, ~10 degrees or ~8 % zoom 0.1-0.2, and
                                    # natural 3 degree / 3 % changes only 0.02-0.05
MIN_TIME_SPAN = 0.01                # consecutive minus random genuine NCC below this: no time structure to scale by
CLASSES = ("near_copy", "transformed_copy", "distinct", "no_reference")
COPY_CLASSES = ("near_copy", "transformed_copy")
GENUINE_ROLES = ("genuine_enrollment", "genuine_verification")
PAIR_KINDS = ("consecutive_low", "consecutive_verification", "random_low", "consecutive_high", "cross_session")
STATS = ("brightness", "contrast", "saturation", "sharpness", "sharpness_norm", "noise", "file_bytes")
QC_KEYS = ("fallback_used", "circle_score", "cx", "cy", "radius", "n_candidates", "centre_in_opening",
           "pupil_found", "pupil_share_of_disc")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
AREAS_CHUNK = 8                     # images per areas task


# --- small helpers -------------------------------------------------------------------------

def data_path(root, relpath):
    """Absolute path of a manifest relpath (forward or back slashes) under the data root."""
    return os.path.join(root, *str(relpath).replace("\\", "/").split("/"))


def in_class_set(subject):
    """The 300 classes of 02_make_splits.py: subject IDs 1141..3279 plus 7000."""
    return subject.between(1141, 3279) | (subject == 7000)


def idx_range(phase, idx):
    """'<phase> <=1009' or '<phase> >=1010' for each row."""
    low = np.asarray(idx) <= LOW_MAX
    return pd.Series(np.where(low, np.asarray(phase, dtype=object) + f" <={LOW_MAX}",
                              np.asarray(phase, dtype=object) + f" >={HIGH_MIN}"), index=getattr(idx, "index", None))


def idx_band(idx):
    """Five-frame bands of the extra block: 1010-1014, 1015-1019, 1020+."""
    idx = np.asarray(idx)
    return np.where(idx < 1015, "1010-1014", np.where(idx < 1020, "1015-1019", "1020+"))


def git_commit():
    """(commit, dirty) of the repo this script runs from; ('unknown', None) without git."""
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        status = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha or "unknown"), (bool(status) if sha else None)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def run_tasks(fn, tasks, workers, label):
    """[fn(task) for task in tasks], in task order, run in --workers processes, printing progress
    and a time estimate."""
    results, t0, n = [None] * len(tasks), time.time(), len(tasks)
    step = max(1, n // 20)

    def progress(done):
        elapsed = time.time() - t0
        print(f"  [{label}] {done}/{n} | {elapsed / 60:.1f} min elapsed | "
              f"about {elapsed / done * (n - done) / 60:.1f} min left", flush=True)

    if workers <= 1 or n <= 1:
        for done, task in enumerate(tasks, 1):
            results[done - 1] = fn(task)
            if done % step == 0 or done == n or done == min(3, n):
                progress(done)
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        position = {pool.submit(fn, task): i for i, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(position), 1):
            results[position[future]] = future.result()
            if done % step == 0 or done == n or done == min(3 * workers, n):
                progress(done)
    return results


# --- reading images ------------------------------------------------------------------------

def png_facts(data):
    """Chunk sequence (repeats collapsed), largest IDAT chunk, bit depth and colour type of a PNG.

    Encoders differ here (libpng/OpenCV write 8 KB IDAT chunks, Pillow 64 KB), so two halves of
    a dataset written by different pipelines usually differ in these fields."""
    if data[:8] != PNG_SIGNATURE:
        return {"png_chunks": "not_png", "png_idat_max": 0, "png_bit_depth": -1, "png_colour_type": -1}
    pos, chunks, idat_max, bit_depth, colour = 8, [], 0, -1, -1
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8].decode("latin-1")
        if kind == "IHDR" and pos + 18 <= len(data):
            bit_depth, colour = data[pos + 16], data[pos + 17]
        if kind == "IDAT":
            idat_max = max(idat_max, length)
        if not chunks or chunks[-1] != kind:
            chunks.append(kind)
        pos += 12 + length
        if kind == "IEND":
            break
    return {"png_chunks": ",".join(chunks), "png_idat_max": idat_max, "png_bit_depth": int(bit_depth),
            "png_colour_type": int(colour)}


def read_image(path):
    """RGB uint8 pixels of one image file, plus its size, PIL mode, byte count and PNG facts."""
    with open(path, "rb") as f:
        data = f.read()
    with Image.open(io.BytesIO(data)) as img:
        width, height, mode = img.width, img.height, img.mode
        rgb = np.asarray(img.convert("RGB"))
    return rgb, {"width": width, "height": height, "mode": mode, "file_bytes": len(data), **png_facts(data)}


def image_stats(rgb):
    """Brightness and contrast (mean and std of grey), mean HSV saturation, sharpness (variance of
    the Laplacian; ``sharpness_norm`` divides it by the grey variance so brightness and contrast
    changes cancel) and noise (Immerkaer 1996 estimate), all in 8-bit grey levels.

    Resampling (rotation, zoom, sub-pixel shift) lowers sharpness_norm and noise; added noise
    raises noise; flips change neither."""
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float64)
    variance = float(grey.var())
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    residual = cv2.filter2D(grey, cv2.CV_64F, kernel)[1:-1, 1:-1]
    return {"brightness": float(grey.mean()), "contrast": float(np.sqrt(variance)),
            "saturation": float(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)[..., 1].mean()),
            "sharpness": sharpness, "sharpness_norm": sharpness / variance if variance > 0 else float("nan"),
            "noise": float(np.sqrt(np.pi / 2) * np.abs(residual).mean() / 6.0)}


def pixel_hashes(rgb):
    """MD5 of the decoded pixels and of their horizontal mirror image (exact duplicate check)."""
    rgb = np.ascontiguousarray(rgb)
    head = f"{rgb.shape[0]}x{rgb.shape[1]}".encode()
    return {"pixel_md5": hashlib.md5(head + rgb.tobytes()).hexdigest(),
            "flip_md5": hashlib.md5(head + np.ascontiguousarray(rgb[:, ::-1]).tobytes()).hexdigest()}


def thumbnail(rgb):
    """Grey 64 x 96 thumbnail, float32 in [0, 1] (area-averaged, so sensor noise mostly cancels)."""
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.resize(grey, (THUMB[1], THUMB[0]), interpolation=cv2.INTER_AREA)


# --- similarity ----------------------------------------------------------------------------

def _blur(x):
    return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)


def _ssim_parts(x):
    """The per-image terms of SSIM (image, local mean, local variance), reusable across pairs."""
    x = np.asarray(x, np.float64)
    mu = _blur(x)
    return x, mu, _blur(x * x) - mu ** 2


def _ssim_from_parts(pa, pb):
    (a, mu_a, var_a), (b, mu_b, var_b) = pa, pb
    cov = _blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = (2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s[5:-5, 5:-5].mean())


def ssim(a, b):
    """Mean SSIM of two 2-D images in [0, 1] (Wang et al. 2004: 11 x 11 Gaussian window, sigma 1.5,
    K1 = 0.01, K2 = 0.03), averaged away from a 5-pixel border as scikit-image does."""
    return _ssim_from_parts(_ssim_parts(a), _ssim_parts(b))


class Search(NamedTuple):
    """What a transformed copy may differ by: rotation angles (degrees), zoom factors and shifts
    (thumbnail pixels). Brightness and contrast need no search: NCC ignores them."""
    angles: tuple
    scales: tuple
    max_shift: int
    margin: int


def make_search(angles=(-20, -10, 0, 10, 20), scales=(0.9, 1.0, 1.1), max_shift=2):
    """A Search grid that always contains the untransformed image (angle 0, zoom 1)."""
    if not 0 <= int(max_shift) <= 12:
        raise ValueError("max_shift must be between 0 and 12 thumbnail pixels")
    return Search(tuple(sorted({float(a) for a in angles} | {0.0})),
                  tuple(sorted({float(s) for s in scales} | {1.0})),
                  int(max_shift), max(MARGIN, int(max_shift)))


def _variants(search):
    return [(flip, angle, scale) for flip in (False, True) for angle in search.angles for scale in search.scales]


def _shifts(search):
    return [(dy, dx) for dy in range(-search.max_shift, search.max_shift + 1)
            for dx in range(-search.max_shift, search.max_shift + 1)]


def _fine_offsets(grid):
    """Half and quarter steps around the best grid value (just 0 for a one-point grid)."""
    if len(grid) < 2:
        return (0.0,)
    step = float(np.min(np.diff(grid)))
    return tuple(step * f for f in (-0.5, -0.25, 0.0, 0.25, 0.5))


def variant(thumb, flip, angle, scale):
    """The thumbnail mirrored (flip), then rotated by *angle* degrees and zoomed by *scale* about
    its centre (bilinear, mirrored border)."""
    img = np.ascontiguousarray(thumb[:, ::-1] if flip else thumb)
    if angle == 0 and scale == 1:
        return img
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(angle), float(scale))
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _centre(img, margin):
    return img[margin:img.shape[0] - margin, margin:img.shape[1] - margin]


def _window(img, margin, dy, dx):
    """The part of a reference that the centre crop of a query covers after a (dy, dx) shift."""
    h, w = img.shape[0] - 2 * margin, img.shape[1] - 2 * margin
    return img[margin + dy:margin + dy + h, margin + dx:margin + dx + w]


def _unit_rows(rows):
    """Rows minus their mean, scaled to unit length (float32): a dot product of two rows is then
    their NCC. Constant rows become zeros (NCC 0)."""
    rows = np.array(rows, dtype=np.float32)
    rows -= rows.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", rows, rows))
    flat = norm <= 1e-6
    rows /= np.where(flat, 1.0, norm)[:, None]
    rows[flat] = 0.0
    return rows


def _templates(thumb, variants, margin):
    return _unit_rows([_centre(variant(thumb, *v), margin).ravel() for v in variants])


def prepare(thumb, search):
    """Precompute a thumbnail for matching: its templates (centre crops of every flip / rotation /
    zoom variant, used when it is the query), its shifted windows (used when it is a reference;
    same order as _shifts) and the SSIM terms of its centre crop."""
    m, s = search.margin, search.max_shift
    h, w = thumb.shape[0] - 2 * m, thumb.shape[1] - 2 * m
    windows = np.lib.stride_tricks.sliding_window_view(thumb[m - s:m + s + h, m - s:m + s + w], (h, w))
    return {"thumb": thumb, "templates": _templates(thumb, _variants(search), m),
            "windows": _unit_rows(windows.reshape(-1, h * w)), "ssim": _ssim_parts(_centre(thumb, m))}


def match(query, refs, search):
    """Best similarity of a prepared *query* to any of the prepared *refs*.

    ncc_id, ssim_id     the images as they are (best over refs; centre crops, so every score
                        uses the same pixels)
    ncc_shift           best NCC over refs x shift only (no flip, rotation or zoom);
                        ncc_best - ncc_shift is what the flip/rotation/zoom adds
    ncc_best            best NCC over refs x flip x rotation x zoom x shift, then refined on the
                        best ref with half and quarter grid steps of rotation, then of zoom
    ssim_best           SSIM at that alignment, after a least-squares brightness/contrast match
                        of the query to the reference
    flip, angle, scale, dy, dx   the alignment; brightness_change (grey levels) and
                        contrast_change (std ratio) of the aligned query against the reference
    best_ref            position of the best reference in *refs* (-1 without refs)
    """
    out = {"n_refs": len(refs), "best_ref": -1}
    if not refs:
        return {**out, **{k: float("nan") for k in MATCH_FIELDS}}
    variants, shifts = _variants(search), _shifts(search)
    id_variant, id_shift = variants.index(ID_VARIANT), shifts.index((0, 0))
    ncc_id = ssim_id = ncc_shift = -np.inf
    best = (-np.inf, 0, 0, 0)
    for k, ref in enumerate(refs):
        scores = ref["windows"] @ query["templates"].T          # (shifts, variants): NCC values
        ncc_id = max(ncc_id, min(1.0, float(scores[id_shift, id_variant])))
        ncc_shift = max(ncc_shift, min(1.0, float(scores[:, id_variant].max())))
        ssim_id = max(ssim_id, _ssim_from_parts(query["ssim"], ref["ssim"]))
        s, v = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if scores[s, v] > best[0]:
            best = (float(scores[s, v]), k, s, v)
    _, k, _, v = best
    flip, angle, scale = variants[v]
    # refine on the best reference: rotation first, then zoom (the two barely interact); each
    # list includes the current best, so the score can only go up
    for axis, offsets in ((1, _fine_offsets(search.angles)), (2, _fine_offsets(search.scales))):
        fine = [(flip, angle + d, scale) if axis == 1 else (flip, angle, scale + d) for d in offsets]
        scores = refs[k]["windows"] @ _templates(query["thumb"], fine, search.margin).T
        s, f = np.unravel_index(int(np.argmax(scores)), scores.shape)
        flip, angle, scale = fine[f]
    dy, dx = shifts[s]
    t = _centre(variant(query["thumb"], flip, angle, scale), search.margin).astype(np.float64)
    w = _window(refs[k]["thumb"], search.margin, dy, dx).astype(np.float64)
    t_mean, w_mean, t_std, w_std = t.mean(), w.mean(), t.std(), w.std()
    gain = float(((t - t_mean) * (w - w_mean)).mean() / t_std ** 2) if t_std > 0 else 0.0
    return {**out, "best_ref": int(k), "ncc_id": ncc_id, "ssim_id": ssim_id, "ncc_shift": ncc_shift,
            "ncc_best": float(np.clip(scores[s, f], -1.0, 1.0)),
            "ssim_best": ssim(gain * (t - t_mean) + w_mean, w),
            "flip": bool(flip), "angle": float(angle), "scale": float(scale), "dy": int(dy), "dx": int(dx),
            "brightness_change": float((t_mean - w_mean) * 255.0),
            "contrast_change": float(t_std / w_std) if w_std > 0 else float("nan")}


def aligned_difference(query_thumb, ref_thumb, row, search):
    """|aligned, brightness-matched query - reference| on the thumbnail crop (for the pair sheets)."""
    t = _centre(variant(query_thumb, bool(row["flip"]), row["angle"], row["scale"]), search.margin)
    w = _window(ref_thumb, search.margin, int(row["dy"]), int(row["dx"]))
    t, w = t.astype(np.float64), w.astype(np.float64)
    gain = ((t - t.mean()) * (w - w.mean())).mean() / t.var() if t.var() > 0 else 0.0
    return np.abs(gain * (t - t.mean()) + w.mean() - w)


# --- part: synthetic -----------------------------------------------------------------------

def select_buckets(manifest, max_buckets=None, seed=0):
    """Enrollment (subject, condition, eye) buckets with at least one frame >= 1010 (a random
    sample of *max_buckets* of them for a smoke run)."""
    enr = manifest[(manifest.phase == "enrollment") & (manifest.idx >= HIGH_MIN)]
    keys = enr[BUCKET].drop_duplicates().sort_values(BUCKET).reset_index(drop=True)
    if max_buckets and max_buckets < len(keys):
        keys = keys.sample(n=max_buckets, random_state=seed).sort_values(BUCKET).reset_index(drop=True)
    return keys


def _bucket_rng(seed, subject, condition, eye):
    """Random numbers that depend only on the bucket, not on worker scheduling."""
    return np.random.default_rng([int(seed), int(subject), zlib.crc32(str(condition).encode()),
                                  0 if eye == "left" else 1])


def score_unit(task):
    """Load one (subject, condition) and score its buckets.

    task: (root, records, search, n_random, seed); records are manifest rows (RECORD_COLS) of the
    selected buckets, both phases. Returns {"images": [...], "frames": [...], "pairs": [...]}.

    frames: every enrollment frame >= 1010 (role "query") against the frames <= 1009 of its
    bucket, and every genuine frame (verification, or enrollment <= 1009) against the other
    genuine frames of its bucket and phase (roles "genuine_verification", "genuine_enrollment").
    pairs: single comparisons for the reference distributions (PAIR_KINDS).
    """
    root, records, search, n_random, seed = task
    images, prepared = [], {}
    for rec in records:
        row = dict(rec)
        try:
            rgb, facts = read_image(data_path(root, rec["relpath"]))
        except (OSError, ValueError, cv2.error) as e:      # a corrupt file must not stop a long run
            images.append({**row, "error": f"{type(e).__name__}: {e}"})
            continue
        images.append({**row, **facts, **image_stats(rgb), **pixel_hashes(rgb)})
        prepared[rec["relpath"]] = prepare(thumbnail(rgb), search)

    frames, pairs = [], []
    table = pd.DataFrame(records)
    table = table[table.relpath.isin(list(prepared))]

    def frame_row(rec, refs, role, extra=None):
        result = match(prepared[rec["relpath"]], [prepared[r["relpath"]] for r in refs], search)
        k = result.pop("best_ref")
        best = refs[k] if k >= 0 else None
        return {**rec, "role": role, **result, "best_ref": best["relpath"] if best else None,
                "best_ref_idx": int(best["idx"]) if best else -1, **(extra or {})}

    def pair_row(kind, a, b):
        result = match(prepared[a["relpath"]], [prepared[b["relpath"]]], search)
        result.pop("best_ref")
        return {"kind": kind, **{c: a[c] for c in ("subject", "condition", "device", "lighting", "eye")},
                "relpath_a": a["relpath"], "relpath_b": b["relpath"], "phase_a": a["phase"], "phase_b": b["phase"],
                "idx_a": int(a["idx"]), "idx_b": int(b["idx"]), **result}

    def consecutive(rows):
        return [(rows[i + 1], rows[i]) for i in range(len(rows) - 1) if rows[i + 1]["idx"] - rows[i]["idx"] == 1]

    for (subject, condition, eye), bucket in table.groupby(BUCKET, sort=True):
        bucket = bucket.sort_values("idx")
        enr = bucket[bucket.phase == "enrollment"].to_dict("records")
        low = [r for r in enr if r["idx"] <= LOW_MAX]
        high = [r for r in enr if r["idx"] >= HIGH_MIN]
        ver = bucket[bucket.phase == "verification"].to_dict("records")
        last_low = max((r["idx"] for r in low), default=-1)
        frames += [frame_row(q, low, "query", {"last_low_idx": last_low}) for q in high]
        if len(low) >= 2:
            frames += [frame_row(g, [r for r in low if r is not g], "genuine_enrollment") for g in low]
        if len(ver) >= 2:
            frames += [frame_row(g, [r for r in ver if r is not g], "genuine_verification") for g in ver]

        # reference pairs: at most n_random of each kind per bucket, drawn at random
        rng = _bucket_rng(seed, subject, condition, eye)
        candidates = {
            "consecutive_low": consecutive(low),
            "consecutive_verification": consecutive(ver),
            "random_low": [(b, a) for i, a in enumerate(low) for b in low[i + 1:] if b["idx"] - a["idx"] >= 2],
            "consecutive_high": consecutive(high),
            "cross_session": [(v, g) for v in ver for g in low],
        }
        for kind in PAIR_KINDS:
            options = candidates[kind]
            pairs += [pair_row(kind, *options[j]) for j in sorted(rng.permutation(len(options))[:n_random])]
    return {"images": images, "frames": frames, "pairs": pairs}


def thresholds(null, quantile):
    """Per-score *quantile* of a genuine reference table: genuine frames scored against their own
    bucket (the main thresholds: the same best-of-bucket statistic as the frames under test), or
    single genuine frame pairs."""
    return {s: float(np.nanquantile(null[s], quantile)) if len(null) and null[s].notna().any() else float("nan")
            for s in SCORES}


def classify(frames, thr):
    """near_copy: beats the genuine thresholds as it is (NCC and SSIM); transformed_copy: beats
    them only after the flip/shift/rotation/zoom search and brightness/contrast match; distinct:
    neither; no_reference: the bucket had no genuine frame to compare with."""
    near = (frames.ncc_id >= thr["ncc_id"]) & (frames.ssim_id >= thr["ssim_id"])
    moved = ~near & (frames.ncc_best >= thr["ncc_best"]) & (frames.ssim_best >= thr["ssim_best"])
    out = np.where(near, "near_copy", np.where(moved, "transformed_copy", "distinct"))
    out = np.where(frames.n_refs.to_numpy() == 0, "no_reference", out)
    return pd.Series(out, index=frames.index)


def _shares(values, categories):
    counts = pd.Series(values).value_counts()
    n = max(int(counts.sum()), 1)
    return {c: float(counts.get(c, 0) / n) for c in categories}


def synthetic_summary(frames):
    """Class shares of the frames >= 1010 by device x lighting x idx band, with the rate at which
    the same rule flags genuine frames of that device and lighting."""
    queries = frames[frames.role == "query"].copy()
    queries["idx_band"] = idx_band(queries.idx)
    genuine = frames[frames.role.isin(GENUINE_ROLES)]
    rows = []
    groups = [((d, l, b), g) for (d, l, b), g in queries.groupby(["device", "lighting", "idx_band"], sort=True)]
    groups.append((("all", "all", "all"), queries))
    for (device, lighting, band), g in groups:
        ref = genuine if device == "all" else genuine[(genuine.device == device) & (genuine.lighting == lighting)]
        rows.append({"device": device, "lighting": lighting, "idx_band": band, "n": len(g),
                     **_shares(g.frame_class, CLASSES),
                     "ncc_best_median": float(g.ncc_best.median()), "ssim_best_median": float(g.ssim_best.median()),
                     "genuine_n": len(ref), "genuine_flag_rate": float(ref.frame_class.isin(COPY_CLASSES).mean())
                     if len(ref) else float("nan")})
    return pd.DataFrame(rows)


def transform_signature(rows):
    """Share of *rows* whose best match needs a flip, rotation or zoom: one that adds at least
    TRANSFORM_GAIN to the NCC of shifts alone (ncc_best - ncc_shift), and which transforms those
    frames used. Natural motion between frames of a sequence is mostly a shift, so genuine frames
    rarely need one; transformed copies do, even when interpolation keeps them just below the copy
    thresholds."""
    if not len(rows):
        return {"n": 0, **{k: float("nan") for k in ("needs_transform", "flip", "rotation", "zoom")}}
    needs = (rows.ncc_best - rows.ncc_shift) >= TRANSFORM_GAIN
    moved = rows[needs]
    return {"n": int(len(rows)), "needs_transform": float(needs.mean()),
            **{k: float(v.mean()) if len(moved) else float("nan")
               for k, v in (("flip", moved.flip.eq(True)), ("rotation", moved.angle != 0), ("zoom", moved.scale != 1))}}


def image_stats_tables(images):
    """Median image statistics by device x lighting x idx range, and within-bucket comparisons:
    enrollment >= 1010 vs <= 1009, and (a genuine-vs-genuine yardstick) verification vs enrollment
    <= 1009. Ratios are per bucket (median of the frames of each range), then summarised."""
    from scipy.stats import wilcoxon

    ok = images[images.get("error", pd.Series(index=images.index, dtype=object)).isna()].copy()
    ok["range"] = np.where(ok.phase == "verification", "verification", idx_range(ok.phase, ok.idx))
    summary = []
    for keys, g in list(ok.groupby(["device", "lighting", "range"], sort=True)) + \
            [(("all", "all", r), g) for r, g in ok.groupby("range", sort=True)]:
        summary.append({"device": keys[0], "lighting": keys[1], "range": keys[2], "n": len(g),
                        **{s: float(g[s].median()) for s in STATS}})

    per_bucket = ok.groupby(BUCKET + ["range"])[list(STATS)].median().unstack("range")
    paired = []
    for a, b in ((f"enrollment >={HIGH_MIN}", f"enrollment <={LOW_MAX}"), ("verification", f"enrollment <={LOW_MAX}")):
        for stat in STATS:
            if (stat, a) not in per_bucket or (stat, b) not in per_bucket:
                continue
            both = per_bucket[[(stat, a), (stat, b)]].dropna()
            va, vb = both[(stat, a)].to_numpy(float), both[(stat, b)].to_numpy(float)
            diff = va - vb
            try:
                p = float(wilcoxon(diff).pvalue) if np.count_nonzero(diff) >= 5 else float("nan")
            except ValueError:
                p = float("nan")
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(vb != 0, va / vb, np.nan)
            paired.append({"comparison": f"{a} vs {b}", "stat": stat, "n_buckets": len(both),
                           "median_a": float(np.median(va)) if len(va) else float("nan"),
                           "median_b": float(np.median(vb)) if len(vb) else float("nan"),
                           "median_ratio": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan"),
                           "median_difference": float(np.median(diff)) if len(diff) else float("nan"),
                           "share_a_higher": float((diff > 0).mean()) if len(diff) else float("nan"),
                           "wilcoxon_p": p})
    sizes = (ok.astype({"width": int, "height": int}).groupby(["phase", "range", "width", "height", "mode"]).size()
             .rename("images").reset_index())
    return pd.DataFrame(summary), pd.DataFrame(paired), sizes


def exact_duplicates(images):
    """Pairs of images with identical decoded pixels ("exact"), or where one is the exact mirror
    image of the other ("mirror"), anywhere among the loaded images."""
    cols = ["relpath", "subject", "condition", "eye", "phase", "idx"]
    ok = images.dropna(subset=["pixel_md5"]) if "pixel_md5" in images else images.iloc[:0]
    left = ok[cols + ["pixel_md5", "flip_md5"]]
    exact = left.merge(left[cols + ["pixel_md5"]], on="pixel_md5", suffixes=("_a", "_b"))
    mirror = left.merge(left[cols + ["pixel_md5"]], left_on="flip_md5", right_on="pixel_md5", suffixes=("_a", "_b"))
    out = pd.concat([exact.assign(kind="exact"), mirror.assign(kind="mirror")], ignore_index=True)
    out = out[out.relpath_a < out.relpath_b]           # each pair once; an image is not its own duplicate
    out = out.assign(same_subject=out.subject_a == out.subject_b,
                     same_bucket=(out.subject_a == out.subject_b) & (out.condition_a == out.condition_b)
                     & (out.eye_a == out.eye_b))
    return out[["kind", "relpath_a", "relpath_b", "phase_a", "phase_b", "idx_a", "idx_b", "same_subject",
                "same_bucket"]].sort_values(["kind", "relpath_a", "relpath_b"]).reset_index(drop=True)


def png_by_range(images):
    """Most common PNG chunk sequences and largest-IDAT sizes per idx range (encoder fingerprints)."""
    if "png_chunks" not in images:
        return {}
    ok = images[images.png_chunks.notna()]
    out = {}
    for name, g in ok.groupby(idx_range(ok.phase, ok.idx), sort=True):
        out[name] = {"chunks": {str(k): int(v) for k, v in g.png_chunks.value_counts().head(3).items()},
                     "largest_idat": {str(int(k)): int(v) for k, v in g.png_idat_max.value_counts().head(3).items()}}
    return out


def pair_summary(pairs):
    """n and quantiles of each score per reference-pair kind."""
    out = {}
    for kind in PAIR_KINDS:
        g = pairs[pairs.kind == kind] if len(pairs) else pairs
        out[kind] = {"n": int(len(g)),
                     **{f"{s}_median": float(g[s].median()) if len(g) else float("nan") for s in SCORES},
                     "ncc_best_p95": float(g.ncc_best.quantile(0.95)) if len(g) else float("nan")}
    return out


def plot_similarity(frames, pairs, thr, path):
    """ECDFs of the best-match scores: frames >= 1010 vs genuine frames (left two panels) and
    the time structure of consecutive frames (right panel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (CVD-safe set)

    def ecdf(ax, values, label, color, style="-"):
        v = np.sort(np.asarray(values, float)[np.isfinite(np.asarray(values, float))])
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), style, color=color, lw=1.8, label=f"{label} (n={len(v)})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    queries, genuine = frames[frames.role == "query"], frames[frames.role.isin(GENUINE_ROLES)]
    for ax, score, title in ((axes[0], "ncc_best", "best NCC (after flip/shift/rotation/zoom)"),
                             (axes[1], "ssim_best", "SSIM at that alignment")):
        ecdf(ax, queries[score], f"frames >={HIGH_MIN} vs frames <={LOW_MAX}", orange)
        ecdf(ax, genuine[score], "genuine frames vs their own bucket", blue)
        if len(pairs):
            ecdf(ax, pairs.loc[pairs.kind == "random_low", score], "random genuine pairs", aqua)
        ax.axvline(thr[score], color="#52514e", lw=1, ls=":")
        ax.text(thr[score], 0.02, " threshold", color="#52514e", fontsize=8)
        ax.set_title(title, fontsize=10)
    if len(pairs):
        ecdf(axes[2], pairs.loc[pairs.kind == "consecutive_high", "ncc_best"],
             f"consecutive frames >={HIGH_MIN}", orange)
        ecdf(axes[2], pairs.loc[pairs.kind.isin(["consecutive_low", "consecutive_verification"]), "ncc_best"],
             "consecutive genuine frames", blue)
        ecdf(axes[2], pairs.loc[pairs.kind == "random_low", "ncc_best"], "random genuine pairs", aqua)
    axes[2].set_title("time structure: best NCC of frame pairs", fontsize=10)
    for ax in axes:
        ax.set_ylabel("share of frames at or below", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=7, loc="best", frameon=False)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _label(row):
    """Two-line tile caption: the class, its best match and how the frame was aligned to it."""
    if row is None or pd.isna(row.get("frame_class")):
        return ""
    cls = row["frame_class"]
    if cls == "no_reference":
        return "no frame <= 1009 to compare"
    moves = (["flip"] if row["flip"] else []) + ([f"rot {row['angle']:g}"] if row["angle"] else []) + \
            ([f"zoom {row['scale']:g}"] if row["scale"] != 1 else []) + \
            ([f"shift {int(row['dy'])},{int(row['dx'])}"] if row["dy"] or row["dx"] else []) + \
            ([f"light {row['brightness_change']:+.0f}"] if abs(row["brightness_change"]) > 5 else [])
    if cls == "distinct":
        return f"distinct\nclosest {int(row['best_ref_idx'])}: {' '.join(moves) or 'as is'} | NCC {row['ncc_best']:.3f}"
    name = "NEAR COPY" if cls == "near_copy" else "TRANSFORMED"
    return f"{name} of {int(row['best_ref_idx'])}\n{' '.join(moves) or 'as is'} | NCC {row['ncc_best']:.3f}"


def contact_sheets(images, frames, root, out_dir, n_buckets, seed, search):
    """One sheet per random bucket (frames <= 1009 on top, >= 1010 below, each labelled with its
    class and best match), plus pair sheets: random frames of each class next to their best match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    by_path = {r["relpath"]: r for r in frames[frames.role == "query"].to_dict("records")}
    ok = images[images.get("error", pd.Series(index=images.index, dtype=object)).isna()]
    enr = ok[ok.phase == "enrollment"]
    candidates = enr.groupby(BUCKET).idx.agg(lambda s: (s <= LOW_MAX).any() and (s >= HIGH_MIN).any())
    candidates = candidates[candidates].index.to_frame(index=False)
    pick = candidates.sample(n=min(n_buckets, len(candidates)), random_state=seed) if len(candidates) else candidates
    cols = 8
    for b in pick.sort_values(BUCKET).itertuples(index=False):
        sel = enr[(enr.subject == b.subject) & (enr.condition == b.condition) & (enr.eye == b.eye)].sort_values("idx")
        low, high = sel[sel.idx <= LOW_MAX], sel[sel.idx >= HIGH_MIN]
        rows_low, rows_high = -(-len(low) // cols), -(-len(high) // cols)
        fig, axes = plt.subplots(rows_low + rows_high, cols, figsize=(2.1 * cols, 1.75 * (rows_low + rows_high) + 0.4),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for start, part in ((0, low), (rows_low * cols, high)):
            for i, r in enumerate(part.itertuples(index=False)):
                ax = axes.ravel()[start + i]
                ax.imshow(read_image(data_path(root, r.relpath))[0])
                label = _label(by_path.get(r.relpath)) if r.idx >= HIGH_MIN else ""
                ax.set_title(f"{r.idx}" + (f": {label}" if label else ""), fontsize=6)
        fig.suptitle(f"subject {b.subject} | {b.condition} | {b.eye} eye | top: frames <={LOW_MAX}, "
                     f"bottom: frames >={HIGH_MIN}", fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"bucket_{b.subject}_{b.condition}_{b.eye}.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)

    queries = frames[(frames.role == "query") & (frames.n_refs > 0)]
    for cls in ("near_copy", "transformed_copy", "distinct"):
        g = queries[queries.frame_class == cls]
        if g.empty:
            continue
        g = g.sample(n=min(8, len(g)), random_state=seed)
        fig, axes = plt.subplots(len(g), 3, figsize=(8.4, 1.9 * len(g)), squeeze=False)
        for row_axes, r in zip(axes, g.to_dict("records")):
            q_rgb = read_image(data_path(root, r["relpath"]))[0]
            b_rgb = read_image(data_path(root, r["best_ref"]))[0]
            diff = aligned_difference(thumbnail(q_rgb), thumbnail(b_rgb), r, search)
            row_axes[0].imshow(q_rgb)
            row_axes[0].set_title(f"{os.path.basename(r['relpath'])}", fontsize=6)
            row_axes[1].imshow(b_rgb)
            row_axes[1].set_title(f"best match {os.path.basename(r['best_ref'])}", fontsize=6)
            row_axes[2].imshow(diff, cmap="gray_r", vmin=0, vmax=0.25)
            row_axes[2].set_title(_label(r).replace("\n", " | "), fontsize=6)
            for ax in row_axes:
                ax.axis("off")
        fig.suptitle(f"{cls}: frame >={HIGH_MIN} | its best match <={LOW_MAX} | |aligned difference| "
                     "(white = identical)", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"pairs_{cls}.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)
    return written


def run_synthetic(manifest, root, args, out_dir):
    """Score the selected buckets and write the synthetic-part outputs; returns (frames, facts)."""
    search = make_search(args.angles, args.scales, args.max_shift)
    os.makedirs(out_dir, exist_ok=True)
    keys = select_buckets(manifest, args.max_buckets, args.seed)
    if keys.empty:
        print("  no enrollment frames >= 1010 in the manifest: nothing to score")
        return pd.DataFrame(), {"buckets_selected": 0, "frames_scored": 0}
    todo = manifest.merge(keys, on=BUCKET)[RECORD_COLS].sort_values(["subject", "condition", "eye", "phase", "idx"])
    tasks = [(root, g.to_dict("records"), search, args.pairs_per_bucket, args.seed)
             for _, g in todo.groupby(["subject", "condition"], sort=True)]
    print(f"  {len(keys)} buckets ({len(tasks)} subject x condition units, {len(todo)} images) | "
          f"search: flip x {len(search.angles)} angles x {len(search.scales)} zooms x shifts +-{search.max_shift} px "
          f"| {args.workers} worker(s)", flush=True)
    results = run_tasks(score_unit, tasks, args.workers, "synthetic")

    images = pd.DataFrame([r for res in results for r in res["images"]]).sort_values("relpath").reset_index(drop=True)
    if "error" not in images:
        images["error"] = np.nan
    frames = pd.DataFrame([r for res in results for r in res["frames"]])
    pairs = pd.DataFrame([r for res in results for r in res["pairs"]])
    if frames.empty:
        print("  no frame could be scored")
        return frames, {"buckets_selected": int(len(keys)), "frames_scored": 0}
    frames = frames.sort_values(["role", "relpath"]).reset_index(drop=True)
    if len(pairs):
        pairs = pairs.sort_values(["kind", "relpath_a", "relpath_b"]).reset_index(drop=True)

    thr = thresholds(frames[frames.role.isin(GENUINE_ROLES)], args.ref_quantile)
    thr_ver = thresholds(frames[frames.role == "genuine_verification"], args.ref_quantile)
    consecutive_pairs = pairs[pairs.kind.isin(["consecutive_low", "consecutive_verification"])] if len(pairs) \
        else pairs
    thr_pairs = thresholds(consecutive_pairs, args.ref_quantile)
    frames["frame_class"] = classify(frames, thr)
    queries = frames[frames.role == "query"]
    scored = queries[queries.frame_class != "no_reference"]
    genuine = frames[frames.role.isin(GENUINE_ROLES)]
    copies = scored[scored.frame_class.isin(COPY_CLASSES)]
    moved = scored[scored.frame_class == "transformed_copy"]
    per_source = copies.best_ref.value_counts()

    summary_table = synthetic_summary(frames)
    stats_summary, stats_paired, sizes = image_stats_tables(images)
    duplicates = exact_duplicates(images)
    images.to_csv(os.path.join(out_dir, "image_stats.csv"), index=False)
    frames.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    summary_table.to_csv(os.path.join(out_dir, "synthetic_summary.csv"), index=False)
    stats_summary.to_csv(os.path.join(out_dir, "image_stats_summary.csv"), index=False)
    stats_paired.to_csv(os.path.join(out_dir, "image_stats_paired.csv"), index=False)
    sizes.to_csv(os.path.join(out_dir, "image_sizes.csv"), index=False)
    duplicates.to_csv(os.path.join(out_dir, "duplicates.csv"), index=False)
    plot_similarity(frames, pairs, thr, os.path.join(out_dir, "fig_similarity.png"))
    sheets = contact_sheets(images, frames, root, os.path.join(out_dir, "contact_sheets"),
                            args.contact_sheets, args.seed, search)

    ps = pair_summary(pairs) if len(pairs) else {}
    coherence = float("nan")
    if ps and all(ps[k]["n"] for k in ("consecutive_low", "random_low", "consecutive_high")):
        span = ps["consecutive_low"]["ncc_best_median"] - ps["random_low"]["ncc_best_median"]
        if span >= MIN_TIME_SPAN:       # a smaller span is noise, and dividing by it gives any number
            coherence = (ps["consecutive_high"]["ncc_best_median"] - ps["random_low"]["ncc_best_median"]) / span
    paired = {f"{r.stat} | {r.comparison}": {"median_ratio": r.median_ratio, "median_difference": r.median_difference,
                                             "share_a_higher": r.share_a_higher, "wilcoxon_p": r.wilcoxon_p,
                                             "n_buckets": int(r.n_buckets)}
              for r in stats_paired.itertuples()} if len(stats_paired) else {}
    facts = {
        "buckets_selected": int(len(keys)), "buckets_with_frames_ge_1010_total": int(len(select_buckets(manifest))),
        "images_loaded": int(len(images)), "images_unreadable": int(images.error.notna().sum()),
        "frames_ge_1010": int(len(queries)), "frames_scored": int(len(scored)),
        "frames_without_reference": int((queries.frame_class == "no_reference").sum()),
        "genuine_frames": {role: int((frames.role == role).sum()) for role in GENUINE_ROLES},
        "search": {"angles": list(search.angles), "scales": list(search.scales), "max_shift_px": search.max_shift,
                   "thumbnail": list(THUMB), "template_margin_px": search.margin},
        "ref_quantile": args.ref_quantile, "thresholds": thr, "thresholds_verification_only": thr_ver,
        "thresholds_consecutive_pairs": thr_pairs,
        "class_share": _shares(scored.frame_class, CLASSES[:3]),
        "copy_share": float(scored.frame_class.isin(COPY_CLASSES).mean()) if len(scored) else float("nan"),
        "copy_share_verification_thresholds": float(classify(scored, thr_ver).isin(COPY_CLASSES).mean())
        if len(scored) else float("nan"),
        "copy_share_consecutive_pair_thresholds": float(classify(scored, thr_pairs).isin(COPY_CLASSES).mean())
        if len(scored) else float("nan"),
        "genuine_flag_rate": float(genuine.frame_class.isin(COPY_CLASSES).mean()) if len(genuine) else float("nan"),
        "genuine_flag_rate_by_role": {role: float(frames.loc[frames.role == role, "frame_class"]
                                                  .isin(COPY_CLASSES).mean()) if (frames.role == role).any()
                                      else float("nan") for role in GENUINE_ROLES},
        "transforms_of_transformed_copies": {
            "n": int(len(moved)), "flip": float(moved.flip.astype(bool).mean()) if len(moved) else float("nan"),
            "rotation": float((moved.angle != 0).mean()) if len(moved) else float("nan"),
            "zoom": float((moved.scale != 1).mean()) if len(moved) else float("nan"),
            "shift": float(((moved.dy != 0) | (moved.dx != 0)).mean()) if len(moved) else float("nan"),
            "brightness_over_5_levels": float((moved.brightness_change.abs() > 5).mean()) if len(moved)
            else float("nan"),
            "contrast_over_5pct": float(((moved.contrast_change - 1).abs() > 0.05).mean()) if len(moved)
            else float("nan")},
        "transform_gain": {"min_gain": TRANSFORM_GAIN, "frames_ge_1010": transform_signature(scored),
                           "genuine": transform_signature(genuine)},
        "copies_per_source_frame": {str(k): int(v) for k, v in per_source.value_counts().sort_index().items()},
        "best_match_is_last_genuine_frame": float((scored.best_ref_idx == scored.last_low_idx).mean())
        if len(scored) else float("nan"),
        "pairs": ps, "time_coherence_of_frames_ge_1010": coherence,
        "image_stats_paired": paired,
        "exact_duplicates": {"exact_pairs": int((duplicates.kind == "exact").sum()),
                             "mirror_pairs": int((duplicates.kind == "mirror").sum()),
                             "pairs_across_subjects": int((~duplicates.same_subject).sum()),
                             "pairs_involving_frames_ge_1010": int(((duplicates.idx_a >= HIGH_MIN) |
                                                                    (duplicates.idx_b >= HIGH_MIN)).sum())},
        "image_sizes": {f"{w}x{h} {mode}": int(n) for (w, h, mode), n in
                        sizes.groupby(["width", "height", "mode"]).images.sum().items()},
        "png_by_range": png_by_range(images),
        "contact_sheets": [os.path.relpath(p, out_dir).replace("\\", "/") for p in sheets],
    }
    facts["verdict"] = synthetic_verdict(facts)
    return frames, facts


def synthetic_verdict(f):
    """A cautious one-line reading of the pixel evidence (the report gives the numbers): copies are
    'found' when at least 5 % of the frames >= 1010 are copies and at least 3x as many as the
    genuine frames the same rule flags (about 1-2 % by construction of the thresholds).

    Without that, two signs that the thresholds missed copies make the reading 'unclear' rather
    than 'no pixel copies found': many more frames >= 1010 than genuine frames need a flip,
    rotation or zoom to match their bucket (when genuine frames of a sequence are almost
    identical, the thresholds sit so close to 1 that interpolated copies fall just below them),
    or consecutive frames >= 1010 are much less alike than consecutive genuine frames (time
    structure below 0.5: not one continuous sequence)."""
    copy, genuine = f.get("copy_share", float("nan")), f.get("genuine_flag_rate", float("nan"))
    if not np.isfinite(copy) or not np.isfinite(genuine):
        return "not enough frames to judge"
    if copy >= max(0.05, 3 * genuine):
        return "synthetic copies found"
    gain = f.get("transform_gain", {})
    moved = gain.get("frames_ge_1010", {}).get("needs_transform", float("nan"))
    moved_genuine = gain.get("genuine", {}).get("needs_transform", float("nan"))
    if np.isfinite(moved) and np.isfinite(moved_genuine) and moved >= max(0.05, 3 * moved_genuine + 0.01):
        return (f"unclear: few frames >= {HIGH_MIN} pass the copy thresholds, but {100 * moved:.0f} % of them match "
                f"their bucket only after a flip, rotation or zoom (genuine frames {100 * moved_genuine:.0f} %); "
                "inspect contact_sheets/pairs_distinct.png")
    coherence = f.get("time_coherence_of_frames_ge_1010", float("nan"))
    if np.isfinite(coherence) and coherence < 0.5:
        return (f"unclear: no copies found, but consecutive frames >= {HIGH_MIN} are much less alike than "
                f"consecutive genuine frames (time structure {coherence:.2f}); inspect the contact sheets and "
                "try a wider --max-shift / --angles / --scales")
    if copy <= 2 * genuine + 0.01:
        return "no pixel copies found"
    return "unclear: inspect the contact sheets"


# --- part: counts --------------------------------------------------------------------------

def count_dataset(manifest):
    """Counts and frame numbering from the manifest (no pixels). Returns (tables, facts)."""
    m = manifest.copy()
    m["in_300"] = in_class_set(m.subject)
    by_folder = (m.groupby(["condition", "device", "lighting", "phase", "eye"])
                 .agg(subjects=("subject", "nunique"), images=("relpath", "size"), images_300=("in_300", "sum"))
                 .reset_index())
    by_index = pd.crosstab(m.idx, m.phase).add_prefix("all_").join(
        pd.crosstab(m[m.in_300].idx, m[m.in_300].phase).add_prefix("classes300_"), how="left").fillna(0).astype(int)
    by_index = by_index.reset_index()

    # one row per bucket and phase; frame numbers are unique within a bucket folder
    low, high = m.idx <= LOW_MAX, m.idx >= HIGH_MIN
    m = m.assign(low=low, high=high, idx_low=m.idx.where(low), idx_high=m.idx.where(high),
                 is_1009=m.idx == LOW_MAX, is_1010=m.idx == HIGH_MIN)
    buckets = (m.groupby(BUCKET + ["device", "lighting", "phase", "in_300"])
               .agg(n=("idx", "size"), n_low=("low", "sum"), n_high=("high", "sum"), idx_min=("idx", "min"),
                    idx_max=("idx", "max"), low_min=("idx_low", "min"), low_max=("idx_low", "max"),
                    high_min=("idx_high", "min"), high_max=("idx_high", "max"), has_1009=("is_1009", "any"),
                    has_1010=("is_1010", "any"))
               .reset_index())
    buckets["low_contiguous"] = (buckets.n_low > 0) & (buckets.low_min == FIRST_IDX) & \
        (buckets.low_max == FIRST_IDX + buckets.n_low - 1)
    buckets["high_contiguous"] = (buckets.n_high > 0) & (buckets.high_min == HIGH_MIN) & \
        (buckets.high_max == HIGH_MIN + buckets.n_high - 1)
    enr_b, ver_b = buckets[buckets.phase == "enrollment"], buckets[buckets.phase == "verification"]
    with_high, ver_high = enr_b[enr_b.n_high > 0], ver_b[ver_b.n_high > 0]
    combos = with_high.groupby(["n_low", "n_high"]).size().sort_values(ascending=False).head(10)

    phase_counts = m.groupby("phase").agg(images=("relpath", "size"), subjects=("subject", "nunique"))
    facts = {
        "images": int(len(m)), "subjects": int(m.subject.nunique()),
        "official_subjects": OFFICIAL_SUBJECTS,
        "subject_differs_from_folder": int((m.subject != m.get("folder_subject", m.subject)).sum()),
        "side_differs_from_eye_folder": int((m.get("side", m.eye.str[0]) != m.eye.str[0]).sum()),
        "by_phase": {p: {"images": int(r.images), "subjects": int(r.subjects),
                         "official_images": OFFICIAL_IMAGES.get(p),
                         "difference_vs_official": int(r.images - OFFICIAL_IMAGES[p]) if p in OFFICIAL_IMAGES else None,
                         "ratio_vs_official": round(float(r.images / OFFICIAL_IMAGES[p]), 4) if p in OFFICIAL_IMAGES
                         else None} for p, r in phase_counts.iterrows()},
        "by_device": {k: int(v) for k, v in m.groupby("device").size().items()},
        "by_lighting": {k: int(v) for k, v in m.groupby("lighting").size().items()},
        "by_device_lighting_phase": {f"{d} | {l} | {p}": int(v) for (d, l, p), v in
                                     m.groupby(["device", "lighting", "phase"]).size().items()},
        "classes300": {"subjects": int(m[m.in_300].subject.nunique()), "images": int(m.in_300.sum())},
        "outside_300": {"subjects": int(m[~m.in_300].subject.nunique()), "images": int((~m.in_300).sum())},
        "share_idx_ge_1010": {p: float((g.idx >= HIGH_MIN).mean()) for p, g in m.groupby("phase")},
        "idx_min_max": {p: [int(g.idx.min()), int(g.idx.max())] for p, g in m.groupby("phase")},
        "buckets": {"enrollment": int(len(enr_b)), "verification": int(len(ver_b)),
                    "enrollment_with_frames_ge_1010": int(len(with_high)),
                    "share_with_1010_but_no_1009": float((with_high.has_1010 & ~with_high.has_1009).mean())
                    if len(with_high) else float("nan"),
                    "verification_share_with_1010_but_no_1009": float((ver_high.has_1010 & ~ver_high.has_1009).mean())
                    if len(ver_high) else float("nan"),
                    "enrollment_low_frames_median": float(enr_b.n_low.median()) if len(enr_b) else float("nan"),
                    "enrollment_high_frames_median": float(with_high.n_high.median()) if len(with_high)
                    else float("nan"),
                    "verification_frames_median": float(ver_b.n.median()) if len(ver_b) else float("nan"),
                    "corr_n_low_n_high": float(with_high[["n_low", "n_high"]].astype(float).corr().iloc[0, 1])
                    if len(with_high) > 2 and with_high.n_low.nunique() > 1 and with_high.n_high.nunique() > 1
                    else float("nan"),
                    "most_common_n_low_n_high": {f"{a}+{b}": int(v) for (a, b), v in combos.items()},
                    "enrollment_total_frames": {str(k): int(v) for k, v in
                                                with_high.n.value_counts().sort_index().items()}},
    }
    return {"counts_by_folder": by_folder, "counts_by_index": by_index, "buckets": buckets}, facts


# --- part: overlap -------------------------------------------------------------------------

def overlap_table(split, copies=None):
    """Share of test and validation images with a close relative in train (test also against
    train + validation, which does not depend on how 02 divided the non-test images).

    Criteria: adjacent (idx +-1, same subject, condition, eye and phase), within_2 (idx +-2),
    same_bucket (any image of that bucket), other_eye_same_frame (the other eye's crop of the
    same frame) and, with *copies* (query relpath -> source relpath of near/transformed copies;
    an empty dict when the synthetic part ran and found none), copy_relation (the image is a copy
    of, or the source of, a reference image).
    """
    s = split.copy()
    s["relpath"] = s.relpath.astype(str).str.replace("\\", "/", regex=False)
    s["group"] = np.where(s.phase == "verification", "verification", idx_range(s.phase, s.idx))
    sources, copy_of = {}, {}
    for query, source in (copies or {}).items():
        copy_of[query] = source
        sources.setdefault(source, set()).add(query)
    rows = []
    for subset, references in (("test", ("train",)), ("test", ("train", "validation")), ("validation", ("train",))):
        ref = s[s.split.isin(references)]
        frames = set(zip(ref.subject, ref.condition, ref.eye, ref.phase, ref.idx))
        buckets = set(zip(ref.subject, ref.condition, ref.eye, ref.phase))
        ref_paths = set(ref.relpath)
        sub = s[s.split == subset]
        other = {"left": "right", "right": "left"}
        found = {"adjacent": [], "within_2": [], "same_bucket": [], "other_eye_same_frame": [], "copy_relation": []}
        for r in sub.itertuples(index=False):
            b = (r.subject, r.condition, r.eye, r.phase)
            found["adjacent"].append(b + (r.idx - 1,) in frames or b + (r.idx + 1,) in frames)
            found["within_2"].append(any(b + (r.idx + d,) in frames for d in (-2, -1, 1, 2)))
            found["same_bucket"].append(b in buckets)
            found["other_eye_same_frame"].append((r.subject, r.condition, other.get(r.eye), r.phase, r.idx) in frames)
            found["copy_relation"].append(copy_of.get(r.relpath) in ref_paths or
                                          bool(sources.get(r.relpath, set()) & ref_paths))
        if copies is None:
            found.pop("copy_relation")
        flags = pd.DataFrame(found, index=sub.index)
        for criterion in flags:
            for group, idx in [("all", sub.index)] + [(g, gi.index) for g, gi in sub.groupby("group", sort=True)]:
                rows.append({"subset": subset, "reference": "+".join(references), "criterion": criterion,
                             "group": group, "n": int(len(idx)), "count": int(flags.loc[idx, criterion].sum()),
                             "share": float(flags.loc[idx, criterion].mean()) if len(idx) else float("nan")})
    return pd.DataFrame(rows)


# --- part: areas ---------------------------------------------------------------------------

def load_like_runner(path):
    """(256, 256, 3) float32 in [0, 1], exactly as 05's load_image (tf.keras.utils.load_img: PIL,
    RGB, nearest-neighbour resize to 256 x 256, then / 255), without importing TensorFlow."""
    with open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0


def relpath_from_link(link_path, eye=None):
    """Data-root relpath of a split-tree link <split>/<subject>/<condition>__<phase>__<file> (02);
    None if the name does not follow that pattern."""
    path = str(link_path).replace("\\", "/")
    parts = os.path.basename(path).split("__")
    found = manifest_script.FNAME_RE.match(parts[-1]) if len(parts) == 3 else None
    if found is None:
        return None
    condition, phase, fname = parts
    if eye not in ("left", "right"):
        eye = "left" if found.group("side").lower() == "l" else "right"
    return f"{condition}/{phase}_{eye}_eye/{os.path.basename(os.path.dirname(path))}/{fname}"


def relpath_under_root(path, root):
    """Relpath (forward slashes) of the file *path* points to (symlinks followed) if it lies under
    *root*, else None."""
    if not root or not isinstance(path, str) or not os.path.exists(path):
        return None
    real, real_root = os.path.realpath(path), os.path.realpath(root)
    try:
        inside = os.path.normcase(os.path.commonpath([real, real_root])) == os.path.normcase(real_root)
    except ValueError:                          # different drives
        return None
    return os.path.relpath(real, real_root).replace(os.sep, "/") if inside else None


def attach_relpaths(images, split_manifest=None, root=None):
    """The images CSV with a relpath column: kept if present; else from 02's manifest (by
    link_path); else the real path of the link under the data root; else parsed from the link
    name. Returns (frame, {source: count})."""
    images = images.copy()
    if "relpath" in images and images.relpath.notna().all():
        images["relpath"] = images.relpath.astype(str).str.replace("\\", "/", regex=False)
        return images, {"csv": int(len(images))}
    images["relpath"] = [relpath_from_link(lp, e) for lp, e in
                         zip(images.link_path, images.eye if "eye" in images else [None] * len(images))]
    source = pd.Series(np.where(images.relpath.isna(), "unknown", "link name"), index=images.index)
    under_root = pd.Series([relpath_under_root(lp, root) for lp in images.link_path], index=images.index)
    images.loc[under_root.notna(), "relpath"] = under_root[under_root.notna()]
    source[under_root.notna()] = "data root"
    if split_manifest is not None and {"link_path", "relpath"} <= set(split_manifest.columns):
        relpaths = split_manifest.relpath.astype(str).str.replace("\\", "/", regex=False)
        hit = images.link_path.map(dict(zip(split_manifest.link_path, relpaths)))
        images.loc[hit.notna(), "relpath"] = hit[hit.notna()]
        source[hit.notna()] = "manifest"
    return images, {k: int(v) for k, v in source.value_counts().items()}


def image_areas(path):
    """Area share of each heuristic region on the 8 x 8 and 16 x 16 grids, plus the mask QC, for one
    image loaded like 05 (so the masks are the ones 05 used for its region masses)."""
    masks, info = get_concept_masks(load_like_runner(path), return_info=True)
    g8, g16 = masks_to_grid(masks, grid=8), masks_to_grid(masks, grid=16)
    row = {f"area_{r}": float(g8[k].mean(dtype=np.float64)) for k, r in enumerate(REGIONS)}
    row.update({f"area16_{r}": float(g16[k].mean(dtype=np.float64)) for k, r in enumerate(REGIONS)})
    row.update({k: info[k] for k in QC_KEYS})
    return row, g8, g16


def _areas_chunk(paths):
    return [image_areas(p) for p in paths]


def run_areas(images, paths, workers):
    """Contract-3 table (relpath, link_path, mask_source, area_<region>, fallback_used, circle_score,
    plus area16_<region> and the other mask QC fields) and the per-cell grids."""
    chunks = [paths[i:i + AREAS_CHUNK] for i in range(0, len(paths), AREAS_CHUNK)]
    results = [r for chunk in run_tasks(_areas_chunk, chunks, workers, "areas") for r in chunk]
    table = pd.DataFrame([r[0] for r in results])
    table.insert(0, "mask_source", "heuristic")
    table.insert(0, "link_path", images.link_path.to_numpy())
    table.insert(0, "relpath", images.relpath.to_numpy())
    grids8 = np.stack([r[1] for r in results]).astype(np.float32)
    grids16 = np.stack([r[2] for r in results]).astype(np.float32)
    return table, grids8, grids16


def resolve_image_paths(images, root):
    """link_path if it exists, else <root>/<relpath> (05's resolve_path rule); None if neither."""
    out = []
    for link, rel in zip(images.link_path, images.relpath):
        if isinstance(link, str) and os.path.exists(link):
            out.append(link)
        elif root and isinstance(rel, str) and os.path.exists(data_path(root, rel)):
            out.append(data_path(root, rel))
        else:
            out.append(None)
    return out


# --- report --------------------------------------------------------------------------------

def _pct(x, digits=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f} %"


def report_lines(summary):
    """report.md: the headline numbers in plain sentences."""
    lines = ["# Dataset check (scripts/16_check_dataset.py)", "",
             f"Code commit {summary['git_commit']}{' (with uncommitted changes)' if summary.get('git_dirty') else ''}, "
             f"run {summary['created']}, {summary['minutes_total']} min. Data root: `{summary.get('data_root')}`.", ""]
    c = summary.get("counts")
    if c:
        enr, ver = c["by_phase"].get("enrollment", {}), c["by_phase"].get("verification", {})
        b = c["buckets"]
        lines += ["## 1. Counts", "",
                  f"- {c['images']:,} images of {c['subjects']} subjects (official VISOB 1.0: {OFFICIAL_SUBJECTS} "
                  f"subjects; the base paper lists 585 classes).",
                  f"- Enrollment {enr.get('images', 0):,} images against {OFFICIAL_IMAGES['enrollment']:,} official "
                  f"({enr.get('difference_vs_official', 0):+,}); verification {ver.get('images', 0):,} against "
                  f"{OFFICIAL_IMAGES['verification']:,} ({ver.get('difference_vs_official', 0):+,}).",
                  f"- The 300 classes of 02 hold {c['classes300']['images']:,} images of "
                  f"{c['classes300']['subjects']} subjects; {c['outside_300']['images']:,} images of "
                  f"{c['outside_300']['subjects']} other subject(s) are outside them.",
                  f"- Frames numbered >= {HIGH_MIN}: {_pct(c['share_idx_ge_1010'].get('enrollment'))} of enrollment, "
                  f"{_pct(c['share_idx_ge_1010'].get('verification'))} of verification images.",
                  f"- {b['enrollment_with_frames_ge_1010']:,} of {b['enrollment']:,} enrollment buckets have frames "
                  f">= {HIGH_MIN}; {_pct(b['share_with_1010_but_no_1009'])} of them have frame {HIGH_MIN} but no "
                  f"frame {LOW_MAX} (verification: {_pct(b['verification_share_with_1010_but_no_1009'])}). A genuine "
                  f"continuous sequence always has {LOW_MAX} before {HIGH_MIN}. Most common (frames <= {LOW_MAX}) + "
                  f"(frames >= {HIGH_MIN}) per bucket: {', '.join(list(b['most_common_n_low_n_high'])[:5])}.", ""]
    s = summary.get("synthetic")
    if s and s.get("frames_scored"):
        thr, ps, tr = s["thresholds"], s["pairs"], s["transforms_of_transformed_copies"]
        cs = s["class_share"]
        lines += ["## 2. Are the frames >= 1010 synthetic?", "",
                  f"**{s['verdict']}.**", "",
                  f"- {s['frames_scored']:,} frames >= {HIGH_MIN} in {s['buckets_selected']:,} buckets (of "
                  f"{s['buckets_with_frames_ge_1010_total']:,}) were matched against the frames <= {LOW_MAX} of their "
                  f"bucket: near copy {_pct(cs['near_copy'])}, transformed copy {_pct(cs['transformed_copy'])}, "
                  f"distinct {_pct(cs['distinct'])}.",
                  f"- The same rule flags {_pct(s['genuine_flag_rate'])} of genuine frames (verification "
                  f"{_pct(s['genuine_flag_rate_by_role']['genuine_verification'])}, enrollment <= {LOW_MAX} "
                  f"{_pct(s['genuine_flag_rate_by_role']['genuine_enrollment'])}). The copy share is "
                  f"{_pct(s['copy_share_verification_thresholds'])} with thresholds from verification frames only, "
                  f"and {_pct(s['copy_share_consecutive_pair_thresholds'])} with thresholds from single consecutive "
                  "genuine pairs (lower thresholds, so more frames count as copies).",
                  f"- Thresholds ({int(100 * s['ref_quantile'])}th percentile of genuine frames): NCC as is "
                  f"{thr['ncc_id']:.4f}, SSIM as is {thr['ssim_id']:.4f}, best NCC {thr['ncc_best']:.4f}, "
                  f"SSIM after alignment {thr['ssim_best']:.4f}."]
        if tr["n"]:
            lines.append(f"- Transformed copies ({tr['n']:,}): flipped {_pct(tr['flip'])}, rotated "
                         f"{_pct(tr['rotation'])}, zoomed {_pct(tr['zoom'])}, shifted {_pct(tr['shift'])}, brightness "
                         f"changed by > 5 levels {_pct(tr['brightness_over_5_levels'])}, contrast by > 5 % "
                         f"{_pct(tr['contrast_over_5pct'])}.")
        tg = s.get("transform_gain")
        if tg and tg["frames_ge_1010"]["n"]:
            q, g = tg["frames_ge_1010"], tg["genuine"]
            lines.append(f"- Every scored frame, copy or not: a flip, rotation or zoom adds at least "
                         f"{tg['min_gain']:g} to the best NCC of shifts alone for {_pct(q['needs_transform'])} of "
                         f"frames >= {HIGH_MIN} "
                         f"(flipped {_pct(q['flip'])}, rotated {_pct(q['rotation'])}, zoomed {_pct(q['zoom'])} of "
                         f"those), against {_pct(g['needs_transform'])} of genuine frames. Natural motion is mostly a "
                         "shift, so a much higher share points to transformed copies even below the copy thresholds.")
        if s["copies_per_source_frame"]:
            lines.append(f"- Copies per source frame (number of copies: number of source frames): "
                         f"{', '.join(f'{k}: {v}' for k, v in s['copies_per_source_frame'].items())}.")
        lines.append(f"- Best match is the last genuine frame of the bucket for "
                     f"{_pct(s['best_match_is_last_genuine_frame'])} of frames >= {HIGH_MIN} (a sequence that simply "
                     f"continues would mostly match its last frame).")
        if ps and ps.get("consecutive_high", {}).get("n"):
            coherence = s["time_coherence_of_frames_ge_1010"]
            lines.append(f"- Time structure (median best NCC): consecutive genuine frames "
                         f"{ps['consecutive_low']['ncc_best_median']:.3f} (enrollment) / "
                         f"{ps['consecutive_verification']['ncc_best_median']:.3f} (verification), random genuine "
                         f"pairs {ps['random_low']['ncc_best_median']:.3f}, consecutive frames >= {HIGH_MIN} "
                         f"{ps['consecutive_high']['ncc_best_median']:.3f}, enrollment vs verification "
                         f"{ps['cross_session']['ncc_best_median']:.3f}. "
                         + (f"On a scale where random pairs are 0 and consecutive genuine frames 1, consecutive frames "
                            f">= {HIGH_MIN} sit at {coherence:.2f}." if np.isfinite(coherence) else
                            f"Consecutive and random genuine pairs differ by less than {MIN_TIME_SPAN:g}, so there is "
                            "no time structure to compare with."))
        for stat in ("sharpness_norm", "noise", "brightness", "contrast", "file_bytes"):
            a = s["image_stats_paired"].get(f"{stat} | enrollment >={HIGH_MIN} vs enrollment <={LOW_MAX}")
            v = s["image_stats_paired"].get(f"{stat} | verification vs enrollment <={LOW_MAX}")
            if a:
                yardstick = f"; verification / enrollment <= {LOW_MAX}: {v['median_ratio']:.3f}" if v else ""
                lines.append(f"- {stat}: frames >= {HIGH_MIN} / frames <= {LOW_MAX} of the same bucket, median ratio "
                             f"{a['median_ratio']:.3f} (higher in {_pct(a['share_a_higher'], 0)} of "
                             f"{a['n_buckets']:,} buckets, Wilcoxon p = {a['wilcoxon_p']:.2g}){yardstick}.")
        d = s["exact_duplicates"]
        fingerprints = "; ".join(f"{name}: {next(iter(v['chunks']), '?')}, largest IDAT "
                                 f"{next(iter(v['largest_idat']), '?')}"
                                 for name, v in s.get("png_by_range", {}).items())
        lines += [f"- Exact pixel duplicates: {d['exact_pairs']} pairs, exact mirror images: {d['mirror_pairs']} "
                  f"pairs ({d['pairs_across_subjects']} across subjects, {d['pairs_involving_frames_ge_1010']} "
                  f"involving frames >= {HIGH_MIN}). Image sizes: "
                  f"{', '.join(f'{k}: {v:,}' for k, v in s['image_sizes'].items())}.",
                  f"- Most common PNG layout per range (a different writer shows up here): {fingerprints}.",
                  f"- Contact sheets: `contact_sheets/` ({len(s['contact_sheets'])} files); distributions: "
                  f"`fig_similarity.png`.", ""]
    o = summary.get("overlap")
    if o:
        lines += ["## 3. Near-duplicate overlap between splits", "", f"Split manifest: `{o['split_manifest']}`.", ""]
        for key, text in (("test vs train", "Test images"), ("test vs train+validation", "Test images, against train "
                                                                                          "+ validation (exact for the "
                                                                                          "checkpoint),"),
                          ("validation vs train", "Validation images")):
            v = o["headline"].get(key)
            if v:
                lines.append(f"- {text}: adjacent frame {_pct(v['adjacent'])}, within 2 frames {_pct(v['within_2'])}, "
                             f"any image of the same bucket {_pct(v['same_bucket'])}, other eye of the same frame "
                             f"{_pct(v['other_eye_same_frame'])}"
                             + (f", copy relation {_pct(v['copy_relation'])}" if "copy_relation" in v else "") + ".")
        scored = o.get("copy_relation_buckets")
        if scored and scored[0] < scored[1]:
            lines.append(f"- The copy relation only knows the copies in the {scored[0]:,} of {scored[1]:,} buckets "
                         "this run scored, so it understates the leak.")
        lines.append("")
    a = summary.get("areas")
    if a:
        lines += ["## 4. Per-image region areas", "",
                  f"- {a['images']:,} images of `{a['images_csv']}` -> `{a['output']}` (relpath from "
                  f"{', '.join(f'{k}: {v}' for k, v in a['relpath_source'].items())}).",
                  "- Mean area share: " + ", ".join(f"{r} {a['mean_area'][r]:.4f}" for r in REGIONS)
                  + f"; Hough fallback used for {_pct(a['fallback_rate'])} of images; largest 8 x 8 vs 16 x 16 area "
                    f"difference {a['max_grid_difference']:.1e} (the shares do not depend on the grid).",
                  ]
        if a.get("frames_ge_1010") is not None:
            lines.append(f"- {a['frames_ge_1010']} of these images are enrollment frames >= {HIGH_MIN}"
                         + (f"; {a['copies']} of them were classified as near or transformed copies"
                            if a.get("copies") is not None else "") + ".")
        lines.append("")
    lines += ["## How to read this", "",
              f"- **Synthetic copies** show up as many frames >= {HIGH_MIN} that match a frame <= {LOW_MAX} far "
              "better than genuine frames ever match each other (copy share many times the genuine flag rate), "
              "through transforms typical of augmentation (flips, rotations, zooms, brightness changes), with "
              "each source frame reused, and as visibly identical content in `contact_sheets/pairs_*.png`. "
              "When genuine frames of a sequence are almost identical, the thresholds sit so close to 1 that "
              "interpolated copies (rotated, zoomed) fall just below them: then the copies show up as a share of "
              "frames needing a flip, rotation or zoom far above that of genuine frames. "
              "Systematic differences in sharpness_norm (resampling blurs) or noise (added noise) between the two "
              "frame ranges of the same bucket, and numbering that restarts at 1010 whatever the number of genuine "
              "frames, support this.",
              f"- **Genuine extra frames** look like this: copy share close to the genuine flag rate, about as few "
              f"frames needing a flip, rotation or zoom as genuine frames, consecutive frames >= {HIGH_MIN} about as "
              "similar as consecutive genuine frames (time structure near 1), image statistics with ratios near 1, "
              "and contact sheets showing natural changes (gaze, blinks, small head motion).",
              "- **Genuine in-between video frames** would also match closely, but only through small natural "
              "motion (shifts, rotations of a few degrees, no flips), with unchanged sharpness, noise and PNG "
              "layout. They are not synthetic, but they leak just the same.",
              "- **Distinct frames with shifted statistics** (or time structure near 0) mean transforms outside "
              "the search grid (large rotations or crops) or frames from another capture: look at the contact "
              "sheets before deciding.",
              f"- If copies are found, frames >= {HIGH_MIN} must be dropped from every test set, and the overlap "
              "section's copy relation tells how much the current split leaks through them.", ""]
    return lines


# --- main ----------------------------------------------------------------------------------

def _floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="CPU checks of the Kaggle VISOB copy (no model needed).")
    ap.add_argument("--parts", default=None,
                    help="comma-separated subset of counts,synthetic,overlap,areas. Default: counts, synthetic "
                         "and overlap, plus areas when --images-csv is given")
    ap.add_argument("--root", default=None,
                    help="VISOB_Augmented_Recent folder (default: searched under /kaggle/input)")
    ap.add_argument("--manifest", default="/kaggle/working/manifest_raw.csv",
                    help="01's manifest; built from --root if the file does not exist")
    ap.add_argument("--split-manifest", default="/kaggle/working/manifest_split.csv",
                    help="02's manifest_split.csv (overlap; also gives exact relpaths for areas)")
    ap.add_argument("--images-csv", default=None,
                    help="images_<tag>.csv of a 05 run or correct_test_<tag>.csv of 04 (link_path, optionally "
                         "relpath): per-image region areas for these images")
    ap.add_argument("--out-dir", default="/kaggle/working/data_check")
    ap.add_argument("--max-buckets", type=int, default=None,
                    help="smoke run: score only this many random enrollment buckets with frames >= 1010")
    ap.add_argument("--max-images", type=int, default=None, help="smoke run: areas for the first N images only")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="processes for the synthetic and areas parts (Kaggle CPU sessions have 4 cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-quantile", type=float, default=0.99,
                    help="percentile of genuine frames' scores used as the copy threshold")
    ap.add_argument("--max-shift", type=int, default=2, help="largest shift searched, in 64 x 96 thumbnail pixels "
                                                               "(2 = 5 native pixels)")
    ap.add_argument("--angles", default="-20,-10,0,10,20", help="rotations searched, degrees (refined by half "
                                                                "and quarter steps around the best)")
    ap.add_argument("--scales", default="0.9,1,1.1", help="zoom factors searched (refined like --angles)")
    ap.add_argument("--pairs-per-bucket", type=int, default=3,
                    help="reference pairs of each kind (consecutive, random, enrollment vs verification) drawn "
                         "per bucket")
    ap.add_argument("--contact-sheets", type=int, default=10, help="number of random buckets drawn as contact sheets")
    args = ap.parse_args()
    args.angles, args.scales = _floats(args.angles), _floats(args.scales)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()] if args.parts else \
        ["counts", "synthetic", "overlap"] + (["areas"] if args.images_csv else [])
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; choose from {list(PARTS)}")

    # check every input before the long part starts
    problems, root = [], args.root
    manifest_needed = bool({"counts", "synthetic"} & set(parts))
    if "synthetic" in parts or (manifest_needed and not os.path.exists(args.manifest)):
        try:
            root = root or manifest_script.find_root()
        except FileNotFoundError as e:
            problems.append(f"data root: {e} (pass --root)")
        if root and not os.path.isdir(root):
            problems.append(f"data root {root} is not a folder")
    if "overlap" in parts and not os.path.exists(args.split_manifest):
        problems.append(f"overlap needs 02's manifest: {args.split_manifest} not found (run 01 and 02, or pass "
                        "--split-manifest)")
    if "areas" in parts and not (args.images_csv and os.path.exists(args.images_csv)):
        problems.append(f"areas needs --images-csv (got {args.images_csv})")
    if "areas" in parts and not root:
        try:
            root = manifest_script.find_root()      # areas only uses it when a link_path is missing
        except FileNotFoundError:
            pass
    if not 0 < args.ref_quantile < 1:
        problems.append(f"--ref-quantile must be between 0 and 1 (got {args.ref_quantile})")
    if "synthetic" in parts:
        try:
            make_search(args.angles, args.scales, args.max_shift)
        except ValueError as e:
            problems.append(f"search grid: {e}")
    split = None
    if os.path.exists(args.split_manifest) and {"overlap", "areas"} & set(parts):
        split = pd.read_csv(args.split_manifest)
    area_images = area_paths = relpath_source = None
    if "areas" in parts and args.images_csv and os.path.exists(args.images_csv):
        # resolve every image now: a missing file must not end the run after the synthetic part
        area_images = pd.read_csv(args.images_csv)
        if args.max_images:
            area_images = area_images.head(args.max_images)
        if "link_path" not in area_images:
            problems.append(f"areas: {args.images_csv} has no link_path column")
        else:
            area_images, relpath_source = attach_relpaths(area_images, split, root)
            area_paths = resolve_image_paths(area_images, root)
            missing = [lp for lp, p in zip(area_images.link_path, area_paths) if p is None]
            if missing:
                problems.append(f"areas: {len(missing)} of {len(area_images)} images not found, e.g. {missing[0]}; "
                                "run 01 and 02 in this session or pass --root")
    if problems:
        raise SystemExit("cannot start:\n  " + "\n  ".join(problems))

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    commit, dirty = git_commit()
    summary = {"script": "scripts/16_check_dataset.py", "git_commit": commit, "git_dirty": dirty,
               "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "parts": parts,
               "args": {k: v for k, v in vars(args).items()}, "data_root": root, "minutes": {}}
    print(f"parts: {parts} | commit {commit}{' (dirty)' if dirty else ''} | out: {args.out_dir}", flush=True)

    manifest = None
    if manifest_needed:
        if os.path.exists(args.manifest):
            manifest, source = pd.read_csv(args.manifest), args.manifest
        else:
            manifest, skipped = manifest_script.build(root)
            source = f"built from {root} ({len(skipped)} files did not match the name pattern)"
        manifest["relpath"] = manifest.relpath.astype(str).str.replace("\\", "/", regex=False)
        summary["manifest"] = {"source": source, "rows": int(len(manifest))}
        print(f"manifest: {len(manifest):,} rows ({source})", flush=True)

    if "counts" in parts:
        t0 = time.time()
        print("\n[counts]", flush=True)
        tables, facts = count_dataset(manifest)
        for name, table in tables.items():
            table.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        summary["counts"] = facts
        summary["minutes"]["counts"] = round((time.time() - t0) / 60, 2)
        print(json.dumps({k: facts[k] for k in ("images", "subjects", "by_phase", "classes300", "outside_300",
                                                "share_idx_ge_1010")}, indent=1))

    frames = pd.DataFrame()
    if "synthetic" in parts:
        t0 = time.time()
        print("\n[synthetic]", flush=True)
        frames, facts = run_synthetic(manifest, root, args, args.out_dir)
        summary["synthetic"] = facts
        summary["minutes"]["synthetic"] = round((time.time() - t0) / 60, 2)
        if facts.get("frames_scored"):
            print(f"  verdict: {facts['verdict']} | classes {facts['class_share']} | genuine flag rate "
                  f"{facts['genuine_flag_rate']:.4f}", flush=True)

    copies = None
    if len(frames):
        found = frames[(frames.role == "query") & frames.frame_class.isin(COPY_CLASSES)]
        copies = dict(zip(found.relpath, found.best_ref))

    if "overlap" in parts:
        t0 = time.time()
        print("\n[overlap]", flush=True)
        table = overlap_table(split, copies)
        table.to_csv(os.path.join(args.out_dir, "overlap.csv"), index=False)
        head = table[table.group == "all"]
        summary["overlap"] = {"split_manifest": args.split_manifest, "copy_relation_from_this_run": copies is not None,
                              "headline": {f"{s} vs {r}": dict(zip(g.criterion, g.share))
                                           for (s, r), g in head.groupby(["subset", "reference"], sort=False)}}
        if copies is not None:      # copies are known only for the buckets this run scored
            summary["overlap"]["copy_relation_buckets"] = [summary["synthetic"]["buckets_selected"],
                                                           summary["synthetic"]["buckets_with_frames_ge_1010_total"]]
        summary["minutes"]["overlap"] = round((time.time() - t0) / 60, 2)
        print(head[["subset", "reference", "criterion", "n", "count", "share"]].round(4).to_string(index=False))

    if "areas" in parts:
        t0 = time.time()
        print("\n[areas]", flush=True)
        images, paths = area_images, area_paths
        print(f"  {len(images)} images from {args.images_csv}", flush=True)
        table, grids8, grids16 = run_areas(images, paths, args.workers)
        stem = os.path.splitext(os.path.basename(args.images_csv))[0]
        out_csv = os.path.join(args.out_dir, f"areas_{stem}.csv")
        table.to_csv(out_csv, index=False)
        np.savez_compressed(os.path.join(args.out_dir, f"areas_{stem}_grids.npz"),
                            relpaths=table.relpath.to_numpy(str), link_paths=table.link_path.to_numpy(str),
                            regions=np.array(REGIONS), grid8=grids8, grid16=grids16,
                            meta=np.array(json.dumps({"mask_source": "heuristic", "code_commit": commit,
                                                      "images_csv": args.images_csv,
                                                      "created": summary["created"]})))
        rel = pd.Series(images.relpath.to_numpy())
        idx = rel.str.extract(r"_(\d+)_[lr]_c\.png$", expand=False).astype(float)
        enr_high = rel.str.contains("/enrollment_", regex=False) & (idx >= HIGH_MIN)
        summary["areas"] = {
            "images_csv": args.images_csv, "output": os.path.basename(out_csv), "images": int(len(table)),
            "relpath_source": relpath_source, "mean_area": {r: float(table[f"area_{r}"].mean()) for r in REGIONS},
            "fallback_rate": float(table.fallback_used.mean()),
            "max_grid_difference": float(max((table[f"area_{r}"] - table[f"area16_{r}"]).abs().max() for r in REGIONS)),
            "frames_ge_1010": int(enr_high.sum()),
            "copies": int(rel[enr_high].isin(set(copies or {})).sum()) if copies is not None else None}
        summary["minutes"]["areas"] = round((time.time() - t0) / 60, 2)
        print(f"  wrote {out_csv} | mean areas {summary['areas']['mean_area']} | fallback "
              f"{summary['areas']['fallback_rate']:.1%}", flush=True)

    summary["minutes_total"] = round((time.time() - t_start) / 60, 2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines(summary)))
    print(f"\nwrote results to {args.out_dir} in {summary['minutes_total']} min")


if __name__ == "__main__":
    main()
