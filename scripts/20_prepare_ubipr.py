"""
scripts/20_prepare_ubipr.py

UBIPr (UBI Periocular, Padole & Proenca, ICB 2012) as VISOB-like eye crops, for the zero-shot
cross-dataset verification of 15_eval_verification.py --modes x1. CPU only, no model.

The dataset (https://iris.di.ubi.pt/ubipr.tar, CC BY-NC-SA 4.0: non-commercial use, cite Padole &
Proenca ICB 2012 and UBIRIS.v2, Proenca et al. TPAMI 2010; never publish images or crops) holds
UBIPeriocular/C<class>_S<session>_I<n>.jpg with one annotation C..._S.._I...txt each (macOS '._*'
files are ignored). Canon EOS 5D, visible light, 4-8 m; every image is one eye, rescaled by the
dataset authors to one size per distance (1001 x 801 at 4 m ... 501 x 401 at 8 m).

Annotation format (checked on all 10,199 files of the archive)
--------------------------------------------------------------
A header line of 25 ';'-separated names (CurrentDir;FileName;Size;Gender;CamDist;<S>PoseAngle;...;
<S>EyeSizeY; with the side S = L or R), then one value per line, each ending in ';'. 'Size' is the
size of the original 4368 x 2912 photograph as three values (2912, 4368, 3: rows, columns,
channels), so there are 27 values for 25 names; parse_annotation aligns them by giving Size every
extra value. Coordinates are (row, column): <S>...PtX is the row and <S>...PtY the column, and
<S>EyeSizeX / <S>EyeSizeY are the image's height / width (true for every file). The manifest stores
them as x = column, y = row.

Eye side and identity (see the summary JSON)
--------------------------------------------
- Odd class IDs are 'L' eyes and even ones 'R' (the header's side), except one image each in
  classes 197, 207 and 328. In this dataset 'L' is the eye with the nose on the crop's right (the
  eye on the left of the photograph; the subject's right eye unless the photograph was mirrored).
  The manifest's eye is the class's majority side; images whose header disagrees are kept but
  marked unusable (side_conflict: they show the other eye or another person).
- Classes 2k-1 and 2k are the two eyes of one person: the crops of one photograph (same capture
  date folder and IMG file) sit in both classes for 5,076 of the 5,083 photographs with two crops.
- Misfiled crops (find_misfiled; unusable): each person was photographed in one run of consecutive
  camera frames per session, so a crop whose frame lies outside every run of its class and eye
  partner on that date belongs to someone else. 12 in the archive: 5 that share a photograph with
  another person's class (C263_S1_I5, C397_S1_I3, C411_S1_I3, C434_S2_I3, C457_S1_I7) and 7 from one
  run of an unlabelled person (2007_12_17, IMG_3488-3502: C32_S1_I6, C39_S2_I5, C40_S2_I5,
  C207_S1_I8, C328_S1_I1, C473_S1_I5, C521_S2_I1), several of them the only session-2 probe of
  their class.
- person_id is the smallest class ID among the classes linked by shared photographs of crops that
  are not misfiled (union-find), so that 15 never scores two classes of one person as impostors,
  including a person enrolled under two IDs (none remains in the archive of 1 Nov 2026).
- A session-1 image taken on a date of its class's session-2 images is marked unusable
  (session1_on_session2_date): templates must come from another session than the probes.

Crop (VISOB framing)
--------------------
VISOB crops are 240 x 160 (3:2) and the eye corners span about three quarters of their width (the
four real VISOB images of results/examples_full.npz: 0.72-0.79), with the corners at about 0.44 of
the height. Each UBIPr crop is a 3:2 box of width K x the corner-to-corner distance (K = 1.3),
centred horizontally on the midpoint of the two eye corners and placed so that the midpoint sits at
ANCHOR_Y = 0.45 of the box height. The corners, not the iris centre, anchor the box: they do not
move with gaze, and 62 % of UBIPr images look 30-45 degrees sideways. Pixels outside the image
repeat the border (replicate), and the box is resized to 240 x 160 (Lanczos), VISOB's native size,
so that 15's load_image stretches it to 256 x 256 exactly as it does VISOB images. No rotation.
Images whose annotation is missing or invalid (unparseable, eye size different from the image,
a corner or the iris outside the image, corners closer than 16 pixels) are skipped and counted.

Writes to --out-dir:
  crops/<name>.png         the 240 x 160 crops (never commit or publish them)
  ubipr_manifest.csv       one row per crop: relpath, class_id, person_id, eye, session, the
                           annotation attributes, landmarks (x = column, y = row, source pixels),
                           crop box, source size, flags, usable
  ubipr_skipped.csv        images left out, with the reason
  qc_contact_sheet.png     48 random crops plus up to 4 VISOB images (--visob-examples), for humans
  ubipr_summary.json       counts per session, class, distance and eye, skipped, flags, misfiled crops,
                           person links, crop parameters, git commit, timing

Usage:
  python scripts/20_prepare_ubipr.py --source /kaggle/working/ubipr.tar --out-dir /kaggle/working/ubipr
  smoke run: add --max-images 200
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME_RE = re.compile(r"(?:^|[/\\])(?P<stem>C(?P<cls>\d+)_S(?P<session>\d+)_I(?P<idx>\d+))\.(?P<ext>jpe?g|txt)$",
                     re.IGNORECASE)
DATE_RE = re.compile(r"(\d{4}_\d{2}_\d{2})")
OUT_SIZE = (240, 160)                  # VISOB's native (width, height)
ASPECT = OUT_SIZE[0] / OUT_SIZE[1]     # 3:2
K = 1.3                                # crop width / corner-to-corner distance
ANCHOR_Y = 0.45                        # height fraction of the box where the corner midpoint sits
MIN_CORNER_DIST = 16                   # pixels
FRAME_WINDOW = 15                      # frames: one capture run of a person (15 photographs per session)
# eye-specific annotation fields (after the side prefix), in file order
EYE_FIELDS = ("PoseAngle", "GazeAngle", "Pigm", "EyeClosure", "HairOcclusion", "Glass",
              "CornerOutPtX", "CornerOutPtY", "IrisCenterPtX", "IrisCenterPtY", "CornerInPtX", "CornerInPtY",
              "EyebrowOutPtX", "EyebrowOutPtY", "EyebrowMidPtX", "EyebrowMidPtY", "EyebrowInPtX", "EyebrowInPtY",
              "EyeSizeX", "EyeSizeY")
COMMON_FIELDS = ("CurrentDir", "FileName", "Size", "Gender", "CamDist")
POINTS = ("corner_out", "iris", "corner_in", "brow_out", "brow_mid", "brow_in")
POINT_FIELDS = dict(zip(POINTS, ("CornerOut", "IrisCenter", "CornerIn", "EyebrowOut", "EyebrowMid", "EyebrowIn")))
FLAGS_UNUSABLE = ("side_conflict", "session1_on_session2_date", "misfiled")


def git_commit():
    """HEAD of the repo, with '-dirty' if tracked files have uncommitted changes; 'unknown' without git."""
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        return (commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")) or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- annotations

def _int(value, name):
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"{name} is not an integer: {value!r}") from None


def parse_annotation(text):
    """One UBIPr annotation as a dict; ValueError if it is not in the expected format.

    Returns current_dir, file_name, capture_date, source_size (rows, cols, channels of the original
    photograph), gender, cam_distance, side ('L' / 'R'), pose, gaze, pigmentation, eye_closure,
    hair_occlusion, glasses, eye_size (height, width of the crop image) and, per landmark of
    POINTS, (x, y) = (column, row).
    """
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) < 2:
        raise ValueError("empty annotation")
    names = [n.strip() for n in lines[0].split(";") if n.strip()]
    values = []
    for line in lines[1:]:
        line = line[:-1] if line.endswith(";") else line
        values += [v.strip() for v in line.split(";")]
    if tuple(names[:len(COMMON_FIELDS)]) != COMMON_FIELDS or len(names) != len(COMMON_FIELDS) + len(EYE_FIELDS):
        raise ValueError(f"unexpected header {lines[0][:80]!r}")
    sides = {n[0] for n in names[len(COMMON_FIELDS):]}
    if len(sides) != 1 or sides - {"L", "R"} or [n[1:] for n in names[len(COMMON_FIELDS):]] != list(EYE_FIELDS):
        raise ValueError(f"eye fields are not all L... or all R...: {names[len(COMMON_FIELDS):][:3]}")
    extra = len(values) - len(names)            # 'Size' takes every extra value (it is 3 values in UBIPr)
    if extra < 0 or extra > 2:
        raise ValueError(f"{len(values)} values for {len(names)} fields")
    size = tuple(_int(v, "Size") for v in values[2:3 + extra])
    rest = dict(zip(names[3:], values[3 + extra:]))
    side = sides.pop()
    eye = {f: rest[side + f] for f in EYE_FIELDS}
    out = {"current_dir": values[0], "file_name": values[1], "source_size": size,
           "gender": rest["Gender"], "cam_distance": _int(rest["CamDist"], "CamDist"), "side": side,
           "pose": _int(eye["PoseAngle"], "PoseAngle"), "gaze": _int(eye["GazeAngle"], "GazeAngle"),
           "pigmentation": eye["Pigm"], "eye_closure": eye["EyeClosure"], "hair_occlusion": eye["HairOcclusion"],
           "glasses": eye["Glass"],
           "eye_size": (_int(eye["EyeSizeX"], "EyeSizeX"), _int(eye["EyeSizeY"], "EyeSizeY"))}
    m = DATE_RE.findall(values[0])
    out["capture_date"] = m[-1] if m else re.split(r"[/\\]", values[0].rstrip("/\\"))[-1]
    for point, field in POINT_FIELDS.items():
        row, col = _int(eye[field + "PtX"], field + "PtX"), _int(eye[field + "PtY"], field + "PtY")
        out[point] = (col, row)
    return out


def check_annotation(ann, width, height):
    """(reason to skip or None, flags kept on the image) for an annotation and its image size."""
    if ann["eye_size"] != (height, width):
        return f"eye size {ann['eye_size']} is not the image's (height, width) ({height}, {width})", []
    inside = {p: 0 <= ann[p][0] < width and 0 <= ann[p][1] < height for p in POINTS}
    for p in ("corner_out", "corner_in", "iris"):
        if not inside[p]:
            return f"{p} {ann[p]} outside the {width} x {height} image", []
    (xo, yo), (xi, yi) = ann["corner_out"], ann["corner_in"]
    if np.hypot(xo - xi, yo - yi) < MIN_CORNER_DIST:
        return f"eye corners closer than {MIN_CORNER_DIST} pixels", []
    flags = []
    if not all(inside[p] for p in ("brow_out", "brow_mid", "brow_in")):
        flags.append("eyebrow_outside")
    if (ann["side"] == "L") != (xo < xi):          # L: outer corner left of the inner one (all but 4 images)
        flags.append("corner_order_swapped")
    return None, flags


# ----------------------------------------------------------------------- crop

def crop_box(corner_a, corner_b, k=K, anchor_y=ANCHOR_Y, aspect=ASPECT):
    """(x0, y0, w, h) in source pixels: a box of width k x the corner distance and the given aspect, the
    corner midpoint centred horizontally and at anchor_y of the height."""
    (xa, ya), (xb, yb) = corner_a, corner_b
    cx, cy = (xa + xb) / 2.0, (ya + yb) / 2.0
    w = k * float(np.hypot(xa - xb, ya - yb))
    h = w / aspect
    return int(round(cx - w / 2)), int(round(cy - anchor_y * h)), max(1, int(round(w))), max(1, int(round(h)))


def crop_image(img, box, size=OUT_SIZE):
    """The box of an (H, W, 3) uint8 image, border pixels repeated outside it, resized to size (w, h).

    Returns (crop as uint8 (h, w, 3), True if the box reached outside the image).
    """
    x0, y0, w, h = box
    xs, ys = np.arange(x0, x0 + w), np.arange(y0, y0 + h)
    padded = bool(xs[0] < 0 or ys[0] < 0 or xs[-1] >= img.shape[1] or ys[-1] >= img.shape[0])
    patch = img[np.clip(ys, 0, img.shape[0] - 1)][:, np.clip(xs, 0, img.shape[1] - 1)]
    out = Image.fromarray(np.ascontiguousarray(patch)).resize(size, Image.LANCZOS)
    return np.asarray(out), padded


# --------------------------------------------------------------------- source

class Source:
    """The archive's image / annotation pairs, from the tar file itself or an extracted folder."""

    def __init__(self, path):
        self.path, self.tar = path, None
        found = []
        if os.path.isfile(path):
            self.tar = tarfile.open(path)
            for member in self.tar.getmembers():
                if member.isfile():
                    found.append((member.name, member))
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    full = os.path.join(root, f)
                    found.append((os.path.relpath(full, path).replace("\\", "/"), full))
        else:
            raise SystemExit(f"--source {path} is neither a file nor a folder")
        self.pairs, self.ignored = defaultdict(dict), Counter()
        for name, handle in found:
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            m = NAME_RE.search(name)
            if base.startswith("._"):
                self.ignored["macOS resource file ('._*')"] += 1
            elif not m:
                self.ignored["not C<class>_S<session>_I<n>.jpg/.txt"] += 1
            else:
                kind = "txt" if m["ext"].lower() == "txt" else "jpg"
                if kind in self.pairs[m["stem"]]:
                    raise SystemExit(f"{m['stem']}.{kind} occurs twice in {path}")
                self.pairs[m["stem"]][kind] = (name, handle)

    def read(self, handle):
        if self.tar is not None:
            return self.tar.extractfile(handle).read()
        with open(handle, "rb") as f:
            return f.read()


def reason_kind(reason):
    """The kind of a skip reason, for the counts: 'no image', 'annotation', 'corner_in outside the image', ..."""
    if reason.startswith(("annotation:", "image:")):
        return reason.split(":")[0]
    if " outside the " in reason:
        return reason.split(" ")[0] + " outside the image"
    if reason.startswith("eye size"):
        return "eye size is not the image size"
    return reason


def stem_key(stem):
    m = re.match(r"C(\d+)_S(\d+)_I(\d+)$", stem)
    return int(m[1]), int(m[2]), int(m[3])


# ------------------------------------------------------------------ identity

def frame_number(file_name):
    """The camera's frame counter of an IMG_<n>[c<k>].JPG file name, or None."""
    m = re.match(r"IMG_(\d+)", str(file_name).strip(), re.IGNORECASE)
    return int(m[1]) if m else None


def find_misfiled(df, window=FRAME_WINDOW):
    """True for crops filed under the wrong class: photographs outside their person's capture run.

    UBIPr photographed each person in one run of consecutive frames per session and date, and each
    photograph gives one crop to each of the person's two classes (the eye partner: the class it shares
    most photographs with, if that is mutual). A crop is misfiled when no other photograph of its class
    or eye partner on the same date lies within +-window frames of its own (in the archive of 1 Nov
    2026: 12 crops, e.g. C39_S2_I5 / C40_S2_I5, the only session-2 images of classes 39 / 40, come
    from another person's run of 2007_12_17, IMG_3488-3502).
    """
    cls, dates, files = df.class_id.to_numpy(), df.capture_date.to_numpy(), df.file_name.to_numpy()
    shared = Counter()
    for idx in df.groupby(["capture_date", "file_name"]).indices.values():
        classes = sorted(set(cls[idx].tolist()))
        shared.update((a, b) for a in classes for b in classes if a != b)
    best = {}
    for (a, b), n in sorted(shared.items()):
        if n > best.get(a, (None, 0))[1]:
            best[a] = (b, n)
    partner = {a: b for a, (b, _) in best.items()}
    group = {c: min(c, partner[c]) if partner.get(partner.get(c)) == c else c for c in set(cls.tolist())}
    frames = [frame_number(f) for f in files]
    runs = defaultdict(list)
    for c, d, f, n in zip(cls, dates, files, frames):
        if n is not None:
            runs[(group[c], d)].append((n, f))
    return np.array([n is not None and not any(abs(m - n) <= window for m, other in runs[(group[c], d)] if other != f)
                     for c, d, f, n in zip(cls, dates, files, frames)], bool)


def link_persons(df, ignore=None):
    """person_id per row: the smallest class ID among the classes linked by shared photographs.

    Two crops come from the same photograph when they share the capture date folder and the IMG file
    name; such classes belong to one person (the two eyes, or one person under two IDs). Rows where
    ignore is True (misfiled crops) create no link. Returns (person_id Series, list of cross-person
    links (class a, class b, photograph) that join classes other than 2k-1 / 2k).
    """
    parent = {c: c for c in df.class_id.unique()}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    cross = []
    used = df if ignore is None else df[~np.asarray(ignore, bool)]
    for (date, file_name), group in used.groupby(["capture_date", "file_name"]):
        classes = sorted(group.class_id.unique())
        for other in classes[1:]:
            a, b = find(classes[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
            if not (classes[0] % 2 == 1 and other == classes[0] + 1):
                cross.append((int(classes[0]), int(other), f"{date}/{file_name}"))
    return df.class_id.map(lambda c: int(find(c))), cross


def mark_conflicts(df):
    """Class side (majority of its headers), dataset-level flags and usable.

    Flags: side_conflict (header side differs from the class's; unusable), session1_on_session2_date
    (unusable; misfiled crops set no session-2 date), misfiled (set by main from find_misfiled; unusable),
    distance_size_mismatch (the image size is not the usual one of its camera distance), gender_conflict
    (differs from the class's majority).
    """
    side = df.groupby("class_id").header_side.agg(lambda s: s.value_counts().index[0])
    df["eye"] = df.class_id.map(side)
    flags = [list(f) for f in df["flags"]]
    for i in np.flatnonzero((df.header_side != df.eye).to_numpy()):
        flags[i].append("side_conflict")
    misfiled = np.array(["misfiled" in f for f in flags], bool)       # another person's photograph: no date
    s2_dates = df[(df.session == 2).to_numpy() & ~misfiled].groupby("class_id").capture_date.agg(set).to_dict()
    for i, (c, s, d) in enumerate(zip(df.class_id, df.session, df.capture_date)):
        if s == 1 and d in s2_dates.get(c, ()):
            flags[i].append("session1_on_session2_date")
    usual = df.groupby("cam_distance").source_w.agg(lambda s: s.value_counts().index[0]).to_dict()
    for i in np.flatnonzero((df.source_w != df.cam_distance.map(usual)).to_numpy()):
        flags[i].append("distance_size_mismatch")
    gender = df.groupby("class_id").gender.agg(lambda s: s.value_counts().index[0])
    for i in np.flatnonzero((df.gender != df.class_id.map(gender)).to_numpy()):
        flags[i].append("gender_conflict")
    df["flags"] = [";".join(f) for f in flags]
    df["usable"] = [int(not any(u in f for u in FLAGS_UNUSABLE)) for f in flags]
    return df


# ------------------------------------------------------------- contact sheet

def visob_tiles(path, size=OUT_SIZE):
    """Up to 4 VISOB images at their native 240 x 160: the clean images of an examples_<tag>.npz
    (05) or the images in a folder; [] if path is None or missing."""
    if not path or not os.path.exists(path):
        return []
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:4]
        return [np.asarray(Image.open(os.path.join(path, f)).convert("RGB").resize(size, Image.BILINEAR))
                for f in files]
    with np.load(path, allow_pickle=False) as z:
        keys = sorted((k for k in z.files if "|clean|" in k and k.endswith("|image")),
                      key=lambda k: (int(k.split("|")[0]), k))
        seen, tiles = set(), []
        for k in keys:
            idx = k.split("|")[0]
            if idx not in seen:
                seen.add(idx)
                tiles.append(np.asarray(Image.fromarray(z[k]).resize(size, Image.BILINEAR)))
        return tiles[:4]


def contact_sheet(crops, labels, visob, path, title, cols=8):
    """Grid of crops (labelled), then a row of VISOB images, saved as PNG."""
    w, h = OUT_SIZE
    rows = (len(crops) + cols - 1) // cols + (1 if visob else 0)
    sheet = Image.new("RGB", (cols * w, max(rows, 1) * (h + 14) + 18), (252, 252, 251))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), title, fill=(11, 11, 11))
    for k, (img, label) in enumerate(list(zip(crops, labels)) + [(v, "VISOB") for v in visob]):
        r, c = (k // cols, k % cols) if k < len(crops) else ((len(crops) + cols - 1) // cols, k - len(crops))
        x, y = c * w, 18 + r * (h + 14)
        sheet.paste(Image.fromarray(img), (x, y))
        draw.text((x + 3, y + h), label, fill=(82, 81, 78))
    sheet.save(path)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="UBIPr eye crops in VISOB's framing, plus a manifest for 15 --modes x1.")
    ap.add_argument("--source", required=True, help="ubipr.tar, or the folder it was extracted to")
    ap.add_argument("--out-dir", default="/kaggle/working/ubipr")
    ap.add_argument("--k", type=float, default=K, help="crop width / eye-corner distance")
    ap.add_argument("--anchor-y", type=float, default=ANCHOR_Y,
                    help="height fraction of the crop where the eye-corner midpoint sits")
    ap.add_argument("--visob-examples", default=None,
                    help="VISOB images for the contact sheet: an examples_<tag>.npz from 05 or a folder of images "
                         "(left out if missing)")
    ap.add_argument("--sheet-crops", type=int, default=48, help="random crops on the contact sheet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=None, help="smoke run: the first N images (sorted by name)")
    args = ap.parse_args()

    t0 = time.time()
    src = Source(args.source)
    stems = sorted(src.pairs, key=stem_key)
    n_found = {"jpg": sum("jpg" in p for p in src.pairs.values()), "txt": sum("txt" in p for p in src.pairs.values())}
    print(f"{args.source}: {n_found['jpg']} images, {n_found['txt']} annotations | ignored: {dict(src.ignored)}")
    if args.max_images:
        stems = stems[:args.max_images]
    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)

    rows, skipped = [], []
    for n, stem in enumerate(stems, 1):
        pair = src.pairs[stem]
        cls, session, idx = stem_key(stem)
        reason = None
        if "jpg" not in pair:
            reason = "no image"
        elif "txt" not in pair:
            reason = "no annotation"
        if reason is None:
            try:
                ann = parse_annotation(src.read(pair["txt"][1]).decode("latin-1"))
            except ValueError as e:
                reason = f"annotation: {e}"
        if reason is None:
            try:
                img = np.asarray(Image.open(io.BytesIO(src.read(pair["jpg"][1]))).convert("RGB"))
            except Exception as e:                         # truncated or unreadable JPEG
                reason = f"image: {e}"
        if reason is None:
            reason, flags = check_annotation(ann, img.shape[1], img.shape[0])
        if reason is not None:
            skipped.append({"name": stem, "class_id": cls, "session": session, "reason": reason})
            continue
        box = crop_box(ann["corner_out"], ann["corner_in"], k=args.k, anchor_y=args.anchor_y)
        crop, padded = crop_image(img, box)
        relpath = f"crops/{stem}.png"
        Image.fromarray(crop).save(os.path.join(args.out_dir, relpath))
        (xo, yo), (xi, yi) = ann["corner_out"], ann["corner_in"]
        row = {"relpath": relpath, "source": pair["jpg"][0], "class_id": cls, "session": session, "image_idx": idx,
               "header_side": ann["side"], "capture_date": ann["capture_date"], "file_name": ann["file_name"],
               "cam_distance": ann["cam_distance"], "pose": ann["pose"], "gaze": ann["gaze"],
               "pigmentation": ann["pigmentation"], "eye_closure": ann["eye_closure"],
               "hair_occlusion": ann["hair_occlusion"], "glasses": ann["glasses"], "gender": ann["gender"]}
        for p in POINTS:
            row[f"{p}_x"], row[f"{p}_y"] = ann[p]
        row.update({"corner_dist": round(float(np.hypot(xo - xi, yo - yi)), 2),
                    "corner_angle_deg": round(float(np.degrees(np.arctan2(yi - yo, xi - xo))), 2),
                    "crop_x0": box[0], "crop_y0": box[1], "crop_w": box[2], "crop_h": box[3],
                    "crop_padded": int(padded),
                    "source_w": img.shape[1], "source_h": img.shape[0],
                    "original_size": "x".join(map(str, ann["source_size"])), "flags": flags})
        rows.append(row)
        if n % 1000 == 0 or n == len(stems):
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n}/{len(stems)} images | {len(skipped)} skipped | {(time.time() - t0) / 60:.1f} min | "
                  f"about {(len(stems) - n) / rate / 60:.1f} min left", flush=True)
    if not rows:
        raise SystemExit("no usable image found")

    df = pd.DataFrame(rows)
    misfiled = find_misfiled(df)
    df["flags"] = [f + ["misfiled"] if m else f for f, m in zip(df["flags"], misfiled)]
    _, cross_all = link_persons(df)                    # every shared photograph, for the record
    df["person_id"], cross = link_persons(df, ignore=misfiled)
    df = mark_conflicts(df)
    first = ["relpath", "class_id", "person_id", "eye", "session", "usable", "flags"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(os.path.join(args.out_dir, "ubipr_manifest.csv"), index=False)
    pd.DataFrame(skipped, columns=["name", "class_id", "session", "reason"]).to_csv(
        os.path.join(args.out_dir, "ubipr_skipped.csv"), index=False)

    rng = np.random.default_rng(args.seed)
    usable = df[df.usable == 1]
    pick = usable.iloc[np.sort(rng.choice(len(usable), min(args.sheet_crops, len(usable)), replace=False))]
    crops = [np.asarray(Image.open(os.path.join(args.out_dir, r)).convert("RGB")) for r in pick.relpath]
    labels = [f"C{c} {e} S{s} {d} m" for c, e, s, d in zip(pick.class_id, pick.eye, pick.session, pick.cam_distance)]
    visob = visob_tiles(args.visob_examples)
    title = (f"UBIPr crops (20_prepare_ubipr.py, k = {args.k}, anchor_y = {args.anchor_y}, 240 x 160)"
             + (f"; last row: VISOB images from {os.path.basename(args.visob_examples)}" if visob else ""))
    contact_sheet(crops, labels, visob, os.path.join(args.out_dir, "qc_contact_sheet.png"), title)

    flag_counts = Counter(f for fl in df["flags"] for f in fl.split(";") if f)
    per_class = df.groupby("class_id")
    both = per_class.session.nunique() == 2
    summary = {
        "source": os.path.abspath(args.source), "git_commit": git_commit(), "max_images": args.max_images,
        "files": {"images": n_found["jpg"], "annotations": n_found["txt"], "ignored": dict(src.ignored)},
        "n_images_considered": len(stems), "n_crops": len(df), "n_usable": int(df.usable.sum()),
        "n_skipped": len(skipped), "skipped_by_reason": dict(Counter(reason_kind(s["reason"]) for s in skipped)),
        "skipped": [f"{s['name']}: {s['reason']}" for s in skipped][:50],
        "flags": dict(flag_counts),
        "flags_unusable": list(FLAGS_UNUSABLE),
        "images_by_session": {int(k): int(v) for k, v in df.session.value_counts().sort_index().items()},
        "classes_by_session": {int(k): int(v) for k, v in df.groupby("session").class_id.nunique().items()},
        "n_classes": int(df.class_id.nunique()), "n_classes_both_sessions": int(both.sum()),
        "class_id_range": [int(df.class_id.min()), int(df.class_id.max())],
        "images_per_class": {"min": int(per_class.size().min()), "median": float(per_class.size().median()),
                             "max": int(per_class.size().max())},
        "n_persons": int(df.person_id.nunique()),
        "classes_per_person": {int(k): int(v) for k, v in
                               df.groupby("person_id").class_id.nunique().value_counts().sort_index().items()},
        "cross_person_links": [f"C{a} ~ C{b} via {photo}" for a, b, photo in cross],
        "cross_person_photos_incl_misfiled": [f"C{a} ~ C{b} via {photo}" for a, b, photo in cross_all],
        "misfiled": sorted((r for r, m in zip(df.relpath, misfiled) if m), key=lambda r: stem_key(r[6:-4])),
        "frame_window": FRAME_WINDOW,
        "eye_by_class": {e: int(v) for e, v in per_class.eye.first().value_counts().items()},
        "odd_classes_are_L": bool((per_class.eye.first()[per_class.eye.first().index % 2 == 1] == "L").all()),
        "even_classes_are_R": bool((per_class.eye.first()[per_class.eye.first().index % 2 == 0] == "R").all()),
        "images_by_cam_distance": {int(k): int(v) for k, v in df.cam_distance.value_counts().sort_index().items()},
        "source_size_by_cam_distance": {f"{d} m": {f"{w}x{h}": int(n) for (w, h), n in g.groupby(
            ["source_w", "source_h"]).size().items()} for d, g in df.groupby("cam_distance")},
        "images_by_glasses": {k: int(v) for k, v in df.glasses.value_counts().items()},
        "x1_protocol": {"templates (session 1, usable)": int(((df.session == 1) & (df.usable == 1)).sum()),
                        "probes (session 2, usable)": int(((df.session == 2) & (df.usable == 1)).sum()),
                        "classes with both": int(df[df.usable == 1].groupby("class_id").session.nunique().eq(2).sum())},
        "crop": {"centre": "midpoint of the two eye corners", "width": f"{args.k} x corner-to-corner distance",
                 "aspect_w_h": "3:2", "anchor_y": args.anchor_y, "border": "replicate", "rotation": "none",
                 "output_size_w_h": list(OUT_SIZE), "resample": "Lanczos", "padded_crops": int(df.crop_padded.sum()),
                 "min_corner_dist_px": MIN_CORNER_DIST},
        "landmark_convention": "annotation <S>...PtX = row, <S>...PtY = column; <S>EyeSizeX/Y = image height/width; "
                               "manifest x = column, y = row",
        "eye_convention": "UBIPr's own L/R: L has the nose on the crop's right (eye on the left of the photograph)",
        "licence": "UBIPr: CC BY-NC-SA 4.0 (non-commercial); cite Padole & Proenca ICB 2012 and Proenca et al. "
                   "TPAMI 2010 (UBIRIS.v2); never publish the images or crops",
        "visob_examples_on_sheet": len(visob), "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "ubipr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{len(df)} crops ({summary['n_usable']} usable) of {summary['n_classes']} classes / {summary['n_persons']} "
          f"persons | skipped {len(skipped)} | flags {dict(flag_counts)}")
    print(f"x1 protocol: {summary['x1_protocol']}")
    print(f"wrote {args.out_dir} ({summary['seconds']} s)")


if __name__ == "__main__":
    sys.exit(main())
