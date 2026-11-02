"""scripts/16_check_dataset.py on a fake VISOB tree of synthetic eye images with known copies."""
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from src.compare.regions import REGIONS, region_grids


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load("16_check_dataset")
COND, OTHER_COND = "oppo_day_light_short", "iphone_dim_light_short"


# --- synthetic eyes ------------------------------------------------------------------------

def eye(subject, gaze=0.0, lid=1.0, gain=1.0, shift=(0, 0), frame=0, noise=3.0):
    """A 160 x 240 RGB periocular-like image: skin texture, tilted brow, an off-centre eye opening
    with lashes, a textured iris, pupil, highlight and a caruncle at the inner corner (so the mirror
    image is not the same eye). gaze moves the iris, shift the whole image; noise differs per frame."""
    base = np.random.default_rng(subject)
    yy, xx = np.mgrid[0:160, 0:240].astype(np.float32)
    skin = 150 + 60 * cv2.GaussianBlur(base.standard_normal((160, 240)).astype(np.float32), (0, 0), 5)
    img = np.stack([skin * 1.1, skin * 0.85, skin * 0.7], -1)
    img[np.abs(yy - 18 - 0.12 * xx + 0.0015 * (xx - 90) ** 2) < 8] = (60, 45, 35)
    cx, cy = 108.0, 88.0
    u = (xx - cx) / 84
    top = cy - 36 * lid * np.sqrt(np.clip(1 - u ** 2, 0, 1)) * (1 + 0.25 * u)
    opening = (np.abs(u) < 1) & (yy > top) & (yy < cy + 26 * np.sqrt(np.clip(1 - u ** 2, 0, 1)))
    img[opening] = (225, 220, 215)
    img[(np.hypot(xx - (cx - 78), yy - cy) < 9) & ~opening | (np.hypot(xx - (cx - 74), yy - cy) < 6)] = (200, 120, 120)
    ix, iy = cx + gaze, cy - 2
    r, theta = np.hypot(xx - ix, yy - iy), np.arctan2(yy - iy, xx - ix)
    iris = (r < 30) & opening
    texture = 70 + 25 * np.sin(14 * theta + subject) * (r / 30)
    img[iris] = np.stack([texture * 1.2, texture, texture * 0.8], -1)[iris]
    img[(r < 11) & opening] = (15, 15, 15)
    img[(np.hypot(xx - ix - 6, yy - iy + 6) < 3) & opening] = 250
    img[(np.abs(yy - top) < 2) & (np.abs(u) < 1)] = (30, 25, 20)
    move = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img = cv2.warpAffine(img.astype(np.float32), move, (240, 160), borderMode=cv2.BORDER_REFLECT)
    img = img * gain + np.random.default_rng([subject, frame]).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).round().astype(np.uint8)


def genuine(subject, k, session=0):
    """Frame k of a capture sequence: the gaze sweeps, the lid and the light change a little."""
    return eye(subject, gaze=(-16, -8, 0, 8, 16)[k % 5] + subject % 3 - 1,
               lid=1 + 0.05 * ((k + subject) % 3 - 1) - 0.08 * session, gain=1 + 0.02 * (k % 5 - 2) + 0.06 * session,
               shift=((k + subject) % 2, (k + 1) % 2), frame=100 * session + k)


def rotate(img, angle, scale=1.0):
    matrix = cv2.getRotationMatrix2D((119.5, 79.5), angle, scale)
    return cv2.warpAffine(img, matrix, (240, 160), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


EXPECTED = {   # frame >= 1010 of subject 1141 -> (how it was made, allowed classes)
    1010: ("exact copy of 1002", {"near_copy"}),
    1011: ("mirror image of 1003", {"transformed_copy"}),
    1012: ("brighter, more contrast 1004", {"near_copy", "transformed_copy"}),
    1013: ("1005 shifted by (4, 2) px", {"transformed_copy"}),
    1014: ("1001 rotated by 10 degrees", {"transformed_copy"}),
    1015: ("new frame, gaze far left, half-closed lid", {"distinct"}),
    1016: ("new frame, gaze far right", {"distinct"}),
}


def _save(root, condition, phase, eye_side, subject, idx, img):
    d = root / condition / f"{phase}_{eye_side}_eye" / str(subject)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(d / f"{subject}_{idx}_{eye_side[0]}_c.png")


def build_tree(root):
    """VISOB-style tree (91 images). Subject 1141 (oppo day, left eye) has genuine frames 1001-1005
    and the frames >= 1010 of EXPECTED; subjects 1142-1147 have genuine frames only (their 1010, and
    1142's 1011, are new frames far along the gaze sweep); all have verification frames 1001-1005.
    Subject 1141 also has a short iphone-dim sequence, and 9001 sits outside the 300 classes."""
    low = {k: genuine(1141, k) for k in range(5)}
    extra = {
        1010: low[1].copy(),
        1011: np.ascontiguousarray(low[2][:, ::-1]),
        1012: np.clip(low[3].astype(np.float32) * 1.15 + 10, 0, 255).round().astype(np.uint8),
        1013: cv2.warpAffine(low[4], np.float32([[1, 0, 4], [0, 1, 2]]), (240, 160), borderMode=cv2.BORDER_REFLECT),
        1014: rotate(low[0], 10),
        1015: eye(1141, gaze=-30, lid=0.55, gain=0.85, shift=(6, 3), frame=50),
        1016: eye(1141, gaze=28, lid=0.7, gain=1.1, shift=(-5, 2), frame=51),
    }
    for k, img in low.items():
        _save(root, COND, "enrollment", "left", 1141, 1001 + k, img)
    for idx, img in extra.items():
        _save(root, COND, "enrollment", "left", 1141, idx, img)
    for subject in range(1142, 1148):
        for k in range(5):
            _save(root, COND, "enrollment", "left", subject, 1001 + k, genuine(subject, k))
        _save(root, COND, "enrollment", "left", subject, 1010,
              eye(subject, gaze=32 if subject % 2 else -32, lid=0.9, gain=1.05, shift=(3, 0), frame=60))
    _save(root, COND, "enrollment", "left", 1142, 1011, eye(1142, gaze=-32, lid=1.1, gain=0.95, shift=(0, 3), frame=61))
    for subject in range(1141, 1148):
        for k in range(5):
            _save(root, COND, "verification", "left", subject, 1001 + k, genuine(subject, k, session=1))
    for k in range(3):
        _save(root, OTHER_COND, "enrollment", "left", 1141, 1001 + k, genuine(1141, k + 1, session=2))
    _save(root, OTHER_COND, "enrollment", "left", 1141, 1010, eye(1141, gaze=30, lid=0.8, frame=70))
    _save(root, OTHER_COND, "verification", "left", 1141, 1001, genuine(1141, 0, session=3))
    for k in range(2):
        _save(root, COND, "enrollment", "left", 9001, 1001 + k, genuine(9001, k))
    return root


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return build_tree(tmp_path_factory.mktemp("visob") / "VISOB_Augmented_Recent")


@pytest.fixture(scope="module")
def manifest(tree):
    m, skipped = check.manifest_script.build(str(tree))
    assert skipped == []
    m["relpath"] = m.relpath.str.replace("\\", "/", regex=False)
    return m


def _args(**kw):
    base = dict(angles=[-20, -10, 0, 10, 20], scales=[0.9, 1.0, 1.1], max_shift=2, max_buckets=None, seed=0,
                pairs_per_bucket=3, workers=1, ref_quantile=0.99, contact_sheets=2)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture(scope="module")
def synthetic(tree, manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_out")
    frames, facts = check.run_synthetic(manifest, str(tree), _args(), str(out))
    return frames, facts, out


# --- similarity building blocks ------------------------------------------------------------

def test_ssim_matches_scikit_image():
    metrics = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.uniform(0, 1, (52, 84)), (0, 0), 2)
    b = np.clip(0.8 * a + 0.1 + 0.05 * rng.standard_normal(a.shape), 0, 1)
    expected = metrics.structural_similarity(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                             data_range=1.0)
    assert check.ssim(a, b) == pytest.approx(expected, abs=1e-6)
    assert check.ssim(a, a) == pytest.approx(1.0)


def test_match_finds_flip_shift_and_ignores_brightness():
    search = check.make_search()
    thumb = check.thumbnail(genuine(1141, 2))
    ref = check.prepare(thumb, search)
    same = check.match(check.prepare(thumb, search), [ref], search)
    assert same["ncc_id"] == pytest.approx(1.0, abs=1e-5) and same["ssim_id"] == pytest.approx(1.0, abs=1e-6)
    mirrored = check.match(check.prepare(np.ascontiguousarray(thumb[:, ::-1]), search), [ref], search)
    assert mirrored["flip"] and mirrored["ncc_best"] == pytest.approx(1.0, abs=1e-5) and mirrored["ncc_id"] < 0.9
    assert mirrored["ncc_shift"] < 0.9                    # shifts alone cannot undo a flip: large transform gain
    shifted = np.roll(thumb, (1, -2), axis=(0, 1))
    moved = check.match(check.prepare(shifted, search), [ref], search)
    assert (moved["dy"], moved["dx"]) == (-1, 2) and moved["ncc_best"] == pytest.approx(1.0, abs=1e-5)
    assert moved["ncc_shift"] == pytest.approx(1.0, abs=1e-5) and moved["ncc_id"] < 0.99   # a shift needs no transform
    darker = check.match(check.prepare(0.7 * thumb + 0.05, search), [ref], search)
    assert darker["ncc_id"] == pytest.approx(1.0, abs=1e-5) and darker["ssim_best"] == pytest.approx(1.0, abs=1e-4)
    assert darker["brightness_change"] < -5 and darker["contrast_change"] == pytest.approx(0.7, abs=1e-3)
    flat = check.match(check.prepare(np.full_like(thumb, 0.5), search), [ref], search)
    assert np.isfinite(flat["ncc_best"]) and flat["ncc_id"] == 0.0           # constant image: NCC 0, no NaN
    empty = check.match(ref, [], search)
    assert empty["n_refs"] == 0 and np.isnan(empty["ncc_best"])


def test_search_grid_always_contains_the_untouched_image():
    search = check.make_search(angles=[5], scales=[1.2], max_shift=1)
    assert 0.0 in search.angles and 1.0 in search.scales and search.margin >= search.max_shift
    with pytest.raises(ValueError):
        check.make_search(max_shift=20)


def test_png_facts():
    import io
    buf = io.BytesIO()
    Image.fromarray(genuine(1141, 0)).save(buf, format="PNG")
    facts = check.png_facts(buf.getvalue())
    assert facts["png_chunks"].startswith("IHDR") and facts["png_chunks"].endswith("IEND")
    assert facts["png_bit_depth"] == 8 and facts["png_colour_type"] == 2 and facts["png_idat_max"] > 0
    assert check.png_facts(b"not a png at all")["png_chunks"] == "not_png"


# --- part: synthetic -----------------------------------------------------------------------

def test_copies_and_distinct_frames_are_classified(synthetic):
    frames, facts, _ = synthetic
    queries = frames[frames.role == "query"].set_index("relpath")
    for idx, (how, allowed) in EXPECTED.items():
        row = queries.loc[f"{COND}/enrollment_left_eye/1141/1141_{idx}_l_c.png"]
        assert row.frame_class in allowed, f"{idx} ({how}) classified {row.frame_class}"
    assert queries.loc[f"{COND}/enrollment_left_eye/1141/1141_1010_l_c.png", "best_ref_idx"] == 1002
    mirror = queries.loc[f"{COND}/enrollment_left_eye/1141/1141_1011_l_c.png"]
    assert mirror.best_ref_idx == 1003 and bool(mirror.flip)
    rotated = queries.loc[f"{COND}/enrollment_left_eye/1141/1141_1014_l_c.png"]
    assert rotated.best_ref_idx == 1001 and rotated.angle == pytest.approx(-10.0)
    others = queries[~queries.index.str.startswith(f"{COND}/enrollment_left_eye/1141/")]
    assert len(others) == 8 and (others.frame_class == "distinct").all()    # genuine frames numbered >= 1010
    genuine_rows = frames[frames.role.isin(check.GENUINE_ROLES)]
    assert genuine_rows.role.value_counts().to_dict() == {"genuine_enrollment": 38, "genuine_verification": 35}
    assert facts["frames_scored"] == len(queries) == 15
    assert facts["class_share"]["near_copy"] + facts["class_share"]["transformed_copy"] == pytest.approx(5 / 15)
    assert facts["genuine_flag_rate"] < 0.1
    assert facts["verdict"] == "synthetic copies found"

    # what a flip/rotation/zoom adds over shifts alone: nothing for exact or shifted copies, a lot for the
    # mirrored and rotated ones, and (almost) never enough for genuine frames
    def gain(idx):
        row = queries.loc[f"{COND}/enrollment_left_eye/1141/1141_{idx}_l_c.png"]
        return row.ncc_best - row.ncc_shift

    assert (frames.ncc_best >= frames.ncc_shift - 1e-6).all()
    assert gain(1010) < 0.01 and gain(1013) < 0.01 and gain(1011) > 0.1 and gain(1014) > 0.1
    assert facts["transform_gain"]["genuine"]["needs_transform"] < 0.05
    assert facts["transform_gain"]["frames_ge_1010"]["needs_transform"] >= 2 / 15


def test_reference_pairs_duplicates_and_outputs(synthetic):
    frames, facts, out = synthetic
    pairs = pd.read_csv(out / "pairs.csv")
    assert set(pairs.kind) == set(check.PAIR_KINDS)
    consecutive = pairs[pairs.kind == "consecutive_low"]
    assert ((consecutive.idx_a - consecutive.idx_b) == 1).all()
    assert (pairs.loc[pairs.kind == "random_low", "idx_a"] - pairs.loc[pairs.kind == "random_low", "idx_b"] >= 2).all()
    dup = pd.read_csv(out / "duplicates.csv")
    as_pairs = set(zip(dup.kind, dup.relpath_a.str[-17:], dup.relpath_b.str[-17:]))
    assert ("exact", "1141_1002_l_c.png", "1141_1010_l_c.png") in as_pairs
    assert ("mirror", "1141_1003_l_c.png", "1141_1011_l_c.png") in as_pairs
    assert facts["exact_duplicates"]["pairs_involving_frames_ge_1010"] == 2
    summary = pd.read_csv(out / "synthetic_summary.csv")
    assert {"device", "lighting", "idx_band", "near_copy", "transformed_copy", "distinct",
            "genuine_flag_rate"} <= set(summary.columns)
    assert summary.loc[summary.device == "all", "n"].item() == 15
    stats = pd.read_csv(out / "image_stats.csv")
    assert {"brightness", "contrast", "sharpness", "noise", "width", "height", "png_chunks"} <= set(stats.columns)
    assert (stats.width == 240).all() and (stats.height == 160).all()
    paired = pd.read_csv(out / "image_stats_paired.csv")
    assert set(paired.comparison) == {"enrollment >=1010 vs enrollment <=1009", "verification vs enrollment <=1009"}
    assert facts["image_sizes"] == {"240x160 RGB": len(stats)}
    sheets = facts["contact_sheets"]
    assert sum(s.startswith("contact_sheets/bucket_") for s in sheets) == 2
    assert {"contact_sheets/pairs_near_copy.png", "contact_sheets/pairs_distinct.png"} <= set(sheets)
    assert all((out / s).exists() for s in sheets) and (out / "fig_similarity.png").exists()


def test_unreadable_file_is_reported_not_fatal(tmp_path):
    root = tmp_path / "VISOB_Augmented_Recent"
    for k in range(3):
        _save(root, COND, "enrollment", "left", 1150, 1001 + k, genuine(1150, k))
        _save(root, COND, "verification", "left", 1150, 1001 + k, genuine(1150, k, session=1))
    _save(root, COND, "enrollment", "left", 1150, 1010, genuine(1150, 1))
    (root / COND / "enrollment_left_eye" / "1150" / "1150_1011_l_c.png").write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    m, _ = check.manifest_script.build(str(root))
    frames, facts = check.run_synthetic(m, str(root), _args(contact_sheets=1), str(tmp_path / "out"))
    assert facts["images_unreadable"] == 1 and facts["images_loaded"] == 8
    assert frames.loc[frames.role == "query", "idx"].tolist() == [1010]
    stats = pd.read_csv(tmp_path / "out" / "image_stats.csv").set_index("idx")
    assert stats.loc[1011, "error"].startswith("UnidentifiedImageError") and stats.error.notna().sum() == 1
    assert facts["image_sizes"] == {"240x160 RGB": 7}


def test_max_buckets_limits_the_smoke_run(tree, manifest, tmp_path):
    frames, facts = check.run_synthetic(manifest, str(tree), _args(max_buckets=1, contact_sheets=0), str(tmp_path))
    assert facts["buckets_selected"] == 1 and facts["buckets_with_frames_ge_1010_total"] == 8
    assert frames[frames.role == "query"][check.BUCKET].drop_duplicates().shape[0] == 1


def test_verdict_rules():
    verdict = check.synthetic_verdict
    few = {"copy_share": 0.01, "genuine_flag_rate": 0.012}

    def gain(frames_ge_1010, genuine_frames):
        return {"transform_gain": {"frames_ge_1010": {"needs_transform": frames_ge_1010},
                                   "genuine": {"needs_transform": genuine_frames}}}

    assert verdict(few) == "no pixel copies found"
    assert verdict({**few, "copy_share": 0.3}) == "synthetic copies found"
    assert verdict({**few, "copy_share": 0.04}).startswith("unclear")
    assert verdict({"copy_share": float("nan"), "genuine_flag_rate": 0.0}) == "not enough frames to judge"
    moved = verdict({**few, **gain(0.4, 0.01)})            # copies kept just below the thresholds
    assert moved.startswith("unclear") and "flip, rotation or zoom" in moved and "40 %" in moved
    assert verdict({**few, **gain(0.03, 0.01)}) == "no pixel copies found"
    assert verdict({**few, "time_coherence_of_frames_ge_1010": 0.1}).startswith("unclear")
    assert verdict({**few, "time_coherence_of_frames_ge_1010": 0.95}) == "no pixel copies found"
    assert verdict({**few, "copy_share": 0.3, **gain(0.4, 0.0)}) == "synthetic copies found"


def test_copies_below_strict_thresholds_make_the_verdict_unclear(tmp_path):
    """Genuine frames that differ only by sensor noise push the thresholds to ~1, so mirrored and rotated
    copies (interpolated) miss them: the verdict must not read 'no pixel copies found'."""
    root = tmp_path / "VISOB_Augmented_Recent"
    for subject in (1160, 1161, 1162):
        low = [eye(subject, frame=k) for k in range(5)]
        for k, img in enumerate(low):
            _save(root, COND, "enrollment", "left", subject, 1001 + k, img)
            _save(root, COND, "verification", "left", subject, 1001 + k, eye(subject, frame=20 + k))
        for k, angle in enumerate((12, -12, 15, -15)):     # mirrored, then rotated (interpolated)
            copy = rotate(np.ascontiguousarray(low[k][:, ::-1]), angle)
            _save(root, COND, "enrollment", "left", subject, 1010 + k, copy)
    m, _ = check.manifest_script.build(str(root))
    frames, facts = check.run_synthetic(m, str(root), _args(contact_sheets=0), str(tmp_path / "out"))
    assert facts["thresholds"]["ncc_best"] > 0.999 and facts["copy_share"] < 0.05
    assert facts["transform_gain"]["frames_ge_1010"]["needs_transform"] == 1.0
    assert facts["transform_gain"]["genuine"]["needs_transform"] == 0.0
    assert facts["verdict"].startswith("unclear") and "flip, rotation or zoom" in facts["verdict"]
    assert np.isnan(facts["time_coherence_of_frames_ge_1010"])      # no time structure among near-identical frames


# --- part: counts --------------------------------------------------------------------------

def test_counts(manifest):
    tables, facts = check.count_dataset(manifest)
    assert facts["images"] == len(manifest) == 91 and facts["subjects"] == 8
    assert facts["by_phase"]["enrollment"]["images"] == 55
    assert facts["by_phase"]["verification"]["difference_vs_official"] == 36 - 46797
    assert facts["outside_300"] == {"subjects": 1, "images": 2}
    assert facts["classes300"] == {"subjects": 7, "images": 89}
    assert facts["share_idx_ge_1010"]["verification"] == 0.0
    assert facts["by_device"] == {"iphone": 5, "oppo": 86} and facts["by_lighting"] == {"day": 86, "dim": 5}
    b = facts["buckets"]
    assert b["enrollment_with_frames_ge_1010"] == 8
    assert b["share_with_1010_but_no_1009"] == 1.0          # every extra block jumps from <= 1005 to 1010
    assert np.isnan(b["verification_share_with_1010_but_no_1009"])   # no verification frame >= 1010: undefined
    buckets = tables["buckets"].set_index(check.BUCKET + ["phase"])
    row = buckets.loc[(1141, COND, "left", "enrollment")]
    assert (row.n_low, row.n_high, bool(row.low_contiguous), bool(row.high_contiguous)) == (5, 7, True, True)
    assert tables["counts_by_folder"].images.sum() == 91
    index = tables["counts_by_index"].set_index("idx")
    assert index.loc[1010, "all_enrollment"] == 8 and index.loc[1001, "classes300_verification"] == 8


# --- part: overlap -------------------------------------------------------------------------

def _split_rows():
    rows = [  # subject, eye, phase, idx, split
        (1141, "left", "enrollment", 1001, "train"), (1141, "left", "enrollment", 1002, "test"),
        (1141, "left", "enrollment", 1003, "validation"), (1141, "left", "enrollment", 1010, "train"),
        (1141, "left", "enrollment", 1011, "test"),
        (1141, "left", "verification", 1001, "train"), (1141, "left", "verification", 1005, "test"),
        (1142, "right", "enrollment", 1001, "test"), (1142, "right", "enrollment", 1002, "test"),
        (1142, "left", "enrollment", 1001, "train"),
    ]
    return pd.DataFrame([{"relpath": f"{COND}/{p}_{e}_eye/{s}/{s}_{i}_{e[0]}_c.png", "subject": s, "condition": COND,
                          "eye": e, "phase": p, "idx": i, "split": sp} for s, e, p, i, sp in rows])


def test_overlap_shares():
    split = _split_rows()
    path = split.set_index(["subject", "eye", "phase", "idx"]).relpath
    enr = {i: path[(1141, "left", "enrollment", i)] for i in (1001, 1002, 1010, 1011)}
    copies = {enr[1010]: enr[1002],     # a train copy of a test image
              enr[1011]: enr[1001]}     # a test copy of a train image
    table = check.overlap_table(split, copies)
    share = table[table.group == "all"].set_index(["subset", "reference", "criterion"]).share
    # test images: 1141 enr 1002 and 1011, 1141 ver 1005, 1142 right 1001 and 1002
    assert share[("test", "train", "adjacent")] == pytest.approx(2 / 5)            # 1002 (1001), 1011 (1010)
    assert share[("test", "train", "same_bucket")] == pytest.approx(3 / 5)
    assert share[("test", "train", "other_eye_same_frame")] == pytest.approx(1 / 5)  # 1142 right 1001
    assert share[("test", "train", "copy_relation")] == pytest.approx(2 / 5)
    assert share[("test", "train+validation", "within_2")] == pytest.approx(2 / 5)
    assert share[("validation", "train", "adjacent")] == 0.0                       # 1003: 1002 is test, 1004 absent
    assert share[("validation", "train", "within_2")] == 1.0                       # 1001 is train
    groups = table[(table.subset == "test") & (table.reference == "train") & (table.criterion == "adjacent")]
    assert set(groups.group) == {"all", "enrollment <=1009", "enrollment >=1010", "verification"}
    assert "copy_relation" not in set(check.overlap_table(split).criterion)
    none_found = check.overlap_table(split, {})                   # synthetic ran and found no copy: 0, not absent
    assert (none_found.loc[none_found.criterion == "copy_relation", "share"] == 0).all()
    assert (none_found.criterion == "copy_relation").any()


# --- part: areas ---------------------------------------------------------------------------

def test_load_like_runner_equals_keras_load_img(tree):
    import tensorflow as tf
    path = str(tree / COND / "enrollment_left_eye" / "1141" / "1141_1001_l_c.png")
    expected = np.asarray(tf.keras.utils.load_img(path, target_size=(256, 256)), dtype=np.float32) / 255.0
    got = check.load_like_runner(path)
    assert got.dtype == np.float32 and got.shape == (256, 256, 3)
    np.testing.assert_array_equal(got, expected)


def test_areas_equal_region_grids(tree):
    paths = sorted(str(p) for p in (tree / COND / "enrollment_left_eye" / "1141").glob("*.png"))[:6]
    rows = [check.image_areas(p) for p in paths]
    images = np.stack([check.load_like_runner(p) for p in paths])
    for grid, prefix in ((8, "area_"), (16, "area16_")):
        expected = region_grids(images, grid).astype(np.float64).mean(axis=(2, 3))
        got = np.array([[r[0][f"{prefix}{name}"] for name in REGIONS] for r in rows])
        np.testing.assert_allclose(got, expected, atol=1e-7)
        np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-6)
    for row, g8, g16 in rows:
        assert isinstance(row["fallback_used"], bool) and "circle_score" in row
        assert g8.shape == (4, 8, 8) and g16.shape == (4, 16, 16)


def test_relpath_from_split_link():
    link = f"/tmp/ocular_recognition/test/2530/{COND}__enrollment__2530_1010_r_c.png"
    assert check.relpath_from_link(link) == f"{COND}/enrollment_right_eye/2530/2530_1010_r_c.png"
    assert check.relpath_from_link(link, eye="right") == f"{COND}/enrollment_right_eye/2530/2530_1010_r_c.png"
    assert check.relpath_from_link("/some/other/file.png") is None
    images = pd.DataFrame({"link_path": [link, "/x/y.png"]})
    manifest = pd.DataFrame({"link_path": ["/x/y.png"], "relpath": ["c\\p\\1\\1_1001_l_c.png"]})
    out, source = check.attach_relpaths(images, manifest)
    assert out.relpath.tolist() == [f"{COND}/enrollment_right_eye/2530/2530_1010_r_c.png", "c/p/1/1_1001_l_c.png"]
    assert source == {"link name": 1, "manifest": 1}


def test_image_paths_fall_back_to_the_data_root(tree, tmp_path):
    rel = f"{COND}/enrollment_left_eye/1141/1141_1001_l_c.png"
    inside, outside = str(tree / COND / "enrollment_left_eye" / "1141" / "1141_1001_l_c.png"), tmp_path / "x.png"
    outside.write_bytes(b"")
    assert check.relpath_under_root(inside, str(tree)) == rel
    assert check.relpath_under_root(str(outside), str(tree)) is None
    assert check.relpath_under_root("/no/file", str(tree)) is None
    images = pd.DataFrame({"link_path": [inside, "/gone/link.png", "/gone/link2.png"],
                           "relpath": [rel, rel, f"{COND}/enrollment_left_eye/1141/missing.png"]})
    paths = check.resolve_image_paths(images, str(tree))       # link first, then <root>/<relpath> (05's rule)
    assert paths[0] == inside and os.path.samefile(paths[1], inside) and paths[2] is None
    assert check.resolve_image_paths(images, None)[1:] == [None, None]


# --- whole script --------------------------------------------------------------------------

def test_script_end_to_end(tree, tmp_path):
    split = _split_rows()
    split_csv = tmp_path / "manifest_split.csv"
    split.to_csv(split_csv, index=False)
    links = sorted(str(p) for p in (tree / COND / "enrollment_left_eye").glob("114[12]/*.png"))[::-1]
    assert len(links) > 2 * check.AREAS_CHUNK                # several chunks over two workers
    images_csv = tmp_path / "images_tiny.csv"
    pd.DataFrame({"image_id": range(len(links)), "link_path": links}).to_csv(images_csv, index=False)
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "16_check_dataset.py"), "--root", str(tree),
           "--manifest", str(tmp_path / "missing_manifest.csv"), "--split-manifest", str(split_csv),
           "--images-csv", str(images_csv), "--workers", "2", "--contact-sheets", "1", "--out-dir", str(out)]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "min left" in result.stdout                     # progress with a time estimate

    summary = json.loads((out / "summary.json").read_text())
    assert summary["parts"] == ["counts", "synthetic", "overlap", "areas"]
    assert summary["git_commit"] and set(summary["minutes"]) == {"counts", "synthetic", "overlap", "areas"}
    assert summary["synthetic"]["verdict"] == "synthetic copies found"
    assert summary["counts"]["images"] == 91 and "built from" in summary["manifest"]["source"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## 2. Are the frames >= 1010 synthetic?" in report and "## How to read this" in report

    areas = pd.read_csv(out / "areas_images_tiny.csv")
    assert list(areas.columns[:3]) == ["relpath", "link_path", "mask_source"]
    assert {f"area_{r}" for r in REGIONS} | {"fallback_used", "circle_score"} <= set(areas.columns)
    assert (areas.mask_source == "heuristic").all() and areas.link_path.tolist() == links
    assert [r.split("/")[-1] for r in areas.relpath] == [os.path.basename(p) for p in links]
    assert areas.relpath.str.startswith(f"{COND}/enrollment_left_eye/114").all()
    expected = np.array([[check.image_areas(p)[0][f"area_{r}"] for r in REGIONS] for p in links])
    np.testing.assert_allclose(areas[[f"area_{r}" for r in REGIONS]].to_numpy(), expected, atol=1e-12)
    grids = np.load(out / "areas_images_tiny_grids.npz")
    assert grids["grid8"].shape == (len(links), 4, 8, 8) and list(grids["regions"]) == list(REGIONS)
    assert list(grids["link_paths"]) == links
    overlap = pd.read_csv(out / "overlap.csv")
    assert {"adjacent", "same_bucket"} <= set(overlap.criterion)
    for name in ("frames.csv", "pairs.csv", "image_stats.csv", "buckets.csv", "counts_by_folder.csv"):
        assert (out / name).exists(), name


def test_script_refuses_to_start_without_inputs(tmp_path):
    cmd = [sys.executable, os.path.join(REPO, "scripts", "16_check_dataset.py"), "--parts", "overlap,areas",
           "--split-manifest", str(tmp_path / "missing.csv"), "--out-dir", str(tmp_path / "out")]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "overlap needs" in result.stderr and "areas needs --images-csv" in result.stderr
    assert not (tmp_path / "out").exists()
    # bad settings and unreachable area images stop the run before the long synthetic part, not after it
    images_csv = tmp_path / "images_x.csv"
    pd.DataFrame({"image_id": [0], "link_path": [str(tmp_path / "gone.png")]}).to_csv(images_csv, index=False)
    cmd = [sys.executable, os.path.join(REPO, "scripts", "16_check_dataset.py"), "--parts", "synthetic,areas",
           "--root", str(tmp_path), "--images-csv", str(images_csv), "--split-manifest", str(tmp_path / "missing.csv"),
           "--ref-quantile", "99", "--max-shift", "20", "--out-dir", str(tmp_path / "out")]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=300)
    assert result.returncode != 0 and not (tmp_path / "out").exists()
    assert "--ref-quantile" in result.stderr and "search grid" in result.stderr
    assert "1 of 1 images not found" in result.stderr
