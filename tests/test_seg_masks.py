"""Learned eye masks (src/xai/seg_masks.py) and scripts 17 and 18, without PyTorch.

The network is replaced by a colour classifier on drawn eyes, so the geometry
(padding, resizing, cropping, the runner's nearest-neighbour stretch), the class
mapping, the mask file, the metrics and the labelling helpers are all checked
exactly. One optional test runs the real model if PyTorch and the weights exist.
"""
import hashlib
import importlib.util
import json
import os
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from conftest import REPO
from src.xai import seg_masks as sm
from src.xai.masks import get_concept_masks, masks_to_grid


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_script = _load("17_make_eye_masks")
validate_script = _load("18_validate_masks")

# Colours the stand-in network recognises, by the NETWORK's class ids (sm.PERIORBITAL_CLASSES).
MOCK_COLOURS = np.array([(200, 150, 120),   # 0 background
                         (60, 40, 30),      # 1 brow
                         (235, 235, 230),   # 2 sclera
                         (40, 70, 160),     # 3 iris
                         (220, 90, 110),    # 4 caruncle
                         (150, 60, 150)],   # 5 lid
                        np.float32)


def drawn_eye(cx=130, cy=85, h=160, w=240, iris=True):
    """A native-size (240 x 160) drawn eye and the network class id of every pixel."""
    yy, xx = np.mgrid[0:h, 0:w]
    ids = np.zeros((h, w), np.uint8)
    ids[:22, 40:200] = 1                                                  # brow
    ids[((xx - 120) / 95.0) ** 2 + ((yy - 88) / 42.0) ** 2 <= 1] = 2       # eye opening (sclera)
    ids[(yy >= 34) & (yy < 50) & (np.abs(xx - 120) < 85) & (ids != 2)] = 5  # upper lid
    if iris:
        ids[(xx - cx) ** 2 + (yy - cy) ** 2 <= 28 ** 2] = 3
    ids[(yy >= 78) & (yy < 98) & (xx >= 28) & (xx < 46)] = 4               # caruncle
    return MOCK_COLOURS[ids].astype(np.uint8), ids


def colour_net(x):
    """Stand-in for the network: logits = minus the squared distance to each class colour."""
    rgb = (np.asarray(x) * 0.5 + 0.5) * 255.0                             # undo the normalisation
    return -((rgb[:, None] - MOCK_COLOURS[None, :, :, None, None]) ** 2).sum(axis=2) / 1000.0


def fake_visob(root, n_subjects=8, iris_missing=()):
    """Drawn eyes saved as native PNGs under VISOB-style relpaths, one subject per image."""
    rels = []
    conditions = [("iphone_day_light_short", "left"), ("oppo_dim_light_short", "right"),
                  ("samsung_office_short", "left"), ("iphone_office_short", "right")]
    for k in range(n_subjects):
        cond, eye = conditions[k % len(conditions)]
        subject = 1141 + k
        rel = f"{cond}/enrollment_{eye}_eye/{subject}/{subject}_1001_{eye[0]}_c.png"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image, _ = drawn_eye(cx=110 + 5 * k, iris=k not in iris_missing)
        Image.fromarray(image).save(path)
        rels.append(rel)
    return rels


# --- geometry -------------------------------------------------------------------

def test_model_geometry_is_the_runners_resize(tmp_path):
    h, w = 160, 240
    yy, xx = np.mgrid[0:h, 0:w]
    native = np.stack([xx, yy, (7 * xx + 3 * yy) % 256], axis=-1).astype(np.uint8)   # pixel = its coordinates
    path = tmp_path / "synthetic_240x160.png"
    Image.fromarray(native).save(path)
    loaded = np.asarray(tf.keras.utils.load_img(str(path), target_size=(256, 256)), dtype=np.uint8)
    runner = _load("05_run_experiment").load_image(str(path))                      # float32 in [0, 1]

    np.testing.assert_array_equal(sm.to_model_geometry(native[..., 0]), loaded[..., 0])   # columns
    np.testing.assert_array_equal(sm.to_model_geometry(native[..., 1]), loaded[..., 1])   # rows
    np.testing.assert_array_equal(sm.model_input_image(native), loaded)
    np.testing.assert_array_equal(np.round(runner * 255).astype(np.uint8), loaded)
    np.testing.assert_array_equal(sm.from_model_geometry(loaded, (h, w)), native)        # exact inverse
    # the stretch repeats 96 rows and 16 columns, as measured on the real model inputs
    assert (np.diff(loaded[:, 0, 1].astype(int)) == 0).sum() == 96
    assert (np.diff(loaded[0, :, 0].astype(int)) == 0).sum() == 16


def test_inverse_geometry_needs_an_upsampled_map():
    with pytest.raises(ValueError):
        sm.from_model_geometry(np.zeros((256, 256), np.uint8), (300, 240))
    with pytest.raises(ValueError):
        sm.to_model_geometry(np.zeros((160, 240), np.float32))


def test_pad_widths():
    assert sm.pad_widths((160, 240), 1.8) == (64, 64, 96, 96)
    assert sm.pad_widths((160, 240), 1.8, "square") == (136, 136, 96, 96)    # 432 x 432 canvas
    assert sm.pad_widths((160, 240), 1.0) == (0, 0, 0, 0)
    with pytest.raises(ValueError):
        sm.pad_widths((160, 240), 0.5)
    with pytest.raises(ValueError):
        sm.pad_widths((160, 240), 1.8, "circle")


def test_preprocess_replicates_edges_and_normalises_like_the_authors():
    image, _ = drawn_eye()
    before = image.copy()
    x, pads = sm.preprocess(image)
    np.testing.assert_array_equal(image, before)
    assert x.shape == (3, 256, 256) and x.dtype == np.float32 and pads == (64, 64, 96, 96)
    assert x.min() >= -1.0 and x.max() <= 1.0
    # the canvas corner is the replicated corner pixel of the crop
    np.testing.assert_allclose(x[:, 0, 0], (image[0, 0] / 255.0 - 0.5) / 0.5, atol=1e-6)
    flat, _ = sm.preprocess(np.full((160, 240, 3), 51, np.uint8))
    np.testing.assert_allclose(flat, (51 / 255.0 - 0.5) / 0.5, atol=1e-6)


@pytest.mark.parametrize("canvas", sm.CANVASES)
def test_segment_puts_regions_back_where_they_were_drawn(canvas):
    image, ids = drawn_eye()
    want = sm.remap(ids)
    got = sm.segment([image], colour_net, factor=1.8, canvas=canvas)[0]
    assert got.shape == (160, 240) and got.dtype == np.uint8
    iou = dict(zip(sm.CLASS_NAMES, sm.iou_dice(got, want, len(sm.CLASS_NAMES))["iou"]))
    assert min(iou["skin"], iou["sclera"], iou["brow"]) > 0.98
    assert min(iou["iris"], iou["lid"]) > 0.94          # a 2-pixel crop error drops the iris to ~0.92
    assert iou["caruncle"] > 0.75                        # 18 x 20 px: boundary pixels dominate
    iris = sm.CLASS_NAMES.index("iris")
    shift = np.argwhere(got == iris).mean(axis=0) - np.argwhere(want == iris).mean(axis=0)
    assert np.abs(shift).max() < 0.3                     # no off-by-one in padding or cropping


def test_segment_crops_back_with_uneven_padding():
    # an odd-sized crop on the square canvas gets one more pixel at the bottom and right: a swapped
    # top/bottom or left/right in the crop-back would shift every region by a pixel
    image, ids = drawn_eye(h=161, w=241)
    assert sm.pad_widths((161, 241), 1.8, "square") == (136, 137, 96, 97)
    got = sm.segment([image], colour_net, canvas="square")[0]
    want = sm.remap(ids)
    assert got.shape == (161, 241)
    assert sm.iou_dice(got, want, len(sm.CLASS_NAMES))["iou"][sm.CLASS_NAMES.index("iris")] > 0.94
    iris = sm.CLASS_NAMES.index("iris")
    shift = np.argwhere(got == iris).mean(axis=0) - np.argwhere(want == iris).mean(axis=0)
    assert np.abs(shift).max() < 0.3


def test_flip_tta_mirrors_the_logits_back():
    image, _ = drawn_eye(cx=150)                         # asymmetric: iris right, caruncle left
    plain = sm.segment([image], colour_net)[0]
    flipped = sm.segment([image], colour_net, flip_tta=True)[0]
    assert (plain == flipped).mean() > 0.995


def test_segment_batches_and_checks_the_network_output():
    images = [drawn_eye(cx=c)[0] for c in (100, 120, 140)]
    one_by_one = [sm.segment([img], colour_net)[0] for img in images]
    batched = sm.segment(images, colour_net, batch_size=2)
    for a, b in zip(one_by_one, batched):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        sm.segment(images, lambda x: np.zeros((len(x), 6, 128, 128), np.float32))


def test_class_mapping_follows_the_authors_ids():
    assert sm.PERIORBITAL_CLASSES == ("background", "brow", "sclera", "iris", "caruncle", "lid")
    assert sm.CLASS_NAMES == ("skin", "iris", "sclera", "lid", "caruncle", "brow")
    for k, name in enumerate(sm.PERIORBITAL_CLASSES):
        assert sm.CLASS_NAMES[sm.PERIORBITAL_TO_CANONICAL[k]] == ("skin" if name == "background" else name)
    logits = np.zeros((6, 4, 4), np.float32)
    logits[sm.PERIORBITAL_CLASSES.index("iris")] = 1.0
    labels = sm.remap(sm.postprocess(logits, (4, 4), (0, 0, 0, 0)))
    assert (labels == sm.CLASS_NAMES.index("iris")).all()
    with pytest.raises(ValueError):
        sm.remap(np.array([6]))


# --- mask file ------------------------------------------------------------------

def test_mask_file_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, (3, 256, 256)).astype(np.uint8)
    labels[0, :5, :5] = sm.IGNORE
    relpaths = ["c/enrollment_left_eye/1/1_1_l_c.png", "c\\enrollment_left_eye\\2\\2_1_l_c.png",
                "c/enrollment_right_eye/3/3_1_r_c.png"]
    meta = {"model": "periorbital", "weights_file": "combine.pth", "weights_sha256": sm.WEIGHTS_SHA256,
            "padding": {"factor": 1.8, "mode": "edge", "canvas": "stretch"}, "code_commit": "abc",
            "created": "2026-10-31T00:00:00+00:00"}
    before = labels.copy()
    path = sm.save_mask_file(str(tmp_path / "masks_t.npz"), relpaths, labels, meta=meta)
    np.testing.assert_array_equal(labels, before)
    mf = sm.load_mask_file(path)
    np.testing.assert_array_equal(mf["labels"], labels)
    assert mf["relpaths"][1] == "c/enrollment_left_eye/2/2_1_l_c.png"           # forward slashes
    assert mf["class_names"] == list(sm.CLASS_NAMES) and mf["meta"] == meta
    assert mf["index"]["c/enrollment_right_eye/3/3_1_r_c.png"] == 2
    with np.load(path, allow_pickle=False) as z:                                 # the raw contract
        assert set(z.files) == {"relpaths", "labels", "class_names", "meta"}
        assert z["labels"].dtype == np.uint8 and z["labels"].shape == (3, 256, 256)
        assert z["relpaths"].dtype.kind == "U" and z["meta"].shape == ()
        assert json.loads(str(z["meta"][()]))["weights_sha256"] == sm.WEIGHTS_SHA256


@pytest.mark.parametrize("change", ["shape", "dtype", "count", "duplicate", "bad_id", "suffix"])
def test_mask_file_rejects_contract_violations(tmp_path, change):
    labels = np.zeros((2, 256, 256), np.uint8)
    relpaths = ["a/b/1/1_1_l_c.png", "a/b/2/2_1_l_c.png"]
    path = str(tmp_path / "masks_t.npz")
    if change == "shape":
        labels = np.zeros((2, 128, 128), np.uint8)
    elif change == "dtype":
        labels = labels.astype(np.int32)
    elif change == "count":
        relpaths = relpaths[:1]
    elif change == "duplicate":
        relpaths = [relpaths[0]] * 2
    elif change == "bad_id":
        labels[1, 0, 0] = 7
    else:
        path = str(tmp_path / "masks_t.data")
    with pytest.raises(ValueError):
        sm.save_mask_file(path, relpaths, labels)


@pytest.mark.parametrize("change", ["missing_key", "dtype", "shape", "count", "bad_id"])
def test_mask_file_loader_checks_the_contract(tmp_path, change):
    # the runner and data-check work packages read files that other code may have written
    z = {"relpaths": np.array(["a/b/1/1_1_l_c.png", "a/b/2/2_1_l_c.png"]),
         "labels": np.zeros((2, 256, 256), np.uint8), "class_names": np.array(sm.CLASS_NAMES),
         "meta": np.array(json.dumps({"model": "x"}))}
    if change == "missing_key":
        del z["meta"]
    elif change == "dtype":
        z["labels"] = z["labels"].astype(np.int64)
    elif change == "shape":
        z["labels"] = np.zeros((2, 160, 240), np.uint8)
    elif change == "count":
        z["relpaths"] = z["relpaths"][:1]
    else:
        z["labels"][0, 10, 10] = 6
    path = tmp_path / "masks_t.npz"
    np.savez_compressed(path, **z)
    with pytest.raises(ValueError):
        sm.load_mask_file(path)


# --- regions and metrics -----------------------------------------------------------

def test_class_grids_average_like_masks_to_grid():
    _, ids = drawn_eye()
    labels = sm.to_model_geometry(sm.remap(ids))
    old = masks_to_grid(sm.to_heuristic_regions(labels), grid=8)         # pupil, iris, sclera, periocular
    new = sm.class_grids(labels, grid=8)
    k = {name: i for i, name in enumerate(sm.CLASS_NAMES)}
    assert new.shape == (6, 8, 8) and new.dtype == np.float32
    assert not old[0].any()
    np.testing.assert_allclose(old[1], new[k["iris"]], atol=1e-6)
    np.testing.assert_allclose(old[2], new[k["sclera"]], atol=1e-6)
    np.testing.assert_allclose(old[3], new[[k[c] for c in ("skin", "lid", "caruncle", "brow")]].sum(axis=0),
                               atol=1e-6)
    areas = sm.class_areas(labels)
    assert list(areas) == [f"area_{c}" for c in sm.CLASS_NAMES]
    assert sum(areas.values()) == pytest.approx(1.0, abs=1e-5)
    assert areas["area_iris"] == pytest.approx(float((labels == k["iris"]).mean()), abs=1e-6)


def test_to_heuristic_regions_and_comparison_labels(images):
    labels = np.array([[0, 1, 2], [3, 4, 5], [255, 1, 0]], np.uint8)
    regions = sm.to_heuristic_regions(labels)
    assert set(regions) == {"pupil", "iris", "sclera", "periocular", "iris_pupil"}
    assert not regions["pupil"].any()
    np.testing.assert_array_equal(regions["iris"], labels == 1)
    np.testing.assert_array_equal(regions["iris_pupil"], regions["iris"])
    np.testing.assert_array_equal(regions["sclera"], labels == 2)
    np.testing.assert_array_equal(regions["periocular"], np.isin(labels, [0, 3, 4, 5]))
    cover = np.stack([regions[r] for r in sm.HEURISTIC_REGIONS]).sum(axis=0)
    np.testing.assert_array_equal(cover, (labels != sm.IGNORE).astype(int))    # disjoint, unknown left out
    np.testing.assert_array_equal(sm.comparison_labels(regions), [[2, 0, 1], [2, 2, 2], [255, 0, 2]])
    heuristic = get_concept_masks(images[0])                 # the heuristic pupil joins the iris
    compared = sm.comparison_labels(heuristic)
    np.testing.assert_array_equal(compared == 0, heuristic["iris"] | heuristic["pupil"])
    assert (compared != sm.IGNORE).all()


def test_iou_dice_by_hand():
    gt = np.array([[0, 0, 1, 1], [0, 2, 2, 255]], np.uint8)
    pred = np.array([[0, 1, 1, 1], [0, 2, 0, 0]], np.uint8)
    r = sm.iou_dice(pred, gt, n_classes=4)
    np.testing.assert_allclose(r["iou"][:3], [2 / 4, 2 / 3, 1 / 2])
    np.testing.assert_allclose(r["dice"][:3], [4 / 6, 4 / 5, 2 / 3])
    assert np.isnan(r["iou"][3]) and np.isnan(r["dice"][3])        # absent from both: undefined, not 1
    np.testing.assert_array_equal(r["gt_px"][:3], [3, 2, 2])           # the ignored pixel is skipped
    unknown = sm.iou_dice(np.full((2, 2), sm.IGNORE, np.uint8), np.zeros((2, 2), np.uint8), 2)
    assert unknown["iou"][0] == 0.0 and np.isnan(unknown["iou"][1])  # unknown predictions are misses
    with pytest.raises(ValueError):
        sm.iou_dice(pred, gt[:1], 4)


# --- labelling helpers ---------------------------------------------------------------

def test_label_file_names():
    rel = "oppo_day_light_short/enrollment_right_eye/1141/1141_1005_r_c.png"
    name = sm.label_filename(rel)
    assert name == "oppo_day_light_short__enrollment_right_eye__1141__1141_1005_r_c.png"
    assert sm.relpath_from_filename(os.path.join("labels", name)) == rel
    assert sm.label_filename(rel.replace("/", "\\")) == name
    with pytest.raises(ValueError):
        sm.label_filename("a__b/c.png")
    assert sm.label_filename("a/_b/c.png") == "a___b__c.png"
    with pytest.raises(ValueError):                      # would also be "a___b__c.png": two images, one name
        sm.label_filename("a_/b/c.png")


def test_label_png_formats(tmp_path):
    labels = np.array([[0, 1, 2], [3, 4, 5], [255, 0, 1]], np.uint8)
    sm.write_label_png(tmp_path / "ids.png", labels)
    sm.write_label_png(tmp_path / "colour.png", labels, colour=True)
    Image.fromarray(labels).convert("LA").save(tmp_path / "ids_alpha.png")     # GIMP grey + alpha export
    for name in ("ids.png", "colour.png", "ids_alpha.png"):
        got, off = sm.read_label_png(tmp_path / name)
        np.testing.assert_array_equal(got, labels)
        assert off == 0
    # indexed PNG in our colours (as CVAT may export it), and indexed PNG holding raw ids
    palette = [int(c) for colour in list(sm.PALETTE) + [sm.IGNORE_COLOUR] for c in colour]
    indexed = Image.fromarray(np.where(labels == sm.IGNORE, 6, labels).astype(np.uint8))
    indexed.putpalette(palette)
    indexed.save(tmp_path / "indexed.png")
    raw = Image.fromarray(labels)
    raw.putpalette([v for k in range(256) for v in (k, 255 - k, (37 * k) % 256)])
    raw.save(tmp_path / "raw_ids.png")
    for name in ("indexed.png", "raw_ids.png"):
        got, off = sm.read_label_png(tmp_path / name)
        np.testing.assert_array_equal(got, labels)
        assert off == 0
    # anti-aliased colours and impossible ids are counted and set to "ignore"
    rgb = sm.colour_labels(labels)
    rgb[0, 0] = (10, 10, 10)
    Image.fromarray(rgb).save(tmp_path / "aliased.png")
    bad = labels.copy()
    bad[0, 0] = 9
    Image.fromarray(bad).save(tmp_path / "bad_ids.png")
    for name in ("aliased.png", "bad_ids.png"):
        got, off = sm.read_label_png(tmp_path / name)
        assert off == 1 and got[0, 0] == sm.IGNORE
        np.testing.assert_array_equal(got.ravel()[1:], labels.ravel()[1:])


def test_strata_and_link_keys():
    rel = "oppo_day_light_short/enrollment_right_eye/1141/1141_1005_r_c.png"
    assert sm.strata_from_relpath(rel) == {"device": "oppo", "lighting": "day", "eye": "right",
                                           "phase": "enrollment", "subject": 1141}
    assert sm.strata_from_relpath("samsung_office_short/verification_left_eye/2530/2530_1010_l_c.png")[
        "lighting"] == "office"
    link = "/tmp/ocular_recognition/test/1141/oppo_day_light_short__enrollment__1141_1005_r_c.png"
    assert sm.link_key(link) == sm.relpath_key(rel) == ("oppo_day_light_short", "enrollment",
                                                         "1141_1005_r_c.png")


def test_stratified_sample_spreads_over_strata_and_subjects():
    rows = [{"device": d, "lighting": light, "eye": eye, "subject": 1000 + 10 * k + s}
            for k, (d, light, eye) in enumerate((d, light, eye) for d in ("iphone", "oppo", "samsung")
                                                for light in ("day", "dim", "office") for eye in ("left", "right"))
            for s in range(5)]
    frame = pd.DataFrame(rows)
    before = frame.copy()
    a = sm.stratified_sample(frame, 40, seed=3)
    pd.testing.assert_frame_equal(frame, before)
    sizes = a.groupby(["device", "lighting", "eye"]).size()
    assert len(a) == 40 and len(sizes) == 18 and set(sizes) == {2, 3}          # 40 = 18 x 2 + 4
    pd.testing.assert_frame_equal(a, sm.stratified_sample(frame, 40, seed=3))    # seeded
    assert not a.index.equals(sm.stratified_sample(frame, 40, seed=4).index)
    small = frame.drop(frame[(frame.device == "oppo") & (frame.lighting == "dim")].index[1:])
    assert len(sm.stratified_sample(small, 40, seed=0)) == 40                    # shortfall filled
    assert len(sm.stratified_sample(frame, 500)) == len(frame)
    # a subject is reused only when the stratum has no other choice
    one = pd.DataFrame({"device": "x", "lighting": "y", "eye": "z", "subject": [1, 1, 1, 1, 2]})
    for seed in range(5):
        assert set(sm.stratified_sample(one, 2, seed=seed).subject) == {1, 2}


def test_image_table_resolves_relpaths_and_files(tmp_path):
    rels = fake_visob(tmp_path / "root", n_subjects=3)
    links = [f"/nowhere/test/{r.split('/')[2]}/{'__'.join(sm.relpath_key(r))}" for r in rels]
    by_link = sm.image_table(pd.DataFrame({"link_path": links}), data_root=str(tmp_path / "root"),
                             known_relpaths=rels)
    assert by_link.relpath.tolist() == rels
    assert all(os.path.exists(p) for p in by_link.path)
    assert by_link.loc[0, ["device", "lighting", "eye", "subject"]].tolist() == ["iphone", "day", "left", 1141]
    by_rel = sm.image_table(pd.DataFrame({"relpath": rels + rels[:1]}))          # no data root, duplicate
    assert len(by_rel) == 3 and by_rel.path.isna().all()
    with pytest.raises(ValueError):
        sm.image_table(pd.DataFrame({"link_path": links}))                         # nothing to recover from
    with pytest.raises(ValueError):
        sm.image_table(pd.DataFrame({"link_path": ["/somewhere/plain_name.png"]}), known_relpaths=rels)
    with pytest.raises(ValueError):
        sm.image_table(pd.DataFrame({"x": [1]}))


def test_fetch_weights_reuses_only_a_matching_file(tmp_path):
    path = tmp_path / "combine.pth"
    path.write_bytes(b"not really weights")
    digest = sm.file_sha256(path)
    assert sm.fetch_weights(str(tmp_path), url="http://invalid.invalid/x", sha256=digest) == str(path)
    with pytest.raises(ValueError):
        sm.fetch_weights(str(tmp_path), url="http://invalid.invalid/x", sha256="0" * 64)


def test_fetch_weights_is_safe_when_several_processes_download_at_once(tmp_path, monkeypatch):
    """train_anatomy.ipynb starts 21 and 17 at the same moment. With one shared .part file both failed on
    22 Nov (one found the file gone, the other a mix of two downloads). Every download now has its own
    temporary file: the writers below all write half, wait for each other, then write the rest."""
    import threading
    data = bytes(range(256)) * 4096                                   # 1 MiB of "weights"
    digest = hashlib.sha256(data).hexdigest()
    n = 4
    barrier = threading.Barrier(n)

    def fake_download(url, filename):
        with open(filename, "wb") as f:
            f.write(data[:len(data) // 2])
            f.flush()
            barrier.wait(timeout=10)                                  # every download is half done
            f.write(data[len(data) // 2:])
        return filename, None

    monkeypatch.setattr(sm.urllib.request, "urlretrieve", fake_download)
    results, errors = [], []

    def fetch():
        try:
            results.append(sm.fetch_weights(str(tmp_path), url="http://x/combine.pth", sha256=digest))
        except Exception as err:                                      # noqa: BLE001 - collected for the assert
            errors.append(err)

    threads = [threading.Thread(target=fetch) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors and results == [str(tmp_path / "combine.pth")] * n
    assert sm.file_sha256(tmp_path / "combine.pth") == digest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["combine.pth"]          # no temporary file left


def test_fetch_weights_retries_and_then_says_why(tmp_path, monkeypatch):
    import urllib.error
    data = b"weights"
    digest = hashlib.sha256(data).hexdigest()
    calls = []

    def flaky(url, filename):
        calls.append(filename)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        with open(filename, "wb") as f:
            f.write(b"truncated" if len(calls) == 2 else data)
        return filename, None

    monkeypatch.setattr(sm.urllib.request, "urlretrieve", flaky)
    path = sm.fetch_weights(str(tmp_path), url="http://x/w", sha256=digest, filename="w.pth", retry_wait=0)
    assert path == str(tmp_path / "w.pth") and len(calls) == 3 and len(set(calls)) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.pth"]

    def broken(url, filename):
        raise urllib.error.URLError("no Internet")

    monkeypatch.setattr(sm.urllib.request, "urlretrieve", broken)
    with pytest.raises(ValueError, match="2 attempt"):
        sm.fetch_weights(str(tmp_path / "other"), url="http://x/w", sha256=digest, attempts=2, retry_wait=0)
    assert list((tmp_path / "other").iterdir()) == []


def test_model_loading_without_torch_says_what_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)      # as if PyTorch were not installed, also where it is
    with pytest.raises(ImportError, match="PyTorch"):
        sm.build_periorbital_model()


# --- scripts 17 and 18 -------------------------------------------------------------

def test_make_masks_writes_the_shared_mask_file(tmp_path):
    rels = fake_visob(tmp_path / "root", n_subjects=6, iris_missing=(5,))
    table = sm.image_table(pd.DataFrame({"relpath": rels}), data_root=str(tmp_path / "root"))
    out = tmp_path / "out"
    summary = make_script.make_masks(table, colour_net, str(out), "t", batch_size=4, qc_per_stratum=1,
                                     meta={"model": "colour stand-in"})

    mf = sm.load_mask_file(out / "masks_t.npz")
    assert mf["relpaths"] == rels and mf["labels"].shape == (6, 256, 256)
    first = np.asarray(Image.open(table.path[0]).convert("RGB"))
    np.testing.assert_array_equal(mf["labels"][0], sm.to_model_geometry(sm.segment([first], colour_net)[0]))
    assert mf["meta"]["model"] == "colour stand-in" and mf["meta"]["native_sizes"] == {"240x160": 6}

    areas = pd.read_csv(out / "mask_areas_t.csv")
    assert list(areas.columns) == (["relpath", "link_path", "mask_source"]
                                   + [f"area_{c}" for c in sm.CLASS_NAMES] + ["empty_iris"])
    assert (areas.mask_source == "periorbital").all() and areas.relpath.tolist() == rels
    np.testing.assert_allclose(areas.filter(like="area_").sum(axis=1), 1.0, atol=1e-5)
    assert areas.empty_iris.tolist() == [False] * 5 + [True]

    saved = json.loads((out / "mask_summary_t.json").read_text())
    assert saved["n_images"] == 6 and saved["n_empty_iris"] == summary["n_empty_iris"] == 1
    assert set(saved["area_quantiles"]) == set(sm.CLASS_NAMES)
    assert (out / "mask_qc_t.png").exists() and (out / "mask_qc_low_iris_t.png").exists()


def test_export_sample_then_validate(tmp_path):
    root = tmp_path / "root"
    rels = fake_visob(root, n_subjects=8)
    table = sm.image_table(pd.DataFrame({"relpath": rels}), data_root=str(root))
    make_script.make_masks(table, colour_net, str(tmp_path / "masks"), "t", batch_size=8, qc_per_stratum=0)
    masks_path = str(tmp_path / "masks" / "masks_t.npz")
    masks = sm.load_mask_file(masks_path)

    pack = tmp_path / "pack"
    sample = validate_script.export_sample(table, str(pack), n=6, n_blind=2, masks=masks, seed=0,
                                           masks_path=masks_path)
    assert len(sample) == 6 and int(sample.prefilled.sum()) == 4 and sample.subject.nunique() == 6
    for r in sample.itertuples():
        assert (pack / "images" / r.name).read_bytes() == (root / r.relpath).read_bytes()
        assert (pack / "prefill" / r.name).exists() == r.prefilled == (pack / "overlays" / r.name).exists()
        if r.prefilled:                                   # the prefill is the model mask at native size
            prefill, off = sm.read_label_png(pack / "prefill" / r.name)
            assert off == 0 and prefill.shape == (160, 240)
            np.testing.assert_array_equal(sm.to_model_geometry(prefill), masks["labels"][masks["index"][r.relpath]])
    assert pd.read_csv(pack / "sample.csv").name.tolist() == sample.name.tolist()
    assert sm.load_mask_file(pack / "model_masks_sample.npz")["relpaths"] == sample.relpath.tolist()
    assert (pack / "labels").is_dir() and "background:0,0,0::" in (pack / "labelmap.txt").read_text()
    with zipfile.ZipFile(pack / "cvat_prefill.zip") as zf:
        names = zf.namelist()
        assert {"labelmap.txt", "ImageSets/Segmentation/default.txt"} <= set(names)
        assert sum(n.startswith("SegmentationClass/") for n in names) == 4

    # hand labels: the model's own masks, at native size (5 images) and at 256 x 256 (1 image)
    model_masks = sm.load_mask_file(pack / "model_masks_sample.npz")
    for i, r in enumerate(sample.itertuples()):
        lab = model_masks["labels"][model_masks["index"][r.relpath]]
        if i == 0:
            sm.write_label_png(pack / "labels" / r.name, lab)
        else:
            sm.write_label_png(pack / "labels" / r.name, sm.from_model_geometry(lab, (160, 240)), colour=i % 2 == 0)
    args = types.SimpleNamespace(labels_dir=str(pack / "labels"), labels_dir_b=str(pack / "labels"),
                                 images_dir=str(pack / "images"), data_root=None,
                                 masks=str(pack / "model_masks_sample.npz"), sample_csv=str(pack / "sample.csv"),
                                 no_heuristic=False, check_only=True, n_boot=200, seed=0,
                                 out_dir=str(tmp_path / "validation"), tag="t")
    validate_script.validate(args)                         # --check-only: label QC, no scores
    out = tmp_path / "validation"
    assert (out / "label_qc_t.csv").exists() and not (out / "validation_t.csv").exists()
    args.check_only = False
    validate_script.validate(args)

    qc = pd.read_csv(out / "label_qc_t.csv")
    assert len(qc) == 6 and qc.ok.all()
    summary = pd.read_csv(out / "validation_summary_t.csv")
    assert set(summary.subset) == {"all", "prefilled", "blind"}
    assert set(summary.method) == {"learned", "heuristic", "annotator B vs A"}
    learned = summary[(summary.method == "learned") & (summary.subset == "all")]
    np.testing.assert_allclose(learned.dropna(subset=["iou_mean"]).iou_mean, 1.0)    # labels == masks
    heuristic = summary[(summary.method == "heuristic") & (summary.subset == "all")]
    assert len(heuristic) == 3 and (heuristic.iou_mean < 1).any()
    report = (out / "validation_t.md").read_text()
    assert "learned minus heuristic" in report.lower() and "human ceiling" in report


def test_annotator_agreement_and_summary_by_hand():
    a = np.zeros((4, 4), np.uint8)
    a[1:3, 1:3] = 1                                      # annotator A: 4 iris pixels
    b = a.copy()
    b[1, 1] = sm.IGNORE                                  # B unsure about one of them: skipped, not a miss
    b[0, 0] = 1                                          # B calls one skin pixel iris
    rows = pd.DataFrame(validate_script.score_annotators(a, b))
    for space in ("6 classes", "3 regions"):
        iris = rows[(rows.space == space) & (rows.cls == "iris")].iloc[0]
        assert (iris.intersection, iris.union, iris.gt_px, iris.pred_px) == (3, 4, 3, 4)
        assert iris.iou == pytest.approx(3 / 4) and iris.dice == pytest.approx(6 / 7)

    per_image = pd.DataFrame({"subset": "all", "method": "learned", "space": "6 classes", "cls": "iris",
                              "relpath": ["x", "y", "z"], "subject": [1, 1, 2],
                              "iou": [0.5, 1.0, np.nan], "dice": [2 / 3, 1.0, np.nan],
                              "intersection": [1, 3, 0], "union": [2, 3, 0], "gt_px": [2, 3, 0],
                              "pred_px": [1, 3, 0]})
    s = validate_script.summarise(per_image, n_boot=50).iloc[0]
    assert (s.n_images, s.n_subjects) == (2, 1)          # the class is absent from both maps of image z
    assert s.iou_mean == pytest.approx(0.75) and s.iou_low == pytest.approx(0.75)   # one subject: no spread
    assert s.iou_pooled == pytest.approx(4 / 5) and s.dice_pooled == pytest.approx(8 / 9)


def test_script_17_runs_a_smoke_run_with_a_stand_in_network(tmp_path, monkeypatch, capsys):
    rels = fake_visob(tmp_path / "root", n_subjects=6)
    pd.DataFrame({"relpath": rels}).to_csv(tmp_path / "list.csv", index=False)
    weights = tmp_path / "combine.pth"
    weights.write_bytes(b"stand-in")
    stand_in_torch = types.SimpleNamespace(__version__="stand-in",
                                           cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(__version__="stand-in"))
    monkeypatch.setattr(sm, "require_torch", lambda: stand_in_torch)
    monkeypatch.setattr(sm, "load_periorbital_model", lambda path, device: ("network", path, device))
    monkeypatch.setattr(sm, "torch_predictor", lambda model, device, amp: colour_net)
    out = tmp_path / "out"
    argv = ["17_make_eye_masks.py", "--csv", str(tmp_path / "list.csv"), "--data-root", str(tmp_path / "root"),
            "--weights", str(weights), "--max-images", "4", "--out-dir", str(out), "--tag", "smoke"]
    monkeypatch.setattr(sys, "argv", argv)
    make_script.main()

    printed = capsys.readouterr().out
    assert "images: 4 of 6 in the input" in printed
    assert "estimate for all 6 input images" in printed  # the smoke run fits in one batch
    mf = sm.load_mask_file(out / "masks_smoke.npz")
    assert len(mf["relpaths"]) == 4 and set(mf["relpaths"]) <= set(rels)
    meta = mf["meta"]
    assert meta["padding"] == {"factor": 1.8, "mode": "edge", "canvas": "stretch"}
    assert meta["weights_sha256"] == sm.WEIGHTS_SHA256 and meta["device"] == "cpu"
    assert {"model", "weights_file", "code_commit", "created"} <= set(meta)
    summary = json.loads((out / "mask_summary_smoke.json").read_text())
    assert summary["n_images"] == 4 and "seconds_model_load" in summary
    assert all(os.path.exists(p) for p in summary["outputs"].values())

    pd.DataFrame({"relpath": []}).to_csv(tmp_path / "empty.csv", index=False)
    monkeypatch.setattr(sys, "argv", argv[:2] + [str(tmp_path / "empty.csv")] + argv[3:])
    with pytest.raises(SystemExit, match="no images"):    # stops before loading the model
        make_script.main()
    table = sm.image_table(pd.DataFrame({"relpath": rels}), data_root=str(tmp_path / "root"))
    with pytest.raises(ValueError):
        make_script.make_masks(table.iloc[:0], colour_net, str(out), "empty")


def test_script_18_exports_a_pack_and_checks_labels(tmp_path, monkeypatch):
    root = tmp_path / "root"
    rels = fake_visob(root, n_subjects=6)
    table = sm.image_table(pd.DataFrame({"relpath": rels}), data_root=str(root))
    summary = make_script.make_masks(table, colour_net, str(tmp_path / "masks"), "t", qc_per_stratum=0)
    assert "qc" not in summary["outputs"] and not (tmp_path / "masks" / "mask_qc_t.png").exists()
    pd.DataFrame({"relpath": rels}).to_csv(tmp_path / "list.csv", index=False)
    pack = tmp_path / "pack"
    monkeypatch.setattr(sys, "argv", ["18_validate_masks.py", "--export-sample", "--csv", str(tmp_path / "list.csv"),
                                      "--data-root", str(root), "--masks", str(tmp_path / "masks" / "masks_t.npz"),
                                      "--n", "4", "--n-blind", "1", "--out-dir", str(pack), "--zip"])
    validate_script.main()
    export = json.loads((pack / "export.json").read_text())
    assert (export["n"], export["n_prefilled"]) == (4, 3) and (tmp_path / "pack.zip").exists()
    subset = sm.load_mask_file(pack / "model_masks_sample.npz")
    assert subset["meta"]["n_images"] == 4 and subset["meta"]["subset_of"] == "masks_t.npz"

    for rel in subset["relpaths"]:                       # labels at native size, plus one broken file
        native = sm.from_model_geometry(subset["labels"][subset["index"][rel]], (160, 240))
        sm.write_label_png(pack / "labels" / sm.label_filename(rel), native)
    (pack / "labels" / "broken.png").write_bytes(b"not a png")
    monkeypatch.setattr(sys, "argv", ["18_validate_masks.py", "--labels-dir", str(pack / "labels"),
                                      "--images-dir", str(pack / "images"), "--check-only",
                                      "--out-dir", str(tmp_path / "qc"), "--tag", "t"])
    validate_script.main()                               # the broken file is flagged, the check goes on
    qc = pd.read_csv(tmp_path / "qc" / "label_qc_t.csv").set_index("file")
    assert len(qc) == 5 and not qc.loc["broken.png", "ok"] and qc.drop("broken.png").ok.all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or not os.environ.get("PERIORBITAL_WEIGHTS"),
                    reason="needs PyTorch and PERIORBITAL_WEIGHTS=<path to combine.pth>")
def test_real_periorbital_model():
    model = sm.load_periorbital_model(os.environ["PERIORBITAL_WEIGHTS"])
    predict = sm.torch_predictor(model)
    image, _ = drawn_eye()
    assert predict(sm.preprocess(image)[0][None]).shape == (1, 6, 256, 256)
    labels = sm.segment([image], predict)[0]
    assert labels.shape == (160, 240) and labels.max() < len(sm.CLASS_NAMES)
    examples = os.environ.get("EYE_EXAMPLES_NPZ")         # e.g. results/examples_full.npz (real eyes)
    if examples:
        with np.load(examples, allow_pickle=False) as z:
            natives = [sm.from_model_geometry(z[f"{i}|clean||||layercam|true|image"], (160, 240)) for i in range(4)]
        for lab in sm.segment(natives, predict, batch_size=4):
            area = float((sm.to_model_geometry(lab) == sm.CLASS_NAMES.index("iris")).mean())
            assert 0.05 < area < 0.25
