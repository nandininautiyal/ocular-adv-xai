"""Mask-aware training data (src/data/masked_generator.py): masks64 files, mask lookup by the 17 / 18 keys,
cell fractions and targets, and the generator: the same images as 03's generator, and image and mask
still aligned after the random augmentation."""
import os

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data import masked_generator as mg
from src.models import anatomy_model as am
from src.xai import seg_masks as sm

import importlib.util
from conftest import REPO

GROUPS = am.DEFAULT_PART_GROUPS
CLASSES = list(sm.CLASS_NAMES)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_script = _load("03_train_baseline")


def aug_datagen():
    """03's training ImageDataGenerator."""
    return ImageDataGenerator(rescale=1.0 / 255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                              horizontal_flip=True, brightness_range=[0.8, 1.2])


def relpath_of(subject, i, condition="cond"):
    return f"{condition}/enrollment_left_eye/{subject}/{subject}_{1001 + i}_l_c.png"


def link_name(subject, i, condition="cond"):
    return f"{condition}__enrollment__{subject}_{1001 + i}_l_c.png"


# Colours that survive brightness 0.8-1.2: class k <-> the set bits of k + 1 over (R, G, B).
CODE = np.array([[200 * ((k + 1) >> b & 1) for b in range(3)] for k in range(6)], np.uint8)


def coded_labels(seed):
    """An asymmetric 256 x 256 label map of 32-pixel blocks over the six classes (no unknown pixels)."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 6, (8, 8))
    blocks[:, :2] = 0                                    # left two columns skin, so flips change the map
    return np.kron(blocks, np.ones((32, 32), np.int64)).astype(np.uint8)


def decode(images):
    """Class of every pixel of coded images (N, H, W, 3) in [0, 1]: set bits of each channel, minus 1."""
    bits = (images > 0.5 * images.max(axis=-1, keepdims=True)) & (images > 0.2)
    code = bits[..., 0] * 1 + bits[..., 1] * 2 + bits[..., 2] * 4
    return code.astype(np.int64) - 1


def coded_tree(root, subjects=(1141, 1142), n=4, split="train"):
    """Images whose pixels encode their own label map; returns {relpath: 256 x 256 labels}."""
    labels = {}
    for s_idx, subject in enumerate(subjects):
        d = root / split / str(subject)
        d.mkdir(parents=True)
        for i in range(n):
            lab = coded_labels(100 * s_idx + i)
            Image.fromarray(CODE[lab]).save(d / link_name(subject, i))
            labels[relpath_of(subject, i)] = lab
    return labels


def write_masks(path, labels_by_relpath, size=64):
    rels = sorted(labels_by_relpath)
    full = np.stack([labels_by_relpath[r] for r in rels])
    if size == 64:
        return mg.save_masks64(str(path), rels, mg.labels_to_64(full), CLASSES, {"source": "test"})
    return sm.save_mask_file(str(path), rels, full, CLASSES, {"source": "test"})


# ---------------------------------------------------------------- masks64 files

def test_labels_to_64_is_pils_nearest_resize():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, (3, 256, 256)).astype(np.uint8)
    labels[0, :10, :10] = sm.IGNORE
    small = mg.labels_to_64(labels)
    assert small.shape == (3, 64, 64) and small.dtype == np.uint8
    for k in range(3):
        want = np.asarray(Image.fromarray(labels[k]).resize((64, 64), Image.NEAREST))
        np.testing.assert_array_equal(small[k], want)
    with pytest.raises(ValueError):
        mg.labels_to_64(labels.astype(np.int32))


def test_masks64_round_trip_and_256_source(tmp_path):
    labels = {relpath_of(1141, i): coded_labels(i) for i in range(3)}
    write_masks(tmp_path / "masks64_t.npz", labels)
    write_masks(tmp_path / "masks_t.npz", labels, size=256)
    small, full = mg.load_label_source(tmp_path / "masks64_t.npz"), mg.load_label_source(tmp_path / "masks_t.npz")
    assert (small["source_size"], full["source_size"]) == (64, 256)
    assert small["relpaths"] == full["relpaths"] == sorted(labels) and small["class_names"] == CLASSES
    np.testing.assert_array_equal(small["labels"], full["labels"])
    assert small["meta"] == {"source": "test"}
    with pytest.raises(ValueError, match="unique"):
        mg.save_masks64(str(tmp_path / "x.npz"), ["a/b/c", "a/b/c"], small["labels"][:2], CLASSES)
    bad = small["labels"].copy()
    bad[0, 0, 0] = 9
    with pytest.raises(ValueError, match="outside"):
        mg.save_masks64(str(tmp_path / "x.npz"), small["relpaths"], bad, CLASSES)


def test_mask_table_matches_links_and_relpaths_like_17():
    rels = [relpath_of(1141, 0), relpath_of(1142, 1, "oppo_day_light_short")]
    table = mg.MaskTable({"relpaths": rels, "labels": np.zeros((2, 64, 64), np.uint8), "class_names": CLASSES})
    links = [f"/split/train/1142/{link_name(1142, 1, 'oppo_day_light_short')}", f"/x/test/1141/{link_name(1141, 0)}",
             f"/x/test/1141/{link_name(1141, 5)}", "/not/a/link.png"]
    assert table.rows_for_links(links).tolist() == [1, 0, -1, -1]
    assert table.rows_for_relpaths(rels[::-1] + ["x.png"]).tolist() == [1, 0, -1]
    assert (table.labels_for([-1])[0] == sm.IGNORE).all()
    with pytest.raises(ValueError, match="two masks"):           # same condition, phase and file name
        mg.MaskTable({"relpaths": [rels[0], rels[0].replace("left_eye", "right_eye")],
                      "labels": np.zeros((2, 64, 64), np.uint8), "class_names": CLASSES})


# ---------------------------------------------------------------- targets

def test_group_fractions_and_targets():
    labels = np.zeros((2, 64, 64), np.uint8)                       # skin everywhere
    labels[0, :8, :8] = CLASSES.index("iris")                      # cell (0, 0): all iris
    labels[0, :8, 8:12] = CLASSES.index("caruncle")                # cell (0, 1): half caruncle (sclera group)
    labels[0, 8:16, :8] = sm.IGNORE                                # cell (1, 0): unknown -> no group
    frac = mg.group_fractions(labels, CLASSES, GROUPS)
    assert frac.shape == (2, 8, 8, 5)
    names = list(GROUPS)
    assert frac[0, 0, 0, names.index("iris")] == 1 and frac[0, 0, 1, names.index("sclera")] == 0.5
    assert frac[0, 1, 0].sum() == 0 and frac[1, ..., names.index("skin")].min() == 1
    np.testing.assert_allclose(frac[0, 2:].sum(-1), 1.0)                # every known pixel is in one group

    eps = 1e-3
    targets, weights = mg.mask_targets(frac, GROUPS, "eye", eps=eps)
    eye = frac[..., :4].sum(-1)
    np.testing.assert_allclose(targets[..., 0], (eye + eps) / (eye + eps).sum(axis=(1, 2), keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(targets.sum(axis=(1, 2)), 1.0, rtol=1e-6)
    # image 0: 96 eye pixels of 4096 (2.3 %); image 1 has no eye pixel: its near-uniform target gets weight 0
    assert weights.tolist() == [[1.0], [0.0]]
    np.testing.assert_allclose(targets[1], 1 / 64)                 # no eye: uniform target
    # the eye-region weight uses min_part_area like the parts mode: 96 / 4096 = 0.0234
    _, weights = mg.mask_targets(frac, GROUPS, "eye", eps=eps, min_part_area=0.03)
    assert weights.tolist() == [[0.0], [0.0]]
    _, weights = mg.mask_targets(frac, GROUPS, "eye", eps=eps, min_part_area=0.0, has_mask=[False, True])
    assert weights.tolist() == [[0.0], [1.0]]

    targets, weights = mg.mask_targets(frac, GROUPS, "parts", eps=eps, min_part_area=0.005,
                                       has_mask=[True, False])
    assert targets.shape == (2, 8, 8, 5)
    # image 0: iris 64 px, sclera 32 px of 4096 (>= 0.5 %), lid and brow absent; image 1 has no mask
    assert weights.tolist() == [[1, 1, 0, 0, 1], [0, 0, 0, 0, 0]]
    np.testing.assert_allclose(targets[0, ..., 0], (frac[0, ..., 0] + eps) / (frac[0, ..., 0] + eps).sum(), rtol=1e-6)

    region = mg.region_inputs(labels, CLASSES, GROUPS, has_mask=[True, False])
    np.testing.assert_allclose(region[0, ..., 0], eye[0])
    assert region.shape == (2, 8, 8, 1) and region[1].max() == 0
    rows = mg.regions_for_rows(mg.MaskTable({"relpaths": [relpath_of(1, 0), relpath_of(2, 0)], "labels": labels,
                                             "class_names": CLASSES}), [0, -1], am.eye_classes(am.anatomy_settings({})))
    np.testing.assert_allclose(rows[0, ..., 0], eye[0])
    np.testing.assert_allclose(rows[1], 0)
    with pytest.raises(ValueError, match="not in the mask"):
        mg.group_fractions(labels, CLASSES, {"cornea": ["cornea"]})


# ---------------------------------------------------------------- the generator

def _generators(root, masks_path, mode="parts", datagen=None, **kwargs):
    classes = train_script.class_list(str(root), splits=("train",))
    datagen = datagen or aug_datagen()
    plain_gen = train_script.ClassListDataGen(str(root / "train"), datagen, classes, batch_size=4)
    masked = mg.MaskedDataGen(str(root / "train"), datagen, classes, masks_path, mode, GROUPS, batch_size=4, **kwargs)
    return plain_gen, masked


def test_images_are_those_of_03s_generator(tmp_path):
    labels = coded_tree(tmp_path)
    write_masks(tmp_path / "m.npz", labels)
    plain_gen, masked = _generators(tmp_path, tmp_path / "m.npz")
    assert len(masked) == len(plain_gen) == 2
    for epoch in range(2):
        np.random.seed(epoch)
        reference = [plain_gen[i] for i in range(2)]
        np.random.seed(epoch)
        ours = [masked[i] for i in range(2)]
        for ([x0, y0], _), ((x1, y1, targets, weights), y2) in zip(reference, ours):
            np.testing.assert_array_equal(x0, x1)       # same calls, same random stream: identical pixels
            np.testing.assert_array_equal(y0, y1)
            np.testing.assert_array_equal(y1, y2)
            assert targets.shape == (4, 8, 8, 5) and weights.shape == (4, 5)
        plain_gen.generator.index_array = masked.generator.index_array = None   # a new epoch's order


def test_image_and_mask_stay_aligned_under_augmentation(tmp_path):
    labels = coded_tree(tmp_path, subjects=(1141, 1142, 1143), n=6)
    write_masks(tmp_path / "m.npz", labels)
    _, masked = _generators(tmp_path, tmp_path / "m.npz")
    np.random.seed(3)
    agree, agree_unaugmented, flipped = [], [], 0
    for i in range(len(masked)):
        index_array = masked.batch_indices(i)
        params_seen = []
        original = masked.generator.image_data_generator.get_random_transform
        masked.generator.image_data_generator.get_random_transform = \
            lambda shape, f=original: params_seen.append(f(shape)) or params_seen[-1]
        try:
            x, _, aug_labels, has_mask = masked.load_batch(index_array)
        finally:
            masked.generator.image_data_generator.get_random_transform = original
        assert has_mask.all()
        flipped += sum(bool(p["flip_horizontal"]) for p in params_seen)
        decoded = decode(x[:, 2::4, 2::4])               # the image pixel each 64 x 64 label pixel stands for
        stored = masked.masks.labels[masked.rows[index_array]]
        for k in range(len(x)):
            known = aug_labels[k] != sm.IGNORE
            agree.append(np.mean(decoded[k][known] == aug_labels[k][known]))
            agree_unaugmented.append(np.mean(decoded[k] == stored[k]))
    assert flipped >= 2                                  # the flips were exercised
    assert min(agree) > 0.97, agree                      # block borders blur under bilinear rotation only
    assert np.mean(agree_unaugmented) < np.mean(agree) - 0.1   # the augmentation really moved the images


def test_validation_batches_are_not_augmented(tmp_path):
    labels = coded_tree(tmp_path)
    write_masks(tmp_path / "m.npz", labels)
    _, masked = _generators(tmp_path, tmp_path / "m.npz", mode="eye", datagen=ImageDataGenerator(rescale=1.0 / 255))
    index_array = masked.batch_indices(0)
    x, _, aug_labels, _ = masked.load_batch(index_array)
    np.testing.assert_array_equal(aug_labels, masked.masks.labels[masked.rows[index_array]])
    np.testing.assert_array_equal(decode(x), np.stack([labels[relpath_of(*subject_frame(masked, j))]
                                                       for j in index_array]))
    (_, _, targets, weights), _ = masked[0]
    assert targets.shape == (4, 8, 8, 1) and (weights == 1).all()


def subject_frame(masked, j):
    """(subject, frame offset) of the iterator's j-th file."""
    name = os.path.basename(masked.generator.filepaths[j])
    subject, frame = name.split("__")[2].split("_")[:2]
    return int(subject), int(frame) - 1001


def test_missing_masks_stop_or_get_weight_zero(tmp_path):
    labels = coded_tree(tmp_path)
    some = {r: v for r, v in labels.items() if not r.endswith("1002_l_c.png")}     # one frame per subject left out
    write_masks(tmp_path / "m.npz", some)
    with pytest.raises(mg.MissingMasksError, match="2 of 8 images"):
        _generators(tmp_path, tmp_path / "m.npz")
    _, masked = _generators(tmp_path, tmp_path / "m.npz", allow_missing=True)
    assert masked.coverage() == {"n_images": 8, "n_with_mask": 6, "share": 0.75}
    np.random.seed(0)
    seen = []
    for i in range(len(masked)):
        (_, _, _, weights), _ = masked[i]
        index_array = masked.generator.index_array[4 * i: 4 * i + 4]
        missing = masked.rows[index_array] < 0
        assert (weights[missing] == 0).all() and (weights[~missing].sum(axis=1) > 0).all()
        seen += missing.tolist()
    assert sum(seen) == 2
    _, region_gen = _generators(tmp_path, tmp_path / "m.npz", mode="region", allow_missing=True)
    (x, y, region), _ = region_gen[0]
    index_array = region_gen.generator.index_array[:4]
    assert region.shape == (4, 8, 8, 1)
    assert (region[region_gen.rows[index_array] < 0] == 0).all()


def test_images_without_enough_eye_get_no_mask_loss(tmp_path):
    """attn_mask: an image whose eye region is (almost) empty gets weight 0, like an absent part in the parts
    mode, and below_min_area counts those images per split (run_<tag>.json records it)."""
    labels = coded_tree(tmp_path)
    empty = sorted(labels)[1]
    labels[empty] = np.zeros_like(labels[empty])                    # skin only: no eye pixel at all
    write_masks(tmp_path / "m.npz", labels)
    plain = ImageDataGenerator(rescale=1.0 / 255)
    _, masked = _generators(tmp_path, tmp_path / "m.npz", mode="eye", datagen=plain)
    assert masked.below_min_area() == 1
    for i in range(len(masked)):
        index_array = masked.batch_indices(i)
        (_, _, targets, weights), _ = masked[i]
        is_empty = np.array([masked.masks.relpaths[masked.rows[j]] == empty for j in index_array])
        assert (weights[is_empty] == 0).all() and (weights[~is_empty] == 1).all()
    _, parts = _generators(tmp_path, tmp_path / "m.npz", mode="parts", datagen=plain)
    counts = parts.below_min_area()
    assert set(counts) == set(GROUPS) and counts["iris"] >= 1 and counts["skin"] == 0
    _, region = _generators(tmp_path, tmp_path / "m.npz", mode="region", datagen=plain)
    assert region.below_min_area() is None


def test_reshuffle_draws_a_new_order(tmp_path):
    labels = coded_tree(tmp_path)
    write_masks(tmp_path / "m.npz", labels)
    for reshuffle in (False, True):
        _, masked = _generators(tmp_path, tmp_path / "m.npz", reshuffle=reshuffle)
        masked[0]
        first = masked.generator.index_array.copy()
        masked.on_epoch_end()
        assert np.array_equal(masked.generator.index_array, first) != reshuffle


def test_label_transform_follows_the_image_transform():
    datagen = aug_datagen()
    lab = np.zeros((64, 64), np.uint8)
    lab[:, :16] = 3                                      # a band on the left
    flip = {"theta": 0, "tx": 0, "ty": 0, "shear": 0, "zx": 1, "zy": 1, "flip_horizontal": True,
            "flip_vertical": False, "brightness": 1.1, "channel_shift_intensity": None}
    out = mg.transform_labels(lab, flip, (256, 256, 3), datagen)
    assert (out[:, -16:] == 3).all() and (out[:, :-16] == 0).all()
    shift = dict(flip, flip_horizontal=False, tx=-32.0)  # Keras' tx moves columns: 32 image px = 8 label px
    image = np.zeros((256, 256, 3), np.float32)
    image[:, :64] = 255
    moved = datagen.apply_transform(image, shift)
    out = mg.transform_labels(lab, shift, image.shape, datagen)
    np.testing.assert_array_equal(out == 3, (moved[2::4, 2::4, 0] > 127))
    down = dict(shift, tx=0.0, ty=-32.0)                 # and ty the rows
    np.testing.assert_array_equal(mg.transform_labels(lab.T.copy(), down, image.shape, datagen) == 3,
                                  datagen.apply_transform(image.transpose(1, 0, 2).copy(), down)[2::4, 2::4, 0] > 127)
    rotated = mg.transform_labels(lab, dict(shift, tx=0, theta=10.0), (256, 256, 3), datagen)
    assert set(np.unique(rotated)) <= {0, 3}             # nearest neighbour: classes never mix
    const = ImageDataGenerator(fill_mode="constant", cval=0.0)
    out = mg.transform_labels(lab, dict(shift, tx=-64.0), (256, 256, 3), const)
    assert (out[:, :16] == sm.IGNORE).all() and (out[:, 16:32] == 3).all()   # constant fill: unknown, not class 0
