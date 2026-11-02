"""
src/data/masked_generator.py

Training data for the mask-supervised heads (src/models/anatomy_model.py):
images with the SAME augmentation as 03_train_baseline.py, plus each image's eye
mask carried through that augmentation, turned into per-cell targets on the
8 x 8 feature grid.

Compact training masks (``masks64_<tag>.npz``, scripts/21_make_train_masks.py)
-----------------------------------------------------------------------------
``relpaths`` (manifest relpaths, "/" separators), ``labels`` uint8 (N, 64, 64):
the 256 x 256 model-geometry labels of src/xai/seg_masks.py resized to 64 x 64
with PIL nearest (pixel (i, j) of the 64 map is pixel (4i+2, 4j+2) of the 256
map), ``class_names`` (skin, iris, sclera, lid, caruncle, brow; 255 = unknown)
and ``meta`` (a 0-d JSON string). 64 x 64 keeps 8 x 8 label pixels per feature
cell, enough for the cell fractions, at 1/16 of the size (about 400 MB for
100,000 images instead of 6.5 GB).

Augmentation
------------
03 augments with ImageDataGenerator(rotation 10, width/height shift 0.1,
horizontal flip, brightness 0.8-1.2). Here every image goes through exactly the
calls the DirectoryIterator makes (load_img, img_to_array,
get_random_transform, apply_transform, standardize, in the same order and with
the same NumPy random stream), so the images are identical to 03's for the same
seed. The label map gets the same transform: the same angle, the shifts scaled
by 64 / 256, the same flip and fill mode, nearest-neighbour (order 0) so that
classes never mix; brightness only touches the image. With fill_mode "constant"
the label fill is 255 (unknown), not the image's cval.

Targets
-------
frac[g] (8 x 8): share of each cell's label pixels in part group g
(anatomy.part_groups; unknown pixels are in no group).
- eye (attn_mask): t = (eye + eps) / sum(eye + eps), eye = sum of every group
  but skin; weight 1 if the eye region covers at least min_part_area of the
  image, else 0 (an image with almost no eye pixels, e.g. a segmenter failure,
  would otherwise get a near-uniform target and pull the map towards average
  pooling).
- parts: t_p = (frac_p + eps) / sum(frac_p + eps) per part; weight 1 if the part
  covers at least min_part_area of the image, else 0.
- region (attnet_gate): the eye-region fraction (N, 8, 8, 1), also at test time.
An image without a mask (only with allow_missing) gets weight 0 (and region 0).
"""

import json
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.anatomy_model import NON_EYE_GROUP
from src.xai import seg_masks as sm

MASK64_SIZE = (64, 64)
IGNORE = sm.IGNORE
MODES = {"attn_mask": "eye", "parts": "parts", "attnet_gate": "region"}   # arch -> generator mode


class MissingMasksError(ValueError):
    """Some images of a split have no mask (pass allow_missing to train with weight 0 for them)."""


# ---------------------------------------------------------------------------
# masks64 files
# ---------------------------------------------------------------------------

def labels_to_64(labels):
    """(N, 256, 256) model-geometry labels -> (N, 64, 64) with PIL nearest (exactly what PIL would pick)."""
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis]
    if labels.shape[1:] != sm.MODEL_SIZE or labels.dtype != np.uint8:
        raise ValueError(f"expected uint8 labels (N, 256, 256), got {labels.dtype} {labels.shape}")
    rows = sm.nearest_index(sm.MODEL_SIZE[0], MASK64_SIZE[0])
    cols = sm.nearest_index(sm.MODEL_SIZE[1], MASK64_SIZE[1])
    return np.ascontiguousarray(labels[:, rows][:, :, cols])


def save_masks64(path, relpaths, labels, class_names=sm.CLASS_NAMES, meta=None):
    """Write a ``masks64_<tag>.npz`` file (compressed, loads without pickle). Returns *path*."""
    if not str(path).endswith(".npz"):
        raise ValueError("mask files must end in .npz")
    relpaths = [str(r).replace("\\", "/") for r in relpaths]
    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.shape[1:] != MASK64_SIZE or labels.dtype != np.uint8:
        raise ValueError(f"labels must be uint8 (N, 64, 64), got {labels.dtype} {labels.shape}")
    if len(relpaths) != len(labels):
        raise ValueError(f"{len(relpaths)} relpaths for {len(labels)} label maps")
    if len(set(relpaths)) != len(relpaths):
        raise ValueError("relpaths must be unique")
    bad = (labels >= len(class_names)) & (labels != IGNORE)
    if bad.any():
        raise ValueError(f"labels contain ids outside 0..{len(class_names) - 1} and {IGNORE}")
    np.savez_compressed(path, relpaths=np.array(relpaths, dtype=str), labels=labels,
                        class_names=np.array(list(class_names), dtype=str),
                        meta=np.array(json.dumps(meta or {}, sort_keys=True)))
    return path


def load_label_source(path):
    """Read a ``masks64_<tag>.npz`` or a 256 ``masks_<tag>.npz`` (scripts/17) as 64 x 64 labels.

    Returns dict: relpaths (list), labels uint8 (N, 64, 64), class_names (list), meta (dict),
    source_size (64 or 256).
    """
    with np.load(path, allow_pickle=False) as z:
        missing = {"relpaths", "labels", "class_names", "meta"} - set(z.files)
        if missing:
            raise ValueError(f"{path} is not a mask file: missing {sorted(missing)}")
        shape = z["labels"].shape
        if len(shape) == 3 and shape[1:] == MASK64_SIZE:
            labels = z["labels"]
            out = {"relpaths": [str(r) for r in z["relpaths"]], "class_names": [str(c) for c in z["class_names"]],
                   "meta": json.loads(str(z["meta"][()])), "source_size": 64}
            if labels.dtype != np.uint8 or len(out["relpaths"]) != len(labels):
                raise ValueError(f"{path}: labels {labels.dtype} {labels.shape} for {len(out['relpaths'])} relpaths")
            out["labels"] = labels
            return out
    full = sm.load_mask_file(path)                     # 256 x 256: checked against the contract, then reduced
    return {"relpaths": full["relpaths"], "labels": labels_to_64(full["labels"]),
            "class_names": full["class_names"], "meta": full["meta"], "source_size": 256}


class MaskTable:
    """Masks keyed like scripts 17 / 18 match images: (condition, phase, file name).

    A split-tree link ``<split>/<subject>/<condition>__<phase>__<file>`` (02) and a
    manifest relpath ``<condition>/<phase>_<eye>_eye/<subject>/<file>`` give the same key
    (src/xai/seg_masks.relpath_key / link_key).
    """

    def __init__(self, source):
        if isinstance(source, (str, os.PathLike)):
            source = load_label_source(source)
        self.relpaths = list(source["relpaths"])
        self.labels = source["labels"]
        self.class_names = list(source["class_names"])
        self.meta = source.get("meta", {})
        self.keys = {}
        for row, rel in enumerate(self.relpaths):
            key = sm.relpath_key(rel)
            if key in self.keys:
                raise ValueError(f"two masks have the key {key}: {self.relpaths[self.keys[key]]} and {rel}")
            self.keys[key] = row

    def __len__(self):
        return len(self.relpaths)

    def _rows(self, names, key_fn):
        rows = []
        for name in names:
            try:
                rows.append(self.keys.get(key_fn(name), -1))
            except ValueError:                           # not a VISOB relpath / split-tree link name
                rows.append(-1)
        return np.array(rows, np.int64)

    def rows_for_links(self, paths):
        """Row of each split-tree link path (-1 where no mask matches)."""
        return self._rows(paths, sm.link_key)

    def rows_for_relpaths(self, relpaths):
        """Row of each manifest relpath (-1 where no mask matches)."""
        return self._rows(relpaths, sm.relpath_key)

    def labels_for(self, rows):
        """(n, 64, 64) labels of *rows*; rows of -1 give all-unknown maps."""
        rows = np.asarray(rows, np.int64)
        out = np.full((len(rows),) + self.labels.shape[1:], IGNORE, np.uint8)
        have = rows >= 0
        out[have] = self.labels[rows[have]]
        return out


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def group_fractions(labels, class_names, groups, grid=8):
    """Share of each grid cell's pixels in each part group: (N, grid, grid, G) float32.

    labels : uint8 (N, h, w) with h, w multiples of *grid*; unknown (255) pixels count for no group
    groups : ordered mapping group name -> class names
    """
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis]
    n, h, w = labels.shape
    if h % grid or w % grid:
        raise ValueError(f"a {h} x {w} label map does not split into {grid} x {grid} cells")
    names = list(class_names)
    out = np.zeros((n, grid, grid, len(groups)), np.float32)
    for g, classes in enumerate(groups.values()):
        missing = [c for c in classes if c not in names]
        if missing:
            raise ValueError(f"classes {missing} are not in the mask file's classes {names}")
        member = np.isin(labels, [names.index(c) for c in classes])
        out[..., g] = member.reshape(n, grid, h // grid, grid, w // grid).mean(axis=(2, 4))
    return out


def eye_fraction(frac, groups):
    """Eye-region share of each cell, (N, grid, grid, 1): the sum of every group but skin."""
    names = list(groups)
    idx = [i for i, g in enumerate(names) if g != NON_EYE_GROUP]
    return frac[..., idx].sum(axis=-1, keepdims=True)


def mask_targets(frac, groups, mode, eps=1e-3, min_part_area=0.005, has_mask=None):
    """Target maps (N, grid, grid, K) and map weights (N, K) for the mask loss.

    mode "eye": one map from the eye region; "parts": one map per group. A map gets
    weight 1 where its region (the eye region, or the group) covers >= min_part_area
    of the image, else 0. Images with has_mask False get weight 0.
    """
    frac = np.asarray(frac, np.float64)                  # normalised in float64, returned as float32
    if mode == "eye":
        maps = eye_fraction(frac, groups)
    elif mode == "parts":
        maps = frac
    else:
        raise ValueError(f"mode must be 'eye' or 'parts', got {mode!r}")
    weights = (maps.mean(axis=(1, 2)) >= min_part_area).astype(np.float32)
    targets = (maps + eps) / (maps + eps).sum(axis=(1, 2), keepdims=True)
    if has_mask is not None:
        weights = weights * np.asarray(has_mask, np.float32)[:, None]
    return targets.astype(np.float32), weights


def region_inputs(labels, class_names, groups, grid=8, has_mask=None):
    """attnet_gate's region input from (N, 64, 64) labels: eye-region fraction per cell, (N, grid, grid, 1).

    Images with has_mask False get region 0 (the gate is then 1/5 everywhere).
    """
    region = eye_fraction(group_fractions(labels, class_names, groups, grid), groups).astype(np.float32)
    if has_mask is not None:
        region = region * np.asarray(has_mask, np.float32)[:, None, None, None]
    return region


def regions_for_rows(table, rows, eye_classes, grid=8):
    """Region input (n, grid, grid, 1) of an attnet_gate model at test time, for rows of *table*.

    eye_classes: from src.models.anatomy_model.region_classes(model). Rows of -1 get region 0.
    """
    rows = np.asarray(rows, np.int64)
    return group_fractions(table.labels_for(rows), table.class_names, {"eye": list(eye_classes)}, grid)


# ---------------------------------------------------------------------------
# augmentation of the label map
# ---------------------------------------------------------------------------

def transform_labels(labels, params, image_shape, datagen):
    """Apply an image's random transform (from ``datagen.get_random_transform``) to its label map.

    Same angle, shear, zoom, flips and fill mode; the shifts are scaled by
    label size / image size; nearest-neighbour; brightness and channel shift are
    left out. labels: uint8 (h, w); image_shape: (H, W, C) channels last.
    """
    from tensorflow.keras.preprocessing.image import apply_affine_transform
    if datagen.data_format != "channels_last":
        raise ValueError("only channels_last generators are supported")
    labels = np.asarray(labels, np.uint8)
    h, w = labels.shape
    big_h, big_w = int(image_shape[0]), int(image_shape[1])
    # apply_affine_transform swaps x and y for channels-last arrays: tx moves the columns, ty the rows
    x = apply_affine_transform(labels[..., np.newaxis].astype(np.float32),
                               params.get("theta", 0), params.get("tx", 0) * w / big_w,
                               params.get("ty", 0) * h / big_h, params.get("shear", 0), params.get("zx", 1),
                               params.get("zy", 1), row_axis=0, col_axis=1, channel_axis=2,
                               fill_mode=datagen.fill_mode, cval=float(IGNORE), order=0)
    if params.get("flip_horizontal", False):
        x = x[:, ::-1]
    if params.get("flip_vertical", False):
        x = x[::-1]
    return np.rint(x[..., 0]).astype(np.uint8)


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------

class MaskedDataGen(tf.keras.utils.Sequence):
    """03's class-folder generator plus masks: ((image, label, targets, weights), label), or
    ((image, label, region), label) for attnet_gate.

    directory, datagen, classes, batch_size, target_size, reshuffle: as 03's ClassListDataGen
        (pass 03's augmenting ImageDataGenerator for train, the plain one for validation / test)
    masks : MaskTable (or a path to a masks64 / 256 mask file)
    mode : "eye" (attn_mask), "parts" or "region" (attnet_gate)
    groups : ordered mapping group -> class names (anatomy.part_groups)
    allow_missing : train on images without a mask with weight 0 (region 0); otherwise
        MissingMasksError with the count
    """

    def __init__(self, directory, datagen, classes, masks, mode, groups, eps=1e-3, min_part_area=0.005,
                 batch_size=32, target_size=(256, 256), class_mode="categorical", reshuffle=False,
                 allow_missing=False, grid=8):
        if mode not in ("eye", "parts", "region"):
            raise ValueError(f"mode must be eye, parts or region, got {mode!r}")
        if class_mode != "categorical":
            raise ValueError("only class_mode='categorical' is supported")
        self.generator = datagen.flow_from_directory(directory, target_size=target_size, batch_size=batch_size,
                                                     class_mode=class_mode, classes=classes)
        self.reshuffle = reshuffle
        self.masks = masks if isinstance(masks, MaskTable) else MaskTable(masks)
        self.mode, self.groups, self.grid = mode, dict(groups), int(grid)
        self.eps, self.min_part_area = float(eps), float(min_part_area)
        self.rows = self.masks.rows_for_links(self.generator.filepaths)
        self.n_missing = int((self.rows < 0).sum())
        for classes_ in self.groups.values():              # fail now, not in the first batch
            group_fractions(np.zeros((1, self.grid, self.grid), np.uint8), self.masks.class_names,
                            {"g": classes_}, self.grid)
        if self.n_missing and not allow_missing:
            examples = [os.path.basename(self.generator.filepaths[i]) for i in np.flatnonzero(self.rows < 0)[:5]]
            raise MissingMasksError(f"{self.n_missing} of {len(self.rows)} images in {directory} have no mask, "
                                    f"e.g. {examples}; make the masks for every split with "
                                    f"21_make_train_masks.py, or pass --allow-missing-masks (weight 0 for them)")

    def coverage(self):
        """{"n_images", "n_with_mask", "share"} of this split."""
        n = len(self.rows)
        return {"n_images": n, "n_with_mask": n - self.n_missing, "share": (n - self.n_missing) / n if n else None}

    def below_min_area(self, chunk=4096):
        """Images of this split with a mask whose target gets weight 0 because its region covers less
        than min_part_area of the image, on the masks before augmentation (an augmented copy can
        differ slightly): mode "eye" an int, mode "parts" {group: int}, mode "region" None."""
        if self.mode == "region":
            return None
        rows = self.rows[self.rows >= 0]
        counts = np.zeros(1 if self.mode == "eye" else len(self.groups), np.int64)
        for start in range(0, len(rows), chunk):
            frac = group_fractions(self.masks.labels_for(rows[start:start + chunk]), self.masks.class_names,
                                   self.groups, self.grid)
            maps = eye_fraction(frac, self.groups) if self.mode == "eye" else frac
            counts += (maps.mean(axis=(1, 2)) < self.min_part_area).sum(axis=0)
        if self.mode == "eye":
            return int(counts[0])
        return {g: int(c) for g, c in zip(self.groups, counts)}

    def __len__(self):
        return len(self.generator)

    def on_epoch_end(self):
        if self.reshuffle:
            self.generator.on_epoch_end()

    def batch_indices(self, idx):
        """Index array of batch *idx*, drawn exactly as DirectoryIterator.__getitem__ draws it."""
        it = self.generator
        if idx >= len(self):
            raise ValueError(f"Asked to retrieve element {idx}, but the Sequence has length {len(self)}")
        if it.seed is not None:
            np.random.seed(it.seed + it.total_batches_seen)
        it.total_batches_seen += 1
        if it.index_array is None:
            it._set_index_array()
        return it.index_array[it.batch_size * idx: it.batch_size * (idx + 1)]

    def load_batch(self, index_array):
        """(images, one-hot labels, augmented 64 x 64 label maps, has_mask) of *index_array*.

        The image path is DirectoryIterator._get_batches_of_transformed_samples line by line.
        """
        it = self.generator
        datagen = it.image_data_generator
        n = len(index_array)
        batch_x = np.zeros((n,) + it.image_shape, dtype=it.dtype)
        batch_labels = np.full((n,) + self.masks.labels.shape[1:], IGNORE, np.uint8)
        filepaths = it.filepaths
        for i, j in enumerate(index_array):
            img = tf.keras.utils.load_img(filepaths[j], color_mode=it.color_mode, target_size=it.target_size,
                                          interpolation=it.interpolation, keep_aspect_ratio=it.keep_aspect_ratio)
            x = tf.keras.utils.img_to_array(img, data_format=it.data_format)
            if hasattr(img, "close"):
                img.close()
            params = datagen.get_random_transform(x.shape)
            x = datagen.apply_transform(x, params)
            x = datagen.standardize(x)
            batch_x[i] = x
            if self.rows[j] >= 0:
                batch_labels[i] = transform_labels(self.masks.labels[self.rows[j]], params, x.shape, datagen)
        batch_y = np.zeros((n, len(it.class_indices)), dtype=it.dtype)
        for i, j in enumerate(index_array):
            batch_y[i, it.classes[j]] = 1.0
        return batch_x, batch_y, batch_labels, self.rows[index_array] >= 0

    def __getitem__(self, idx):
        x, y, labels, has_mask = self.load_batch(self.batch_indices(idx))
        if self.mode == "region":
            return (x, y, region_inputs(labels, self.masks.class_names, self.groups, self.grid, has_mask)), y
        frac = group_fractions(labels, self.masks.class_names, self.groups, self.grid)
        targets, weights = mask_targets(frac, self.groups, self.mode, self.eps, self.min_part_area, has_mask)
        return (x, y, targets, weights), y
