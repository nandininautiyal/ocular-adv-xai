"""
src/xai/seg_masks.py

Anatomical eye masks from a learned segmentation model, in the exact geometry
the recognition model sees.

The heuristic masks in src/xai/masks.py are a Hough circle plus one fixed
ellipse, so "periocular" is the same 69 % of every image. This module replaces
them with the periorbital segmentation model of Nahass et al. (Ophthalmology
Science 2025, arXiv:2409.20407): DeepLabV3-ResNet101, weights ``combine.pth``
from huggingface.co/grnahass/periorbital_segmentation (MIT), loading code
re-implemented from github.com/aiolab/periorbital-dataset (MIT). It was trained
on 256 x 256 eye crops of face photos, with a lot of skin around the eye, so a
tight VISOB crop is first placed in a larger canvas.

Pipeline for one image
----------------------
1. Native crop (VISOB: 240 x 160, width x height), RGB uint8.
2. Replicate-pad it by a factor (``factor``, default 1.8) and resize the canvas
   to 256 x 256. ``canvas="stretch"`` pads each side by the same factor, so the eye
   keeps the 1.5x vertical stretch of the model input (the geometry of the
   verified 4-image pilot); ``canvas="square"`` pads to a square and keeps the
   native aspect ratio.
3. Normalise like the authors' code, (x / 255 - 0.5) / 0.5, and run the network.
4. Resize the logits back to the canvas (bilinear), crop the native region,
   take the arg-max and map the model's classes to ours (``CLASS_NAMES``).
5. Resize the label map to the recognition model's input geometry, 256 x 256
   with PIL nearest-neighbour, exactly like
   ``tf.keras.utils.load_img(path, target_size=(256, 256))`` resizes the image,
   so label pixel (i, j) belongs to input pixel (i, j).

Classes
-------
``CLASS_NAMES`` = skin, iris, sclera, lid, caruncle, brow (ids 0-5); 255 means
unknown / ignore (hand labels only). The model's own ids are 0 background,
1 brow, 2 sclera, 3 iris, 4 caruncle, 5 lid, as in the authors' package
(periorbital_package 0.1.3, pkg_utils.extract_and_split_masks). The model has
no pupil class: its iris includes the pupil, and so does ours. Its "lid" is the
upper lid from the crease to the lid margin, upper lashes included; the lower
lid is labelled background, which becomes skin.

Mask file (shared contract with the runner and data-check work packages)
------------------------------------------------------------------------
``masks_<tag>.npz``: ``relpaths`` (manifest relpath per image), ``labels``
(uint8, N x 256 x 256, model-input geometry), ``class_names`` and ``meta``
(a 0-d string array holding JSON). See :func:`save_mask_file`.

PyTorch and torchvision are imported only when the network is built, so
everything else here (geometry, mask files, metrics) works without them.
The masks are zero-shot: validate them against hand labels
(scripts/18_validate_masks.py) before reporting region-level results.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image

MODEL_SIZE = (256, 256)          # (height, width) of the recognition model's input
NET_SIZE = 256                   # input side of the segmentation network
DEFAULT_PAD_FACTOR = 1.8
CANVASES = ("stretch", "square")

CLASS_NAMES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")
IGNORE = 255

# The segmentation model's own output ids, and where each one goes in CLASS_NAMES.
PERIORBITAL_CLASSES = ("background", "brow", "sclera", "iris", "caruncle", "lid")
PERIORBITAL_TO_CANONICAL = np.array(
    [CLASS_NAMES.index("skin" if name == "background" else name) for name in PERIORBITAL_CLASSES],
    dtype=np.uint8)

# The four regions of src/compare/regions.py, and the three that both mask
# sources can be compared on ("iris" there includes the pupil).
HEURISTIC_REGIONS = ("pupil", "iris", "sclera", "periocular")
COMPARISON_REGIONS = ("iris", "sclera", "periocular")

# Label colours for overlays and hand labelling (docs/mask_labelling.md).
PALETTE = np.array([(0, 0, 0),         # skin (and everything else): black
                    (0, 0, 255),       # iris: blue
                    (0, 255, 0),       # sclera: green
                    (255, 0, 255),     # lid: magenta
                    (255, 255, 0),     # caruncle: yellow
                    (255, 0, 0)],      # brow: red
                   dtype=np.uint8)
IGNORE_COLOUR = (128, 128, 128)

WEIGHTS_FILE = "combine.pth"
WEIGHTS_SHA256 = "126a7070cc327e55f6bae5e10448eebdfb0180964babe5897c9566255741b21b"
WEIGHTS_REPO = "grnahass/periorbital_segmentation"
WEIGHTS_REVISION = "e4bc5232e3eebbb2742881a65d788ec28e3ea002"
WEIGHTS_URL = f"https://huggingface.co/{WEIGHTS_REPO}/resolve/{WEIGHTS_REVISION}/{WEIGHTS_FILE}"


# --- geometry -------------------------------------------------------------------

def _check_labels_2d(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 label map, got shape {labels.shape} and dtype {labels.dtype}")
    return labels


def _check_rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 image (H, W, 3), got shape {image.shape} and dtype {image.dtype}")
    return image


def to_model_geometry(labels, size=MODEL_SIZE):
    """Resize a native label map the way the runner resizes the image.

    ``tf.keras.utils.load_img(path, target_size=(256, 256))`` uses PIL's
    nearest-neighbour resize; this is the same call on the label map.

    Parameters
    ----------
    labels : ndarray (H, W), uint8 — native geometry
    size : (height, width) of the model input

    Returns
    -------
    ndarray (height, width), uint8
    """
    labels = _check_labels_2d(labels)
    out = Image.fromarray(labels).resize((size[1], size[0]), Image.NEAREST)
    return np.array(out, dtype=np.uint8)


def model_input_image(image, size=MODEL_SIZE):
    """The recognition model's view of a native RGB uint8 image (PIL nearest, as load_img)."""
    image = _check_rgb_uint8(image)
    return np.array(Image.fromarray(image).resize((size[1], size[0]), Image.NEAREST), dtype=np.uint8)


def nearest_index(n_in, n_out):
    """Source index of every output pixel along one axis of PIL's nearest resize.

    Taken from PIL itself (not re-derived), so it matches load_img on any Pillow version.
    """
    line = Image.fromarray(np.arange(n_in, dtype=np.int32)[None, :])
    return np.array(line.resize((n_out, 1), Image.NEAREST), dtype=np.int64)[0]


def from_model_geometry(array, native_hw):
    """Undo :func:`to_model_geometry` exactly (also works for (H, W, C) images).

    Upsampling by nearest neighbour copies every native pixel at least once, so
    the native map is recovered by reading one copy of each pixel.

    Parameters
    ----------
    array : ndarray (256, 256) or (256, 256, C) in model-input geometry
    native_hw : (height, width) of the native image, e.g. (160, 240)
    """
    array = np.asarray(array)
    h, w = int(native_hw[0]), int(native_hw[1])
    picks = []
    for n_native, n_model in ((h, array.shape[0]), (w, array.shape[1])):
        src = nearest_index(n_native, n_model)
        first = np.searchsorted(src, np.arange(n_native))
        if first.max(initial=0) >= n_model or not (src[np.minimum(first, n_model - 1)] == np.arange(n_native)).all():
            raise ValueError(f"a {n_model}-pixel axis does not contain every one of {n_native} native pixels "
                             f"(the native image is larger than the model input)")
        picks.append(first)
    return array[np.ix_(picks[0], picks[1])].copy()


def pad_widths(native_hw, factor=DEFAULT_PAD_FACTOR, canvas="stretch"):
    """Replicate padding that places the crop in a larger canvas.

    Parameters
    ----------
    native_hw : (height, width) of the crop
    factor : canvas side / crop side (>= 1)
    canvas : "stretch" pads each axis by ``factor`` (aspect ratio of the crop kept,
        so the resize to 256 x 256 stretches it like the model input);
        "square" pads to a square of side ``factor * max(height, width)``

    Returns
    -------
    (top, bottom, left, right) : ints
    """
    h, w = int(native_hw[0]), int(native_hw[1])
    if factor < 1:
        raise ValueError("pad factor must be >= 1")
    if canvas == "stretch":
        ph = int(math.floor((factor - 1.0) * h / 2.0 + 0.5))
        pw = int(math.floor((factor - 1.0) * w / 2.0 + 0.5))
        return ph, ph, pw, pw
    if canvas == "square":
        side = max(int(math.floor(factor * max(h, w) + 0.5)), h, w)
        top, left = (side - h) // 2, (side - w) // 2
        return top, side - h - top, left, side - w - left
    raise ValueError(f"canvas must be one of {CANVASES}, got {canvas!r}")


def preprocess(image, factor=DEFAULT_PAD_FACTOR, canvas="stretch", size=NET_SIZE):
    """Native RGB uint8 crop -> network input and the padding used.

    Returns
    -------
    x : ndarray (3, size, size), float32 in [-1, 1] — (x / 255 - 0.5) / 0.5 of the resized canvas
    pads : (top, bottom, left, right)
    """
    image = _check_rgb_uint8(image)
    pads = pad_widths(image.shape[:2], factor, canvas)
    top, bottom, left, right = pads
    padded = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="edge")
    resized = np.asarray(Image.fromarray(padded).resize((size, size), Image.BILINEAR), dtype=np.float32)
    x = (resized / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32), pads


def postprocess(logits, native_hw, pads):
    """Network logits for the padded canvas -> the network's class ids on the native crop.

    The logits are resized back to the canvas with bilinear interpolation
    before the arg-max, so boundaries are not quantised to the 256 grid.

    Parameters
    ----------
    logits : ndarray (C, s, s)
    native_hw : (height, width) of the crop
    pads : (top, bottom, left, right) from :func:`preprocess`

    Returns
    -------
    ndarray (height, width), uint8 — ids in the network's own class order
    """
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"expected logits of shape (C, H, W), got {logits.shape}")
    h, w = int(native_hw[0]), int(native_hw[1])
    top, bottom, left, right = pads
    hc, wc = h + top + bottom, w + left + right
    up = cv2.resize(np.ascontiguousarray(logits.transpose(1, 2, 0)), (wc, hc), interpolation=cv2.INTER_LINEAR)
    if up.ndim == 2:
        up = up[..., None]
    return up[top:top + h, left:left + w].argmax(axis=-1).astype(np.uint8)


def remap(ids, class_map=PERIORBITAL_TO_CANONICAL):
    """Map the network's class ids to ``CLASS_NAMES`` ids (background -> skin, ...)."""
    ids = np.asarray(ids)
    class_map = np.asarray(class_map, dtype=np.uint8)
    if ids.size and int(ids.max()) >= len(class_map):
        raise ValueError(f"class id {int(ids.max())} has no mapping (the map covers {len(class_map)} classes)")
    return class_map[ids]


def _predict_checked(predict_fn, x, size):
    logits = np.asarray(predict_fn(x), dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != len(x) or logits.shape[2:] != (size, size):
        raise ValueError(f"predict_fn returned shape {logits.shape}, expected ({len(x)}, C, {size}, {size})")
    return logits


def segment(images, predict_fn, batch_size=16, factor=DEFAULT_PAD_FACTOR, canvas="stretch",
            flip_tta=False, class_map=PERIORBITAL_TO_CANONICAL, size=NET_SIZE):
    """Canonical label maps for native crops, in NATIVE geometry.

    Use :func:`to_model_geometry` on each map to get the runner's geometry.

    Parameters
    ----------
    images : list of ndarray (H, W, 3), uint8 — native crops (sizes may differ)
    predict_fn : callable, float32 (N, 3, size, size) -> logits (N, C, size, size);
        :func:`torch_predictor` for the real network, any function in tests
    batch_size : images per call of ``predict_fn``
    factor, canvas : padding, see :func:`pad_widths`
    flip_tta : also run the mirrored image and average the (re-mirrored) logits
    class_map : network id -> ``CLASS_NAMES`` id

    Returns
    -------
    list of ndarray (H, W), uint8
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    labels = []
    for start in range(0, len(images), batch_size):
        chunk = [_check_rgb_uint8(img) for img in images[start:start + batch_size]]
        prepared = [preprocess(img, factor, canvas, size) for img in chunk]
        x = np.stack([p[0] for p in prepared])
        logits = _predict_checked(predict_fn, x, size)
        if flip_tta:
            mirrored = _predict_checked(predict_fn, np.ascontiguousarray(x[..., ::-1]), size)
            logits = 0.5 * (logits + mirrored[..., ::-1])
        for img, (_, pads), lg in zip(chunk, prepared, logits):
            labels.append(remap(postprocess(lg, img.shape[:2], pads), class_map))
    return labels


# --- the network (PyTorch, imported lazily) -------------------------------------------

def require_torch():
    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError as err:
        raise ImportError("The periorbital segmentation model needs PyTorch and torchvision "
                          "(preinstalled on Kaggle; elsewhere: pip install torch torchvision). "
                          "Mask files, geometry and metrics in this module work without them.") from err
    return torch


def file_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_weights(dest_dir, url=WEIGHTS_URL, sha256=WEIGHTS_SHA256, filename=WEIGHTS_FILE, attempts=3,
                  retry_wait=5.0):
    """Path to the pinned weights file, downloading it into *dest_dir* if needed.

    The file is checked against *sha256* before it is used; a file already in *dest_dir* with another
    sha256 raises. Safe when several processes fetch at once (train_anatomy.ipynb starts 21 and 17 on two
    GPUs at the same moment): each download goes to its own temporary file in *dest_dir* and is moved
    into place atomically only after its sha256 matches, so no process reads or replaces another one's
    half-written file. (A shared ``combine.pth.part`` made both jobs fail on 22 Nov: one found the file
    gone, the other a mix of two downloads.) A failed download or a mismatch is tried again, *attempts*
    times in all, *retry_wait* x attempt seconds apart, before it raises.
    """
    path = os.path.join(dest_dir, filename)
    if os.path.exists(path):
        if file_sha256(path) == sha256:
            return path
        raise ValueError(f"{path} exists but its sha256 is not the pinned {sha256}; delete it or pass the "
                         f"right file")
    os.makedirs(dest_dir, exist_ok=True)
    error = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        fd, partial = tempfile.mkstemp(prefix=f"{filename}.", suffix=".part", dir=dest_dir)
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, partial)
            digest = file_sha256(partial)
            if digest == sha256:
                os.replace(partial, path)
                return path
            error = ValueError(f"downloaded {url} has sha256 {digest}, expected {sha256}")
        except OSError as err:                    # network errors (urllib's URLError is an OSError)
            error = err
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        if os.path.exists(path) and file_sha256(path) == sha256:     # another process finished meanwhile
            return path
        if attempt < attempts:
            time.sleep(retry_wait * attempt)
    raise ValueError(f"could not fetch {url} with sha256 {sha256} in {attempts} attempt(s): {error}") from error


def build_periorbital_model(aux_loss=True):
    """DeepLabV3-ResNet101 with the authors' 6-class head (full_tester.py build_model), untrained."""
    require_torch()
    from torch import nn
    from torchvision.models.segmentation import deeplabv3_resnet101

    try:
        model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_loss=aux_loss)
    except TypeError:                                    # torchvision < 0.13
        model = deeplabv3_resnet101(pretrained=False, pretrained_backbone=False, aux_loss=aux_loss)
    # The authors replace the whole DeepLab head (ASPP included) with this small head.
    model.classifier = nn.Sequential(
        nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, len(PERIORBITAL_CLASSES), kernel_size=1, stride=1),
    )
    return model


def load_periorbital_model(weights, device="cpu", expected_sha256=WEIGHTS_SHA256):
    """Load ``combine.pth`` (sha256-checked, tensors only) into the network, in eval mode.

    Parameters
    ----------
    weights : path to the weights file
    device : torch device string ("cpu", "cuda", "cuda:0")
    expected_sha256 : refuse a different file; None skips the check (tests only)
    """
    torch = require_torch()
    if expected_sha256 is not None:
        digest = file_sha256(weights)
        if digest != expected_sha256:
            raise ValueError(f"{weights} has sha256 {digest}, expected {expected_sha256}")
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError as err:
        raise RuntimeError("this PyTorch is too old for torch.load(weights_only=True); use torch >= 1.13") from err
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model = build_periorbital_model(aux_loss=any(k.startswith("aux_classifier.") for k in state))
    model.load_state_dict(state, strict=True)
    model.aux_classifier = None                          # the auxiliary head is unused at inference
    return model.to(device).eval()


def torch_predictor(model, device="cpu", amp=False):
    """Wrap the network as ``predict_fn`` for :func:`segment` (NumPy in, NumPy logits out)."""
    torch = require_torch()
    device = str(device)

    def predict(x):
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
            if amp and device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(batch)["out"]
            else:
                out = model(batch)["out"]
            return out.float().cpu().numpy()

    return predict


# --- mask files (contract 1) ------------------------------------------------------------

def _clean_relpath(relpath):
    return str(relpath).replace("\\", "/")


def _check_label_values(labels, n_classes, chunk=256):
    labels = np.asarray(labels, dtype=np.uint8)
    counts = np.zeros(256, np.int64)
    for start in range(0, len(labels), chunk):          # bincount works in int64: keep the copy small
        counts += np.bincount(labels[start:start + chunk].ravel(), minlength=256)
    bad = np.flatnonzero(counts[n_classes:IGNORE])
    if bad.size:
        raise ValueError(f"labels contain ids {[int(b) + n_classes for b in bad[:5]]} outside 0..{n_classes - 1} "
                         f"and {IGNORE}")


def save_mask_file(path, relpaths, labels, class_names=CLASS_NAMES, meta=None):
    """Write a ``masks_<tag>.npz`` file (compressed; loads without pickle).

    Parameters
    ----------
    path : output path ending in .npz
    relpaths : manifest relpath of each image (data-root relative; "/" separators are enforced)
    labels : ndarray (N, 256, 256), uint8 — model-input geometry, ids into *class_names*, 255 = unknown
    class_names : names of the label ids
    meta : dict, stored as JSON (model, weights file and sha256, padding, code commit, created, ...)

    Returns
    -------
    path
    """
    if not str(path).endswith(".npz"):
        raise ValueError("mask files must end in .npz")
    relpaths = [_clean_relpath(r) for r in relpaths]
    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.shape[1:] != MODEL_SIZE or labels.dtype != np.uint8:
        raise ValueError(f"labels must be uint8 with shape (N, {MODEL_SIZE[0]}, {MODEL_SIZE[1]}), "
                         f"got {labels.dtype} {labels.shape}")
    if len(relpaths) != len(labels):
        raise ValueError(f"{len(relpaths)} relpaths for {len(labels)} label maps")
    if len(set(relpaths)) != len(relpaths):
        raise ValueError("relpaths must be unique")
    _check_label_values(labels, len(class_names))
    np.savez_compressed(path, relpaths=np.array(relpaths, dtype=str), labels=labels,
                        class_names=np.array(list(class_names), dtype=str),
                        meta=np.array(json.dumps(meta or {}, sort_keys=True)))
    return path


def load_mask_file(path):
    """Read a mask file written by :func:`save_mask_file` and check it against the contract.

    Returns
    -------
    dict with ``relpaths`` (list of str), ``labels`` (uint8 N x 256 x 256),
    ``class_names`` (list of str), ``meta`` (dict) and ``index`` (relpath -> row)
    """
    with np.load(path, allow_pickle=False) as z:
        missing = {"relpaths", "labels", "class_names", "meta"} - set(z.files)
        if missing:
            raise ValueError(f"{path} is not a mask file: missing {sorted(missing)}")
        relpaths = [str(r) for r in z["relpaths"]]
        labels = z["labels"]
        class_names = [str(c) for c in z["class_names"]]
        meta = json.loads(str(z["meta"][()]))
    if labels.ndim != 3 or labels.shape[1:] != MODEL_SIZE or labels.dtype != np.uint8:
        raise ValueError(f"{path}: labels are {labels.dtype} {labels.shape}, expected uint8 (N, 256, 256)")
    if len(relpaths) != len(labels):
        raise ValueError(f"{path}: {len(relpaths)} relpaths for {len(labels)} label maps")
    _check_label_values(labels, len(class_names))
    return {"relpaths": relpaths, "labels": labels, "class_names": class_names, "meta": meta,
            "index": {r: i for i, r in enumerate(relpaths)}}


# --- regions on the heatmap grid ------------------------------------------------------

def class_grids(labels, n_classes=len(CLASS_NAMES), grid=8):
    """Fraction of every grid cell covered by each class, area-averaged like ``masks_to_grid``.

    Parameters
    ----------
    labels : ndarray (256, 256), uint8 — model-input geometry
    n_classes : number of class ids (unknown pixels count for no class)
    grid : side of the heatmap (8 for Xception)

    Returns
    -------
    ndarray (n_classes, grid, grid), float32
    """
    labels = _check_labels_2d(labels)
    if labels.shape != MODEL_SIZE:
        raise ValueError(f"labels must have shape {MODEL_SIZE}, got {labels.shape}")
    if grid < 1:
        raise ValueError("grid must be a positive integer.")
    return np.stack([cv2.resize((labels == k).astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
                     for k in range(n_classes)]).astype(np.float32)


def class_areas(labels, class_names=CLASS_NAMES, grid=8):
    """``area_<class>``: each class's share of the image, measured on the heatmap grid."""
    grids = class_grids(labels, len(class_names), grid)
    return {f"area_{name}": float(g.mean()) for name, g in zip(class_names, grids)}


def to_heuristic_regions(labels, class_names=CLASS_NAMES):
    """Map anatomical labels to the regions of src/compare/regions.py.

    iris -> iris, sclera -> sclera, every other known class -> periocular;
    the pupil is empty because the learned iris already includes it. Unknown
    (255) pixels belong to no region.

    Returns
    -------
    dict of boolean (H, W) masks with the keys ``get_concept_masks`` returns:
    pupil, iris, sclera, periocular, iris_pupil
    """
    labels = np.asarray(labels)
    names = list(class_names)
    iris = labels == names.index("iris")
    sclera = labels == names.index("sclera")
    periocular = (labels != IGNORE) & ~iris & ~sclera
    return {"pupil": np.zeros(labels.shape, bool), "iris": iris, "sclera": sclera,
            "periocular": periocular, "iris_pupil": iris.copy()}


def comparison_labels(regions):
    """Label map over ``COMPARISON_REGIONS`` (0 iris incl. pupil, 1 sclera, 2 periocular, 255 unknown).

    *regions* is a dict from ``get_concept_masks`` or :func:`to_heuristic_regions`,
    so heuristic and learned masks can be scored on the same three regions.
    """
    iris = np.asarray(regions["iris"], bool) | np.asarray(regions["pupil"], bool)
    out = np.full(iris.shape, IGNORE, np.uint8)
    out[np.asarray(regions["periocular"], bool)] = COMPARISON_REGIONS.index("periocular")
    out[np.asarray(regions["sclera"], bool)] = COMPARISON_REGIONS.index("sclera")
    out[iris] = COMPARISON_REGIONS.index("iris")
    return out


# --- agreement metrics ---------------------------------------------------------------

def iou_dice(pred, gt, n_classes, ignore=IGNORE):
    """Per-class IoU and Dice of a predicted label map against a reference.

    Pixels that are unknown in the reference are skipped; unknown pixels in the
    prediction count as misses. A class absent from both maps gets NaN, not 1.

    Returns
    -------
    dict of arrays of length *n_classes*: ``iou``, ``dice``, ``intersection``,
    ``union``, ``pred_px``, ``gt_px``
    """
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction {pred.shape} and reference {gt.shape} differ in shape")
    valid = gt != ignore
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    if g.size and g.max() >= n_classes:
        raise ValueError(f"reference contains id {int(g.max())} >= n_classes {n_classes}")
    p = np.where(p < n_classes, p, n_classes)            # unknown / out-of-range predictions: no class
    conf = np.bincount(g * (n_classes + 1) + p, minlength=n_classes * (n_classes + 1))
    conf = conf.reshape(n_classes, n_classes + 1)
    inter = np.diag(conf[:, :n_classes]).astype(np.float64)
    gt_px = conf.sum(axis=1).astype(np.float64)
    pred_px = conf[:, :n_classes].sum(axis=0).astype(np.float64)
    union = gt_px + pred_px - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
        dice = np.where(gt_px + pred_px > 0, 2 * inter / (gt_px + pred_px), np.nan)
    return {"iou": iou, "dice": dice, "intersection": inter, "union": union, "pred_px": pred_px, "gt_px": gt_px}


# --- hand labelling: file names, colours, sampling --------------------------------------------

def label_filename(relpath):
    """File name of an image's hand label (and of its copy for labelling): relpath with "/" -> "__"."""
    relpath = _clean_relpath(relpath)
    name = relpath.replace("/", "__")
    # "a_/b" and "a/_b" would both become "a___b": refuse any relpath the name cannot be decoded back to
    if "__" in relpath or relpath_from_filename(name) != relpath:
        raise ValueError(f"relpath {relpath!r} cannot be encoded as a file name (it contains '__', or a "
                         f"folder or file name next to a '/' starts or ends with '_')")
    return name


def relpath_from_filename(name):
    """Inverse of :func:`label_filename`."""
    return os.path.basename(str(name)).replace("__", "/")


def colour_labels(labels):
    """RGB uint8 picture of a label map in ``PALETTE`` colours (unknown pixels grey)."""
    labels = _check_labels_2d(labels)
    out = np.empty(labels.shape + (3,), np.uint8)
    out[:] = IGNORE_COLOUR
    for k, colour in enumerate(PALETTE):
        out[labels == k] = colour
    return out


def labels_from_colours(rgb):
    """Class ids of an RGB picture in ``PALETTE`` colours.

    Returns
    -------
    (labels, n_off_palette) : uint8 (H, W) with unknown colours set to 255, and how many pixels that was
    """
    rgb = _check_rgb_uint8(rgb)
    code = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    out = np.full(rgb.shape[:2], IGNORE, np.uint8)
    known = np.zeros(rgb.shape[:2], bool)
    for k, (r, g, b) in enumerate(PALETTE.astype(np.int64)):
        hit = code == ((r << 16) | (g << 8) | b)
        out[hit] = k
        known |= hit
    known |= code == ((IGNORE_COLOUR[0] << 16) | (IGNORE_COLOUR[1] << 8) | IGNORE_COLOUR[2])
    return out, int((~known).sum())


def read_label_png(path, n_classes=len(CLASS_NAMES)):
    """Read a hand label: single-channel ids (0..n-1, 255) or a ``PALETTE``-coloured picture.

    Palette (indexed) PNGs are read by colour when their colours are ours, and
    as raw ids otherwise. An alpha channel is ignored.

    Returns
    -------
    (labels, n_off_palette) : uint8 (H, W), and the number of pixels set to 255
    because their colour or id was not valid
    """
    with Image.open(path) as img:
        img.load()
        mode = img.mode
        if mode == "P":
            labels, off = labels_from_colours(np.asarray(img.convert("RGB")))
            if off == 0:
                return labels, 0
            raw = np.asarray(img, dtype=np.int64)
        elif mode in ("RGB", "RGBA"):
            return labels_from_colours(np.asarray(img.convert("RGB")))
        elif mode == "LA":                               # grey-level ids saved with an alpha channel (GIMP)
            raw = np.asarray(img.convert("L")).astype(np.int64)
        else:
            raw = np.asarray(img).astype(np.int64)
    if raw.ndim != 2:
        raise ValueError(f"{path}: cannot read a label map from a {mode} image")
    bad = ~(((raw >= 0) & (raw < n_classes)) | (raw == IGNORE))
    return np.where(bad, IGNORE, raw).astype(np.uint8), int(bad.sum())


def write_label_png(path, labels, colour=False):
    """Save a label map as single-channel ids (default) or as a ``PALETTE`` colour picture."""
    labels = _check_labels_2d(labels)
    Image.fromarray(colour_labels(labels) if colour else labels).save(path)


def overlay(image, labels, alpha=0.45):
    """Tint each labelled class of an RGB uint8 image in its colour (skin is left untinted)."""
    image = _check_rgb_uint8(image)
    labels = _check_labels_2d(labels)
    if image.shape[:2] != labels.shape:
        raise ValueError(f"image {image.shape[:2]} and labels {labels.shape} differ in size")
    colours = colour_labels(labels).astype(np.float32)
    out = image.astype(np.float32)
    tint = labels != CLASS_NAMES.index("skin")
    out[tint] = (1 - alpha) * out[tint] + alpha * colours[tint]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def relpath_key(relpath):
    """(condition, phase, file name) of a VISOB relpath: what a split-tree link name keeps of it."""
    parts = _clean_relpath(relpath).split("/")
    if len(parts) < 3:
        raise ValueError(f"not a VISOB relpath: {relpath!r}")
    return parts[0], parts[1].split("_")[0], parts[-1]


def link_key(link_path):
    """(condition, phase, file name) of a link made by scripts/02_make_splits.py:
    ``<split-dir>/<split>/<subject>/<condition>__<phase>__<file>``."""
    parts = os.path.basename(_clean_relpath(link_path)).split("__")
    if len(parts) != 3:
        raise ValueError(f"not a split-tree link name: {link_path!r}")
    return tuple(parts)


def image_table(frame, data_root=None, known_relpaths=()):
    """Normalise an image list for the mask scripts.

    Parameters
    ----------
    frame : DataFrame with a ``relpath`` and/or ``link_path`` column (manifest,
        correct_test_<tag>.csv from 04, images_<tag>.csv from 05)
    data_root : VISOB root; used to find files whose link no longer exists and to
        read relpaths off symlinks
    known_relpaths : relpaths (e.g. splits/visob_test.csv) used to recover the relpath
        of rows that only have a link_path

    Returns
    -------
    DataFrame, one row per distinct relpath (first occurrence kept): relpath,
    link_path ("" if none), path (readable file or None), device, lighting, eye,
    phase, subject
    """
    import pandas as pd

    if "relpath" not in frame and "link_path" not in frame:
        raise ValueError("the image list needs a relpath or a link_path column")
    links = frame["link_path"].fillna("").astype(str).tolist() if "link_path" in frame else [""] * len(frame)
    rels = frame["relpath"].tolist() if "relpath" in frame else [None] * len(frame)
    lookup = {relpath_key(r): _clean_relpath(r) for r in known_relpaths}
    out = []
    for rel, link in zip(rels, links):
        if not isinstance(rel, str) or not rel:
            try:
                rel = lookup.get(link_key(link)) if link else None
            except ValueError:                           # not a split-tree link name
                rel = None
            if rel is None and link and data_root and os.path.islink(link):
                rel = os.path.relpath(os.readlink(link), data_root)
            if rel is None:
                raise ValueError(f"cannot find the relpath of {link!r}: give a CSV with a relpath column, "
                                 f"or the pinned test list / --data-root")
        rel = _clean_relpath(rel)
        path = None
        if link and os.path.exists(link):
            path = link
        elif data_root and os.path.exists(os.path.join(data_root, rel)):
            path = os.path.join(data_root, rel)
        out.append({"relpath": rel, "link_path": link, "path": path, **strata_from_relpath(rel)})
    table = pd.DataFrame(out, columns=["relpath", "link_path", "path", "device", "lighting", "eye", "phase",
                                       "subject"])
    return table.drop_duplicates("relpath").reset_index(drop=True)


_PHASE_EYE = re.compile(r"^(?P<phase>enrollment|verification)_(?P<eye>left|right)_eye$", re.IGNORECASE)


def strata_from_relpath(relpath):
    """Phone, lighting, eye, phase and subject parsed from a VISOB relpath
    (``<device>_<lighting>_short/<phase>_<eye>_eye/<subject>/<subject>_<idx>_<l|r>_c.png``),
    the same way scripts/01_build_manifest.py parses them; "unknown" / -1 if a part is missing."""
    parts = _clean_relpath(relpath).split("/")
    condition = parts[0].lower() if len(parts) > 1 else ""
    lighting = next((name for name in ("office", "day", "dim") if name in condition), "unknown")
    match = _PHASE_EYE.match(parts[1]) if len(parts) > 2 else None
    subject = parts[-1].split("_")[0]
    return {"device": condition.split("_")[0] if condition else "unknown", "lighting": lighting,
            "eye": match.group("eye").lower() if match else "unknown",
            "phase": match.group("phase").lower() if match else "unknown",
            "subject": int(subject) if subject.isdigit() else -1}


def stratified_sample(frame, n, keys=("device", "lighting", "eye"), group="subject", seed=0):
    """About *n* rows spread evenly over the strata (combinations of *keys*).

    Each stratum gets n // k rows (k strata) and a seeded random n % k strata
    get one more; if a stratum has too few rows, the shortfall is filled from
    the rows left over in all strata. Rows of a subject (*group*) that is not
    used yet are preferred, so the images are as independent as the data
    allow. Seeded, and the input frame is not changed.

    Returns
    -------
    DataFrame with the chosen rows, in the input order
    """
    rng = np.random.default_rng(seed)
    n = min(int(n), len(frame))
    if n <= 0:
        return frame.iloc[:0]
    strata = [np.asarray(idx) for _, idx in sorted(frame.groupby(list(keys), sort=True).indices.items())]
    quota = np.full(len(strata), n // len(strata))
    quota[rng.permutation(len(strata))[:n % len(strata)]] += 1
    groups = frame[group].to_numpy() if group in frame else np.arange(len(frame))
    used, picked = set(), []

    def take(candidates, count):
        order = rng.permutation(candidates)
        chosen, seen = [], set()
        for i in order:                                  # first one row per not-yet-used subject
            if len(chosen) < count and groups[i] not in used and groups[i] not in seen:
                chosen.append(i)
                seen.add(groups[i])
        taken = set(chosen)
        for i in order:                                  # then repeats, only if needed
            if len(chosen) < count and i not in taken:
                chosen.append(i)
                taken.add(i)
        used.update(groups[i] for i in chosen)
        return chosen

    for s in rng.permutation(len(strata)):
        picked += take(strata[s], min(quota[s], len(strata[s])))
    if len(picked) < n:                                  # small strata: fill from what is left
        rest = np.setdiff1d(np.arange(len(frame)), picked)
        picked += take(rest, n - len(picked))
    return frame.iloc[sorted(picked)]
