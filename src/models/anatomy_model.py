"""
src/models/anatomy_model.py

Anatomy-guided heads for the Xception + ArcFace recognition model, and the
mask-supervised trainer behind them.

Why this module exists
----------------------
The paper explains periocular matching at the level of eye parts (iris,
sclera, lid, brow, skin) under adaptive attack. The base paper's model
(Flatten -> Dense -> ArcFace) has no part structure, so every part-level
statement about it has to come from a post-hoc method. The heads here give the
model that structure while keeping its input, backbone, ArcFace head and
recipe:

- ``gap``          GlobalAveragePooling head (a plain pooling baseline);
- ``attn``         one learned attention map over the 8 x 8 feature cells, no mask loss;
- ``attn_mask``    the same architecture, trained with a KL loss that pulls the map
                   towards the eye region of the training mask (masks are needed
                   only during training, never at test time);
- ``parts``        one attention map per part group, a linear embedding per part and
                   their sum: the margin-free cosine of a class is exactly the sum of
                   the per-part terms (``part_receipt``);
- ``attnet_gate``  the fixed gate of AttNet (Zhao & Kumar, TIFS 2018): features x5
                   inside the eye region and x1/5 outside, then the ``gap`` head. It
                   needs the eye mask at test time too (a third input ``region``).

Every head ends in the vendored ArcFace layer (same s and m), the backbone is the
same Xception on raw [0, 1] pixels with the same ``fine_tune_at``, and the CAM
target ``block14_sepconv2_act`` exists in every model, so ``load_checkpoint``,
``get_margin_free``, ``predict_logits``, LayerCAM / Eigen-CAM and
``src.eval.verification.embed`` work unchanged on the dual-input models.

Pitfalls
--------
- ``find_feature_layer`` returns ``block14_sepconv2_act`` (the features BEFORE the
  attention / gate). Pass the layer explicitly when a post-attention map is meant.
- An 8 x 8 cell of Xception sees most of the 256 x 256 image, so "part" means "what
  the attention map selects on the feature grid", not a pixel-exact region. The
  receipt is exact for the score, not for pixels.
- ``attnet_gate`` has three inputs; the helpers that take only images need
  :func:`bind_region` (XAI on a fixed set of images) or :func:`embed_regions`.

Image contract: float32 (N, 256, 256, 3) in [0, 1]; dual input [image, label],
zero label at inference.
"""

import os
import weakref

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.recognition_model import ArcFace, build_model

ARCHS = ("baseline", "gap", "attn", "attn_mask", "parts", "attnet_gate")
MASK_LOSS_ARCHS = ("attn_mask", "parts")           # trained with the mask loss (MaskTrainer)
MASK_ARCHS = ("attn_mask", "parts", "attnet_gate")  # need a masks file to train
FEATURE_LAYER = "block14_sepconv2_act"

DEFAULT_PART_GROUPS = {"iris": ["iris"], "sclera": ["sclera", "caruncle"], "lid": ["lid"], "brow": ["brow"],
                       "skin": ["skin"]}
NON_EYE_GROUP = "skin"     # the eye region is every group except this one
ANATOMY_DEFAULTS = {
    "part_groups": DEFAULT_PART_GROUPS,
    "hidden": 256,             # width of the attention MLP (1 x 1 convolutions)
    "mask_loss_weight": 1.0,   # lambda
    "eps": 1.0e-3,             # added to every cell of a target map before normalising
    "min_part_area": 0.005,    # a part map counts in the loss only if the part covers this share of the image
    "gate_inside": 5.0,        # AttNet gains: x5 inside the eye region ...
    "gate_outside": 0.2,       # ... and x1/5 outside
}


def anatomy_settings(cfg):
    """The ``anatomy`` section of *cfg* with defaults filled in, checked.

    ``part_groups`` keeps its order (YAML mappings keep it); groups must not share
    a class, and one group must be ``skin`` (the complement of the eye region).
    """
    section = dict((cfg or {}).get("anatomy") or {})
    unknown = set(section) - set(ANATOMY_DEFAULTS)
    if unknown:                                    # a typo would otherwise fall back to the default silently
        raise ValueError(f"unknown anatomy settings {sorted(unknown)}; known: {sorted(ANATOMY_DEFAULTS)}")
    out = {k: section.get(k, v) for k, v in ANATOMY_DEFAULTS.items()}
    groups = {str(name): [str(c) for c in classes] for name, classes in dict(out["part_groups"]).items()}
    seen = {}
    for name, classes in groups.items():
        if not classes:
            raise ValueError(f"part group {name!r} has no classes")
        for c in classes:
            if c in seen:
                raise ValueError(f"class {c!r} is in part groups {seen[c]!r} and {name!r}")
            seen[c] = name
    if NON_EYE_GROUP not in groups or len(groups) < 2:
        raise ValueError(f"part_groups needs a {NON_EYE_GROUP!r} group and at least one eye group, got {list(groups)}")
    out["part_groups"] = groups
    for key in ("hidden",):
        out[key] = int(out[key])
    for key in ("mask_loss_weight", "eps", "min_part_area", "gate_inside", "gate_outside"):
        out[key] = float(out[key])
    if out["eps"] <= 0:
        raise ValueError("anatomy.eps must be > 0")
    return out


def eye_groups(settings):
    """Names of the groups that form the eye region (every group except skin), in order."""
    return [g for g in settings["part_groups"] if g != NON_EYE_GROUP]


def eye_classes(settings):
    """Mask classes of the eye region: the classes of every group except skin."""
    return [c for g in eye_groups(settings) for c in settings["part_groups"][g]]


def region_classes(model):
    """Mask classes an attnet_gate model's region input is made of (saved in its gate layer)."""
    names = model.get_layer("region_gate").eye_classes
    return list(names) if names is not None else eye_classes(anatomy_settings({}))


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="anatomy")
class AttentionPool(tf.keras.layers.Layer):
    """K attention maps over the cells of a feature map, and the features they pool.

    a_k = softmax over the H x W cells of a 2-layer 1 x 1 MLP (hidden units, ReLU);
    z_k = sum over cells of a_k * F. With a zero-initialised output bias every map
    starts close to uniform, i.e. close to global average pooling.

    Input F (N, H, W, C). Returns [z (N, K, C), maps (N, H, W, K)].
    """

    def __init__(self, hidden=256, n_maps=1, map_names=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden = int(hidden)
        self.n_maps = int(n_maps)
        self.map_names = list(map_names) if map_names is not None else [f"map{k}" for k in range(self.n_maps)]
        if len(self.map_names) != self.n_maps:
            raise ValueError(f"{len(self.map_names)} map names for {self.n_maps} maps")

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.w1 = self.add_weight(name="w1", shape=(channels, self.hidden), initializer="glorot_uniform")
        self.b1 = self.add_weight(name="b1", shape=(self.hidden,), initializer="zeros")
        self.w2 = self.add_weight(name="w2", shape=(self.hidden, self.n_maps), initializer="glorot_uniform")
        self.b2 = self.add_weight(name="b2", shape=(self.n_maps,), initializer="zeros")
        super().build(input_shape)

    def call(self, features):
        shape = tf.shape(features)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        hidden = tf.nn.relu(tf.einsum("nhwc,cd->nhwd", features, self.w1) + self.b1)
        logits = tf.einsum("nhwd,dk->nhwk", hidden, self.w2) + self.b2
        flat = tf.nn.softmax(tf.reshape(logits, (n, h * w, self.n_maps)), axis=1)
        pooled = tf.einsum("nsk,nsc->nkc", flat, tf.reshape(features, (n, h * w, c)))
        return [pooled, tf.reshape(flat, (n, h, w, self.n_maps))]

    def compute_output_shape(self, input_shape):
        n, h, w, c = input_shape
        return [tf.TensorShape((n, self.n_maps, c)), tf.TensorShape((n, h, w, self.n_maps))]

    def get_config(self):
        config = super().get_config()
        config.update({"hidden": self.hidden, "n_maps": self.n_maps, "map_names": self.map_names})
        return config


@tf.keras.utils.register_keras_serializable(package="anatomy")
class PartEmbedding(tf.keras.layers.Layer):
    """Linear map per part: e_p = W_p z_p + b_p, no nonlinearity (keeps the score additive).

    Input z (N, P, C); weights (P, C, units), bias (P, units); output (N, P, units).
    Each W_p is initialised like a Dense(C -> units) layer (Glorot uniform per part).
    """

    def __init__(self, units, part_names=None, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.part_names = list(part_names) if part_names is not None else None

    def build(self, input_shape):
        parts, channels = int(input_shape[1]), int(input_shape[2])
        if self.part_names is not None and len(self.part_names) != parts:
            raise ValueError(f"{len(self.part_names)} part names for {parts} parts")
        limit = float(np.sqrt(6.0 / (channels + self.units)))
        self.kernel = self.add_weight(name="kernel", shape=(parts, channels, self.units),
                                      initializer=tf.keras.initializers.RandomUniform(-limit, limit))
        self.bias = self.add_weight(name="bias", shape=(parts, self.units), initializer="zeros")
        super().build(input_shape)

    def call(self, z):
        return tf.einsum("npc,pcd->npd", z, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[1], self.units))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "part_names": self.part_names})
        return config


@tf.keras.utils.register_keras_serializable(package="anatomy")
class SumParts(tf.keras.layers.Layer):
    """Sum of the part embeddings: (N, P, D) -> (N, D)."""

    def call(self, parts):
        return tf.reduce_sum(parts, axis=1)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[2]))


@tf.keras.utils.register_keras_serializable(package="anatomy")
class RegionGate(tf.keras.layers.Layer):
    """AttNet's fixed gate: F * (outside + (inside - outside) * region).

    region (N, H, W, 1) is the eye-region fraction of each feature cell, so a cell
    fully inside the eye is multiplied by ``inside`` (5) and one fully outside by
    ``outside`` (1/5). Inputs [F, region]; no weights. ``eye_classes`` records which mask
    classes the region was computed from in training, so evaluation feeds the same.
    """

    def __init__(self, inside=5.0, outside=0.2, eye_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.inside, self.outside = float(inside), float(outside)
        self.eye_classes = list(eye_classes) if eye_classes is not None else None   # what "region" was made of

    def call(self, inputs):
        features, region = inputs
        region = tf.cast(region, features.dtype)
        return features * (self.outside + (self.inside - self.outside) * region)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"inside": self.inside, "outside": self.outside, "eye_classes": self.eye_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="anatomy")
class FixedRegion(tf.keras.layers.Layer):
    """A constant region input (see :func:`bind_region`): returns the stored (B, H, W, 1)
    region, broadcast over the batch when B == 1. Not part of any trained model."""

    def __init__(self, region=None, **kwargs):
        super().__init__(**kwargs)
        self.region = None if region is None else np.asarray(region, np.float32)

    def call(self, image):
        region = tf.constant(self.region)
        if self.region.shape[0] == 1:
            return tf.broadcast_to(region, tf.concat([tf.shape(image)[:1], tf.shape(region)[1:]], axis=0))
        return region

    def get_config(self):
        config = super().get_config()
        config.update({"region": None if self.region is None else self.region.tolist()})
        return config


CUSTOM_OBJECTS = {"ArcFace": ArcFace, "AttentionPool": AttentionPool, "PartEmbedding": PartEmbedding,
                  "SumParts": SumParts, "RegionGate": RegionGate, "FixedRegion": FixedRegion}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _backbone(cfg):
    """(image input, feature tensor) of the backbone, with its layers frozen up to fine_tune_at.

    ``cfg["model"]["backbone"]``: "xception" (default; ImageNet weights unless
    ``backbone_weights`` says otherwise, exactly the vendored create_model's base) or
    "tiny" (2 convolutions for CPU tests; last activation ``block14_sepconv2_act``,
    8 x 8 for a 256 x 256 input, random weights).
    """
    model_cfg = cfg["model"]
    kind = model_cfg.get("backbone", "xception")
    shape = tuple(model_cfg["input_shape"])
    if kind == "xception":
        base = tf.keras.applications.Xception(input_shape=shape, include_top=False,
                                              weights=model_cfg.get("backbone_weights", "imagenet"))
    elif kind == "tiny":
        image = tf.keras.Input(shape, name="image")
        x = tf.keras.layers.Conv2D(8, 3, strides=4, padding="same", name="tiny_conv1")(image)
        x = tf.keras.layers.Activation("relu", name="tiny_conv1_act")(x)
        x = tf.keras.layers.Conv2D(16, 3, strides=8, padding="same", name="tiny_conv2")(x)
        x = tf.keras.layers.Activation("relu", name=FEATURE_LAYER)(x)
        base = tf.keras.Model(image, x, name="tiny_backbone")
    else:
        raise ValueError(f"unknown backbone {kind!r}; use 'xception' or 'tiny'")
    base.trainable = True                        # as create_model: train from fine_tune_at on
    for layer in base.layers[:model_cfg["fine_tune_at"]]:
        layer.trainable = False
    if base.layers[-1].name != FEATURE_LAYER:
        raise ValueError(f"the backbone must end in {FEATURE_LAYER}, got {base.layers[-1].name}")
    return base.input, base.output


def build_anatomy_model(cfg, arch):
    """Compiled inference model of *arch* (see the module docstring), built from *cfg*.

    ``baseline`` is the vendored create_model (recognition_model.build_model). The
    other archs share its input, backbone, Dropout(0.5) and ArcFace head (s, m from
    cfg) and its compile settings (Adam at the cfg learning rate, categorical
    cross-entropy, accuracy). Inputs [image, label], plus ``region`` (N, 8, 8, 1) for
    ``attnet_gate``. The model is named ``anatomy_<arch>`` so :func:`model_arch`
    can tell ``attn`` and ``attn_mask`` apart after a reload.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    if arch == "baseline":
        return build_model(cfg)
    settings = anatomy_settings(cfg)
    n_classes, units = int(cfg["model"]["n_classes"]), int(cfg["model"]["dense_units"])
    image, features = _backbone(cfg)
    label = tf.keras.Input((n_classes,), name="label")
    inputs = [image, label]
    grid = tuple(int(d) for d in features.shape[1:3])
    channels = int(features.shape[-1])

    if arch in ("gap", "attnet_gate"):
        if arch == "attnet_gate":
            region = tf.keras.Input(grid + (1,), name="region")
            inputs.append(region)
            features = RegionGate(settings["gate_inside"], settings["gate_outside"], eye_classes=eye_classes(settings),
                                  name="region_gate")([features, region])
        x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(features)
        x = tf.keras.layers.Dense(units, activation="relu", name="head_dense")(x)
    elif arch in ("attn", "attn_mask"):
        pooled, _ = AttentionPool(settings["hidden"], 1, map_names=["eye"], name="attention_pool")(features)
        x = tf.keras.layers.Reshape((channels,), name="attention_squeeze")(pooled)
        x = tf.keras.layers.Dense(units, activation="relu", name="head_dense")(x)
    else:                                          # parts
        names = list(settings["part_groups"])
        pooled, _ = AttentionPool(settings["hidden"], len(names), map_names=names, name="attention_pool")(features)
        x = PartEmbedding(units, part_names=names, name="part_embedding")(pooled)
        x = SumParts(name="part_sum")(x)
    x = tf.keras.layers.Dropout(0.5, name="head_dropout")(x)
    output = ArcFace(n_classes=n_classes, s=cfg["arcface"]["scale"], m=cfg["arcface"]["margin"],
                     name="arcface")([x, label])
    model = tf.keras.Model(inputs=inputs, outputs=output, name=f"anatomy_{arch}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    return model


def model_arch(model):
    """Arch name of a model built here (from its name, else its layers); ``baseline`` otherwise."""
    name = getattr(model, "name", "")
    if name.startswith("anatomy_") and name[len("anatomy_"):] in ARCHS:
        return name[len("anatomy_"):]
    names = {layer.name for layer in model.layers}
    for layer_name, arch in (("region_gate", "attnet_gate"), ("part_embedding", "parts"),
                             ("attention_pool", "attn"), ("head_gap", "gap")):
        if layer_name in names:
            return arch
    return "baseline"


def part_names(model):
    """Names of the part maps of a ``parts`` model (from the saved layer config), in order."""
    if "part_embedding" not in {layer.name for layer in model.layers}:
        raise ValueError(f"model {model.name!r} ({model_arch(model)}) has no part embedding")
    names = model.get_layer("part_embedding").part_names
    return list(names) if names is not None else [f"part{k}" for k in range(attention_n_maps(model))]


def attention_n_maps(model):
    return int(model.get_layer("attention_pool").n_maps)


def feature_grid(model):
    """(H, W) of the CAM feature map (8 x 8 for Xception at 256 x 256)."""
    return tuple(int(d) for d in model.get_layer(FEATURE_LAYER).output.shape[1:3])


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_SUB_MODELS = weakref.WeakKeyDictionary()


def _sub_model(model, key, build):
    per_model = _SUB_MODELS.setdefault(model, {})
    if key not in per_model:
        per_model[key] = build()
    return per_model[key]


def _find_arcface(model):
    for layer in model.layers:
        if type(layer).__name__ == "ArcFace":
            return layer
    raise ValueError("model has no ArcFace layer")


def _batches(images, batch_size):
    x = np.asarray(images, np.float32)
    if x.ndim == 3:
        x = x[np.newaxis]
    if x.ndim != 4:
        raise ValueError(f"expected (N, H, W, 3) images, got shape {x.shape}")
    return x, range(0, len(x), max(1, int(batch_size)))


def attention_maps(model, images, batch_size=32):
    """Attention maps (N, H, W, K) float32 (each map sums to 1 over its cells).

    K = 1 for attn / attn_mask, P for parts. Raises for a model without ``attention_pool``.
    """
    if "attention_pool" not in {layer.name for layer in model.layers}:
        raise ValueError(f"model {model.name!r} ({model_arch(model)}) has no attention_pool layer")
    sub = _sub_model(model, "attention", lambda: tf.keras.Model(
        model.inputs[0], model.get_layer("attention_pool").output[1]))
    x, starts = _batches(images, batch_size)
    parts = [np.asarray(sub(x[i:i + batch_size], training=False), np.float32) for i in starts]
    k = attention_n_maps(model)
    return np.concatenate(parts) if parts else np.zeros((0,) + feature_grid(model) + (k,), np.float32)


def part_receipt(model, images, class_ids, batch_size=32, units="cosine"):
    """Per-part terms of the margin-free cosine of class ``class_ids[n]`` for image n: (N, P) float32.

    For the parts head e = sum_p e_p, and the margin-free cosine of class c is
    e . normalize(W_c) / ||e|| = sum_p e_p . normalize(W_c) / ||e||, so the terms sum
    exactly (to float32 rounding) to the cosine. ``units="logit"`` multiplies them by
    ArcFace's s, so they sum to the margin-free logit of predict_logits (the units of the
    ASA scores in src/xai/shapley.py). The sum is computed in float64 from the layer
    outputs, with ArcFace's l2_normalize floor.
    """
    if model_arch(model) != "parts":
        raise ValueError(f"part_receipt needs a parts model, got {model_arch(model)}")
    if units not in ("cosine", "logit"):
        raise ValueError(f"units must be 'cosine' or 'logit', got {units!r}")
    sub = _sub_model(model, "parts", lambda: tf.keras.Model(model.inputs[0], model.get_layer("part_embedding").output))
    arcface = _find_arcface(model)
    weights = np.asarray(arcface.W, np.float64)
    weights = weights / np.sqrt(np.maximum((weights ** 2).sum(axis=0, keepdims=True), 1e-12))
    x, starts = _batches(images, batch_size)
    class_ids = np.asarray(class_ids, np.int64).reshape(-1)
    if len(class_ids) != len(x):
        raise ValueError(f"{len(class_ids)} class ids for {len(x)} images")
    out = []
    for i in starts:
        e_parts = np.asarray(sub(x[i:i + batch_size], training=False), np.float64)       # (n, P, D)
        e = e_parts.sum(axis=1)
        norm = np.sqrt(np.maximum((e ** 2).sum(axis=1), 1e-12))
        w = weights[:, class_ids[i:i + batch_size]].T                                     # (n, D)
        out.append(np.einsum("npd,nd->np", e_parts, w) / norm[:, None])
    receipt = np.concatenate(out) if out else np.zeros((0, attention_n_maps(model)))
    if units == "logit":
        receipt = receipt * float(arcface.s)
    return receipt.astype(np.float32)


def arcface_scale(model):
    """ArcFace's scale s of a model (margin-free logit = s x cosine)."""
    return float(_find_arcface(model).s)


def embed_regions(model, images, regions, batch_size=64):
    """L2-normalised ArcFace-input embeddings of an ``attnet_gate`` model, fed its region input.

    The three-input counterpart of src.eval.verification.embed: images (N, 256, 256, 3)
    and regions (N, 8, 8, 1) (eye-region fraction per cell).
    """
    from src.eval.verification import l2_normalise
    if len(model.inputs) != 3:
        raise ValueError("embed_regions is for the three-input attnet_gate model; use verification.embed")
    sub = _sub_model(model, "embed_regions", lambda: tf.keras.Model(
        [model.inputs[0], model.inputs[2]], _find_arcface(model).input[0]))
    x, starts = _batches(images, batch_size)
    r = np.asarray(regions, np.float32).reshape((len(x),) + tuple(model.inputs[2].shape[1:]))
    parts = [np.asarray(sub([x[i:i + batch_size], r[i:i + batch_size]], training=False)) for i in starts]
    if not parts:
        return np.zeros((0, int(sub.output.shape[-1])), np.float32)
    return l2_normalise(np.concatenate(parts))


def bind_region(model, region):
    """Dual-input [image, label] copy of an ``attnet_gate`` model with its region input fixed.

    region: (8, 8, 1) or (1, 8, 8, 1) (broadcast to any batch, e.g. one image for
    LayerCAM) or (B, 8, 8, 1) (only batches of exactly these B images). The copy is
    rebuilt from the layer configs and gets the trained weights by layer name, so
    get_margin_free, predict_logits, LayerCAM and verification.embed work on it. It is
    a fresh model (a few seconds for Xception): bind once per set of images.
    """
    if model_arch(model) != "attnet_gate":
        raise ValueError(f"bind_region needs an attnet_gate model, got {model_arch(model)}")
    region = np.asarray(region, np.float32)
    if region.ndim == 3:
        region = region[np.newaxis]
    gate = model.get_layer("region_gate")
    arcface = _find_arcface(model)
    names = {layer.name for layer in model.layers}
    cfg = {"model": {"backbone": "tiny" if "tiny_conv1" in names else "xception", "backbone_weights": None,
                     "input_shape": list(model.inputs[0].shape[1:]), "n_classes": int(arcface.n_classes),
                     "dense_units": int(model.get_layer("head_dense").units), "fine_tune_at": 0}}
    image, features = _backbone(cfg)
    label = tf.keras.Input((cfg["model"]["n_classes"],), name="label")
    fixed = FixedRegion(region, name="fixed_region")(image)
    x = RegionGate(gate.inside, gate.outside, eye_classes=gate.eye_classes, name="region_gate")([features, fixed])
    x = tf.keras.layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = tf.keras.layers.Dense(cfg["model"]["dense_units"], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.5, name="head_dropout")(x)
    output = ArcFace(n_classes=arcface.n_classes, s=arcface.s, m=arcface.m, name="arcface")([x, label])
    bound = tf.keras.Model([image, label], output, name="anatomy_attnet_gate_bound")
    for layer in bound.layers:
        if layer.weights:
            source = arcface if type(layer).__name__ == "ArcFace" else model.get_layer(layer.name)
            layer.set_weights(source.get_weights())
    return bound


# ---------------------------------------------------------------------------
# Mask loss and trainer
# ---------------------------------------------------------------------------

def mask_kl(targets, maps, weights, floor=1e-12):
    """Weighted mean over maps of KL(t || a) = sum over cells of t * log(t / a).

    targets, maps : (N, H, W, K), each map summing to 1 over its cells
    weights       : (N, K); maps with weight 0 do not count; 0 if no map counts
    """
    t = tf.cast(targets, tf.float32)
    a = tf.cast(maps, tf.float32)
    kl = tf.reduce_sum(t * (tf.math.log(t + floor) - tf.math.log(a + floor)), axis=[1, 2])   # (N, K)
    w = tf.cast(weights, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(w * kl), tf.reduce_sum(w))


class MaskTrainer(tf.keras.Model):
    """Trains an ``attn_mask`` / ``parts`` inference model with CE + lambda * mask KL.

    The data are ((image, label, targets, weights), label) from
    src.data.masked_generator. ``train_step`` runs in graph mode, so the
    cross-entropy is the compiled CategoricalCrossentropy on the ArcFace output,
    which reads the logits of its Softmax op exactly as model.fit does for the
    baseline (never eager cross-entropy on probabilities). Logged: ``loss`` (the
    total objective, which ModelCheckpoint / EarlyStopping monitor as val_loss),
    ``ce``, ``mask_loss`` (unweighted KL) and ``accuracy``.

    Everything that saves or restores goes to the inference model: ``save`` (so
    ModelCheckpoint writes the plain dual-input model that load_checkpoint reads),
    ``save_weights``, ``get_weights`` / ``set_weights`` (so EarlyStopping's best
    weights and 03's ResumeState, which reads them from the checkpoint, line up).
    The trainer shares the inference model's optimizer, so the checkpoint carries the
    optimizer state as the baseline's does; BackupAndRestore backs up the trainer.
    """

    def __init__(self, net, mask_loss_weight=1.0, **kwargs):
        super().__init__(name=f"{net.name}_trainer", **kwargs)
        self.net = net
        maps = net.get_layer("attention_pool").output[1]
        self.maps_model = tf.keras.Model(net.inputs[:2], [net.output, maps], name=f"{net.name}_with_maps")
        self.mask_loss_weight = float(mask_loss_weight)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_tracker = tf.keras.metrics.Mean(name="ce")
        self.mask_tracker = tf.keras.metrics.Mean(name="mask_loss")

    def call(self, inputs, training=None):
        return self.net(list(inputs)[:len(self.net.inputs)], training=training)

    @property
    def metrics(self):
        compiled = self.compiled_metrics.metrics if self.compiled_metrics is not None else []
        return [self.loss_tracker, self.ce_tracker, self.mask_tracker] + list(compiled)

    def _objective(self, data, training):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        image, label, targets, weights = x
        probs, maps = self.maps_model([image, label], training=training)
        ce = self.compiled_loss(y, probs, regularization_losses=self.net.losses)
        mask = mask_kl(targets, maps, weights)
        return y, probs, ce, mask, ce + self.mask_loss_weight * mask

    def _track(self, y, probs, ce, mask, total):
        self.loss_tracker.update_state(total)
        self.ce_tracker.update_state(ce)
        self.mask_tracker.update_state(mask)
        self.compiled_metrics.update_state(y, probs)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        with tf.GradientTape() as tape:
            y, probs, ce, mask, total = self._objective(data, training=True)
        self.optimizer.minimize(total, self.net.trainable_variables, tape=tape)
        return self._track(y, probs, ce, mask, total)

    def test_step(self, data):
        return self._track(*self._objective(data, training=False))

    def save(self, filepath, *args, **kwargs):
        return self.net.save(filepath, *args, **kwargs)

    def save_weights(self, filepath, *args, **kwargs):
        return self.net.save_weights(filepath, *args, **kwargs)

    def get_weights(self):
        return self.net.get_weights()

    def set_weights(self, weights):
        return self.net.set_weights(weights)

    def count_params(self):
        return self.net.count_params()


def make_trainer(net, cfg, mask_loss_weight=None):
    """Compiled :class:`MaskTrainer` for an attn_mask / parts model from :func:`build_anatomy_model`.

    Compiled like the model (same loss and metrics) with the model's own optimizer.
    *mask_loss_weight* overrides ``cfg["anatomy"]["mask_loss_weight"]``.
    """
    arch = model_arch(net)
    if arch not in MASK_LOSS_ARCHS:
        raise ValueError(f"the mask trainer is for {MASK_LOSS_ARCHS}, got {arch}")
    weight = anatomy_settings(cfg)["mask_loss_weight"] if mask_loss_weight is None else float(mask_loss_weight)
    trainer = MaskTrainer(net, weight)
    trainer.compile(optimizer=net.optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    return trainer
