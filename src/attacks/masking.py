"""
src/attacks/masking.py

Gradient-masking check for the attention models: an attack that rescales the
attention softmax, and the gradient and attention statistics behind the check.

Why this module exists
----------------------
"Towards Understanding and Improving Adversarial Robustness of Vision
Transformers" (IEEE 10658195) shows that the softmax inside attention can
saturate, so its gradient underflows and PGD or CW look weaker than they are
(gradient masking, a false robustness result). Their Adaptive Attention Scaling
(AAS) attack rescales the pre-softmax values while attacking. The attention
heads of src/models/anatomy_model.py (``attn``, ``attn_mask``, ``parts``) have
one such softmax, over the 8 x 8 feature cells of AttentionPool. (The ArcFace
softmax is already out of the way: every attack uses the margin-free logits.)
A robustness difference between an attention model and the baseline only counts
once the check of scripts/24_gradient_masking.py passes.

with_attention_temperature(model, tau)
    A model on the same weights whose attention logits are divided by tau (the
    AttentionPool temperature). tau > 1 flattens a saturated softmax.
aas_pgd
    PGD on the margin loss in the style of AAS. At every step each model of a
    small temperature grid (default 1, 2, 4, 8, 16) proposes the PGD step along
    its own gradient, and each image takes the proposal after which the attack
    loss of the unscaled model (tau = 1) is largest. Everything else is
    ``whitebox.pgd`` (random start, step size, projection, last iterate), so the
    two differ only in where the gradient comes from; without an attention pool,
    or with the grid (1,), aas_pgd returns exactly pgd's image. The image is
    judged like every other attack: on the unscaled model, after the caller
    rounds it to 8 bits within whole levels of the budget.

    Deviation from the paper, on purpose: AAS adapts the scaling of the
    pre-softmax values of the attention layers of a ViT during the attack; here a
    single temperature on our one attention pool is chosen at every step from a
    fixed grid. We chose the grid over a gradient step on log tau because it
    needs no second step size, is deterministic and keeps tau = 1 (PGD's own
    step) among the candidates. The candidates are compared by the unscaled
    model's loss after the step, not by each scaled model's own loss: a large tau
    changes the model, so its own loss says how fooled the changed model is, not
    the real one (on a tiny attention model, picking by the scaled loss chose
    tau = 16 most of the time and gave a weaker attack than PGD). The price is
    one forward and one backward pass per grid value and step, plus one forward
    pass of the unscaled model per proposal (about 6-7 PGD steps per step).
gradient_stats, attention_stats
    Share of exactly zero and of non-finite input-gradient components, and the
    gradient norms, per image; the largest attention weight and the normalised
    entropy (entropy / log(cells), 1 = uniform, 0 = one cell) of every map.

Image contract
--------------
- Input  : float32, shape (256, 256, 3) or (N, 256, 256, 3), range [0, 1]
- Output : float32, same shape as the input, range [0, 1] (float: round it with
           src.attacks.budget.quantise like the output of whitebox.pgd)
"""

import os
import weakref

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.attacks.whitebox import _labels, _loss_gradient, _output, _project
from src.models.anatomy_model import attention_maps
from src.models.arcface_inference import as_batch, find_arcface_layer, get_margin_free
from src.models.recognition_model import ArcFace

DEFAULT_TAUS = (1.0, 2.0, 4.0, 8.0, 16.0)
POOL_LAYER = "attention_pool"

_SCALED = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Temperature-scaled models
# ---------------------------------------------------------------------------

def has_attention(model):
    """True if *model* has an AttentionPool named ``attention_pool`` (attn, attn_mask, parts)."""
    return POOL_LAYER in {layer.name for layer in model.layers}


def check_taus(taus):
    """The temperature grid as a tuple of floats: positive, finite and distinct."""
    taus = tuple(float(t) for t in np.atleast_1d(np.asarray(taus, dtype=np.float64)))
    if not taus or not all(np.isfinite(t) and t > 0 for t in taus):
        raise ValueError(f"temperatures must be positive numbers, got {taus}")
    if len(set(taus)) != len(taus):
        raise ValueError(f"temperatures must be distinct, got {taus}")
    return taus


def _build_scaled(model, tau):
    pool = model.get_layer(POOL_LAYER)
    arcface = find_arcface_layer(model)
    # the parts before and after the pool, as sub-models of the trained graph (they share its layers)
    before = tf.keras.Model(model.inputs[0], pool.input, name="before_attention")
    after = tf.keras.Model(pool.output[0], arcface.input[0], name="after_attention")
    scaled_pool = pool.shared_copy(pool.temperature * tau)
    # a fresh ArcFace on the same W, so that get_margin_free finds its input in this graph
    head = ArcFace(n_classes=arcface.n_classes, s=arcface.s, m=arcface.m, regularizer=arcface.regularizer,
                   name=arcface.name)
    head.W = arcface.W
    head.built = True
    image = tf.keras.Input(tuple(model.inputs[0].shape[1:]), name="image")
    label = tf.keras.Input(tuple(model.inputs[1].shape[1:]), name="label")
    pooled, _ = scaled_pool(before(image))
    output = head([after(pooled), label])
    return tf.keras.Model([image, label], output, name=model.name)


def with_attention_temperature(model, tau):
    """Model on the same weights as *model* whose attention logits are divided by *tau*.

    Parameters
    ----------
    model : tf.keras.Model with dual inputs [image, label] and an ``attention_pool`` layer
    tau : float > 0 — multiplies the pool's temperature (1 in every trained model)

    Returns
    -------
    A dual-input model with the same name (so ``model_arch`` still works) that shares
    every variable with *model*: nothing is copied, and a weight update of one shows in
    the other. ``predict_logits``, ``get_margin_free`` and ``attention_maps`` work on it;
    tau = 1 gives exactly *model*'s outputs. The layers before and after the pool are
    nested sub-models, so LayerCAM, ``part_receipt`` and layer lookups by name are for
    the original model. Built once per (model, tau) and cached.
    """
    if not has_attention(model):
        raise ValueError(f"model {model.name!r} has no {POOL_LAYER!r} layer to scale")
    if len(model.inputs) != 2:
        raise ValueError("with_attention_temperature needs a dual-input [image, label] model")
    tau = check_taus([tau])[0]
    per_model = _SCALED.setdefault(model, {})
    if tau not in per_model:
        per_model[tau] = _build_scaled(model, tau)
    return per_model[tau]


# ---------------------------------------------------------------------------
# AAS-style PGD
# ---------------------------------------------------------------------------

def _per_image_loss(logits, labels, loss):
    """Per-image attack loss (the summand of whitebox._loss_gradient): margin or cross-entropy."""
    if loss == "ce":
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    one_hot = tf.one_hot(labels, tf.shape(logits)[1])
    true_logit = tf.reduce_sum(one_hot * logits, axis=1)
    best_other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
    return best_other - true_logit


def _scaled_nets(model, taus):
    """[(tau, margin-free net)] for the grid; [(1, net)] for a model without attention."""
    if not has_attention(model):
        return [(1.0, get_margin_free(model))]
    return [(tau, get_margin_free(model if tau == 1.0 else with_attention_temperature(model, tau)))
            for tau in taus]


def aas_step(nets, net, x, x0, labels, alpha, epsilon, loss="margin"):
    """One projected sign step per image, along the gradient of the best scaled model.

    nets : [(tau, margin-free net)], e.g. from the grid of :func:`aas_pgd`
    net  : the unscaled margin-free net (tau = 1), which judges the candidates
    Every scaled model proposes the PGD step along its own gradient; each image takes
    the proposal after which the unscaled model's attack loss is largest (ties go to
    the first entry). With one entry this is exactly the step of ``whitebox.pgd``.

    Returns (next iterate tensor like x, chosen tau per image (N,) ndarray).
    """
    def propose(candidate_net):
        gradient = _loss_gradient(candidate_net, x, labels, loss)
        return _project(x + alpha * tf.sign(gradient), x0, epsilon)

    if len(nets) == 1:
        return propose(nets[0][1]), np.full(int(x.shape[0]), nets[0][0])
    proposals = [propose(candidate_net) for _, candidate_net in nets]
    losses = np.stack([_per_image_loss(net(p), labels, loss).numpy() for p in proposals])    # (G, N)
    choice = np.argmax(losses, axis=0)                                                      # first maximum
    stacked = tf.stack(proposals, axis=1)                                                   # (N, G, H, W, 3)
    index = np.stack([np.arange(len(choice)), choice], axis=1).astype(np.int32)
    return tf.gather_nd(stacked, index), np.asarray([nets[g][0] for g in choice], np.float64)


def aas_pgd(image, model, epsilon, label=None, steps=10, step_size=None, random_start=True, seed=0,
            taus=DEFAULT_TAUS, loss="margin", return_info=False):
    """PGD whose step follows the temperature-scaled model that helps the unscaled one most.

    In the style of Adaptive Attention Scaling (IEEE 10658195; see the module docstring
    for how it deviates). Same random start, step size, projection and last iterate as
    ``whitebox.pgd`` with the same seed; the true labels are those of the unscaled model.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget on the [0, 1] scale
    label : int, sequence of int, or None — true class; None uses the clean prediction
    steps : int
    step_size : float or None — defaults to 2.5 * epsilon / steps
    random_start : bool
    seed : int — seed for the random start (as in whitebox.pgd)
    taus : sequence of float — attention temperatures tried at every step, in order of
        preference for ties (ignored, i.e. plain PGD, for a model without an attention pool)
    loss : {"margin", "ce"} — attack loss; defaults to "margin"
    return_info : bool — also return {"taus": (steps, N) chosen temperature per step and
        image (N = 1 for a single image)}

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    if loss not in ("ce", "margin"):
        raise ValueError(f"loss must be 'ce' or 'margin', got {loss!r}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if not epsilon > 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    nets = _scaled_nets(model, check_taus(taus))
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    alpha = 2.5 * epsilon / steps if step_size is None else step_size
    x = tf.identity(x0)
    if random_start:            # the start of whitebox.pgd for the same seed
        noise = tf.random.stateless_uniform(tf.shape(x0), seed=[seed, 0], minval=-epsilon, maxval=epsilon)
        x = _project(x0 + noise, x0, epsilon)
    chosen = []
    for _ in range(steps):
        x, tau = aas_step(nets, net, x, x0, labels, alpha, epsilon, loss)
        chosen.append(tau)
    out = _output(x, single)
    if not return_info:
        return out
    return out, {"taus": np.stack(chosen)}


# ---------------------------------------------------------------------------
# Statistics for the check
# ---------------------------------------------------------------------------

def gradient_stats(model, images, labels, loss="margin", taus=None):
    """Input-gradient statistics of the attack loss, per image.

    Returns a dict of (N,) arrays: ``zero_share`` (components exactly 0),
    ``nonfinite_share`` (NaN or inf), ``l2``, ``linf`` and ``l1`` (norms of the finite
    components). With *taus* (attention models only), also ``l2_tau<t>``: the gradient
    norm of the model scaled by t, which grows with t when the softmax saturates.
    """
    x, _ = as_batch(images)
    labels = tf.constant(np.asarray(labels, np.int32).reshape(-1))
    out = _stats(_loss_gradient(get_margin_free(model), x, labels, loss).numpy())
    if taus is not None and has_attention(model):
        for tau in check_taus(taus):
            target = model if tau == 1.0 else with_attention_temperature(model, tau)
            g = _loss_gradient(get_margin_free(target), x, labels, loss).numpy()
            out[f"l2_tau{tau:g}"] = _stats(g)["l2"]
    return out


def _stats(gradient):
    g = np.asarray(gradient, np.float64).reshape(len(gradient), -1)
    finite = np.isfinite(g)
    clean = np.where(finite, g, 0.0)
    return {"zero_share": (g == 0).mean(axis=1), "nonfinite_share": (~finite).mean(axis=1),
            "l2": np.sqrt((clean ** 2).sum(axis=1)), "linf": np.abs(clean).max(axis=1),
            "l1": np.abs(clean).sum(axis=1)}


def map_saturation(maps):
    """Largest weight and normalised entropy of attention maps (N, H, W, K), each map summing to 1.

    Returns (max_weight (N, K), entropy_norm (N, K)); entropy_norm = -sum a log a / log(H W).
    """
    a = np.asarray(maps, np.float64)
    n, h, w, k = a.shape
    flat = a.reshape(n, h * w, k)
    entropy = -(np.where(flat > 0, flat * np.log(np.where(flat > 0, flat, 1.0)), 0.0)).sum(axis=1)
    return flat.max(axis=1), entropy / np.log(h * w)


def attention_stats(model, images, batch_size=32):
    """:func:`map_saturation` of the attention maps of *model* on *images*; None without attention."""
    if not has_attention(model):
        return None
    return map_saturation(attention_maps(model, images, batch_size=batch_size))

