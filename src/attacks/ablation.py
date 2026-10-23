"""
src/attacks/ablation.py

One configurable iterative attack that covers both the vendored ABRobOcular
BIM / MIM / FGSM and their standard formulations, so the weakness of the
vendored attacks can be attributed to each flaw separately.

Flaws of the vendored ``_face`` attacks (third_party/ABRobOcular/AB_Rob_Ocular/
AB_Rob_Ocular_Tensor.py) that this module can switch on and off:

    loss_space="double_softmax"  The model already ends in a softmax (ArcFace layer);
                                 the vendored loss applies ``tf.nn.softmax`` again and
                                 takes cross-entropy of that.
    step_size = 5 * epsilon      BIM / MIM step ``iter_eps = 0.005`` for ``epsilon = 0.001``.
    projection="end"             The iterate is clipped onto the epsilon ball only once,
                                 after the last step, instead of after every step.
    momentum_norm=False          MIM accumulates raw gradients instead of L1-normalised ones.

``VENDORED`` and ``STANDARD`` hold the two ends; ``ablation_configs`` lists the
one-flaw-at-a-time variants. With ``VENDORED`` settings the result equals the
vendored code (tested in tests/test_ablation.py).
"""

import numpy as np
import tensorflow as tf

from src.models.arcface_inference import as_batch, get_margin_free, get_n_classes

LOSS_SPACES = ("double_softmax", "probabilities", "logits_ce", "logits_margin")


def _loss(model, net, x, labels, loss_space):
    if loss_space in ("double_softmax", "probabilities"):
        zeros = tf.zeros([tf.shape(x)[0], get_n_classes(model)], tf.float32)
        probs = model([x, zeros], training=False)
        if loss_space == "double_softmax":
            probs = tf.nn.softmax(probs)
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
    logits = net(x)
    if loss_space == "logits_ce":
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    one_hot = tf.one_hot(labels, tf.shape(logits)[1])
    true_logit = tf.reduce_sum(one_hot * logits, axis=1)
    best_other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
    return tf.reduce_sum(best_other - true_logit)


def iterative_attack(image, model, epsilon, steps=10, step_size=None, projection="each",
                     loss_space="logits_ce", decay=None, momentum_norm=True, label=None):
    """Configurable BIM (``decay=None``) or MIM (``decay`` set); ``steps=1, step_size=epsilon`` is FGSM.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget
    steps : int
    step_size : float or None — defaults to epsilon / steps
    projection : {"each", "end"} — clip onto the epsilon ball after every step, or once at the end
    loss_space : one of :data:`LOSS_SPACES`
    decay : float or None — momentum decay (MIM); None for BIM
    momentum_norm : bool — L1-normalise gradients before accumulating (standard MIM)
    label : int, sequence, or None — target label; None uses the clean prediction

    Returns
    -------
    adversarial : ndarray, same shape as *image*
    """
    if loss_space not in LOSS_SPACES:
        raise ValueError(f"loss_space must be one of {LOSS_SPACES}, got {loss_space!r}")
    if projection not in ("each", "end"):
        raise ValueError(f"projection must be 'each' or 'end', got {projection!r}")
    x0, single = as_batch(image)
    net = get_margin_free(model)
    if label is None:
        labels = tf.argmax(net(x0), axis=1, output_type=tf.int32)
    else:
        labels = tf.broadcast_to(tf.reshape(tf.constant(label, tf.int32), [-1]), [tf.shape(x0)[0]])
    alpha = epsilon / steps if step_size is None else step_size
    x = tf.identity(x0)
    momentum = tf.zeros_like(x0)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = _loss(model, net, x, labels, loss_space)
        gradient = tape.gradient(value, x)
        if decay is not None:
            if momentum_norm:
                gradient = gradient / tf.maximum(
                    tf.reduce_mean(tf.abs(gradient), axis=[1, 2, 3], keepdims=True), 1e-12)
            momentum = decay * momentum + gradient
            gradient = momentum
        x = tf.clip_by_value(x + alpha * tf.sign(gradient), 0.0, 1.0)
        if projection == "each":
            x = tf.clip_by_value(x, x0 - epsilon, x0 + epsilon)
    if projection == "end":
        x = tf.clip_by_value(x, x0 - epsilon, x0 + epsilon)
    out = np.clip(x.numpy(), 0.0, 1.0).astype(np.float32)
    return out[0] if single else out


def ablation_configs(epsilon):
    """Named configurations, from the vendored attack to the standard one, one fix at a time.

    Returns
    -------
    dict name -> (attack family, kwargs for :func:`iterative_attack`)
    """
    vendored_step = 5.0 * epsilon            # iter_eps = 0.005 for epsilon = 0.001
    out = {
        "fgsm/vendored": ("fgsm", dict(steps=1, step_size=epsilon, loss_space="double_softmax")),
        "fgsm/single_softmax": ("fgsm", dict(steps=1, step_size=epsilon, loss_space="probabilities")),
        "fgsm/logits_ce": ("fgsm", dict(steps=1, step_size=epsilon, loss_space="logits_ce")),
        "fgsm/logits_margin": ("fgsm", dict(steps=1, step_size=epsilon, loss_space="logits_margin")),
    }
    for family, extra in (("bim", {}), ("mim", {"decay": 1.0})):
        vendored = dict(steps=10, step_size=vendored_step, projection="end", loss_space="double_softmax",
                        momentum_norm=False, **extra)
        out[f"{family}/vendored"] = (family, vendored)
        out[f"{family}/fix_projection"] = (family, {**vendored, "projection": "each"})
        out[f"{family}/fix_step"] = (family, {**vendored, "step_size": None})
        out[f"{family}/fix_projection+step"] = (family, {**vendored, "projection": "each", "step_size": None})
        out[f"{family}/+single_softmax"] = (family, {**vendored, "projection": "each", "step_size": None,
                                                     "loss_space": "probabilities"})
        out[f"{family}/+logits_ce"] = (family, {**vendored, "projection": "each", "step_size": None,
                                                "loss_space": "logits_ce", "momentum_norm": True})
        out[f"{family}/standard_margin"] = (family, {**vendored, "projection": "each", "step_size": None,
                                                     "loss_space": "logits_margin", "momentum_norm": True})
    return out
