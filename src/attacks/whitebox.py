"""
src/attacks/whitebox.py

White-box attacks on the Xception + ArcFace dual-input model.

Image contract
--------------
- Input  : float32, shape (256, 256, 3) or (N, 256, 256, 3), range [0, 1]
- Output : float32, same shape as the input, range [0, 1]

Two families
------------
fgsm, bim, pgd, mim, cw
    Standard formulations on the margin-free ArcFace logits (margin loss by
    default; cross-entropy remains selectable, projection onto the epsilon ball
    after every step). Use these for
    the XAI study. ``epsilon`` is an L-inf budget on the [0, 1] pixel scale and
    has no default on purpose: pick it explicitly (e.g. 1/255, 2/255, 8/255).
    Cross-entropy gradients underflow in float32 once the top-1 / top-2 logit
    gap passes ~87; the trained model's gaps are ~30-40, but an image the
    sign-gradient attacks cannot move may be saturated rather than robust.
    CW's margin loss does not saturate.
paper_fgsm, paper_bim, paper_mim, paper_pgd, paper_cw
    Thin wrappers around the vendored ABRobOcular ``_face`` functions with the
    paper's settings (epsilon = 0.001, Section 6.1). Use them only to reproduce
    the paper's ASR table. The vendored code has bugs that change what the
    attacks are:
      * every ``_face`` attack applies softmax a second time to the model output
        (the ArcFace layer already returns probabilities);
      * bim/mim step by ``iter_eps`` (0.005 > epsilon) and project onto the
        epsilon ball only once, after the last step;
      * pgd never uses ``epsilon`` (no projection) and has no random start;
      * cw is not Carlini-Wagner: it has no distance term, re-creates Adam every
        step (so each step is lr * sign(grad)), and adds N(0, 0.1) noise between
        restarts, so its perturbation is neither minimal nor deterministic.

``third_party/`` stays unmodified: the vendored module is loaded straight from
its file, which also avoids the package ``__init__`` that imports PyTorch.
"""

import importlib.util
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.arcface_inference import as_batch, get_margin_free, get_n_classes

_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ABRobOcular")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _labels(net, x, label):
    """Attack target: the given label(s), or the model's clean prediction."""
    if label is None:
        return tf.argmax(net(x), axis=1, output_type=tf.int32)
    labels = tf.reshape(tf.constant(label, dtype=tf.int32), [-1])
    return tf.broadcast_to(labels, [tf.shape(x)[0]])


def _loss_gradient(net, x, labels, loss="margin"):
    """Gradient of the attack loss w.r.t. the images.

    loss="ce"     : summed cross-entropy from logits (textbook). With ArcFace scale s = 64 the
                    softmax of a confident prediction rounds p_y to exactly 1.0 in float32, so the
                    true-class term of the gradient (p_y - 1) is exactly zero and only the
                    runner-up terms remain.
    loss="margin" : (best other logit) - (true logit). Never saturates.
    """
    if loss not in ("ce", "margin"):
        raise ValueError(f"loss must be 'ce' or 'margin', got {loss!r}")
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = net(x)
        if loss == "ce":
            value = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        else:
            one_hot = tf.one_hot(labels, tf.shape(logits)[1])
            true_logit = tf.reduce_sum(one_hot * logits, axis=1)
            best_other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
            value = tf.reduce_sum(best_other - true_logit)
    return tape.gradient(value, x)


def _project(x_adv, x, epsilon):
    """Clip onto the L-inf ball of radius *epsilon* around *x*, then onto [0, 1]."""
    return tf.clip_by_value(tf.clip_by_value(x_adv, x - epsilon, x + epsilon), 0.0, 1.0)


def _output(x, single):
    out = np.clip(x.numpy(), 0.0, 1.0).astype(np.float32)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Standard attacks — each returns the same shape as its input
# ---------------------------------------------------------------------------

def fgsm(image, model, epsilon, label=None, loss="margin"):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget on the [0, 1] scale
    label : int, sequence of int, or None — true class; None uses the clean prediction
    loss : {"margin", "ce"} — attack loss; defaults to "margin"

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    x, single = as_batch(image)
    net = get_margin_free(model)
    gradient = _loss_gradient(net, x, _labels(net, x, label), loss)
    return _output(_project(x + epsilon * tf.sign(gradient), x, epsilon), single)


def bim(image, model, epsilon, label=None, steps=10, step_size=None, loss="margin"):
    """Basic Iterative Method (Kurakin et al., 2017): iterated FGSM, projected every step.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget
    label : int, sequence of int, or None
    steps : int — number of gradient steps
    step_size : float or None — defaults to epsilon / steps
    loss : {"margin", "ce"} — attack loss; defaults to "margin"

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    alpha = epsilon / steps if step_size is None else step_size
    x = tf.identity(x0)
    for _ in range(steps):
        x = _project(x + alpha * tf.sign(_loss_gradient(net, x, labels, loss)), x0, epsilon)
    return _output(x, single)


def pgd(image, model, epsilon, label=None, steps=10, step_size=None, random_start=True, seed=0, loss="margin"):
    """Projected Gradient Descent (Madry et al., 2018).

    Uniform random start inside the epsilon ball (seeded, so the function is
    deterministic), steps of 2.5 * epsilon / steps, projection every step.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget
    label : int, sequence of int, or None
    steps : int
    step_size : float or None — defaults to 2.5 * epsilon / steps
    random_start : bool
    seed : int — seed for the random start
    loss : {"margin", "ce"} — attack loss; defaults to "margin"

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    alpha = 2.5 * epsilon / steps if step_size is None else step_size
    x = tf.identity(x0)
    if random_start:
        noise = tf.random.stateless_uniform(tf.shape(x0), seed=[seed, 0],
                                            minval=-epsilon, maxval=epsilon)
        x = _project(x0 + noise, x0, epsilon)
    for _ in range(steps):
        x = _project(x + alpha * tf.sign(_loss_gradient(net, x, labels, loss)), x0, epsilon)
    return _output(x, single)


def mim(image, model, epsilon, label=None, steps=10, decay=1.0, step_size=None, loss="margin"):
    """Momentum Iterative Method (Dong et al., 2018).

    The momentum accumulates L1-normalised gradients; every step is projected.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget
    label : int, sequence of int, or None
    steps : int
    decay : float — momentum decay factor (mu)
    step_size : float or None — defaults to epsilon / steps
    loss : {"margin", "ce"} — attack loss; defaults to "margin"

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    alpha = epsilon / steps if step_size is None else step_size
    x = tf.identity(x0)
    momentum = tf.zeros_like(x0)
    for _ in range(steps):
        gradient = _loss_gradient(net, x, labels, loss)
        l1 = tf.reduce_mean(tf.abs(gradient), axis=[1, 2, 3], keepdims=True)
        momentum = decay * momentum + gradient / tf.maximum(l1, 1e-12)
        x = _project(x + alpha * tf.sign(momentum), x0, epsilon)
    return _output(x, single)


def _cw_check(net, one_hot, candidates, x0, kappa):
    """CW success (misclassified by more than kappa) and squared L2 to *x0* of each candidate."""
    logits = net(tf.constant(candidates))
    real = tf.reduce_sum(one_hot * logits, axis=1)
    other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
    return (other > real + kappa).numpy(), np.square(candidates - x0).sum(axis=(1, 2, 3))


def cw(image, model, epsilon=None, label=None, kappa=0.0, steps=50, binary_steps=4,
       initial_const=1.0, learning_rate=0.01, quantise_fn=None):
    """Carlini-Wagner L2 attack, untargeted (Carlini & Wagner, 2017).

    Minimises ||x' - x||^2 + c * max(Z_y - max_{j != y} Z_j, -kappa) in tanh
    space with Adam, with a binary search over c per image. It has no epsilon:
    the perturbation size is whatever CW needs. An iterate counts as a success
    when it is misclassified by more than kappa logits (max_{j != y} Z_j > Z_y + kappa,
    as in Carlini's reference code); the success with the smallest L2 is returned.
    Images for which no success was found are returned unchanged (as in
    CleverHans); callers should check success.

    8-bit images: with kappa = 0 the result sits on the decision boundary, and
    rounding it to the 8-bit grid undoes many successes (CW succeeded on 100 %
    of the float ablation pilot, but on 62.5 % of the 8-bit 600-image run).
    With *quantise_fn* every candidate is rounded before the success check: the
    binary search and the choice of the best candidate use the rounded image and
    its own L2, and the rounded image is returned. The clean image should
    already be on the grid (it is when loaded from a PNG). Each step then costs
    one extra forward pass.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : ignored, CW has no budget (kept so registry calls are uniform)
    label : int, sequence of int, or None
    kappa : float — confidence margin, in logits (the clean top-1 / top-2 gap is ~30-40)
    steps : int — Adam steps per binary-search step
    binary_steps : int — binary-search steps over c
    initial_const : float — starting value of c
    learning_rate : float
    quantise_fn : callable or None — images (N, 256, 256, 3) -> images of the same
        shape, applied to every candidate before the success check, e.g.
        ``src.attacks.budget.quantise`` (8-bit rounding)

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    n = int(x0.shape[0])
    one_hot = tf.one_hot(labels, get_n_classes(model))
    w0 = tf.atanh(tf.clip_by_value(x0 * 2.0 - 1.0, -1 + 1e-6, 1 - 1e-6))

    lower = np.zeros(n)
    upper = np.full(n, 1e10)
    const = np.full(n, initial_const)
    best_l2 = np.full(n, np.inf)
    reference = x0.numpy()
    best = reference.copy()

    for _ in range(binary_steps):
        w = tf.Variable(w0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        c = tf.constant(const, dtype=tf.float32)
        succeeded = np.zeros(n, dtype=bool)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                x_adv = (tf.tanh(w) + 1.0) / 2.0
                logits = net(x_adv)
                real = tf.reduce_sum(one_hot * logits, axis=1)
                other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
                margin = tf.maximum(real - other, -kappa)
                l2 = tf.reduce_sum(tf.square(x_adv - x0), axis=[1, 2, 3])
                loss = tf.reduce_sum(l2 + c * margin)
            optimizer.apply_gradients([(tape.gradient(loss, w), w)])

            candidates = x_adv.numpy()
            if quantise_fn is None:
                found, l2_np = (other > real + kappa).numpy(), l2.numpy()
            else:
                rounded = np.asarray(quantise_fn(candidates), dtype=np.float32)
                if rounded.shape != candidates.shape:
                    raise ValueError(f"quantise_fn must keep the shape {candidates.shape}, "
                                     f"got {rounded.shape}")
                candidates = rounded
                found, l2_np = _cw_check(net, one_hot, candidates, reference, kappa)
            improved = found & (l2_np < best_l2)
            best_l2[improved] = l2_np[improved]
            best[improved] = candidates[improved]
            succeeded |= found

        upper = np.where(succeeded, np.minimum(upper, const), upper)
        lower = np.where(succeeded, lower, np.maximum(lower, const))
        const = np.where(upper < 1e9, (lower + upper) / 2.0, const * 10.0)

    return _output(tf.constant(best), single)


# ---------------------------------------------------------------------------
# Paper replication — vendored ABRobOcular code with the paper's settings
# ---------------------------------------------------------------------------

_VENDORED = {}


def _vendored_attacks():
    """Load OcularAdversarialAttacks from its file (skips the torch-importing __init__)."""
    if "attacks" not in _VENDORED:
        path = os.path.join(_THIRD_PARTY, "AB_Rob_Ocular", "AB_Rob_Ocular_Tensor.py")
        spec = importlib.util.spec_from_file_location("ab_rob_ocular_tensor", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VENDORED["attacks"] = module.OcularAdversarialAttacks
    return _VENDORED["attacks"]


def _paper(method_name, image, model, **kwargs):
    """Run a vendored ``_face`` attack image by image with a zero label vector."""
    x, single = as_batch(image)
    method = getattr(_vendored_attacks(), method_name)
    dummy = np.zeros((1, get_n_classes(model)), dtype="float32")
    out = [np.squeeze(method(img, dummy, model, **kwargs), axis=0) for img in x.numpy()]
    out = np.clip(np.stack(out), 0.0, 1.0).astype(np.float32)
    return out[0] if single else out


def paper_fgsm(image, model, epsilon=0.001):
    """Vendored FGSM, paper settings. See the module docstring for its bugs."""
    return _paper("fgsm_attack_face", image, model, epsilon=epsilon)


def paper_bim(image, model, epsilon=0.001, iter_eps=0.005, iterations=10):
    """Vendored BIM, paper settings (projects onto the epsilon ball only at the end)."""
    return _paper("bim_attack_face", image, model, epsilon=epsilon,
                  iter_eps=iter_eps, iterations=iterations)


def paper_mim(image, model, epsilon=0.001, iter_eps=0.005, iterations=10, decay_factor=1.0):
    """Vendored MIM, paper settings (projects onto the epsilon ball only at the end)."""
    return _paper("mim_attack_face", image, model, epsilon=epsilon, iter_eps=iter_eps,
                  iterations=iterations, decay_factor=decay_factor)


def paper_pgd(image, model, epsilon=0.001, num_steps=10, step_size=0.0001):
    """Vendored PGD, paper settings. *epsilon* is ignored by the vendored code."""
    return _paper("pgd_attack_face", image, model, epsilon=epsilon,
                  num_steps=num_steps, step_size=step_size)


def paper_cw(image, model, epsilon=None, learning_rate=0.001, max_iter=10, num_starts=5):
    """Vendored "CW" (not Carlini-Wagner; non-deterministic). Epsilon is ignored for registry uniformity."""
    return _paper("cw_attack_face", image, model, learning_rate=learning_rate,
                  max_iter=max_iter, num_starts=num_starts)


# ---------------------------------------------------------------------------
# Registries for easy iteration in evaluation scripts
# ---------------------------------------------------------------------------

ATTACK_REGISTRY = {
    "fgsm": fgsm,
    "bim": bim,
    "pgd": pgd,
    "mim": mim,
    "cw": cw,
}

PAPER_ATTACK_REGISTRY = {
    "fgsm": paper_fgsm,
    "bim": paper_bim,
    "pgd": paper_pgd,
    "mim": paper_mim,
    "cw": paper_cw,
}
