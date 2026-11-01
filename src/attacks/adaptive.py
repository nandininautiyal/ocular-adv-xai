"""
src/attacks/adaptive.py

Adaptive white-box attacks: attacks that know which inference-time defence the
system applies, and attack the model *through* it.

Why this module exists
----------------------
JPEG compression and bit-depth reduction (src/defences/inference.py) run in
numpy, so no gradient reaches the input through them. An attack computed on the
undefended model and defended afterwards (scripts/05_run_experiment.py) only
measures a non-adaptive attacker; a defence that "restores" the label against
it can still be broken by an attacker who knows it is there. Robustness and
restoration claims are judged against adaptive attacks (Athalye et al., ICML
2018; Tramer et al., NeurIPS 2020).

Attacks
-------
bpda_pgd
    PGD on the margin-free logits of D(x). The forward pass uses the defended
    image; the backward pass treats the defence as the identity (Backward Pass
    Differentiable Approximation, BPDA): x_def = x + stop_gradient(D(x) - x),
    with D computed outside the tape at every step. A plain GradientTape gives
    no gradient at all through the numpy defences.
eot_bpda_pgd
    BPDA-PGD against a randomised defence: every step sums the BPDA gradients of
    n_samples defences drawn at random (Expectation over Transformation, EOT),
    e.g. JPEG with a random quality (random_jpeg).
transform_robust_pgd
    Maximises the summed margin loss over several transforms of the same image,
    e.g. [identity, D]. The result has to fool the model with and without the
    defence: the attack that a detector comparing x with D(x) (explanation
    consistency, Feature Squeezing) would face.

8-bit protocol
--------------
With quantise_8bit=True (the default) every forward pass sees what a deployed
system receives: the iterate rounded to the 8-bit grid, within the budget
around the clean image, then defended. The rounding is straight-through
(rounded forward, identity backward) and the iterate itself stays in float, so
steps smaller than one level add up. For a clean image on the 8-bit grid (every
image loaded from a PNG) the rounding equals ``quantise`` in
scripts/05_run_experiment.py: the largest whole number of levels within the
budget (e.g. 7 levels for epsilon = 0.03). Budgets below one level (1/255)
cannot change such an image and raise ValueError.

Every iterate is scored after rounding and the defence, and the attack returns
the one with the lowest defended margin (the most adversarial; at least as good
as the last step). Success is therefore judged after the defence, on exactly the
8-bit image that is returned.

Margins are the true-class logit minus the best other logit (margin-free
logits); a negative margin means the image is misclassified.

Image contract
--------------
- Input  : float32, shape (256, 256, 3) or (N, 256, 256, 3), range [0, 1]
- Output : float32, same shape as the input, range [0, 1]; on the 8-bit grid
           and within the budget when quantise_8bit=True
"""

import functools
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.attacks.whitebox import _labels, _project
from src.defences.inference import jpeg_compress
from src.models.arcface_inference import as_batch, get_margin_free

_LEVELS = 255.0
_TOLERANCE = 1e-3      # budgets are written to ~5 decimals (0.0078431 * 255 = 1.99999)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def identity(images):
    """The undefended image; pair it with a defence in ``transforms``."""
    return np.asarray(images, dtype=np.float32)


def random_jpeg(min_quality=60, max_quality=90):
    """Randomised JPEG for EOT.

    Returns ``sample(rng)``, which draws a quality uniformly from the integers in
    [min_quality, max_quality] with the numpy Generator *rng* and returns
    ``jpeg_compress`` with that quality.
    """
    if not 1 <= min_quality <= max_quality <= 100:
        raise ValueError(f"need 1 <= min_quality <= max_quality <= 100, got {min_quality}, {max_quality}")

    def sample(rng):
        return functools.partial(jpeg_compress, quality=int(rng.integers(min_quality, max_quality + 1)))
    return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _margin(logits, labels):
    """True-class logit minus the best other logit, per image (negative = misclassified)."""
    one_hot = tf.one_hot(labels, tf.shape(logits)[1])
    true_logit = tf.reduce_sum(one_hot * logits, axis=1)
    best_other = tf.reduce_max(logits - 1e9 * one_hot, axis=1)
    return true_logit - best_other


def _attack_loss(logits, labels, loss):
    """Summed loss the attack maximises: the negative margin, or cross-entropy."""
    if loss == "ce":
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return -tf.reduce_sum(_margin(logits, labels))


def _as_transforms(transforms):
    transforms = (transforms,) if callable(transforms) else tuple(transforms)
    if not transforms:
        raise ValueError("need at least one transform")
    return transforms


def _grid_rounding(clean, epsilon):
    """Function that rounds images to the nearest 8-bit level within *epsilon* of *clean*.

    Same arithmetic as ``quantise`` in scripts/05_run_experiment.py (rounding in float32, so
    half-level ties go the same way): around a clean value on the grid, the largest whole
    number of levels within the budget. Around a value off the grid, the levels strictly
    within the budget.
    """
    levels = np.floor(epsilon * _LEVELS + _TOLERANCE)
    if levels < 1:
        raise ValueError(f"epsilon = {epsilon} is below one 8-bit level (1/255) and cannot change an "
                         f"8-bit image; use quantise_8bit=False for a float-domain attack")
    scaled = np.asarray(clean, np.float64) * _LEVELS
    nearest = np.round(scaled)
    on_grid = np.abs(scaled - nearest) < 1e-4            # float32 copies of k/255 are within ~1.5e-5 of k
    low = np.where(on_grid, nearest - levels, np.ceil(scaled - epsilon * _LEVELS - 1e-9))
    high = np.where(on_grid, nearest + levels, np.floor(scaled + epsilon * _LEVELS + 1e-9))
    # whole levels 0..255 are exact in float32, which keeps the per-step rounding cheap
    low = np.clip(low, 0.0, _LEVELS).astype(np.float32)
    high = np.clip(high, 0.0, _LEVELS).astype(np.float32)

    def rounding(images):
        values = np.clip(np.asarray(images, np.float32), 0.0, 1.0)     # a new array: safe to work in place
        values *= np.float32(_LEVELS)
        np.round(values, out=values)
        np.clip(values, low, high, out=values)
        values /= np.float32(_LEVELS)
        return values
    return rounding


def _margins_and_gradient(net, x, seen, labels, transforms, loss, gradient=True):
    """Margin of every transform of *seen*, and the BPDA gradient of the summed loss w.r.t. *x*.

    x     : the float iterate (tensor) the gradient is taken with respect to
    seen  : what the forward pass starts from (x itself, or x rounded to 8 bits; ndarray)
    Each transform runs outside the tape. x_def has the value t(seen) and the identity as its
    gradient (BPDA, with straight-through rounding): x + stop_gradient(t(seen) - x), written as
    t(seen) + (x - stop_gradient(x)) so that the value is exactly t(seen) in float32.

    Returns margins (T, N) as an ndarray, and the gradient (tensor; None if gradient=False).
    """
    margins, total = [], None
    for transform in transforms:
        z = tf.constant(np.asarray(transform(seen), dtype=np.float32))
        if not gradient:
            margins.append(_margin(net(z), labels).numpy())
            continue
        with tf.GradientTape() as tape:
            tape.watch(x)
            x_def = z + (x - tf.stop_gradient(x))
            logits = net(x_def)
            value = _attack_loss(logits, labels, loss)
        margins.append(_margin(logits, labels).numpy())
        step = tape.gradient(value, x)
        total = step if total is None else total + step
    return np.stack(margins), total


def _attack(image, model, epsilon, label, steps, step_size, random_start, seed, loss, quantise_8bit,
            transforms_at, reduce, return_info):
    """Projected sign-gradient ascent through ``transforms_at(step)``, keeping the best iterate.

    reduce : (T, N) margins -> (N,) score; the iterate with the lowest score is returned.
    """
    if loss not in ("ce", "margin"):
        raise ValueError(f"loss must be 'ce' or 'margin', got {loss!r}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if not epsilon > 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    x0, single = as_batch(image)
    net = get_margin_free(model)
    labels = _labels(net, x0, label)
    alpha = 2.5 * epsilon / steps if step_size is None else step_size
    rounding = _grid_rounding(x0.numpy(), epsilon) if quantise_8bit else identity

    x = tf.identity(x0)
    if random_start:            # same start as whitebox.pgd for the same seed
        noise = tf.random.stateless_uniform(tf.shape(x0), seed=[seed, 0],
                                            minval=-epsilon, maxval=epsilon)
        x = _project(x0 + noise, x0, epsilon)

    best, best_margin, history = None, None, []
    for step in range(steps + 1):            # steps + 1 iterates: the start and one per step
        seen = rounding(x.numpy())
        margins, gradient = _margins_and_gradient(net, x, seen, labels, transforms_at(step), loss,
                                                  gradient=step < steps)
        margin = reduce(margins)
        history.append(margin)
        if best is None:
            best, best_margin = seen.copy(), margin.copy()
        else:
            better = margin < best_margin
            best[better], best_margin[better] = seen[better], margin[better]
        if step < steps:
            x = _project(x + alpha * tf.sign(gradient), x0, epsilon)

    out = np.clip(best, 0.0, 1.0).astype(np.float32)
    out = out[0] if single else out
    if not return_info:
        return out
    return out, {"margin": best_margin, "success": best_margin < 0, "history": np.stack(history)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bpda_gradient(image, model, transforms, label=None, loss="margin"):
    """BPDA gradient of the attack loss summed over *transforms* of *image*, and their margins.

    The forward pass of each transform uses its real output, the backward pass the
    identity. A plain GradientTape gives no gradient at all through the numpy defences.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    transforms : callable or sequence of callables, images -> images of the same shape
    label : int, sequence of int, or None — true class; None uses the clean prediction
    loss : {"margin", "ce"} — the loss whose gradient (ascent direction) is returned

    Returns
    -------
    gradient : ndarray, same shape as *image*
    margins : ndarray (T, N) — margin of each transformed image (N = 1 for a single image)
    """
    x, single = as_batch(image)
    if loss not in ("ce", "margin"):
        raise ValueError(f"loss must be 'ce' or 'margin', got {loss!r}")
    net = get_margin_free(model)
    margins, gradient = _margins_and_gradient(net, x, x.numpy(), _labels(net, x, label),
                                              _as_transforms(transforms), loss)
    gradient = gradient.numpy()
    return (gradient[0] if single else gradient), margins


def bpda_pgd(image, model, epsilon, defence, label=None, steps=10, step_size=None, random_start=True,
             seed=0, loss="margin", quantise_8bit=True, return_info=False):
    """PGD through a non-differentiable defence with BPDA (Athalye et al., 2018).

    Same steps, step size and random start as ``whitebox.pgd``, but every step attacks
    D(round(x)) and the result is the iterate whose defended 8-bit image has the lowest
    margin.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — L-inf budget on the [0, 1] scale
    defence : callable, images -> images (e.g. functools.partial(jpeg_compress, quality=75))
    label : int, sequence of int, or None — true class; None uses the clean prediction
    steps : int
    step_size : float or None — defaults to 2.5 * epsilon / steps
    random_start : bool
    seed : int — seed for the random start
    loss : {"margin", "ce"} — attack loss; defaults to "margin"
    quantise_8bit : bool — attack and return 8-bit images (see the module docstring)
    return_info : bool — also return a dict with ``margin`` (N,) and ``success`` (N,) of
        D(result), and ``history`` (steps + 1, N), the defended margin of every iterate
        (N = 1 for a single image)

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    return _attack(image, model, epsilon, label, steps, step_size, random_start, seed, loss,
                   quantise_8bit, transforms_at=lambda step: (defence,),
                   reduce=lambda margins: margins[0], return_info=return_info)


def eot_bpda_pgd(image, model, epsilon, sample_defence, n_samples=8, label=None, steps=10, step_size=None,
                 random_start=True, seed=0, loss="margin", quantise_8bit=True, return_info=False):
    """BPDA-PGD with Expectation over Transformation, for a randomised defence.

    Every step draws *n_samples* defences with ``sample_defence(rng)`` and follows the sum of
    their BPDA gradients. The iterate with the lowest mean margin over the draws of its own
    step is returned (a noisy estimate for a randomised defence).

    Parameters
    ----------
    image, model, epsilon, label, steps, step_size, random_start, loss, quantise_8bit,
    return_info : as in :func:`bpda_pgd` (``margin`` and ``history`` are means over the draws)
    sample_defence : callable, numpy Generator -> defence callable (e.g. :func:`random_jpeg`)
    n_samples : int — defences drawn per step
    seed : int — seeds the random start and the draws

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    return _attack(image, model, epsilon, label, steps, step_size, random_start, seed, loss,
                   quantise_8bit, transforms_at=lambda step: [sample_defence(rng) for _ in range(n_samples)],
                   reduce=lambda margins: margins.mean(axis=0), return_info=return_info)


def transform_robust_pgd(image, model, epsilon, transforms, label=None, steps=10, step_size=None,
                         random_start=True, seed=0, loss="margin", quantise_8bit=True, return_info=False):
    """PGD on the margin loss summed over several transforms of the image, e.g. [identity, D].

    The defence gradients use BPDA. The result is the iterate whose worst transform (the
    highest margin) is lowest, so success means the model is fooled under every transform.

    Parameters
    ----------
    image, model, epsilon, label, steps, step_size, random_start, seed, loss, quantise_8bit,
    return_info : as in :func:`bpda_pgd` (``margin`` and ``history`` are the worst case
        over the transforms)
    transforms : sequence of callables, images -> images (use :func:`identity` for no defence)

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    transforms = _as_transforms(transforms)
    return _attack(image, model, epsilon, label, steps, step_size, random_start, seed, loss,
                   quantise_8bit, transforms_at=lambda step: transforms,
                   reduce=lambda margins: margins.max(axis=0), return_info=return_info)
