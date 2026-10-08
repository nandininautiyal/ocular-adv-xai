"""
src/attacks/whitebox.py

Thin, uniform wrappers around the vendored ABRobOcular ``_face`` attack
variants.  Each wrapper handles the ArcFace dual-input model signature
(dummy zero-label injection) so callers never have to think about it.

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
- Output : float32, shape (256, 256, 3), range [0, 1]

Every function returns a *single image* (no batch dimension).

We do NOT rewrite the attack logic — we delegate to the vendored code and
keep ``third_party/`` unmodified.
"""

import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

# --- import vendored attack implementations --------------------------------
_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "ABRobOcular")
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from AB_Rob_Ocular.AB_Rob_Ocular_Tensor import OcularAdversarialAttacks  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_labels(n_classes):
    """Zero one-hot vector — suppresses the ArcFace margin during attack."""
    return np.zeros((1, n_classes), dtype="float32")


def _squeeze(adv):
    """Remove batch dimension and ensure float32 in [0, 1]."""
    out = np.squeeze(adv).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API — each returns shape (256, 256, 3)
# ---------------------------------------------------------------------------

def fgsm(image, model, epsilon=0.01, n_classes=300):
    """Fast Gradient Sign Method (FGSM) for the ArcFace model.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — perturbation magnitude
    n_classes : int

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    dummy = _dummy_labels(n_classes)
    adv = OcularAdversarialAttacks.fgsm_attack_face(
        image, dummy, model, epsilon=epsilon,
    )
    return _squeeze(adv)


def bim(image, model, epsilon=0.01, n_classes=300, iterations=10,
        iter_eps=0.005):
    """Basic Iterative Method (BIM) for the ArcFace model.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — total perturbation budget
    n_classes : int
    iterations : int — number of gradient steps
    iter_eps : float — step size per iteration

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    dummy = _dummy_labels(n_classes)
    adv = OcularAdversarialAttacks.bim_attack_face(
        image, dummy, model, epsilon=epsilon,
        iter_eps=iter_eps, iterations=iterations,
    )
    return _squeeze(adv)


def mim(image, model, epsilon=0.01, n_classes=300, iterations=10,
        iter_eps=0.005, decay_factor=1.0):
    """Momentum Iterative Method (MIM) for the ArcFace model.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — total perturbation budget
    n_classes : int
    iterations : int
    iter_eps : float — step size per iteration
    decay_factor : float — momentum decay

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    dummy = _dummy_labels(n_classes)
    adv = OcularAdversarialAttacks.mim_attack_face(
        image, dummy, model, epsilon=epsilon,
        iter_eps=iter_eps, iterations=iterations,
        decay_factor=decay_factor,
    )
    return _squeeze(adv)


def pgd(image, model, epsilon=0.01, n_classes=300, num_steps=10,
        step_size=0.001):
    """Projected Gradient Descent (PGD) for the ArcFace model.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    epsilon : float — Linf perturbation budget
    n_classes : int
    num_steps : int
    step_size : float

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    dummy = _dummy_labels(n_classes)
    adv = OcularAdversarialAttacks.pgd_attack_face(
        image, dummy, model, epsilon=epsilon,
        num_steps=num_steps, step_size=step_size,
    )
    return _squeeze(adv)


def cw(image, model, n_classes=300, learning_rate=0.001, max_iter=10,
       num_starts=5):
    """Carlini–Wagner (CW) L2 attack for the ArcFace model.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    model : tf.keras.Model with dual inputs [image, label]
    n_classes : int
    learning_rate : float
    max_iter : int
    num_starts : int

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    dummy = _dummy_labels(n_classes)
    adv = OcularAdversarialAttacks.cw_attack_face(
        image, dummy, model,
        learning_rate=learning_rate,
        max_iter=max_iter,
        num_starts=num_starts,
    )
    return _squeeze(adv)


# ---------------------------------------------------------------------------
# Registry for easy iteration in evaluation scripts
# ---------------------------------------------------------------------------

ATTACK_REGISTRY = {
    "fgsm": fgsm,
    "bim": bim,
    "mim": mim,
    "pgd": pgd,
    "cw": cw,
}
