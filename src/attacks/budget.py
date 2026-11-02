"""
src/attacks/budget.py

8-bit perturbation budgets.

A deployed recogniser receives images saved with 8 bits per channel, so an
L-inf budget only means something as a whole number of grey levels (one level
is 1/255 on the [0, 1] scale). Budgets are written on the command line as
rounded decimals (0.0078431 for 2/255); this module turns them into levels and
back, and rounds attacked images to the 8-bit grid without ever going past the
budget.

levels(epsilon)
    Whole number of levels in a budget. Raises ValueError when the decimal does
    not name a whole number of levels: 0.0078 is 1.989 levels, a typo for 2/255
    that the rounding in ``quantise`` would silently turn into 1 level (the bug
    class fixed in PR #30), and 0.001 is a quarter of a level.
levels_to_epsilon(n)
    n / 255.
quantise(images, reference=None, epsilon=None)
    Round to the 8-bit grid, then clip to the budget around the clean images.
    Same semantics as ``quantise`` in scripts/05_run_experiment.py and
    scripts/11_run_detection.py on main (da52f8c): the budget is floored, not
    validated, so check budgets with ``levels`` first.
"""

import math

import numpy as np

LEVEL_TOLERANCE = 1e-3     # budgets are written to ~5 decimals: 0.0078431 * 255 = 1.99999


def levels(epsilon):
    """Whole number of 8-bit levels in the L-inf budget *epsilon* ([0, 1] scale).

    Parameters
    ----------
    epsilon : float — e.g. 0.0039216, 0.0078431, 0.0156863 (1, 2 and 4 levels)

    Returns
    -------
    n : int in [0, 255]

    Raises
    ------
    ValueError
        If epsilon * 255 is not within 1e-3 of a whole number in [0, 255]
        (0.0078 is 1.989 levels, 0.001 is 0.255 levels).
    """
    value = float(epsilon) * 255.0
    n = round(value) if math.isfinite(value) else None
    if n is None or abs(value - n) > LEVEL_TOLERANCE or not 0 <= n <= 255:
        raise ValueError(f"epsilon = {epsilon!r} is {value:.4g} 8-bit levels, not a whole number "
                         f"in [0, 255]; write budgets as n/255 to 7 decimals (0.0078431 = 2 levels)")
    return int(n)


def levels_to_epsilon(n):
    """L-inf budget of *n* whole 8-bit levels, n / 255.

    Raises ValueError unless *n* is a whole number in [0, 255].
    """
    try:
        whole = int(n)
    except (TypeError, ValueError, OverflowError):
        whole = None
    if whole is None or whole != n or not 0 <= whole <= 255:
        raise ValueError(f"n must be a whole number of 8-bit levels in [0, 255], got {n!r}")
    return whole / 255.0


def quantise(images, reference=None, epsilon=None):
    """Round to the 8-bit grid a saved image has: what a deployed system actually receives.

    With *reference* (the clean images, already on the grid) and *epsilon*, the result is
    also clipped to the largest whole number of 8-bit levels within the budget, so rounding
    never pushes a perturbation past epsilon (e.g. epsilon = 0.03 allows 7 levels, not 8).

    Parameters
    ----------
    images : ndarray (..., 3), float — attacked or noisy images
    reference : ndarray of the same shape, or None — the clean images, on the 8-bit grid
    epsilon : float or None — L-inf budget; floored to whole levels (with a 1e-3 tolerance),
        not validated: call :func:`levels` first to reject budgets such as 0.0078

    Returns
    -------
    quantised : ndarray, same shape, float32, on the 8-bit grid
    """
    images = np.asarray(images, dtype=np.float32)
    out = np.round(np.clip(images, 0.0, 1.0) * 255.0) / 255.0
    if reference is not None and epsilon is not None:
        # budgets are written to ~5 decimals (0.0078431 * 255 = 1.99999), so allow for that
        budget = np.floor(epsilon * 255.0 + 1e-3) / 255.0
        reference = np.asarray(reference, dtype=np.float32)
        out = np.clip(out, reference - budget, reference + budget)
    return out.astype(np.float32)
