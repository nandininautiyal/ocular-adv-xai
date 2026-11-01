"""8-bit budgets: whole levels only, and the same rounding as the runner scripts on main."""
import importlib.util
import os

import numpy as np
import pytest

from conftest import REPO
from src.attacks.budget import levels, levels_to_epsilon, quantise


def quantise_main(images, reference=None, epsilon=None):
    """Verbatim copy of ``quantise`` in scripts/05_run_experiment.py on main (da52f8c)."""
    images = np.asarray(images, dtype=np.float32)
    out = np.round(np.clip(images, 0.0, 1.0) * 255.0) / 255.0
    if reference is not None and epsilon is not None:
        # budgets are written to ~5 decimals (0.0078431 * 255 = 1.99999), so allow for that
        levels = np.floor(epsilon * 255.0 + 1e-3) / 255.0
        reference = np.asarray(reference, dtype=np.float32)
        out = np.clip(out, reference - levels, reference + levels)
    return out.astype(np.float32)


def _script_quantise(script):
    spec = importlib.util.spec_from_file_location(script, os.path.join(REPO, "scripts", f"{script}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.quantise


def test_default_budgets_are_whole_levels():
    assert levels(0.0039216) == 1
    assert levels(0.0078431) == 2
    assert levels(0.0156863) == 4
    assert levels(2 / 255) == 2 and levels(0.0) == 0 and levels(1.0) == 255
    assert isinstance(levels(np.float32(0.0078431)), int)


@pytest.mark.parametrize("epsilon", [0.0078, 0.0039, 0.001, 0.03, -1 / 255, 256 / 255,
                                     float("nan"), float("inf")])
def test_budgets_that_are_not_whole_levels_are_rejected(epsilon):
    # 0.0078 is 1.989 levels: quantise would floor it to 1 level without a word
    with pytest.raises(ValueError):
        levels(epsilon)


def test_levels_to_epsilon_round_trips():
    assert levels_to_epsilon(1) == pytest.approx(1 / 255)
    assert levels_to_epsilon(np.int64(4)) == pytest.approx(4 / 255)
    assert all(levels(levels_to_epsilon(n)) == n for n in range(256))
    for bad in (1.5, -1, 256, "2", None, float("nan")):
        with pytest.raises(ValueError):
            levels_to_epsilon(bad)


def test_quantise_equals_the_main_function_on_random_data():
    rng = np.random.default_rng(0)
    for trial in range(20):
        clean = (np.round(rng.uniform(0, 1, (3, 16, 16, 3)) * 255) / 255).astype(np.float32)
        adv = (clean + rng.uniform(-0.1, 0.1, clean.shape)).astype(np.float32)   # leaves [0, 1] too
        for eps in (0.0039216, 0.0078431, 0.0156863, 2 / 255, 0.03, 0.001, 0.0078, 0.5):
            np.testing.assert_array_equal(quantise(adv, clean, eps), quantise_main(adv, clean, eps))
        np.testing.assert_array_equal(quantise(adv), quantise_main(adv))
        np.testing.assert_array_equal(quantise(adv, clean), quantise_main(adv, clean))   # no budget
        assert quantise(adv, clean, 0.0078431).dtype == np.float32


@pytest.mark.parametrize("script", ["05_run_experiment", "11_run_detection"])
def test_quantise_main_copy_matches_the_scripts(script):
    # the scripts now import src.attacks.budget.quantise; this keeps them tied to main's rounding
    rng = np.random.default_rng(1)
    clean = (np.round(rng.uniform(0, 1, (2, 8, 8, 3)) * 255) / 255).astype(np.float32)
    adv = (clean + rng.uniform(-0.05, 0.05, clean.shape)).astype(np.float32)
    old = _script_quantise(script)
    for eps in (0.0039216, 0.0078431, 0.03, None):
        np.testing.assert_array_equal(old(adv, clean, eps), quantise_main(adv, clean, eps))


def test_quantise_stays_on_grid_within_budget_and_leaves_input_untouched():
    rng = np.random.default_rng(2)
    clean = (np.round(rng.uniform(0, 1, (2, 8, 8, 3)) * 255) / 255).astype(np.float32)
    adv = (clean + 0.02 * rng.standard_normal(clean.shape)).astype(np.float32)
    before = adv.copy()
    for n in (1, 2, 4):
        q = quantise(adv, clean, levels_to_epsilon(n))
        assert np.allclose(q * 255, np.round(q * 255), atol=1e-4)
        assert np.abs(q - clean).max() * 255 == pytest.approx(n, abs=1e-3)
        np.testing.assert_allclose(quantise(q, clean, levels_to_epsilon(n)), q, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(adv, before)
