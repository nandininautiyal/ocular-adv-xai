"""8-bit rounding of attacked images must keep them on the grid and inside the attack budget."""
import importlib.util
import os
import sys

import numpy as np
import pytest

from conftest import REPO


def _script(script):
    spec = importlib.util.spec_from_file_location(script, os.path.join(REPO, "scripts", f"{script}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quantise(script):
    return _script(script).quantise


@pytest.mark.parametrize("script", ["05_run_experiment", "11_run_detection"])
def test_quantise_stays_on_grid_and_within_budget(script):
    quantise = _quantise(script)
    rng = np.random.default_rng(0)
    clean = np.round(rng.uniform(0, 1, (2, 8, 8, 3)) * 255) / 255
    for eps in (1 / 255, 2 / 255, 0.03, 0.001):
        adv = np.clip(clean + eps * np.sign(rng.uniform(-1, 1, clean.shape)), 0, 1)
        q = quantise(adv, clean, eps)
        assert np.allclose(q * 255, np.round(q * 255), atol=1e-4)            # on the 8-bit grid
        assert np.abs(q - clean).max() <= eps + 1e-6                        # never past the budget
    assert np.abs(quantise(clean + 0.001, clean, 0.001) - clean).max() < 1e-6  # below one level: erased
    # without a reference it is plain rounding
    assert quantise(np.array([0.5 / 255 + 1e-4]))[0] == pytest.approx(1 / 255)


@pytest.mark.parametrize("script", ["05_run_experiment", "11_run_detection"])
def test_scripts_use_the_shared_quantise(script):
    # one implementation for both runners (src/attacks/budget.py), so they cannot drift apart
    from src.attacks.budget import quantise
    assert _quantise(script) is quantise


@pytest.mark.parametrize("levels", ["1,2.5", "0", "256"])
def test_detection_levels_must_be_whole_levels(levels, monkeypatch, capsys):
    # --levels is read before the checkpoint is loaded, so the run stops at once
    detection = _script("11_run_detection")
    monkeypatch.setattr(sys, "argv", ["11_run_detection.py", "--ckpt", "unused.keras", "--correct-csv",
                                      "unused.csv", "--levels", levels])
    with pytest.raises(SystemExit) as stop:
        detection.main()
    assert stop.value.code == 2 and "--levels" in capsys.readouterr().err


@pytest.mark.parametrize("script", ["05_run_experiment", "11_run_detection"])
def test_default_budgets_keep_their_levels(script):
    # the defaults are 1, 2 and 4 levels written as rounded decimals; none may lose a level
    quantise = _quantise(script)
    clean = np.full((1, 4, 4, 3), 128 / 255, dtype=np.float32)
    for eps, levels in zip((0.0039216, 0.0078431, 0.0156863), (1, 2, 4)):
        q = quantise(clean + eps, clean, eps)
        assert np.abs(q - clean).max() * 255 == pytest.approx(levels, abs=1e-3)
