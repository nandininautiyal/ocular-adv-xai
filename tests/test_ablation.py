"""The ablation attack must reproduce the vendored ABRobOcular attacks exactly at VENDORED settings."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from conftest import REPO
from src.attacks.ablation import ablation_configs, iterative_attack
from src.attacks.whitebox import paper_bim, paper_fgsm, paper_mim, paper_pgd

EPS = 0.001


def _agreement(a, b):
    """Fraction of pixels that agree to float precision."""
    return float(np.mean(np.abs(a - b) < 1e-6))


def test_model_output_is_already_a_softmax(model, images, zeros):
    """The vendored loss applies softmax to this output a second time."""
    probs = model([images, zeros], training=False).numpy()
    assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name, vendored", [
    ("fgsm/vendored", lambda x, m: paper_fgsm(x, m, epsilon=EPS)),
    ("bim/vendored", lambda x, m: paper_bim(x, m, epsilon=EPS)),
    ("mim/vendored", lambda x, m: paper_mim(x, m, epsilon=EPS)),
])
def test_vendored_settings_reproduce_the_vendored_code(name, vendored, model, images):
    _, kwargs = ablation_configs(EPS)[name]
    ours = iterative_attack(images, model, EPS, **kwargs)
    theirs = vendored(images, model)
    assert _agreement(ours, theirs) > 0.999


def test_vendored_pgd_ignores_epsilon(model, images):
    """Its budget is num_steps * step_size; epsilon is never used."""
    np.testing.assert_array_equal(paper_pgd(images[:2], model, epsilon=0.001),
                                  paper_pgd(images[:2], model, epsilon=0.5))


def test_projection_end_lets_the_path_leave_the_ball(model, images):
    """With 5x steps and one final clip, the result differs from per-step projection."""
    configs = ablation_configs(EPS)
    end = iterative_attack(images, model, EPS, **configs["bim/vendored"][1])
    each = iterative_attack(images, model, EPS, **configs["bim/fix_projection"][1])
    for adv in (end, each):
        assert np.abs(adv - images).max() <= EPS + 1e-6
    assert _agreement(end, each) < 0.999


def test_vendored_cw_is_unbounded(model, images):
    """The vendored 'CW' adds N(0, 0.1) noise between restarts and has no distance term."""
    from src.attacks.whitebox import paper_cw
    tf.random.set_seed(0)
    adv = paper_cw(images[:2], model)
    assert np.abs(adv - images[:2]).max() > 0.1


def test_every_config_respects_the_budget(model, images):
    for name, (_, kwargs) in ablation_configs(EPS).items():
        adv = iterative_attack(images[:2], model, EPS, **kwargs)
        assert np.abs(adv - images[:2]).max() <= EPS + 1e-6, name


def test_bad_arguments():
    with pytest.raises(ValueError):
        iterative_attack(np.zeros((256, 256, 3), np.float32), None, EPS, loss_space="nope")
    with pytest.raises(ValueError):
        iterative_attack(np.zeros((256, 256, 3), np.float32), None, EPS, projection="sometimes")


def test_ablation_script_end_to_end(tmp_path, model, images, labels):
    ckpt = tmp_path / "tiny.keras"
    model.save(ckpt)
    rows = []
    for i, (img, label) in enumerate(zip(images, labels)):
        path = tmp_path / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        rows.append({"link_path": str(path), "true_idx": int(label)})
    correct_csv = tmp_path / "correct.csv"
    pd.DataFrame(rows).to_csv(correct_csv, index=False)
    out = tmp_path / "ablation"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "13_attack_ablation.py"), "--ckpt", str(ckpt),
           "--correct-csv", str(correct_csv), "--n", "5", "--epsilons", "0.001,0.03", "--batch-size", "3",
           "--out-dir", str(out), "--tag", "tiny"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    table = pd.read_csv(out / "ablation_tiny.csv")
    assert set(ablation_configs(0.001)) | {"pgd/vendored", "pgd/standard_margin", "cw/vendored",
                                           "cw/standard_l2"} == set(table.config)
    bounded = table[table.family != "cw"]
    assert set(bounded.epsilon.round(4)) == {0.001, 0.03}
    assert table.asr.between(0, 1).all() and (bounded.max_linf <= bounded.epsilon + 1e-6).all()
    assert {"asr_ci_low", "asr_ci_high", "n"} <= set(table.columns)
