"""End-to-end: run scripts/05_run_experiment.py the way the docs say, on the tiny model."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from conftest import REPO


def test_run_experiment_end_to_end(tmp_path, model, images, labels):
    ckpt = tmp_path / "tiny.keras"
    model.save(ckpt)

    rows = []
    for i, (img, label) in enumerate(zip(images, labels)):
        path = tmp_path / f"{label}_{i}_l_c.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(path)
        rows.append({"link_path": str(path), "true_idx": int(label), "true_subject": 1000 + int(label),
                     "eye": "left" if i % 2 else "right", "device": "test", "lighting": "office",
                     "phase": "enrollment"})
    correct_csv = tmp_path / "correct_test_tiny.csv"
    pd.DataFrame(rows).to_csv(correct_csv, index=False)

    out_dir = tmp_path / "out"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "05_run_experiment.py"),
           "--ckpt", str(ckpt), "--correct-csv", str(correct_csv), "--per-class", "1",
           "--attacks", "fgsm,pgd,cw,patch,monocle", "--epsilons", "0.004,0.03",
           "--batch-size", "3", "--save-examples", "1", "--out-dir", str(out_dir), "--tag", "tiny"]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]

    table = pd.read_csv(out_dir / "rows_tiny.csv")
    assert table.image_id.nunique() == 5
    assert set(table.condition) == {"clean", "noise", "defence_on_clean", "attack", "attack+defence"}
    assert set(table.xai) == {"layercam", "eigencam"}
    clean = table[table.condition == "clean"]
    assert clean.correct.all() and (clean.spearman.round(6) == 1.0).all()
    defended = table[table.condition == "attack+defence"]
    assert defended.vs_defended_clean_spearman.notna().all()
    assert table[table.condition != "attack+defence"].vs_defended_clean_spearman.isna().all()
    linf = table[(table.condition == "attack") & (table.attack == "fgsm")].groupby("epsilon").linf.max()
    assert (linf <= linf.index + 1e-6).all()

    summary = json.loads((out_dir / "summary_tiny.json").read_text())
    assert summary["n_images"] == 5
    assert summary["attack_success_rate"]["pgd@0.03"]["rate"] > 0
    assert summary["attack_success_rate"]["cw"]["rate"] > 0
    assert not any(k.startswith("cw@") for k in summary["attack_success_rate"])   # CW runs once
    assert summary["restoration_rate_among_successful_attacks"]
    assert "noise@0.03" in summary["noise_control_flip_rate"]
    assert set(summary["defence_on_clean_flip_rate"]) == {"clean+jpeg", "clean+bit_depth"}
    assert (out_dir / "summary_tiny.csv").exists()
    assert len(np.load(out_dir / "examples_tiny.npz").files) > 0
