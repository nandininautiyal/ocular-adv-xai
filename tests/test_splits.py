"""The data split must not depend on the order the disk lists files in."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from conftest import REPO


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest_script = _load("01_build_manifest")
split_script = _load("02_make_splits")


def _fake_visob(root, subjects=(1141, 1142, 1143), per_folder=10):
    for subject in subjects:
        d = root / "oppo_day_light_short" / "enrollment_left_eye" / str(subject)
        d.mkdir(parents=True)
        for idx in range(1000, 1000 + per_folder):
            (d / f"{subject}_{idx}_l_c.png").write_bytes(b"")
    return root


def _manifest(n_subjects=3, per_class=20):
    rows = [{"relpath": f"c/p/{s}/{s}_{i}_l_c.png", "subject": s}
            for s in range(1141, 1141 + n_subjects) for i in range(per_class)]
    return pd.DataFrame(rows)


def test_manifest_rows_are_sorted(tmp_path):
    df, skipped = manifest_script.build(str(_fake_visob(tmp_path)))
    assert skipped == [] and len(df) == 30
    assert df.relpath.tolist() == sorted(df.relpath)


def test_split_ignores_row_order():
    m = _manifest()
    a = split_script.assign_splits(m, seed=42)
    b = split_script.assign_splits(m.sample(frac=1.0, random_state=7), seed=42)
    pd.testing.assert_frame_equal(a, b)
    counts = a.groupby("subject").split.value_counts().unstack()
    assert (counts["train"] == 12).all() and (counts["validation"] == 4).all() and (counts["test"] == 4).all()


def test_pinned_test_list_is_respected_and_order_independent():
    m = _manifest()
    pinned = m.groupby("subject").head(5).relpath
    a = split_script.assign_splits(m, seed=42, test_relpaths=pinned)
    b = split_script.assign_splits(m.iloc[::-1], seed=42, test_relpaths=pinned)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.loc[a.split == "test", "relpath"]) == set(pinned)
    counts = a[a.split != "test"].groupby("subject").split.value_counts().unstack()
    assert (counts["train"] == 11).all() and (counts["validation"] == 4).all()   # 15 left: 75/25


def test_pinned_test_list_must_exist_in_manifest():
    with pytest.raises(ValueError, match="not in the manifest"):
        split_script.assign_splits(_manifest(), test_relpaths=["missing.png"])


def test_committed_test_list_matches_the_baseline_run():
    test = pd.read_csv(os.path.join(REPO, "splits", "visob_test.csv"))
    assert len(test) == 13868 and test.relpath.is_unique and test.subject.nunique() == 300
    assert test.subject.between(1141, 3279).sum() + (test.subject == 7000).sum() == len(test)
    assert np.all(test.relpath.str.endswith("_c.png"))
