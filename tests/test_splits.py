"""The data split must not depend on the order the disk lists files in; the cross-session split must
keep near-duplicate buckets whole; class indices must be the same in every split."""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

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


# ------------------------------------------------------------ new protocols

CONDITIONS = ("iphone_day_light_short", "iphone_office_short", "oppo_dim_light_short", "samsung_office_short")
BUCKET = ["subject", "condition", "eye", "phase"]


def _bucket_manifest():
    """1141, 1142 (8 enrollment buckets each), 1143 (one bucket), 7000 (two), and 1000, 4000 outside the 300."""
    rows = []

    def add(subject, condition, eye, phase, frames):
        for idx in range(1001, 1001 + frames):
            rows.append({"relpath": f"{condition}/{phase}_{eye}_eye/{subject}/{subject}_{idx}_{eye[0]}_c.png",
                         "subject": subject, "idx": idx, "eye": eye, "phase": phase, "condition": condition})

    for subject in (1141, 1142, 1000, 4000):
        for condition in CONDITIONS:
            for eye in ("left", "right"):
                add(subject, condition, eye, "enrollment", 4)
                add(subject, condition, eye, "verification", 2)
    add(1143, CONDITIONS[0], "left", "enrollment", 5)
    add(1143, CONDITIONS[0], "left", "verification", 3)
    add(7000, CONDITIONS[1], "right", "enrollment", 3)
    add(7000, CONDITIONS[2], "right", "enrollment", 3)
    add(7000, CONDITIONS[1], "right", "verification", 2)
    return pd.DataFrame(rows)


def _classes(m):
    return m[split_script.in_paper_classes(m.subject)]


def test_cross_session_keeps_buckets_whole_and_tests_on_verification():
    m = _classes(_bucket_manifest())
    a = split_script.assign_cross_session(m, seed=42)
    pd.testing.assert_frame_equal(a, split_script.assign_cross_session(m.sample(frac=1.0, random_state=3), seed=42))
    assert set(a.phase[a.split == "test"]) == {"verification"} and set(a.split[a.phase == "verification"]) == {"test"}
    assert a.groupby(BUCKET).split.nunique().max() == 1                      # no bucket is split
    train = set(map(tuple, a[a.split == "train"][BUCKET].values))
    assert not train & set(map(tuple, a[a.split == "validation"][BUCKET].values))
    counts = a[a.split != "test"].groupby("subject").split.value_counts().unstack(fill_value=0)
    assert (counts.loc[[1141, 1142, 7000]] > 0).all().all()                  # at least 1 train and 1 validation
    assert counts.loc[1143, "train"] == 5 and counts.loc[1143, "validation"] == 0   # a single bucket stays in train
    val_buckets = a[a.split == "validation"].drop_duplicates(BUCKET).groupby("subject").size()
    assert val_buckets.to_dict() == {1141: 2, 1142: 2, 7000: 1}              # 25 % of 8 buckets; 1 of 2
    half = split_script.assign_cross_session(m, seed=42, val_frac=0.5)
    assert half[half.split == "validation"].drop_duplicates(BUCKET).groupby("subject").size().loc[1141] == 4


def test_cross_session_rejects_unknown_phases():
    m = _classes(_bucket_manifest()).copy()
    m.loc[m.index[0], "phase"] = "other"
    with pytest.raises(ValueError, match="phase"):
        split_script.assign_cross_session(m)


def test_open_set_is_disjoint_from_the_training_classes():
    m = _bucket_manifest()
    o = split_script.select_open_set(m.sample(frac=1.0, random_state=0))
    assert set(o.subject) == {1000, 4000} and not split_script.in_paper_classes(o.subject).any()
    assert set(o.phase) == {"enrollment", "verification"} and (o.split == "test").all()
    assert o.relpath.tolist() == sorted(o.relpath) and len(o) == m.subject.isin([1000, 4000]).sum()


def _run(monkeypatch, *args):
    """Run 02's main() in-process; symlinks become small files (Windows needs a privilege for them)."""
    def fake_symlink(src, dst):
        with open(dst, "w") as f:
            f.write(src)
    monkeypatch.setattr(os, "symlink", fake_symlink)
    monkeypatch.setattr(sys, "argv", ["02_make_splits.py", *map(str, args)])
    split_script.main()


def test_random_protocol_cli_is_unchanged(tmp_path, monkeypatch):
    m = _bucket_manifest()
    m.to_csv(tmp_path / "manifest_raw.csv", index=False)
    _classes(m).groupby("subject").head(2)[["relpath", "subject"]].to_csv(tmp_path / "pinned.csv", index=False)
    common = ["--manifest", tmp_path / "manifest_raw.csv", "--root", tmp_path / "data", "--out-dir", tmp_path / "tree",
              "--test-list", tmp_path / "pinned.csv"]
    _run(monkeypatch, *common, "--out-manifest", tmp_path / "default.csv")
    _run(monkeypatch, *common, "--out-manifest", tmp_path / "random.csv", "--protocol", "random")
    a, b = pd.read_csv(tmp_path / "default.csv"), pd.read_csv(tmp_path / "random.csv")
    pd.testing.assert_frame_equal(a, b)
    expected = split_script.assign_splits(_classes(m), 42, pd.read_csv(tmp_path / "pinned.csv").relpath)
    pd.testing.assert_frame_equal(a[["relpath", "subject", "split"]], expected[["relpath", "subject", "split"]])
    first = a.iloc[0]
    assert first.link_path == os.path.join(str(tmp_path / "tree"), first.split, str(first.subject),
                                           f"{first.condition}__{first.phase}__{os.path.basename(first.relpath)}")
    assert open(first.link_path).read() == os.path.join(str(tmp_path / "data"), first.relpath)
    assert split_script.OUTPUTS["random"] == ("/tmp/ocular_recognition", "/kaggle/working/manifest_split.csv")


def test_cross_session_and_open_set_cli(tmp_path, monkeypatch, capsys):
    _bucket_manifest().to_csv(tmp_path / "manifest_raw.csv", index=False)
    common = ["--manifest", tmp_path / "manifest_raw.csv", "--root", tmp_path / "data",
              "--write-split-csvs", tmp_path / "splits"]
    _run(monkeypatch, *common, "--protocol", "cross_session", "--out-dir", tmp_path / "cs",
         "--out-manifest", tmp_path / "cs.csv")
    out = capsys.readouterr().out
    assert "buckets shared by train and validation: 0" in out
    assert "1 classes have no validation images" in out and "[1143]" in out
    assert sorted(os.listdir(tmp_path / "cs")) == ["test", "train", "validation"]
    lists = {s: pd.read_csv(tmp_path / "splits" / f"cross_session_{s}.csv") for s in ("train", "validation", "test")}
    assert all(list(df.columns) == ["relpath", "subject"] and df.relpath.is_monotonic_increasing
               for df in lists.values())
    assert not set(lists["train"].relpath) & set(lists["validation"].relpath)
    assert lists["test"].relpath.str.contains("/verification_").all()

    _run(monkeypatch, *common, "--protocol", "open_set", "--out-dir", tmp_path / "no_tree",
         "--out-manifest", tmp_path / "open.csv")
    assert not (tmp_path / "no_tree").exists()                                # nothing to train: no tree
    open_set = pd.read_csv(tmp_path / "splits" / "open_set_test.csv")
    assert set(open_set.subject) == {1000, 4000}
    assert not (tmp_path / "splits" / "open_set_train.csv").exists()
    assert pd.read_csv(tmp_path / "open.csv").link_path.str.startswith(str(tmp_path / "data")).all()


def test_max_frame_leaves_out_late_frames(tmp_path, monkeypatch, capsys):
    m = _bucket_manifest()
    m.to_csv(tmp_path / "manifest_raw.csv", index=False)
    common = ["--manifest", tmp_path / "manifest_raw.csv", "--root", tmp_path / "data", "--max-frame", 1002]
    _run(monkeypatch, *common, "--protocol", "cross_session", "--out-dir", tmp_path / "cs",
         "--out-manifest", tmp_path / "cs.csv")
    cs = pd.read_csv(tmp_path / "cs.csv")
    paper = m[split_script.in_paper_classes(m.subject)]
    assert cs.idx.max() == 1002 and len(cs) == int((paper.idx <= 1002).sum())
    assert set(cs.split) == {"train", "validation", "test"}
    late = m[m.idx > 1002]                                # counted on the whole manifest, before the classes
    assert (f"left out {int((late.phase == 'enrollment').sum())} enrollment, "
            f"{int((late.phase == 'verification').sum())} verification") in capsys.readouterr().out
    _run(monkeypatch, *common, "--protocol", "open_set", "--out-manifest", tmp_path / "open.csv")
    assert pd.read_csv(tmp_path / "open.csv").idx.max() == 1002
    with pytest.raises(SystemExit):                       # the pinned random test list holds late frames
        _run(monkeypatch, *common, "--protocol", "random", "--out-dir", tmp_path / "r")
    kept, dropped = split_script.drop_late_frames(m.drop(columns="idx"), 1002)   # no idx column: parsed
    assert len(kept) == int((m.idx <= 1002).sum()) and sum(dropped.values()) == int((m.idx > 1002).sum())


def test_val_frac_must_be_a_fraction(tmp_path, monkeypatch):
    _bucket_manifest().to_csv(tmp_path / "manifest_raw.csv", index=False)
    with pytest.raises(SystemExit):                        # 25 meant as 25 %: all but one bucket would validate
        _run(monkeypatch, "--manifest", tmp_path / "manifest_raw.csv", "--root", tmp_path / "data",
             "--protocol", "cross_session", "--val-frac", "25", "--out-dir", tmp_path / "cs",
             "--out-manifest", tmp_path / "cs.csv")
    assert not (tmp_path / "cs").exists()


# ----------------------------------------------- 03: class indices per split

@pytest.fixture(scope="module")
def train_script():
    return _load("03_train_baseline")


def _class_tree(root, layout):
    """layout: split -> {class folder: number of images}; 4 x 4 PNGs named like 02's links."""
    for split, classes in layout.items():
        for name, n in classes.items():
            d = root / split / name
            d.mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (4, 4), (40 * i, 0, 0)).save(d / f"cond__enrollment__{name}_{1001 + i}_l_c.png")
    return root


def test_explicit_class_list_reproduces_the_default_indices(tmp_path, train_script):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    names = {"1141": 2, "1142": 1, "7000": 2, "999": 1}       # '999' sorts last as a string, as Keras sorts it
    root = _class_tree(tmp_path, {split: names for split in ("train", "validation", "test")})
    classes = train_script.class_list(str(root))
    assert classes == ["1141", "1142", "7000", "999"]
    gen = ImageDataGenerator()
    for split in ("train", "validation", "test"):
        default = gen.flow_from_directory(str(root / split), target_size=(8, 8), shuffle=False)
        explicit = gen.flow_from_directory(str(root / split), target_size=(8, 8), shuffle=False, classes=classes)
        assert explicit.class_indices == default.class_indices
        np.testing.assert_array_equal(explicit.classes, default.classes)
        assert explicit.filenames == default.filenames


def test_class_indices_survive_a_missing_class(tmp_path, train_script):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    full = {"1141": 2, "1142": 1, "7000": 2}
    root = _class_tree(tmp_path, {"train": full, "validation": {"1141": 1, "7000": 1}, "test": full})
    classes = train_script.class_list(str(root))
    plain = ImageDataGenerator(rescale=1.0 / 255)
    train = train_script.ClassListDataGen(str(root / "train"), plain, classes, batch_size=8, target_size=(8, 8))
    val = train_script.ClassListDataGen(str(root / "validation"), plain, classes, batch_size=4, target_size=(8, 8))
    assert val.generator.class_indices == train.generator.class_indices == {"1141": 0, "1142": 1, "7000": 2}
    (x, y), _ = val[0]
    assert x.shape == (2, 8, 8, 3) and y.shape == (2, 3) and sorted(y.argmax(1)) == [0, 2]
    # without the list, validation's 7000 would take index 1, which is 1142 in train
    assert plain.flow_from_directory(str(root / "validation"), target_size=(8, 8)).class_indices["7000"] == 1

    missing = {s: train_script.missing_classes(g.generator, classes)
               for s, g in (("train", train), ("validation", val))}
    assert missing == {"train": [], "validation": ["1142"]}
    assert train_script.check_coverage(missing) == ["WARNING: 1 classes have no validation images, e.g. ['1142']"]
    with pytest.raises(SystemExit, match="no training images"):
        train_script.check_coverage({"train": ["1142"], "validation": []})
    assert train_script.images_by_phase(train.generator.filenames) == {"enrollment": 5}
    assert train_script.images_by_phase(["1141/plain_name.png"]) == {"unknown": 1}


def test_train_baseline_end_to_end(tmp_path, monkeypatch, capsys, train_script):
    """03's main() on a tree whose validation split lacks a class, with the tiny model instead of Xception."""
    import tensorflow as tf
    from conftest import build_tiny_model

    def tiny_model(cfg):
        model = build_tiny_model(cfg["model"]["n_classes"])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy())
        return model

    root = _class_tree(tmp_path / "tree", {"train": {"1141": 3, "1142": 3, "7000": 3},
                                           "validation": {"1141": 1, "7000": 1},
                                           "test": {"1141": 1, "1142": 1, "7000": 1}})
    monkeypatch.setattr(train_script, "build_model", tiny_model)
    monkeypatch.setattr(sys, "argv", ["03_train_baseline.py", "--split-dir", str(root), "--n-classes", "3",
                                      "--epochs", "1", "--batch-size", "4", "--verbose", "0", "--tag", "t",
                                      "--out-dir", str(tmp_path / "models")])
    train_script.main()
    assert "WARNING: 1 classes have no validation images, e.g. ['1142']" in capsys.readouterr().out
    out = tmp_path / "models"
    assert json.loads((out / "class_index_t.json").read_text()) == {"1141": 0, "1142": 1, "7000": 2}
    result = json.loads((out / "results_t.json").read_text())
    assert result["images_by_phase"]["train"] == {"enrollment": 9}
    assert result["class_index"].endswith("class_index_t.json")
    assert (out / "Xception_ArcFace_t.keras").exists()

    (root / "test").rename(root / "elsewhere")                                # a missing split fails at once
    with pytest.raises(SystemExit, match="not found"):
        train_script.main()
