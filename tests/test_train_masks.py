"""scripts/21_make_train_masks.py: 64 x 64 training masks from a mask file of 17 or straight from the
segmenter (a colour stand-in network, as in test_seg_masks.py), and the part-presence report."""
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from src.data import masked_generator as mg
from src.models import anatomy_model as am
from src.xai import seg_masks as sm
from test_seg_masks import colour_net, fake_visob


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load("21_make_train_masks")
GROUPS = am.DEFAULT_PART_GROUPS
CLASSES = list(sm.CLASS_NAMES)


def test_part_presence_counts_images_over_each_threshold():
    labels = np.zeros((4, 64, 64), np.uint8)                       # 4096 pixels per image
    labels[0, :10, :10] = CLASSES.index("brow")                    # 100 px = 2.44 %
    labels[1, :4, :4] = CLASSES.index("brow")                      # 16 px = 0.39 %
    labels[2, :20, :20] = CLASSES.index("brow")                    # 400 px = 9.8 %
    labels[3, :, :32] = sm.IGNORE                                  # half unknown, no brow
    labels[3, :8, 32:40] = CLASSES.index("caruncle")               # 64 px -> sclera group = 1.56 %
    table = script.part_presence(labels, CLASSES, GROUPS, ["train", "train", "test", "test"])
    assert list(table.columns) == ["split", "group", "n_images", "mean_area", "share_ge_0.5pct", "share_ge_2pct",
                                   "share_ge_5pct"]
    assert table.split.unique().tolist() == ["test", "train", "all"]
    brow = table[table.group == "brow"].set_index("split")
    assert brow.loc["train", "n_images"] == 2 and brow.loc["all", "n_images"] == 4
    assert brow.loc["train", "share_ge_0.5pct"] == 0.5 and brow.loc["train", "share_ge_2pct"] == 0.5
    assert brow.loc["test", "share_ge_5pct"] == 0.5 and brow.loc["all", "share_ge_0.5pct"] == 0.5
    assert brow.loc["all", "mean_area"] == pytest.approx((100 + 16 + 400) / 4096 / 4)
    sclera = table[(table.group == "sclera") & (table.split == "test")].iloc[0]
    assert sclera.mean_area == pytest.approx(64 / 4096 / 2) and sclera["share_ge_0.5pct"] == 0.5
    skin = table[(table.group == "skin") & (table.split == "test")].iloc[0]  # unknown pixels count in the image
    assert skin.mean_area == pytest.approx((1 - 400 / 4096 + 0.5 - 64 / 4096) / 2)
    single = script.part_presence(labels, CLASSES, GROUPS, ["all"] * 4)
    assert single.split.unique().tolist() == ["all"]


def test_convert_a_mask_file_of_17(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    rels = [f"cond/enrollment_left_eye/{s}/{s}_100{i}_l_c.png" for s in (1141, 1142) for i in range(3)]
    full = rng.integers(0, 6, (6, 256, 256)).astype(np.uint8)
    sm.save_mask_file(str(tmp_path / "masks_test.npz"), rels, full, CLASSES,
                      {"weights_sha256": sm.WEIGHTS_SHA256, "model": "periorbital"})
    report = pd.DataFrame({"relpath": rels, "split": ["train"] * 3 + ["test"] * 3})
    report.to_csv(tmp_path / "report.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["21_make_train_masks.py", "--from-mask-file", str(tmp_path / "masks_test.npz"),
                                      "--report-csv", str(tmp_path / "report.csv"), "--out-dir", str(out),
                                      "--tag", "t"])
    script.main()
    printed = capsys.readouterr().out
    assert "part presence" in printed and "brow covers >= 0.5 % of the image" in printed

    small = mg.load_label_source(str(out / "masks64_t.npz"))
    assert small["source_size"] == 64 and small["relpaths"] == rels and small["class_names"] == CLASSES
    np.testing.assert_array_equal(small["labels"], mg.labels_to_64(full))
    meta = small["meta"]
    assert meta["weights_sha256"] == sm.WEIGHTS_SHA256 and meta["source_meta"]["model"] == "periorbital"
    assert "converted from" in meta["source"] and "code_commit" in meta and meta["created"]

    presence = pd.read_csv(out / "part_presence_t.csv")
    assert set(presence.split) == {"train", "test", "all"} and set(presence.group) == set(GROUPS)
    expected = script.part_presence(small["labels"], CLASSES, GROUPS, report.split)
    pd.testing.assert_frame_equal(presence, expected, check_dtype=False)
    saved = json.loads((out / "part_presence_t.json").read_text())
    assert saved["thresholds"] == [0.005, 0.02, 0.05] and len(saved["rows"]) == len(presence)
    summary = json.loads((out / "masks64_summary_t.json").read_text())
    assert summary["n_images"] == 6 and summary["images_per_split"] == {"train": 3, "test": 3}
    assert summary["weights_sha256"] == sm.WEIGHTS_SHA256 and "code_commit" in summary

    monkeypatch.setattr(sys, "argv", ["21_make_train_masks.py", "--from-mask-file", str(tmp_path / "masks_test.npz"),
                                      "--max-images", "2", "--out-dir", str(out), "--tag", "smoke"])
    script.main()                                                  # smoke run, no splits: one split "all"
    assert len(mg.load_label_source(str(out / "masks64_smoke.npz"))["relpaths"]) == 2
    assert set(pd.read_csv(out / "part_presence_smoke.csv").split) == {"all"}


def test_segment_images_with_17s_pipeline(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    rels = fake_visob(root, n_subjects=8)
    manifest = pd.DataFrame({"relpath": rels, "split": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2})
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    weights = tmp_path / "combine.pth"
    weights.write_bytes(b"stand-in")
    monkeypatch.setattr(sm, "require_torch", lambda: types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(sm, "load_periorbital_model", lambda path, device: ("network", path, device))
    monkeypatch.setattr(sm, "torch_predictor", lambda model, device, amp: colour_net)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["21_make_train_masks.py", "--csv", str(tmp_path / "manifest.csv"),
                                      "--splits", "train,test", "--data-root", str(root), "--weights", str(weights),
                                      "--batch-size", "3", "--out-dir", str(out), "--tag", "seg"])
    script.main()
    printed = capsys.readouterr().out
    assert "images: 6 of 6" in printed and "min left" in printed

    small = mg.load_label_source(str(out / "masks64_seg.npz"))
    kept = manifest[manifest.split.isin(["train", "test"])]
    assert small["relpaths"] == kept.relpath.tolist()
    natives = [np.asarray(Image.open(root / r).convert("RGB")) for r in kept.relpath]
    want = mg.labels_to_64(np.stack([sm.to_model_geometry(lab) for lab in sm.segment(natives, colour_net)]))
    np.testing.assert_array_equal(small["labels"], want)
    assert (small["labels"] == CLASSES.index("brow")).any() and small["meta"]["weights_sha256"] == sm.WEIGHTS_SHA256
    presence = pd.read_csv(out / "part_presence_seg.csv")
    assert set(presence.split) == {"train", "test", "all"}
    assert presence[(presence.group == "iris") & (presence.split == "all")]["share_ge_0.5pct"].iloc[0] == 1.0

    monkeypatch.setattr(sys, "argv", ["21_make_train_masks.py", "--csv", str(tmp_path / "manifest.csv"),
                                      "--data-root", str(root), "--weights", str(weights), "--max-images", "3",
                                      "--out-dir", str(out), "--tag", "smoke"])
    script.main()
    printed = capsys.readouterr().out
    assert "images: 3 of 8" in printed and "estimate for all 8 input images" in printed
    assert len(mg.load_label_source(str(out / "masks64_smoke.npz"))["relpaths"]) == 3


def test_split_lookup_from_a_split_tree(tmp_path):
    for split, name in (("train", "cond__enrollment__1141_1001_l_c.png"), ("test", "cond__verification__1141_1001_l_c.png")):
        d = tmp_path / split / "1141"
        d.mkdir(parents=True)
        (d / name).write_bytes(b"")
    keys = script.split_lookup(str(tmp_path), ["train", "test"])
    rels = ["cond/enrollment_left_eye/1141/1141_1001_l_c.png", "cond/verification_left_eye/1141/1141_1001_l_c.png",
            "cond/enrollment_left_eye/1141/1141_1002_l_c.png"]
    assert script.splits_of(rels, keys) == ["train", "test", "unknown"]
    assert script.splits_of(rels, {}) == ["all"] * 3
