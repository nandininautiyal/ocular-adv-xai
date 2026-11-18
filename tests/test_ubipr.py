"""Tests of scripts/20_prepare_ubipr.py (annotation parsing, crop, identity links, outputs) and of the
zero-shot cross-dataset mode x1 of scripts/15_eval_verification.py, on synthetic UBIPr-like files only:
tiny drawn eyes and annotations written in the real files' format (no real images)."""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import REPO
from test_verification import _fake_visob

SCRIPT = os.path.join(REPO, "scripts", "20_prepare_ubipr.py")


def _prep():
    spec = importlib.util.spec_from_file_location("prep20", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _prep()


def annotation(side, size, corner_out, iris, corner_in, brows=None, cam=8, date="2007_12_11", file_name="IMG_1028.JPG",
               gender="Male", glasses="No", pose=0, gaze=0, crlf=False):
    """An annotation in UBIPr's format: header with a leading space, one 'value;' per line, Size as three
    lines (2912, 4368, 3), coordinates as (row, column), EyeSizeX / EyeSizeY = image height / width.
    Points are given as (x, y) = (column, row)."""
    w, h = size
    brows = brows or [(corner_out[0], max(corner_out[1] - 20, 0)), (iris[0], max(iris[1] - 25, 0)),
                      (corner_in[0], max(corner_in[1] - 20, 0))]
    names = list(prep.COMMON_FIELDS) + [side + f for f in prep.EYE_FIELDS]
    values = [f"C:\\Users\\someone\\Periocular dataset UBI\\original\\{date}\\", file_name, "2912", "4368", "3", gender,
              str(cam), str(pose), str(gaze), "Light", "No", "No", glasses]
    for x, y in [corner_out, iris, corner_in] + list(brows):
        values += [str(y), str(x)]
    values += [str(h), str(w)]
    newline = "\r\n" if crlf else "\n"
    return " " + ";".join(names) + ";" + newline + newline.join(f" {v};" for v in values)


def eye_image(size, corner_out, iris, corner_in, colour):
    """A tiny drawn eye: skin, an elliptic sclera between the corners, a coloured iris disk."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float32) + np.array([0.75, 0.55, 0.45])
    cx, cy = (corner_out[0] + corner_in[0]) / 2, (corner_out[1] + corner_in[1]) / 2
    a = abs(corner_out[0] - corner_in[0]) / 2
    img[((xx - cx) / a) ** 2 + ((yy - cy) / (0.4 * a)) ** 2 < 1] = 0.95
    img[(xx - iris[0]) ** 2 + (yy - iris[1]) ** 2 < (0.35 * a) ** 2] = colour
    return Image.fromarray((img * 255).round().astype(np.uint8))


def jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ------------------------------------------------------------------ parsing

def test_parse_annotation_aligns_size_and_swaps_row_column():
    text = annotation("R", (120, 96), corner_out=(90, 50), iris=(60, 46), corner_in=(30, 52), cam=6, gaze=-30,
                      glasses="Yes", gender="Female", date="2008_01_15")
    ann = prep.parse_annotation(text)
    assert ann["source_size"] == (2912, 4368, 3) and ann["side"] == "R"
    assert ann["eye_size"] == (96, 120)                                   # (height, width)
    assert ann["corner_out"] == (90, 50) and ann["iris"] == (60, 46) and ann["corner_in"] == (30, 52)
    assert (ann["cam_distance"], ann["gaze"], ann["pose"]) == (6, -30, 0)
    assert (ann["glasses"], ann["gender"], ann["pigmentation"], ann["eye_closure"]) == ("Yes", "Female", "Light", "No")
    assert ann["capture_date"] == "2008_01_15" and ann["file_name"] == "IMG_1028.JPG"
    left = prep.parse_annotation(annotation("L", (120, 96), (30, 50), (60, 46), (90, 52), crlf=True))
    assert left["side"] == "L" and left["corner_in"] == (90, 52)


def test_parse_annotation_rejects_malformed_files():
    good = annotation("L", (120, 96), (30, 50), (60, 46), (90, 52))
    with pytest.raises(ValueError, match="empty"):
        prep.parse_annotation("")
    with pytest.raises(ValueError, match="values for"):
        prep.parse_annotation(good.rsplit("\n", 4)[0])                    # 4 values missing
    with pytest.raises(ValueError, match="all L"):
        prep.parse_annotation(good.replace("LGlass", "RGlass", 1))        # mixed sides
    with pytest.raises(ValueError, match="not an integer"):
        prep.parse_annotation(good.replace(" 8;", " eight;", 1))          # CamDist
    with pytest.raises(ValueError, match="header"):
        prep.parse_annotation(good.replace("CamDist", "Distance", 1))


def test_check_annotation():
    ann = prep.parse_annotation(annotation("L", (120, 96), (30, 50), (60, 46), (90, 52)))
    assert prep.check_annotation(ann, 120, 96) == (None, [])
    assert "eye size" in prep.check_annotation(ann, 96, 120)[0]          # width and height swapped
    out = prep.parse_annotation(annotation("L", (120, 96), (30, 50), (60, 46), (130, 52)))
    assert prep.check_annotation(out, 120, 96)[0].startswith("corner_in")
    close = prep.parse_annotation(annotation("L", (120, 96), (55, 50), (60, 46), (65, 52)))
    assert "closer than" in prep.check_annotation(close, 120, 96)[0]
    swapped = prep.parse_annotation(annotation("R", (120, 96), (30, 50), (60, 46), (90, 52),
                                               brows=[(30, -1), (60, 20), (90, 30)]))
    assert prep.check_annotation(swapped, 120, 96) == (None, ["eyebrow_outside", "corner_order_swapped"])


# --------------------------------------------------------------------- crop

def test_crop_box_frames_the_corners_like_visob():
    x0, y0, w, h = prep.crop_box((100, 80), (200, 80), k=1.3, anchor_y=0.45)
    assert (w, h) == (130, 87) and x0 == 85 and y0 == round(80 - 0.45 * 130 / 1.5)
    assert prep.crop_box((200, 80), (100, 80)) == prep.crop_box((100, 80), (200, 80))   # corner order is irrelevant


def test_crop_image_replicates_the_border_and_resizes_to_visob():
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :40] = (255, 0, 0)
    img[:, 40:] = (0, 0, 255)
    crop, padded = prep.crop_image(img, (-20, -10, 80, 40))               # reaches over the left / top border
    assert padded and crop.shape == (160, 240, 3) and crop.dtype == np.uint8
    assert (crop[:, :170] == (255, 0, 0)).all()                           # replicated red, no black padding
    assert (crop[:, 190:] == (0, 0, 255)).all()                           # x 40-59: blue from 180 on
    assert not prep.crop_image(img, (10, 10, 30, 20))[1]
    marker = np.zeros((96, 120, 3), np.uint8)
    marker[49:52, 59:62] = 255                                            # the corner midpoint (60, 50)
    box = prep.crop_box((30, 50), (90, 50), k=1.3, anchor_y=0.45)
    crop = prep.crop_image(marker, box)[0].astype(float).sum(axis=2)
    yy, xx = np.nonzero(crop > 0.5 * crop.max())
    assert abs(xx.mean() - 120) < 6 and abs(yy.mean() - 0.45 * 160) < 6


# ----------------------------------------------------------------- identity

def test_find_misfiled_and_person_links():
    # persons 1/2 (frames 100-102 and 300-301) and 3/4 (frames 200-202) on one date; C1 at frame 201 shares
    # 3/4's photograph (misfiled), C3 at frame 500 is nobody's run (misfiled), C2_S2 on another date is its
    # own run with its partner (kept), and a 'c' retake of frame 102 counts as frame 102
    rows = [(1, "d1", "IMG_100.JPG"), (2, "d1", "IMG_100.JPG"), (1, "d1", "IMG_101.JPG"), (2, "d1", "IMG_101.JPG"),
            (1, "d1", "IMG_102c2.JPG"), (2, "d1", "IMG_102c2.JPG"),
            (3, "d1", "IMG_200.JPG"), (4, "d1", "IMG_200.JPG"), (3, "d1", "IMG_202.JPG"), (4, "d1", "IMG_202.JPG"),
            (4, "d1", "IMG_201.JPG"), (1, "d1", "IMG_201.JPG"), (3, "d1", "IMG_500.JPG"),
            (1, "d2", "IMG_300.JPG"), (2, "d2", "IMG_300.JPG"), (2, "d2", "IMG_301.JPG"), (2, "d2", "IMG_x.JPG")]
    df = pd.DataFrame(rows, columns=["class_id", "capture_date", "file_name"])
    misfiled = prep.find_misfiled(df)
    assert df.index[misfiled].tolist() == [11, 12]                       # C1 at 201, C3 at 500; IMG_x: no frame
    assert prep.frame_number("IMG_102c2.JPG") == 102 and prep.frame_number("DSC_1.JPG") is None
    person, cross = prep.link_persons(df)
    assert cross == [(1, 4, "d1/IMG_201.JPG")] and set(person) == {1}   # the misfiled photograph joins 1/2 and 3/4
    person, cross = prep.link_persons(df, ignore=misfiled)
    assert cross == [] and dict(zip(df.class_id, person)) == {1: 1, 2: 1, 3: 3, 4: 3}
    assert not prep.find_misfiled(df, window=400).any()                  # a wider window accepts every frame


# --------------------------------------------------------------- the script

# (class, session, image) -> capture date, IMG file, camera distance, glasses. Classes 2k-1 (L) and 2k (R) share
# their photographs; C5_S2_I1 shares one with C7_S1_I1 inside the capture run of 7/8 (frames 521-532 of 2008_01_15:
# one person under two IDs), which joins persons 5/6 and 7/8.
SIZE = {4: (150, 120), 8: (120, 96)}
COLOURS = {1: (0.2, 0.3, 0.8), 3: (0.6, 0.3, 0.1), 5: (0.1, 0.6, 0.2), 7: (0.5, 0.1, 0.5)}


def _records():
    recs = []
    for person in (1, 3, 5, 7):
        for side, cls in (("L", person), ("R", person + 1)):
            sessions = (1,) if person == 7 else (1, 2)
            for session in sessions:
                for idx in (1, 2):
                    cam = 4 if idx == 1 else 8
                    date = f"2007_12_1{session}" if session == 1 and person != 7 else "2008_01_15"
                    file_name = f"IMG_53{idx}.JPG" if person == 7 else f"IMG_{person}{session}{idx}.JPG"
                    recs.append({"cls": cls, "side": side, "session": session, "idx": idx, "cam": cam, "date": date,
                                 "file": file_name, "colour": COLOURS[person],
                                 "glasses": "Yes" if person == 3 else "No"})
    for r in recs:
        if (r["cls"], r["session"], r["idx"]) == (7, 1, 1):
            r["date"], r["file"] = "2008_01_15", "IMG_521.JPG"           # the photograph of C5_S2_I1
    return recs


def _write_source(write):
    """Calls write(name, bytes) for every file of a synthetic archive; returns the expected skips."""
    for r in _records():
        w, h = SIZE[r["cam"]]
        y = h // 2
        out, inn = ((w // 4, y), (3 * w // 4, y)) if r["side"] == "L" else ((3 * w // 4, y), (w // 4, y))
        iris = (w // 2 + 3, y - 2)
        stem = f"C{r['cls']}_S{r['session']}_I{r['idx']}"
        write(f"UBIPeriocular/{stem}.jpg", jpeg_bytes(eye_image((w, h), out, iris, inn, r["colour"])))
        write(f"UBIPeriocular/{stem}.txt", annotation(r["side"], (w, h), out, iris, inn, cam=r["cam"], date=r["date"],
                                                      file_name=r["file"], glasses=r["glasses"]).encode("latin-1"))
        write(f"UBIPeriocular/._{stem}.txt", b"\x00\x05\x16\x07 macOS resource fork")
    w, h = SIZE[8]
    eye = eye_image((w, h), (30, 48), (60, 46), (90, 48), COLOURS[3])
    # C3_S1_I3: the header says R in an L class (side_conflict, unusable); C4_S1_I3: a corner outside the image;
    # C6_S1_I3: no annotation; C6_S1_I4: taken on the date of C6's session-2 images (unusable); C2_S2_I3: a frame far
    # outside every capture run of classes 1 / 2 (misfiled: another person's photograph; unusable)
    extra = {"C3_S1_I3": annotation("R", (w, h), (90, 48), (60, 46), (30, 48), date="2007_12_11",
                                    file_name="IMG_313.JPG"),
             "C4_S1_I3": annotation("R", (w, h), (90, 48), (60, 46), (300, 48), file_name="IMG_901.JPG"),
             "C6_S1_I3": None,
             "C6_S1_I4": annotation("R", (w, h), (90, 48), (60, 46), (30, 48), date="2008_01_15",
                                    file_name="IMG_523.JPG"),
             "C2_S2_I3": annotation("R", (w, h), (90, 48), (60, 46), (30, 48), date="2007_12_17",
                                    file_name="IMG_3492.JPG")}
    for stem, text in extra.items():
        write(f"UBIPeriocular/{stem}.jpg", jpeg_bytes(eye))
        if text is not None:
            write(f"UBIPeriocular/{stem}.txt", text.encode("latin-1"))
    write("UBIPeriocular/readme.pdf", b"%PDF-1.4 not an image")
    return {"C4_S1_I3": "corner_in outside the image", "C6_S1_I3": "no annotation"}


def make_tar(path):
    with tarfile.open(path, "w") as tar:
        def write(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        expected = _write_source(write)
    return expected


def make_folder(root):
    def write(name, data):
        path = os.path.join(root, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return _write_source(write)


def run_prepare(source, out, *extra):
    cmd = [sys.executable, SCRIPT, "--source", str(source), "--out-dir", str(out), "--sheet-crops", "6", *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    return pd.read_csv(out / "ubipr_manifest.csv", keep_default_na=False), \
        json.loads((out / "ubipr_summary.json").read_text())


def test_prepare_from_tar_and_folder(tmp_path):
    expected_skips = make_tar(tmp_path / "ubipr.tar")
    visob = tmp_path / "visob"
    visob.mkdir()
    Image.new("RGB", (240, 160), (90, 60, 50)).save(visob / "a.png")
    df, summary = run_prepare(tmp_path / "ubipr.tar", tmp_path / "out", "--visob-examples", str(visob))
    n_regular = len(_records())
    assert len(df) == n_regular + 3 and summary["n_skipped"] == 2
    assert summary["skipped_by_reason"] == {"corner_in outside the image": 1, "no annotation": 1}
    skipped = pd.read_csv(tmp_path / "out" / "ubipr_skipped.csv")
    assert dict(zip(skipped.name, skipped.reason.str.split(" ").str[0])) == {"C4_S1_I3": "corner_in", "C6_S1_I3": "no"}
    assert set(expected_skips) == set(skipped.name)
    assert summary["files"]["ignored"] == {"macOS resource file ('._*')": n_regular,
                                           "not C<class>_S<session>_I<n>.jpg/.txt": 1}
    # identity: 2k-1 / 2k share photographs; C5_S2_I1 ~ C7_S1_I1 joins persons 5 and 7
    assert dict(zip(df.class_id, df.person_id)) == {1: 1, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 5}
    assert summary["cross_person_links"] == ["C5 ~ C7 via 2008_01_15/IMG_521.JPG"]
    assert summary["cross_person_photos_incl_misfiled"] == summary["cross_person_links"]
    assert summary["misfiled"] == ["crops/C2_S2_I3.png"] and summary["frame_window"] == 15
    assert summary["n_persons"] == 3 and summary["classes_per_person"] == {"2": 2, "4": 1}
    assert summary["odd_classes_are_L"] and summary["even_classes_are_R"]
    assert (df.eye == np.where(df.class_id % 2 == 1, "L", "R")).all()
    flagged = df.set_index("relpath")
    assert flagged.loc["crops/C3_S1_I3.png", "flags"] == "side_conflict"
    assert flagged.loc["crops/C3_S1_I3.png", "header_side"] == "R" and flagged.loc["crops/C3_S1_I3.png", "eye"] == "L"
    assert flagged.loc["crops/C6_S1_I4.png", "flags"] == "session1_on_session2_date"
    assert flagged.loc["crops/C2_S2_I3.png", "flags"] == "misfiled"
    assert df.usable.sum() == n_regular and set(df.relpath[df.usable == 0]) == {
        "crops/C3_S1_I3.png", "crops/C6_S1_I4.png", "crops/C2_S2_I3.png"}
    assert summary["flags"] == {"side_conflict": 1, "session1_on_session2_date": 1, "misfiled": 1}
    # columns, landmarks (x = column, y = row) and crops
    for col in ("relpath", "class_id", "person_id", "eye", "session", "cam_distance", "pose", "gaze", "pigmentation",
                "eye_closure", "hair_occlusion", "glasses", "gender", "corner_out_x", "corner_out_y", "iris_x",
                "iris_y", "corner_in_x", "corner_in_y", "brow_mid_y", "crop_x0", "crop_y0", "crop_w", "crop_h",
                "source_w", "source_h", "capture_date", "file_name", "flags", "usable"):
        assert col in df.columns, col
    row = flagged.loc["crops/C1_S1_I1.png"]                               # 150 x 120, L: outer corner on the left
    assert (row.source_w, row.source_h, row.corner_out_x, row.corner_out_y, row.corner_in_x) == (150, 120, 37, 60, 112)
    assert row.crop_w == round(1.3 * 75) and row.crop_h == round(1.3 * 75 / 1.5)
    for rel in df.relpath:
        with Image.open(tmp_path / "out" / rel) as im:
            assert im.size == (240, 160) and im.mode == "RGB"
    assert summary["images_by_session"] == {"1": 18, "2": 13} and summary["n_classes_both_sessions"] == 6
    assert summary["x1_protocol"] == {"templates (session 1, usable)": 16, "probes (session 2, usable)": 12,
                                      "classes with both": 6}
    assert summary["crop"]["anchor_y"] == 0.45 and summary["visob_examples_on_sheet"] == 1 and summary["git_commit"]
    with Image.open(tmp_path / "out" / "qc_contact_sheet.png") as sheet:
        assert sheet.size[0] == 8 * 240

    # an extracted folder gives the same manifest
    make_folder(tmp_path / "extracted")
    df2, summary2 = run_prepare(tmp_path / "extracted", tmp_path / "out2")
    pd.testing.assert_frame_equal(df.drop(columns="source"), df2.drop(columns="source"))
    assert summary2["visob_examples_on_sheet"] == 0
    # smoke run: the first N images by class, session and index
    df3, summary3 = run_prepare(tmp_path / "ubipr.tar", tmp_path / "out3", "--max-images", "5")
    assert summary3["n_images_considered"] == 5 and set(df3.class_id) == {1, 2}


# ----------------------------------------------------- 15 --modes x1 (tiny model)

def _run15(tmp_path, ckpt, out, *args):
    cmd = [sys.executable, os.path.join(REPO, "scripts", "15_eval_verification.py"), "--ckpt", str(ckpt),
           "--n-boot", "40", "--batch-size", "16", "--workers", "2", "--sample-scores", "30", "--out-dir", str(out),
           "--tag", "t", *args]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    return (pd.read_csv(out / "verification_metrics_t.csv"), pd.read_csv(out / "verification_scores_t.csv"),
            json.loads((out / "verification_summary_t.json").read_text()))


def test_eval_verification_x1_and_p3_unchanged(tmp_path, model):
    make_tar(tmp_path / "ubipr.tar")
    run_prepare(tmp_path / "ubipr.tar", tmp_path / "ubipr")
    ext = ["--external-manifest", str(tmp_path / "ubipr" / "ubipr_manifest.csv")]
    ckpt = tmp_path / "Xception_ArcFace_tiny.keras"
    model.save(ckpt)
    (tmp_path / "class_index_tiny.json").write_text(json.dumps({"1141": 0, "1142": 1}))

    # x1 alone: no VISOB root is needed (none exists under /kaggle/input here)
    metrics, scores, summary = _run15(tmp_path, ckpt, tmp_path / "x1", "--modes", "x1", *ext, "--save-embeddings")
    assert metrics["mode"].unique().tolist() == ["x1"]
    assert metrics.subset.tolist() == ["all", "cam_distance=4", "cam_distance=8", "glasses=No", "glasses=Yes"]
    overall = metrics.iloc[0]
    # 12 probes (classes 1-6, session 2), 8 templates; per eye 4 templates. Impostors: other persons only, so the
    # probes of class 5 (6) never meet the template of class 7 (8), the same person: 16 per eye, 4 left out
    assert (overall.n_probes, overall.n_templates, overall.n_genuine, overall.n_impostor) == (12, 8, 12, 32)
    assert overall.n_subjects == 3                                        # persons, the bootstrap clusters
    assert overall.rank1_method.endswith("(other classes of the probe's person left out)")
    assert overall.caveat.startswith("zero-shot cross-dataset") and "ubipr" in overall.caveat
    assert metrics.eer.between(0, 1).all() and metrics.rank1.between(0, 1).all()
    by_subset = metrics.set_index("subset")
    assert by_subset.loc["glasses=Yes", "n_probes"] == 4 and by_subset.loc["cam_distance=4", "n_probes"] == 6
    x1 = summary["per_mode"]["x1"]
    assert x1["n_same_person_comparisons_left_out"] == 4 and x1["n_unusable_images_left_out"] == 3
    assert x1["n_persons"] == 3 and x1["n_classes_with_template"] == 8 and x1["n_probes_without_template"] == 0
    assert summary["external"]["name"] == "ubipr" and summary["n_images_embedded"] == 0
    assert summary["data_root"] is None and summary["embedding_dim"] == 32
    imp = scores[scores.kind == "impostor"]
    persons = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 5}
    assert (imp.subject.map(persons) != imp.other_subject.map(persons)).all()
    assert {"person", "cam_distance", "glasses"} <= set(scores.columns) and "device" not in scores.columns
    emb = np.load(tmp_path / "x1" / "embeddings_ubipr_t.npz")
    assert emb["embeddings"].shape == (28, 32)
    assert not (tmp_path / "x1" / "embeddings_t.npz").exists()

    # p3 alone and p3 next to x1: identical p3 rows and score samples; x1 identical to x1 alone
    root = _fake_visob(tmp_path / "VISOB_Augmented_Recent")
    rel = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.png"))
    pd.DataFrame({"relpath": rel}).to_csv(tmp_path / "manifest_raw.csv", index=False)
    visob = ["--manifest", str(tmp_path / "manifest_raw.csv"), "--data-root", str(root)]
    p3_metrics, p3_scores, p3_summary = _run15(tmp_path, ckpt, tmp_path / "p3", "--modes", "p3", *visob)
    both_metrics, both_scores, both_summary = _run15(tmp_path, ckpt, tmp_path / "both", "--modes", "x1,p3", *visob,
                                                     *ext)
    assert both_metrics["mode"].tolist()[:len(p3_metrics)] == ["p3"] * len(p3_metrics)
    pd.testing.assert_frame_equal(both_metrics[both_metrics["mode"] == "p3"].reset_index(drop=True), p3_metrics)
    pd.testing.assert_frame_equal(both_metrics[both_metrics["mode"] == "x1"].reset_index(drop=True), metrics)
    p3_part = both_scores[both_scores["mode"] == "p3"].dropna(axis=1, how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(p3_part, p3_scores, check_dtype=False)
    assert "external" not in p3_summary and set(p3_summary) | {"external"} == set(both_summary)
    assert p3_summary["per_mode"]["p3"]["eer"] == both_summary["per_mode"]["p3"]["eer"]
    assert p3_metrics.subset.tolist() == ["all", "device=iphone", "device=samsung", "lighting=day", "lighting=office"]


def test_eval_verification_x1_alonso2025(tmp_path, model):
    from src.eval.verification import embed
    make_tar(tmp_path / "ubipr.tar")
    _, prep_summary = run_prepare(tmp_path / "ubipr.tar", tmp_path / "ubipr")
    ext = ["--modes", "x1", "--external-manifest", str(tmp_path / "ubipr" / "ubipr_manifest.csv")]
    ckpt = tmp_path / "Xception_ArcFace_tiny.keras"
    model.save(ckpt)

    # ours stays the default: the same metrics file byte for byte, and no new summary keys
    _run15(tmp_path, ckpt, tmp_path / "default", *ext)
    _, _, ours = _run15(tmp_path, ckpt, tmp_path / "ours", *ext, "--x1-protocol", "ours")
    assert (tmp_path / "default" / "verification_metrics_t.csv").read_bytes() == \
        (tmp_path / "ours" / "verification_metrics_t.csv").read_bytes()
    assert (tmp_path / "default" / "verification_scores_t.csv").read_bytes() == \
        (tmp_path / "ours" / "verification_scores_t.csv").read_bytes()
    assert "protocol" not in ours["external"] and "pair_counts" not in ours["per_mode"]["x1"]

    metrics, scores, summary = _run15(tmp_path, ckpt, tmp_path / "alonso", *ext, "--x1-protocol", "alonso2025",
                                      "--save-embeddings")
    # usable crops: classes 1-6 at 8 m (D1) and 4 m (D5) in both sessions, classes 7 / 8 (person 5) in session 1;
    # persons 1 = {1, 2}, 3 = {3, 4}, 5 = {5, 6, 7, 8}. Session-1 images per distance: 2, 2, 4; session-2: 2, 2, 2.
    # genuine intra: (4 + 4 + 8) x 2 distances = 32; cross D1-D5: 2 x 4 + 2 x 4 + 4 x 6 = 40;
    # impostor per combination: 2 x 4 + 2 x 4 + 4 x 4 = 32, 3 combinations
    assert metrics.subset.tolist() == ["all", "intra", "cross", "D1(8m)-D1(8m)", "D1(8m)-D5(4m)", "D5(4m)-D5(4m)"]
    by = metrics.set_index("subset")
    assert (by.loc["all", "n_genuine"], by.loc["all", "n_impostor"]) == (72, 96)
    assert (by.loc["intra", "n_genuine"], by.loc["cross", "n_genuine"]) == (32, 40)
    assert by.loc["D1(8m)-D5(4m)", "n_impostor"] == 32 and by.loc["all", "n_subjects"] == 3
    assert by.loc["all", "n_probes"] == 16                              # session-1 images
    assert metrics.rank1.isna().all() and metrics.eer.between(0, 1).all()
    assert metrics.protocol.iloc[0].startswith("zero-shot cross-dataset verification, pairing of Alonso-Fernandez")
    assert "7.6 Rs" in metrics.caveat.iloc[0]
    x1 = summary["per_mode"]["x1"]
    counts = x1["pair_counts"]
    assert (counts["persons"], counts["images"], counts["images_flipped"]) == (3, 28, 14)
    assert (counts["genuine"], counts["impostor"], counts["genuine_intra"], counts["genuine_cross"]) == (72, 96, 32, 40)
    assert counts["paper"]["genuine"] == 8600 and counts["paper"]["impostor"] == 438600 and not counts["matches_paper"]
    assert counts["nominal_for_these_persons"]["genuine"] == 300
    assert counts["by_combination"]["D1(8m)-D5(4m)"] == {"genuine": 40, "impostor": 32}
    assert counts["eye_class_session_distance_cells_missing"] == 3 * 20 - 28
    assert x1["x1_protocol"] == "alonso2025" and x1["frontal_rule"] == "closest"
    assert summary["external"]["protocol"] == "alonso2025" and summary["external"]["pair_counts"] == counts
    assert set(scores.kind) == {"genuine", "impostor"} and "combination" in scores.columns
    persons = {1: 1, 2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 5}
    gen, imp = scores[scores.kind == "genuine"], scores[scores.kind == "impostor"]
    assert (gen.subject.map(persons) == gen.other_subject.map(persons)).all()
    assert (imp.subject.map(persons) != imp.other_subject.map(persons)).all()
    assert (gen.subject != gen.other_subject).any()                   # left eye vs right eye of one person
    # 20 reports the same counts without a model
    assert prep_summary["alonso2025_protocol"]["closest"] == {"persons": 3, "images": 28, "genuine": 72,
                                                              "impostor": 96, "nominal_for_these_persons": [300, 360]}

    # 'L' crops are embedded mirrored, 'R' crops as they are
    script = importlib.util.spec_from_file_location("eval15_ubipr", os.path.join(REPO, "scripts",
                                                                                 "15_eval_verification.py"))
    eval15 = importlib.util.module_from_spec(script)
    script.loader.exec_module(eval15)
    saved = np.load(tmp_path / "alonso" / "embeddings_ubipr_t.npz")
    manifest = pd.read_csv(tmp_path / "ubipr" / "ubipr_manifest.csv").set_index("relpath")
    for rel in ("crops/C1_S1_I1.png", "crops/C2_S1_I1.png"):
        path = str(tmp_path / "ubipr" / rel)
        flip = manifest.loc[rel, "eye"] == "L"
        image = eval15.load_image(path, flip=flip)
        np.testing.assert_array_equal(image, eval15.load_image(path)[:, ::-1] if flip else eval15.load_image(path))
        k = saved["relpaths"].tolist().index(rel)
        np.testing.assert_allclose(saved["embeddings"][k].astype(np.float32), embed(model, image)[0], atol=2e-3)
    assert not np.array_equal(eval15.load_image(str(tmp_path / "ubipr" / "crops" / "C1_S1_I1.png"), flip=True),
                              eval15.load_image(str(tmp_path / "ubipr" / "crops" / "C1_S1_I1.png")))

    # strict frontal rule: every synthetic image is annotated pose 0 and gaze 0, so the same images are selected
    strict, _, strict_summary = _run15(tmp_path, ckpt, tmp_path / "strict", *ext, "--x1-protocol", "alonso2025",
                                       "--x1-frontal", "strict", "--max-subjects", "2")
    assert strict_summary["per_mode"]["x1"]["frontal_rule"] == "strict"
    assert strict_summary["per_mode"]["x1"]["pair_counts"]["persons"] == 2            # smoke run: first 2 persons
