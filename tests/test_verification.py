"""Verification metrics (src/eval/verification.py) on synthetic scores with known error rates,
embeddings of the tiny model, and an end-to-end run of scripts/15_eval_verification.py."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image
from scipy.stats import norm

from conftest import REPO, make_images
from src.eval.verification import (N_BINS, THRESHOLDS, bootstrap_metrics, bootstrap_picks, build_templates,
                                   cmc, embed, eer, error_rates, error_rates_from_counts, far_key,
                                   metrics_from_counts, pair_counts, rank1, rank1_ci, ranks, roc, score_bins,
                                   tar_at_far, template_counts, template_scores)
from src.models.arcface_inference import find_arcface_layer, predict_logits


def _counts(scores):
    return np.bincount(score_bins(scores), minlength=N_BINS)


def _gaussians(n=200_000, seed=0):
    """Genuine N(0.6, 0.1) and impostor N(0.2, 0.1): d' = 4, so EER = Phi(-2)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.6, 0.1, n), rng.normal(0.2, 0.1, n)


# --------------------------------------------------------------- raw scores

def test_roc_runs_from_nothing_to_everything_accepted():
    far, tar, thr = roc([0.9, 0.8, 0.8, 0.4], [0.8, 0.3, 0.2, 0.1])
    assert (far[0], tar[0], thr[0]) == (0.0, 0.0, np.inf) and far[-1] == tar[-1] == 1.0
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0) and np.all(np.diff(thr) < 0)
    # the tie at 0.8 is one operating point: two genuine and one impostor scores enter together
    k = np.flatnonzero(thr == 0.8)[0]
    assert tar[k] == 0.75 and far[k] == 0.25


def test_eer_hand_computed():
    # threshold 0.7 -> FRR 1/3, FAR 1/4; threshold 0.4 -> FRR 0, FAR 1/4: the segment crosses FAR = FRR at 1/4
    value, threshold = eer([0.9, 0.8, 0.4], [0.7, 0.3, 0.2, 0.1])
    assert value == pytest.approx(0.25) and threshold == 0.4
    # a tie at 0.6 moves FAR and FRR together, from (0, 3/4) to (1/4, 0): the segment crosses FAR = FRR 3/4
    # of the way along, at 3/16 (not min max(FAR, FRR) = 1/4, nor the mean at either end, 3/8 or 1/8)
    value, threshold = eer([0.9, 0.6, 0.6, 0.6], [0.6, 0.1, 0.1, 0.1])
    assert value == pytest.approx(3 / 16) and threshold == 0.6
    assert eer([0.9, 0.8], [0.1, 0.2])[0] == 0.0                  # separable scores
    assert np.isnan(eer([], [0.1])[0]) and np.isnan(eer([0.1], [])[0])


def test_eer_of_uniform_scores():
    # FAR(t) = 1 - t on [0, 1], FRR(t) = t - 0.5 on [0.5, 1.5]: they cross at t = 0.75, EER = 0.25
    impostor, genuine = np.linspace(0.0, 1.0, 10_001), np.linspace(0.5, 1.5, 10_001)
    value, threshold = eer(genuine, impostor)
    assert value == pytest.approx(0.25, abs=2e-4) and threshold == pytest.approx(0.75, abs=2e-4)


def test_eer_and_tar_of_gaussian_scores():
    genuine, impostor = _gaussians()
    assert eer(genuine, impostor)[0] == pytest.approx(norm.cdf(-2.0), abs=2e-3)
    tar, threshold, achieved = tar_at_far(genuine, impostor, 1e-3)
    assert achieved <= 1e-3 and threshold == pytest.approx(0.2 + 0.1 * norm.ppf(0.999), abs=0.01)
    assert tar == pytest.approx(norm.cdf(4.0 - norm.ppf(0.999)), abs=0.02)


def test_tar_at_far_is_an_operating_point():
    impostor = np.arange(100_000) / 100_000                         # FAR(t) = 1 - t
    genuine = (50_000 + np.arange(100_000)) / 100_000               # TAR(t) = 1.5 - t, same float grid
    tar, threshold, achieved = tar_at_far(genuine, impostor, 1e-2)
    assert achieved == 1e-2 and threshold == 0.99 and tar == 0.51
    frr, far = error_rates(genuine, impostor, threshold)
    assert far == pytest.approx(achieved) and frr == pytest.approx(1 - tar)
    # a target below 1 / n_impostor: only thresholds above every impostor score qualify
    tar, threshold, achieved = tar_at_far([0.9, 0.95], [0.1, 0.5], 1e-3)
    assert achieved == 0.0 and tar == 1.0 and threshold == 0.9


def test_error_rates():
    frr, far = error_rates([0.9, 0.5, 0.4], [0.5, 0.1], 0.5)
    assert frr == pytest.approx(1 / 3) and far == 0.5               # a score equal to the threshold is accepted
    # no threshold (e.g. TAR@FAR without impostor scores) gives no operating point, not FRR 1 / FAR 0
    assert np.isnan(error_rates([0.9], [0.1], np.nan)).all()
    assert np.isnan(error_rates_from_counts(_counts([0.9]), _counts([0.1]), np.nan)).all()


# -------------------------------------------------------------- grid counts

def test_far_key():
    assert far_key(1e-2) == "1e-2" and far_key(1e-3) == "1e-3" and far_key(0.05) == "5e-2"


def test_score_grid():
    np.testing.assert_array_equal(score_bins([-1.0, -0.99995, 0.5, 1.0, 1.0001]), [0, 0, 15000, N_BINS - 1,
                                                                                   N_BINS - 1])
    assert THRESHOLDS[15000] == 0.5


def test_metrics_from_counts_match_the_raw_scores():
    genuine, impostor = _gaussians(seed=1)
    m = metrics_from_counts(_counts(genuine), _counts(impostor))
    assert m["n_genuine"] == m["n_impostor"] == len(genuine)
    assert m["eer"] == pytest.approx(eer(genuine, impostor)[0], abs=1e-4)
    for target in (1e-2, 1e-3):
        key = far_key(target)
        assert m[f"tar_at_far_{key}"] == pytest.approx(tar_at_far(genuine, impostor, target)[0], abs=1e-3)
        assert m[f"far_achieved_{key}"] <= target
        frr, far = error_rates_from_counts(_counts(genuine), _counts(impostor), m[f"thr_at_far_{key}"])
        assert far == pytest.approx(m[f"far_achieved_{key}"]) and frr == pytest.approx(1 - m[f"tar_at_far_{key}"])
    # the counts give exactly the metrics of the scores rounded down to the grid
    rounded = THRESHOLDS[score_bins(genuine)], THRESHOLDS[score_bins(impostor)]
    assert (m["eer"], m["eer_threshold"]) == eer(*rounded)
    assert m["tar_at_far_1e-3"] == tar_at_far(*rounded, 1e-3)[0]


# --------------------------------------------------------------- comparisons

def _clusters(n_subjects=6, per_subject=4, dim=16, spread=0.3, seed=0):
    """L2-normalised embeddings: one random direction per subject plus noise, alternating eyes."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_subjects, dim))
    subjects = np.repeat(np.arange(100, 100 + n_subjects), per_subject)
    emb = centres[subjects - 100] + spread * rng.normal(size=(len(subjects), dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    eyes = np.tile(["left", "right"], len(subjects) // 2)
    return emb.astype(np.float32), subjects, eyes


def test_pair_counts_match_brute_force():
    emb, subjects, eyes = _clusters()
    out = pair_counts(emb, subjects, eyes, max_block=7)            # tiny blocks: exercise the block loop
    genuine, impostor = [], []
    best = {}
    for i in range(len(emb)):
        for j in range(len(emb)):
            if i == j or eyes[i] != eyes[j]:
                continue
            score = float(emb[i] @ emb[j])
            (genuine if subjects[i] == subjects[j] else impostor).append(score)
            key = "gen" if subjects[i] == subjects[j] else "imp"
            best[i, key] = max(best.get((i, key), -np.inf), score)
    np.testing.assert_array_equal(out["subjects"], np.unique(subjects))
    np.testing.assert_array_equal(out["genuine"].sum(0), _counts(genuine))
    np.testing.assert_array_equal(out["impostor"].sum(0), _counts(impostor))
    # per eye: 6 subjects x 2 images, so 6 x 2 ordered genuine pairs and 12 x 11 - 12 impostor ones
    assert out["genuine"].sum() == 2 * 12 and out["impostor"].sum() == 2 * 120
    # a row holds the comparisons whose first image is of that subject: 4 images x 1 partner / x 10 others
    assert out["genuine"][0].sum() == 4 and out["impostor"][0].sum() == 40
    expected = [float(best[i, "gen"] > best[i, "imp"]) for i in range(len(emb))]
    np.testing.assert_array_equal(out["nn_correct"], expected)


def test_pair_counts_without_partner_and_without_groups():
    emb, subjects, _ = _clusters(n_subjects=3, per_subject=1)
    out = pair_counts(emb, subjects)
    assert out["genuine"].sum() == 0 and out["impostor"].sum() == 6
    assert np.isnan(out["nn_correct"]).all()
    # an exact tie between the best genuine and the best impostor neighbour counts as wrong
    tie = pair_counts(np.array([[1, 0], [1, 0], [1, 0]], np.float32), np.array([1, 1, 2]))
    np.testing.assert_array_equal(tie["nn_correct"], [0.0, 0.0, np.nan])


def test_templates_scores_and_ranks():
    emb, subjects, eyes = _clusters(n_subjects=4, per_subject=6, spread=0.05)
    templates, t_subjects, t_eyes = build_templates(emb, subjects, eyes)
    assert len(templates) == 8 and list(t_subjects[:2]) == [100, 100] and list(t_eyes[:2]) == ["left", "right"]
    mean = emb[(subjects == 101) & (eyes == "right")].mean(0)
    np.testing.assert_allclose(templates[3], mean / np.linalg.norm(mean), atol=1e-6)

    probes, p_subjects, p_eyes = emb[:5], subjects[:5], eyes[:5]
    scores, own = template_scores(templates, t_subjects, probes, p_subjects, t_eyes, p_eyes)
    assert scores.shape == (5, 8) and own.sum(1).tolist() == [1] * 5
    assert np.isnan(scores[0, 1]) and not np.isnan(scores[0, 0])  # probe 0 is a left eye
    np.testing.assert_array_equal(ranks(scores, own), [1] * 5)    # tight clusters: own template wins

    hand = np.array([[0.9, 0.2, np.nan], [0.5, 0.5, 0.1], [0.1, 0.3, 0.2]], np.float32)
    hand_own = np.array([[True, False, False], [False, True, False], [False, False, False]])
    np.testing.assert_array_equal(ranks(hand, hand_own), [1, 2, 0])  # a tie counts against the probe
    np.testing.assert_allclose(cmc([1, 2, 0, 3], max_rank=3), [1 / 3, 2 / 3, 1.0])
    assert rank1([1, 2, 0, 3]) == pytest.approx(1 / 3) and np.isnan(rank1([0]))

    counts = template_counts(scores, own, p_subjects, max_block=3)
    np.testing.assert_array_equal(counts["subjects"], [100])
    np.testing.assert_array_equal(counts["genuine"][0], _counts(scores[own]))
    np.testing.assert_array_equal(counts["impostor"][0], _counts(scores[~own & ~np.isnan(scores)]))


# ------------------------------------------------------- confidence intervals

def _subject_counts(n_subjects=40, per_subject=200, seed=2):
    rng = np.random.default_rng(seed)
    gen = np.stack([_counts(rng.normal(0.6, 0.1, per_subject)) for _ in range(n_subjects)])
    imp = np.stack([_counts(rng.normal(0.2, 0.1, 20 * per_subject)) for _ in range(n_subjects)])
    return gen, imp


def test_bootstrap_brackets_the_estimate_and_is_seeded():
    gen, imp = _subject_counts()
    point = metrics_from_counts(gen.sum(0), imp.sum(0))
    ci = bootstrap_metrics(gen, imp, n_boot=200, seed=3)
    for name in ("eer", "tar_at_far_1e-2", "tar_at_far_1e-3"):
        low, high = ci[name]
        assert low <= point[name] <= high and high - low < 0.1
    assert ci == bootstrap_metrics(gen, imp, n_boot=200, seed=3)
    assert np.isnan(bootstrap_metrics(gen, imp, n_boot=0)["eer"][0])


def test_bootstrap_resamples_whole_subjects():
    # one perfect and one hopeless subject: a resample holds 0, 1 or 2 copies of each, never a mixture of
    # their single comparisons, so the EER of a resample is that of one subject or of both together
    good = _counts([0.9, 0.95]), _counts([0.1, 0.2])
    bad = _counts([0.1, 0.2]), _counts([0.9, 0.95])
    gen, imp = np.stack([good[0], bad[0]]), np.stack([good[1], bad[1]])
    together = metrics_from_counts(gen.sum(0), imp.sum(0))["eer"]
    low, high = bootstrap_metrics(gen, imp, n_boot=200)["eer"]
    assert low == 0.0 and high == 1.0 and 0 < together < 1


def test_bootstrap_equals_resampling_the_scores_of_whole_subjects():
    # adding count rows must give exactly what pooling the drawn subjects' scores gives, a subject drawn
    # twice counting twice (scores rounded down to the grid, as the counts store them)
    rng = np.random.default_rng(4)
    gen = [rng.normal(0.6, 0.15, rng.integers(5, 30)) for _ in range(12)]
    imp = [rng.normal(0.2, 0.15, rng.integers(50, 300)) for _ in range(12)]
    ci = bootstrap_metrics(np.stack([_counts(g) for g in gen]), np.stack([_counts(i) for i in imp]),
                           n_boot=200, seed=5)
    boots = {name: [] for name in ci}
    for pick in bootstrap_picks(12, 200, seed=5):
        g = np.concatenate([THRESHOLDS[score_bins(gen[k])] for k in pick])
        i = np.concatenate([THRESHOLDS[score_bins(imp[k])] for k in pick])
        boots["eer"].append(eer(g, i)[0])
        for target in (1e-2, 1e-3):
            boots[f"tar_at_far_{far_key(target)}"].append(tar_at_far(g, i, target)[0])
    for name, values in boots.items():
        np.testing.assert_allclose(ci[name], np.quantile(values, [0.025, 0.975]), rtol=0, atol=1e-12)


def test_rank1_ci_reuses_the_cluster_bootstrap():
    correct = np.r_[np.ones(30), np.zeros(10), np.nan]
    subjects = np.r_[np.repeat(np.arange(20), 2), 99]
    est, low, high = rank1_ci(correct, subjects, n_boot=300)
    assert est == pytest.approx(0.75) and low <= est <= high
    assert rank1_ci(correct, subjects, n_boot=0)[0] == pytest.approx(0.75)
    picks = bootstrap_picks(20, 3, seed=0)
    rng = np.random.default_rng(0)                                  # stats.cluster_bootstrap_ci draws these
    for pick in picks:
        np.testing.assert_array_equal(pick, rng.integers(0, 20, 20))


# --------------------------------------------------------------- embeddings

def test_embed_is_the_normalised_arcface_input(model, images):
    emb = embed(model, images, batch_size=2)
    assert emb.shape == (5, 32) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    arcface = find_arcface_layer(model)
    weights = tf.nn.l2_normalize(arcface.W, axis=0).numpy()
    np.testing.assert_allclose(arcface.s * emb @ weights, predict_logits(model, images), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(embed(model, images[2]), emb[2:3], atol=1e-5)
    assert embed(model, images[:0]).shape == (0, 32)


# ------------------------------------------------------------------ the CLI

SEEN, UNSEEN = (1141, 1142, 1143), (1000, 4000, 5000)
CONDITIONS = ("iphone_day_light_short", "samsung_office_short")


def _fake_visob(root, frames=3):
    """A VISOB-like tree: each subject is one synthetic disk image plus per-frame noise, saved at 64 x 64."""
    disks = make_images(6)
    rng = np.random.default_rng(0)
    for k, subject in enumerate(SEEN + UNSEEN):
        for condition in CONDITIONS:
            for phase in ("enrollment", "verification"):
                for eye in ("left", "right"):
                    d = root / condition / f"{phase}_{eye}_eye" / str(subject)
                    d.mkdir(parents=True)
                    for idx in range(1001, 1001 + frames):
                        img = np.clip(disks[k] + rng.uniform(-0.03, 0.03, disks[k].shape), 0, 1)[::4, ::4]
                        Image.fromarray((img * 255).round().astype(np.uint8)).save(
                            d / f"{subject}_{idx}_{eye[0]}_c.png")
    return root


def test_eval_verification_end_to_end(tmp_path, model):
    spec = importlib.util.spec_from_file_location("manifest01", os.path.join(REPO, "scripts", "01_build_manifest.py"))
    manifest_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest_script)
    root = _fake_visob(tmp_path / "VISOB_Augmented_Recent")
    manifest, skipped = manifest_script.build(str(root))
    assert not skipped and len(manifest) == 6 * 2 * 2 * 2 * 3
    manifest.to_csv(tmp_path / "manifest_raw.csv", index=False)
    manifest[manifest.subject.isin(SEEN)][["relpath", "subject"]].to_csv(tmp_path / "test.csv", index=False)

    # what 03 writes next to the checkpoint: 1143 was not a training class of this model
    ckpt = tmp_path / "Xception_ArcFace_tiny.keras"
    model.save(ckpt)
    (tmp_path / "class_index_tiny.json").write_text(json.dumps({"1141": 0, "1142": 1}))
    (tmp_path / "results_tiny.json").write_text(json.dumps({"images_by_phase": {"train": {"enrollment": 24}}}))

    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(REPO, "scripts", "15_eval_verification.py"), "--ckpt", str(ckpt),
           "--modes", "p3,p1,p2", "--manifest", str(tmp_path / "manifest_raw.csv"), "--data-root", str(root),
           "--test-list", str(tmp_path / "test.csv"), "--n-boot", "50", "--batch-size", "16", "--workers", "2",
           "--threshold", "0.5", "--sample-scores", "40", "--save-embeddings", "--out-dir", str(out), "--tag", "tiny"]
    result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]

    metrics = pd.read_csv(out / "verification_metrics_tiny.csv")
    assert metrics["mode"].unique().tolist() == ["p1", "p2", "p3"]
    assert set(metrics.subset) == {"all", "device=iphone", "device=samsung", "lighting=day", "lighting=office"}
    overall = metrics[metrics.subset == "all"].set_index("mode")
    assert overall.n_subjects.to_dict() == {"p1": 3, "p2": 2, "p3": 3}
    # p1: 24 images per subject, 12 per eye -> 3 x 2 x 66 genuine and 2 x (36 x 35 / 2 - 3 x 66) impostor pairs
    assert overall.loc["p1", "n_genuine"] == 396 and overall.loc["p1", "n_impostor"] == 864
    # p2 / p3: one template per subject and eye; each verification image is a probe
    assert overall.loc["p2", "n_templates"] == 4 and overall.loc["p2", "n_probes"] == 24
    assert overall.loc["p3", "n_genuine"] == 36 and overall.loc["p3", "n_impostor"] == 36 * 2
    assert overall.loc["p2", "eer"] == 0 and overall.loc["p2", "rank1"] == 1    # trained subjects separate
    assert metrics.eer.between(0, 1).all() and metrics.rank1.between(0, 1).all()
    assert (metrics.eer_ci_low <= metrics.eer + 1e-12).all() and (metrics.eer <= metrics.eer_ci_high + 1e-12).all()
    assert {"far_at_threshold", "frr_at_global_thr_1e-3", "tar_at_far_1e-2_ci_high"} <= set(metrics.columns)
    assert metrics.caveat[metrics["mode"] == "p1"].str.contains("inflated").all()
    assert metrics.caveat[metrics["mode"] == "p1"].str.contains("without verification images").all()

    scores = pd.read_csv(out / "verification_scores_tiny.csv")
    assert scores.groupby(["mode", "kind"]).size().to_dict() == {
        ("p1", "genuine"): 40, ("p1", "impostor"): 40, ("p2", "genuine"): 24, ("p2", "impostor"): 40,
        ("p3", "genuine"): 36, ("p3", "impostor"): 40}
    assert (scores[scores.kind == "impostor"].subject != scores[scores.kind == "impostor"].other_subject).all()

    summary = json.loads((out / "verification_summary_tiny.json").read_text())
    assert summary["modes"] == ["p1", "p2", "p3"] and summary["training_images_by_phase"] == {"enrollment": 24}
    assert summary["training_subjects"].endswith("class_index_tiny.json") and summary["git_commit"]
    assert len(summary["per_mode"]["p3"]["cmc_rank_1_to_20"]) == 20
    assert summary["per_mode"]["p3"]["n_probes_without_template"] == 0
    p2_notes = summary["per_mode"]["p2"]["caveats"]                            # trained on enrollment only,
    assert not any("verification images" in n for n in p2_notes)               # but only 24 impostor scores
    assert any("rule of 30" in n for n in p2_notes)
    assert (out / "verification_roc_det_tiny.png").stat().st_size > 10_000
    embeddings = np.load(out / "embeddings_tiny.npz")
    assert embeddings["embeddings"].shape == (6 * 24, 32) and len(embeddings["relpaths"]) == 6 * 24   # p2 within p1


def _eval_script():
    spec = importlib.util.spec_from_file_location("eval15", os.path.join(REPO, "scripts", "15_eval_verification.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_training_provenance_sets_the_caveats(tmp_path):
    script = _eval_script()
    ckpt = tmp_path / "Xception_ArcFace_cs.keras"
    assert script.training_info(str(ckpt)) == (None, "subject IDs 1141-3279 plus 7000 (no class index found)", None)
    assert "unknown" in script.caveats("p2", None)[0]
    (tmp_path / "results_cs.json").write_text(json.dumps({"test_acc_margin_free": 0.98}))   # written by the old 03
    assert script.training_info(str(ckpt))[2] == {} and "random split" in script.caveats("p2", {})[0]
    (tmp_path / "class_index_cs.json").write_text(json.dumps({"1141": 0, "7000": 1}))
    (tmp_path / "results_cs.json").write_text(json.dumps({"images_by_phase": {"train": {"enrollment": 9}}}))
    seen, source, phases = script.training_info(str(ckpt))
    assert seen == {1141, 7000} and source.endswith("class_index_cs.json") and phases == {"enrollment": 9}
    assert script.caveats("p2", phases) == [] and script.caveats("p3", phases) == []
    assert len(script.caveats("p1", phases)) == 2                             # inflated, and holds training images
    assert "inflated" in script.caveats("p2", {"enrollment": 9, "verification": 4})[0]
    assert script.in_paper_classes([1140, 1141, 3279, 3280, 7000]).tolist() == [False, True, True, False, True]
    with pytest.raises(ValueError, match="unexpected relpath"):
        script.image_table(["not/a/visob/path.png"])


def test_image_selection_keeps_p3_open_set(tmp_path):
    script = _eval_script()
    relpaths = [f"oppo_dim_light_short/{phase}_left_eye/{s}/{s}_1001_l_c.png"
                for s in (1141, 1143, 1000, 4000, 5000) for phase in ("enrollment", "verification")]
    pd.DataFrame({"relpath": relpaths}).to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame({"relpath": relpaths[:4]}).to_csv(tmp_path / "test.csv", index=False)    # 1141 and 1143
    args = argparse.Namespace(test_list=str(tmp_path / "test.csv"), manifest=str(tmp_path / "manifest.csv"),
                              p3_list=None, max_subjects=None)

    def subjects(selected, mode):
        return sorted(selected[mode].subject.unique())

    selected = script.select_images(["p1", "p2", "p3"], args, {1141})         # a model trained on 1141 only
    assert subjects(selected, "p1") == [1141, 1143] and subjects(selected, "p2") == [1141]
    assert subjects(selected, "p3") == [1000, 4000, 5000]                      # 1143 is one of the 300 IDs
    assert subjects(script.select_images(["p2"], args, None), "p2") == [1141, 1143]   # no class index
    assert subjects(script.select_images(["p3"], args, {1141, 1000}), "p3") == [4000, 5000]   # 1000 trained on
    args.max_subjects = 2                                                        # smoke run: first N subjects
    assert subjects(script.select_images(["p3"], args, {1141}), "p3") == [1000, 4000]
    args.max_subjects, args.p3_list = None, str(tmp_path / "manifest.csv")       # a list with training subjects
    with pytest.raises(SystemExit, match="training subjects"):
        script.select_images(["p3"], args, {1141})
