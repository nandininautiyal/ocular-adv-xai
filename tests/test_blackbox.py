import numpy as np
import pytest

from src.attacks.blackbox import detect_eye, make_monocle_template, monocle_blending, patch_occlusion


def test_patch_position_is_row_then_col(images):
    adv = patch_occlusion(images[0], patch_size=20, position=(10, 50))
    changed = np.argwhere((adv != images[0]).any(axis=2))
    assert changed[:, 0].min() >= 10 and changed[:, 0].max() < 30     # rows
    assert changed[:, 1].min() >= 50 and changed[:, 1].max() < 70     # cols
    assert (adv[10:30, 50:70] == 0).all()


def test_patch_batch_and_input_untouched(images):
    before = images.copy()
    adv = patch_occlusion(images)
    np.testing.assert_array_equal(images, before)
    assert adv.shape == images.shape and (adv[:, 96:160, 96:160] == 0).all()


def test_monocle_template():
    t = make_monocle_template(100)
    assert t.shape == (100, 100, 4) and t.dtype == np.float32
    assert 0.0 <= t.min() and t.max() <= 1.0
    assert t[50, 50, 3] < 0.2          # lens is nearly transparent
    assert t[50, 92, 3] > 0.8          # rim is opaque
    assert t[0, 0, 3] == 0.0           # corners are empty


def test_monocle_only_changes_the_eye_box(images):
    box = (60, 70, 100, 80)
    adv = monocle_blending(images[0], eye_box=box)
    changed = np.argwhere((np.abs(adv - images[0]) > 1e-6).any(axis=2))
    assert len(changed) > 0
    assert changed[:, 1].min() >= 60 and changed[:, 1].max() < 160
    assert changed[:, 0].min() >= 70 and changed[:, 0].max() < 150
    assert adv.min() >= 0.0 and adv.max() <= 1.0


def test_monocle_left_eye_is_mirrored():
    img = np.full((256, 256, 3), 0.5, np.float32)
    box = (64, 64, 128, 128)
    right = monocle_blending(img, eye_box=box, side="right")
    left = monocle_blending(img, eye_box=box, side="left")
    np.testing.assert_allclose(left[64:192, 64:192], right[64:192, 64:192][:, ::-1], atol=1e-5)


def test_monocle_fallback_box_is_central():
    img = np.full((256, 256, 3), 0.5, np.float32)
    adv = monocle_blending(img, detect=False)
    changed = np.argwhere((np.abs(adv - img) > 1e-6).any(axis=2))
    assert changed.min() >= 70 and changed.max() < 186


def test_detect_eye_handles_no_eye():
    assert detect_eye(np.full((256, 256, 3), 0.5, np.float32)) is None


def test_detect_eye_without_haar_cascades_warns(monkeypatch):
    import cv2
    monkeypatch.delattr(cv2, "CascadeClassifier", raising=False)
    with pytest.warns(UserWarning, match="Haar"):
        assert detect_eye(np.full((256, 256, 3), 0.5, np.float32)) is None
