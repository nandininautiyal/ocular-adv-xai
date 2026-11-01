import numpy as np
import pytest

import src.attacks.blackbox as blackbox
from src.attacks.blackbox import (IRIS_LABEL, detect_eye, locate_eye, make_monocle_template,
                                  monocle_blending, patch_occlusion)
from src.xai.masks import get_concept_masks

YY, XX = np.mgrid[0:256, 0:256]
CENTRAL_BOX = (70, 70, 115, 115)        # 45 % of the width, centred


def dark_disc(cx, cy, r):
    """Light image with a dark disc: an iris the Hough transform can find."""
    img = np.full((256, 256, 3), 0.8, np.float32)
    img[(YY - cy) ** 2 + (XX - cx) ** 2 < r ** 2] = 0.15
    return img


def iris_labels(cx, cy, r, iris=IRIS_LABEL):
    """Eye-mask label map: skin (0) everywhere, iris inside the disc."""
    labels = np.zeros((256, 256), np.uint8)
    labels[(YY - cy) ** 2 + (XX - cx) ** 2 < r ** 2] = iris
    return labels


def changed_centre(adv, img):
    """(x, y) centre of the bounding box of the pixels the monocle changed."""
    changed = np.argwhere((np.abs(adv - img) > 1e-6).any(axis=2))
    (r0, c0), (r1, c1) = changed.min(axis=0), changed.max(axis=0)
    return (c0 + c1) / 2.0, (r0 + r1) / 2.0


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


# --- monocle placement: Haar box, then the iris, then the image centre -------------

def test_monocle_prefers_the_haar_box(monkeypatch):
    monkeypatch.setattr(blackbox, "detect_eye", lambda image: (60, 70, 100, 80))
    img = np.full((256, 256, 3), 0.5, np.float32)
    adv, placement = monocle_blending(img, eye_labels=iris_labels(90, 100, 30), return_placement=True)
    assert placement == "haar"
    np.testing.assert_array_equal(adv, monocle_blending(img, eye_box=(60, 70, 100, 80)))


def test_monocle_is_centred_on_the_iris_label():
    img = np.full((256, 256, 3), 0.5, np.float32)       # Haar finds no eye here
    labels = iris_labels(90, 100, 30)
    labels[:10] = 255                                    # unknown pixels are not iris
    before = img.copy(), labels.copy()
    adv, placement = monocle_blending(img, eye_labels=labels, return_placement=True)
    assert placement == "iris"
    cx, cy = changed_centre(adv, img)
    assert abs(cx - 90) <= 1 and abs(cy - 100) <= 1
    np.testing.assert_array_equal(img, before[0])
    np.testing.assert_array_equal(labels, before[1])
    # another label id for the iris, and a larger box
    box, placement = locate_eye(img, eye_labels=iris_labels(90, 100, 30, iris=4), iris_label=4)
    assert placement == "iris" and box == (33, 43, 115, 115)
    box, _ = locate_eye(img, eye_labels=labels, box_share=0.7)
    assert box == (1, 11, 179, 179)                     # side round(0.7 * 256), same centre
    with pytest.raises(ValueError):
        locate_eye(img, eye_labels=labels, box_share=0)


def test_monocle_without_mask_uses_the_hough_iris_circle():
    img = dark_disc(90, 100, 40)
    _, info = get_concept_masks(img, return_info=True)
    assert not info["fallback_used"]                     # precondition: Hough finds the disc
    assert abs(info["cx"] - 90) <= 4 and abs(info["cy"] - 100) <= 4
    box, placement = locate_eye(img, detect=False)
    assert placement == "iris" and box[2:] == (115, 115)
    assert (box[0] + 57, box[1] + 57) == (info["cx"], info["cy"])   # box centre = circle centre
    adv, placement = monocle_blending(img, detect=False, return_placement=True)
    assert placement == "iris"
    np.testing.assert_array_equal(adv, monocle_blending(img, eye_box=box))


def test_monocle_falls_back_to_the_centre():
    flat = np.full((256, 256, 3), 0.5, np.float32)       # no Haar eye, no Hough circle
    adv, placement = monocle_blending(flat, return_placement=True)
    assert placement == "centre_fallback"
    np.testing.assert_array_equal(adv, monocle_blending(flat, eye_box=CENTRAL_BOX))
    # a mask that finds no iris wins over a Hough circle that would be found
    disc = dark_disc(90, 100, 40)
    assert locate_eye(disc, eye_labels=np.zeros((256, 256), np.uint8), detect=False) == (
        CENTRAL_BOX, "centre_fallback")
    # the old placement (Haar, else the centre) is still available
    assert locate_eye(disc, detect=False, use_iris=False) == (CENTRAL_BOX, "centre_fallback")


def test_monocle_placement_flag_keeps_the_old_call_signature():
    img = np.full((256, 256, 3), 0.5, np.float32)
    assert isinstance(monocle_blending(img, detect=False), np.ndarray)
    adv, placement = monocle_blending(img, eye_box=(60, 70, 100, 80), return_placement=True)
    assert placement == "given"
    np.testing.assert_array_equal(adv, monocle_blending(img, eye_box=(60, 70, 100, 80)))
    # an iris at the border gives a box that sticks out; only its inside part is blended
    adv, placement = monocle_blending(img, eye_labels=iris_labels(5, 250, 20), return_placement=True)
    assert placement == "iris" and adv.shape == img.shape and not np.array_equal(adv, img)
    with pytest.raises(ValueError):
        monocle_blending(img, eye_labels=np.zeros((128, 128), np.uint8))
