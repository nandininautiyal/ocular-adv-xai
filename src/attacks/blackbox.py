"""
src/attacks/blackbox.py

Implementations of the two ocular-specific black-box attacks described
in the ABRobOcular paper (Section 3.3): Patch Occlusion and Monocle Blending.

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
           (patch_occlusion also accepts a batch (N, 256, 256, 3))
- Output : float32, same shape as the input, range [0, 1]

The vendored repo ships no code and no template image for either attack, so
both follow the paper's text:
- Patch positions are (X, Y) = (column, row) of the top-left corner, as in the
  paper's Fig. 7 and Tables 4-5; (96, 96) with a 64 x 64 patch is the most
  damaging position on VISOB (ASR 58.8 %).
- Monocle Blending: Haar-cascade eye box -> pupil centre (Eq. 7) -> template
  resized to the eye box and flipped for left eyes -> alpha blend (Eq. 8). The
  paper's monocle image was not released; ``make_monocle_template`` draws a
  stand-in modelled on its Fig. 4 (gold rim, faint lens, bead chain).
"""

import warnings

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Patch Occlusion
# ---------------------------------------------------------------------------

def patch_occlusion(image, patch_size=64, position=(96, 96), mask_value=0.0):
    """Apply a square occlusion patch to the image.

    According to the paper, the most critical position for VISOB is (96, 96)
    with a patch size of 64x64, which covers the iris.

    Parameters
    ----------
    image : ndarray (256, 256, 3) or (N, 256, 256, 3), float32, [0, 1]
    patch_size : int, side length of the square patch
    position : tuple (x, y) — column and row of the top-left corner (paper convention)
    mask_value : float in [0, 1], value to fill the patch with (usually black=0)

    Returns
    -------
    adv_image : ndarray, same shape as *image*, float32, [0, 1]
    """
    adv = np.array(image, dtype=np.float32, copy=True)
    x, y = position
    x0, y0 = max(x, 0), max(y, 0)
    adv[..., y0:y + patch_size, x0:x + patch_size, :] = mask_value
    return adv


# ---------------------------------------------------------------------------
# Monocle Blending
# ---------------------------------------------------------------------------

def make_monocle_template(diameter=128):
    """Draw an RGBA monocle (gold rim, faint lens, bead chain on the right).

    Parameters
    ----------
    diameter : int — side length of the square template in pixels

    Returns
    -------
    template : ndarray (diameter, diameter, 4), float32, [0, 1] — RGB + alpha
    """
    d = float(diameter)
    yy, xx = np.mgrid[0:diameter, 0:diameter].astype(np.float32) + 0.5
    r = np.hypot(xx - d / 2, yy - d / 2)

    rgb = np.zeros((diameter, diameter, 3), np.float32)
    alpha = np.zeros((diameter, diameter), np.float32)

    lens = r < 0.40 * d
    rgb[lens] = (0.92, 0.95, 1.00)
    alpha[lens] = 0.12

    rim = np.exp(-((r - 0.42 * d) / (0.025 * d)) ** 2)        # soft-edged gold ring
    gold = np.array([0.86, 0.66, 0.30], np.float32)
    mask = rim > alpha
    rgb[mask] = gold
    alpha = np.maximum(alpha, 0.9 * rim)

    for cy, radius in [(0.62, 0.022), (0.69, 0.022), (0.76, 0.022), (0.86, 0.040)]:
        bead = np.hypot(xx - 0.87 * d, yy - cy * d) < radius * d
        rgb[bead] = gold
        alpha[bead] = 0.9

    return np.dstack([rgb, alpha]).astype(np.float32)


def detect_eye(image):
    """Locate the eye with OpenCV's Haar cascade (paper, Section 3.3, stage 1).

    Parameters
    ----------
    image : ndarray (H, W, 3), float32, [0, 1]

    Returns
    -------
    box : tuple (x, y, w, h) of the largest detection, or None if no eye is
        found or Haar cascades are unavailable (OpenCV 5 removed them; use
        opencv-python-headless<5)
    """
    import cv2  # optional dependency; only this function needs it

    if not hasattr(cv2, "CascadeClassifier"):
        warnings.warn(f"OpenCV {cv2.__version__} has no Haar cascades (removed in 5.0); "
                      "monocle_blending falls back to a central eye box. "
                      "Install opencv-python-headless<5 to follow the paper.", stacklevel=2)
        return None

    gray = cv2.cvtColor((np.clip(image, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    min_side = max(16, min(gray.shape) // 8)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(min_side, min_side))
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def monocle_blending(image, monocle_template=None, eye_box=None, side=None, alpha=1.0,
                     detect=True):
    """Simulate a physical Monocle attack via alpha blending (paper Eq. 7-8).

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    monocle_template : ndarray (H, W, 4), float32, [0, 1] with alpha channel,
        or None for :func:`make_monocle_template`
    eye_box : tuple (x, y, w, h) or None
        Eye bounding box. If None, it is detected with :func:`detect_eye`
        (when *detect* is True); if that fails, a central box covering 45 % of
        the image width is used.
    side : {"left", "right", None}
        Eye side. The template is mirrored for left eyes so that the chain sits
        on the same side of the face for both eyes.
    alpha : float in [0, 1], overall opacity of the monocle
    detect : bool, whether to run the Haar cascade when *eye_box* is None

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    adv = np.array(image, dtype=np.float32, copy=True)
    h, w = adv.shape[:2]

    if eye_box is None and detect:
        eye_box = detect_eye(adv)
    if eye_box is None:
        side_len = int(round(0.45 * w))
        eye_box = ((w - side_len) // 2, (h - side_len) // 2, side_len, side_len)
    bx, by, bw, bh = eye_box

    template = make_monocle_template() if monocle_template is None else monocle_template
    if side == "left":
        template = template[:, ::-1]
    template = tf.image.resize(template, (bh, bw), antialias=True).numpy().clip(0.0, 1.0)

    # Pupil centre = centre of the eye box (Eq. 7); the template is centred on it.
    px, py = bx + bw / 2.0, by + bh / 2.0
    x, y = int(round(px - bw / 2.0)), int(round(py - bh / 2.0))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 <= x0 or y1 <= y0:
        return adv  # template entirely outside the image

    roi = template[y0 - y:y1 - y, x0 - x:x1 - x]
    a = alpha * roi[..., 3:4]
    adv[y0:y1, x0:x1] = a * roi[..., :3] + (1.0 - a) * adv[y0:y1, x0:x1]
    return np.clip(adv, 0.0, 1.0)
