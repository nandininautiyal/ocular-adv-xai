"""
src/attacks/blackbox.py

Implementations of the two ocular-specific black-box attacks described
in the ABRobOcular paper: Patch Occlusion and Monocle Blending.

Image contract
--------------
- Input  : float32, shape (256, 256, 3), range [0, 1]
- Output : float32, shape (256, 256, 3), range [0, 1]
"""

import numpy as np

# ---------------------------------------------------------------------------
# Patch Occlusion
# ---------------------------------------------------------------------------

def patch_occlusion(image, patch_size=64, position=(96, 96), mask_value=0.0):
    """Apply a rectangular occlusion patch to the image.
    
    According to the paper, the most critical position for VISOB is (96, 96)
    with a patch size of 64x64, which covers the iris.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    patch_size : int, side length of the square patch
    position : tuple (y, x), top-left corner of the patch
    mask_value : float in [0, 1], value to fill the patch with (usually black=0)

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    adv = image.copy()
    y, x = position
    
    # Ensure bounds
    y_end = min(y + patch_size, adv.shape[0])
    x_end = min(x + patch_size, adv.shape[1])
    y_start = max(y, 0)
    x_start = max(x, 0)
    
    adv[y_start:y_end, x_start:x_end, :] = mask_value
    return adv


# ---------------------------------------------------------------------------
# Monocle Blending
# ---------------------------------------------------------------------------

def monocle_blending(image, monocle_template, pupil_center=None, alpha=1.0):
    """Simulate a physical Monocle attack via alpha blending.
    
    If pupil_center is not provided, it approximates the geometric center.

    Parameters
    ----------
    image : ndarray (256, 256, 3), float32, [0, 1]
    monocle_template : ndarray (H, W, 4), float32, [0, 1] with alpha channel
    pupil_center : tuple (y, x) or None
    alpha : float, overall transparency of the monocle

    Returns
    -------
    adv_image : ndarray (256, 256, 3), float32, [0, 1]
    """
    adv = image.copy()
    h, w = adv.shape[:2]
    
    if pupil_center is None:
        # Fallback: approximate geometric center
        pupil_center = (h // 2, w // 2)
        
    py, px = pupil_center
    mh, mw = monocle_template.shape[:2]
    
    # Calculate top-left placement such that the monocle centers on the pupil
    y = py - mh // 2
    x = px - mw // 2
    
    # Find valid intersecting region between image and monocle
    y_start = max(y, 0)
    y_end = min(y + mh, h)
    x_start = max(x, 0)
    x_end = min(x + mw, w)
    
    my_start = y_start - y
    my_end = my_start + (y_end - y_start)
    mx_start = x_start - x
    mx_end = mx_start + (x_end - x_start)
    
    if y_end <= y_start or x_end <= x_start:
        return adv  # Monocle is completely out of bounds
        
    # Extract the ROI from the monocle template
    monocle_roi = monocle_template[my_start:my_end, mx_start:mx_end]
    
    # Extract alpha channel and scale by overall alpha
    monocle_alpha = monocle_roi[..., 3:4] * alpha
    monocle_rgb = monocle_roi[..., :3]
    
    # Alpha blend
    roi = adv[y_start:y_end, x_start:x_end]
    blended = monocle_alpha * monocle_rgb + (1.0 - monocle_alpha) * roi
    adv[y_start:y_end, x_start:x_end] = blended
    
    return np.clip(adv, 0.0, 1.0)
