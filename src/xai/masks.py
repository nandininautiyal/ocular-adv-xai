
import cv2
import numpy as np


MASK_SIZE = (256, 256)


def _to_rgb_float(image):
    """Convert an RGB image to float32 in [0, 1] and resize to 256x256."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or infinite values.")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point image values must be in [0, 1].")

    return cv2.resize(image, MASK_SIZE, interpolation=cv2.INTER_LINEAR)


def get_concept_masks(image):
    """
    Return estimated anatomical masks at 256x256.

    Returns boolean arrays:
        pupil, iris, sclera, periocular, iris_pupil

    This is a classical OpenCV baseline. Its estimates require validation
    against manually annotated masks before research use.
    """
    rgb = _to_rgb_float(image)
    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    min_dim = min(h, w)

    # Detect candidate iris circles.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 4),
        param1=100,
        param2=30,
        minRadius=max(10, int(min_dim * 0.14)),
        maxRadius=int(min_dim * 0.40),
    )

    # A geometric fallback if Hough detection finds no circle.
    if circles is not None:
        candidates = np.round(circles[0]).astype(int)
        cx, cy, radius = min(
            candidates,
            key=lambda c: (
                ((c[0] - w / 2) / w) ** 2
                + ((c[1] - h / 2) / h) ** 2
            ),
        )
    else:
        cx, cy = int(w * 0.50), int(h * 0.48)
        radius = int(min_dim * 0.27)

    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    radius = int(np.clip(radius, 10, min_dim // 2))

    # Estimate the visible eye opening.
    # This ellipse is approximate; it does not detect eyelid boundaries.
    opening = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        opening,
        (int(w * 0.50), int(h * 0.47)),
        (int(w * 0.39), int(h * 0.25)),
        0, 0, 360, 1, -1,
    )

    # Candidate iris, restricted to the estimated visible eye opening.
    iris_pupil = np.zeros((h, w), np.uint8)
    cv2.circle(iris_pupil, (cx, cy), radius, 1, -1)
    iris_pupil = (iris_pupil.astype(bool) & opening.astype(bool))

    # Estimate the pupil using a conservative central dark-region threshold.
    # Reflections and poor contrast can make this estimate unreliable.
    pupil_candidate = (
        (gray < 45)
        & iris_pupil
    ).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pupil_candidate, connectivity=8
    )

    pupil = np.zeros((h, w), np.uint8)
    if count > 1:
        valid = []
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            px, py = centroids[i]
            if area >= 3:
                distance = ((px - cx) / max(radius, 1)) ** 2
                distance += ((py - cy) / max(radius, 1)) ** 2
                valid.append((distance, -area, i))

        if valid:
            _, _, best = min(valid)
            pupil[labels == best] = 1

    pupil = pupil.astype(bool) & iris_pupil.astype(bool)
    iris = iris_pupil.astype(bool) & ~pupil

    # Sclera is the estimated eye opening outside the iris/pupil.
    sclera = (
        opening.astype(bool)
        & ~iris
        & ~pupil
    )

    # Remaining pixels form the periocular region.
    periocular = ~(pupil | iris | sclera)

    masks = {
        "pupil": pupil.astype(bool),
        "iris": iris.astype(bool),
        "sclera": sclera.astype(bool),
        "periocular": periocular.astype(bool),
        "iris_pupil": (pupil | iris).astype(bool),
    }

    return masks


def masks_to_grid(masks, grid=8):
    """
    Area-average the four concept masks into a (4, grid, grid) array.

    Channel order: pupil, iris, sclera, periocular.
    Values represent the fraction of each grid cell covered by each mask.
    """
    if grid < 1:
        raise ValueError("grid must be a positive integer.")

    names = ("pupil", "iris", "sclera", "periocular")
    output = []

    for name in names:
        if name not in masks:
            raise KeyError(f"Missing mask: {name}")

        mask = np.asarray(masks[name], dtype=np.float32)
        if mask.shape != MASK_SIZE:
            raise ValueError(
                f"{name} must have shape {MASK_SIZE}, got {mask.shape}"
            )

        reduced = cv2.resize(
            mask, (grid, grid), interpolation=cv2.INTER_AREA
        )
        output.append(reduced)

    return np.stack(output).astype(np.float32)
