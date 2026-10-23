import cv2
import numpy as np


MASK_SIZE = (256, 256)

# Tunable constants (untested defaults: validate against hand-labelled masks).
HOUGH_PARAM2 = 30          # accumulator threshold; lower gives more candidates
EDGE_WEIGHT = 1.0          # weight of the boundary-edge score when ranking circles
CENTRE_WEIGHT = 2.0        # penalty on distance from the expected eye centre
PUPIL_MAX_SHARE = 0.40     # pupil area / iris-disc area upper bound (anatomical prior)
PUPIL_MIN_SHARE = 0.03     # below this the pupil is reported as not found
CLOSE_KERNEL = 7           # closes specular-highlight holes in the pupil


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


def _select_circle(gray, candidates, w, h):
    """Rank Hough circles by edge strength along their perimeter minus distance
    from the expected eye centre (instead of by distance alone).

    Returns ((cx, cy, r), score) or (None, nan) if no candidate has enough of
    its perimeter inside the image."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    scale = float(np.percentile(grad, 95)) + 1e-6
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    best, best_score = None, -np.inf
    for cx, cy, r in candidates:
        xs = np.round(cx + r * np.cos(theta)).astype(int)
        ys = np.round(cy + r * np.sin(theta)).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if inside.mean() < 0.5:
            continue
        edge = float(grad[ys[inside], xs[inside]].mean()) / scale
        dist = float(np.hypot((cx - w * 0.5) / w, (cy - h * 0.47) / h))
        score = EDGE_WEIGHT * edge - CENTRE_WEIGHT * dist
        if score > best_score:
            best, best_score = (int(cx), int(cy), int(r)), score
    return best, (best_score if best is not None else float("nan"))


def _filled_hull(binary, cx, cy, radius):
    """Close small holes, keep the component that is large and near the disc
    centre, then fill it to its convex hull (specular highlights sit inside the
    pupil and would otherwise cut holes in it)."""
    binary = binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros(binary.shape, bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    near = [
        i for i in range(1, count)
        if np.hypot(centroids[i][0] - cx, centroids[i][1] - cy) <= 0.7 * radius
    ]
    pool = near if near else list(range(1, count))
    best = max(pool, key=lambda i: areas[i - 1])

    contours, _ = cv2.findContours(
        (labels == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(np.vstack(contours))
    out = np.zeros(binary.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def _estimate_pupil(gray, disc, cx, cy, radius):
    """Darkest compact region inside the iris disc, capped at a plausible size.

    Starts from an Otsu threshold on the disc pixels and lowers it until the
    pupil covers at most PUPIL_MAX_SHARE of the disc. Returns (mask, threshold)."""
    empty = np.zeros(gray.shape, bool)
    values = gray[disc]
    if values.size < 50:
        return empty, float("nan")

    otsu, _ = cv2.threshold(
        values.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold, floor = float(otsu), float(values.min())
    disc_area = float(disc.sum())

    pupil = empty
    for _ in range(40):
        pupil = _filled_hull(disc & (gray <= threshold), cx, cy, radius) & disc
        if pupil.sum() / disc_area <= PUPIL_MAX_SHARE or threshold <= floor:
            break
        threshold -= 3.0

    if pupil.sum() / disc_area < PUPIL_MIN_SHARE:
        return empty, threshold
    return pupil, threshold


def get_concept_masks(image, return_info=False):
    """
    Return estimated anatomical masks at 256x256.

    Returns boolean arrays:
        pupil, iris, sclera, periocular, iris_pupil

    With return_info=True returns (masks, info), where info holds the circle
    used, whether the geometric fallback was needed and the pupil threshold, so
    failures can be counted instead of silently averaged over.

    This is a classical OpenCV baseline. Its estimates require validation
    against manually annotated masks before research use. The sclera and
    periocular masks come from a fixed eye-opening ellipse, not from eyelid
    detection: treat them as priors, not segmentations.
    """
    rgb = _to_rgb_float(image)
    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    min_dim = min(h, w)

    # Detect candidate iris circles, then pick the best-scoring one.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 4),
        param1=100,
        param2=HOUGH_PARAM2,
        minRadius=max(10, int(min_dim * 0.14)),
        maxRadius=int(min_dim * 0.40),
    )

    chosen, score, n_candidates = None, float("nan"), 0
    if circles is not None:
        candidates = np.round(circles[0]).astype(int)
        n_candidates = len(candidates)
        chosen, score = _select_circle(gray, candidates, w, h)

    # A geometric fallback if Hough detection finds no usable circle.
    fallback = chosen is None
    if fallback:
        cx, cy = int(w * 0.50), int(h * 0.48)
        radius = int(min_dim * 0.27)
    else:
        cx, cy, radius = chosen

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
    opening = opening.astype(bool)

    # Candidate iris, restricted to the estimated visible eye opening.
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (cx, cy), radius, 1, -1)
    iris_pupil = disc.astype(bool) & opening

    # Pupil: darkest compact region inside the disc, size-capped.
    pupil, threshold = _estimate_pupil(gray, iris_pupil, cx, cy, radius)
    iris = iris_pupil & ~pupil

    # Sclera is the estimated eye opening outside the iris/pupil.
    sclera = opening & ~iris & ~pupil

    # Remaining pixels form the periocular region.
    periocular = ~(pupil | iris | sclera)

    masks = {
        "pupil": pupil.astype(bool),
        "iris": iris.astype(bool),
        "sclera": sclera.astype(bool),
        "periocular": periocular.astype(bool),
        "iris_pupil": (pupil | iris).astype(bool),
    }

    if not return_info:
        return masks

    info = {
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "fallback_used": bool(fallback),
        "n_candidates": int(n_candidates),
        "circle_score": float(score),
        "centre_in_opening": bool(opening[cy, cx]),
        "pupil_found": bool(pupil.any()),
        "pupil_threshold": float(threshold),
        "pupil_share_of_disc": float(pupil.sum() / max(int(iris_pupil.sum()), 1)),
    }
    return masks, info


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