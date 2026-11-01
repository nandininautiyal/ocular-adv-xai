import numpy as np
import pytest

from src.attacks.blackbox import monocle_blending, patch_occlusion
from src.compare.regions import (REGIONS, enrichment, grid_shape, grids_from_labels, heuristic_masks, label_masks,
                                 masks_to_grids, normalised_entropy, occlusion_grids, occlusion_mask, region_areas,
                                 region_grids, region_mass, region_mass_named)
from src.xai.masks import get_concept_masks, masks_to_grid

EYE_CLASSES = ("skin", "iris", "sclera", "lid", "caruncle", "brow")


def eye_labels(n=2, seed=0):
    """Synthetic label maps (N, 256, 256) uint8 with all six classes and a band of unknown (255)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256]
    out = []
    for _ in range(n):
        cx, cy = rng.integers(90, 166, size=2)
        labels = np.zeros((256, 256), np.uint8)                                  # skin
        labels[np.abs(yy - cy) < 60] = 3                                          # lid band
        labels[((xx - cx) / 100.0) ** 2 + ((yy - cy) / 45.0) ** 2 < 1] = 2        # sclera
        labels[(xx - cx) ** 2 + (yy - cy) ** 2 < 35 ** 2] = 1                     # iris (with pupil)
        labels[(xx - (cx - 95)) ** 2 + (yy - cy) ** 2 < 8 ** 2] = 4               # caruncle
        labels[:20] = 5                                                           # brow
        out.append(labels)
    return np.stack(out)


# --- src/xai/masks.py --------------------------------------------------------

def test_masks_partition_the_image(images):
    masks = get_concept_masks(images[0])
    regions = np.stack([masks[r] for r in REGIONS])
    assert regions.shape == (4, 256, 256) and regions.dtype == bool
    assert (regions.sum(axis=0) == 1).all()                        # disjoint and covering
    np.testing.assert_array_equal(masks["iris_pupil"], masks["iris"] | masks["pupil"])


def test_masks_accept_uint8_and_leave_input_untouched(images):
    img8 = (images[0] * 255).round().astype(np.uint8)
    before = img8.copy()
    masks = get_concept_masks(img8)
    np.testing.assert_array_equal(img8, before)
    assert masks["periocular"].any()


def test_masks_return_info_keeps_masks_and_reports_qc(images):
    plain = get_concept_masks(images[0])
    masks, info = get_concept_masks(images[0], return_info=True)
    for name in plain:
        np.testing.assert_array_equal(masks[name], plain[name])
    for key in ("fallback_used", "centre_in_opening", "n_candidates", "circle_score",
                "pupil_found", "pupil_threshold", "pupil_share_of_disc"):
        assert key in info
    assert isinstance(info["fallback_used"], bool)
    assert info["pupil_found"] == bool(masks["pupil"].any())


def test_masks_reject_bad_input():
    with pytest.raises(ValueError):
        get_concept_masks(np.zeros((256, 256), np.float32))
    with pytest.raises(ValueError):
        get_concept_masks(np.full((256, 256, 3), 2.0, np.float32))


def test_masks_to_grid_cells_sum_to_one(images):
    grid = masks_to_grid(get_concept_masks(images[0]), grid=8)
    assert grid.shape == (4, 8, 8)
    np.testing.assert_allclose(grid.sum(axis=0), 1.0, atol=1e-5)


# --- src/compare/regions.py --------------------------------------------------

def test_region_grids_batch(images):
    grids = region_grids(images[:2], grid=8)
    assert grids.shape == (2, 4, 8, 8) and grids.dtype == np.float32


def test_region_mass_sums_to_one_and_follows_the_heatmap():
    grids = np.zeros((4, 8, 8), np.float32)
    grids[1, :4] = 1.0          # iris: top half
    grids[3, 4:] = 1.0          # periocular: bottom half
    top = np.zeros((8, 8), np.float32)
    top[:4] = 1.0
    mass = region_mass(top, grids)
    assert set(mass) == {f"mass_{r}" for r in REGIONS}
    assert mass["mass_iris"] == pytest.approx(1.0) and mass["mass_periocular"] == pytest.approx(0.0)
    uniform = region_mass(np.ones((8, 8)), grids)
    assert uniform["mass_iris"] == pytest.approx(0.5)
    assert sum(uniform.values()) == pytest.approx(1.0)


def test_region_mass_edge_cases():
    grids = np.full((4, 8, 8), 0.25, np.float32)
    assert all(np.isnan(v) for v in region_mass(np.zeros((8, 8)), grids).values())
    with pytest.raises(ValueError):
        region_mass(np.ones((8, 8)), grids[:3])
    with pytest.raises(ValueError):
        region_mass(-np.ones((8, 8)), grids)


def test_region_grids_unchanged_for_square_grids(images):
    for grid in (8, 16, (8, 8)):
        side = grid_shape(grid)[0]
        old = np.stack([masks_to_grid(get_concept_masks(img), grid=side) for img in images[:2]])
        np.testing.assert_array_equal(region_grids(images[:2], grid), old)


def test_region_grids_non_square(images):
    grids = region_grids(images[:2], (5, 8))
    assert grids.shape == (2, 4, 5, 8) and grids.dtype == np.float32
    np.testing.assert_allclose(grids.sum(axis=1), 1.0, atol=1e-5)        # still a partition
    masks = heuristic_masks(images[:2])
    assert masks.shape == (2, 4, 256, 256) and masks.dtype == bool
    for g, m in zip(grids, masks):
        areas = region_areas(g, REGIONS)
        for r, region_mask in zip(REGIONS, m):
            assert areas[f"area_{r}"] == pytest.approx(region_mask.mean(), abs=1e-6)


def test_grid_shape():
    assert grid_shape(8) == (8, 8) and grid_shape(np.int64(16)) == (16, 16)
    assert grid_shape((5, 8)) == (5, 8) and grid_shape([7, 3]) == (7, 3)
    for bad in (0, (0, 8), (8, -1)):
        with pytest.raises(ValueError):
            grid_shape(bad)


def test_masks_to_grids_matches_masks_to_grid_and_keeps_leading_axes(images):
    masks = get_concept_masks(images[0])
    stacked = np.stack([masks[r] for r in REGIONS])
    np.testing.assert_array_equal(masks_to_grids(stacked, 8), masks_to_grid(masks, grid=8))
    assert masks_to_grids(np.zeros((2, 3, 256, 256), bool), (4, 2)).shape == (2, 3, 4, 2)


@pytest.mark.parametrize("grid", [8, 16, (5, 8), (7, 3)])
def test_grids_from_labels_areas_match_pixel_fractions(grid):
    labels = eye_labels()
    labels[:, 200:210, :50] = 255                       # unknown pixels belong to no class
    grids, names = grids_from_labels(labels, EYE_CLASSES, grid)
    assert names == EYE_CLASSES
    assert grids.shape == (2, 6) + grid_shape(grid) and grids.dtype == np.float32
    unknown = masks_to_grids(labels == 255, grid)
    np.testing.assert_allclose(grids.sum(axis=1) + unknown, 1.0, atol=1e-5)
    for g, lab in zip(grids, labels):
        areas = region_areas(g, names)
        assert set(areas) == {f"area_{n}" for n in EYE_CLASSES}
        for k, name in enumerate(names):
            assert areas[f"area_{name}"] == pytest.approx((lab == k).mean(), abs=1e-6)


def test_grids_from_labels_rejects_bad_labels():
    labels = eye_labels(1)
    with pytest.raises(ValueError):
        grids_from_labels(labels[0], EYE_CLASSES, 8)                  # needs (N, H, W)
    with pytest.raises(ValueError):
        grids_from_labels(labels.astype(np.float32), EYE_CLASSES, 8)  # label ids are integers
    with pytest.raises(ValueError):
        grids_from_labels(labels, EYE_CLASSES[:5], 8)                 # label 5 has no name
    with pytest.raises(ValueError):
        grids_from_labels(labels, ("skin", "iris", "iris", "lid", "caruncle", "brow"), 8)
    masks = label_masks(labels, EYE_CLASSES)
    assert masks.shape == (1, 6, 256, 256) and (masks.sum(axis=1) == 1).all()


def test_named_masses_sum_to_one_and_match_region_mass(images):
    rng = np.random.default_rng(0)
    grids, names = grids_from_labels(eye_labels(), EYE_CLASSES, 8)
    for g in grids:
        mass = region_mass_named(rng.uniform(0, 1, (8, 8)), g, names)
        assert set(mass) == {f"mass_{n}" for n in EYE_CLASSES}
        assert sum(mass.values()) == pytest.approx(1.0)
    heuristic = region_grids(images[:1], 8)[0]
    heatmap = rng.uniform(0, 1, (8, 8))
    assert region_mass_named(heatmap, heuristic, REGIONS) == region_mass(heatmap, heuristic)
    assert set(region_mass_named(heatmap, heuristic, REGIONS, prefix="massvis")) == {
        f"massvis_{r}" for r in REGIONS}
    assert all(np.isnan(v) for v in region_mass_named(np.zeros((8, 8)), grids[0], names).values())
    with pytest.raises(ValueError):
        region_mass_named(heatmap, grids[0], names[:5])
    non_square = grids_from_labels(eye_labels(1), EYE_CLASSES, (5, 8))[0][0]
    assert sum(region_mass_named(np.ones((5, 8)), non_square, names).values()) == pytest.approx(1.0)


def test_enrichment_is_one_for_a_uniform_map():
    grids, names = grids_from_labels(eye_labels(1), EYE_CLASSES, 8)
    areas = region_areas(grids[0], names)
    mass = region_mass_named(np.ones((8, 8)), grids[0], names)
    enr = enrichment(mass, areas)
    assert set(enr) == {f"enr_{n}" for n in EYE_CLASSES}
    for name in names:
        assert mass[f"mass_{name}"] == pytest.approx(areas[f"area_{name}"])
        assert enr[f"enr_{name}"] == pytest.approx(1.0)
    peaked = np.zeros((8, 8))
    peaked[0] = 1.0                                   # all weight on the top row (the brow)
    assert enrichment(region_mass_named(peaked, grids[0], names), areas)["enr_brow"] > 1.0
    assert np.isnan(enrichment({"mass_iris": 0.3}, {"area_iris": 0.0})["enr_iris"])


def test_normalised_entropy():
    assert normalised_entropy(np.ones((8, 8))) == pytest.approx(1.0)
    one_hot = np.zeros((8, 8))
    one_hot[3, 4] = 0.7
    assert normalised_entropy(one_hot) == 0.0
    two = np.zeros((8, 8))
    two[0, :2] = 1.0
    assert normalised_entropy(two) == pytest.approx(np.log(2) / np.log(64))
    assert normalised_entropy(np.ones((5, 8))) == pytest.approx(1.0)
    assert np.isnan(normalised_entropy(np.zeros((8, 8))))
    with pytest.raises(ValueError):
        normalised_entropy(-np.ones((8, 8)))


def test_occlusion_aware_mass_renormalises_over_visible_pixels():
    labels = np.zeros((1, 256, 256), np.uint8)
    labels[:, :, 128:] = 1                            # left half skin, right half iris
    occluded = np.zeros((256, 256), bool)
    occluded[:128, 128:] = True                       # top half of the iris is covered
    masks = label_masks(labels, ("skin", "iris"))[0]
    grids, visible = occlusion_grids(masks, occluded, 8)
    mass = region_mass_named(np.ones((8, 8)), grids, ("skin", "iris"), visible=visible, prefix="massvis")
    assert mass == pytest.approx({"massvis_skin": 2 / 3, "massvis_iris": 1 / 3})
    # nothing occluded: the plain masses
    grids, visible = occlusion_grids(masks, np.zeros((256, 256), bool), 8)
    heatmap = np.random.default_rng(0).uniform(0, 1, (8, 8))
    plain = region_mass_named(heatmap, masks_to_grids(masks, 8), ("skin", "iris"))
    vis = region_mass_named(heatmap, grids, ("skin", "iris"), visible=visible)
    assert vis == pytest.approx(plain)
    # all heatmap weight under the occluder: undefined
    hot = np.zeros((8, 8))
    hot[0, 7] = 1.0
    grids, visible = occlusion_grids(masks, occluded, 8)
    assert all(np.isnan(v) for v in region_mass_named(hot, grids, ("skin", "iris"), visible=visible).values())


def test_occlusion_aware_mass_is_nan_for_a_region_the_occluder_covers():
    # A region hidden completely has no measurable mass: 0 would make its dmassvis read "no shift".
    names = ("skin", "iris", "sclera", "lid")                 # no lid pixels in this image
    labels = np.zeros((1, 256, 256), np.uint8)
    labels[:, 96:160, 96:160] = 1                             # the iris is exactly under the patch
    labels[:, :16] = 2
    masks = label_masks(labels, names)[0]
    occluded = np.zeros((256, 256), bool)
    occluded[96:160, 96:160] = True
    heatmap = np.random.default_rng(2).uniform(0.1, 1.0, (8, 8))
    grids, visible = occlusion_grids(masks, occluded, 8)
    mass = region_mass_named(heatmap, grids, names, visible=visible, prefix="massvis")
    assert np.isnan(mass["massvis_iris"]) and np.isnan(mass["massvis_lid"])
    assert mass["massvis_skin"] + mass["massvis_sclera"] == pytest.approx(1.0)
    occluded[96, 96] = False                                  # one visible iris pixel is enough
    grids, visible = occlusion_grids(masks, occluded, 8)
    assert 0 < region_mass_named(heatmap, grids, names, visible=visible)["mass_iris"] < 1e-2
    # plain masses are unchanged: an empty region has mass 0, as in region_mass
    assert region_mass_named(heatmap, masks_to_grids(masks, 8), names)["mass_lid"] == 0.0


def test_occlusion_aware_mass_equals_pixel_level_reweighting():
    # Each cell's weight spread evenly over its 32 x 32 pixels, occluded pixels set to 0,
    # renormalised: the definition, computed pixel by pixel.
    rng = np.random.default_rng(1)
    labels = eye_labels(1, seed=3)
    masks = label_masks(labels, EYE_CLASSES)[0]
    occluded = np.zeros((256, 256), bool)
    occluded[40:139, 77:190] = True                   # not aligned with the cells
    heatmap = rng.uniform(0, 1, (8, 8))
    grids, visible = occlusion_grids(masks, occluded, 8)
    mass = region_mass_named(heatmap, grids, EYE_CLASSES, visible=visible, prefix="massvis")
    pixel_weight = np.kron(heatmap, np.ones((32, 32))) * ~occluded
    hidden = [name for name, region in zip(EYE_CLASSES, masks) if not (region & ~occluded).any()]
    assert hidden == ["iris"]                         # covered completely: not measured
    for name, region in zip(EYE_CLASSES, masks):
        expected = (pixel_weight * region).sum() / pixel_weight.sum() if name not in hidden else np.nan
        assert mass[f"massvis_{name}"] == pytest.approx(expected, rel=1e-5, abs=1e-7, nan_ok=True)
    assert np.nansum(list(mass.values())) == pytest.approx(1.0)
    batch_grids, batch_visible = occlusion_grids(masks[np.newaxis], occluded[np.newaxis], 8)
    np.testing.assert_array_equal(batch_grids[0], grids)
    np.testing.assert_array_equal(batch_visible[0], visible)
    with pytest.raises(ValueError):
        region_mass_named(heatmap, grids, EYE_CLASSES, visible=visible[:4])


def test_occlusion_mask_finds_the_occluder(images):
    patched = patch_occlusion(images[:2], patch_size=64, position=(96, 96))
    occluded = occlusion_mask(images[:2], patched)
    assert occluded.shape == (2, 256, 256)
    expected = np.zeros((256, 256), bool)
    expected[96:160, 96:160] = True
    assert (occluded == expected).all()
    assert occluded.mean() == pytest.approx(64 * 64 / 256 ** 2)
    flat = np.full((256, 256, 3), 0.5, np.float32)
    monocle = monocle_blending(flat, detect=False, use_iris=False)
    covered = occlusion_mask(flat, monocle)
    assert 0 < covered.mean() < 0.25 and not covered[:60].any()     # inside the central box
