import numpy as np
import pytest

from src.compare.regions import REGIONS, region_grids, region_mass
from src.xai.masks import get_concept_masks, masks_to_grid


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
