"""Area-baseline null for region shifts: lam, residuals, enrichment, area sources."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_images
from src.analysis.region_null import (HEURISTIC_PERIOCULAR_AREA, HEURISTIC_REGIONS, add_null_columns,
                                      attach_areas, enrichment, fit_area_null, null_table, pooled_r2,
                                      regions_in)
from src.compare.regions import region_mass
from src.xai.masks import get_concept_masks, masks_to_grid


def test_fit_recovers_a_known_lambda():
    rng = np.random.default_rng(0)
    clean, area = rng.dirichlet(np.ones(4), 50), rng.dirichlet(np.full(4, 5.0), 50)
    fit = fit_area_null(0.3 * (area - clean), clean, area, min_gap=0.0)
    np.testing.assert_allclose(fit["lam"], 0.3)
    np.testing.assert_allclose(fit["res"], 0.0, atol=1e-12)
    np.testing.assert_allclose(fit["r2"], 1.0)
    one_area = fit_area_null(0.5 * (area[0] - clean), clean, area[0], min_gap=0.0)     # (R,) areas broadcast
    np.testing.assert_allclose(one_area["lam"], 0.5)


def test_residual_is_zero_under_pure_diffusion():
    """A map mixed with the uniform map moves every region towards its area share by the same fraction."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, (8, 8))                       # every cell belongs to one region
    grids = np.stack([(labels == k).astype(float) for k in range(4)])
    area = grids.mean(axis=(1, 2))
    for lam in (0.1, 0.4, 0.8):
        clean = rng.gamma(1.0, 1.0, (8, 8)) * np.where(labels < 2, 6.0, 1.0)   # concentrated, like an eye map
        clean /= clean.sum()
        attacked = (1 - lam) * clean + lam * np.full((8, 8), 1 / 64)
        m_clean = np.array(list(region_mass(clean, grids).values()))
        m_attacked = np.array(list(region_mass(attacked, grids).values()))
        fit = fit_area_null((m_attacked - m_clean)[None], m_clean[None], area)
        assert fit["lam"][0] == pytest.approx(lam, abs=1e-12)
        np.testing.assert_allclose(fit["res"], 0.0, atol=1e-12)
        assert fit["r2"][0] == pytest.approx(1.0)


def test_relocation_leaves_a_residual_and_small_gaps_are_undefined():
    area = np.array([0.07, 0.14, 0.09, 0.70])
    fit = fit_area_null([[-0.1, 0.1, 0.0, 0.0]], [[0.2, 0.3, 0.2, 0.3]], area)   # pupil -> iris: not diffusion
    assert fit["r2"][0] < 0.01 and np.abs(fit["res"]).max() > 0.09
    flat = fit_area_null([[0.01, -0.01]], [[0.5, 0.5]], [0.51, 0.49])               # clean map ~ area shares
    assert np.isnan(flat["lam"][0]) and np.isnan(flat["r2"][0])


def test_enrichment():
    np.testing.assert_allclose(enrichment([0.2, 0.7, 0.1], [0.4, 0.7, 0.0]), [0.5, 1.0, np.nan])


def test_heuristic_periocular_area_is_the_same_for_every_image():
    for img in make_images(3):
        masks = get_concept_masks(img)
        assert masks["periocular"].mean() == pytest.approx(HEURISTIC_PERIOCULAR_AREA, abs=1e-12)
        assert masks_to_grid(masks, grid=8)[3].mean() == pytest.approx(HEURISTIC_PERIOCULAR_AREA, abs=1e-6)
    assert round(HEURISTIC_PERIOCULAR_AREA, 4) == 0.6925


def test_regions_follow_the_columns():
    cols = ["mass_skin", "dmass_skin", "massvis_skin", "dmassvis_skin", "mass_iris", "dmass_iris", "mass_lid"]
    assert regions_in(cols) == ["skin", "iris"]                 # mass_lid has no dmass_lid
    shuffled = [f"{p}_{r}" for r in ("iris", "periocular", "pupil", "sclera") for p in ("mass", "dmass")]
    assert regions_in(shuffled) == list(HEURISTIC_REGIONS)     # heuristic regions keep their usual order


def _region_rows():
    return pd.DataFrame({"image_id": [0, 0, 1], "true_idx": [5, 5, 6],
                         "mass_iris": [0.2, 0.3, 0.4], "dmass_iris": [0.0, 0.1, 0.1],
                         "mass_skin": [0.8, 0.7, 0.6], "dmass_skin": [0.0, -0.1, -0.1]})


def test_attach_areas_takes_columns_then_the_area_csv():
    rows, regions = _region_rows(), ["iris", "skin"]
    images = pd.DataFrame({"image_id": [0, 1], "link_path": ["/tmp/a.png", "/tmp/b.png"]})
    areas = pd.DataFrame({"relpath": ["x/a.png", "x/b.png"], "link_path": ["\\tmp\\a.png", "/tmp/b.png"],
                          "area_iris": [0.1, 0.2], "area_skin": [0.9, 0.8]})
    frame, source = attach_areas(rows, regions, areas, images)            # image_id -> link_path -> areas
    assert source == "areas_csv" and frame.area_iris.tolist() == [0.1, 0.1, 0.2]
    frame, _ = attach_areas(rows.assign(relpath=["x/b.png", "x/b.png", "x/a.png"]), regions, areas)
    assert frame.area_iris.tolist() == [0.2, 0.2, 0.1]                    # relpath wins when both have it
    assert attach_areas(frame, regions)[1] == "columns"
    assert attach_areas(rows, regions)[1] is None
    assert "area_iris" not in rows                                         # input untouched
    with pytest.raises(ValueError, match="no areas"):
        attach_areas(rows, regions, areas.iloc[:1], images)
    with pytest.raises(ValueError, match="images_<tag>"):
        attach_areas(rows, regions, areas.drop(columns="relpath"))
    with pytest.raises(ValueError, match="area_skin"):
        attach_areas(rows, regions, areas.drop(columns="area_skin"), images)


def _heuristic_rows(n=40, lam=0.25, seed=3):
    """Rows whose shift is pure diffusion with known areas: 2 rows per subject."""
    rng = np.random.default_rng(seed)
    clean = rng.dirichlet(np.full(4, 2.0), n)
    area = rng.dirichlet(np.full(4, 4.0), n)
    dm = lam * (area - clean)
    data = {"true_idx": np.arange(n) // 2, "group": "a"}
    for k, r in enumerate(HEURISTIC_REGIONS):
        data.update({f"mass_{r}": clean[:, k] + dm[:, k], f"dmass_{r}": dm[:, k], f"area_{r}": area[:, k]})
    return pd.DataFrame(data)


def test_null_table_with_per_image_areas():
    regions = list(HEURISTIC_REGIONS)
    frame = add_null_columns(_heuristic_rows(), regions, "columns")
    table = null_table(frame, ["group"], regions, "columns", n_boot=200)
    assert table.region.tolist() == regions and (table.area_source == "columns").all()
    assert table.lam_median.iloc[0] == pytest.approx(0.25) and table.r2.iloc[0] == pytest.approx(1.0)
    assert table.r2_chance.iloc[0] == pytest.approx(1 / 3)                # masses sum to 1: 3 free directions
    np.testing.assert_allclose(table.res_median, 0.0, atol=1e-12)
    assert table.lam_ci_low.iloc[0] <= 0.25 <= table.lam_ci_high.iloc[0]
    few = null_table(frame.head(6), ["group"], regions, "columns", n_boot=50, min_n=10)
    assert few.lam_ci_low.isna().all() and few.p_res.isna().all() and np.isfinite(few.lam_median).all()


def test_periocular_only_without_areas():
    rows = _heuristic_rows().drop(columns=[f"area_{r}" for r in HEURISTIC_REGIONS])
    frame = add_null_columns(rows, list(HEURISTIC_REGIONS), None)
    clean = rows.mass_periocular - rows.dmass_periocular
    gap = HEURISTIC_PERIOCULAR_AREA - clean
    expected = np.where(gap.abs() > 0.05, rows.dmass_periocular / gap, np.nan)
    np.testing.assert_allclose(frame.lam, expected)
    np.testing.assert_allclose(frame.enr_periocular, rows.mass_periocular / HEURISTIC_PERIOCULAR_AREA)
    table = null_table(frame, ["group"], list(HEURISTIC_REGIONS), None, n_boot=100)
    assert table.region.tolist() == ["periocular"] and table.area_source.iloc[0] == "periocular_only"
    assert np.isnan(table.r2.iloc[0]) and np.isnan(table.p_res.iloc[0])
    assert add_null_columns(_region_rows(), ["iris", "skin"], None) is None   # other masks need areas


def test_pooled_r2():
    dm = np.array([[0.1, -0.1], [0.2, -0.2]])
    res = np.array([[0.1, 0.0], [0.0, 0.0]])
    est, lo, hi = pooled_r2(dm, res, [0, 1], n_boot=100)
    assert est == pytest.approx(1 - 0.01 / 0.1) and lo <= est <= hi
