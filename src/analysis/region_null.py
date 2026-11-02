"""
src/analysis/region_null.py

Area-baseline null for region shifts: does an attack move heatmap mass to a
particular eye region, or does the map just lose its structure?

A map that spreads out (or decorrelates from the image) drifts towards a
uniform map, and a uniform map puts on each region exactly the region's area
share. Under that null every region moves towards its area share by the same
fraction ``lam`` of its gap:

    dmass_r = lam * (area_r - clean_mass_r)

``lam`` is fitted per row by least squares over the regions. What the null
cannot explain is the residual ``res_r = dmass_r - lam * (area_r - clean_mass_r)``;
``R^2`` says how much of the shift the null explains. Enrichment
``mass_r / area_r`` compares a region's mass with what a uniform map would give
it (1 = the same).

Areas are shares of the map grid, averaged like src/compare/regions.py does.
With the heuristic masks (src/xai/masks.py), the periocular region is the
outside of one fixed ellipse, so its area is the same for every image
(:data:`HEURISTIC_PERIOCULAR_AREA`), while pupil, iris and sclera areas vary
per image and were not stored by the 27 Oct run. Without per-image areas only
the periocular enrichment and ``lam`` (the share of the gap to a uniform map
that the shift closes) are exact, and the null cannot be tested: inside versus
outside the ellipse are two regions, and one ``lam`` always fits two regions
perfectly.
"""

import numpy as np
import pandas as pd

from src.analysis.stats import (cluster_bootstrap_counts, cluster_bootstrap_median, finite_median, holm,
                                paired_subject_ci, paired_subject_test)

HEURISTIC_REGIONS = ("pupil", "iris", "sclera", "periocular")
# 20,155 of the 65,536 pixels lie inside the fixed eye-opening ellipse of src/xai/masks.py
# (centre (128, 120), half-axes (99, 64)); "periocular" is everything outside it.
HEURISTIC_PERIOCULAR_AREA = 1.0 - 20155 / 65536          # 0.692459
MIN_GAP = 0.05            # below this distance to the area shares the direction of the null is undefined


def regions_in(columns):
    """Regions with both a ``mass_<region>`` and a ``dmass_<region>`` column: the heuristic
    regions in their usual order (pupil, iris, sclera, periocular), others in column order."""
    columns = list(columns)
    found = [c[len("mass_"):] for c in columns if c.startswith("mass_") and f"d{c}" in columns]
    return list(HEURISTIC_REGIONS) if set(found) == set(HEURISTIC_REGIONS) else found


def enrichment(mass, area):
    """``mass / area`` (1 = as much mass as a uniform map); NaN where the area is 0."""
    mass, area = np.asarray(mass, np.float64), np.asarray(area, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, mass / np.where(area > 0, area, 1.0), np.nan)


def fit_area_null(dmass, clean_mass, area, min_gap=MIN_GAP):
    """Least-squares fit of ``dmass_r = lam * (area_r - clean_mass_r)`` for every row.

    Parameters
    ----------
    dmass, clean_mass : array (N, R) — change of region mass and mass of the clean map
    area : array (N, R) or (R,) — area share of each region
    min_gap : rows whose clean masses lie within this (Euclidean) distance of the
        area shares get NaN: the null has no direction to move them in

    Returns
    -------
    dict with ``lam`` (N,), ``pred`` (N, R) = lam * (area - clean_mass),
    ``res`` (N, R) = dmass - pred and ``r2`` (N,) = 1 - |res|^2 / |dmass|^2
    (NaN where dmass is all zero or lam is undefined)
    """
    dm = np.atleast_2d(np.asarray(dmass, np.float64))
    gap = np.broadcast_to(np.asarray(area, np.float64), dm.shape) - np.atleast_2d(np.asarray(clean_mass, np.float64))
    gg = (gap * gap).sum(1)
    lam = np.where(np.sqrt(gg) > min_gap, (gap * dm).sum(1) / np.where(gg > 0, gg, 1.0), np.nan)
    pred = lam[:, None] * gap
    res = dm - pred
    ss = (dm * dm).sum(1)
    r2 = np.where((ss > 0) & np.isfinite(lam), 1.0 - (res * res).sum(1) / np.where(ss > 0, ss, 1.0), np.nan)
    return {"lam": lam, "pred": pred, "res": res, "r2": r2}


def _path_key(series):
    return series.astype(str).str.replace("\\", "/", regex=False)


def attach_areas(rows, regions, areas=None, images=None):
    """Copy of *rows* with ``area_<region>`` columns, and where they came from.

    Sources, in order: ``area_<region>`` columns already in the rows (the v2 runner
    writes them); else a per-image area CSV (*areas*: relpath and/or link_path plus
    ``area_<region>``), joined on relpath when both tables have it, else on link_path,
    which rows that only carry image_id get from *images* (the run's images_<tag>.csv).

    Returns
    -------
    (frame, source) with source "columns", "areas_csv", or None when there are no
    per-image areas

    Raises
    ------
    ValueError if the area CSV lacks a region, cannot be joined, or misses images
    """
    need = [f"area_{r}" for r in regions]
    if all(c in rows for c in need):
        return rows.copy(), "columns"
    if areas is None:
        return rows.copy(), None
    missing = [c for c in need if c not in areas]
    if missing:
        raise ValueError(f"the area CSV has no column(s) {missing}")
    frame = rows.drop(columns=[c for c in need if c in rows])
    if "relpath" in frame and "relpath" in areas:
        key = "relpath"
    elif "link_path" in areas:
        key = "link_path"
        if "link_path" not in frame:
            if images is None or not {"image_id", "link_path"} <= set(images.columns):
                raise ValueError("the rows have neither relpath nor link_path: pass the run's "
                                 "images_<tag>.csv to map image_id to link_path")
            frame = frame.merge(images[["image_id", "link_path"]].drop_duplicates("image_id"),
                                on="image_id", how="left", validate="many_to_one")
    else:
        raise ValueError("the area CSV needs a relpath column (rows with relpath) or a link_path column")
    table = areas[[key] + need].copy()
    table[key] = _path_key(table[key])
    if table.duplicated(key).any():
        raise ValueError(f"the area CSV lists some {key} more than once")
    frame["_key"] = _path_key(frame[key])
    frame = frame.merge(table.rename(columns={key: "_key"}), on="_key", how="left").drop(columns="_key")
    lost = frame[need].isna().any(axis=1)
    if lost.any():
        raise ValueError(f"{int(lost.sum())} rows ({frame.loc[lost, key].nunique()} images) have no "
                         f"areas in the area CSV")
    return frame, "areas_csv"


def add_null_columns(rows, regions, area_source, min_gap=MIN_GAP):
    """Copy of *rows* with the per-row null, or None if it cannot be computed.

    With per-image areas (*area_source* not None): ``lam``, ``r2_null`` and, per
    region, ``pred_<r>``, ``res_<r>``, ``enr_<r>`` and ``enr_clean_<r>``. Without
    them, for the heuristic regions only: ``lam`` (gap closed) and
    ``enr_periocular`` / ``enr_clean_periocular`` from the exact periocular area.
    """
    frame = rows.copy()
    mass = frame[[f"mass_{r}" for r in regions]].to_numpy(np.float64)
    dm = frame[[f"dmass_{r}" for r in regions]].to_numpy(np.float64)
    clean = mass - dm
    if area_source is not None:
        area = frame[[f"area_{r}" for r in regions]].to_numpy(np.float64)
        fit = fit_area_null(dm, clean, area, min_gap)
        frame["lam"], frame["r2_null"] = fit["lam"], fit["r2"]
        for k, r in enumerate(regions):
            frame[f"pred_{r}"], frame[f"res_{r}"] = fit["pred"][:, k], fit["res"][:, k]
            frame[f"enr_{r}"] = enrichment(mass[:, k], area[:, k])
            frame[f"enr_clean_{r}"] = enrichment(clean[:, k], area[:, k])
        return frame
    if set(regions) != set(HEURISTIC_REGIONS):
        return None
    k = list(regions).index("periocular")
    fit = fit_area_null(dm[:, [k]], clean[:, [k]], [HEURISTIC_PERIOCULAR_AREA], min_gap)
    frame["lam"] = fit["lam"]
    frame["area_periocular"] = HEURISTIC_PERIOCULAR_AREA
    frame["enr_periocular"] = mass[:, k] / HEURISTIC_PERIOCULAR_AREA
    frame["enr_clean_periocular"] = clean[:, k] / HEURISTIC_PERIOCULAR_AREA
    return frame


def pooled_r2(dmass, res, groups, n_boot=2000, level=0.95, seed=0):
    """R^2 of the null pooled over rows and regions, 1 - sum(res^2) / sum(dmass^2), with a
    subject-cluster bootstrap CI. Rows with any NaN are left out.

    Returns
    -------
    (estimate, low, high) : floats
    """
    dm, res, groups = np.asarray(dmass, np.float64), np.asarray(res, np.float64), np.asarray(groups)
    keep = np.isfinite(dm).all(1) & np.isfinite(res).all(1)
    if not keep.any():
        return float("nan"), float("nan"), float("nan")
    codes, counts = cluster_bootstrap_counts(groups[keep], n_boot, seed)
    ss_res = np.bincount(codes, weights=(res[keep] ** 2).sum(1))
    ss_tot = np.bincount(codes, weights=(dm[keep] ** 2).sum(1))
    if ss_tot.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = 1.0 - (counts @ ss_res) / (counts @ ss_tot)
    alpha = (1.0 - level) / 2
    return (float(1.0 - ss_res.sum() / ss_tot.sum()), float(np.nanquantile(boots, alpha)),
            float(np.nanquantile(boots, 1 - alpha)))


def null_table(rows, keys, regions, area_source, n_boot=2000, seed=0, min_n=10):
    """Summary of the per-row null (rows from :func:`add_null_columns`) per group and region.

    Group columns: n, n_subjects, n_fit (rows with a defined lam), lam median with CI and,
    with per-image areas, the pooled R^2 with CI and its chance level (1 / the number of
    free directions: regions - 1 when the masses sum to 1). Region columns: medians of
    area, clean mass, mass, enrichment (with CI) and dmass; the share of maps above the
    area share; and, with per-image areas, the null's prediction and the residual (median
    with CI, per-subject mean with CI, Wilcoxon p vs 0, Holm within the table). Groups with
    fewer than *min_n* rows get medians only (no CI or p).
    """
    shown = list(regions) if area_source is not None else ["periocular"]

    def median_ci(values, subjects, enough):
        if not enough:
            return finite_median(values), np.nan, np.nan
        return cluster_bootstrap_median(values, subjects, n_boot, seed=seed)

    out = []
    for key, g in rows.groupby(list(keys), dropna=False, sort=True):
        subjects = g.true_idx.to_numpy()
        lam = g.lam.to_numpy(np.float64)
        enough = len(g) >= min_n
        base = {**dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                "area_source": area_source or "periocular_only", "n": len(g),
                "n_subjects": int(g.true_idx.nunique()), "n_fit": int(np.isfinite(lam).sum())}
        est, lo, hi = median_ci(lam, subjects, enough)
        base.update({"lam_median": est, "lam_ci_low": lo, "lam_ci_high": hi,
                     "r2": np.nan, "r2_ci_low": np.nan, "r2_ci_high": np.nan, "r2_chance": np.nan})
        if area_source is not None:
            dm = g[[f"dmass_{r}" for r in regions]].to_numpy(np.float64)
            res = g[[f"res_{r}" for r in regions]].to_numpy(np.float64)
            fitted = np.isfinite(lam)
            r2, r2_lo, r2_hi = pooled_r2(dm[fitted], res[fitted], subjects[fitted], n_boot, seed=seed)
            sums = g[[f"mass_{r}" for r in regions]].to_numpy(np.float64).sum(1)
            free = len(regions) - 1 if np.allclose(sums, 1.0, atol=1e-4) else len(regions)
            base.update({"r2": r2, "r2_ci_low": r2_lo if enough else np.nan, "r2_ci_high": r2_hi if enough else np.nan,
                         "r2_chance": 1.0 / free if free > 0 else np.nan})
        for region in shown:
            mass, dmass = g[f"mass_{region}"].to_numpy(np.float64), g[f"dmass_{region}"].to_numpy(np.float64)
            area = g[f"area_{region}"].to_numpy(np.float64)
            enr = g[f"enr_{region}"].to_numpy(np.float64)
            e, e_lo, e_hi = median_ci(enr, subjects, enough)
            row = {**base, "region": region, "area_median": finite_median(area),
                   "clean_mass_median": finite_median(mass - dmass), "mass_median": finite_median(mass),
                   "enr_clean_median": finite_median(g[f"enr_clean_{region}"]),
                   "enr_median": e, "enr_ci_low": e_lo, "enr_ci_high": e_hi,
                   "frac_above_area": float(np.mean(mass > area)),
                   "frac_clean_above_area": float(np.mean(mass - dmass > area)),
                   "dmass_median": finite_median(dmass),
                   "pred_median": np.nan, "res_median": np.nan, "res_ci_low": np.nan, "res_ci_high": np.nan,
                   "res_subject_mean": np.nan, "res_subject_ci_low": np.nan, "res_subject_ci_high": np.nan,
                   "p_res": np.nan}
            if area_source is not None:
                res_r = g[f"res_{region}"].to_numpy(np.float64)
                r_est, r_lo, r_hi = median_ci(res_r, subjects, enough)
                row.update({"pred_median": finite_median(g[f"pred_{region}"]),
                            "res_median": r_est, "res_ci_low": r_lo, "res_ci_high": r_hi})
                if enough:
                    s_est, s_lo, s_hi = paired_subject_ci(res_r, 0.0, subjects, n_boot, seed=seed)
                    test = paired_subject_test(res_r, np.zeros(len(res_r)), subjects)
                    row.update({"res_subject_mean": s_est, "res_subject_ci_low": s_lo,
                                "res_subject_ci_high": s_hi, "p_res": test["p"]})
            out.append(row)
    table = pd.DataFrame(out)
    if len(table):
        table["p_res_holm"] = holm(table.p_res)
    return table
