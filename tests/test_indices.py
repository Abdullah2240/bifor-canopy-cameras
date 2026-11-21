import numpy as np
import pytest

from bifor import calib, indices


def leaf(wl, red=0.05, nir=0.45, edge=715.0, width=12.0):
    """Smooth vegetation-like spectrum: low red, logistic red edge, high NIR."""
    return red + (nir - red) / (1 + np.exp(-(wl - edge) / width))


@pytest.fixture(scope="module")
def a5(cfg):
    return calib.load(cfg.calib_dir, "A5")


def test_ndvi_ndre_match_hand_calculation(cfg, a5):
    s = leaf(a5.wl)
    r = lambda nm: s[np.argmin(abs(a5.wl - nm))]
    ndvi = (r(794) - r(665)) / (r(794) + r(665))
    ndre = (r(794) - r(717)) / (r(794) + r(717))
    assert indices.compute_one("NDVI", s, a5.wl, cfg.band_tol) == pytest.approx(ndvi)
    assert indices.compute_one("NDRE", s, a5.wl, cfg.band_tol) == pytest.approx(ndre)
    assert 0.7 < ndvi < 0.9


def test_rep_tracks_red_edge_position(cfg, a5):
    lo = indices.compute_one("REP-LE", leaf(a5.wl, edge=705), a5.wl, cfg.band_tol)
    hi = indices.compute_one("REP-LE", leaf(a5.wl, edge=725), a5.wl, cfg.band_tol)
    assert 700 < lo < hi < 745
    assert hi - lo > 8


def test_rep_extrapolation_in_red_edge(cfg, a5):
    v = indices.compute_one("REP-extrap", leaf(a5.wl), a5.wl, cfg.band_tol)
    assert 695 < v < 740


@pytest.mark.parametrize("name", ["PSRI", "PRI"])
def test_out_of_range_indices_switched_off(cfg, a5, name):
    res = indices.resolve(indices.REGISTRY[name], a5.wl, cfg.band_tol)
    assert not res.enabled
    assert "nm" in res.reason


def test_water_band_index_off_because_sensor_tops_out_at_952(cfg, a5):
    res = indices.resolve(indices.REGISTRY["WI/WBI"], a5.wl, cfg.band_tol)
    assert not res.enabled and "962" in res.reason


def test_resolution_is_per_camera(cfg, a5):
    a6 = calib.load(cfg.calib_dir, "A6")
    r5 = indices.resolve(indices.REGISTRY["NDRE"], a5.wl, cfg.band_tol)
    r6 = indices.resolve(indices.REGISTRY["NDRE"], a6.wl, cfg.band_tol)
    assert r5.enabled and r6.enabled
    assert r5.bands != r6.bands


def test_ratios_survive_exposure_scaling(cfg, a5):
    s = leaf(a5.wl)
    v1 = indices.compute_all(s, a5.wl, cfg.band_tol)
    v2 = indices.compute_all(s * 3.7, a5.wl, cfg.band_tol)
    for name, idx in indices.REGISTRY.items():
        if name in v1 and idx.scale_invariant:
            assert v2[name] == pytest.approx(v1[name], rel=1e-9, abs=1e-12), name


def test_every_catalog_index_is_in_registry_or_explained():
    from bifor.indices import catalog_shorts
    missing = [s for s in catalog_shorts() if s not in indices.REGISTRY and s not in indices.NOT_COMPUTED]
    assert missing == []


def test_compute_works_on_pixel_arrays(cfg, a5):
    cube = np.broadcast_to(leaf(a5.wl)[:, None, None], (24, 4, 5))
    m = indices.compute_one("NDVI", cube, a5.wl, cfg.band_tol)
    assert m.shape == (4, 5)


@pytest.mark.parametrize("name", ["Ctr5", "Ctr2", "REP-extrap", "NDRE", "MTCI"])
def test_both_cameras_use_the_same_filter_slot(cfg, name):
    a5, a6 = calib.load(cfg.calib_dir, "A5"), calib.load(cfg.calib_dir, "A6")
    r5 = indices.resolve(indices.REGISTRY[name], a5.wl, cfg.band_tol)
    r6 = indices.resolve(indices.REGISTRY[name], a6.wl, cfg.band_tol)
    assert [abs(x - y) < 2.0 for x, y in zip(r5.bands, r6.bands)] == [True] * len(r5.bands)
