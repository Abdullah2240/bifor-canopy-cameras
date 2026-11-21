import numpy as np
import pytest
import tifffile

from bifor import calib, reader
from tests.conftest import FIXTURES


def test_raw_demosaic_equals_cube(local_pair):
    cube_path, raw_path = local_pair
    cube = tifffile.imread(cube_path)
    bands, bit = reader.read_bands(raw_path)
    assert bit == 16
    assert np.array_equal(bands, cube)


def test_cube_and_raw_decode_identically(cfg, local_pair):
    cal = calib.load(cfg.calib_dir, "A5")
    a = reader.decode(local_pair[0], cal, None, cfg)
    b = reader.decode(local_pair[1], cal, None, cfg)
    np.testing.assert_array_equal(a.bands, b.bands)
    assert a.bands.shape == (24, reader.H, reader.W)


def test_saturated_pixels_masked_in_16bit(cfg, local_pair):
    cal = calib.load(cfg.calib_dir, "A5")
    raw, _ = reader.read_bands(local_pair[0])
    d = reader.decode(local_pair[0], cal, None, cfg)
    expected = (raw >= cfg.sat_16bit).any(0)
    assert d.sat.sum() == expected.sum() > 0
    assert np.isnan(d.bands[:, d.sat]).all()


def test_8bit_mosaic_scaled_to_10bit_and_255_is_saturated(cfg, tmp_path):
    m = np.full((1088, 2048), 100, np.uint8)
    m[0, 0] = 255
    p = tmp_path / "45-image_20251205_120000.png"
    from PIL import Image
    Image.fromarray(m).save(p)
    bands, bit = reader.read_bands(p)
    assert bit == 8
    d = reader.decode(p, calib.load(cfg.calib_dir, "A5"), None, cfg)
    assert d.sat[0, 0] and d.sat.sum() == 1
    assert d.dn_mean == pytest.approx(400, rel=0.01)


def test_dark_subtraction_removes_offset(cfg, local_pair, local_dark):
    cal = calib.load(cfg.calib_dir, "A5")
    dark = reader.load_dark(local_dark)
    a = reader.decode(local_pair[0], cal, None, cfg)
    b = reader.decode(local_pair[0], cal, dark, cfg)
    assert dark.mean() > 0
    assert np.nanmean(a.bands) != pytest.approx(np.nanmean(b.bands))


def test_no_per_band_normalisation(cfg, local_pair):
    """Corrected bands must equal M @ dn exactly: no hidden rescaling per band."""
    cal = calib.load(cfg.calib_dir, "A5")
    raw, _ = reader.read_bands(local_pair[0])
    dn = raw.astype(np.float64) / 64.0
    expected = (cal.M @ dn.reshape(25, -1)).reshape(24, reader.H, reader.W)
    d = reader.decode(local_pair[0], cal, None, cfg)
    ok = ~d.sat
    np.testing.assert_allclose(d.bands[:, ok], expected[:, ok], rtol=1e-4, atol=1e-3)


def test_wrong_shape_fails_loudly(tmp_path):
    p = tmp_path / "x_20260101_120000.tiff"
    tifffile.imwrite(p, np.zeros((100, 100), np.uint16))
    with pytest.raises(ValueError, match="2048"):
        reader.read_bands(p)


def test_camera_wavelengths_differ(cfg):
    a5, a6 = calib.load(cfg.calib_dir, "A5"), calib.load(cfg.calib_dir, "A6")
    assert a5.M.shape == (24, 25) and a5.wl.shape == (24,)
    diff = a5.wl - a6.wl
    assert 0.3 < diff.max() < 2.0 and diff.min() > -0.5


@pytest.mark.skipif(not (FIXTURES / "may").exists(), reason="May fixture not copied from D: yet")
def test_may_canopy_ndvi_above_point3(cfg):
    from bifor import indices, roi
    cube = sorted((FIXTURES / "may").glob("*45-*_cube.tiff"))[0]
    dark = sorted((FIXTURES / "may").glob("*45-*dark*"))[0]
    cal = calib.load(cfg.calib_dir, "A5")
    d = reader.decode(cube, cal, reader.load_dark(dark), cfg)
    masks = roi.load(cfg.roi_dir, "A5", cfg.versions["roi"])
    spec = np.nanmedian(d.bands[:, masks["canopy_lower"]], axis=1)
    assert indices.compute_one("NDVI", spec, cal.wl, cfg.band_tol) > 0.3
