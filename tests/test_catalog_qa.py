import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bifor import catalog, qa


def make_tree(root):
    for cam, pre in (("057400066245", "45"), ("057400066246", "46")):
        d = root / "some" / cam
        d.mkdir(parents=True)
        (d / f"{pre}-image_20251205_120000.png").write_bytes(b"x")
        (d / f"{pre}-image_20260210_120000.tiff").write_bytes(b"x")
        (d / f"{pre}-image_20260515_120000_cube.tiff").write_bytes(b"x")
        (d / f"{pre}-image_20260515_120000_raw.tiff").write_bytes(b"x")
        (d / "notes.txt").write_bytes(b"x")


def test_scan_pairs_cube_and_raw_and_tags_formats(cfg, tmp_path):
    make_tree(tmp_path)
    df = catalog.scan(tmp_path, cfg)
    assert len(df) == 6
    may = df[(df.camera == "A5") & (df.ts_local.dt.month == 5)].iloc[0]
    assert may.format == "pair" and may.read_path.endswith("_cube.tiff") and may.raw_path.endswith("_raw.tiff")
    assert set(df.format) == {"png", "tiff", "pair"}
    assert df.frame_id.is_unique


def test_local_time_converted_to_utc_across_clock_change():
    local = pd.Series(pd.to_datetime(["2025-12-05 12:00", "2026-05-15 12:00", "2026-03-29 01:30"]))
    uk = catalog.to_utc(local, "Europe/London")
    assert uk.iloc[0].hour == 12 and uk.iloc[1].hour == 11   # GMT in winter, BST in May
    assert pd.isna(uk.iloc[2])                               # hour skipped by the clock change


def test_utc_filenames_keep_the_clock_change_hour(cfg, tmp_path):
    assert cfg.filename_tz == "UTC"
    d = tmp_path / "057400066245"
    d.mkdir()
    (d / "45-image_20260329_013000_cube.tiff").write_bytes(b"x")
    df = catalog.scan(tmp_path, cfg)
    assert len(df) == 1 and df.ts_utc.iloc[0].hour == 1


def test_sun_position_columns(cfg):
    ts = pd.Series(pd.to_datetime(["2026-06-21 12:10", "2026-12-21 12:10", "2026-06-21 00:00"], utc=True))
    s = qa.sun_position(ts, cfg.lat, cfg.lon)
    assert s.sun_elev.iloc[0] == pytest.approx(60.6, abs=0.6)
    assert s.sun_elev.iloc[1] == pytest.approx(13.8, abs=0.6)
    assert s.sun_elev.iloc[2] < 0
    assert abs(s.min_from_noon.iloc[0]) < 10


def test_shift_detection_finds_injected_offset():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(217, 409)).cumsum(0).cumsum(1)
    moved = np.roll(ref, (3, -2), axis=(0, 1))
    dy, dx = qa.shift(ref, moved)
    assert (round(dy), round(dx)) == (3, -2)


def test_sharpness_drops_when_blurred():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    img = rng.normal(100, 10, (217, 409))
    assert qa.sharpness(gaussian_filter(img, 3)) < qa.sharpness(img) / 5


def test_apply_qa_flags_each_reason(cfg):
    from dataclasses import replace
    rules = replace(cfg, qa={**cfg.qa, "min_sun_elev": 10.0, "max_sat_frac": 0.25, "max_light_jump": 0.15,
                             "max_cloud_pct": 85.0, "max_shift_px": 2.0, "min_sharpness": 0.0})
    base = dict(sun_elev=40.0, sat_frac=0.0, light_jump=0.0, cloud_pct=10.0, precip_mm=0.0, shift_px=0.0, sharpness=1.0)
    rows = [base, {**base, "sun_elev": 5.0}, {**base, "sat_frac": 0.5}, {**base, "light_jump": 0.5},
            {**base, "cloud_pct": 99.0}, {**base, "shift_px": 5.0}]
    out = qa.apply(pd.DataFrame(rows), rules)
    assert out.qa_pass.tolist() == [True, False, False, False, False, False]
    assert out.qa_reason.iloc[1] == "low_sun" and out.qa_reason.iloc[5] == "camera_shift"


def test_light_jump_vs_neighbours():
    t = pd.date_range("2026-05-01 10:00", periods=7, freq="10min", tz="UTC")
    df = pd.DataFrame({"camera": "A5", "ts_utc": t, "light_ratio": [1, 1, 1, 2, 1, 1, 1.0]})
    j = qa.light_jump(df)
    assert j.iloc[3] > 0.5 and j.drop(index=3).max() < 0.05


def test_masked_shift_finds_offset_of_masked_structure():
    """Pylon-only registration must track the pylons, not the mask outline."""
    rng = np.random.default_rng(5)
    scene = rng.normal(100, 5, (217, 409))
    mask = np.zeros((217, 409), bool)
    for x in (40, 120, 200, 280, 360):
        mask[20:200, x:x + 6] = True
        mask[60:64, x - 15:x + 21] = True
    ref = np.where(mask, 20.0, scene)
    moved = np.roll(np.where(mask, 20.0, rng.normal(100, 5, (217, 409))), (3, -2), axis=(0, 1))
    from scipy.ndimage import binary_dilation
    dy, dx = qa.shift(ref, moved, ref_mask=binary_dilation(mask, iterations=3))
    assert (round(dy), round(dx)) == (3, -2)


def test_masked_shift_reports_no_match_when_offset_is_outside_window(cfg):
    """Real offsets are -4..+1 px. A match on the search-window edge means the
    true peak lies outside it (on A5 the ~8 px pylon lattice repeat), so it is
    reported as no match rather than as an edge value."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    rng = np.random.default_rng(9)
    ref = gaussian_filter(rng.normal(0, 1, (217, 409)), 4) * 100 + 100
    mask = np.zeros((217, 409), bool)
    for x in (40, 120, 200, 280, 360):
        mask[20:200, x:x + 12] = True
    moved = np.roll(ref, (8, 1), axis=(0, 1))
    dy, dx = qa.shift(ref, moved, ref_mask=binary_dilation(mask, iterations=2), max_px=cfg.qa["max_search_px"])
    assert np.isnan(dy) and np.isnan(dx)
    dy, dx = qa.shift(ref, np.roll(ref, (-2, 1), axis=(0, 1)), ref_mask=mask, max_px=cfg.qa["max_search_px"])
    assert (dy, dx) == (-2.0, 1.0)


def test_day_offset_falls_back_when_no_frame_registers(cfg):
    from bifor import pipeline
    assert pipeline.day_offset([(np.nan, np.nan)] * 3, cfg, "A5") == tuple(cfg.qa["fallback_shift"]["A5"])
    assert pipeline.day_offset([(-2.0, 1.0), (np.nan, np.nan), (-1.0, 1.0), (-2.0, 0.0)], cfg, "A5") == (-2.0, 1.0)


def test_fog_and_frost_rules_from_labelled_frames(cfg):
    from dataclasses import replace
    rules = replace(cfg, qa={**cfg.qa, "min_sharpness": {"A5": 0.065, "A6": 0.08},
                             "max_sat_frac_leafless": 0.05, "leafless_months": [11, 12, 1, 2, 3]})
    base = dict(sun_elev=30.0, sat_frac=0.0, light_jump=0.0, cloud_pct=50.0, precip_mm=0.0, shift_px=0.0, sharpness=0.2)
    rows = [
        {**base, "camera": "A5", "ts_utc": pd.Timestamp("2026-01-08 12:00", tz="UTC")},                      # clear
        {**base, "camera": "A5", "ts_utc": pd.Timestamp("2026-01-08 12:00", tz="UTC"), "sharpness": 0.05},   # fog
        {**base, "camera": "A6", "ts_utc": pd.Timestamp("2026-01-08 12:00", tz="UTC"), "sharpness": 0.07},   # fog on A6
        {**base, "camera": "A5", "ts_utc": pd.Timestamp("2026-01-08 12:00", tz="UTC"), "sharpness": 0.07},   # fine on A5
        {**base, "camera": "A5", "ts_utc": pd.Timestamp("2026-01-02 12:00", tz="UTC"), "sat_frac": 0.10},    # frost
        {**base, "camera": "A5", "ts_utc": pd.Timestamp("2026-05-20 12:00", tz="UTC"), "sat_frac": 0.10},    # sunlit leaves
    ]
    out = qa.apply(pd.DataFrame(rows), rules)
    assert out.qa_reason.tolist() == ["", "blur_fog", "blur_fog", "", "frost_snow", ""]
