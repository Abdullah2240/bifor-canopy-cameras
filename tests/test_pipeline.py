import pandas as pd
import pytest

from bifor import pipeline, store


@pytest.fixture
def local_run(cfg, tmp_path):
    if not (cfg.local_sample / "057400066245").exists():
        pytest.skip("local cubes missing")
    work = tmp_path / "work"
    cat = pipeline.scan(cfg, raw_root=cfg.local_sample, work_dir=work)
    return work, cat


def test_scan_finds_local_frames_both_cameras(local_run):
    _, cat = local_run
    assert set(cat.camera) == {"A5", "A6"}
    assert len(cat) == 40


def test_process_writes_tables_and_rerun_is_idempotent(cfg, local_run):
    work, _ = local_run
    n1 = pipeline.process(cfg, work_dir=work, workers=1)
    sizes = {t: len(store.read(work / t)) for t in ("frames", "roi_spectra", "indices")}
    n2 = pipeline.process(cfg, work_dir=work, workers=1)
    assert n1 > 0 and n2 == 0
    assert sizes == {t: len(store.read(work / t)) for t in ("frames", "roi_spectra", "indices")}
    ind = store.read(work / "indices")
    assert {"code_v", "roi_v", "calib_v"} <= set(ind.columns)
    ndvi = ind[(ind["index"] == "NDVI") & (ind.roi == "canopy_lower")].value
    assert ndvi.between(0.0, 0.5).all()


def test_version_bump_reprocesses(cfg, local_run):
    work, _ = local_run
    pipeline.process(cfg, work_dir=work, workers=1, limit=2)
    from dataclasses import replace
    bumped = replace(cfg, versions={**cfg.versions, "code": "9.9.9"})
    assert pipeline.process(bumped, work_dir=work, workers=1, limit=2) == 2


def _night_tree(root, levels):
    import numpy as np
    import tifffile
    d = root / "057400066245"
    d.mkdir(parents=True)
    for k, v in enumerate(levels):
        cube = np.full((25, 217, 409), v * 64, np.uint16)
        tifffile.imwrite(d / f"45-image_202601{10 + k:02d}_000500_cube.tiff", cube)
    return root


def test_master_dark_ignores_bright_nights(cfg, tmp_path):
    raw = _night_tree(tmp_path / "raw", [10, 10, 11, 12, 70])
    work = tmp_path / "work"
    pipeline.scan(cfg, raw_root=raw, work_dir=work)
    out = pipeline.build_master_darks(cfg, work)
    import numpy as np
    dark = np.load(out[("A5", 16)])
    assert dark.shape == (25, 217, 409)
    assert float(dark.mean()) == pytest.approx(10.0, abs=0.6)


def test_format_bit_depth_mismatch_is_recorded(cfg, tmp_path):
    import numpy as np
    import tifffile
    p = tmp_path / "45-image_20260210_120000.tiff"
    tifffile.imwrite(p, np.full((1088, 2048), 500, np.uint16))   # tiff era should be 8-bit
    row = {"frame_id": "A5_x", "ts_utc": pd.Timestamp("2026-02-10 12:00", tz="UTC"), "sun_elev": 20.0,
           "min_from_noon": 0.0, "format": "tiff", "read_path": str(p)}
    f, _, _ = pipeline.process_day(cfg, "A5", [row], None, None)
    assert "bit depth" in f.error.iloc[0]


def test_in_window_drops_frames_before_record_start(cfg):
    from dataclasses import replace
    c = replace(cfg, window={"start_utc": "2026-03-10T10:31:00Z"})
    df = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-02-01 12:00", "2026-03-10 10:30", "2026-03-10 10:31",
                                                 "2026-05-01 12:00"], utc=True)})
    kept = pipeline.in_window(df, c)
    assert list(kept.ts_utc.dt.strftime("%m-%d %H:%M")) == ["03-10 10:31", "05-01 12:00"]
    assert len(pipeline.in_window(df, replace(cfg, window={}))) == 4
