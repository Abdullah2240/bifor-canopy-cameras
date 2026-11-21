"""Stages: scan -> roi reference -> process frames -> daily -> fit."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import binary_dilation

from . import calib, catalog, indices, qa, reader, roi, store, timeseries, weather

NIR_NM = 800.0


def _work(cfg, work_dir) -> Path:
    w = Path(work_dir or cfg.work_dir)
    w.mkdir(parents=True, exist_ok=True)
    return w


def log_run(work: Path, stage: str, **info) -> None:
    with open(work / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": stage, "at": pd.Timestamp.now(tz="UTC").isoformat(), **info}, default=str) + "\n")


# ---------- scan ----------

def scan(cfg, raw_root=None, work_dir=None) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    t = time.time()
    cat = catalog.scan(Path(raw_root or cfg.raw_root), cfg)
    cat = pd.concat([cat, qa.sun_position(cat.ts_utc, cfg.lat, cfg.lon)], axis=1)
    cat.to_parquet(work / "catalog.parquet", index=False)
    log_run(work, "scan", frames=len(cat), seconds=round(time.time() - t, 1),
            by_camera=cat.camera.value_counts().to_dict(), by_format=cat.format.value_counts().to_dict())
    return cat


def sample(cfg, every_days: int = 14, work_dir=None, seed: int = 0) -> pd.DataFrame:
    """Pilot subset: every Nth calendar day per camera (all frames of that day)."""
    work = _work(cfg, work_dir)
    cat = pd.read_parquet(work / "catalog.parquet")
    day = cat.ts_local.dt.normalize()
    start = day.min()
    pick = ((day - start).dt.days % every_days) == 0
    sub = cat[pick]
    sub.to_parquet(work / "catalog_sample.parquet", index=False)
    log_run(work, "sample", every_days=every_days, days=int(sub.ts_local.dt.date.nunique()), frames=len(sub))
    return sub


# ---------- darks ----------

FORMAT_BITS = {"png": 8, "tiff": 8, "pair": 16, "raw": 16}


def build_master_darks(cfg, work_dir=None) -> dict[tuple[str, int], Path]:
    """Per camera and bit depth: per-pixel median of the darkest frame of each
    night, keeping only nights at or below the configured quantile of nightly
    means. Bright nights (twilight in late May, A5's December nights) drop out."""
    work = _work(cfg, work_dir)
    out_dir = work / "darks"
    out_dir.mkdir(exist_ok=True)
    cat = pd.read_parquet(work / "catalog.parquet")
    night = cat[cat.sun_elev < -12].copy()
    night["night"] = (night.ts_utc - pd.Timedelta(hours=12)).dt.date
    pick = night.loc[night.groupby(["camera", "night"]).sun_elev.idxmin()]
    pick = pick.assign(bits=pick.format.map(FORMAT_BITS))
    means = []
    for r in pick.itertuples():
        means.append(float(reader.load_dark(Path(r.read_path)).mean()))
    pick["dark_mean"] = means
    paths = {}
    for (cam, bits), g in pick.groupby(["camera", "bits"]):
        keep = g[g.dark_mean <= g.dark_mean.quantile(cfg.master_dark_quantile)]
        stack = np.stack([reader.load_dark(Path(p)) for p in keep.read_path])
        p = out_dir / f"{cam}_{bits}bit.npy"
        np.save(p, np.median(stack, axis=0).astype(np.float32))
        paths[(cam, int(bits))] = p
        pick.loc[g.index, "used"] = g.index.isin(keep.index)
    pick.drop(columns=["read_path", "raw_path", "cube_path", "mosaic_path"]).to_csv(out_dir / "nights.csv", index=False)
    log_run(work, "master_darks", groups={f"{c}_{b}": int(len(g)) for (c, b), g in pick.groupby(["camera", "bits"])})
    return paths


def master_dark(work: Path, camera: str, fmt: str) -> tuple[Path | None, str | None]:
    bits = FORMAT_BITS[fmt]
    p = work / "darks" / f"{camera}_{bits}bit.npy"
    return (p, f"master_{camera}_{bits}bit") if p.exists() else (None, None)


def pick_darks(cat: pd.DataFrame, cfg) -> pd.DataFrame:
    """One dark frame per camera per local night: the frame nearest dark_hour_local,
    sun well below the horizon."""
    night = cat[(cat.sun_elev < -12) & (cat.ts_local.dt.hour.between(0, 4))].copy()
    if night.empty:
        return pd.DataFrame(columns=["camera", "date", "format", "dark_path", "dark_id"])
    night["date"] = night.ts_local.dt.normalize()
    night["dist"] = (night.ts_local.dt.hour * 60 + night.ts_local.dt.minute - cfg.dark_hour_local * 60).abs()
    best = night.sort_values("dist").groupby(["camera", "date"], as_index=False).first()
    return best.rename(columns={"read_path": "dark_path", "frame_id": "dark_id"})[
        ["camera", "date", "format", "dark_path", "dark_id"]]


def _dark_for(darks: pd.DataFrame, camera: str, date: pd.Timestamp, fmt: str):
    d = darks[darks.camera == camera]
    if d.empty:
        return None, None
    same = d[d.format == fmt]
    d = same if not same.empty else d
    i = (d.date - date).abs().idxmin()
    return d.loc[i, "dark_path"], d.loc[i, "dark_id"]


# ---------- per-day worker ----------

def day_offset(shifts: list[tuple[float, float]], cfg, camera: str) -> tuple[float, float]:
    """Median of the frames that registered; the camera's typical offset if none did
    (fog, flat light, or a day of only one or two frames)."""
    ok = [(dy, dx) for dy, dx in shifts if np.isfinite(dy) and np.isfinite(dx)]
    if not ok:
        fb = cfg.qa["fallback_shift"][camera]
        return float(fb[0]), float(fb[1])
    a = np.array(ok)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def process_day(cfg, camera: str, rows: list[dict], dark_path, dark_id):
    cal = calib.load(cfg.calib_dir, camera)
    rv = cfg.versions["roi"]
    masks = roi.load(cfg.roi_dir, camera, rv)
    ref = roi.reference(cfg.roi_dir, camera, rv)
    # Register on pylons and their edges only: leaves change all season, the steel does not.
    struct = binary_dilation(roi.exclusion(cfg.roi_dir, camera, rv), iterations=2)
    dark = (np.load(dark_path) if str(dark_path).endswith(".npy") else reader.load_dark(dark_path)) if dark_path else None
    nir_i = int(np.argmin(np.abs(cal.wl - NIR_NM)))
    vers = {"code_v": cfg.versions["code"], "roi_v": rv, "calib_v": cal.version}
    fr, sp, ix = [], [], []

    # Pass 1: read each frame once, measure its offset from the reference.
    loaded = []
    for r in rows:
        meta = {"frame_id": r["frame_id"], "camera": camera, "ts_utc": r["ts_utc"], "sun_elev": r["sun_elev"],
                "min_from_noon": r["min_from_noon"], "format": r["format"], "dark_id": dark_id, **vers}
        try:
            raw, bit = reader.read_bands(Path(r["read_path"]))
            if bit != FORMAT_BITS[r["format"]]:
                raise ValueError(f"bit depth {bit} does not match format {r['format']}")
        except Exception as e:  # recorded per frame and reported by the CLI; one bad file must not stop a day
            print(f"error {r['frame_id']}: {e}", file=sys.stderr)
            fr.append({**meta, "error": f"{type(e).__name__}: {e}"})
            continue
        dn = reader.to_dn10(raw, bit) - (dark if dark is not None else 0)
        nir = (cal.M[nir_i] @ dn.reshape(25, -1)).reshape(reader.H, reader.W)
        nir[(raw >= (cfg.sat_8bit if bit == 8 else cfg.sat_16bit)).any(0)] = np.nan
        dy, dx = (qa.shift(ref, nir, ref_mask=struct, max_px=int(cfg.qa["max_search_px"]))
                  if ref is not None else (np.nan, np.nan))
        loaded.append((meta, raw, bit, dy, dx))

    # The camera drifts by a few pixels over weeks; within a day the estimate
    # jitters by ~1 px with moving shadows, so masks follow the day's median.
    day_dy, day_dx = day_offset([(x[3], x[4]) for x in loaded], cfg, camera)
    m = roi.shift_masks(masks, day_dy, day_dx) if np.isfinite(day_dy) else masks
    canopy = m.get("canopy", np.logical_or.reduce(list(m.values())))

    # Pass 2: decode and extract with the shifted masks.
    for meta, raw, bit, dy, dx in loaded:
        base = {"frame_id": meta["frame_id"], "camera": camera}
        d = reader.decode_bands(raw, bit, cal, dark, cfg)
        nir = d.bands[nir_i]
        fr.append({**meta, "error": "", "bit_depth": d.bit_depth, "dn_mean": d.dn_mean,
                   "sat_frac": float(d.sat[canopy].mean()),
                   "light_ratio": qa.light_ratio(nir, canopy),
                   "sharpness": qa.sharpness(np.where(canopy, nir, np.nan)),
                   "shift_dy": dy, "shift_dx": dx, "shift_px": float(np.hypot(dy - day_dy, dx - day_dx)),
                   "day_shift_dy": day_dy, "day_shift_dx": day_dx})
        for name, s in roi.spectra(d.bands, m).items():
            for j, nm in enumerate(cal.wl):
                sp.append({**base, "roi": name, "band_nm": round(float(nm), 1), "median": s["median"][j],
                           "p10": s["p10"][j], "p90": s["p90"][j], "n_valid": s["n_valid"], **vers})
            if s["n_valid"] == 0:
                continue
            for iname, v in indices.compute_all(s["median"], cal.wl, cfg.band_tol).items():
                ix.append({**base, "roi": name, "index": iname, "value": float(v), **vers})
    return pd.DataFrame(fr), pd.DataFrame(sp), pd.DataFrame(ix)


# ---------- process ----------

def _versions(cfg, camera: str) -> dict:
    return {"code_v": cfg.versions["code"], "roi_v": cfg.versions["roi"],
            "calib_v": calib.load(cfg.calib_dir, camera).version}


def done_ids(work: Path, cfg) -> set[str]:
    f = store.read(work / "frames", ["frame_id", "camera", *store.VERSION_COLS])
    if f.empty:
        return set()
    keep = []
    for cam, g in f.groupby("camera"):
        keep.append(store.current(g, _versions(cfg, cam)))
    return set(pd.concat(keep).frame_id)


def process(cfg, work_dir=None, workers: int = 1, limit: int | None = None, use_sample: bool = False,
            min_sun_elev: float = 0.0) -> int:
    work = _work(cfg, work_dir)
    cat = pd.read_parquet(work / ("catalog_sample.parquet" if use_sample else "catalog.parquet"))
    if not (work / "darks" / "nights.csv").exists():
        build_master_darks(cfg, work)
    todo = cat[(cat.sun_elev > min_sun_elev) & ~cat.frame_id.isin(done_ids(work, cfg))]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        return 0
    todo = todo.assign(date=todo.ts_local.dt.normalize())
    jobs = []
    for (cam, date, fmt), g in todo.groupby(["camera", "date", "format"]):
        dp, did = master_dark(work, cam, fmt)
        jobs.append((cam, g.to_dict("records"), dp, did))
    t, n, errors = time.time(), 0, 0

    def _save(res):
        nonlocal n, errors
        f, s, i = res
        store.write(s, work / "roi_spectra")
        store.write(i, work / "indices")
        store.write(f, work / "frames")       # last, so a crash never marks a frame done without its data
        n += len(f)
        errors += int((f.get("error", pd.Series(dtype=str)) != "").sum())

    if workers <= 1:
        for j in jobs:
            _save(process_day(cfg, *j))
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(process_day, cfg, *j) for j in jobs]
            for k, fu in enumerate(as_completed(futs), 1):
                _save(fu.result())
                if k % 10 == 0:
                    print(f"  {k}/{len(jobs)} camera-days, {n} frames", file=sys.stderr)
    log_run(work, "process", frames=n, errors=errors, jobs=len(jobs), workers=workers,
            seconds=round(time.time() - t, 1), versions=cfg.versions)
    return n


# ---------- frames view, weather, daily, fit ----------

def fetch_weather(cfg, work_dir=None) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    cat = pd.read_parquet(work / "catalog.parquet", columns=["ts_utc"])
    start, end = cat.ts_utc.min().date().isoformat(), cat.ts_utc.max().date().isoformat()
    wx = weather.fetch(cfg.lat, cfg.lon, start, end, work / "cache" / f"openmeteo_{start}_{end}.json")
    wx.to_parquet(work / "weather.parquet", index=False)
    return wx


def in_window(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Keep rows on or after the valid-record start (config [window] start_utc)."""
    return df if cfg.start_utc is None else df[df.ts_utc >= cfg.start_utc]


def frames_view(cfg, work_dir=None) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    f = store.read(work / "frames")
    if f.empty:
        return f
    f = pd.concat([store.current(g, _versions(cfg, c)) for c, g in f.groupby("camera")])
    f = in_window(f[f.error == ""], cfg).reset_index(drop=True)
    f["light_jump"] = qa.light_jump(f)
    if (work / "weather.parquet").exists():
        f = weather.join(f, pd.read_parquet(work / "weather.parquet"), cfg.lat, cfg.lon)
    return qa.apply(f, cfg)


def indices_view(cfg, work_dir=None) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    i = store.read(work / "indices")
    return pd.concat([store.current(g, _versions(cfg, c)) for c, g in i.groupby("camera")]) if not i.empty else i


def build_daily(cfg, work_dir=None) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    d = timeseries.daily(indices_view(cfg, work), frames_view(cfg, work), cfg.noon_window_min)
    d = d.sort_values("date")
    d["outlier"] = d.groupby(["camera", "roi", "index"])["median"].transform(timeseries.hampel).astype(bool)
    d.to_parquet(work / "daily.parquet", index=False)
    log_run(work, "daily", rows=len(d), days=int(d.date.nunique()))
    return d


def flag_fit(fit: dict, last_doy: float) -> dict:
    """reliable = curve rises, the half-way date and its CI end at least
    3 days before the record does (otherwise the plateau was never seen), and
    the 95% CI on that date is no wider than 30 days."""
    edge = fit["mid50_hi"] >= last_doy - 3 or fit["mid50"] >= last_doy - 3
    rises = fit["hi"] > fit["lo"]
    narrow = (fit["mid50_hi"] - fit["mid50_lo"]) <= 30
    return {**fit, "mid_at_record_end": bool(edge), "rises": bool(rises), "ci_narrow": bool(narrow),
            "reliable": bool(rises and not edge and narrow)}


def _fit_one(job) -> dict:
    (cam, r, name), doy, y, w, n_boot = job
    try:
        return {"camera": cam, "roi": r, "index": name, "error": "",
                **flag_fit(timeseries.fit_season(doy, y, w, n_boot=n_boot), float(doy.max()))}
    except (RuntimeError, ValueError) as e:
        return {"camera": cam, "roi": r, "index": name, "error": str(e), "n_days": len(doy)}


def fit_all(cfg, work_dir=None, min_days: int = 15, n_boot: int = 300, workers: int = 6) -> pd.DataFrame:
    work = _work(cfg, work_dir)
    d = pd.read_parquet(work / "daily.parquet")
    d = d[~d.outlier]
    inv = {n for n, i in indices.REGISTRY.items() if i.scale_invariant}
    jobs = []
    for (cam, r, name), g in d[d["index"].isin(inv)].groupby(["camera", "roi", "index"]):
        if len(g) < min_days:
            continue
        doy = (g.date - pd.Timestamp("2026-01-01")).dt.days.to_numpy(float)
        jobs.append(((cam, r, name), doy, g["median"].to_numpy(), g.n_frames.to_numpy(float), n_boot))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_fit_one, jobs, chunksize=4))
    out = pd.DataFrame(rows)
    out.to_parquet(work / "phenology.parquet", index=False)
    log_run(work, "fit", fits=len(out), failed=int((out.get("error", pd.Series(dtype=str)) != "").sum()))
    return out


# ---------- ROI reference ----------

def build_roi_reference(cfg, camera: str, frames: list[Path], dark: Path | None) -> np.ndarray:
    cal = calib.load(cfg.calib_dir, camera)
    dk = reader.load_dark(dark) if dark else None
    nd, sat, nir = [], [], []
    for p in frames:
        d = reader.decode(p, cal, dk, cfg)
        nd.append(indices.compute_one("NDVI", d.bands, cal.wl, cfg.band_tol))
        sat.append(d.sat)
        nir.append(d.bands[int(np.argmin(np.abs(cal.wl - NIR_NM)))])
    return roi.build_reference(cfg.roi_dir, camera, cfg.versions["roi"], np.array(nd), np.array(sat), np.array(nir))
