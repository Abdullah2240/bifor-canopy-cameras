"""Build the self-contained data bundle for the hosted dashboard (app/data).

Needs the work directory (D:). Everything the dashboard and the HTML report
quote comes from summary.json, written here from the current outputs.
Run after: daily, fit, figures, simulations, patterns, scripts/lane_a_prime.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bifor import calib, config, indices, pipeline, reader, timeseries as ts  # noqa: E402

cfg = config.load()
W = cfg.work_dir
OUT = ROOT / "app" / "data"
D0 = pd.Timestamp("2026-01-01")
KEY_IDX = ["NDVI", "kNDVI", "NDRE", "CIre", "MTCI", "RVI / SR", "NDI(704,665)", "VOG1", "REDSI", "REP-LE"]
ROIS = ["canopy", "canopy_upper", "canopy_lower", "understory"]


def doy_date(x):
    return None if x is None or not np.isfinite(x) else (D0 + pd.Timedelta(days=float(x))).strftime("%Y-%m-%d")


def iso(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
(OUT / "frames").mkdir()

# ---------- tables ----------
daily = pd.read_parquet(W / "daily.parquet")
ph = pd.read_parquet(W / "phenology.parquet")
lane = pd.read_csv(W / "lane_a_prime.csv", parse_dates=["half_A5", "half_A6"])
daily[daily["index"].isin(KEY_IDX) & daily.roi.isin(ROIS)].to_parquet(OUT / "daily.parquet", index=False)
ph.to_parquet(OUT / "phenology.parquet", index=False)
lane.to_csv(OUT / "lane_a_prime.csv", index=False)

frames = pipeline.frames_view(cfg)
near = frames[frames.min_from_noon.abs() <= cfg.noon_window_min]
qa = (near.assign(date=near.ts_utc.dt.strftime("%Y-%m-%d"), reason=near.qa_reason.replace("", "passed"))
      .groupby(["camera", "date", "reason"]).size().rename("n").reset_index())
qa.to_csv(OUT / "qa_daily.csv", index=False)

cals = {c: calib.load(cfg.calib_dir, c) for c in ("A5", "A6")}
cat = indices.resolution_table(cals, cfg.band_tol)
if (W / "citations.csv").exists():
    cat = cat.merge(pd.read_csv(W / "citations.csv")[["index", "status", "doi"]], on="index", how="left")
cat.to_csv(OUT / "index_catalog.csv", index=False)

# ---------- summary numbers ----------
dk = daily[~daily.outlier]
S: dict = {"window": {"start_utc": str(cfg.start_utc), "leafless_end": cfg.leafless_end,
                      "first_day": iso(dk.date.min()), "last_day": iso(dk.date.max()),
                      "days": {c: int(g.date.nunique()) for c, g in dk.groupby("camera")},
                      "frames_near_noon": int(len(near)), "frames_passed": int(near.qa_pass.sum()),
                      "frames_per_day_median": float(near[near.qa_pass].groupby(
                          [near.camera, near.ts_utc.dt.date]).size().median())},
           "treatment": cfg.treatment}

gu = []
for name in ("NDVI", "kNDVI", "NDRE", "CIre"):
    for cam in ("A5", "A6"):
        p = ph[(ph["index"] == name) & (ph.roi == "canopy") & (ph.camera == cam) & (ph.error == "")]
        s = dk[(dk["index"] == name) & (dk.roi == "canopy") & (dk.camera == cam)].set_index("date")["median"]
        row = {"index": name, "camera": cam, "half_direct": iso(ts.half_rise_date(s, cfg.leafless_end)),
               "p80_direct": iso(ts.half_rise_date(s, cfg.leafless_end, level=0.8))}
        if len(p):
            p = p.iloc[0]
            row |= {"sos20": doy_date(p.sos20), "mid50": doy_date(p.mid50), "mid50_lo": doy_date(p.mid50_lo),
                    "mid50_hi": doy_date(p.mid50_hi), "reliable": bool(p.reliable)}
        gu.append(row)
S["greenup"] = gu

lag = {}
for cam in ("A5", "A6"):
    h = {r["index"]: r["half_direct"] for r in gu if r["camera"] == cam}
    if h.get("NDVI") and h.get("NDRE"):
        lag[cam] = {"ndvi": h["NDVI"], "ndre": h["NDRE"], "cire": h.get("CIre"),
                    "lag_days": (pd.Timestamp(h["NDRE"]) - pd.Timestamp(h["NDVI"])).days}
S["lag"] = lag

layers = []
for cam in ("A5", "A6"):
    for r in ROIS:
        s = dk[(dk["index"] == "NDVI") & (dk.roi == r) & (dk.camera == cam)].set_index("date")["median"]
        if len(s) > 30:
            layers.append({"camera": cam, "roi": r, "half_direct": iso(ts.half_rise_date(s, cfg.leafless_end))})
S["layers"] = layers

w = dk[(dk["index"] == "NDVI") & (dk.roi == "canopy")].pivot_table(index="date", columns="camera", values="median").dropna()
leaf = w[w.index < cfg.leafless_end]
S["noise"] = {"rho_all": float(w.A5.corr(w.A6)), "rho_leafless": float(leaf.A5.corr(leaf.A6)),
              "day_sd_leafless": float(pd.concat([leaf.A5.diff(), leaf.A6.diff()]).std() / np.sqrt(2)),
              "n_leafless_days": int(len(leaf))}

l = lane[(lane["index"] == "NDVI") & (lane.roi == "canopy")]
S["lane_a_prime"] = {k: (None if pd.isna(v) else (iso(v) if isinstance(v, pd.Timestamp) else
                                                  (float(v) if isinstance(v, (int, float, np.number)) else v)))
                     for k, v in l.iloc[0].items()} if len(l) else {}
if len(l):
    n5 = pd.DataFrame(gu).set_index(["index", "camera"])
    a, b = n5.loc[("NDVI", "A5"), "p80_direct"], n5.loc[("NDVI", "A6"), "p80_direct"]
    S["lane_a_prime"]["direct80_shift"] = (pd.Timestamp(a) - pd.Timestamp(b)).days if a and b else None
    S["lane_a_prime"]["fit_A5_reliable"] = bool(ph[(ph["index"] == "NDVI") & (ph.roi == "canopy") & (ph.camera == "A5")].reliable.iloc[0])
    S["lane_a_prime"]["fit_A6_reliable"] = bool(ph[(ph["index"] == "NDVI") & (ph.roi == "canopy") & (ph.camera == "A6")].reliable.iloc[0])
S["lane_a_prime_all"] = {"n": int(len(lane)), "fit_shift_median": float(lane["shift"].median()),
                         "direct_shift_median": float(lane.direct_shift.median()),
                         "share_A6_earlier_fit": float((lane["shift"] > 0).mean())}

mdd = pd.read_csv(W / "sim" / "s4_min_detectable_shift.csv")
S["power"] = mdd.to_dict(orient="records")

stages = pd.read_csv(W / "patterns" / "p5_stages.csv", parse_dates=["date"])
leaf_stage = {}
for cam, g in stages.groupby("camera"):
    g = g.sort_values("date")
    bare = set(g[g.date < cfg.leafless_end].stage)          # stages seen while leafless
    last_bare = g[g.stage.isin(bare)].date.max()
    after = g[g.date > last_bare].date
    leaf_stage[cam] = iso(after.min()) if len(after) else None
S["spectral_leafed_from"] = leaf_stage

summ = (W / "patterns" / "summary.txt").read_text(encoding="utf-8") if (W / "patterns" / "summary.txt").exists() else ""
S["patterns_text"] = summ.splitlines()
import re  # noqa: E402
m = re.search(r"clearness r=([-+]?[0-9]*.?[0-9]+)", summ)
S["clear_r"] = float(m.group(1)) if m else None
(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")

# ---------- figures ----------
FIGS = {"sim": ["s1_driver_shares", "s1_cab_lai_grids", "s2_band_bias", "s3_exposure_dark_quant", "s4_power_curves",
                "s5_mildew", "s6_water_reach", "s7_qa_detection"],
        "": ["roi_overlay", "band_ruler", "data_health_calendar", "qa_breakdown"]}
for sub, names in FIGS.items():
    for n in names:
        src = W / "figures" / sub / f"{n}.png"
        if src.exists():
            shutil.copy(src, OUT / "figures" / f"{n}.png")
for n in ("p4_red_edge", "p5_spectral_stages", "p6_weather"):
    src = W / "patterns" / f"{n}.png"
    if src.exists():
        shutil.copy(src, OUT / "figures" / f"{n}.png")

# ---------- example frames (what the cameras see) ----------
from PIL import Image  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

DAYS = ["2026-03-15", "2026-04-15", "2026-04-29", "2026-05-06", "2026-05-15", "2026-06-02"]
ok = frames[frames.qa_pass]
shown = []
for cam in ("A5", "A6"):
    cal = cals[cam]
    for day in DAYS:
        g = ok[(ok.camera == cam) & (ok.ts_utc.dt.strftime("%Y-%m-%d") == day)]
        if g.empty:
            continue
        r = g.loc[g.min_from_noon.abs().idxmin()]
        catrow = pd.read_parquet(W / "catalog.parquet", filters=[("frame_id", "==", r.frame_id)]).iloc[0]
        dp, _ = pipeline.master_dark(W, cam, catrow.format)
        d = reader.decode(Path(catrow.read_path), cal, np.load(dp) if dp else None, cfg)
        R = indices.Spec(d.bands, cal.wl)
        rgb = np.dstack([R(860), R(716), R(674)])
        for k in range(3):
            lo, hi = np.nanpercentile(rgb[..., k], [2, 98])
            rgb[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo + 1e-9), 0, 1)
        Image.fromarray((np.nan_to_num(rgb, nan=1.0) * 255).astype(np.uint8)).save(OUT / "frames" / f"{cam}_{day}_fc.png")
        v = indices.compute_one("NDVI", d.bands, cal.wl, cfg.band_tol)
        vm = cm.RdYlGn(np.clip((np.nan_to_num(v, nan=0) - 0.0) / 0.8, 0, 1))[..., :3]
        Image.fromarray((vm * 255).astype(np.uint8)).save(OUT / "frames" / f"{cam}_{day}_ndvi.png")
        shown.append({"camera": cam, "date": day, "time_utc": r.ts_utc.strftime("%H:%M")})
pd.DataFrame(shown).to_csv(OUT / "frames" / "frames.csv", index=False)

size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"bundle {OUT}: {size / 1e6:.1f} MB")
print(json.dumps({k: S[k] for k in ("window", "lag", "noise", "lane_a_prime", "spectral_leafed_from")}, indent=1, default=str))
