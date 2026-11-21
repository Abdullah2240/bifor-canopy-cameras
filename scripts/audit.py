"""Data audit: catalog statistics, a weekly noon-frame sample through the full
reader, dark-frame levels, and monthly contact sheets. Reads D:\\BIFOR only."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import calib, config, indices, pipeline, reader  # noqa: E402

cfg = config.load()
WORK = cfg.work_dir
OUT = WORK / "audit"
OUT.mkdir(parents=True, exist_ok=True)
cat = pd.read_parquet(WORK / "catalog.parquet")
cat["month"] = cat.ts_utc.dt.strftime("%Y-%m")
cat["date"] = cat.ts_local.dt.normalize()


def catalog_stats():
    rows = []
    for (cam, m), g in cat.groupby(["camera", "month"]):
        dt = g.ts_utc.sort_values().diff().dt.total_seconds().div(60)
        rows.append({"camera": cam, "month": m, "frames": len(g), "days": g.date.nunique(),
                     "daylight_frames": int((g.sun_elev > 15).sum()),
                     "formats": ", ".join(f"{k}:{v}" for k, v in g.format.value_counts().items()),
                     "median_interval_min": round(dt.median(), 1), "max_gap_h": round(dt.max() / 60, 1),
                     "GB": round(g.size_bytes.sum() / 1e9, 2)})
    stats = pd.DataFrame(rows)
    gaps = []
    for cam, g in cat.groupby("camera"):
        t = g.ts_utc.sort_values().reset_index(drop=True)
        d = t.diff()
        for i in np.where(d > pd.Timedelta(hours=2))[0]:
            gaps.append({"camera": cam, "from": t[i - 1], "to": t[i], "hours": round(d[i].total_seconds() / 3600, 1)})
    gaps = pd.DataFrame(gaps)
    fmt = (cat.sort_values("ts_utc").groupby(["camera", "format"]).ts_utc.agg(["min", "max", "size"]).reset_index())
    noon_days = (cat[cat.sun_elev > 15].groupby(["camera", "date"]).size().rename("daylight_frames").reset_index())
    return stats, gaps, fmt, noon_days


def weekly_sample() -> pd.DataFrame:
    day = cat[(cat.min_from_noon.abs() < 30)]
    first = day.date.min()
    day = day[((day.date - first).dt.days % 7) == 0]
    pick = day.loc[day.groupby(["camera", "date"]).min_from_noon.apply(lambda s: s.abs().idxmin())]
    return pick


def measure(pick: pd.DataFrame) -> pd.DataFrame:
    darks = pipeline.pick_darks(cat, cfg)
    out = []
    for (cam, date, fmt), g in pick.groupby(["camera", "date", "format"]):
        dp, did = pipeline._dark_for(darks, cam, date, fmt)
        f, _, i = pipeline.process_day(cfg, cam, g.to_dict("records"), dp, did)
        if not i.empty:
            piv = i[i["index"].isin(["NDVI", "NDRE", "CIre", "REP-LE"])].pivot_table(
                index="frame_id", columns=["roi", "index"], values="value")
            piv.columns = [f"{r}:{n}" for r, n in piv.columns]
            f = f.merge(piv.reset_index(), on="frame_id", how="left")
        if dp:
            dk = reader.load_dark(Path(dp))
            f["dark_dn_mean"] = float(dk.mean())
        out.append(f)
        print(cam, date.date(), fmt, file=sys.stderr)
    return pd.concat(out, ignore_index=True)


def contact_sheet(pick: pd.DataFrame, cam: str):
    darks = pipeline.pick_darks(cat, cfg)
    cal = calib.load(cfg.calib_dir, cam)
    g = pick[pick.camera == cam].copy()
    g["month"] = g.ts_utc.dt.strftime("%Y-%m")
    rows = g.groupby("month").apply(lambda x: x.iloc[len(x) // 2], include_groups=False)
    n = len(rows)
    fig, ax = plt.subplots(n, 2, figsize=(14, 3.3 * n))
    for k, (m, r) in enumerate(rows.iterrows()):
        dp, _ = pipeline._dark_for(darks, cam, r.date, r.format)
        d = reader.decode(Path(r.read_path), cal, reader.load_dark(Path(dp)) if dp else None, cfg)
        R = indices.Spec(d.bands, cal.wl)
        rgb = np.dstack([R(860), R(716), R(674)])
        for c in range(3):
            lo, hi = np.nanpercentile(rgb[..., c], [2, 98])
            rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-9), 0, 1)
        ax[k, 0].imshow(np.nan_to_num(rgb, nan=1.0))
        nd = indices.compute_one("NDVI", d.bands, cal.wl, cfg.band_tol)
        ax[k, 1].imshow(nd, cmap="RdYlGn", vmin=-0.1, vmax=0.7)
        ax[k, 0].set_title(f"{cam} {r.ts_utc:%Y-%m-%d %H:%M} UTC  {r.format} {d.bit_depth}-bit  sat {d.sat.mean():.3f}", fontsize=9)
        ax[k, 1].set_title(f"NDVI  median {np.nanmedian(nd):.2f}", fontsize=9)
        for a in ax[k]:
            a.axis("off")
    fig.tight_layout()
    p = OUT / f"contact_{cam}.png"
    fig.savefig(p, dpi=55)
    plt.close(fig)
    return p


if __name__ == "__main__":
    stats, gaps, fmt, noon_days = catalog_stats()
    stats.to_csv(OUT / "by_month.csv", index=False)
    gaps.to_csv(OUT / "gaps.csv", index=False)
    fmt.to_csv(OUT / "formats.csv", index=False)
    noon_days.to_csv(OUT / "daylight_per_day.csv", index=False)
    print(stats.to_string(index=False))
    print(fmt.to_string(index=False))
    print(f"gaps > 2h: {len(gaps)}")
    print(gaps.sort_values("hours", ascending=False).head(15).to_string(index=False))
    pick = weekly_sample()
    m = measure(pick)
    m.to_csv(OUT / "weekly_noon_sample.csv", index=False)
    for cam in ("A5", "A6"):
        print(contact_sheet(pick, cam))
