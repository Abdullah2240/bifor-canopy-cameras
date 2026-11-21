"""Data figures: band ruler, ROI overlays, data-health calendar, season curves,
A5-A6 difference, QA breakdown. All write PNGs to <work>/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import calib, indices, pipeline, roi, timeseries

CAM_COLOR = {"A5": "#1f6f8b", "A6": "#c0562f"}
KEY_INDICES = ["NDVI", "NDRE", "CIre", "MTCI", "REP-LE", "kNDVI"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def band_ruler(cfg, out: Path) -> Path:
    cals = {c: calib.load(cfg.calib_dir, c) for c in ("A5", "A6")}
    names = list(indices.REGISTRY)
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(names) + 1.6))
    for k, (cam, cal) in enumerate(cals.items()):
        ax.vlines(cal.wl, -1.9 + 0.5 * k, -1.5 + 0.5 * k, color=CAM_COLOR[cam], lw=1.6, label=f"{cam} bands")
    for i, n in enumerate(names):
        y = i
        for k, (cam, cal) in enumerate(cals.items()):
            r = indices.resolve(indices.REGISTRY[n], cal.wl, cfg.band_tol)
            if r.enabled:
                ax.plot(r.bands, [y + 0.15 * (k - 0.5)] * len(r.bands), "o", ms=3.5, color=CAM_COLOR[cam])
        r5 = indices.resolve(indices.REGISTRY[n], cals["A5"].wl, cfg.band_tol)
        label = n + ("" if r5.enabled else "  (off)")
        ax.text(655, y, label, ha="right", va="center", fontsize=8, color="0.2" if r5.enabled else "0.55")
        if not r5.enabled:
            ax.plot(indices.REGISTRY[n].wavelengths, [y] * len(indices.REGISTRY[n].wavelengths), "x", color="0.6", ms=4)
    ax.axvspan(673.8 - 5, 951.9 + 5, color="#e9f1e4", zorder=-1)
    ax.set_xlim(490, 980)
    ax.set_ylim(-2.2, len(names))
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("wavelength (nm)  —  shaded: sensor coverage after m0 correction")
    ax.set_title("Bands each index uses, per camera (x = needed wavelength outside the sensor)")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out, "band_ruler")


def roi_overlay(cfg, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, cam in zip(axes, ("A5", "A6")):
        ref = roi.reference(cfg.roi_dir, cam, cfg.versions["roi"])
        ex = roi.exclusion(cfg.roi_dir, cam, cfg.versions["roi"])
        lo, hi = np.nanpercentile(ref, [2, 98])
        ax.imshow(np.clip((ref - lo) / (hi - lo), 0, 1), cmap="gray")
        ax.imshow(np.ma.masked_where(~ex, ex), cmap="autumn", alpha=0.45)
        for k, (name, coords) in enumerate(roi.polygons(cfg.roi_dir, cam, cfg.versions["roi"]).items()):
            xy = np.asarray(coords)
            ax.plot(xy[:, 0], xy[:, 1], lw=1.4, color=plt.cm.tab10(k), label=name)
        ax.set_title(f"{cam}: ROIs over reference NIR (orange = excluded pylons/sky)")
        ax.axis("off")
        ax.legend(fontsize=7, loc="lower left")
    return _save(fig, out, "roi_overlay")


def data_health(cfg, work: Path, out: Path) -> Path:
    cat = pipeline.in_window(pd.read_parquet(work / "catalog.parquet"), cfg)
    f = pipeline.frames_view(cfg, work)
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    for ax, cam in zip(axes, ("A5", "A6")):
        c = cat[cat.camera == cam]
        day = c.ts_utc.dt.floor("D")
        allf = c.groupby(day).size()
        ax.bar(allf.index, allf.values, width=1, color="0.85", label="all frames")
        if not f.empty:
            g = f[(f.camera == cam) & (f.min_from_noon.abs() <= cfg.noon_window_min)]
            n = g.groupby(g.ts_utc.dt.floor("D")).qa_pass.agg(["size", "sum"])
            ax.bar(n.index, n["size"], width=1, color="#f0b37e", label="processed near noon")
            ax.bar(n.index, n["sum"], width=1, color=CAM_COLOR[cam], label="passed QA")
        for fmt, col in (("png", "#ddd"), ("tiff", "#eee")):
            s = c[c.format == fmt].ts_utc
            if len(s):
                ax.axvspan(s.min(), s.max(), color=col, alpha=0.35, zorder=-2)
                ax.text(s.min(), ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, f"{fmt} 8-bit", fontsize=7)
        ax.set_ylabel(f"{cam}\nframes/day")
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("Data health: frames per day, near-noon frames processed, and frames passing QA")
    return _save(fig, out, "data_health_calendar")


def qa_breakdown(cfg, work: Path, out: Path) -> Path | None:
    f = pipeline.frames_view(cfg, work)
    if f.empty:
        return None
    f = f[f.min_from_noon.abs() <= cfg.noon_window_min]
    t = f.assign(month=f.ts_utc.dt.strftime("%Y-%m"), reason=f.qa_reason.replace("", "pass"))
    tab = t.groupby(["camera", "month", "reason"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, cam in zip(axes, ("A5", "A6")):
        if cam in tab.index.get_level_values(0):
            d = tab.loc[cam]
            (d.T / d.sum(1)).T.plot.bar(stacked=True, ax=ax, width=0.85, colormap="tab20", legend=cam == "A6")
        ax.set_title(f"{cam}: near-noon frames by QA outcome")
        ax.set_xlabel("")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1))
    return _save(fig, out, "qa_breakdown")


def season_curves(cfg, work: Path, out: Path, roi_name: str = "canopy") -> Path | None:
    if not (work / "daily.parquet").exists():
        return None
    d = pd.read_parquet(work / "daily.parquet")
    ph = pd.read_parquet(work / "phenology.parquet") if (work / "phenology.parquet").exists() else pd.DataFrame()
    names = [n for n in KEY_INDICES if n in set(d["index"])]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.4 * len(names)), sharex=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        for cam in ("A5", "A6"):
            g = d[(d.camera == cam) & (d.roi == roi_name) & (d["index"] == n)].sort_values("date")
            if g.empty:
                continue
            ok = g[~g.outlier]
            ax.fill_between(ok.date, ok.q25, ok.q75, color=CAM_COLOR[cam], alpha=0.15, lw=0)
            ax.plot(ok.date, ok["median"], ".", ms=3, color=CAM_COLOR[cam], label=f"{cam} daily median")
            ax.plot(g[g.outlier].date, g[g.outlier]["median"], "x", ms=4, color="0.5")
            if not ph.empty:
                p = ph[(ph.camera == cam) & (ph.roi == roi_name) & (ph["index"] == n) & (ph.error == "")]
                if len(p):
                    p = p.iloc[0]
                    t = np.arange(ok.date.min().dayofyear - 366 if ok.date.min().year == 2025 else ok.date.min().dayofyear,
                                  ok.date.max().dayofyear + 1)
                    x = pd.Timestamp("2026-01-01") + pd.to_timedelta(t, "D")
                    ok_fit = bool(p.get("reliable", True))
                    ax.plot(x, timeseries.logistic(t, p.lo, p.hi, p.t0, p.k), color=CAM_COLOR[cam],
                            lw=1.4 if ok_fit else 0.9, ls="-" if ok_fit else ":")
                    mid = pd.Timestamp("2026-01-01") + pd.Timedelta(days=p.mid50)
                    if ok_fit:
                        lo = pd.Timestamp("2026-01-01") + pd.Timedelta(days=p.mid50_lo)
                        hi = pd.Timestamp("2026-01-01") + pd.Timedelta(days=p.mid50_hi)
                        ax.axvspan(lo, hi, color=CAM_COLOR[cam], alpha=0.12)
                        ax.axvline(mid, color=CAM_COLOR[cam], ls="--", lw=0.9)
                    y = ax.get_ylim()[1] - (0 if cam == "A5" else 0.12 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
                    label = f" {cam} mid {mid:%d %b}" if ok_fit else f" {cam}: fit not reliable"
                    ax.text(ok.date.min(), y, label, color=CAM_COLOR[cam], fontsize=7, va="top")
        ax.set_ylabel(n)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle(f"Lane D: daily midday index ({roi_name} ROI), IQR band, spring logistic fit (dotted = not reliable), "
                 "half-green date with 95% CI (relative reflectance)")
    return _save(fig, out, f"season_{roi_name}")


def camera_difference(cfg, work: Path, out: Path, index: str = "NDVI", roi_name: str = "canopy") -> Path | None:
    if not (work / "daily.parquet").exists():
        return None
    d = pd.read_parquet(work / "daily.parquet")
    d = d[(d["index"] == index) & (d.roi == roi_name) & ~d.outlier]
    w = d.pivot_table(index="date", columns="camera", values="median").dropna()
    if w.empty or not {"A5", "A6"} <= set(w.columns):
        return None
    diff = (w.A5 - w.A6)
    roll = diff.rolling("14D", min_periods=4)
    m, se = roll.mean(), roll.std() / np.sqrt(roll.count())
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(diff.index, diff, ".", ms=3, color="0.4", label="daily A5 - A6")
    ax.plot(m.index, m, color="k", lw=1.3, label="14-day mean")
    ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color="k", alpha=0.12, label="95% CI")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel(f"{index} difference")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7)
    ax.set_title(f"A5 minus A6, {index}, {roi_name} ROI (differences include view geometry, not only treatment)")
    return _save(fig, out, f"camera_difference_{index}")


def render_all(cfg, work_dir=None) -> list[Path]:
    work = Path(work_dir or cfg.work_dir)
    out = work / "figures"
    ps = [band_ruler(cfg, out), roi_overlay(cfg, out)]
    if (work / "frames").exists():
        ps += [data_health(cfg, work, out), qa_breakdown(cfg, work, out)]
    for r in ("canopy", "canopy_upper", "canopy_lower"):
        ps.append(season_curves(cfg, work, out, r))
    for n in ("NDVI", "NDRE", "REP-LE"):
        ps.append(camera_difference(cfg, work, out, n))
    return [p for p in ps if p]
