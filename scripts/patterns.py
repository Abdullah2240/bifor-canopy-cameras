"""Second line of analysis: patterns in the full-run tables beyond the green-up fit.

1. Diurnal cycle: index vs minutes from solar noon, by month.
2. A5 vs A6: day-to-day correlation and best lag of the green-up.
3. Which layer greens first: upper vs lower canopy vs understorey (A5), half-dates.
4. Red-edge position through spring.
5. Spectral stages: PCA + k-means on daily 24-band canopy spectra (normalised).
6. Weather: index scatter vs clearness, cloud and rain.
7. Anomaly days: residuals from the fit beyond 3 robust SD.
Writes D:\\BIFOR_work\\patterns\\*.png|csv and returns a text summary."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import config, pipeline, store, timeseries  # noqa: E402

cfg = config.load()
W = cfg.work_dir
OUT = W / "patterns"
OUT.mkdir(exist_ok=True)
D0 = pd.Timestamp("2026-01-01")
COL = {"A5": "#1f6f8b", "A6": "#c0562f"}
lines = []


def say(s):
    print(s)
    lines.append(s)


f = pipeline.frames_view(cfg)
ind = pipeline.indices_view(cfg)
daily = pd.read_parquet(W / "daily.parquet")
ph = pd.read_parquet(W / "phenology.parquet")
x = ind.merge(f[["frame_id", "ts_utc", "min_from_noon", "qa_pass", "clearness", "cloud_pct", "precip_mm", "sun_elev"]],
              on="frame_id")
x = x[x.qa_pass]
x["month"] = x.ts_utc.dt.strftime("%Y-%m")

# 1. diurnal
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
rows = []
for ax, cam in zip(axes, ("A5", "A6")):
    g = x[(x.camera == cam) & (x.roi == "canopy") & (x["index"] == "NDVI")]
    g = g.assign(bin=(g.min_from_noon // 30) * 30)
    for m, gm in g.groupby("month"):
        dev = gm.value - gm.groupby(gm.ts_utc.dt.date).value.transform("median")
        prof = dev.groupby(gm.bin).median()
        prof = prof[gm.groupby("bin").size() > 20]
        ax.plot(prof.index / 60, prof.values, label=m)
        if len(prof):
            rows.append({"camera": cam, "month": m, "am_minus_pm": prof[prof.index < -60].mean() - prof[prof.index > 60].mean(),
                         "noon_minus_edges": prof[prof.index.to_series().abs() <= 30].mean() - prof[prof.index.to_series().abs() > 120].mean()})
    ax.set_title(f"{cam}: NDVI minus that day's median, by time from solar noon")
    ax.set_xlabel("hours from solar noon")
    ax.axvline(0, color="0.6", lw=0.8)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "p1_diurnal.png", dpi=110); plt.close(fig)
di = pd.DataFrame(rows); di.to_csv(OUT / "p1_diurnal.csv", index=False)
say(f"1. Diurnal: median morning-minus-afternoon NDVI {di.am_minus_pm.median():+.3f}; "
    f"noon-minus-edges {di.noon_minus_edges.median():+.3f} (per month and camera, n={len(di)}).")

# 2. A5 vs A6
d = daily[(daily.roi == "canopy") & ~daily.outlier]
for n in ("NDVI", "NDRE", "REP-LE"):
    w = d[d["index"] == n].pivot_table(index="date", columns="camera", values="median").dropna()
    win = w[w.index < cfg.leafless_end]
    best = None
    for lag in range(-10, 11):
        a = w.A5.shift(lag)
        r = a.corr(w.A6)
        if best is None or r > best[1]:
            best = (lag, r)
    say(f"2. {n}: A5-A6 daily correlation {w.A5.corr(w.A6):.2f} all season, {win.A5.corr(win.A6):.2f} in leafless "
        f"period to {cfg.leafless_end} (shared weather/light). Best lag {best[0]:+d} d (r={best[1]:.2f}).")

# 3. layer order (A5 has upper, lower, understory; A6 upper, lower)
p = ph[(ph.error == "") & (ph["index"].isin(["NDVI", "NDRE", "CIre", "REP-LE"]))].copy()
for c in ("sos20", "mid50", "mid50_lo", "mid50_hi"):
    p[c + "_date"] = (D0 + pd.to_timedelta(p[c], "D")).dt.strftime("%d %b")
lay = p[p.roi.isin(["canopy", "canopy_upper", "canopy_lower", "understory", "evergreen"])][
    ["camera", "roi", "index", "sos20_date", "mid50_date", "mid50_lo_date", "mid50_hi_date", "mid50", "rmse"]]
lay.sort_values(["index", "camera", "mid50"]).to_csv(OUT / "p3_layer_order.csv", index=False)
for (cam, n), g in lay.groupby(["camera", "index"]):
    if n == "NDVI":
        say(f"3. {cam} NDVI half-green dates by layer: " +
            ", ".join(f"{r.roi} {r.mid50_date}" for r in g.sort_values("mid50").itertuples()))

# 4. red-edge position
rep = d[d["index"] == "REP-LE"].pivot_table(index="date", columns="camera", values="median")
fig, ax = plt.subplots(figsize=(12, 3.3))
for cam in rep:
    ax.plot(rep.index, rep[cam], ".", ms=3, color=COL[cam], label=cam)
    ax.plot(rep.index, rep[cam].rolling(7, center=True, min_periods=3).median(), color=COL[cam])
ax.set_ylabel("red-edge position (nm)"); ax.legend(fontsize=7)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_title("Red-edge position, canopy ROI (daily median, 7-day running median)")
fig.savefig(OUT / "p4_red_edge.png", dpi=110, bbox_inches="tight"); plt.close(fig)
early = rep[rep.index < cfg.leafless_end].median(); late = rep[rep.index >= "2026-05-25"].median()
say(f"4. Red-edge position leafless (to {cfg.leafless_end}) vs last 10 days: " + ", ".join(f"{c} {early[c]:.1f} -> {late[c]:.1f} nm" for c in rep))

# 5. spectral stages
sp = store.read(W / "roi_spectra")
sp = sp[(sp.code_v == cfg.versions["code"]) & (sp.roi == "canopy")]
sp = sp.merge(f[f.qa_pass & (f.min_from_noon.abs() <= cfg.noon_window_min)][["frame_id", "ts_utc"]], on="frame_id")
sp["date"] = sp.ts_utc.dt.normalize().dt.tz_localize(None)
sp["slot"] = sp.groupby("camera").band_nm.rank(method="dense").astype(int) - 1   # same filter on both cameras
S = sp.groupby(["camera", "date", "slot"])["median"].median().unstack()
S = S.div(S.mean(1), axis=0)                         # shape only: removes exposure
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
Z = S.dropna()
Z = Z - Z.groupby(level=0).transform("mean")         # remove each camera's own mean shape (view geometry)
pc = PCA(3).fit(Z.values)
sc = pc.transform(Z.values)
km = KMeans(4, n_init=20, random_state=0).fit(sc[:, :2])
lab = pd.Series(km.labels_, index=Z.index)
order = lab.groupby(level=1).first().groupby(lab.groupby(level=1).first()).apply(lambda s: s.index.min()).sort_values()
remap = {k: i for i, k in enumerate(order.index)}
lab = lab.map(remap)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for cam in ("A5", "A6"):
    if cam in lab.index.get_level_values(0):
        l = lab.loc[cam]
        axes[0].scatter(l.index, [0 if cam == "A5" else 1] * len(l), c=l.values, cmap="viridis", s=14)
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["A5", "A6"])
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[0].set_title("Spectral stage per day (k-means on PCA of normalised canopy spectra)")
cents = Z.groupby(lab.values).mean()
from bifor import indices as _ix  # noqa: E402
for k, row in cents.iterrows():
    axes[1].plot(_ix.slot_wl()[row.index.to_numpy()], row.values, label=f"stage {k}", color=plt.cm.viridis(k / 3))
axes[1].set_xlabel("nm"); axes[1].set_title("Mean normalised spectrum per stage"); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "p5_spectral_stages.png", dpi=110); plt.close(fig)
lab.rename("stage").reset_index().to_csv(OUT / "p5_stages.csv", index=False)
trans = {}
for cam in lab.index.get_level_values(0).unique():
    l = lab.loc[cam].sort_index()
    runs = l[l != l.shift()]                          # first day of each run of the same stage
    long = [(d, k) for d, k in runs.items() if (l.loc[d:d + pd.Timedelta(days=6)] == k).mean() > 0.7]
    trans[cam] = ", ".join(f"{d:%d %b}->s{k}" for d, k in long)
say(f"5. PCA: first 3 components explain {', '.join(f'{v:.0%}' for v in pc.explained_variance_ratio_)}. "
    f"First day of each spectral stage: {trans}")

# 6. weather
g = x[(x.roi == "canopy") & (x["index"] == "NDVI") & (x.min_from_noon.abs() <= cfg.noon_window_min)].copy()
g["dev"] = g.value - g.groupby(["camera", g.ts_utc.dt.date]).value.transform("median")
dd = daily[(daily.roi == "canopy") & (daily["index"] == "NDVI") & ~daily.outlier].copy()
wxd = f.assign(date=f.ts_utc.dt.normalize().dt.tz_localize(None)).groupby(["camera", "date"]).agg(
    clear=("clearness", "median"), cloud=("cloud_pct", "median"), rain=("precip_mm", "sum")).reset_index()
dd = dd.merge(wxd, on=["camera", "date"])
leafless = dd[dd.date < cfg.leafless_end]
r_clear = leafless[["median", "clear"]].corr().iloc[0, 1]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for cam in ("A5", "A6"):
    q = leafless[leafless.camera == cam]
    axes[0].scatter(q.clear, q["median"], s=10, color=COL[cam], label=cam)
axes[0].set_xlabel("daily clearness (measured / clear-sky sunlight)"); axes[0].set_ylabel("daily NDVI, leafless period")
axes[0].legend(fontsize=7); axes[0].set_title("Does light change leafless NDVI?")
gb = g.assign(cb=pd.cut(g.clearness, [0, .3, .5, .7, .9, 1.5])).groupby("cb", observed=True).dev.apply(lambda s: s.abs().median())
axes[1].bar(gb.index.astype(str), gb.values, color="#7a8b6f")
axes[1].set_xlabel("clearness"); axes[1].set_ylabel("median |frame - day median|")
axes[1].set_title("Within-day NDVI scatter by clearness")
fig.tight_layout(); fig.savefig(OUT / "p6_weather.png", dpi=110); plt.close(fig)
say(f"6. Weather: leafless-period daily NDVI vs clearness r={r_clear:+.2f}. Within-day scatter by clearness: "
    + ", ".join(f"{k} {v:.3f}" for k, v in gb.items()))

# 7. anomaly days
an = []
for (cam, r_, n), gg in daily[(daily.roi == "canopy") & daily["index"].isin(["NDVI", "NDRE"])].groupby(["camera", "roi", "index"]):
    pp = ph[(ph.camera == cam) & (ph.roi == r_) & (ph["index"] == n) & (ph.error == "")]
    if pp.empty:
        continue
    pp = pp.iloc[0]
    t = (gg.date - D0).dt.days
    res = gg["median"] - timeseries.logistic(t, pp.lo, pp.hi, pp.t0, pp.k)
    mad = 1.4826 * (res - res.median()).abs().median()
    out = gg[(res - res.median()).abs() > 3 * mad].assign(resid=res)
    an.append(out)
an = pd.concat(an) if an else pd.DataFrame()
if len(an):
    an = an.merge(wxd, on=["camera", "date"], how="left")
    an.to_csv(OUT / "p7_anomaly_days.csv", index=False)
    common = an.groupby(["date", "index"]).camera.nunique()
    both = common[common == 2].reset_index().date.dt.strftime("%d %b").unique()
    say(f"7. Anomaly days (|residual| > 3 robust SD): {an.date.nunique()} dates; on both cameras: {', '.join(both[:12])}"
        f"{' ...' if len(both) > 12 else ''}.")
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
