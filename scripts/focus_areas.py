"""Per focus area (catalog_data.py areas 1-4): which indices carry the signal on
this sensor, combining
  - S1 PROSAIL driver shares (chlorophyll, LAI, geometry),
  - S2 A5-A6 band-placement bias,
  - observed seasonal signal-to-noise: (end-of-record median - leafless median) / leafless day-to-day SD,
  - fitted green-up date and reliability.
Writes D:\\BIFOR_work\\focus\\focus_areas.csv and focus_snr.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import config, indices  # noqa: E402

cfg = config.load()
W = cfg.work_dir
OUT = W / "focus"
OUT.mkdir(exist_ok=True)
AREA = {1: "Chlorophyll & nitrogen", 2: "Photosynthesis / GPP", 3: "Canopy structure / biomass",
        4: "Stress / disease / water / senescence"}
D0 = pd.Timestamp("2026-01-01")

s1 = pd.read_csv(W / "sim" / "s1_sensitivity_shares.csv", index_col=0)
s2 = pd.read_csv(W / "sim" / "s2_band_bias.csv").set_index("index")
daily = pd.read_parquet(W / "daily.parquet")
daily = daily[(daily.roi == "canopy") & ~daily.outlier]
ph = pd.read_parquet(W / "phenology.parquet")
ph = ph[ph.roi == "canopy"]

rows = []
for name, idx in indices.REGISTRY.items():
    entry = indices.catalog_entry(name)
    areas = entry.get("area", [])
    if not areas:
        continue
    for cam in ("A5", "A6"):
        d = daily[(daily.camera == cam) & (daily["index"] == name)]
        if d.empty:
            continue
        win = d[d.date < cfg.leafless_end]["median"]
        jun = d[d.date >= "2026-05-25"]["median"]
        snr = (jun.median() - win.median()) / win.std() if len(win) > 10 and len(jun) else np.nan
        p = ph[(ph.camera == cam) & (ph["index"] == name)]
        rows.append({
            "index": name, "areas": ",".join(str(a) for a in areas), "camera": cam,
            "scale_invariant": idx.scale_invariant,
            "s1_chlorophyll": s1["cab"].get(name, np.nan), "s1_lai": s1["lai"].get(name, np.nan),
            "s1_geometry": s1[["tts", "tto", "psi"]].sum(1).get(name, np.nan),
            "s2_bias_pct_range": s2["bias_pct_of_range"].get(name, np.nan),
            "season_snr": snr,
            "half_green": (D0 + pd.Timedelta(days=float(p.mid50.iloc[0]))).strftime("%d %b") if len(p) else "",
            "fit_reliable": bool(p.reliable.iloc[0]) if len(p) else False,
        })
t = pd.DataFrame(rows)
t.to_csv(OUT / "focus_areas.csv", index=False)

summary = []
for a, label in AREA.items():
    g = t[t.areas.str.contains(str(a)) & t.scale_invariant]
    g = g.groupby("index").agg(snr=("season_snr", "mean"), chl=("s1_chlorophyll", "first"), lai=("s1_lai", "first"),
                               geo=("s1_geometry", "first"), bias=("s2_bias_pct_range", "first"),
                               reliable=("fit_reliable", "any"),
                               half_green=("half_green", lambda s: " / ".join(x for x in s if x)))
    g = g.sort_values("snr", ascending=False)
    summary.append((label, g))
    print(f"\n== {a}. {label}")
    print(g.round(2).head(8).to_string())

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True)
for ax, (label, g) in zip(axes, summary):
    g = g.head(10).iloc[::-1]
    col = ["#2e7d32" if c >= 0.6 else "#1565c0" if l >= 0.3 else "#9e9e9e" for c, l in zip(g.chl.fillna(0), g.lai.fillna(0))]
    ax.barh(g.index, g.snr, color=col)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("season change / leafless day SD")
fig.suptitle("Focus areas: seasonal signal-to-noise of each index (green = mainly chlorophyll in S1, blue = mainly leaf area)")
fig.tight_layout()
fig.savefig(OUT / "focus_snr.png", dpi=110)
plt.close(fig)
