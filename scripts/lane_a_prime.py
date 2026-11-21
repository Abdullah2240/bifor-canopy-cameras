"""Lane A': green-up timing difference between the two cameras (A5 minus A6, days).

Fitted half-way dates come from a joint moving-block bootstrap on the days both
cameras share, so the weather noise they have in common is kept. A model-free
half-rise date is reported alongside. Treatment labels come from [treatment] in
bifor.toml and are a working assumption (docs/assumptions_and_tenets.md, T1).
Output: <work>/lane_a_prime.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import config, indices, timeseries as ts  # noqa: E402

cfg = config.load()
W = cfg.work_dir
D0 = pd.Timestamp("2026-01-01")
ROIS = ["canopy", "canopy_upper", "canopy_lower"]
KEY = ["NDVI", "kNDVI", "NDRE", "CIre", "MTCI", "RVI / SR", "NDI(704,665)", "VOG1", "REDSI", "REP-LE"]

d = pd.read_parquet(W / "daily.parquet")
d = d[~d.outlier & d.roi.isin(ROIS)]
inv = {n for n, i in indices.REGISTRY.items() if i.scale_invariant} & set(KEY)
rows = []
for (r, name), g in d[d["index"].isin(inv)].groupby(["roi", "index"]):
    w = g.pivot_table(index="date", columns="camera", values="median").dropna()
    if not {"A5", "A6"} <= set(w.columns) or len(w) < 30:
        continue
    t = (w.index - D0).days.to_numpy(float)
    row = {"roi": r, "index": name, "n_shared_days": len(w), "rho": float(w.A5.corr(w.A6))}
    try:
        row |= ts.paired_shift(t, w.A5.to_numpy(), w.A6.to_numpy(), n_boot=200)
    except (RuntimeError, ValueError) as e:
        row["error"] = str(e)
    h5 = ts.half_rise_date(w.A5, cfg.leafless_end)
    h6 = ts.half_rise_date(w.A6, cfg.leafless_end)
    row |= {"half_A5": h5, "half_A6": h6,
            "direct_shift": (h5 - h6).days if pd.notna(h5) and pd.notna(h6) else None}
    rows.append(row)
    print(f"{r:13s} {name:14s} fit A5-A6 {row.get('shift', float('nan')):+5.1f} d "
          f"[{row.get('lo', float('nan')):+5.1f}, {row.get('hi', float('nan')):+5.1f}]  direct {row['direct_shift']}")
out = pd.DataFrame(rows)
out["label_A5"], out["label_A6"] = cfg.treatment.get("A5", "unknown"), cfg.treatment.get("A6", "unknown")
out.to_csv(W / "lane_a_prime.csv", index=False)
print(f"wrote {len(out)} rows -> {W / 'lane_a_prime.csv'}")
