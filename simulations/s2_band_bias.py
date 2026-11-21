"""S2. How much do A5 and A6 disagree on the same canopy purely because their
bands sit 0.4-1.5 nm apart? This is the floor any real A5-A6 difference must
clear before it can be read as a treatment effect."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CAL, FIG, OUT, TOL, all_enabled, bands, canopy, sample_params
from bifor import indices

rng = np.random.default_rng(2)
S = np.array([canopy(**p) for p in sample_params(1500, rng)])
names = all_enabled()
v5 = indices.compute_all(bands(S, "A5"), CAL["A5"].wl, TOL, names)
v6 = indices.compute_all(bands(S, "A6"), CAL["A6"].wl, TOL, names)
rows = []
for n in names:
    a, b = np.asarray(v5[n], float), np.asarray(v6[n], float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    spread = np.nanpercentile(a[ok], 95) - np.nanpercentile(a[ok], 5)
    r6 = indices.resolve(indices.REGISTRY[n], CAL["A6"].wl, TOL)
    r5 = indices.resolve(indices.REGISTRY[n], CAL["A5"].wl, TOL)
    rows.append({"index": n, "mean_bias": d.mean(), "sd": d.std(), "bias_pct_of_range": 100 * abs(d.mean()) / spread,
                 "A5_bands": r5.bands, "A6_bands": r6.bands})
t = pd.DataFrame(rows).sort_values("bias_pct_of_range", ascending=False)
t.to_csv(OUT / "s2_band_bias.csv", index=False)
print(t.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1))
ax.barh(t["index"], t.bias_pct_of_range, color="#c0562f")
ax.invert_yaxis()
ax.axvline(2, color="0.5", ls="--", lw=0.8)
ax.set_xlabel("|mean A5 − A6| as % of the index's natural range (5–95%)")
ax.set_title("S2: camera disagreement from band placement alone")
fig.savefig(FIG / "s2_band_bias.png", bbox_inches="tight", dpi=110)
plt.close(fig)
