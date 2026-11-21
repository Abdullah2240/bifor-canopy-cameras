"""S1. Which indices track chlorophyll, which track LAI, and which react to view
and sun geometry, using A5's real bands.

Method: 3000 PROSAIL canopies with all parameters drawn at random. For each
index, the standardised rank-regression coefficient (SRRC) on each parameter
gives its share of influence. Plus Cab x LAI response grids for key indices."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import CAL, FIG, OUT, TOL, all_enabled, bands, canopy, sample_params
from bifor import indices

rng = np.random.default_rng(1)
P = sample_params(3000, rng)
S = np.array([canopy(**p) for p in P])
B = bands(S, "A5")
names = all_enabled()
V = indices.compute_all(B, CAL["A5"].wl, TOL, names)
X = pd.DataFrame(P)[["cab", "lai", "cw", "cm", "cbrown", "n", "lidfa", "tts", "tto", "psi", "car"]]
Xr = np.column_stack([rankdata(X[c]) for c in X]); Xr = (Xr - Xr.mean(0)) / Xr.std(0)
rows = []
for n in names:
    y = np.asarray(V[n], float)
    ok = np.isfinite(y)
    yr = rankdata(y[ok]); yr = (yr - yr.mean()) / yr.std()
    coef, *_ = np.linalg.lstsq(Xr[ok], yr, rcond=None)
    pred = Xr[ok] @ coef
    r2 = 1 - ((yr - pred) ** 2).sum() / (yr ** 2).sum()
    rows.append({"index": n, **dict(zip(X.columns, coef)), "R2_rank": r2})
t = pd.DataFrame(rows).set_index("index")
share = t[X.columns].pow(2).div(t[X.columns].pow(2).sum(1), axis=0)
share["R2_rank"] = t.R2_rank
share["main_driver"] = share[X.columns].idxmax(1)
share.round(3).to_csv(OUT / "s1_sensitivity_shares.csv")
print(share.sort_values("cab", ascending=False).round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 0.32 * len(share) + 1.5))
order = share.sort_values("cab", ascending=True).index
left = np.zeros(len(order))
cols = {"cab": "#2e7d32", "lai": "#1565c0", "cw": "#00838f", "cbrown": "#8d6e63", "cm": "#9e9d24",
        "n": "#bdbdbd", "lidfa": "#7e57c2", "tts": "#ef6c00", "tto": "#c62828", "psi": "#f48fb1", "car": "#fdd835"}
for c in X.columns:
    v = share.loc[order, c].to_numpy()
    ax.barh(order, v, left=left, color=cols[c], label=c)
    left += v
ax.set_xlabel("share of rank variance explained (SRRC²)")
ax.set_title("S1: what drives each index on A5's bands (3000 PROSAIL canopies)")
ax.legend(ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.04))
fig.savefig(FIG / "s1_driver_shares.png", bbox_inches="tight", dpi=110)
plt.close(fig)

cab = np.linspace(10, 80, 15); lai = np.linspace(0.3, 6.5, 15)
key = ["NDVI", "NDRE", "CIre", "MTCI", "REP-LE", "kNDVI"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
grid = np.array([[canopy(cab=c, lai=l) for c in cab] for l in lai])
Bg = bands(grid, "A5")
for ax, n in zip(axes.ravel(), key):
    v = indices.compute_one(n, Bg, CAL["A5"].wl, TOL)
    im = ax.contourf(cab, lai, v, levels=14, cmap="viridis")
    ax.set_title(n); ax.set_xlabel("Cab (µg/cm²)"); ax.set_ylabel("LAI")
    fig.colorbar(im, ax=ax)
fig.suptitle("S1: index response to chlorophyll and leaf area (PROSAIL, A5 bands, sun 40°, view 60°)")
fig.tight_layout()
fig.savefig(FIG / "s1_cab_lai_grids.png", dpi=110)
plt.close(fig)
