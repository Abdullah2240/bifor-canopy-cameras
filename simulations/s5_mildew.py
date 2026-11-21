"""S5. Oak powdery mildew (Erysiphe alphitoides): which indices respond, and
what infected fraction of the canopy is needed before the change beats the
day-to-day noise seen in the real data?

Infected foliage: chlorophyll down 40%, some brown pigment, and a white
mycelium coat modelled as a flat +0.06 additive reflectance. Canopy signal is
an area-weighted mix of healthy and infected canopies (mid-season, LAI 4)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CAL, CFG, FIG, OUT, TOL, bands, canopy, enabled_invariant
from bifor import indices

wl = CAL["A5"].wl
healthy = canopy(cab=45, lai=4.0)
infected = canopy(cab=27, cbrown=0.3, lai=4.0) + 0.06
frac = np.linspace(0, 0.6, 25)
mix = np.array([(1 - f) * healthy + f * infected for f in frac])
names = enabled_invariant()
V = indices.compute_all(bands(mix, "A5"), wl, TOL, names)

# PROSAIL index levels do not match the cameras' relative-reflectance levels
# (simulated leafy NDVI ~0.85, observed ~0.5). So both the simulated change and
# the observed noise are expressed as a fraction of their own seasonal range:
#   simulated range = leafless (LAI 0.3) to leafy (LAI 4) canopy,
#   observed range  = 5th-95th percentile of the daily series Dec-Jun,
#   observed noise  = day-to-day SD in the stable leafless period (Jan-Mar).
from bifor import indices as _ix
bare = _ix.compute_all(bands(canopy(cab=20, lai=0.3), "A5"), wl, TOL, names)
leafy = _ix.compute_all(bands(healthy, "A5"), wl, TOL, names)
sim_range = {n: abs(float(leafy[n]) - float(bare[n])) for n in names}
noise = {}
dp = CFG.work_dir / "daily.parquet"
if dp.exists():
    d = pd.read_parquet(dp)
    d = d[(d.camera == "A5") & (d.roi == "canopy") & ~d.outlier]
    obs_range = d.groupby("index")["median"].apply(lambda s: s.quantile(0.95) - s.quantile(0.05))
    sd = d[d.date < CFG.leafless_end].groupby("index")["median"].std()
    noise = (sd / obs_range).to_dict()      # noise as a fraction of the observed season range
rows = []
for n in names:
    v = np.asarray(V[n], float)
    dv = (v - v[0]) / max(sim_range[n], 1e-12)     # fraction of the simulated season range
    sd = noise.get(n, np.nan)
    det = frac[np.argmax(np.abs(dv) > 2 * sd)] if np.isfinite(sd) and (np.abs(dv) > 2 * sd).any() else np.nan
    rows.append({"index": n, "change_at_20pct_of_range": dv[np.argmin(np.abs(frac - 0.2))], "day_sd_of_range": sd,
                 "snr_at_20pct": abs(dv[np.argmin(np.abs(frac - 0.2))]) / sd if sd == sd else np.nan,
                 "min_detectable_fraction": det})
t = pd.DataFrame(rows).sort_values("snr_at_20pct", ascending=False)
t.to_csv(OUT / "s5_mildew.csv", index=False)
print(t.round(4).to_string(index=False))

top = t.dropna(subset=["snr_at_20pct"]).head(8)["index"].tolist() or names[:8]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for n in top:
    v = np.asarray(V[n], float)
    sd = noise.get(n, np.nan)
    axes[0].plot(frac * 100, (v - v[0]) / max(sim_range[n], 1e-12) / (sd if sd == sd else 1), label=n)
axes[0].axhline(2, color="0.5", ls="--", lw=0.8); axes[0].axhline(-2, color="0.5", ls="--", lw=0.8)
axes[0].set_xlabel("infected canopy area (%)"); axes[0].set_ylabel("change / observed day-to-day SD")
axes[0].set_title("Index change in units of real noise (±2 = detectable on one day)")
axes[0].legend(fontsize=7)
axes[1].plot(wl, bands(healthy, "A5"), label="healthy", color="#2e7d32")
axes[1].plot(wl, bands(0.7 * healthy + 0.3 * infected, "A5"), label="30% infected", color="#8d6e63")
axes[1].set_xlabel("nm"); axes[1].set_ylabel("canopy reflectance"); axes[1].legend(fontsize=8)
axes[1].set_title("Canopy spectrum on A5 bands")
fig.suptitle("S5: powdery mildew sensitivity (PROSAIL mixture model)")
fig.tight_layout()
fig.savefig(FIG / "s5_mildew.png", dpi=110)
plt.close(fig)
