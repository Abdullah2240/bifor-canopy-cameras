"""S3. Auto-exposure, dark offset, and 8-bit quantisation.

Signal model per band: DN = clip(round(g * R * E + offset)), with the scene
exposed so the mean DN sits near 500 of 1023 (as on the cameras). Compare
indices from DN (after the pipeline's dark subtraction, which may be wrong by
`dark_error` DN) against indices from true reflectance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CAL, FIG, OUT, TOL, bands, canopy, sample_params
from bifor import indices

rng = np.random.default_rng(3)
S = np.array([canopy(**p) for p in sample_params(800, rng)])
R = bands(S, "A5")                                   # (24, n) reflectance
wl = CAL["A5"].wl
key = ["NDVI", "NDRE", "CIre", "MTCI", "REP-LE", "SR705", "DVI", "NIRv"]
truth = indices.compute_all(R, wl, TOL, key)


def observe(R, gain_mult=1.0, offset=10.0, dark_error=0.0, bits=10, noise_dn=1.0):
    g = 500.0 / R.mean(0) * gain_mult
    dn10 = R * g + offset + rng.normal(0, noise_dn, R.shape)
    if bits == 8:
        dn10 = np.round(dn10 / 4) * 4
    dn10 = np.clip(np.round(dn10), 0, 1023)
    return dn10 - (offset + dark_error)


rows = []
for gm in (0.3, 1.0, 3.0):
    for de in (0, 2, 5, 10, 20, 60):
        for bits in (10, 8):
            v = indices.compute_all(observe(R, gm, 10, de, bits), wl, TOL, key)
            for n in key:
                e = np.asarray(v[n], float) - np.asarray(truth[n], float)
                rel = n in ("DVI", "NIRv")
                if rel:   # exposure-dependent: compare after scaling to the truth's mean
                    e = np.asarray(v[n], float) / np.nanmean(v[n]) * np.nanmean(truth[n]) - truth[n]
                rows.append({"gain_mult": gm, "dark_error_dn": de, "bits": bits, "index": n,
                             "bias": np.nanmean(e), "rmse": np.sqrt(np.nanmean(e ** 2))})
t = pd.DataFrame(rows)
t.to_csv(OUT / "s3_exposure_dark_quant.csv", index=False)
print(t[(t.gain_mult == 1.0)].pivot_table(index=["index", "bits"], columns="dark_error_dn", values="rmse").round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for n in ["NDVI", "NDRE", "CIre", "REP-LE"]:
    g = t[(t["index"] == n) & (t.gain_mult == 1) & (t.bits == 10)]
    axes[0].plot(g.dark_error_dn, g.bias / np.nanstd(truth[n]), "o-", label=n)
axes[0].set_xlabel("dark offset left in the signal (DN of 1023)"); axes[0].set_ylabel("bias / natural SD of index")
axes[0].set_title("Dark offset error"); axes[0].legend(fontsize=7); axes[0].axvline(70 - 8, color="0.5", ls=":")
axes[0].text(62, axes[0].get_ylim()[0], " A5 Dec\n per-night dark", fontsize=7, va="bottom")
for n in ["NDVI", "NDRE", "CIre", "SR705"]:
    g = t[(t["index"] == n) & (t.dark_error_dn == 0) & (t.bits == 10)]
    axes[1].plot(g.gain_mult, g.rmse, "o-", label=n)
axes[1].set_xscale("log"); axes[1].set_xlabel("exposure multiplier"); axes[1].set_ylabel("RMSE vs true reflectance")
axes[1].set_title("Ratios ignore exposure (with correct dark)"); axes[1].legend(fontsize=7)
g = t[(t.dark_error_dn == 0) & (t.gain_mult == 1)].pivot_table(index="index", columns="bits", values="rmse")
g.plot.bar(ax=axes[2], color=["#c0562f", "#1f6f8b"])
axes[2].set_title("8-bit vs 10-bit quantisation, single frame"); axes[2].set_ylabel("RMSE")
fig.suptitle("S3: exposure, dark offset and bit depth (PROSAIL canopies, A5 bands, mean signal 500 DN)")
fig.tight_layout()
fig.savefig(FIG / "s3_exposure_dark_quant.png", dpi=110)
plt.close(fig)
