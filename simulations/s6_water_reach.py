"""S6. Lane C: what can the 930-952 nm bands say about canopy water?

The classic water band index uses 970 nm; this sensor stops at 952 nm, and
940 nm is also where atmospheric water vapour absorbs sunlight. With no
reflectance panel the camera sees R * E, so vapour changes in E look like
canopy water changes. Compare an index's response to leaf water content (Cw)
against its response to precipitable water vapour (PWV) and sun angle.

Vapour model: optical depth tau = 0.35 * PWV[cm] * airmass * exp(-0.5((nm-935)/18)^2).
A rough band shape, enough to compare magnitudes, not a radiative transfer run."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CAL, FIG, OUT, WL, bands, canopy

wl = CAL["A5"].wl
j869, j944, j952, j904 = (int(np.argmin(np.abs(wl - x))) for x in (869, 944, 952, 904))


def tau(pwv, airmass):
    return 0.35 * pwv * airmass * np.exp(-0.5 * ((WL - 935) / 18) ** 2)


def seen(cw, pwv, airmass):
    return bands(canopy(cw=cw) * np.exp(-tau(pwv, airmass)), "A5")


ratios = {"R869/R944": (j869, j944), "R869/R952": (j869, j952), "R904/R952": (j904, j952)}
cws = np.linspace(0.004, 0.03, 12)
pwvs = np.linspace(0.6, 3.2, 12)
ams = np.array([1.15, 1.5, 2.5, 4.0])
rows = []
for name, (a, b) in ratios.items():
    v_cw = np.array([seen(c, 1.8, 1.5) for c in cws])
    v_pw = np.array([seen(0.012, p, 1.5) for p in pwvs])
    v_am = np.array([seen(0.012, 1.8, m) for m in ams])
    r_cw, r_pw, r_am = (x[:, a] / x[:, b] for x in (v_cw, v_pw, v_am))
    rows.append({"ratio": name, "change_over_Cw_range": np.ptp(r_cw), "change_over_PWV_range": np.ptp(r_pw),
                 "change_over_airmass_range": np.ptp(r_am),
                 "vapour_to_leafwater": np.ptp(r_pw) / max(np.ptp(r_cw), 1e-9)})
    if name == "R869/R944":
        keep = (r_cw, r_pw, r_am)
t = pd.DataFrame(rows)
t.to_csv(OUT / "s6_water_reach.csv", index=False)
print(t.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for c in (0.005, 0.012, 0.025):
    axes[0].plot(WL, canopy(cw=c), label=f"Cw {c}")
axes[0].axvspan(wl.min(), wl.max(), color="#e9f1e4"); axes[0].axvline(970, color="0.5", ls=":")
axes[0].text(972, 0.05, "970 nm\nwater band", fontsize=7)
axes[0].set_xlim(650, 1100); axes[0].set_title("Canopy reflectance vs leaf water (shaded: sensor)")
axes[0].legend(fontsize=7)
axes[1].plot(WL, np.exp(-tau(1.8, 1.5)), color="#1565c0")
axes[1].axvspan(wl.min(), wl.max(), color="#e9f1e4"); axes[1].set_xlim(650, 1100)
axes[1].set_title("Assumed vapour transmission (PWV 1.8 cm, airmass 1.5)")
axes[2].bar(t.ratio, t.vapour_to_leafwater, color="#c0562f")
axes[2].axhline(1, color="0.5", ls="--"); axes[2].set_yscale("log")
axes[2].set_title("Vapour effect ÷ leaf-water effect (>1 = vapour wins)")
fig.suptitle("S6: reach of the 930-952 nm bands for canopy water (Lane C)")
fig.tight_layout()
fig.savefig(FIG / "s6_water_reach.png", dpi=110)
plt.close(fig)
