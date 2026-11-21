"""S4. Lane A' power: can two rings' green-up dates be told apart?

Each simulated ring pair shares the season shape and the camera cadence. Daily
noise has a part common to both cameras (weather, sun) with correlation rho and
an independent part. Cloudy/rainy days are dropped at random. The record is
truncated at the real end of data (3 Jun = day 154), before the plateau,
and starts at the valid-record start (config [window]).

The test statistic is |t0_B - t0_A| from separate logistic fits; its 95%
critical value comes from no-shift simulations, so the test needs no
assumption about the fits' standard errors. Truth parameters and noise come
from the full-run fits when phenology.parquet exists."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CFG, FIG, OUT
from bifor import timeseries as ts

START_DOY = float((CFG.start_utc.tz_convert(None).normalize() - pd.Timestamp("2026-01-01")).days) if CFG.start_utc is not None else -31.0
DAYS = np.arange(START_DOY, 155, dtype=float)
N_SIM = int(__import__("os").environ.get("S4_NSIM", 300))
truth = dict(lo=0.20, hi=0.62, t0=135.0, k=0.09)
noise_obs = None
ph_p, d_p = CFG.work_dir / "phenology.parquet", CFG.work_dir / "daily.parquet"
if ph_p.exists() and d_p.exists():
    ph = pd.read_parquet(ph_p)
    p = ph[(ph["index"] == "NDVI") & (ph.roi == "canopy") & (ph.error == "")]
    if len(p):
        truth = dict(lo=float(p.lo.mean()), hi=float(p.hi.mean()), t0=float(p.t0.mean()), k=float(p.k.mean()))
        noise_obs = float(p.rmse.mean())
print("truth", {k: round(v, 3) for k, v in truth.items()}, "observed residual sd", noise_obs)


def pair(rng, shift, day_sd, rho, keep_p):
    keep = rng.random(DAYS.size) < keep_p
    d = DAYS[keep]
    common = rng.normal(0, day_sd * np.sqrt(rho), d.size)
    out = []
    for s in (0.0, shift):
        y = ts.logistic(d, truth["lo"], truth["hi"], truth["t0"] + s, truth["k"])
        y = y + common + rng.normal(0, day_sd * np.sqrt(1 - rho), d.size)
        out.append(ts._t0_with_se(d, y, np.ones_like(d))[0])
    return out[1] - out[0]


def power_curve(day_sd, rho, keep_p, shifts, n=N_SIM, seed=0):
    rng = np.random.default_rng(seed)
    null = np.abs([pair(rng, 0.0, day_sd, rho, keep_p) for _ in range(n)])
    crit = np.percentile(null, 95)
    pw = [np.mean(np.abs([pair(rng, s, day_sd, rho, keep_p) for _ in range(n)]) > crit) for s in shifts]
    return crit, pw


shifts = np.arange(0, 16, 1.0)
sds = sorted({0.02, 0.04, 0.06} | ({round(noise_obs, 3)} if noise_obs else set()))
RHOS = (0.0, float(__import__("os").environ.get("S4_RHO", 0.8)))


def _scenario(args):
    sd, rho = args
    return sd, rho, power_curve(sd, rho, 0.85, shifts, seed=int(sd * 1000) + int(rho * 10))


def main():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(__import__("os").environ.get("S4_WORKERS", 3))) as ex:
        results = {(sd, rho): r for sd, rho, r in ex.map(_scenario, [(s, r) for r in RHOS for s in sds])}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, rho in zip(axes, RHOS):
        for sd in sds:
            crit, pw = results[(sd, rho)]
            rows += [{"day_sd": sd, "rho": rho, "shift_days": s, "power": p, "crit_days": crit} for s, p in zip(shifts, pw)]
            lab = f"day SD {sd:.3f}" + (" (observed)" if noise_obs and abs(sd - noise_obs) < 1e-6 else "")
            ax.plot(shifts, pw, "o-", ms=3, label=lab)
        ax.axhline(0.8, color="0.5", ls="--", lw=0.8)
        ax.set_title(f"camera noise correlation ρ = {rho}")
        ax.set_xlabel("true green-up shift between rings (days)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("power (5% test)")
    fig.suptitle("S4: power to detect a green-up date shift between two rings (NDVI-like season, data ending 3 Jun)")
    fig.tight_layout()
    fig.savefig(FIG / "s4_power_curves.png", dpi=110)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s4_power.csv", index=False)
    mds = t[t.power >= 0.8].groupby(["day_sd", "rho"]).shift_days.min().rename("min_detectable_shift_days")
    print(mds.to_string())
    mds.to_csv(OUT / "s4_min_detectable_shift.csv")


if __name__ == "__main__":
    main()
