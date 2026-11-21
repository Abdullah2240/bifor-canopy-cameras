"""Daily series, outlier filter, green-up fits and shift-detection power."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

LEVELS = {"sos20": 0.2, "mid50": 0.5, "mat80": 0.8}


def daily(ind: pd.DataFrame, frames: pd.DataFrame, window_min: int) -> pd.DataFrame:
    f = frames[frames.qa_pass & (frames.min_from_noon.abs() <= window_min)]
    d = ind.merge(f[["frame_id", "ts_utc"]], on="frame_id")
    d = d[np.isfinite(d.value)]
    d["date"] = d.ts_utc.dt.tz_convert("UTC").dt.date
    g = d.groupby(["date", "camera", "roi", "index"]).value
    out = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                        "n_frames": g.size()}).reset_index()
    out["date"] = pd.to_datetime(out.date)
    return out


def hampel(s: pd.Series, window: int = 7, k: float = 3.0) -> pd.Series:
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median() * 1.4826
    return ((s - med).abs() > k * mad.clip(lower=1e-9)).fillna(False)


def logistic(t, lo, hi, t0, k):
    return lo + (hi - lo) / (1 + np.exp(-k * (t - t0)))


def double_logistic(t, lo, hi, t0, k, t1, k1):
    return lo + (hi - lo) * (1 / (1 + np.exp(-k * (t - t0))) - 1 / (1 + np.exp(-k1 * (t - t1))))


def _level_dates(p) -> dict[str, float]:
    _, _, t0, k = p
    return {name: t0 + np.log(q / (1 - q)) / k for name, q in LEVELS.items()}


def _fit(t, y, w):
    p0 = [np.percentile(y, 5), np.percentile(y, 95), t[np.argmin(np.abs(y - np.median(y)))], 0.1]
    bounds = ([-np.inf, -np.inf, t.min(), 0.005], [np.inf, np.inf, t.max(), 2.0])
    sigma = 1 / np.sqrt(np.clip(w, 1e-6, None))
    p, _ = curve_fit(logistic, t, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    return p


def fit_greenup(t, y, w, n_boot: int = 300, block: int = 7, seed: int = 0) -> dict:
    """Spring logistic fit (autumn limb absent in this dataset) with a moving-block
    residual bootstrap for the 20/50/80% green-up dates."""
    t, y, w = map(np.asarray, (t, y, w))
    p = _fit(t, y, w)
    fitted = logistic(t, *p)
    res = y - fitted
    out = {"lo": p[0], "hi": p[1], "t0": p[2], "k": p[3], "rmse": float(np.sqrt(np.mean(res ** 2))),
           "n_days": int(t.size), **_level_dates(p)}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in LEVELS}
    n = t.size
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n // block + 1)
        r = np.concatenate([res[s:s + block] for s in starts])[:n]
        try:
            pb = _fit(t, fitted + r, w)
        except RuntimeError:
            continue
        for k, v in _level_dates(pb).items():
            boots[k].append(v)
    for k, v in boots.items():
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 10 else (np.nan, np.nan)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    out["n_boot_ok"] = len(boots["mid50"])
    return out


def _fit_double(t, y, w):
    p0 = [np.percentile(y, 5), np.percentile(y, 95), np.percentile(t, 25), 0.1, np.percentile(t, 80), 0.1]
    lo_t, hi_t = t.min(), t.max()
    bounds = ([-np.inf, -np.inf, lo_t, 0.005, lo_t, 0.005], [np.inf, np.inf, hi_t, 2.0, hi_t, 2.0])
    p, _ = curve_fit(double_logistic, t, y, p0=p0, sigma=1 / np.sqrt(np.clip(w, 1e-6, None)), bounds=bounds,
                     maxfev=40000)
    return p


def fit_season(t, y, w, n_boot: int = 300, block: int = 7, seed: int = 0) -> dict:
    """Spring logistic, or a double logistic when the record reaches autumn
    (beyond day 240) and the double form is clearly better (AIC lower by 10).
    The double fit adds the autumn half-way date eos50 with a block-bootstrap CI."""
    t, y, w = map(np.asarray, (t, y, w))
    single = fit_greenup(t, y, w, n_boot=n_boot, block=block, seed=seed)
    single.update(model="logistic", eos50=np.nan, eos50_lo=np.nan, eos50_hi=np.nan)
    if t.max() < 240:
        return single
    try:
        p = _fit_double(t, y, w)
    except RuntimeError:
        return single
    n = t.size
    aic_double = n * np.log(np.sum((y - double_logistic(t, *p)) ** 2) / n) + 2 * 6
    aic_single = n * np.log(single["rmse"] ** 2) + 2 * 4
    if aic_double > aic_single - 10:
        return single
    fitted = double_logistic(t, *p)
    res = y - fitted
    rng = np.random.default_rng(seed)
    boots = {"mid50": [], "eos50": [], "sos20": [], "mat80": []}
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n // block + 1)
        r = np.concatenate([res[s:s + block] for s in starts])[:n]
        try:
            pb = _fit_double(t, fitted + r, w)
        except RuntimeError:
            continue
        for k, v in _level_dates(pb[:4]).items():
            boots[k].append(v)
        boots["eos50"].append(pb[4])
    out = {"lo": p[0], "hi": p[1], "t0": p[2], "k": p[3], "t1": p[4], "k1": p[5],
           "rmse": float(np.sqrt(np.mean(res ** 2))), "n_days": int(n), "model": "double_logistic",
           **_level_dates(p[:4]), "eos50": float(p[4])}
    for k, v in boots.items():
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 10 else (np.nan, np.nan)
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    out["n_boot_ok"] = len(boots["mid50"])
    return out


def synth_season(rng, t0: float, noise: float = 0.1, frames_per_day: int = 12,
                 clear_prob: float = 0.6, days=np.arange(-30, 160)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic daily-median NDVI series: per-frame noise, cloudy days dropped."""
    keep = rng.random(days.size) < clear_prob
    d = days[keep]
    truth = logistic(d, 0.15, 0.6, t0, 0.12)
    n = rng.integers(3, frames_per_day + 1, d.size)
    y = np.array([np.median(m + rng.normal(0, noise, k)) for m, k in zip(truth, n)])
    return d.astype(float), y, n.astype(float)


def _t0_with_se(t, y, w) -> tuple[float, float]:
    p0 = [np.percentile(y, 5), np.percentile(y, 95), t[np.argmin(np.abs(y - np.median(y)))], 0.1]
    bounds = ([-np.inf, -np.inf, t.min(), 0.005], [np.inf, np.inf, t.max(), 2.0])
    p, cov = curve_fit(logistic, t, y, p0=p0, sigma=1 / np.sqrt(w), bounds=bounds, maxfev=20000)
    return p[2], float(np.sqrt(cov[2, 2]))


def shift_detect_power(shift_days: float, n_sim: int = 200, seed: int = 0, noise: float = 0.1,
                       clear_prob: float = 0.6, frames_per_day: int = 12) -> float:
    """Fraction of simulated ring pairs where the mid-green-up difference is
    significant at 5%. Simulated daily noise is independent, so the fit's own
    covariance gives the standard error (real data uses the block bootstrap)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        a, sa = _t0_with_se(*synth_season(rng, 120.0, noise, frames_per_day, clear_prob))
        b, sb = _t0_with_se(*synth_season(rng, 120.0 + shift_days, noise, frames_per_day, clear_prob))
        hits += abs(b - a) > 1.96 * np.hypot(sa, sb)
    return hits / n_sim


def half_rise_date(s: pd.Series, base_end: str, end_days: int = 7, smooth: int = 7, level: float = 0.5):
    """Model-free green-up date: first day after which the running median stays at or
    above half of the rise from the leafless baseline (days before base_end) to the
    last end_days. 'Stays' matters: red-edge curves rise in two steps with a pause
    near the half-way level, so a first touch is unstable."""
    s = s.sort_index()
    base = s[s.index < pd.Timestamp(base_end)].median()
    top = s.iloc[-end_days:].median()
    r = s.rolling(smooth, center=True, min_periods=1).median()
    r = r[r.index >= pd.Timestamp(base_end)]
    above = ((r - base) / (top - base)) >= level
    below = above.index[~above.to_numpy()]
    if not above.any():
        return pd.NaT
    after = above.index[above.index > below[-1]] if len(below) else above.index
    return after[0] if len(after) else pd.NaT


def paired_shift(t, ya, yb, n_boot: int = 300, block: int = 7, seed: int = 0) -> dict:
    """Half-way date of camera a minus camera b (days), with a moving-block bootstrap
    that draws the same day-blocks for both cameras so their shared day-to-day
    noise is kept."""
    t, ya, yb = map(np.asarray, (t, ya, yb))
    w = np.ones_like(t)
    pa, pb = _fit(t, ya, w), _fit(t, yb, w)
    fa, fb = logistic(t, *pa), logistic(t, *pb)
    ra, rb = ya - fa, yb - fb
    rng = np.random.default_rng(seed)
    n, diffs = t.size, []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        try:
            diffs.append(_fit(t, fa + ra[idx], w)[2] - _fit(t, fb + rb[idx], w)[2])
        except RuntimeError:
            continue
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) > 10 else (np.nan, np.nan)
    return {"shift": float(pa[2] - pb[2]), "lo": float(lo), "hi": float(hi), "n_boot_ok": len(diffs),
            "t0_a": float(pa[2]), "t0_b": float(pb[2]), "n_days": int(n)}
