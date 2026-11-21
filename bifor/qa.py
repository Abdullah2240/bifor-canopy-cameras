"""Per-frame quality metrics and the pass/fail rule.

Metrics are stored as columns; the pass rule is applied when tables are read,
so thresholds can change without reprocessing frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from scipy.ndimage import laplace
from skimage.registration import phase_cross_correlation


def sun_position(ts_utc: pd.Series, lat: float, lon: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(ts_utc)
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon)
    noon_min = 720 - 4 * lon - sp["equation_of_time"].to_numpy()
    minute = idx.hour * 60 + idx.minute + idx.second / 60
    return pd.DataFrame({"sun_elev": sp["apparent_elevation"].to_numpy(),
                         "sun_az": sp["azimuth"].to_numpy(),
                         "min_from_noon": np.asarray(minute - noon_min)}, index=ts_utc.index)


def _fill(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    return np.where(np.isfinite(img), img, np.nanmedian(img))


def shift(ref: np.ndarray, img: np.ndarray, ref_mask: np.ndarray | None = None,
          max_px: int = 5) -> tuple[float, float]:
    """How far img has moved relative to ref, in pixels (dy, dx).

    With ref_mask, uses masked normalised cross-correlation (whole pixels) so
    only the masked structure (pylons and their edges) drives the match.
    Blanking the rest of both images instead makes the mask outline itself
    dominate, and the result is always ~0."""
    if ref_mask is None:
        s, _, _ = phase_cross_correlation(_fill(ref), _fill(img), upsample_factor=10)
        return float(-s[0]), float(-s[1])
    return _masked_search(ref, img, ref_mask, max_px)


def _masked_search(ref, img, ref_mask, max_px: int = 5) -> tuple[float, float]:
    """Brute-force search over +-max_px for the offset that maximises the
    Pearson correlation of ref and img on the masked pixels. A small window
    cannot alias onto the next pylon the way a global FFT match can. Keep it
    under the ~8 px vertical repeat of the pylon cross-bracing."""
    H, W = ref.shape
    m = ref_mask & np.isfinite(ref)
    m[:max_px], m[-max_px:], m[:, :max_px], m[:, -max_px:] = False, False, False, False
    yy, xx = np.nonzero(m)
    a = ref[yy, xx]
    a = (a - a.mean()) / (a.std() + 1e-12)
    best, arg = -np.inf, (0, 0)
    for dy in range(-max_px, max_px + 1):
        for dx in range(-max_px, max_px + 1):
            b = img[yy + dy, xx + dx]
            ok = np.isfinite(b)
            if ok.sum() < 0.5 * b.size:
                continue
            bb = b[ok]
            r = float(np.dot(a[ok], (bb - bb.mean()) / (bb.std() + 1e-12)) / ok.sum())
            if r > best:
                best, arg = r, (dy, dx)
    if max(abs(arg[0]), abs(arg[1])) == max_px:
        return float("nan"), float("nan")      # peak on the window edge: the true match lies outside it
    return float(arg[0]), float(arg[1])


def sharpness(img: np.ndarray) -> float:
    a = _fill(img)
    return float(np.var(laplace(a)) / max(np.mean(a) ** 2, 1e-12))


def light_ratio(nir: np.ndarray, mask: np.ndarray) -> float:
    """Sunlit/shaded contrast in the canopy (p90/p10 of NIR). Drops under cloud."""
    v = nir[mask & np.isfinite(nir)]
    if v.size < 100:
        return float("nan")
    p10, p90 = np.percentile(v, [10, 90])
    return float(p90 / p10) if p10 > 0 else float("nan")


def light_jump(df: pd.DataFrame, window: str = "61min") -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("camera"):
        g = g.sort_values("ts_utc")
        s = pd.Series(g.light_ratio.to_numpy(), index=pd.DatetimeIndex(g.ts_utc))
        med = s.rolling(window, center=True, min_periods=3).median()
        out.loc[g.index] = np.abs(s.to_numpy() / med.to_numpy() - 1)
    return out


RULES = [
    ("low_sun", lambda d, q: d.sun_elev < q["min_sun_elev"]),
    ("saturated", lambda d, q: d.sat_frac > q["max_sat_frac"]),
    ("light_jump", lambda d, q: d.light_jump > q["max_light_jump"]),
    ("cloud", lambda d, q: d.cloud_pct > q["max_cloud_pct"]),
    ("rain", lambda d, q: d.precip_mm > q["max_precip_mm"]),
    ("camera_shift", lambda d, q: d.shift_px > q["max_shift_px"]),
    ("blur_fog", lambda d, q: d.sharpness < _per_camera(d, q["min_sharpness"])),
    ("frost_snow", lambda d, q: _leafless(d, q) & (d.sat_frac > q.get("max_sat_frac_leafless", np.inf))),
]


def _per_camera(d: pd.DataFrame, v) -> pd.Series | float:
    """A threshold given as {camera: value} or as one number for all cameras."""
    if isinstance(v, dict):
        return d.camera.map(v).astype(float)
    return float(v)


def _leafless(d: pd.DataFrame, q: dict) -> pd.Series:
    """In leafless months nothing in the canopy ROI should clip except frost, snow or ice."""
    months = q.get("leafless_months", [])
    if "ts_utc" not in d or not months:
        return pd.Series(False, index=d.index)
    return d.ts_utc.dt.month.isin(months)


def apply(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    for c in ("light_jump", "cloud_pct", "precip_mm", "shift_px", "sharpness"):
        if c not in df:
            df[c] = np.nan
    reason = pd.Series("", index=df.index)
    for name, rule in RULES:
        hit = rule(df, cfg.qa).fillna(False) & (reason == "")
        reason[hit] = name
    df["qa_reason"] = reason
    df["qa_pass"] = reason == ""
    return df
