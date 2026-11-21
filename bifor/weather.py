"""Hourly cloud cover and rain for the site from the Open-Meteo archive (no key)."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float, start: str, end: str, cache: Path) -> pd.DataFrame:
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        d = json.loads(cache.read_text())
    else:
        r = httpx.get(URL, params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                   "hourly": "cloud_cover,precipitation,shortwave_radiation", "timezone": "UTC"},
                      timeout=60)
        r.raise_for_status()
        d = r.json()
        cache.write_text(json.dumps(d))
    h = d["hourly"]
    return pd.DataFrame({"hour_utc": pd.to_datetime(h["time"], utc=True), "cloud_pct": h["cloud_cover"],
                         "precip_mm": h["precipitation"], "sw_rad": h["shortwave_radiation"]})


def join(frames: pd.DataFrame, wx: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Attach the hour's weather. Open-Meteo radiation is the mean of the
    preceding hour, so a frame at 12:20 takes the 13:00 value."""
    import pvlib
    f = frames.copy()
    f["hour_utc"] = f.ts_utc.dt.ceil("h")
    f = f.merge(wx, on="hour_utc", how="left").drop(columns="hour_utc")
    ghi = pvlib.location.Location(lat, lon).get_clearsky(pd.DatetimeIndex(f.ts_utc))["ghi"].to_numpy()
    f["clearness"] = (f.sw_rad.to_numpy() / pd.Series(ghi).clip(lower=20).to_numpy()).clip(0, 1.5)
    return f
