from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    raw_root: Path
    work_dir: Path
    calib_dir: Path
    roi_dir: Path
    local_sample: Path
    lat: float
    lon: float
    filename_tz: str
    cameras: dict[str, str]
    sat_16bit: int
    sat_8bit: int
    dark_hour_local: int
    master_dark_quantile: float = 0.4
    qa: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    window: dict = field(default_factory=dict)
    treatment: dict = field(default_factory=dict)

    @property
    def start_utc(self):
        import pandas as pd
        s = self.window.get("start_utc")
        return pd.Timestamp(s) if s else None

    @property
    def leafless_end(self) -> str:
        return str(self.window.get("leafless_end", "2026-04-01"))

    @property
    def band_tol(self) -> float:
        return float(self.qa["band_tolerance_nm"])

    @property
    def noon_window_min(self) -> int:
        return int(self.qa["noon_window_min"])


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q).resolve()


def load(path: Path | None = None) -> Config:
    path = Path(path or os.environ.get("BIFOR_CONFIG", ROOT / "bifor.toml"))
    d = tomllib.loads(path.read_text(encoding="utf-8"))
    p, dec = d["paths"], d["decode"]
    return Config(
        raw_root=_resolve(os.environ.get("BIFOR_RAW", p["raw_root"])),
        work_dir=_resolve(os.environ.get("BIFOR_WORK", p["work_dir"])),
        calib_dir=_resolve(p["calib_dir"]),
        roi_dir=_resolve(p["roi_dir"]),
        local_sample=_resolve(p["local_sample"]),
        lat=float(d["site"]["lat"]),
        lon=float(d["site"]["lon"]),
        filename_tz=d["site"]["filename_tz"],
        cameras={str(k): v for k, v in d["cameras"].items()},
        sat_16bit=int(dec["sat_16bit"]),
        sat_8bit=int(dec["sat_8bit"]),
        dark_hour_local=int(dec["dark_hour_local"]),
        master_dark_quantile=float(dec.get("master_dark_quantile", 0.4)),
        qa=dict(d["qa"]),
        versions=dict(d["versions"]),
        window=dict(d.get("window", {})),
        treatment=dict(d.get("treatment", {})),
    )
