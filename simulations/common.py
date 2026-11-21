"""Shared simulation helpers: PROSAIL canopy spectra, camera band resampling,
output folders."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import prosail

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import calib, config, indices  # noqa: E402

CFG = config.load()
WL = np.arange(400, 2501, dtype=float)
OUT = CFG.work_dir / "sim"
FIG = CFG.work_dir / "figures" / "sim"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CAL = {c: calib.load(CFG.calib_dir, c) for c in ("A5", "A6")}
TOL = CFG.band_tol

# Oak canopy defaults (PROSPECT-D + 4SAIL). Ranges follow typical broadleaf values.
BASE = dict(n=1.6, cab=45.0, car=9.0, cbrown=0.0, cw=0.012, cm=0.009, lai=4.0, lidfa=57.0,
            hspot=0.05, tts=40.0, tto=60.0, psi=90.0, ant=1.0)
RANGES = dict(n=(1.2, 2.2), cab=(10, 80), car=(3, 18), cbrown=(0.0, 0.6), cw=(0.004, 0.03),
              cm=(0.004, 0.015), lai=(0.3, 6.5), lidfa=(30, 75), tts=(30, 77), tto=(40, 80), psi=(0, 180))


def canopy(**kw) -> np.ndarray:
    p = {**BASE, **kw}
    return prosail.run_prosail(p["n"], p["cab"], p["car"], p["cbrown"], p["cw"], p["cm"], p["lai"], p["lidfa"],
                               p["hspot"], p["tts"], p["tto"], p["psi"], ant=p["ant"], prospect_version="D",
                               typelidf=2, rsoil=1.0, psoil=0.5, factor="SDR")


def srf(cam: str) -> np.ndarray:
    """Gaussian response per corrected band from the calibration centres and FWHM."""
    c = CAL[cam]
    s = c.fwhm / 2.3548
    r = np.exp(-0.5 * ((WL[None, :] - c.wl[:, None]) / s[:, None]) ** 2)
    return r / r.sum(1, keepdims=True)


SRF = {c: srf(c) for c in CAL}


def bands(spec: np.ndarray, cam: str = "A5") -> np.ndarray:
    """(…, 2101) spectra -> (24, …) camera bands."""
    return np.moveaxis(spec @ SRF[cam].T, -1, 0)


def sample_params(n: int, rng, fixed: dict | None = None) -> list[dict]:
    out = []
    for _ in range(n):
        p = {k: rng.uniform(*v) for k, v in RANGES.items()}
        out.append({**p, **(fixed or {})})
    return out


def enabled_invariant() -> list[str]:
    w = CAL["A5"].wl
    return [n for n, i in indices.REGISTRY.items() if i.scale_invariant and indices.resolve(i, w, TOL).enabled]


def all_enabled() -> list[str]:
    w = CAL["A5"].wl
    return [n for n, i in indices.REGISTRY.items() if indices.resolve(i, w, TOL).enabled]
