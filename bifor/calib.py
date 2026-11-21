from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calib:
    camera: str
    M: np.ndarray          # (24, 25): raw cube-order bands -> corrected virtual bands
    wl: np.ndarray         # (24,) virtual band centres, nm
    fwhm: np.ndarray       # (24,)
    raw_wl: np.ndarray     # (25,) raw bands in cube order
    version: str


@lru_cache(maxsize=8)
def load(calib_dir: Path, camera: str) -> Calib:
    path = Path(calib_dir) / f"{camera}_calib.json"
    d = json.loads(path.read_text())
    M = np.asarray(d["correction_matrix"], dtype=np.float64)
    if M.shape != (24, 25):
        raise ValueError(f"{path}: correction matrix shape {M.shape}, expected (24, 25)")
    return Calib(
        camera=camera,
        M=M,
        wl=np.asarray(d["virtual_band_wavelength_nm"], dtype=np.float64),
        fwhm=np.asarray(d["virtual_band_fwhm_nm"], dtype=np.float64),
        raw_wl=np.asarray(d["raw_band_wavelength_nm"], dtype=np.float64),
        version=f"{d['correction_matrix_name']}-{d['correction_algorithm']}".replace(" ", ""),
    )
