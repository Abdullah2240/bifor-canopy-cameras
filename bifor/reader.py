"""Any frame format -> 24-band corrected cube at true wavelengths.

Raw mosaics are 2048x1088 with a 5x5 filter pattern; band k sits at mosaic
offset (k // 5, k % 5). 16-bit files hold 10-bit data shifted left 6 bits;
8-bit files hold the top 8 bits. Both are brought to 10-bit DN.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .calib import Calib

H, W = 217, 409
MOSAIC_SHAPE = (1088, 2048)


@dataclass
class Decoded:
    bands: np.ndarray   # (24, H, W) float32, NaN where saturated
    sat: np.ndarray     # (H, W) bool
    bit_depth: int
    dn_mean: float      # mean 10-bit DN before dark subtraction


def demosaic(mosaic: np.ndarray) -> np.ndarray:
    return np.stack([mosaic[oy::5, ox::5][:H, :W] for oy in range(5) for ox in range(5)])


def read_bands(path: Path) -> tuple[np.ndarray, int]:
    """Return (25, H, W) native-dtype bands and bit depth (8 or 16)."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        a = np.asarray(Image.open(path))
    else:
        a = tifffile.imread(path)
    if a.ndim == 3 and a.shape == (25, H, W):
        bands = a
    elif a.shape == MOSAIC_SHAPE:
        bands = demosaic(a)
    else:
        raise ValueError(f"{path.name}: shape {a.shape}, expected mosaic 1088x2048 or cube (25, {H}, {W})")
    if a.dtype == np.uint8:
        return bands, 8
    if a.dtype == np.uint16:
        return bands, 16
    raise ValueError(f"{path.name}: unsupported dtype {a.dtype}")


def to_dn10(bands: np.ndarray, bit: int) -> np.ndarray:
    return bands.astype(np.float32) * (4.0 if bit == 8 else 1.0 / 64.0)


def load_dark(path: Path) -> np.ndarray:
    bands, bit = read_bands(path)
    return to_dn10(bands, bit)


def decode(path: Path, calib: Calib, dark: np.ndarray | None, cfg) -> Decoded:
    raw, bit = read_bands(path)
    return decode_bands(raw, bit, calib, dark, cfg)


def decode_bands(raw: np.ndarray, bit: int, calib: Calib, dark: np.ndarray | None, cfg) -> Decoded:
    sat = (raw >= (cfg.sat_8bit if bit == 8 else cfg.sat_16bit)).any(0)
    dn = to_dn10(raw, bit)
    dn_mean = float(dn.mean())
    if dark is not None:
        dn = dn - dark
    corr = (calib.M @ dn.reshape(25, -1).astype(np.float64)).reshape(24, H, W).astype(np.float32)
    corr[:, sat] = np.nan
    return Decoded(corr, sat, bit, dn_mean)
