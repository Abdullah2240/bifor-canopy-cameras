"""Regions of interest: pixel polygons per camera plus a static structure mask.

rois/<CAM>_<ver>.geojson   polygons in pixel coordinates (x = column, y = row)
rois/<CAM>_<ver>_exclude.npy  pylons, towers and sky (built by `bifor roi-ref`)
rois/<CAM>_<ver>_ref.npy      reference NIR image for camera-shift detection
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import polygon as draw_polygon

from .reader import H, W


def _paths(roi_dir: Path, camera: str, version: str):
    base = Path(roi_dir) / f"{camera}_{version}"
    return base.with_suffix(".geojson"), Path(f"{base}_exclude.npy"), Path(f"{base}_ref.npy")


def polygons(roi_dir: Path, camera: str, version: str) -> dict[str, list]:
    gj, _, _ = _paths(roi_dir, camera, version)
    d = json.loads(gj.read_text())
    return {f["properties"]["name"]: f["geometry"]["coordinates"][0] for f in d["features"]}


def load(roi_dir: Path, camera: str, version: str) -> dict[str, np.ndarray]:
    _, excl_p, _ = _paths(roi_dir, camera, version)
    excl = np.load(excl_p) if excl_p.exists() else np.zeros((H, W), bool)
    masks = {}
    for name, coords in polygons(roi_dir, camera, version).items():
        xy = np.asarray(coords, dtype=float)
        m = np.zeros((H, W), bool)
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], (H, W))
        m[rr, cc] = True
        masks[name] = m & ~excl
    return masks


def exclusion(roi_dir: Path, camera: str, version: str) -> np.ndarray:
    _, excl_p, _ = _paths(roi_dir, camera, version)
    return np.load(excl_p) if excl_p.exists() else np.zeros((H, W), bool)


def reference(roi_dir: Path, camera: str, version: str) -> np.ndarray | None:
    _, _, ref_p = _paths(roi_dir, camera, version)
    return np.load(ref_p) if ref_p.exists() else None


def build_reference(roi_dir: Path, camera: str, version: str, ndvi_stack: np.ndarray,
                    sat_stack: np.ndarray, nir_stack: np.ndarray, ndvi_max: float = -0.02) -> np.ndarray:
    """Static exclusion = pixels whose median NDVI is below ndvi_max (painted pylons, towers)
    or that saturate in most frames (sky), grown by one pixel."""
    _, excl_p, ref_p = _paths(roi_dir, camera, version)
    excl = (np.nanmedian(ndvi_stack, 0) < ndvi_max) | (sat_stack.mean(0) > 0.5)
    excl = binary_dilation(excl, iterations=1)
    np.save(excl_p, excl)
    np.save(ref_p, np.nanmedian(nir_stack, 0).astype(np.float32))
    return excl


def shift_masks(masks: dict[str, np.ndarray], dy: float, dx: float) -> dict[str, np.ndarray]:
    iy, ix = int(round(dy)), int(round(dx))
    if iy == 0 and ix == 0:
        return masks
    out = {}
    for k, m in masks.items():
        s = np.roll(m, (iy, ix), axis=(0, 1))
        if iy > 0: s[:iy] = False
        if iy < 0: s[iy:] = False
        if ix > 0: s[:, :ix] = False
        if ix < 0: s[:, ix:] = False
        out[k] = s
    return out


def spectra(bands: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Per ROI: median, p10, p90 per band and count of valid pixels."""
    out = {}
    for name, m in masks.items():
        v = bands[:, m]
        ok = np.isfinite(v).all(0)
        v = v[:, ok]
        if v.shape[1] == 0:
            nan = np.full(bands.shape[0], np.nan)
            out[name] = {"median": nan, "p10": nan, "p90": nan, "n_valid": 0}
            continue
        p10, med, p90 = np.percentile(v, [10, 50, 90], axis=1)
        out[name] = {"median": med, "p10": p10, "p90": p90, "n_valid": int(v.shape[1])}
    return out
