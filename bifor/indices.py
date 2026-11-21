"""Wavelength-based index registry, resolved per camera.

Each index names the nominal wavelengths it needs. For a given camera the
engine picks the nearest corrected band; if that band is further away than
the tolerance the index is switched off with the reason recorded.

scale_invariant=False marks indices that change when the whole spectrum is
multiplied by a constant (differences, sums, products with a band). The
cameras auto-expose and there is no reflectance panel, so these only carry
relative meaning within a frame and are kept out of default time series.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from .config import ROOT


@lru_cache(maxsize=1)
def slot_wl() -> np.ndarray:
    """Mean centre of each filter slot across A5 and A6. Bands are picked by slot
    so both cameras compute an index from the same filters; A6 sits 0.4-1.5 nm
    below A5, enough to flip a nearest-band choice when a target falls midway."""
    from . import calib, config
    cfg = config.load()
    return (calib.load(cfg.calib_dir, "A5").wl + calib.load(cfg.calib_dir, "A6").wl) / 2


def _slot(nm: float, wl: np.ndarray) -> int:
    ref = slot_wl() if len(wl) == len(slot_wl()) else wl
    return int(np.argmin(np.abs(ref - nm)))


class Spec:
    """Band accessor over a (24, ...) array at known centres."""

    def __init__(self, x: np.ndarray, wl: np.ndarray):
        self.x, self.wl = np.asarray(x, dtype=np.float64), np.asarray(wl, dtype=np.float64)

    def i(self, nm: float) -> int:
        return _slot(nm, self.wl)

    def __call__(self, nm: float) -> np.ndarray:
        return self.x[self.i(nm)]

    def at(self, nm: float) -> float:
        return float(self.wl[self.i(nm)])

    def deriv(self) -> tuple[np.ndarray, np.ndarray]:
        dw = np.diff(self.wl).reshape((-1,) + (1,) * (self.x.ndim - 1))
        return (self.wl[1:] + self.wl[:-1]) / 2, np.diff(self.x, axis=0) / dw


@dataclass(frozen=True)
class Index:
    name: str
    wavelengths: tuple[float, ...]
    fn: Callable[[Spec], np.ndarray]
    kind: str = "ratio"                 # ratio | nd | position | other
    scale_invariant: bool = True
    needs_absolute: bool = False
    note: str = ""


@dataclass(frozen=True)
class Resolution:
    enabled: bool
    bands: tuple[float, ...] = ()
    reason: str = ""


def nd(a, b):
    return (a - b) / (a + b)


def _ndvi(R):
    return nd(R(794), R(665))


def _rep_le(R):
    l1, l2 = R.at(704), R.at(742)
    return l1 + (l2 - l1) * ((R(665) + R(781)) / 2 - R(704)) / (R(742) - R(704))


def _rep_extrap(R):
    mid, d = R.deriv()
    ref = slot_wl() if len(R.wl) == len(slot_wl()) else R.wl
    ref_mid = (ref[1:] + ref[:-1]) / 2
    k = [int(np.argmin(np.abs(ref_mid - t))) for t in (680, 694, 724, 760)]
    s1 = (d[k[1]] - d[k[0]]) / (mid[k[1]] - mid[k[0]])
    s2 = (d[k[3]] - d[k[2]]) / (mid[k[3]] - mid[k[2]])
    c1, c2 = d[k[0]] - s1 * mid[k[0]], d[k[2]] - s2 * mid[k[2]]
    return (c2 - c1) / (s1 - s2)


def _sdr(R):
    mid, d = R.deriv()
    keep = (mid >= 680 - 10) & (mid <= 760)
    return d[keep].sum(axis=0)


def _naoc(R):
    keep = (R.wl >= 665 - 10) & (R.wl <= 781 + 5)
    return (1 - R.x[keep]).sum(axis=0)


_I = Index
REGISTRY: dict[str, Index] = {i.name: i for i in [
    _I("NDVI", (794, 665), _ndvi, "nd"),
    _I("NDRE", (794, 717), lambda R: nd(R(794), R(717)), "nd"),
    _I("CIre", (755, 717), lambda R: R(755) / R(717) - 1),
    _I("MTCI", (755, 704, 678), lambda R: (R(755) - R(704)) / (R(704) - R(678))),
    _I("IRECI", (781, 665, 704, 742), lambda R: (R(781) - R(665)) / (R(704) / R(742)), scale_invariant=False),
    _I("S2REP", (781, 665, 704, 742), lambda R: 704 + 35 * ((R(781) + R(665)) / 2 - R(704)) / (R(742) - R(704)), "position",
       note="Nominal constants as published; REP-LE uses the camera's real band centres."),
    _I("REP-LE", (665, 704, 742, 781), _rep_le, "position"),
    _I("REP-extrap", (680, 694, 724, 760), _rep_extrap, "position"),
    _I("Maccioni", (781, 704, 678), lambda R: (R(781) - R(704)) / (R(781) - R(678))),
    _I("SR705", (755, 717), lambda R: R(755) / R(717)),
    _I("ND705", (755, 717), lambda R: nd(R(755), R(717)), "nd"),
    _I("VOG1", (742, 717), lambda R: R(742) / R(717)),
    _I("VOG2", (730, 742, 717), lambda R: (R(730) - R(742)) / (R(717) + R(730))),
    _I("NDI(704,665)", (704, 665), lambda R: nd(R(704), R(665)), "nd"),
    _I("NIRv", (794, 665), lambda R: _ndvi(R) * R(794), scale_invariant=False),
    _I("kNDVI", (794, 665), lambda R: np.tanh(_ndvi(R) ** 2), "nd"),
    _I("EVI2", (794, 665), lambda R: 2.5 * (R(794) - R(665)) / (R(794) + 2.4 * R(665) + 1),
       scale_invariant=False, needs_absolute=True),
    _I("SCCCI", (794, 717, 665), lambda R: nd(R(794), R(717)) / _ndvi(R)),
    _I("RVI / SR", (794, 665), lambda R: R(794) / R(665)),
    _I("RDVI", (794, 665), lambda R: (R(794) - R(665)) / np.sqrt(R(794) + R(665)), scale_invariant=False),
    _I("DVI", (794, 665), lambda R: R(794) - R(665), scale_invariant=False),
    _I("TTVI", (859, 742, 781), lambda R: 0.5 * np.abs(43 * (R(859) - R(742)) - 125 * (R(781) - R(742))),
       scale_invariant=False),
    _I("NAOC", (665, 781), _naoc, scale_invariant=False, needs_absolute=True),
    _I("ISI", (870, 755), lambda R: R(870) - R(755), scale_invariant=False),
    _I("RVSI", (717, 755, 742), lambda R: (R(717) + R(755)) / 2 - R(742), scale_invariant=False),
    _I("REDSI", (781, 665, 704), lambda R: (40 * (R(781) - R(665)) - 118 * (R(704) - R(665))) / (2 * R(665))),
    _I("WI/WBI", (897, 962), lambda R: R(897) / R(962)),
    _I("Ctr2", (691, 755), lambda R: R(691) / R(755)),
    _I("Ctr4", (704, 755), lambda R: R(704) / R(755)),
    _I("Ctr5", (691, 665), lambda R: R(691) / R(665)),
    _I("NDI45", (755, 717), lambda R: nd(R(755), R(717)), "nd", note="Same formula as ND705 on this sensor."),
    _I("SDr", (680, 760), _sdr, scale_invariant=False),
    _I("PSRI", (678, 500, 750), lambda R: (R(678) - R(500)) / R(750)),
    _I("PRI", (531, 570), lambda R: nd(R(531), R(570)), "nd"),
]}

NOT_COMPUTED: dict[str, str] = {
    "R755+R936 pair": "A pair of regression features, not one index. Both bands are in roi_spectra.",
}


@lru_cache(maxsize=1)
def _catalog():
    path = ROOT.parent / "BIFOR_analysis" / "catalog_data.py"
    spec = importlib.util.spec_from_file_location("catalog_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def catalog_shorts() -> list[str]:
    return [i["short"] for i in _catalog().INDICES]


def catalog_entry(name: str) -> dict:
    for i in _catalog().INDICES:
        if i["short"] == name:
            return i
    for i in _catalog().EXCLUDED:
        if i["short"] == name:
            return i
    return {}


def resolve(idx: Index, wl: np.ndarray, tol: float) -> Resolution:
    wl = np.asarray(wl)
    bands = []
    for nm in idx.wavelengths:
        j = _slot(nm, wl)
        d = abs(wl[j] - nm)
        if d > tol:
            return Resolution(False, reason=f"needs {nm:g} nm; nearest band {wl[j]:.1f} nm is {d:.1f} nm away "
                                            f"(sensor covers {wl.min():.1f}-{wl.max():.1f} nm)")
        bands.append(round(float(wl[j]), 1))
    if idx.needs_absolute:
        return Resolution(True, tuple(bands), "computed, but assumes absolute reflectance (no panel in view)")
    return Resolution(True, tuple(bands))


def compute_all(x: np.ndarray, wl: np.ndarray, tol: float, names=None) -> dict[str, np.ndarray]:
    R = Spec(x, wl)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for name in names or REGISTRY:
            idx = REGISTRY[name]
            if resolve(idx, wl, tol).enabled:
                out[name] = idx.fn(R)
    return out


def compute_one(name: str, x: np.ndarray, wl: np.ndarray, tol: float):
    res = resolve(REGISTRY[name], wl, tol)
    if not res.enabled:
        raise ValueError(f"{name} is switched off: {res.reason}")
    return compute_all(x, wl, tol, [name])[name]


def resolution_table(calibs: dict, tol: float):
    import pandas as pd
    rows = []
    for name, idx in REGISTRY.items():
        entry = catalog_entry(name)
        row = {"index": name, "wavelengths_nm": ", ".join(f"{w:g}" for w in idx.wavelengths),
               "focus_areas": str(entry.get("area", "")), "scale_invariant": idx.scale_invariant,
               "citation": entry.get("backing", "").split(";")[0], "note": idx.note}
        for cam, cal in calibs.items():
            r = resolve(idx, cal.wl, tol)
            row[f"{cam}_bands"] = ", ".join(f"{b:.1f}" for b in r.bands) if r.enabled else "off"
            row[f"{cam}_reason"] = r.reason
        rows.append(row)
    for name, reason in NOT_COMPUTED.items():
        rows.append({"index": name, "note": reason})
    return pd.DataFrame(rows)
