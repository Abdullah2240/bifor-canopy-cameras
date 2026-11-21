"""Walk the raw drive once and record one row per frame (camera + timestamp)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

TS = re.compile(r"_(\d{8})_(\d{6})")
EXTS = {".png", ".tif", ".tiff"}


def _walk(root: Path):
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif os.path.splitext(e.name)[1].lower() in EXTS:
                    yield e


def scan(root: Path, cfg) -> pd.DataFrame:
    frames: dict[tuple[str, str], dict] = {}
    for e in _walk(Path(root)):
        parts = Path(e.path).parts
        sensor = next((p for p in parts if p in cfg.cameras), None)
        m = TS.search(e.name)
        if sensor is None or m is None:
            continue
        key = (cfg.cameras[sensor], m.group(1) + m.group(2))
        row = frames.setdefault(key, {"camera": key[0], "sensor": sensor, "stamp": key[1], "size_bytes": 0})
        row["size_bytes"] += e.stat().st_size
        name = e.name.lower()
        if "_cube" in name:
            row["cube_path"] = e.path
        elif "_raw" in name:
            row["raw_path"] = e.path
        else:
            row["mosaic_path"] = e.path
    if not frames:
        raise FileNotFoundError(f"no camera frames found under {root}")
    df = pd.DataFrame(frames.values())
    for c in ("cube_path", "raw_path", "mosaic_path"):
        if c not in df:
            df[c] = None
    df["format"] = df.apply(_format, axis=1)
    df["read_path"] = df.cube_path.fillna(df.raw_path).fillna(df.mosaic_path)
    df["ts_local"] = pd.to_datetime(df.stamp, format="%Y%m%d%H%M%S")
    df["ts_utc"] = to_utc(df.ts_local, cfg.filename_tz)
    bad = df.ts_utc.isna()
    if bad.any():
        print(f"warning: {bad.sum()} frames fall in a clock-change gap and are dropped", file=sys.stderr)
        df = df[~bad]
    df["frame_id"] = df.camera + "_" + df.ts_utc.dt.strftime("%Y%m%dT%H%M%S")
    cols = ["frame_id", "camera", "sensor", "ts_local", "ts_utc", "format", "read_path",
            "raw_path", "cube_path", "mosaic_path", "size_bytes"]
    return df[cols].sort_values(["camera", "ts_utc"]).reset_index(drop=True)


def _format(r) -> str:
    if isinstance(r.cube_path, str) or isinstance(r.raw_path, str):
        return "pair" if isinstance(r.cube_path, str) else "raw"
    return "png" if r.mosaic_path.lower().endswith(".png") else "tiff"


def to_utc(local: pd.Series, tz: str) -> pd.Series:
    s = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    return s.dt.tz_convert("UTC")
