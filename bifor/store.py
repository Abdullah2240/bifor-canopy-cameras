"""Parquet tables as folders of part files, partitioned by camera."""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

VERSION_COLS = ("code_v", "roi_v", "calib_v")


def write(df: pd.DataFrame, table: Path) -> None:
    if df.empty:
        return
    for cam, g in df.groupby("camera"):
        d = Path(table) / f"camera={cam}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".part-{uuid.uuid4().hex}.tmp"
        g.drop(columns="camera").to_parquet(tmp, index=False)
        tmp.rename(d / f"part-{uuid.uuid4().hex}.parquet")


def read(table: Path, columns: list[str] | None = None) -> pd.DataFrame:
    table = Path(table)
    parts = sorted(table.glob("camera=*/part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=columns or [])
    frames = []
    for p in parts:
        cols = [c for c in columns if c != "camera"] if columns else None
        f = pd.read_parquet(p, columns=cols)
        f["camera"] = p.parent.name.split("=", 1)[1]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def current(df: pd.DataFrame, versions: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = pd.Series(True, index=df.index)
    for col, v in versions.items():
        keep &= df[col] == v
    return df[keep]
