from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import typer

from . import calib, citations, config, indices, pipeline

app = typer.Typer(no_args_is_help=True, help="BIFoR-FACE spectral pipeline")


def _cfg():
    return config.load()


@app.command()
def scan(raw_root: Path = typer.Option(None), work_dir: Path = typer.Option(None)):
    """Walk the raw drive and write catalog.parquet with sun position per frame."""
    cat = pipeline.scan(_cfg(), raw_root, work_dir)
    typer.echo(f"{len(cat)} frames")
    typer.echo(cat.groupby(["camera", "format"]).size().to_string())


@app.command()
def sample(every_days: int = 14, work_dir: Path = typer.Option(None)):
    """Pick a pilot subset: every Nth day, both cameras."""
    s = pipeline.sample(_cfg(), every_days, work_dir)
    typer.echo(f"{len(s)} frames on {s.ts_local.dt.date.nunique()} days")


@app.command()
def process(workers: int = os.cpu_count() // 2 or 1, limit: int = 0, use_sample: bool = False,
            work_dir: Path = typer.Option(None)):
    """Decode, QA-measure, extract ROI spectra and indices for new frames."""
    n = pipeline.process(_cfg(), work_dir, workers, limit or None, use_sample)
    typer.echo(f"{n} frames processed")


@app.command()
def weather(work_dir: Path = typer.Option(None)):
    """Fetch hourly cloud cover and rain for the catalog period."""
    wx = pipeline.fetch_weather(_cfg(), work_dir)
    typer.echo(f"{len(wx)} hours")


@app.command()
def daily(work_dir: Path = typer.Option(None)):
    d = pipeline.build_daily(_cfg(), work_dir)
    typer.echo(f"{len(d)} daily rows over {d.date.nunique()} days")


@app.command()
def fit(n_boot: int = 300, min_days: int = 15, work_dir: Path = typer.Option(None)):
    f = pipeline.fit_all(_cfg(), work_dir, min_days, n_boot)
    typer.echo(f"{len(f)} fits")


@app.command("resolve-bands")
def resolve_bands(work_dir: Path = typer.Option(None)):
    """Write index_resolution.csv: which band each camera uses per index, or why it is off."""
    cfg = _cfg()
    t = indices.resolution_table({c: calib.load(cfg.calib_dir, c) for c in ("A5", "A6")}, cfg.band_tol)
    out = Path(work_dir or cfg.work_dir) / "index_resolution.csv"
    t.to_csv(out, index=False)
    typer.echo(f"{out}  ({(t.A5_bands != 'off').sum()} enabled on A5)")


@app.command("check-citations")
def check_citations(work_dir: Path = typer.Option(None)):
    """Verify each index citation against Crossref (first author + year)."""
    cfg = _cfg()
    t = citations.run()
    out = Path(work_dir or cfg.work_dir) / "citations.csv"
    t.to_csv(out, index=False)
    typer.echo(t.status.value_counts().to_string())


@app.command()
def figures(work_dir: Path = typer.Option(None)):
    """Render all data figures into <work>/figures."""
    from . import figures as fg
    for p in fg.render_all(_cfg(), work_dir):
        typer.echo(p)


@app.command()
def run(use_sample: bool = False, workers: int = os.cpu_count() // 2 or 1, work_dir: Path = typer.Option(None)):
    """process -> weather -> daily -> fit -> figures."""
    cfg = _cfg()
    typer.echo(f"processed {pipeline.process(cfg, work_dir, workers, None, use_sample)} frames")
    pipeline.fetch_weather(cfg, work_dir)
    pipeline.build_daily(cfg, work_dir)
    pipeline.fit_all(cfg, work_dir)
    from . import figures as fg
    for p in fg.render_all(cfg, work_dir):
        typer.echo(p)


if __name__ == "__main__":
    app()
