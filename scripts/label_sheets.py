"""Render 200 near-noon frames (stratified by camera and month) as numbered
false-colour contact sheets for visual QA labelling."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import calib, config, indices, pipeline, reader  # noqa: E402

cfg = config.load()
OUT = cfg.work_dir / "qa_label"
OUT.mkdir(exist_ok=True)
cat = pd.read_parquet(cfg.work_dir / "catalog.parquet")
c = cat[(cat.min_from_noon.abs() <= cfg.noon_window_min) & (cat.sun_elev > 5)].copy()
c["month"] = c.ts_utc.dt.strftime("%m")
pick = c.groupby(["camera", "month"]).sample(n=15, random_state=1)
pick = pick.sample(frac=1, random_state=2).head(200).reset_index(drop=True)
pick["n"] = range(len(pick))
pick[["n", "frame_id", "camera", "ts_utc"]].to_csv(OUT / "sample.csv", index=False)
for s in range(0, len(pick), 20):
    fig, axes = plt.subplots(5, 4, figsize=(16, 11))
    for ax, r in zip(axes.ravel(), pick.iloc[s:s + 20].itertuples()):
        cal = calib.load(cfg.calib_dir, r.camera)
        dp, _ = pipeline.master_dark(cfg.work_dir, r.camera, r.format)
        d = reader.decode(Path(r.read_path), cal, np.load(dp), cfg)
        R = indices.Spec(d.bands, cal.wl)
        rgb = np.dstack([R(860), R(716), R(674)])
        for k in range(3):
            lo, hi = np.nanpercentile(rgb[..., k], [2, 98])
            rgb[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo + 1e-9), 0, 1)
        ax.imshow(np.nan_to_num(rgb, nan=1.0))
        ax.set_title(f"#{r.n} {r.camera} {r.ts_utc:%d %b %H:%M}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"sheet_{s // 20:02d}.png", dpi=55)
    plt.close(fig)
print("ok")
