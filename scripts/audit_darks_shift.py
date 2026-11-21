"""Dark-frame survey (darkest frame of every night) and pylon-only shift test."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import config, reader, qa, roi, calib, pipeline

cfg = config.load(); W = cfg.work_dir; OUT = W / "audit"
cat = pd.read_parquet(W / "catalog.parquet")
night = cat[cat.sun_elev < -12].copy()
night["night"] = (night.ts_utc - pd.Timedelta(hours=12)).dt.date
pick = night.loc[night.groupby(["camera", "night"]).sun_elev.idxmin()]
rows = []
for r in pick.itertuples():
    b, bit = reader.read_bands(Path(r.read_path))
    dn = reader.to_dn10(b, bit)
    rows.append({"camera": r.camera, "night": r.night, "frame_id": r.frame_id, "ts_utc": r.ts_utc, "sun_elev": r.sun_elev,
                 "format": r.format, "dark_mean": float(dn.mean()), "dark_p99": float(np.percentile(dn, 99)),
                 "band_spread": float(dn.reshape(25, -1).mean(1).std())})
d = pd.DataFrame(rows); d.to_csv(OUT / "darks.csv", index=False)
print(d.groupby(["camera", d.ts_utc.dt.strftime("%Y-%m")]).dark_mean.describe()[["count", "min", "50%", "max"]].round(1))
print(d.sort_values("dark_mean", ascending=False).head(8)[["frame_id", "sun_elev", "dark_mean", "dark_p99"]].round(1))

# shift: whole image vs pylon-only
m = pd.read_csv(OUT / "weekly_noon_sample.csv")
cat_i = cat.set_index("frame_id")
out = []
for cam in ("A5", "A6"):
    cal = calib.load(cfg.calib_dir, cam)
    ref = roi.reference(cfg.roi_dir, cam, "v1")
    excl = np.load(cfg.roi_dir / f"{cam}_v1_exclude.npy")
    nir_i = int(np.argmin(abs(cal.wl - 800)))
    for fid in m[m.frame_id.str.startswith(cam)].frame_id:
        dec = reader.decode(Path(cat_i.loc[fid, "read_path"]), cal, None, cfg)
        nir = dec.bands[nir_i]
        whole = qa.shift(ref, nir)
        st = qa.shift(np.where(excl, ref, np.nan), np.where(excl, nir, np.nan))
        out.append({"frame_id": fid, "whole_dy": whole[0], "whole_dx": whole[1], "pylon_dy": st[0], "pylon_dx": st[1]})
s = pd.DataFrame(out).round(2); s.to_csv(OUT / "shift_test.csv", index=False)
print(s.to_string(index=False))
