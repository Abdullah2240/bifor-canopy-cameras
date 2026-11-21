"""S7. Do the QA flags catch what they are meant to catch?

Real near-noon frames from the pilot days (read-only from D:) are altered:
- camera shift: rolled by 0-5 px; measured by pylon-only phase correlation;
- passing cloud: direct light removed by blending the frame toward a blurred,
  flatter copy (sunlit/shade contrast falls), inserted into a sequence of
  untouched neighbours; measured by light_jump.
Reports hit rate per size and false-alarm rate at the configured thresholds."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

from common import CAL, CFG, FIG, OUT
from bifor import pipeline, qa, reader, roi

cat = pd.read_parquet(CFG.work_dir / "catalog.parquet")
f = pipeline.frames_view(CFG)
ok = f[f.qa_pass & (f.min_from_noon.abs() < 60)].groupby("camera").head(30)
rows_s, rows_c = [], []
rng = np.random.default_rng(7)
for cam, g in ok.groupby("camera"):
    cal = CAL[cam]
    nir_i = int(np.argmin(np.abs(cal.wl - 800)))
    from scipy.ndimage import binary_dilation
    struct = binary_dilation(roi.exclusion(CFG.roi_dir, cam, "v1"), iterations=2)
    ref = roi.reference(CFG.roi_dir, cam, "v1")
    canopy = roi.load(CFG.roi_dir, cam, "v1")["canopy"]
    dark_p, _ = pipeline.master_dark(CFG.work_dir, cam, "pair")
    paths = cat.set_index("frame_id").loc[g.frame_id, ["read_path", "format"]]
    for fid, (p, fmt) in paths.iterrows():
        dp, _ = pipeline.master_dark(CFG.work_dir, cam, fmt)
        nir = reader.decode(Path(p), cal, np.load(dp), CFG).bands[nir_i]
        y0, x0 = qa.shift(ref, nir, ref_mask=struct)          # the frame's own offset (the 'day median' here)
        for s in (0, 1, 2, 3, 4, 5, 6):
            ang = rng.uniform(0, 2 * np.pi)
            dy, dx = int(round(s * np.sin(ang))), int(round(s * np.cos(ang)))
            moved = np.roll(nir, (dy, dx), axis=(0, 1))
            my, mx = qa.shift(ref, moved, ref_mask=struct)
            rows_s.append({"camera": cam, "true_px": float(np.hypot(dy, dx)),
                           "measured_px": float(np.hypot(my - y0, mx - x0)),
                           "err_px": float(np.hypot(my - y0 - dy, mx - x0 - dx))})
        base = qa.light_ratio(nir, canopy)
        flat = gaussian_filter(np.nan_to_num(nir, nan=np.nanmedian(nir)), 6)
        for a in (0.0, 0.2, 0.4, 0.6, 0.8):
            img = (1 - a) * nir + a * flat
            seq = [base * (1 + rng.normal(0, 0.01)) for _ in range(6)]
            seq.insert(3, qa.light_ratio(img, canopy))
            df = pd.DataFrame({"camera": cam, "ts_utc": pd.date_range("2026-05-01 10:00", periods=7, freq="10min", tz="UTC"),
                               "light_ratio": seq})
            rows_c.append({"camera": cam, "cloud_strength": a, "light_jump": float(qa.light_jump(df).iloc[3])})
S, C = pd.DataFrame(rows_s), pd.DataFrame(rows_c)
S.to_csv(OUT / "s7_shift.csv", index=False); C.to_csv(OUT / "s7_cloud.csv", index=False)
thr_s, thr_c = CFG.qa["max_shift_px"], CFG.qa["max_light_jump"]
S["flag"] = S.measured_px > thr_s
C["flag"] = C.light_jump > thr_c
print(S.groupby(S.true_px.round()).agg(flag_rate=("flag", "mean"), mean_err_px=("err_px", "mean"), exact=("err_px", lambda e: (e < 0.5).mean())).round(3))
print(C.groupby("cloud_strength").agg(flag_rate=("flag", "mean"), median_jump=("light_jump", "median")).round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
g = S.groupby(S.true_px.round()).flag.mean()
axes[0].bar(g.index.astype(str), g.values, color="#1f6f8b")
axes[0].set_xlabel("true shift (px)"); axes[0].set_ylabel(f"fraction flagged (> {thr_s} px)")
axes[0].set_title("Camera-shift flag (masked pylon registration)")
g = C.groupby("cloud_strength").flag.mean()
axes[1].bar(g.index.astype(str), g.values, color="#c0562f")
axes[1].set_xlabel("simulated loss of direct light (0 = none)"); axes[1].set_ylabel(f"fraction flagged (> {thr_c})")
axes[1].set_title("Light-jump flag for a single cloudy frame")
fig.suptitle("S7: QA flag detection on real pilot frames with injected faults")
fig.tight_layout()
fig.savefig(FIG / "s7_qa_detection.png", dpi=110)
plt.close(fig)
