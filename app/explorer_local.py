"""Local BIFoR explorer with frame inspector and QA labelling (needs the work drive). Run: streamlit run app/explorer_local.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import calib, config, indices, pipeline, reader, roi, timeseries  # noqa: E402

st.set_page_config(page_title="BIFoR canopy explorer", layout="wide")
cfg = config.load()
W = cfg.work_dir
D0 = pd.Timestamp("2026-01-01")


@st.cache_data
def load(name):
    p = W / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def frames():
    return pipeline.frames_view(cfg)


st.title("BIFoR-FACE canopy explorer")
st.caption("Two tower cameras (A5, A6), 24 bands 674–952 nm, Dec 2025 – Jun 2026. Values are relative reflectance "
           "(auto-exposure, no reflectance panel).")
tabs = st.tabs(["Season", "Compare cameras", "Data health", "Frame inspector", "Index catalog", "Simulations"])

daily, ph = load("daily.parquet"), load("phenology.parquet")

with tabs[0]:
    if daily.empty:
        st.info("No daily table yet. Run `python -m bifor.cli daily`.")
    else:
        c1, c2, c3 = st.columns(3)
        idx = c1.selectbox("Index", sorted(daily["index"].unique()), index=sorted(daily["index"].unique()).index("NDVI"))
        r = c2.selectbox("Region", sorted(daily.roi.unique()), index=sorted(daily.roi.unique()).index("canopy"))
        show_out = c3.checkbox("Show outlier days", False)
        d = daily[(daily["index"] == idx) & (daily.roi == r)]
        if not show_out:
            d = d[~d.outlier]
        wide = d.pivot_table(index="date", columns="camera", values="median")
        st.line_chart(wide)
        p = ph[(ph["index"] == idx) & (ph.roi == r)] if not ph.empty else pd.DataFrame()
        if len(p):
            p = p.copy()
            for c in ("sos20", "mid50", "mat80", "mid50_lo", "mid50_hi"):
                p[c] = (D0 + pd.to_timedelta(p[c], "D")).dt.strftime("%d %b")
            st.subheader("Green-up dates (spring logistic fit, 95% bootstrap CI)")
            st.dataframe(p[["camera", "sos20", "mid50", "mid50_lo", "mid50_hi", "mat80", "lo", "hi", "rmse", "n_days"]],
                         hide_index=True)
            st.caption("Data ends 3 Jun, before the canopy plateaus: the 80% date and the upper level are weakly constrained.")
        st.download_button("Download this series (CSV)", d.to_csv(index=False), f"{idx}_{r}.csv")

with tabs[1]:
    if not daily.empty:
        idx2 = st.selectbox("Index ", ["NDVI", "NDRE", "CIre", "REP-LE", "MTCI", "kNDVI"], key="cmp")
        d = daily[(daily["index"] == idx2) & (daily.roi == "canopy") & ~daily.outlier]
        w = d.pivot_table(index="date", columns="camera", values="median").dropna()
        if {"A5", "A6"} <= set(w.columns):
            w["A5 - A6"] = w.A5 - w.A6
            st.line_chart(w[["A5 - A6"]])
            st.write(f"Day-to-day correlation A5 vs A6: **{w.A5.corr(w.A6):.2f}** over {len(w)} days. "
                     "Shared swings come from weather and light; the difference removes them.")
        st.caption("Camera views differ (A5 oblique across the ring, A6 looking down), so a constant offset is "
                   "geometry, not treatment. Timing differences are the comparable quantity.")

with tabs[2]:
    f = frames()
    if f.empty:
        st.info("No processed frames yet.")
    else:
        n = f[f.min_from_noon.abs() <= cfg.noon_window_min]
        cam = st.radio("Camera", ["A5", "A6"], horizontal=True)
        g = n[n.camera == cam]
        per_day = g.groupby(g.ts_utc.dt.date).agg(near_noon=("qa_pass", "size"), passed=("qa_pass", "sum"))
        st.bar_chart(per_day)
        st.write(g.qa_reason.replace("", "pass").value_counts())
        sh = f[f.camera == cam].groupby(f.ts_utc.dt.date)[["day_shift_dy", "day_shift_dx"]].first()
        st.subheader("Camera offset from the 30 Mar reference (pixels)")
        st.line_chart(sh)

with tabs[3]:
    cat = load("catalog.parquet")
    if cat.empty:
        st.info("No catalog yet.")
    else:
        c1, c2 = st.columns(2)
        cam = c1.selectbox("Camera ", ["A5", "A6"])
        day = c2.date_input("Day", value=pd.Timestamp("2026-05-20").date())
        c = cat[(cat.camera == cam) & (cat.ts_utc.dt.date == day) & (cat.sun_elev > 5)]
        if c.empty:
            st.warning("No daylight frames that day.")
        else:
            fid = st.select_slider("Frame (UTC)", options=list(c.frame_id), value=c.loc[c.min_from_noon.abs().idxmin(), "frame_id"])
            r = c.set_index("frame_id").loc[fid]
            cal = calib.load(cfg.calib_dir, cam)
            dp, _ = pipeline.master_dark(W, cam, r.format)
            d = reader.decode(Path(r.read_path), cal, np.load(dp) if dp else None, cfg)
            R = indices.Spec(d.bands, cal.wl)
            rgb = np.dstack([R(860), R(716), R(674)])
            for k in range(3):
                lo, hi = np.nanpercentile(rgb[..., k], [2, 98])
                rgb[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo + 1e-9), 0, 1)
            im_idx = st.selectbox("Index map", ["NDVI", "NDRE", "CIre", "REP-LE"])
            v = indices.compute_one(im_idx, d.bands, cal.wl, cfg.band_tol)
            lo, hi = np.nanpercentile(v, [2, 98])
            import matplotlib.cm as cm
            vm = cm.RdYlGn(np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1))[..., :3]
            a, b = st.columns(2)
            a.image(np.nan_to_num(rgb, nan=1.0), caption=f"{fid} false colour 860/716/674 nm", width="stretch")
            b.image(vm, caption=f"{im_idx} ({lo:.2f} to {hi:.2f})", width="stretch")
            masks = roi.load(cfg.roi_dir, cam, cfg.versions["roi"])
            spec = roi.spectra(d.bands, masks)
            st.line_chart(pd.DataFrame({k: v["median"] for k, v in spec.items()}, index=np.round(cal.wl, 1)))
            fr = frames()
            row = fr[fr.frame_id == fid]
            if len(row):
                st.write(row[["qa_pass", "qa_reason", "sat_frac", "light_ratio", "light_jump", "shift_px",
                              "cloud_pct", "clearness", "precip_mm"]].T.astype(str))
            st.subheader("Label this frame (builds the QA check set)")
            lab = st.radio("Label", ["clear", "cloudy", "fog/haze", "rain/wet", "frost/snow", "camera moved", "other"],
                           horizontal=True)
            if st.button("Save label"):
                p = W / "qa_labels.csv"
                pd.DataFrame([{"frame_id": fid, "label": lab, "at": pd.Timestamp.now(tz="UTC")}]).to_csv(
                    p, mode="a", header=not p.exists(), index=False)
                st.success(f"Saved {fid} as {lab}")

with tabs[4]:
    cals = {c: calib.load(cfg.calib_dir, c) for c in ("A5", "A6")}
    t = indices.resolution_table(cals, cfg.band_tol)
    cit = W / "citations.csv"
    if cit.exists():
        t = t.merge(pd.read_csv(cit)[["index", "status", "doi"]], on="index", how="left")
    st.dataframe(t, hide_index=True, width="stretch")

with tabs[5]:
    figs = sorted((W / "figures" / "sim").glob("*.png"))
    if not figs:
        st.info("Run the simulations in simulations/.")
    for p in figs:
        st.image(str(p), caption=p.stem, width="stretch")
