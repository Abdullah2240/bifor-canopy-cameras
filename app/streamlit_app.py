"""BIFoR-FACE canopy cameras: season 2026 dashboard. Reads only app/data."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

import content as C  # noqa: E402

C = importlib.reload(C)  # pick up content changes on hot redeploys

LOCAL_DATA = Path(os.environ.get("BIFOR_APP_DATA", Path(__file__).resolve().parent / "data"))
DATA_REPO = "Abdullah2240/bifor-canopy-data"
D0 = pd.Timestamp("2026-01-01")
CAM_COL = {"A5": "#C0703A", "A6": "#2E7D7A"}
IDX_COL = {"NDVI": "#3F7D3A", "kNDVI": "#8DB06A", "NDRE": "#B0558A", "CIre": "#6A5ACD"}

st.set_page_config(page_title="BIFoR canopy cameras 2026", page_icon="🌳", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
.lede {font-size: 1.08rem; line-height: 1.6; opacity: .92; max-width: 70ch;}
.note {border-left: 3px solid #C0703A; padding: .55rem .9rem; margin: .6rem 0 1rem; font-size: .95rem;
       background: rgba(192,112,58,.08); border-radius: 0 6px 6px 0;}
.tenet {border: 1px solid rgba(128,128,128,.28); border-radius: 8px; padding: .8rem 1rem; margin-bottom: .7rem;}
.tenet b.k {font-family: ui-monospace, monospace; font-size: .85rem; opacity: .7; margin-right: .4rem;}
.conf {font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; padding: .1rem .45rem;
       border-radius: 99px; border: 1px solid currentColor; margin-left: .4rem; opacity: .8;}
.small {font-size: .88rem; opacity: .85;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data…")
def fetch_bundle(token: str) -> Path:
    """Download the private data bundle once per server process."""
    import io
    import tempfile
    import urllib.request
    import zipfile
    req = urllib.request.Request(f"https://api.github.com/repos/{DATA_REPO}/zipball/main",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    out = Path(tempfile.mkdtemp(prefix="bifor_"))
    z.extractall(out)
    root = next(out.iterdir())
    return root / "data" if (root / "data").exists() else root


def resolve_data() -> Path:
    if (LOCAL_DATA / "summary.json").exists():
        return LOCAL_DATA
    try:
        token = st.secrets.get("BIFOR_DATA_TOKEN")
    except Exception:
        token = None
    if not token:
        st.error("The data for this dashboard is not configured. Add BIFOR_DATA_TOKEN to the app's secrets.")
        st.stop()
    try:
        return fetch_bundle(token)
    except Exception as e:
        st.error(f"Could not load the data bundle ({type(e).__name__}).")
        st.stop()


DATA = resolve_data()


@st.cache_data
def load():
    s = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(DATA / "daily.parquet")
    ph = pd.read_parquet(DATA / "phenology.parquet")
    lane = pd.read_csv(DATA / "lane_a_prime.csv")
    qa = pd.read_csv(DATA / "qa_daily.csv", parse_dates=["date"])
    cat = pd.read_csv(DATA / "index_catalog.csv")
    fr = pd.read_csv(DATA / "frames" / "frames.csv") if (DATA / "frames" / "frames.csv").exists() else pd.DataFrame()
    return s, daily, ph, lane, qa, cat, fr


S, daily, ph, lane, qa, cat, frames = load()
T = S.get("treatment", {})


def lab(cam):
    return f"{cam} (assumed {T[cam]})" if cam in T else cam


dt = C.dt


def logistic(t, lo, hi, t0, k):
    return lo + (hi - lo) / (1 + np.exp(-k * (t - t0)))


@st.cache_data(show_spinner=False)
def _b64(path: str) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode()


def show_img(target, path, caption=""):
    """Inline the image as data so it does not depend on the server's media endpoint."""
    p = Path(path)
    if not p.exists():
        return
    cap = f"<div class='small' style='margin-top:.3rem'>{caption}</div>" if caption else ""
    target.markdown(f"<img src='data:image/png;base64,{_b64(str(p))}' alt='{caption}' "
                    f"style='width:100%;height:auto;border-radius:6px;display:block'>{cap}", unsafe_allow_html=True)


def figure(path, caption):
    p = DATA / "figures" / f"{path}.png"
    if p.exists():
        with st.container(border=True):
            show_img(st, p, caption)


def chart_layout(fig, y_title, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
                      yaxis_title=y_title, hovermode="x unified")
    return fig


W = S["window"]
gu = pd.DataFrame(S.get("greenup", []))
lag = S.get("lag", {})
NOISE = S.get("noise", {})
LA = S.get("lane_a_prime", {})
POWER = pd.DataFrame(S.get("power", []))

PAGES = ["Overview", "Assumptions", "Methods", "Season explorer", "Leaves before chlorophyll",
         "eCO2 comparison", "What the cameras see", "Simulations", "Data quality", "Index catalog",
         "Open questions", "Research roadmap"]
with st.sidebar:
    st.markdown("### BIFoR canopy cameras")
    st.caption(f"Season 2026 · {dt(W['first_day'])} – {dt(W['last_day'])}")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Relative reflectance only. Treatment labels are a working assumption; see Assumptions.")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Spring green-up of the BIFoR-FACE oak canopy, seen by two tower hyperspectral cameras")
    st.markdown(
        f"<p class='lede'>Two snapshot hyperspectral cameras (A5 and A6) on the BIFoR-FACE towers record the oak "
        f"canopy in 24 bands from 674 to 952 nm, roughly every 11 minutes. This dashboard covers the valid record, "
        f"{dt(W['first_day'])} to {dt(W['last_day'])} 2026, and reports what the imagery alone can show: when the "
        f"canopy greens up, how chlorophyll follows leaf expansion, how the two cameras compare, and what this sensor "
        f"can and cannot measure.</p>", unsafe_allow_html=True)
    nd = gu[gu["index"] == "NDVI"].set_index("camera") if len(gu) else pd.DataFrame()
    c = st.columns(4)
    c[0].metric("Days analysed", f"{W['days'].get('A5', 0)} / {W['days'].get('A6', 0)}", help="A5 / A6, daily values after QA")
    c[1].metric("Frames passed QA", f"{W['frames_passed']:,}", help=f"of {W['frames_near_noon']:,} frames within 2 h of solar noon")
    if len(nd):
        c[2].metric("NDVI half-way (direct)", f"{dt(nd.loc['A5', 'half_direct'])} / {dt(nd.loc['A6', 'half_direct'])}",
                    help="A5 / A6: first day after which the 7-day median stays above half of the observed rise")
    if lag:
        c[3].metric("Chlorophyll lag", " / ".join(f"{v['lag_days']} d" for v in lag.values()),
                    help="NDRE half-way date minus NDVI half-way date, A5 / A6")
    st.subheader("Findings at a glance")
    lf = S.get("spectral_leafed_from", {})
    items = []
    if len(nd):
        items.append(f"**Green-up timing.** NDVI reaches half of its spring rise on {dt(nd.loc['A5', 'half_direct'])} "
                     f"(A5) and {dt(nd.loc['A6', 'half_direct'])} (A6), measured directly from the daily series.")
    if lag:
        items.append("**Leaves come before chlorophyll.** The red-edge chlorophyll index NDRE reaches its half-way "
                     "point " + " and ".join(f"{v['lag_days']} days ({k})" for k, v in lag.items()) +
                     " after NDVI. The red-edge indices are still rising when the record ends, so this lag is a minimum.")
    if lf:
        items.append("**Leaf-out without an index.** Clustering the full spectrum by shape alone places the switch to "
                     "a leafed canopy on " + " and ".join(f"{dt(v)} ({k})" for k, v in lf.items()) + ".")
    if LA:
        items.append(C.camera_glance(S).replace("<b>", "**").replace("</b>", "**"))
    items.append("**A quality-control method for this camera type**, and simulations showing which indices track "
                 "chlorophyll, which track leaf amount, and what is out of reach (leaf water).")
    for i in items:
        st.markdown(f"- {i}")
    if len(frames):
        st.subheader("The canopy through the season")
        st.caption("False colour (860 / 716 / 674 nm) near solar noon. Full gallery under 'What the cameras see'.")
        f6 = frames[frames.camera == "A6"]
        cols = st.columns(len(f6) or 1)
        for col, r in zip(cols, f6.itertuples()):
            show_img(col, DATA / "frames" / f"A6_{r.date}_fc.png", dt(r.date))

# ---------------------------------------------------------------- Assumptions
elif page == "Assumptions":
    st.title("Assumptions and working tenets")
    st.markdown("<p class='lede'>Every result here rests on the tenets below. Each one states its basis and how "
                "confident we are in it. If any of them changes, the analysis can be rerun with the new value.</p>",
                unsafe_allow_html=True)
    TEN = C.tenets(S)
    for k, title, conf, body in TEN:
        st.markdown(f"<div class='tenet'><b class='k'>{k}</b><b>{title}</b><span class='conf'>{conf}</span>"
                    f"<div class='small' style='margin-top:.35rem'>{body}</div></div>", unsafe_allow_html=True)
    st.subheader("References")
    st.markdown("\n".join(f"- {t} [doi:{u.split('doi.org/')[1]}]({u})" for t, u in C.REFERENCES),
                unsafe_allow_html=True)

# ---------------------------------------------------------------- Methods
elif page == "Methods":
    st.title("Methods")
    for head, body in C.methods(S):
        st.markdown(f"**{head}.** {body}")

# ---------------------------------------------------------------- Season explorer
elif page == "Season explorer":
    st.title("Season explorer")
    c1, c2, c3 = st.columns(3)
    idxs = sorted(daily["index"].unique())
    rois = sorted(daily.roi.unique())
    idx = c1.selectbox("Index", idxs, index=idxs.index("NDVI") if "NDVI" in idxs else 0)
    roi = c2.selectbox("Region", rois, index=rois.index("canopy") if "canopy" in rois else 0)
    show_fit = c3.toggle("Show fitted curve", True)
    d = daily[(daily["index"] == idx) & (daily.roi == roi) & ~daily.outlier]
    fig = go.Figure()
    for cam in ("A5", "A6"):
        g = d[d.camera == cam].sort_values("date")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=pd.concat([g.date, g.date[::-1]]), y=pd.concat([g.q75, g.q25[::-1]]),
                                 fill="toself", fillcolor=CAM_COL[cam], opacity=0.15, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=g.date, y=g["median"], mode="markers", name=lab(cam),
                                 marker=dict(color=CAM_COL[cam], size=5)))
        p = ph[(ph["index"] == idx) & (ph.roi == roi) & (ph.camera == cam) & (ph.error == "")]
        if show_fit and len(p):
            p = p.iloc[0]
            t = np.arange((g.date.min() - D0).days, (g.date.max() - D0).days + 1)
            fig.add_trace(go.Scatter(x=D0 + pd.to_timedelta(t, "D"), y=logistic(t, p.lo, p.hi, p.t0, p.k),
                                     mode="lines", name=f"{cam} fit", line=dict(color=CAM_COL[cam], width=2)))
    st.plotly_chart(chart_layout(fig, idx), width="stretch", theme="streamlit")
    st.caption("Points: daily median of quality-passed frames near solar noon. Shading: interquartile range of frames "
               "within the day.")
    p = ph[(ph["index"] == idx) & (ph.roi == roi) & (ph.error == "")].copy()
    if len(p):
        for col in ("sos20", "mid50", "mid50_lo", "mid50_hi", "mat80"):
            p[col] = (D0 + pd.to_timedelta(p[col], "D")).dt.strftime("%d %b")
        p["camera"] = p.camera.map(lab)
        st.markdown("**Fitted green-up dates** (20%, 50% with 95% CI, 80%)")
        st.dataframe(p[["camera", "sos20", "mid50", "mid50_lo", "mid50_hi", "mat80", "reliable", "n_days"]],
                     hide_index=True, width="stretch")
        st.caption("'reliable' means the curve rises, its 50% date and interval end at least 3 days before the record, "
                   "and the interval is 30 days or narrower. The record ends before the canopy plateaus, so the 80% "
                   "date is weakly constrained.")
    st.download_button("Download this series (CSV)", d.to_csv(index=False), f"{idx}_{roi}.csv".replace(" ", "_").replace("/", "-"))

# ---------------------------------------------------------------- Lag
elif page == "Leaves before chlorophyll":
    st.title("Leaves come before chlorophyll")
    st.markdown("<p class='lede'>NDVI responds mostly to the amount of leaf; the red-edge indices NDRE and CIre respond "
                "mostly to leaf chlorophyll (simulation S1). Scaling each index from its leafless baseline (0) to the "
                "end of the record (1) shows when each reaches half of its rise.</p>", unsafe_allow_html=True)
    cam = st.radio("Camera", ["A5", "A6"], horizontal=True, index=1, format_func=lab)
    fig = go.Figure()
    for n in ("NDVI", "kNDVI", "NDRE", "CIre"):
        g = daily[(daily["index"] == n) & (daily.roi == "canopy") & (daily.camera == cam) & ~daily.outlier]
        s = g.set_index("date")["median"].sort_index()
        if s.empty:
            continue
        base = s[s.index < W["leafless_end"]].median()
        top = s.iloc[-7:].median()
        z = ((s - base) / (top - base)).rolling(7, center=True, min_periods=3).median()
        fig.add_trace(go.Scatter(x=z.index, y=z.values, name=n, mode="lines", line=dict(color=IDX_COL[n], width=2.5)))
    fig.add_hline(y=0.5, line_dash="dot", opacity=0.6)
    fig.update_xaxes(range=["2026-04-01", W["last_day"]])
    st.plotly_chart(chart_layout(fig, "scaled rise (0 = leafless, 1 = end of record)"), width="stretch", theme="streamlit")
    if lag:
        t = pd.DataFrame([{"camera": lab(k), "NDVI half-way": dt(v["ndvi"]), "NDRE half-way": dt(v["ndre"]),
                           "CIre half-way": dt(v.get("cire")), "lag (NDRE − NDVI)": f"{v['lag_days']} days"}
                          for k, v in lag.items()])
        st.dataframe(t, hide_index=True, width="stretch")
    st.markdown("<div class='note'>The red-edge indices are still rising on the last day, so the true lag is at least "
                "this long. Fitted mid-points for the red-edge indices fall at the end of the record and are not used "
                "here.</div>", unsafe_allow_html=True)
    lay = pd.DataFrame(S.get("layers", []))
    if len(lay):
        st.subheader("Canopy layers")
        lay["half_direct"] = lay.half_direct.map(dt)
        st.dataframe(lay.pivot(index="roi", columns="camera", values="half_direct"), width="stretch")
        st.caption("NDVI half-way date by layer. Differences of a few days are within the day-to-day noise; no clear "
                   "vertical sequence is resolved.")

# ---------------------------------------------------------------- eCO2
elif page == "eCO2 comparison":
    st.title("Camera comparison under the elevated-CO2 assumption")
    st.markdown(f"<div class='note'><b>Read with Assumption T1.</b> We do not know which ring each camera views. "
                f"Following the published finding that elevated CO2 advances leaf-out by about 3 days (Gao et al., 2026), "
                f"A6 is labelled elevated CO2 and A5 ambient. Because that label was chosen after seeing which camera "
                f"finishes greening first, the difference below is <b>not evidence</b> of a CO2 effect. It shows whether the "
                f"observed difference is the size that published effect would produce. The viewing angles also differ "
                f"(A5 oblique, A6 from above), and angle alone can move green-up dates.</div>", unsafe_allow_html=True)
    if LA:
        c = st.columns(3)
        c[0].metric("A5 − A6, direct half-way", f"{LA.get('direct_shift', 0):+.0f} d", help="positive: A6 earlier")
        c[1].metric("A5 − A6, direct 80%", f"{LA.get('direct80_shift', 0):+d} d", help="positive: A6 earlier")
        c[2].metric("A5 − A6, fitted half-way", f"{LA['shift']:+.1f} d",
                    help=f"95% CI {LA['lo']:+.1f} to {LA['hi']:+.1f} (joint bootstrap)")
        st.markdown(C.camera_text(S), unsafe_allow_html=True)
    st.subheader("All indices and regions")
    t = lane.copy()
    t["fitted A5 − A6 (d)"] = t["shift"].round(1)
    t["95% CI"] = t.apply(lambda r: f"{r.lo:+.1f} to {r.hi:+.1f}", axis=1)
    t["direct A5 − A6 (d)"] = t.direct_shift
    t["A5–A6 correlation"] = t.rho.round(2)
    st.dataframe(t[["roi", "index", "fitted A5 − A6 (d)", "95% CI", "direct A5 − A6 (d)", "A5–A6 correlation",
                    "n_shared_days"]], hide_index=True, width="stretch")
    st.caption("Positive values mean A6 reaches half-way earlier. Red-edge indices are still rising at the end of the "
               "record, so their fitted dates are less reliable.")
    st.subheader("Could this design detect a 3-day shift?")
    if len(POWER):
        st.dataframe(POWER.rename(columns={"day_sd": "day-to-day SD", "rho": "shared noise (ρ)",
                                           "min_detectable_shift_days": "shift detected with 80% power (days)"}),
                     hide_index=True)
    figure("s4_power_curves", "S4: probability of detecting a green-up shift between two views, by size of shift.")
    st.caption("The simulation assumes 80% shared noise; the measured value is higher, so these detectable shifts are "
               "conservative. "
               f"Measured on this record: NDVI correlation between cameras {NOISE.get('rho_all', float('nan')):.2f} "
               f"overall and {NOISE.get('rho_leafless', float('nan')):.2f} in the leafless period; day-to-day SD "
               f"{NOISE.get('day_sd_leafless', float('nan')):.3f}.")

# ---------------------------------------------------------------- Frames
elif page == "What the cameras see":
    st.title("What the cameras see")
    st.caption("Frames closest to solar noon that passed QA. Left: false colour (860 / 716 / 674 nm). Right: NDVI "
               "(red low, green high).")
    figure("roi_overlay", "Canopy regions used for the daily values.")
    for cam in ("A5", "A6"):
        st.subheader(lab(cam))
        g = frames[frames.camera == cam]
        for r in g.itertuples():
            c1, c2 = st.columns(2)
            show_img(c1, DATA / "frames" / f"{cam}_{r.date}_fc.png", f"{dt(r.date)}, {r.time_utc} UTC")
            show_img(c2, DATA / "frames" / f"{cam}_{r.date}_ndvi.png", "NDVI")

# ---------------------------------------------------------------- Simulations
elif page == "Simulations":
    st.title("Simulations: what this sensor can measure")
    SIMS = C.SIMS
    for f, title, text in SIMS:
        st.subheader(title)
        st.markdown(text)
        figure(f, title)

# ---------------------------------------------------------------- QA
elif page == "Data quality":
    st.title("Data quality")
    st.markdown(f"{W['frames_passed']:,} of {W['frames_near_noon']:,} frames within two hours of solar noon passed "
                f"every check ({W['frames_passed'] / max(W['frames_near_noon'], 1):.0%}).")
    cam = st.radio("Camera", ["A5", "A6"], horizontal=True)
    q = qa[qa.camera == cam].pivot_table(index="date", columns="reason", values="n", aggfunc="sum").fillna(0)
    fig = go.Figure()
    for col in q.columns:
        fig.add_trace(go.Bar(x=q.index, y=q[col], name=col))
    fig.update_layout(barmode="stack")
    st.plotly_chart(chart_layout(fig, "frames near noon", 360), width="stretch", theme="streamlit")
    st.subheader("Problems found and corrected")
    st.dataframe(pd.DataFrame(C.QA_FIXES, columns=["Problem", "Effect if missed", "Correction"]),
                 hide_index=True, width="stretch")
    if S.get("clear_r") is not None:
        st.markdown(f"**Illumination check.** In the leafless period ({S['noise']['n_leafless_days']} days), daily NDVI "
                    f"reads lower on clear days (r = {S['clear_r']:+.2f} against daily clearness). This is bark and "
                    "shadow under direct sun, and is why daily values use the median of many frames.")
    st.caption("The QA thresholds were set using 200 visually labelled frames. Those labels have not yet been "
               "independently confirmed.")

# ---------------------------------------------------------------- Catalog
elif page == "Index catalog":
    st.title("Index catalog")
    st.caption("Each index resolved to the cameras' bands. Citation status from a DOI check.")
    st.dataframe(cat, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Open questions
elif page == "Open questions":
    st.title("Open questions and curiosities")
    st.markdown("<p class='lede'>These are not conclusions. They are the questions this record raises, and the next "
                "steps each would need.</p>", unsafe_allow_html=True)
    Q = C.QUESTIONS
    for qn, body in Q:
        st.markdown(f"<div class='tenet'><b>{qn}</b><div class='small' style='margin-top:.35rem'>{body}</div></div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- Roadmap
elif page == "Research roadmap":
    st.title("Research roadmap: from this record to publication")
    st.markdown("<p class='lede'>What this record already supports, and what each next step would need. Routes are "
                "ordered by how little new data they require. Each states what exists now, the one thing that unlocks "
                "it, and the expected output; none promises a result.</p>", unsafe_allow_html=True)
    for n, title, status, q, now, needs, out in C.ROADMAP:
        st.markdown(f"<div class='tenet'><b class='k'>Route {n}</b><b>{title}</b><span class='conf'>{status}</span>"
                    f"<div class='small' style='margin-top:.45rem'><b>Question.</b> {q}</div>"
                    f"<div class='small'><b>In place now.</b> {now}</div>"
                    f"<div class='small'><b>Unlocked by.</b> {needs}</div>"
                    f"<div class='small'><b>Output.</b> {out}</div></div>", unsafe_allow_html=True)
    st.subheader("What would move this forward")
    NEEDS = ["**Ring and treatment for each camera** (route 2)",
             "**Imagery from June onward**, with the reference panel in view (route 3)",
             "**Site measurements for the same period**: gas exchange, leaf nitrogen and chlorophyll, canopy "
             "temperature, and ground budburst dates (route 4, and validation for all routes)",
             "**Summer field surveys of powdery mildew** (route 5)"]
    st.markdown(chr(10).join(f"- {x}" for x in NEEDS))
