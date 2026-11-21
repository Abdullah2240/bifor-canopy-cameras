"""Standalone HTML report for the site lead, built from the dashboard bundle (app/data).

Run after scripts/export_app_data.py. Output: docs/BIFoR_canopy_cameras_2026.html
(self-contained, works offline, light and dark mode).
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import content as C  # noqa: E402

DATA = ROOT / "app" / "data"
OUT = ROOT / "docs" / "BIFoR_canopy_cameras_2026.html"
D0 = pd.Timestamp("2026-01-01")
CAM_COL = {"A5": "#C0703A", "A6": "#2E7D7A"}
IDX_COL = {"NDVI": "#3F7D3A", "kNDVI": "#8DB06A", "NDRE": "#B0558A", "CIre": "#6A5ACD"}

S = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
daily = pd.read_parquet(DATA / "daily.parquet")
ph = pd.read_parquet(DATA / "phenology.parquet")
lane = pd.read_csv(DATA / "lane_a_prime.csv")
frames = pd.read_csv(DATA / "frames" / "frames.csv")
W, T, LA, NOISE, lag = S["window"], S["treatment"], S["lane_a_prime"], S["noise"], S["lag"]
gu = pd.DataFrame(S["greenup"])
dt = C.dt
first_plot = [True]


def lab(cam):
    return f"{cam} (assumed {T[cam]})" if cam in T else cam


def b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def img(name, caption, sub="figures"):
    p = DATA / sub / f"{name}.png"
    return (f'<figure><img src="{b64(p)}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
            if p.exists() else "")


def plot(fig, y_title, height=400) -> str:
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=30, b=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8a8f87", family="Inter, system-ui, sans-serif"),
                      legend=dict(orientation="h", y=1.1), yaxis_title=y_title, hovermode="x unified")
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    out = pio.to_html(fig, full_html=False, include_plotlyjs=first_plot[0], config={"displaylogo": False})
    first_plot[0] = False
    return f'<div class="chart">{out}</div>'


def logistic(t, lo, hi, t0, k):
    return lo + (hi - lo) / (1 + np.exp(-k * (t - t0)))


# ---------- charts ----------
fig = go.Figure()
for cam in ("A5", "A6"):
    g = daily[(daily["index"] == "NDVI") & (daily.roi == "canopy") & (daily.camera == cam) & ~daily.outlier].sort_values("date")
    fig.add_trace(go.Scatter(x=g.date, y=g["median"], mode="markers", name=lab(cam), marker=dict(color=CAM_COL[cam], size=5)))
    p = ph[(ph["index"] == "NDVI") & (ph.roi == "canopy") & (ph.camera == cam) & (ph.error == "")]
    if len(p):
        p = p.iloc[0]
        t = np.arange((g.date.min() - D0).days, (g.date.max() - D0).days + 1)
        fig.add_trace(go.Scatter(x=D0 + pd.to_timedelta(t, "D"), y=logistic(t, p.lo, p.hi, p.t0, p.k), mode="lines",
                                 name=f"{cam} fit", line=dict(color=CAM_COL[cam], width=2)))
season_chart = plot(fig, "canopy NDVI")

lag_charts = []
for cam in ("A5", "A6"):
    fig = go.Figure()
    for n in ("NDVI", "kNDVI", "NDRE", "CIre"):
        s = daily[(daily["index"] == n) & (daily.roi == "canopy") & (daily.camera == cam) & ~daily.outlier].set_index("date")["median"].sort_index()
        if s.empty:
            continue
        base, top = s[s.index < W["leafless_end"]].median(), s.iloc[-7:].median()
        z = ((s - base) / (top - base)).rolling(7, center=True, min_periods=3).median()
        fig.add_trace(go.Scatter(x=z.index, y=z.values, name=n, mode="lines", line=dict(color=IDX_COL[n], width=2.5)))
    fig.add_hline(y=0.5, line_dash="dot", opacity=0.6)
    fig.update_xaxes(range=["2026-04-01", W["last_day"]])
    fig.update_layout(title=dict(text=lab(cam), x=0.01, font=dict(size=13)))
    lag_charts.append(plot(fig, "scaled rise", 340))


def table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


nd = gu[gu["index"] == "NDVI"].set_index("camera")
lf = S.get("spectral_leafed_from", {})
gu_t = gu.copy()
gu_t["camera"] = gu_t.camera.map(lab)
for c in ("half_direct", "sos20", "mid50", "mid50_lo", "mid50_hi"):
    if c in gu_t:
        gu_t[c] = gu_t[c].map(dt)
gu_t = gu_t.rename(columns={"half_direct": "half-way (direct)", "sos20": "20% (fit)", "mid50": "50% (fit)",
                            "mid50_lo": "50% CI low", "mid50_hi": "50% CI high"})
lag_t = pd.DataFrame([{"camera": lab(k), "NDVI half-way": dt(v["ndvi"]), "NDRE half-way": dt(v["ndre"]),
                       "CIre half-way": dt(v.get("cire")), "lag (NDRE − NDVI)": f"{v['lag_days']} days"} for k, v in lag.items()])
lay = pd.DataFrame(S.get("layers", []))
lay_t = lay.assign(half_direct=lay.half_direct.map(dt)).pivot(index="roi", columns="camera", values="half_direct").reset_index() if len(lay) else pd.DataFrame()
ln = lane.copy()
ln["fitted A5 − A6 (d)"] = ln["shift"].round(1)
ln["95% CI"] = ln.apply(lambda r: f"{r.lo:+.1f} to {r.hi:+.1f}", axis=1)
ln = ln.rename(columns={"direct_shift": "direct A5 − A6 (d)", "rho": "correlation", "n_shared_days": "shared days"})
ln["correlation"] = ln.correlation.round(2)
ln_t = ln[["roi", "index", "fitted A5 − A6 (d)", "95% CI", "direct A5 − A6 (d)", "correlation", "shared days"]]
pw = pd.DataFrame(S["power"]).rename(columns={"day_sd": "day-to-day SD", "rho": "shared noise (ρ)",
                                             "min_detectable_shift_days": "shift detected with 80% power (days)"})

# ---------- sections ----------
illum = (f"<p><b>Illumination check.</b> In the leafless period ({NOISE['n_leafless_days']} days), daily NDVI reads lower on "
         f"clear days (r = {S['clear_r']:+.2f} against daily clearness): bark and shadow under direct sun, which is why "
         f"daily values use the median of many frames.</p>") if S.get('clear_r') is not None else ''
glance = []
glance.append(f"<b>Green-up timing.</b> NDVI reaches half of its spring rise on {dt(nd.loc['A5', 'half_direct'])} (A5) "
              f"and {dt(nd.loc['A6', 'half_direct'])} (A6), measured directly from the daily series.")
glance.append("<b>Leaves come before chlorophyll.</b> NDRE reaches its half-way point " +
              " and ".join(f"{v['lag_days']} days ({k})" for k, v in lag.items()) +
              " after NDVI; the red-edge indices are still rising when the record ends, so this is a minimum.")
if lf:
    glance.append("<b>Leaf-out without an index.</b> Clustering the full spectrum by shape alone places the switch to a "
                  "leafed canopy on " + " and ".join(f"{dt(v)} ({k})" for k, v in lf.items()) + ".")
glance.append(C.camera_glance(S))
glance.append("<b>Method.</b> A quality-control protocol for this camera type, and simulations showing which indices "
              "track chlorophyll, which track leaf amount, and what is out of reach.")

tenets = "".join(f"<div class='card'><div class='ch'><span class='k'>{k}</span><b>{t}</b><span class='conf'>{c}</span>"
                 f"</div><div>{b}</div></div>" for k, t, c, b in C.tenets(S))
methods = "".join(f"<p><b>{h}.</b> {b}</p>" for h, b in C.methods(S))
sims = "".join(f"<div class='sim'><h3>{t}</h3><p>{x}</p>{img(f, t)}</div>" for f, t, x in C.SIMS)
qa_t = table(pd.DataFrame(C.QA_FIXES, columns=["Problem", "Effect if missed", "Correction"]))
qs = "".join(f"<div class='card'><b>{q}</b><div>{b}</div></div>" for q, b in C.QUESTIONS)
refs = "".join(f"<li>{t} <a href='{u}'>{u.replace('https://', '')}</a></li>" for t, u in C.REFERENCES)
strip = "".join(f"<figure class='fr'><img src='{b64(DATA / 'frames' / f'{r.camera}_{r.date}_fc.png')}' "
                f"alt='{r.camera} {r.date}'><figcaption>{r.camera} · {dt(r.date)}</figcaption></figure>"
                for r in frames.itertuples())

CSS = """
:root{--bg:#F7F5EF;--panel:#FFFFFF;--ink:#1E2A20;--muted:#5E665C;--line:#DCD7CA;--accent:#2E6B3A;--warn:#C0703A}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121613;--panel:#1A201B;--ink:#E4E7DF;--muted:#A3AA9F;--line:#2D352E;--accent:#8CC08E;--warn:#E09A64}}
:root[data-theme=dark]{--bg:#121613;--panel:#1A201B;--ink:#E4E7DF;--muted:#A3AA9F;--line:#2D352E;--accent:#8CC08E;--warn:#E09A64}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:48px 16px 80px}
h1,h2,h3{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.01em;line-height:1.25}
h1{font-size:2.1rem;margin:.2rem 0 1rem}h2{font-size:1.5rem;margin:3rem 0 .8rem;padding-top:1rem;border-top:1px solid var(--line)}
h3{font-size:1.1rem;margin:1.6rem 0 .4rem}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;color:var(--muted)}
.lede{font-size:1.1rem;color:var(--muted);max-width:68ch}
nav.toc{display:flex;flex-wrap:wrap;gap:.4rem 1rem;font-size:.9rem;margin:1.5rem 0}
a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:1.5rem 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-family:Georgia,serif;font-size:1.5rem}.kpi .l{font-size:.8rem;color:var(--muted)}
ul.glance li{margin:.45rem 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0;font-size:.95rem}
.ch{margin-bottom:.35rem}.k{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--muted);margin-right:.5rem}
.conf{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;border:1px solid var(--line);border-radius:99px;padding:1px 8px;margin-left:.5rem;color:var(--muted)}
.note{border-left:3px solid var(--warn);background:color-mix(in srgb,var(--warn) 9%,transparent);padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0;font-size:.95rem}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:6px;margin:12px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
figure{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}
figure img{width:100%;height:auto;display:block}figcaption{font-size:.82rem;color:#5E665C;margin-top:6px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.fr{background:var(--panel);padding:4px}.fr figcaption{color:var(--muted)}
.tblw{overflow-x:auto}table.tbl{border-collapse:collapse;width:100%;font-size:.88rem;margin:10px 0;background:var(--panel)}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
.tbl th{font-weight:600;color:var(--muted)}
.small{font-size:.88rem;color:var(--muted)}
.themebtn{position:fixed;top:12px;right:12px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:99px;padding:4px 12px;font-size:.8rem;cursor:pointer}
footer{margin-top:4rem;font-size:.85rem;color:var(--muted)}
"""

body = f"""
<button class="themebtn" onclick="var r=document.documentElement;r.dataset.theme=(r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches))?'light':'dark'">Light / dark</button>
<main>
<div class="eyebrow">BIFoR-FACE · tower hyperspectral cameras · season 2026</div>
<h1>Spring green-up of the BIFoR-FACE oak canopy, seen by two tower hyperspectral cameras</h1>
<p class="lede">Two snapshot hyperspectral cameras (A5 and A6) record the oak canopy in 24 bands from 674 to 952 nm,
roughly every 11 minutes. This report covers the valid record, {dt(W['first_day'])} to {dt(W['last_day'])} 2026, and
sets out what the imagery alone can show, how it was obtained, and which questions it opens.</p>
<nav class="toc"><a href="#glance">1 At a glance</a><a href="#assumptions">2 Assumptions</a><a href="#methods">3 Methods</a>
<a href="#greenup">4 Green-up</a><a href="#eco2">5 eCO2 comparison</a><a href="#sims">6 Simulations</a>
<a href="#qa">7 Data quality</a><a href="#questions">8 Open questions</a><a href="#refs">References</a></nav>

<div class="kpis">
<div class="kpi"><div class="v">{W['days'].get('A5', 0)} / {W['days'].get('A6', 0)}</div><div class="l">days analysed, A5 / A6</div></div>
<div class="kpi"><div class="v">{W['frames_passed']:,}</div><div class="l">frames passed QA, of {W['frames_near_noon']:,} near noon</div></div>
<div class="kpi"><div class="v">{dt(nd.loc['A5', 'half_direct'])} / {dt(nd.loc['A6', 'half_direct'])}</div><div class="l">NDVI half-way, A5 / A6</div></div>
<div class="kpi"><div class="v">{' / '.join(f"{v['lag_days']} d" for v in lag.values())}</div><div class="l">chlorophyll lag after NDVI, A5 / A6</div></div>
</div>
<div class="strip">{strip}</div>
<p class="small">False colour (860 / 716 / 674 nm), frames closest to solar noon that passed QA.</p>

<h2 id="glance">1 At a glance</h2>
<ul class="glance">{''.join(f'<li>{g}</li>' for g in glance)}</ul>

<h2 id="assumptions">2 Assumptions and working tenets</h2>
<p>Every result rests on the tenets below. Each states its basis and confidence; any of them can be revised and the
analysis rerun.</p>
{tenets}

<h2 id="methods">3 Methods</h2>
{methods}
{img('roi_overlay', 'Canopy regions used for the daily values.')}

<h2 id="greenup">4 Green-up timing</h2>
<h3>Season curve</h3>
{season_chart}
<p class="small">Daily median of quality-passed frames near solar noon, with the fitted spring logistic curve.</p>
<div class="tblw">{table(gu_t)}</div>
<p class="small">'reliable' marks fits whose 50% date and interval end at least 3 days before the record does and whose
interval is 30 days or narrower. The direct half-way date is preferred where they disagree.</p>
<h3>Leaves come before chlorophyll</h3>
<p>NDVI responds mostly to leaf amount; the red-edge indices NDRE and CIre respond mostly to leaf chlorophyll
(Simulation S1). Each index is scaled from its leafless baseline (0) to the end of the record (1).</p>
<div class="grid2">{''.join(lag_charts)}</div>
<div class="tblw">{table(lag_t)}</div>
<div class="note">The red-edge indices are still rising on the last day, so the true lag is at least this long.</div>
<h3>Leaf-out from spectral shape alone</h3>
{img('p5_spectral_stages', 'Daily spectral stage from principal components and k-means (left) and the mean spectrum of each stage (right).')}
<h3>Canopy layers</h3>
<div class="tblw">{table(lay_t) if len(lay_t) else ''}</div>
<p class="small">NDVI half-way date by layer. Differences of a few days are within the day-to-day noise; no clear vertical
sequence is resolved.</p>

<h2 id="eco2">5 Camera comparison under the elevated-CO2 assumption</h2>
<div class="note"><b>Read with Assumption T1.</b> Which ring each camera views is not known. Following the published finding
that elevated CO2 advances leaf-out by about 3 days (Gao et al., 2026), A6 is labelled elevated CO2 and A5 ambient.
Because that label was chosen after seeing which camera finishes greening first, the difference below is <b>not evidence</b> of a
CO2 effect; it shows whether the observed difference is the size that published effect would produce. The viewing
angles also differ (A5 oblique, A6 from above), and angle alone can move green-up dates.</div>
<p>{C.camera_text(S)}</p>
<div class="tblw">{table(ln_t)}</div>
<h3>Could this design detect a 3-day shift?</h3>
<div class="tblw">{table(pw)}</div>
<p class="small">The simulation assumes 80% shared noise; the measured value is higher, so these detectable shifts are conservative. Measured on this record: NDVI correlation between cameras {NOISE['rho_all']:.2f} overall and
{NOISE['rho_leafless']:.2f} in the leafless period; day-to-day SD {NOISE['day_sd_leafless']:.3f}.</p>
{img('s4_power_curves', 'S4: probability of detecting a green-up shift between two views, by size of shift.')}

<h2 id="sims">6 Simulations: what this sensor can measure</h2>
{sims}

<h2 id="qa">7 Data quality</h2>
<p>{W['frames_passed']:,} of {W['frames_near_noon']:,} frames within two hours of solar noon passed every check
({W['frames_passed'] / max(W['frames_near_noon'], 1):.0%}).</p>
<div class="tblw">{qa_t}</div>
{illum}
{img('data_health_calendar', 'Frames per day, near-noon frames, and frames passing QA.')}
<p class="small">QA thresholds were set using 200 visually labelled frames, which have not yet been independently confirmed.</p>

<h2 id="questions">8 Open questions and curiosities</h2>
<p>These are not conclusions. They are the questions this record raises, and what each would need.</p>
{qs}

<h2 id="refs">References</h2>
<ol class="small">{refs}</ol>
<footer>Interactive version: the BIFoR canopy cameras dashboard. Values are relative reflectance.</footer>
</main>"""

page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
        f"initial-scale=1'><title>BIFoR canopy cameras 2026</title><style>{CSS}</style></head><body>{body}</body></html>")
OUT.write_text(page, encoding="utf-8")
print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} MB")
