"""Build docs/BIFOR_season_report.html: one self-contained offline report
(plotly.js, data and figures embedded). Reads D:\\BIFOR_work only."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifor import config, pipeline  # noqa: E402

cfg = config.load()
W = cfg.work_dir
OUT = Path(__file__).resolve().parents[1] / "docs" / "BIFOR_season_report.html"
D0 = pd.Timestamp("2026-01-01")
S = W / "figures" / "sim"
F = W / "figures"
P = W / "patterns"


def img(path: Path, alt: str) -> str:
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{alt}" loading="lazy">'


def fig(path, alt, caption, num=None):
    label = f"<b>Figure {num}.</b> " if num else ""
    return f'<figure>{img(path, alt)}<figcaption>{label}{caption}</figcaption></figure>'


def a(href, text):
    return f'<a href="{href}">{text}</a>'


def ext(href, text):
    return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'


# ---------- data for the interactive charts ----------
IDX = ["NDVI", "kNDVI", "NDRE", "CIre", "MTCI", "RVI / SR", "NDI(704,665)", "VOG1", "REDSI", "REP-LE"]
ROIS = ["canopy", "canopy_upper", "canopy_lower", "understory"]
daily = pd.read_parquet(W / "daily.parquet")
daily = daily[daily["index"].isin(IDX) & daily.roi.isin(ROIS)]
series = {}
for (cam, r, n), g in daily.groupby(["camera", "roi", "index"]):
    g = g.sort_values("date")
    series.setdefault(n, {}).setdefault(r, {})[cam] = {
        "d": g.date.dt.strftime("%Y-%m-%d").tolist(),
        "m": g["median"].round(4).tolist(), "q25": g.q25.round(4).tolist(), "q75": g.q75.round(4).tolist(),
        "n": g.n_frames.astype(int).tolist(), "out": g.outlier.astype(bool).tolist()}
ph = pd.read_parquet(W / "phenology.parquet")
fits = {}
for r in ph[ph["index"].isin(IDX) & ph.roi.isin(ROIS) & (ph.error == "")].itertuples():
    fits.setdefault(r.index, {}).setdefault(r.roi, {})[r.camera] = {
        "lo": r.lo, "hi": r.hi, "t0": r.t0, "k": r.k, "reliable": bool(r.reliable),
        "sos20": r.sos20, "mid50": r.mid50, "mid50_lo": r.mid50_lo, "mid50_hi": r.mid50_hi}

frames = pipeline.frames_view(cfg)
near = frames[frames.min_from_noon.abs() <= cfg.noon_window_min]
health = {}
for cam, g in near.groupby("camera"):
    h = g.groupby(g.ts_utc.dt.strftime("%Y-%m-%d")).qa_pass.agg(["size", "sum"])
    health[cam] = {"d": h.index.tolist(), "near": h["size"].astype(int).tolist(), "ok": h["sum"].astype(int).tolist()}
qa_counts = near.qa_reason.replace("", "passed").value_counts().to_dict()
pass_rate = float(near.qa_pass.mean())
DATA = json.dumps({"series": series, "fits": fits, "health": health}, separators=(",", ":"))


def d(x):
    if not np.isfinite(x):
        return "—"
    t = D0 + pd.Timedelta(days=float(x))
    return f"{t.day} {t:%b}"


def fit_row(cam, roi, idx):
    p = ph[(ph.camera == cam) & (ph.roi == roi) & (ph["index"] == idx)]
    return p.iloc[0] if len(p) else None


rows = []
for cam, roi, idx, label in [("A6", "canopy", "NDVI", "A6, whole canopy"), ("A6", "canopy_lower", "NDVI", "A6, lower canopy"),
                             ("A6", "canopy_upper", "NDVI", "A6, upper canopy"), ("A5", "canopy", "NDVI", "A5, whole canopy"),
                             ("A5", "understory", "NDVI", "A5, understorey"), ("A6", "canopy", "kNDVI", "A6, whole canopy"),
                             ("A5", "canopy", "NDRE", "A5, whole canopy"), ("A6", "canopy", "NDRE", "A6, whole canopy")]:
    p = fit_row(cam, roi, idx)
    if p is None:
        continue
    ok = bool(p.reliable)
    rows.append(f"<tr><td>{label}</td><td>{idx}</td><td class='num'>{d(p.sos20)}</td>"
                f"<td class='num'><b>{d(p.mid50)}</b></td><td class='num'>{d(p.mid50_lo)} – {d(p.mid50_hi)}</td>"
                f"<td><span class='pill {'ok' if ok else 'warn'}'>{'meets criteria' if ok else 'not reliable'}</span></td></tr>")
greenup_table = "\n".join(rows)

bm = pd.read_csv(W / "audit" / "by_month.csv")
month_rows = "\n".join(
    f"<tr><td>{r.camera}</td><td>{pd.Timestamp(r.month + '-01'):%b %Y}</td><td class='num'>{r.frames:,}</td>"
    f"<td class='num'>{r.days}</td><td class='num'>{r.daylight_frames:,}</td><td>{r.formats}</td>"
    f"<td class='num'>{r.max_gap_h:g} h</td><td class='num'>{r.GB:g}</td></tr>" for r in bm.itertuples())

cit = pd.read_csv(W / "citations.csv")
cit_counts = cit.status.value_counts().to_dict()
ref_rows = []
for r in cit.itertuples():
    if r.status not in ("verified", "no_doi"):
        continue
    doi = r.doi if isinstance(r.doi, str) and r.doi else ""
    title = r.found_title if isinstance(getattr(r, "found_title", ""), str) else ""
    src = ext(f"https://doi.org/{doi}", doi) if doi else "<span class='note'>no DOI (report or conference paper)</span>"
    cite = r.citation if isinstance(r.citation, str) else ""
    ref_rows.append(f"<tr><td id='ref-{r.index.replace(' ', '').replace('/', '').replace('(', '').replace(')', '').replace(',', '')}'>{r.index}</td>"
                    f"<td>{cite}</td><td>{title}</td><td>{src}</td></tr>")
ref_table = "\n".join(ref_rows)

qa_list = "; ".join(f"{k.replace('_', ' ')}: {v:,}" for k, v in qa_counts.items() if k != "passed")

# ---------- all fitted indices (canopy) ----------
canopy_fits = ph[(ph.roi == "canopy") & (ph.error == "")]
fit_rows = []
for name, g in sorted(canopy_fits.groupby("index"), key=lambda kv: kv[1].mid50.min()):
    cells = []
    for cam in ("A5", "A6"):
        r = g[g.camera == cam]
        if r.empty:
            cells.append("<td>—</td><td>—</td>")
            continue
        r = r.iloc[0]
        ok = bool(r.reliable)
        why = "" if ok else (" (mid-point at record end)" if r.mid_at_record_end else
                             " (interval > 30 days)" if not r.ci_narrow else " (does not rise)")
        cells.append(f"<td class='num'>{d(r.mid50)} <span class='note'>({d(r.mid50_lo)} – {d(r.mid50_hi)})</span></td>"
                     f"<td><span class='pill {'ok' if ok else 'warn'}'>{'meets' if ok else 'fails'}</span><span class='note'>{why}</span></td>")
    fit_rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
fit_table = ("<details class='more'><summary>All " + str(canopy_fits['index'].nunique()) +
             " fitted indices, whole canopy</summary><div class='tablewrap'><table>"
             "<caption>Mid-point (50%) with 95% CI. A fit meets the criteria if the curve rises, the mid-point and its interval end "
             "at least 3 days before the record, and the interval is 30 days or narrower. The 10 exposure-dependent indices are "
             "computed but not fitted, because automatic exposure changes their absolute level.</caption>"
             "<thead><tr><th>Index</th><th>A5 mid-point</th><th>A5 fit</th><th>A6 mid-point</th><th>A6 fit</th></tr></thead><tbody>"
             + "\n".join(fit_rows) + "</tbody></table></div></details>")


def chart(cid, caption):
    return f'<figure class="viz"><div id="{cid}" class="chart mini"></div><figcaption>{caption}</figcaption></figure>'


# ---------- findings ----------
FINDINGS = [
    ("f1", "F1", "Spring phenology baseline", "ok", "data-backed",
     "Canopy green-up begins on 26–27 April. The canopy reaches half of its seasonal NDVI increase by 2 May on camera A6 "
     "(95% CI 29 April – 5 May) and by 9 May on camera A5 (95% CI 3–29 May).",
     f"166–170 daily values per camera, each the median of approximately 20 quality-controlled frames within two hours of "
     f"solar noon. Spring logistic model with moving-block bootstrap intervals. The two cameras agree on the onset date "
     f"to within one day. A direct, model-free measure (date of reaching half the observed winter-to-end rise) gives 5 May (A5) "
     f"and 3 May (A6), consistent with the fits ({a('#explorer', 'interactive charts')}; {a('#tbl-greenup', 'Table 2')}).",
     f"Values are relative reflectance ({a('#g-relative', 'see Glossary')}). The record ends on 3 June, before the canopy plateau."),
    ("f2", "F2", "Leaf expansion precedes chlorophyll accumulation", "ok", "data-backed",
     "All indices begin rising together at the end of April, but NDVI reaches half of its observed rise on 3–5 May, whereas the "
     "red-edge chlorophyll indices (NDRE, CIre) reach it on 18–19 May: a lag of approximately two weeks on both cameras.",
     f"Measured directly from the daily series, without model fitting: the date on which each index first reaches half of its "
     f"rise from the winter median to the end-of-record median. Simulation {a('#s1', 'S1')} confirms that the red-edge indices "
     f"are 82–95% chlorophyll-driven, whereas NDVI is dominated by leaf area and viewing geometry.",
     "The rise is measured to the end of the record (3 June); the red-edge indices are still increasing, so the true lag may be "
     "longer. Fitted mid-points for the red-edge indices fall at the record end and are not used."),
    ("f3", "F3", "Canopy layers green up near-simultaneously", "hyp", "inconclusive",
     "No clear vertical sequence is resolved. Measured directly, the layers reach half of their NDVI rise within 0–2 days of each "
     "other: A5 understorey 3 May, lower and upper canopy 5 May; A6 lower and upper canopy both 3 May.",
     f"Direct half-rise dates per canopy layer on both cameras ({a('#fig-roi', 'Figure 7')} shows the layers). Fitted mid-points "
     f"suggest an earlier understorey on A5 (4 May versus 9–11 May), but their confidence intervals overlap and the direct measure "
     f"does not support a difference.",
     "A layer sequence of a few days may exist below the resolution of this record. Ground observations of budburst by layer "
     f"would resolve it ({a('#q6', 'Request 6')})."),
    ("f4", "F4", "Leaf-out detected without an index or model", "ok", "data-backed",
     "Clustering of the full 24-band spectral shape places the transition to the leafed state on 3 May for both cameras.",
     f"Principal component analysis and k-means clustering of daily canopy spectra, with each camera's mean spectrum removed "
     f"({a('#fig-stages', 'Figure 9')}).",
     f"A methodological result that independently corroborates {a('#f1', 'F1')}."),
    ("f5", "F5", "Illumination biases leafless-season NDVI", "ok", "data-backed",
     "On leafless winter days, NDVI is lower under clear skies (r = −0.65 against clearness index). Cloud cover does not "
     "alter within-day scatter.",
     f"Daily NDVI matched to hourly {ext('https://open-meteo.com/', 'Open-Meteo')} reanalysis and modelled clear-sky "
     f"irradiance ({a('#fig-weather', 'Figure 10')}).",
     "Provides a correction term for winter baselines in phenocam studies."),
    ("f6", "F6", "Quality-control protocol for tower snapshot hyperspectral cameras", "ok", "data-backed",
     f"Seven sources of error specific to this camera type were identified and quantified ({a('#fixes', 'Section 9')}): "
     f"unreliable nightly dark frames (NDVI bias up to 0.22), 1–4 pixel pointing drift, registration aliasing on the tower "
     f"lattice, inconsistent filter selection between cameras (up to 394% of an index's range), fog and frost, unsuitable "
     f"model cloud cover, and UTC timestamps.",
     f"Full-archive audit, pilot run, simulations {a('#s2', 'S2')}, {a('#s3', 'S3')} and {a('#s7', 'S7')}, and 200 labelled frames.",
     "The labelled frames will be cross-checked against site knowledge."),
    ("f7", "F7", "Measurement capability of a red-edge-only camera", "ok", "data-backed",
     "Red-edge indices measure chlorophyll; NDVI-family indices measure leaf amount and are sensitive to view geometry; "
     "red-band ratios respond first to a mildew-like coating; the 930–952 nm bands are dominated by atmospheric water vapour.",
     f"{ext('https://doi.org/10.1016/j.rse.2008.01.026', 'PROSAIL')} simulations {a('#s1', 'S1')}, {a('#s2', 'S2')}, "
     f"{a('#s5', 'S5')} and {a('#s6', 'S6')} using the cameras' calibrated band positions.",
     "Simulation-based guidance for other users of this sensor."),
    ("f8", "F8", "Effect of eCO2 on spring green-up timing", "warn", "requires one input",
     "A difference of approximately 3 days in green-up date between an eCO2 ring and an ambient ring would be detected "
     "with 80% power.",
     f"Simulation {a('#s4', 'S4')} using this season's fitted curve, day-to-day noise (SD 0.041) and the measured shared "
     f"noise between cameras (correlation 0.87).",
     f"The ring and treatment viewed by each camera ({a('#q1', 'Request 1')}). Only timing, not absolute level, is comparable "
     f"between the two views."),
    ("f9", "F9", "Effect of eCO2 on autumn senescence", "hyp", "hypothesis",
     "Not testable with the current record.",
     "The spring-and-autumn (double logistic) model is implemented and validated on a simulated full season.",
     f"Imagery from July to November ({a('#q3', 'Request 3')})."),
    ("f10", "F10", "Red-edge chlorophyll as a proxy for photosynthetic capacity", "hyp", "hypothesis",
     "Red-edge chlorophyll may track photosynthetic capacity under eCO2 and the observed +1.3 °C canopy warming.",
     "Only the chlorophyll component is observable with this sensor.",
     f"Site gas-exchange, canopy temperature and leaf nitrogen data ({a('#q5', 'Request 5')})."),
    ("f11", "F11", "Early detection of oak powdery mildew", "hyp", "hypothesis",
     "Red-band ratios are expected to detect oak powdery mildew at approximately 5–15% canopy infection.",
     f"Simulation {a('#s5', 'S5')} only.",
     "Summer imagery and field infection surveys."),
    ("f12", "F12", "Canopy water use", "bad", "ruled out",
     "Not measurable with these bands.",
     f"Simulation {a('#s6', 'S6')}: atmospheric water vapour dominates by a factor of 6–18, and the 970 nm water feature lies "
     f"beyond the 952 nm sensor limit.",
     "Would require a reflectance panel and a sensor extending beyond 970 nm."),
]
PILL = {"ok": "ok", "warn": "warn", "hyp": "hyp", "bad": "bad"}
VIS = {
    "f1": chart("viz-f1", "Daily canopy NDVI on both cameras with the fitted spring curves; shaded bands are the 95% CI of the "
                "mid-point. Expand the table below for all fitted indices.") + fit_table,
    "f2": chart("viz-f2", "Camera A6, whole canopy, 7-day running median. Each index is scaled from its winter median (0) to its "
                "end-of-record median (1); the dotted line marks the half-rise level. NDVI crosses it on 3 May, NDRE and CIre on 19 May."),
    "f3": chart("viz-f3", "NDVI by canopy layer, 7-day running median, scaled from winter median (0) to end-of-record median (1). "
                "Solid lines: camera A5; dashed: camera A6. The layers cross the half-rise level within 0–2 days of each other."),
    "f4": fig(P / "p5_spectral_stages.png", "Spectral stages", "Daily spectral stage from clustering (left) and the mean spectrum of each stage (right). Both cameras enter the leafed stage (yellow) on 3 May."),
    "f5": fig(P / "p6_weather.png", "Illumination effect", "Left: leafless-season daily NDVI against clearness. Right: within-day scatter by clearness class."),
    "f6": fig(S / "s3_exposure_dark_quant.png", "Dark offset effect", "Example of a quantified error source: index bias from a residual dark offset (left). The full list is in <a href='#fixes'>Section 9</a>."),
    "f7": fig(S / "s1_driver_shares.png", "Index drivers", "Share of each index's variation explained by each simulated canopy factor (<a href='#s1'>S1</a>)."),
    "f8": fig(S / "s4_power_curves.png", "Power curves", "Probability of detecting a green-up shift between rings; right panel corresponds to the measured shared noise (<a href='#s4'>S4</a>)."),
    "f11": fig(S / "s5_mildew.png", "Mildew response", "Index response to increasing infected canopy fraction (<a href='#s5'>S5</a>)."),
    "f12": fig(S / "s6_water_reach.png", "Water sensitivity", "Atmospheric vapour versus leaf-water sensitivity of the 930–952 nm bands (<a href='#s6'>S6</a>)."),
}
cards = "\n".join(
    f'<article class="finding" id="{fid}"><div class="ftext"><div class="head"><h3><span class="tag">{code}</span>{title}</h3>'
    f'<span class="pill {PILL[c]}">{label}</span></div>'
    f'<p><b>Finding.</b> {fnd}</p><p><b>Evidence.</b> {ev}</p><p class="small"><b>Limitations and requirements.</b> {lim}</p></div>'
    f'<div class="fvis">{VIS.get(fid, "")}</div></article>'
    for fid, code, title, c, label, fnd, ev, lim in FINDINGS)

# ---------- simulations ----------
SIMS = [
    ("s1", "S1", "Drivers of each index", S / "s1_driver_shares.png", S / "s1_cab_lai_grids.png",
     "Share of each index's rank variance explained by each simulated factor. cab: leaf chlorophyll; lai: leaf area index; "
     "cw: leaf water; cm: leaf dry matter; cbrown: brown pigment; n: leaf structure; lidfa: mean leaf angle; tts: solar "
     "zenith angle; tto: view zenith angle; psi: relative azimuth; car: carotenoids.",
     "Response of six indices to chlorophyll (horizontal, µg cm⁻²) and leaf area index (vertical). Contours parallel to the "
     "vertical axis indicate chlorophyll control; contours parallel to the horizontal axis indicate leaf-area control.",
     f"3,000 {ext('https://doi.org/10.1016/j.rse.2008.01.026', 'PROSAIL')} oak canopies with all leaf, canopy and "
     f"sun–view parameters drawn at random, resampled to camera A5's calibrated bands.",
     "Red-edge indices (REP, MTCI, Maccioni, SCCCI) are approximately 95% chlorophyll-driven. NDVI is dominated by leaf "
     "area (43%) and viewing geometry (30%).",
     f"Red-edge indices should be used for chlorophyll and NDVI-family indices for leaf amount ({a('#f2', 'F2')}, {a('#f7', 'F7')})."),
    ("s2", "S2", "Agreement between the two cameras", S / "s2_band_bias.png", None,
     "Mean A5 − A6 difference on identical simulated canopies, as a percentage of each index's natural range. Dashed line: 2%.",
     None,
     "The same 1,500 simulated canopies viewed through the filters of A5 and A6, which differ by 0.4–1.5 nm.",
     "NDVI and kNDVI agree to within 0.1% of their range; red-edge indices differ by 3–6%; MTCI by 10%. NAOC differs by 37% "
     "because it sums many bands and assumes absolute reflectance. The analysis also revealed inconsistent filter selection "
     "for Ctr5 (394%), corrected to 7%.",
     f"Cross-camera comparisons should rely on NDVI-family indices ({a('#f8', 'F8')}). MTCI should be interpreted with "
     f"caution and NAOC excluded."),
    ("s3", "S3", "Exposure, dark offset and bit depth", S / "s3_exposure_dark_quant.png", None,
     "Left: bias from a residual dark offset (DN, raw units of 1023); the dotted line marks the level of A5's December night "
     "frames. Centre: index error versus exposure (logarithmic); flat lines indicate exposure invariance. Right: single-frame "
     "RMSE for 8-bit and 10-bit data.",
     None,
     "Automatic exposure, residual dark offset and 8-bit quantisation applied to simulated signals at the cameras' operating level.",
     "Ratio indices are exposure-invariant. A 60 DN dark error, as implied by A5's December night frames, biases NDVI by "
     "approximately 0.22. 8-bit quantisation adds only minor noise.",
     f"Dark-frame treatment is the dominant correction; hence the use of a single master dark per camera ({a('#fixes', 'Section 9')})."),
    ("s4", "S4", "Power to detect a green-up shift between rings", S / "s4_power_curves.png", None,
     "Probability of detecting a green-up shift (days) between two rings. Left: independent daily noise. Right: 80% shared "
     "noise (measured: 87%). 'Day SD' is the day-to-day standard deviation of NDVI; 0.041 was measured. Dashed line: 80% power.",
     None,
     "Paired simulated seasons using this year's fitted curve and noise, with one season shifted by 0–15 days and both "
     "truncated on 3 June.",
     "With the measured shared noise, a 3-day shift is detected with 80% power; with independent noise, approximately 6 days.",
     f"The eCO2 green-up comparison ({a('#f8', 'F8')}) is feasible once the ring viewed by each camera is confirmed."),
    ("s5", "S5", "Oak powdery mildew", S / "s5_mildew.png", None,
     "Left: index change with increasing infected canopy fraction, in units of observed day-to-day noise; values beyond ±2 "
     "are detectable on a single day. Right: healthy and 30%-infected canopy spectra on A5's bands.",
     None,
     "Infected foliage modelled with reduced chlorophyll, brown pigment and a white surface coating, mixed with healthy "
     "canopy at 0–60%.",
     "Red-band ratios (simple ratio, Ctr5, REDSI) respond first, at approximately 5–15% infection. NDVI and NDRE respond weakly.",
     f"Red-band ratios should be monitored in summer imagery ({a('#f11', 'F11')})."),
    ("s6", "S6", "Sensitivity to canopy water", S / "s6_water_reach.png", None,
     "Left: canopy reflectance at three leaf water contents (Cw, g cm⁻²); shading marks the sensor range and the dotted line "
     "the 970 nm water feature. Centre: atmospheric transmission for 1.8 cm precipitable water vapour (PWV) at airmass 1.5. "
     "Right: ratio of vapour-driven to leaf-water-driven change for three band ratios (R869/R944 denotes reflectance at 869 nm "
     "divided by reflectance at 944 nm); values above 1 indicate vapour dominance.",
     None,
     "Leaf water content varied against atmospheric water vapour, observed through the 930–952 nm bands.",
     "Water vapour changes these bands 6–18 times more than leaf water; the 970 nm feature lies outside the sensor range.",
     f"Canopy water use cannot be measured with these cameras ({a('#f12', 'F12')})."),
    ("s7", "S7", "Performance of the quality checks", S / "s7_qa_detection.png", None,
     "Left: fraction of frames flagged as displaced, by injected shift (pixels). Right: fraction flagged for an abrupt "
     "illumination change, by fraction of direct light removed.",
     None,
     "Real near-noon frames with injected camera shifts and simulated passing cloud.",
     "Shifts of 0–5 pixels are recovered exactly. Strong passing cloud is flagged in 77–82% of cases with no false alarms. "
     "The test also identified and led to the correction of a registration fault.",
     f"The checks perform as intended; light haze is not flagged but has negligible effect on ratio indices ({a('#f6', 'F6')})."),
]
sim_html = []
for sid, code, title, pa, pb, cap_a, cap_b, design, result, impl in SIMS:
    figs = fig(pa, title, cap_a) + (fig(pb, title, cap_b) if pb else "")
    sim_html.append(f'<div class="sim" id="{sid}"><div><h3><span class="tag">{code}</span>{title}</h3><dl>'
                    f'<dt>Design</dt><dd>{design}</dd><dt>Result</dt><dd>{result}</dd><dt>Implication</dt><dd>{impl}</dd>'
                    f'</dl></div><div class="figs">{figs}</div></div>')

GLOSS = [
    ("lane", "Lane", f"One of the five research questions defined for this project (D, A′, A, B, C); see {a('#questions', 'Section 3')}."),
    ("eco2", "eCO2 / ambient", "Elevated CO2: FACE rings receiving approximately 150 ppm additional carbon dioxide. Ambient: control rings receiving unenriched air."),
    ("face", "FACE ring", f"A circular forest plot surrounded by towers that release CO2-enriched or ambient air ({ext('https://www.birmingham.ac.uk/research/bifor', 'BIFoR-FACE')})."),
    ("a5a6", "A5, A6", "The two tower cameras, named after their sensor serial numbers (…245 and …246). A5 views a ring obliquely; A6 views the canopy from above."),
    ("band", "Band", "A narrow wavelength interval measured by the camera; 24 bands span 674–952 nm (red, red edge and near infrared)."),
    ("nm", "nm", "Nanometre, the unit of wavelength."),
    ("index", "Index", "A combination of bands that responds to a canopy property such as greenness or chlorophyll."),
    ("rnotation", "R869", "Reflectance at 869 nm; R869/R944 denotes the ratio of two bands."),
    ("rededge", "Red edge", "The steep rise in leaf reflectance between approximately 690 and 750 nm; its position and slope respond to chlorophyll."),
    ("ndvi", "NDVI, kNDVI", f"Normalised Difference Vegetation Index and its kernel form ({ext('https://doi.org/10.1126/sciadv.abc7447', 'Camps-Valls et al., 2021')}); respond mainly to leaf amount."),
    ("ndre", "NDRE, CIre", "Red-edge normalised difference and red-edge chlorophyll index; respond mainly to chlorophyll."),
    ("rep", "REP", "Red-edge position (nm); shifts to longer wavelengths as chlorophyll increases; undefined on leafless branches."),
    ("stress", "REDSI, Ctr2/4/5", "Stress and disease indices responding to chlorophyll loss or changes in leaf surface."),
    ("relative", "Relative reflectance", "The cameras set exposure automatically and no reference panel is in view; ratios and timing are reliable, absolute levels are not."),
    ("dark", "Dark frame / master dark", "An image acquired in darkness to remove the sensor's baseline signal; the master dark is the per-pixel median of the darkest nights."),
    ("drift", "Pixel drift", "Gradual change of a camera's pointing by a few pixels; region masks are shifted to follow it."),
    ("roi", "Region of interest", "The image area analysed (canopy layers), with towers and sky masked."),
    ("qa", "Quality checks (QA)", "Automatic rejection of unsuitable frames: low sun, rain, fog, frost, saturation, camera displacement and abrupt illumination change."),
    ("fit", "Logistic fit", "An S-shaped model fitted to daily values; its 20%, 50% and 80% points mark onset, mid-point and near-completion of green-up."),
    ("ci", "Bootstrap CI", "A 95% confidence interval obtained by refitting resampled blocks of residuals."),
    ("power", "Power", "The probability of detecting a true effect of a given size; 80% is the conventional target."),
    ("prosail", "PROSAIL", f"A coupled leaf and canopy radiative transfer model ({ext('https://doi.org/10.1016/j.rse.2008.01.026', 'Jacquemoud et al., 2009')})."),
    ("pwv", "PWV, airmass", "Precipitable water vapour (total column water, cm) and the relative length of the solar path through the atmosphere."),
    ("senescence", "Senescence", "Autumn ageing, yellowing and fall of leaves."),
]
gloss = "\n".join(f'<dt id="g-{gid}">{term}</dt><dd>{text}</dd>' for gid, term, text in GLOSS)

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light">
<title>BIFoR Canopy Season 2026</title>
<style>
/* Layout: a single reading column on white, a pale-green title band, numbered sections, figures and charts full column width. */
:root{
  color-scheme: light;
  --bg:#FFFFFF; --band:#EEF5EC; --surface:#FFFFFF; --tint:#F5F9F3;
  --ink:#16271D; --muted:#4F6357; --line:#D7E3D6; --rule:#B9CDB6;
  --forest:#2D5F35; --leaf:#5E8C4A; --moss:#8BAA6E;
  --a5:#2B6A8E; --a6:#B4652A;
  --ok:#2F7A45; --warn:#9A6A12; --hyp:#5A6C8A; --bad:#A33A33;
  --display:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --body:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --mono:"Cascadia Mono",Consolas,ui-monospace,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.62 var(--body)}
.wrap{max-width:1100px;margin:0 auto;padding-left:max(20px,env(safe-area-inset-left,0px));padding-right:max(20px,env(safe-area-inset-right,0px))}
a{color:var(--forest);text-decoration:underline;text-decoration-color:var(--moss);text-underline-offset:2px}
a:hover,a:focus-visible{color:var(--leaf)}
h1,h2,h3{font-family:var(--display);font-weight:600;line-height:1.2;text-wrap:balance;margin:0;color:var(--forest)}
h1{font-size:clamp(2rem,4.6vw,3rem);letter-spacing:-.01em;color:var(--ink)}
h2{font-size:1.65rem;display:flex;gap:.6rem;align-items:baseline}
h2 .no{font:600 .9rem/1 var(--mono);color:var(--leaf)}
h3{font-size:1.1rem}
p{margin:0 0 .75rem;max-width:74ch}
code{font:.86em var(--mono);background:var(--tint);padding:.05em .35em;border-radius:3px}
.small,.note{font-size:.88rem;color:var(--muted)}
.titleband{background:var(--band);border-bottom:3px solid var(--forest)}
.titleband .wrap{padding-block:46px 30px;display:grid;gap:12px}
.eyebrow{font:600 .74rem/1.3 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--leaf)}
.lede{font-size:1.1rem;color:var(--muted);max-width:70ch;margin:0}
.meta{font-size:.86rem;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin-top:6px}
.stat{background:var(--surface);padding:14px 16px;display:grid;gap:2px}
.stat b{font:600 1.4rem/1.15 var(--display);color:var(--forest)}
.stat span{font-size:.84rem;color:var(--muted)}
nav.toc{border-bottom:1px solid var(--line);background:var(--bg);position:sticky;top:0;z-index:5}
nav.toc .wrap{display:flex;flex-wrap:wrap;gap:4px 18px;padding-block:10px}
nav.toc a{color:var(--muted);text-decoration:none;font-size:.86rem}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--forest);text-decoration:underline}
main section{padding:46px 0 8px;display:grid;gap:14px;scroll-margin-top:56px;border-bottom:1px solid var(--line)}
main section:last-of-type{border-bottom:0}
ul.summary{margin:0;padding-left:20px}
ul.summary li{margin-bottom:.5rem;max-width:82ch}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:.9rem}
select{font:inherit;padding:5px 8px;border-radius:5px;border:1px solid var(--rule);background:#fff;color:var(--ink)}
label{color:var(--muted)}
.chart{width:100%;height:440px}
.chart.small{height:300px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.9rem;background:#fff}
caption{caption-side:top;text-align:left;font-size:.88rem;color:var(--muted);padding:8px 12px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 .7rem/1.4 var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--forest);background:var(--band)}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;font:600 .68rem/1 var(--mono);padding:4px 8px;border-radius:999px;letter-spacing:.03em;white-space:nowrap;border:1px solid currentColor}
.pill.ok{color:var(--ok);background:#EAF4EC}
.pill.warn{color:var(--warn);background:#FBF3E2}
.pill.hyp{color:var(--hyp);background:#EEF1F6}
.pill.bad{color:var(--bad);background:#F8ECEA}
.findings{display:grid;gap:0}
.finding{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:22px;align-items:start;padding:22px 0;border-top:1px solid var(--line);scroll-margin-top:60px}
.finding .head{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}
.finding p{font-size:.94rem;margin:0 0 .5rem}
.fvis{display:grid;gap:10px;min-width:0}
.finding:target{background:var(--tint)}
@media (max-width:860px){ .finding{grid-template-columns:1fr} }
.chart.mini{height:300px}
details.more{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--tint)}
details.more summary{cursor:pointer;font-weight:600;color:var(--forest);font-size:.9rem}
details.more .tablewrap{margin-top:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--moss);border-radius:8px;padding:16px;display:grid;gap:6px;align-content:start;scroll-margin-top:60px}
.card .head{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.card p{font-size:.93rem;margin:0}
.tag{font:600 .78rem/1 var(--mono);color:var(--leaf);margin-right:.5rem}
.card:target,.sim:target,dt:target{background:var(--tint)}
figure{margin:0;border:1px solid var(--line);border-radius:8px;padding:10px;display:grid;gap:8px;background:#fff;scroll-margin-top:60px}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:.86rem;color:var(--muted)}
.sim{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:20px;align-items:start;padding:20px 0;border-top:1px solid var(--line);scroll-margin-top:60px}
.sim dl{margin:0;display:grid;gap:4px}
.sim dt{font:600 .7rem/1.6 var(--mono);text-transform:uppercase;letter-spacing:.1em;color:var(--leaf)}
.sim dd{margin:0 0 8px}
.figs{display:grid;gap:10px}
@media (max-width:820px){ .sim{grid-template-columns:1fr} }
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.gloss{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:.93rem;margin:0}
.gloss dt{font-weight:600;color:var(--forest);scroll-margin-top:60px}
.gloss dd{margin:0;color:var(--muted)}
@media (max-width:600px){ .gloss{grid-template-columns:1fr} .gloss dd{margin-bottom:8px} }
footer{border-top:3px solid var(--forest);background:var(--band);margin-top:30px}
footer .wrap{padding-block:18px 26px;font-size:.84rem;color:var(--muted)}
@media print{ nav.toc{position:static} .chart{height:360px} }
</style>
<script>__PLOTLY__</script>
</head>
<body>
<div class="titleband"><div class="wrap">
  <span class="eyebrow">BIFoR-FACE · tower hyperspectral cameras A5 and A6 · 1 December 2025 – 3 June 2026</span>
  <h1>BIFoR Canopy Season 2026</h1>
  <p class="lede">A full-archive study of the canopy camera record. Every daylight frame from both towers was processed through a single, tested pipeline. This report presents the results, the strength of evidence for each, and their potential for publication.</p>
  <p class="meta">Site: <a href="https://www.birmingham.ac.uk/research/bifor" target="_blank" rel="noopener">Birmingham Institute of Forest Research FACE facility</a>, Staffordshire, UK · All values are relative reflectance (<a href="#g-relative">see Glossary</a>).</p>
  <div class="stats">
    <div class="stat"><b>23,103</b><span>daylight frames processed</span></div>
    <div class="stat"><b>26–27 April</b><span>onset of green-up, both cameras</span></div>
    <div class="stat"><b>2 May / 9 May</b><span>NDVI mid-point, A6 / A5</span></div>
    <div class="stat"><b>≈3 days</b><span>detectable eCO2 shift in green-up date</span></div>
    <div class="stat"><b>__PASS__%</b><span>of near-noon frames pass quality control</span></div>
  </div>
</div></div>

<nav class="toc" aria-label="Contents"><div class="wrap">
  <a href="#paper">1 Findings</a><a href="#summary">2 Summary</a><a href="#questions">3 Research questions</a>
  <a href="#explorer">4 Seasonal trajectories</a><a href="#outcomes">5 Outcomes</a><a href="#focus">6 Index guidance</a>
  <a href="#sims">7 Simulations</a><a href="#audit">8 Dataset</a><a href="#fixes">9 Corrections</a>
  <a href="#patterns">10 Patterns</a><a href="#open">11 Requests</a><a href="#refs">12 References</a><a href="#glossary">13 Glossary</a>
</div></nav>

<main class="wrap">

<section id="paper">
  <h2><span class="no">1</span>Principal findings and publication potential</h2>
  <p>The findings below derive from the complete record rather than a sample. Each is graded by strength of evidence:
  <span class="pill ok">data-backed</span> demonstrated by this dataset and suitable for publication with the stated limitations;
  <span class="pill warn">requires one input</span> analysis complete, pending one item of information;
  <span class="pill hyp">hypothesis</span> a direction supported in principle but not yet demonstrated;
  <span class="pill bad">ruled out</span> tested and not achievable with this instrument.</p>
  <div class="findings">
__CARDS__
  </div>
  <div class="tablewrap"><table>
    <caption>Table 1. Proposed publications.</caption>
    <thead><tr><th>Proposed paper</th><th>Findings</th><th>Status</th></tr></thead>
    <tbody>
      <tr><td><b>Methods and baseline:</b> obtaining a reliable seasonal record from tower hyperspectral cameras, and what it reveals</td><td><a href="#f1">F1</a>–<a href="#f7">F7</a></td><td><span class="pill ok">ready to draft</span></td></tr>
      <tr><td><b>Elevated CO2 and canopy phenology</b></td><td><a href="#f8">F8</a>, subsequently <a href="#f9">F9</a></td><td><span class="pill warn">awaiting ring assignment</span></td></tr>
      <tr><td><b>Linking canopy spectra to physiology</b></td><td><a href="#f10">F10</a>, <a href="#f11">F11</a></td><td><span class="pill hyp">awaiting site data</span></td></tr>
    </tbody></table></div>
</section>

<section id="summary">
  <h2><span class="no">2</span>Summary</h2>
  <ul class="summary">
    <li><b>Green-up timing.</b> Both cameras record the onset of green-up on 26–27 April, with the canopy NDVI mid-point on 2 May (A6) and 9 May (A5) (<a href="#f1">F1</a>).</li>
    <li><b>Sequence of leaf-out.</b> Leaf expansion precedes chlorophyll accumulation by approximately two weeks (<a href="#f2">F2</a>). Canopy layers green up within 0–2 days of each other; no vertical sequence is resolved (<a href="#f3">F3</a>).</li>
    <li><b>Feasibility of the CO2 comparison.</b> Because the two cameras share most of their day-to-day variation, a 3-day difference in green-up between an eCO2 and an ambient ring would be detectable (<a href="#f8">F8</a>). Confirmation of the ring viewed by each camera is required (<a href="#q1">Request 1</a>).</li>
    <li><b>Scope.</b> Autumn senescence requires imagery after June; photosynthesis requires site gas-exchange data; canopy water use is not measurable with this sensor (<a href="#outcomes">Section 5</a>).</li>
    <li><b>Data quality.</b> Several instrument-specific error sources were identified and corrected before analysis; each would otherwise have distorted the seasonal curve (<a href="#fixes">Section 9</a>).</li>
  </ul>
</section>

<section id="questions">
  <h2><span class="no">3</span>Research questions</h2>
  <p><b>Experimental setting.</b> <a href="https://www.birmingham.ac.uk/research/bifor" target="_blank" rel="noopener">BIFoR-FACE</a> is a mature oak forest in Staffordshire. Selected circular plots (<a href="#g-face">rings</a>) receive approximately 150 ppm additional CO2 throughout the growing season (<a href="#g-eco2">eCO2</a>); paired control rings receive ambient air.</p>
  <p><b>Instrumentation.</b> Two tower-mounted hyperspectral cameras (<a href="#g-a5a6">A5 and A6</a>) record the canopy every 10–12 minutes in 24 <a href="#g-band">bands</a> from 674 to 952 nm.</p>
  <p><b>Lanes.</b> The analysis was planned as five research questions, referred to as <a href="#g-lane">lanes</a>. The original letters are retained for consistency with earlier documents.</p>
  <div class="tablewrap"><table>
    <caption>Research questions and the extent to which this dataset can address them.</caption>
    <thead><tr><th>Lane</th><th>Title</th><th>Question</th><th>Capability of this dataset</th><th>Status</th></tr></thead>
    <tbody>
      <tr id="lane-d"><td>D</td><td><b>Seasonal baseline</b></td><td>What does the winter-to-summer trajectory look like on these cameras?</td><td>Fully addressed: daily values throughout the season and green-up dates (<a href="#f1">F1</a>).</td><td><span class="pill ok">delivered</span></td></tr>
      <tr id="lane-a2"><td>A′</td><td><b>eCO2 and green-up timing</b></td><td>Does canopy green-up in an eCO2 ring differ in timing from an ambient ring? (spring counterpart of Lane A)</td><td>Method established; a 3-day difference is detectable (<a href="#f8">F8</a>).</td><td><span class="pill warn">requires one input</span></td></tr>
      <tr id="lane-a"><td>A</td><td><b>eCO2 and senescence</b></td><td>Does eCO2 alter the timing of autumn <a href="#g-senescence">senescence</a>?</td><td>Not addressable: the record ends on 3 June (<a href="#f9">F9</a>).</td><td><span class="pill hyp">needs autumn data</span></td></tr>
      <tr id="lane-b"><td>B</td><td><b>Photosynthesis and warming</b></td><td>Can the cameras track photosynthesis and the +1.3 °C canopy warming observed under eCO2?</td><td>Chlorophyll component only; requires site measurements (<a href="#f10">F10</a>).</td><td><span class="pill hyp">needs site data</span></td></tr>
      <tr id="lane-c"><td>C</td><td><b>Water use</b></td><td>Can the cameras detect the reduced water use observed under eCO2?</td><td>Not addressable: leaf water cannot be separated from atmospheric vapour (<a href="#f12">F12</a>).</td><td><span class="pill bad">ruled out</span></td></tr>
    </tbody></table></div>
</section>

<section id="explorer">
  <h2><span class="no">4</span>Seasonal trajectories</h2>
  <p>Each point is a daily value: the median of approximately 20 quality-controlled frames within two hours of solar noon. Shading shows the interquartile range within the day. Solid curves are fits meeting the <a href="#g-fit">reliability criteria</a>; dotted curves do not. Hover for values; drag to zoom; double-click to reset.</p>
  <div class="panel">
    <div class="controls">
      <label for="idx">Index</label><select id="idx"></select>
      <label for="roi">Region</label><select id="roi"></select>
      <label><input type="checkbox" id="showOut"> show excluded outlier days</label>
    </div>
    <div id="season" class="chart"></div>
    <p class="note" id="fitnote"></p>
  </div>
  <div class="panel">
    <h3>Difference between cameras (A5 − A6)</h3>
    <p class="note">The two cameras view the canopy from different angles (A5 obliquely, A6 from above), so a constant offset reflects geometry. The quantity relevant to a ring comparison is the timing of change (<a href="#f8">F8</a>).</p>
    <div id="diff" class="chart small"></div>
  </div>
  <div class="tablewrap" id="tbl-greenup"><table>
    <caption>Table 2. Green-up dates from the spring logistic model with 95% bootstrap confidence intervals.</caption>
    <thead><tr><th>Camera and region</th><th>Index</th><th>Onset (20%)</th><th>Mid-point (50%)</th><th>95% CI of mid-point</th><th>Fit</th></tr></thead>
    <tbody>__GREENUP__</tbody></table></div>
</section>

<section id="outcomes">
  <h2><span class="no">5</span>Outcomes by research question</h2>
  <div class="cards">
    <article class="card"><div class="head"><h3>Seasonal baseline <span class="note">(<a href="#lane-d">Lane D</a>)</span></h3><span class="pill ok">delivered</span></div>
      <p>Daily record for 166–170 days per camera with green-up dates and confidence intervals. Onset 26–27 April; NDVI mid-point 2 May (A6) and 9 May (A5). See <a href="#f1">F1</a>–<a href="#f5">F5</a>.</p></article>
    <article class="card"><div class="head"><h3>eCO2 and green-up timing <span class="note">(<a href="#lane-a2">Lane A′</a>)</span></h3><span class="pill warn">requires one input</span></div>
      <p>Simulation with this season's curve and noise shows a 3-day shift is detectable with 80% power, owing to the shared weather signal between cameras (correlation 0.87). Requires the ring viewed by each camera (<a href="#q1">Request 1</a>).</p></article>
    <article class="card"><div class="head"><h3>eCO2 and senescence <span class="note">(<a href="#lane-a">Lane A</a>)</span></h3><span class="pill hyp">needs autumn data</span></div>
      <p>The record ends on 3 June. The spring-and-autumn model is implemented and validated on a simulated full season (<a href="#q3">Request 3</a>).</p></article>
    <article class="card"><div class="head"><h3>Photosynthesis and warming <span class="note">(<a href="#lane-b">Lane B</a>)</span></h3><span class="pill hyp">needs site data</span></div>
      <p>The sensor has no photochemical (PRI) bands and no thermal channel. Red-edge indices track chlorophyll (82–95% in <a href="#s1">S1</a>), a proxy for photosynthetic capacity; linking to photosynthesis requires site measurements (<a href="#q5">Request 5</a>).</p></article>
    <article class="card"><div class="head"><h3>Water use <span class="note">(<a href="#lane-c">Lane C</a>)</span></h3><span class="pill bad">ruled out</span></div>
      <p>The leaf water feature at 970 nm lies beyond the 952 nm sensor limit, and atmospheric water vapour dominates the 930–952 nm bands by a factor of 6–18 (<a href="#s6">S6</a>).</p></article>
  </div>
</section>

<section id="focus">
  <h2><span class="no">6</span>Recommended indices by focus area</h2>
  <p>Each index was assessed on its simulated drivers (<a href="#s1">S1</a>), its agreement between cameras (<a href="#s2">S2</a>), and its observed seasonal signal relative to day-to-day noise. Source publications are listed in <a href="#refs">Section 12</a>.</p>
  <div class="tablewrap"><table>
    <caption>Table 3. Recommended indices for each focus area of the project.</caption>
    <thead><tr><th>Focus area</th><th>Recommended indices</th><th>Basis</th></tr></thead>
    <tbody>
      <tr><td>Chlorophyll and nitrogen</td><td><b>REP-extrap, VOG1, CIre</b></td><td>82–95% chlorophyll-driven; seasonal change 8–11 times the day-to-day noise; still rising at the end of the record. Nitrogen requires calibration against leaf samples.</td></tr>
      <tr><td>Photosynthesis and GPP</td><td><b>kNDVI, simple ratio</b> (light capture); <b>CIre, NDRE</b> (chlorophyll)</td><td>kNDVI and simple ratio provide reliable A6 dates (6 and 8 May). SCCCI and VOG2, proposed in the literature for photosynthetic capacity, are weak or inversely signed in this canopy.</td></tr>
      <tr><td>Canopy structure</td><td><b>kNDVI, NDVI</b></td><td>Driven by leaf area and view geometry; lowest between-camera bias (0.1%); reliable dates 2–9 May.</td></tr>
      <tr><td>Stress and disease</td><td><b>REDSI, CIre, NDRE</b>; red-band ratios for mildew</td><td>A mildew-like surface coating raises red reflectance; simple ratio, Ctr5 and REDSI respond at approximately 5–15% infection (<a href="#s5">S5</a>).</td></tr>
    </tbody></table></div>
</section>

<section id="sims">
  <h2><span class="no">7</span>Sensor simulations</h2>
  <p>Seven simulations established what this sensor can and cannot measure before the field data were interpreted. Canopy spectra were generated with the <a href="https://doi.org/10.1016/j.rse.2008.01.026" target="_blank" rel="noopener">PROSAIL</a> radiative transfer model and resampled to each camera's calibrated bands.</p>
  __SIMS__
</section>

<section id="audit">
  <h2><span class="no">8</span>Dataset and data quality</h2>
  <p>All files were read in place without modification. The archive holds 44,523 time stamps (approximately 67,000 files, as March–June frames comprise a cube and a raw image) from two cameras, recorded every 10–12 minutes, day and night.</p>
  <div class="tablewrap"><table>
    <caption>Table 4. Archive contents by camera and month.</caption>
    <thead><tr><th>Camera</th><th>Month</th><th>Frames</th><th>Days</th><th>Daylight frames</th><th>Formats</th><th>Longest gap</th><th>GB</th></tr></thead>
    <tbody>__MONTHS__</tbody></table></div>
  <p class="note">The file format changes twice: 8-bit PNG until 12 January, 8-bit TIFF until 10 March, then 16-bit cube and raw pairs. No discontinuity in index values occurs at either change. December is incomplete (gaps of up to 70 hours) and the noon sun reaches only 14°.</p>
  <div class="panel"><h3>Frames per day</h3><p class="note">Grey: frames acquired within two hours of solar noon. Coloured: frames passing all <a href="#g-qa">quality checks</a> (__PASS__% overall). Exclusions: __QALIST__.</p><div id="health" class="chart small"></div></div>
  <div class="two">
    __HEALTH_FIG__
    __QA_FIG__
  </div>
  <div class="two">
    __CONTACT_A5__
    __CONTACT_A6__
  </div>
  __ROI_FIG__
  __RULER_FIG__
</section>

<section id="fixes">
  <h2><span class="no">9</span>Methodological corrections</h2>
  <p>Each issue below was identified from the data or by simulation. Each would have distorted the seasonal record had it not been corrected (<a href="#f6">F6</a>).</p>
  <div class="tablewrap"><table>
    <caption>Table 5. Error sources identified and their treatment.</caption>
    <thead><tr><th>Issue</th><th>Consequence if uncorrected</th><th>Treatment</th></tr></thead>
    <tbody>
      <tr><td>File time stamps are in UTC, not local time</td><td>One-hour error after the 29 March clock change</td><td>Confirmed by frames at 01:00–02:00 on 29 March, an hour absent from UK local time</td></tr>
      <tr><td>Nightly <a href="#g-dark">dark frames</a> unreliable: A5 December nights ≈70 DN versus ≈9 DN; May nights not fully dark</td><td>NDVI bias of up to 0.22 (<a href="#s3">S3</a>)</td><td>Single master dark per camera and bit depth from the darkest 40% of nights</td></tr>
      <tr><td>Per-band rescaling in an earlier processing script</td><td>Negative NDVI in May</td><td>Removed; an automated test prevents its reintroduction</td></tr>
      <tr><td>Different filters selected for some indices on the two cameras</td><td>Ctr5 disagreement of 394% of its range (<a href="#s2">S2</a>)</td><td>Bands assigned by filter position common to both cameras; now 7%</td></tr>
      <tr><td>Camera <a href="#g-drift">pointing drift</a> of 1–4 pixels</td><td>Tower steel enters canopy regions, more in winter, producing a spurious trend</td><td>Daily offset measured on the tower structure; region masks shifted accordingly</td></tr>
      <tr><td>Registration aliasing on the 8-pixel tower lattice (9 days)</td><td>Masks displaced by 8 pixels</td><td>Search window limited to ±5 pixels; edge matches rejected (<a href="#s7">S7</a>)</td></tr>
      <tr><td>Model cloud cover ≈100% in most hours</td><td>A cloud criterion would have excluded ≈80% of frames</td><td>Criterion disabled; cloud does not affect within-day scatter (<a href="#f5">F5</a>)</td></tr>
      <tr><td>Fog and frost not detected by standard checks</td><td>Unsuitable days in the winter baseline</td><td>Sharpness and leafless-season saturation criteria, calibrated on 200 labelled frames</td></tr>
      <tr><td>Index catalogue: formula given for NDI45 corresponds to ND705</td><td>An incorrect index reported under that name</td><td>NDI45 as defined by <a href="https://doi.org/10.3390/s110707063" target="_blank" rel="noopener">Delegido et al. (2011)</a> is computed as NDI(704,665)</td></tr>
    </tbody></table></div>
</section>

<section id="patterns">
  <h2><span class="no">10</span>Additional patterns</h2>
  <div class="two">
    __P5__
    __P6__
    __P4__
    __P1__
  </div>
</section>

<section id="open">
  <h2><span class="no">11</span>Information requested from the site</h2>
  <p>Requests are listed in order of priority; each answer enables a specific next step.</p>
  <div class="tablewrap"><table>
    <caption>Table 6. Information requests.</caption>
    <thead><tr><th>Request</th><th>Context</th><th>Enables</th></tr></thead>
    <tbody>
      <tr id="q1"><td><b>1. Which FACE ring does each camera view, and what is its treatment?</b></td><td>A5 views a ring obliquely with sky in frame; A6 views a canopy from above. The imagery does not identify the ring or treatment.</td><td><a href="#f8">F8</a> (<a href="#lane-a2">Lane A′</a>); analysis ready.</td></tr>
      <tr id="q2"><td><b>2. Are the two rings comparable in species composition and age?</b></td><td>A5 greens more gradually than A6 (mid-point 9 May versus 2 May), which may reflect treatment, viewing angle or stand differences.</td><td>Separation of a CO2 effect from stand differences.</td></tr>
      <tr id="q3"><td><b>3. Is imagery available after 3 June 2026?</b></td><td>The record ends at 06:17 on 3 June, while the canopy is still greening.</td><td><a href="#f9">F9</a> (<a href="#lane-a">Lane A</a>) and the full-season curve.</td></tr>
      <tr id="q4"><td><b>4. Was November 2025 recorded elsewhere?</b></td><td>The November folders contain no images; December includes gaps of up to 70 hours.</td><td>A more complete winter baseline.</td></tr>
      <tr id="q5"><td><b>5. Site measurements, December–June: gas exchange, canopy temperature, sap flow, leaf nitrogen and chlorophyll, LAI, flux or GPP.</b></td><td>The cameras observe greenness and chlorophyll only.</td><td><a href="#f10">F10</a> (<a href="#lane-b">Lane B</a>) and calibrated chlorophyll estimates.</td></tr>
      <tr id="q6"><td><b>6. Ground observations of budburst or leaf-out in spring 2026.</b></td><td>The cameras indicate onset on 26–27 April and mid-point in early May.</td><td>Independent validation of <a href="#f1">F1</a>–<a href="#f3">F3</a>.</td></tr>
      <tr id="q7"><td><b>7. Camera maintenance or settings log.</b></td><td>A5 December night frames are ≈8× brighter than in January; both cameras drift 1–4 pixels; the file format changed on 12 January and 10 March.</td><td>Confirmation that corrections correspond to recorded events.</td></tr>
      <tr id="q8"><td><b>8. Installation of a grey reflectance panel in view for the next season.</b></td><td>Without a reference, results are relative only.</td><td>Absolute reflectance, comparison with satellite data, and any assessment of <a href="#lane-c">Lane C</a>.</td></tr>
    </tbody></table></div>
  <p class="note">The 200 frames used to calibrate the fog and frost criteria were labelled visually by our team and will be cross-checked. Of the index source publications, __CIT_OK__ are verified against their DOI records and __CIT_NODOI__ are reports or conference papers without a DOI (<a href="#refs">Section 12</a>).</p>
</section>

<section id="refs">
  <h2><span class="no">12</span>References</h2>
  <p>Index source publications, verified against <a href="https://www.crossref.org/" target="_blank" rel="noopener">Crossref</a>. Model and data sources: PROSAIL (<a href="https://doi.org/10.1016/j.rse.2008.01.026" target="_blank" rel="noopener">Jacquemoud et al., 2009</a>); weather reanalysis (<a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo</a>); solar position (<a href="https://pvlib-python.readthedocs.io/" target="_blank" rel="noopener">pvlib</a>).</p>
  <div class="tablewrap"><table>
    <caption>Table 7. Index references.</caption>
    <thead><tr><th>Index</th><th>Source</th><th>Title</th><th>DOI</th></tr></thead>
    <tbody>__REFS__</tbody></table></div>
</section>

<section id="glossary">
  <h2><span class="no">13</span>Glossary</h2>
  <dl class="gloss">
__GLOSS__
  </dl>
</section>

</main>

<footer><div class="wrap">BIFoR-FACE canopy camera study · complete record 1 December 2025 – 3 June 2026 · processing pipeline v__CODEV__, quality rules v__QAV__ · prepared __BUILT__ · all values are relative reflectance.</div></footer>

<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const D0 = Date.UTC(2026,0,1);
const toDate = t => new Date(D0 + t*86400000);
const logistic = (t,p) => p.lo + (p.hi-p.lo)/(1+Math.exp(-p.k*(t-p.t0)));
const fmt = t => toDate(t).toLocaleDateString('en-GB',{day:'numeric',month:'short',timeZone:'UTC'});
const ROI_LABEL = {canopy:'Whole canopy',canopy_upper:'Upper canopy',canopy_lower:'Lower canopy',understory:'Understorey (A5 only)'};
const idxSel = document.getElementById('idx'), roiSel = document.getElementById('roi'), outBox = document.getElementById('showOut');
Object.keys(DATA.series).forEach(k => idxSel.add(new Option(k,k)));
['canopy','canopy_upper','canopy_lower','understory'].forEach(k => roiSel.add(new Option(ROI_LABEL[k],k)));
idxSel.value = 'NDVI';
function layout(extra){
  return Object.assign({paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',
    font:{family:css('--body'),color:css('--ink'),size:12},margin:{l:56,r:14,t:10,b:40},
    xaxis:{gridcolor:css('--line'),linecolor:css('--rule'),zeroline:false},
    yaxis:{gridcolor:css('--line'),linecolor:css('--rule'),zeroline:false},
    legend:{orientation:'h',y:1.08,x:0},hovermode:'x unified'},extra||{});
}
const CFG = {displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d']};
function drawSeason(){
  const idx = idxSel.value, roi = roiSel.value, showOut = outBox.checked;
  const S = (DATA.series[idx]||{})[roi]||{}, Fi = ((DATA.fits[idx]||{})[roi])||{};
  const traces = [], shapes = [], notes = [];
  for (const cam of ['A5','A6']){
    const s = S[cam]; if(!s) continue;
    const col = css(cam==='A5'?'--a5':'--a6');
    const keep = s.d.map((_,i)=> showOut || !s.out[i]);
    const pick = a => a.filter((_,i)=>keep[i]);
    const d = pick(s.d);
    traces.push({x:d.concat(d.slice().reverse()),y:pick(s.q75).concat(pick(s.q25).reverse()),fill:'toself',
      fillcolor:col+'22',line:{width:0},hoverinfo:'skip',showlegend:false,type:'scatter'});
    traces.push({x:d,y:pick(s.m),mode:'markers',marker:{size:5,color:col},name:cam+' daily median',
      customdata:pick(s.n),hovertemplate:cam+': %{y:.3f} (%{customdata} frames)<extra></extra>',type:'scatter'});
    const p = Fi[cam];
    if(p){
      const t0 = (new Date(d[0])-D0)/86400000, t1 = (new Date(d[d.length-1])-D0)/86400000;
      const ts=[],ys=[]; for(let t=t0;t<=t1;t+=1){ts.push(toDate(t));ys.push(logistic(t,p));}
      traces.push({x:ts,y:ys,mode:'lines',line:{color:col,width:p.reliable?2.2:1.2,dash:p.reliable?'solid':'dot'},
        name:cam+' fit'+(p.reliable?'':' (does not meet criteria)'),hoverinfo:'skip',type:'scatter'});
      if(p.reliable){
        shapes.push({type:'rect',xref:'x',yref:'paper',x0:toDate(p.mid50_lo),x1:toDate(p.mid50_hi),y0:0,y1:1,fillcolor:col,opacity:.09,line:{width:0}});
        shapes.push({type:'line',xref:'x',yref:'paper',x0:toDate(p.mid50),x1:toDate(p.mid50),y0:0,y1:1,line:{color:col,width:1.2,dash:'dash'}});
        notes.push(`${cam}: onset ${fmt(p.sos20)}, mid-point ${fmt(p.mid50)} (95% CI ${fmt(p.mid50_lo)} – ${fmt(p.mid50_hi)})`);
      } else notes.push(`${cam}: fit does not meet the reliability criteria (mid-point at the end of the record, or interval too wide)`);
    }
  }
  Plotly.react('season',traces,layout({shapes,yaxis:{title:idx,gridcolor:css('--line'),linecolor:css('--rule'),zeroline:false}}),CFG);
  document.getElementById('fitnote').textContent = notes.join(' · ') || 'No data for this combination.';
  const a = S.A5, b = S.A6, dx=[], dy=[];
  if(a && b){ const m = new Map(b.d.map((x,i)=>[x,b.m[i]]));
    a.d.forEach((x,i)=>{ if(m.has(x) && !a.out[i]) {dx.push(x); dy.push(a.m[i]-m.get(x));} }); }
  Plotly.react('diff',[{x:dx,y:dy,mode:'markers',marker:{size:4,color:css('--forest')},name:'A5 − A6',type:'scatter',
    hovertemplate:'%{y:.3f}<extra></extra>'}],layout({yaxis:{title:idx+' (A5 − A6)',gridcolor:css('--line'),zeroline:true,zerolinecolor:css('--rule')},showlegend:false}),CFG);
}
function drawHealth(){
  const tr=[];
  for(const cam of ['A5','A6']){ const h=DATA.health[cam]; if(!h) continue; const col=css(cam==='A5'?'--a5':'--a6');
    tr.push({x:h.d,y:h.near,type:'bar',name:cam+' acquired',marker:{color:'#D7E3D6'},hovertemplate:cam+' acquired %{y}<extra></extra>'});
    tr.push({x:h.d,y:h.ok,type:'bar',name:cam+' passed',marker:{color:col},hovertemplate:cam+' passed %{y}<extra></extra>'}); }
  Plotly.react('health',tr,layout({barmode:'group',bargap:0.05,yaxis:{title:'frames per day',gridcolor:css('--line')}}),CFG);
}
function med7(v){ return v.map((_,i)=>{ const w=v.slice(Math.max(0,i-3),i+4).filter(x=>x!==null).sort((a,b)=>a-b); return w.length? w[Math.floor(w.length/2)] : null; }); }
function scaled(idx, roi, cam){
  const s=((DATA.series[idx]||{})[roi]||{})[cam]; if(!s) return null;
  const keep=s.d.map((_,i)=>!s.out[i]); const d=s.d.filter((_,i)=>keep[i]); const m=med7(s.m.filter((_,i)=>keep[i]));
  const medOf=a=>{const z=a.filter(x=>x!==null).sort((p,q)=>p-q); return z[Math.floor(z.length/2)];};
  const lo=medOf(m.filter((_,i)=>d[i]<'2026-04-01')), hi=medOf(m.filter((_,i)=>d[i]>='2026-05-27'));
  return {d, y:m.map(v=>v===null?null:(v-lo)/(hi-lo))};
}
function drawLines(id, items, ytitle){
  const tr=items.map(it=>{ const s=scaled(it.idx,it.roi,it.cam); return s?{x:s.d,y:s.y,mode:'lines',name:it.label,line:{color:it.color,width:2,dash:it.dash||'solid'},type:'scatter',hovertemplate:it.label+': %{y:.2f}<extra></extra>'}:null; }).filter(Boolean);
  const half={type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:0.5,y1:0.5,line:{color:css('--muted'),width:1,dash:'dot'}};
  Plotly.newPlot(id,tr,layout({shapes:[half],yaxis:{title:ytitle,gridcolor:css('--line'),range:[-0.3,1.3]},xaxis:{gridcolor:css('--line'),range:['2026-03-15','2026-06-04']}}),CFG);
}
function drawF1(){
  const traces=[], shapes=[]; const S=(DATA.series.NDVI||{}).canopy||{}, Fi=(DATA.fits.NDVI||{}).canopy||{};
  for(const cam of ['A5','A6']){ const s=S[cam]; if(!s) continue; const col=css(cam==='A5'?'--a5':'--a6');
    const d=s.d.filter((_,i)=>!s.out[i]), m=s.m.filter((_,i)=>!s.out[i]);
    traces.push({x:d,y:m,mode:'markers',marker:{size:4,color:col},name:cam,type:'scatter',hovertemplate:cam+': %{y:.3f}<extra></extra>'});
    const p=Fi[cam]; if(p){ const ts=[],ys=[]; for(let t=-30;t<=153;t++){ts.push(toDate(t));ys.push(logistic(t,p));}
      traces.push({x:ts,y:ys,mode:'lines',line:{color:col,width:2,dash:p.reliable?'solid':'dot'},name:cam+' fit',type:'scatter',hoverinfo:'skip'});
      if(p.reliable){ shapes.push({type:'rect',xref:'x',yref:'paper',x0:toDate(p.mid50_lo),x1:toDate(p.mid50_hi),y0:0,y1:1,fillcolor:col,opacity:.10,line:{width:0}});
        shapes.push({type:'line',xref:'x',yref:'paper',x0:toDate(p.mid50),x1:toDate(p.mid50),y0:0,y1:1,line:{color:col,width:1.2,dash:'dash'}}); } } }
  Plotly.newPlot('viz-f1',traces,layout({shapes,yaxis:{title:'NDVI',gridcolor:css('--line')}}),CFG);
}
drawF1();
drawLines('viz-f2',[{idx:'NDVI',roi:'canopy',cam:'A6',label:'NDVI',color:'#2D5F35'},{idx:'kNDVI',roi:'canopy',cam:'A6',label:'kNDVI',color:'#8BAA6E'},
  {idx:'NDRE',roi:'canopy',cam:'A6',label:'NDRE',color:'#2B6A8E'},{idx:'CIre',roi:'canopy',cam:'A6',label:'CIre',color:'#B4652A'}],'fraction of observed rise');
drawLines('viz-f3',[{idx:'NDVI',roi:'understory',cam:'A5',label:'A5 understorey',color:'#8BAA6E'},{idx:'NDVI',roi:'canopy_lower',cam:'A5',label:'A5 lower canopy',color:'#5E8C4A'},
  {idx:'NDVI',roi:'canopy_upper',cam:'A5',label:'A5 upper canopy',color:'#2D5F35'},{idx:'NDVI',roi:'canopy_lower',cam:'A6',label:'A6 lower canopy',color:'#C98A4B',dash:'dash'},
  {idx:'NDVI',roi:'canopy_upper',cam:'A6',label:'A6 upper canopy',color:'#8A4B1E',dash:'dash'}],'fraction of observed NDVI rise');
[idxSel,roiSel,outBox].forEach(e=>e.addEventListener('change',drawSeason));
drawSeason(); drawHealth();
</script>
</body>
</html>
"""

vers = cfg.versions
repl = {
    "__PLOTLY__": plotly.offline.get_plotlyjs(),
    "__DATA__": DATA,
    "__PASS__": f"{pass_rate * 100:.0f}",
    "__QALIST__": qa_list,
    "__CARDS__": cards,
    "__GREENUP__": greenup_table,
    "__MONTHS__": month_rows,
    "__SIMS__": "\n".join(sim_html),
    "__REFS__": ref_table,
    "__GLOSS__": gloss,
    "__HEALTH_FIG__": fig(F / "data_health_calendar.png", "Data availability calendar", "Frames per day for each camera: all frames, near-noon frames and frames passing quality control. Shading marks the 8-bit periods.", 1),
    "__QA_FIG__": fig(F / "qa_breakdown.png", "Quality-control outcomes by month", "Reasons for exclusion of near-noon frames, by month.", 2),
    "__CONTACT_A5__": fig(W / "audit" / "contact_A5.png", "Camera A5, monthly frames", "Camera A5, one noon frame per month: false-colour composite (left) and NDVI (right). A5 views the ring obliquely, with sky at upper right.", 3),
    "__CONTACT_A6__": fig(W / "audit" / "contact_A6.png", "Camera A6, monthly frames", "Camera A6, one noon frame per month. A6 views the canopy from above; no sky is in frame.", 4),
    "__ROI_FIG__": '<div id="fig-roi">' + fig(F / "roi_overlay.png", "Regions of interest", "Regions analysed on each camera. Areas shown in orange are excluded: painted tower structure and, for A5, sky.", 7) + "</div>",
    "__RULER_FIG__": fig(F / "band_ruler.png", "Band assignment", "Bands used by each index on each camera. Crosses mark required wavelengths outside the sensor range; those indices are not computed.", 8),
    "__P5__": '<div id="fig-stages">' + fig(P / "p5_spectral_stages.png", "Spectral stages", "Days grouped by spectral shape alone, without an index or model. Both cameras enter the leafed stage (yellow) on 3 May (<a href='#f4'>F4</a>).", 9) + "</div>",
    "__P6__": '<div id="fig-weather">' + fig(P / "p6_weather.png", "Illumination effect", "Leafless-season NDVI is lower under clear skies; within-day scatter is unaffected by cloud (<a href='#f5'>F5</a>).", 10) + "</div>",
    "__P4__": fig(P / "p4_red_edge.png", "Red-edge position", "A6 red-edge position shifts from 711 to 721 nm with leaf-out. A5 winter values are undefined (no red edge on bare branches).", 11),
    "__P1__": fig(P / "p1_diurnal.png", "Diurnal stability", "NDVI varies by less than 0.006 between morning, noon and afternoon, supporting the ±2-hour noon window.", 12),
    "__CIT_OK__": str(cit_counts.get("verified", 0)),
    "__CIT_NODOI__": str(cit_counts.get("no_doi", 0)),
    "__CODEV__": vers["code"], "__QAV__": vers["qa"],
    "__BUILT__": pd.Timestamp.now().strftime("%d %B %Y"),
}
html = HTML
for k, v in repl.items():
    html = html.replace(k, v)
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
