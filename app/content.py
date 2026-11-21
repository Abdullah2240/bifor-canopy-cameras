"""Text shared by the dashboard and the HTML report. Numbers come from summary.json."""
from __future__ import annotations

import pandas as pd


def dt(x) -> str:
    if not x or (not isinstance(x, str) and pd.isna(x)):
        return "n/a"
    t = pd.Timestamp(x)
    return f"{t.day} {t:%b}"


REFERENCES = [
    ("Gao, Y., Ma, L., Zhao, G., et al. (2026). Elevated CO2 advances vegetation phenology. <i>Functional Ecology</i>, "
     "40(9), 2946–2958.", "https://doi.org/10.1111/1365-2435.70402"),
    ("Jones, A. L., Gardner, A., Hayes, F., Pfrang, C., & Jeffers, E. S. (2025). The effect of elevated CO2 on "
     "hyperspectral leaf reflectance in mature trees. <i>Trees</i>, 39(4).", "https://doi.org/10.1007/s00468-025-02650-w"),
    ("Hagan Brown, W., Gloor, E., Fyfe, R., MacKenzie, A. R., Harper, N. J., Ganderton, P., et al. (2025). Elevated CO2 "
     "increases the canopy temperature of mature <i>Quercus robur</i>. <i>Global Change Biology</i>, 31, e70565.",
     "https://doi.org/10.1111/gcb.70565"),
    ("Jacquemoud, S., Verhoef, W., Baret, F., et al. (2009). PROSPECT + SAIL models: a review of use for vegetation "
     "characterization. <i>Remote Sensing of Environment</i>, 113, S56–S66.", "https://doi.org/10.1016/j.rse.2008.01.026"),
]


def tenets(S) -> list[tuple[str, str, str, str]]:
    W = S["window"]
    return [
        ("T1", "Treatment labels: A6 = elevated CO2, A5 = ambient", "assumption",
         "The ring and treatment viewed by each camera are not known. A meta-analysis of 624 records from 39 "
         "experiments found that elevated CO2 advances leaf-out by 3.0 days on average (Gao et al., 2026). Both cameras "
         "start greening together, but A6 completes its green-up sooner on the NDVI-family indices, which compare "
         "fairly across the two cameras (S2). A6 is therefore labelled as the elevated-CO2 view."
         "<div class='note'><b>Limits.</b> The label was chosen from the observed result, so the camera difference "
         "cannot count as evidence of a CO2 effect; it can only be compared with the published effect size. The two "
         "cameras also view the canopy at different angles (A5 obliquely, A6 from above), which can shift dates on "
         "its own. Gao et al. pool many species and experimental set-ups; the estimate is not specific to mature oak "
         "in a FACE ring. At BIFoR itself, elevated CO2 leaves overall oak leaf reflectance unchanged but raises "
         "PSRI (Jones et al., 2025) and warms the canopy (Hagan Brown et al., 2025).</div>"),
        ("T2", f"Valid record: {dt(W['first_day'])} – {dt(W['last_day'])} 2026", "site guidance",
         "Data before March 2026 is excluded on site guidance. The record starts with the first 16-bit cube files on "
         "10 March at 10:31 UTC and ends with the last frame on 3 June. No later imagery exists."),
        ("T3", "The two views are comparable stands", "site guidance",
         "Both cameras view the same species (<i>Quercus robur</i> canopy) in the same forest."),
        ("T4", "No camera changes during the record", "assumption",
         "No maintenance or settings changes are known for March–June. Slow pointing drift of 1–2 pixels is "
         "measured and corrected every day."),
        ("T5", "Relative reflectance only", "fact",
         "The cameras auto-expose and no reference panel was in view, so only band ratios and normalised indices are "
         "used. Absolute reflectance and comparison with satellite products are not possible for this record."),
        ("T6", "Imagery only, no ground truth", "scope",
         "Results are checked internally: the two cameras against each other, several independent indices, and a "
         "model-free spectral clustering. There is no ground observation in this analysis."),
        ("T7", "Timestamps are UTC", "fact",
         "Frames continue through 01:00–02:00 on 29 March 2026, an hour that does not exist in UK local time."),
    ]


def methods(S) -> list[tuple[str, str]]:
    W = S["window"]
    return [
        ("Cameras and data",
         "Two snapshot hyperspectral cameras on BIFoR-FACE towers (A5, sensor …245; A6, sensor …246), 25 filter bands "
         "of which 24 are usable, 674–952 nm. A5 views a ring obliquely with sky in view; A6 looks down onto the "
         f"canopy. Frames every ~11 minutes, day and night. Valid record {dt(W['first_day'])} – {dt(W['last_day'])} 2026."),
        ("1. Decoding and dark correction",
         "Each frame is read as a 25-band cube. One master dark per camera (per-pixel median of the darkest nights) is "
         "subtracted; per-night darks were rejected because twilight contaminates them. Saturated pixels are masked."),
        ("2. Pointing drift",
         "The cameras move by 1–2 pixels over the record. Each day's offset is measured by masked correlation on the "
         "tower structure within ±5 pixels, and the region masks follow it. Frames more than 4 pixels from their "
         "day's offset are flagged."),
        ("3. Quality control",
         "Frames are kept when the sun is above 10°, saturation is below 25% of the canopy, there is no abrupt light "
         "change against neighbouring frames, no rain in the hour, the image is sharp (fog and haze filter), and "
         "there is no frost or snow saturation. Thresholds were set from simulations and 200 visually labelled frames."),
        ("4. Regions and indices",
         "Fixed canopy regions (whole canopy, upper, lower, understorey). Indices are resolved by filter slot so both "
         "cameras use the same physical bands. Only ratio and normalised indices are used, because they cancel the "
         "automatic exposure."),
        ("5. Daily series",
         "Each day's value is the median of quality-passed frames within 2 hours of solar noon (median "
         f"{W.get('frames_per_day_median', 0):.0f} frames per camera-day). Outlier days are flagged with a Hampel filter."),
        ("6. Green-up dates",
         "(a) A spring logistic curve fitted to each daily series, with 95% intervals from a 7-day moving-block "
         "bootstrap. (b) A model-free half-rise date: the first day after which the 7-day running median stays at or above half of the rise "
         f"from the leafless baseline ({dt(W['first_day'])} – {dt(W['leafless_end'])}) to the last week of the record. "
         "Both are reported; where they disagree, the direct measure is preferred because the record ends before the "
         "canopy plateaus."),
        ("7. Camera comparison",
         "Half-way dates of A5 and A6 are differenced on the days both cameras share, with a joint block bootstrap that "
         "resamples the same days for both cameras, so their shared weather noise is kept."),
        ("8. Spectral clustering",
         "Daily canopy spectra (24 bands, each camera's mean removed) are reduced by principal components and grouped "
         "by k-means, without any index or model."),
        ("9. Simulations",
         "PROSAIL canopy simulations (Jacquemoud et al., 2009) on the cameras' calibrated bands test what each index "
         "responds to (S1), camera agreement (S2), exposure and dark errors (S3), the power to detect a timing shift "
         "(S4), mildew (S5), leaf water (S6) and the QA checks (S7)."),
    ]


SIMS = [
    ("s1_driver_shares", "S1 · What drives each index",
     "3,000 simulated oak canopies (PROSAIL) on A5's bands. Red-edge indices are mostly driven by chlorophyll; NDVI by "
     "leaf area and viewing geometry. Use red-edge indices for chlorophyll and NDVI-family indices for leaf amount."),
    ("s2_band_bias", "S2 · Do the two cameras agree?",
     "Identical canopies seen through each camera's filters, which differ by 0.4–1.5 nm. NDVI-family indices agree "
     "closely; red-edge indices differ by a few per cent. Cross-camera timing should lean on NDVI-family indices."),
    ("s3_exposure_dark_quant", "S3 · Exposure, dark offset, bit depth",
     "Ratio indices cancel automatic exposure. A residual dark offset is the largest error source, hence the master "
     "dark per camera."),
    ("s4_power_curves", "S4 · Power to detect a timing shift",
     "Paired simulated seasons with this record's curve and noise, truncated on 3 June."),
    ("s5_mildew", "S5 · Oak powdery mildew",
     "Infected foliage mixed into healthy canopy. Red-band ratios respond first, at roughly 5–15% infection; NDVI and "
     "NDRE respond weakly."),
    ("s6_water_reach", "S6 · Leaf water",
     "Atmospheric water vapour changes the 930–952 nm bands far more than leaf water, and the 970 nm water feature is "
     "beyond the sensor. Canopy water is out of reach for this camera."),
    ("s7_qa_detection", "S7 · Do the quality checks work?",
     "Real frames with injected shifts and simulated passing cloud. Shifts are recovered; strong passing cloud is "
     "flagged without false alarms."),
]

QA_FIXES = [
    ("Nightly dark frames unreliable (twilight)", "Index bias", "One master dark per camera from the darkest nights"),
    ("Pointing drift of 1–2 pixels", "Tower steel leaks into canopy regions", "Daily offset measured on the tower; masks follow it"),
    ("Tower lattice repeats every ~8 pixels", "Registration can lock onto the wrong repeat", "Search limited to ±5 px; edge matches rejected"),
    ("Filters differ slightly between cameras", "Some indices picked different bands per camera", "Bands resolved by filter slot"),
    ("Fog, haze and frost pass simple checks", "Bad days in the baseline", "Sharpness and leafless-saturation rules"),
    ("Model cloud cover near 100% most hours", "Would reject most good frames", "Measured light change used instead"),
    ("File timestamps are UTC", "One-hour error after the clock change", "All times handled in UTC"),
]

QUESTIONS = [
    ("Is the camera difference a CO2 effect?",
     "Knowing which ring each camera views would turn the camera comparison into a real treatment test. The design "
     "can detect a shift of a few days (S4)."),
    ("Why does the red-edge rise pause in mid-May?",
     "On both cameras the red-edge indices rise in two steps: up in late April and early May, a pause around "
     "12 May, then a second rise from about 19 May. NDVI shows only a small pause. Whether this reflects weather, "
     "leaf development stages or canopy structure is open; air temperature and leaf sampling would separate them."),
    ("How long is the chlorophyll lag?",
     "The red-edge indices are still rising when the record ends. Imagery through June and July would close the curve."),
    ("Does elevated CO2 change autumn senescence?",
     "Needs imagery through autumn. A spring-and-autumn curve is already implemented and tested on simulated seasons."),
    ("Does red-edge chlorophyll track photosynthetic capacity?",
     "Would need leaf gas exchange, leaf nitrogen and canopy temperature from the site."),
    ("Do canopy layers green up in sequence?",
     "Differences of a few days may exist below this record's resolution; ground budburst records by layer would "
     "settle it."),
    ("Can powdery mildew be detected early?",
     "Simulation suggests red-band ratios respond at roughly 5–15% infection. Needs summer imagery and field surveys."),
    ("Can results be put in absolute terms?",
     "A reference panel is now in view. Future imagery could be converted to absolute reflectance and compared with "
     "satellite data."),
]


def camera_glance(S) -> str:
    L = S.get("lane_a_prime", {})
    return (f"<b>Camera difference.</b> Both cameras reach half of their NDVI rise on the same day "
            f"({dt(L.get('half_A5'))}); A6 then finishes sooner, reaching 80% {abs(L.get('direct80_shift') or 0)} days "
            f"before A5. See the camera comparison for how this is, and is not, interpreted.")


def camera_text(S) -> str:
    L, N = S.get("lane_a_prime", {}), S.get("noise", {})
    rel = "" if L.get("fit_A5_reliable") else (
        " The A5 fit does not meet the reliability criteria, because A5 keeps rising to the end of the record, so "
        "the fitted difference is driven by the shape of the late-May curve rather than by the timing of green-up.")
    return (f"On canopy NDVI, over {L.get('n_shared_days', 0):.0f} shared days, the two direct measures give "
            f"<b>{L.get('direct_shift', 0):+.0f} days at half-way</b> (both {dt(L.get('half_A5'))}) and "
            f"<b>{L.get('direct80_shift', 0):+d} days at 80%</b> (A5 − A6; positive means A6 is earlier). The fitted "
            f"half-way dates differ by {L.get('shift', float('nan')):+.1f} days (95% CI {L.get('lo', float('nan')):+.1f} "
            f"to {L.get('hi', float('nan')):+.1f}, joint bootstrap).{rel} Taken together, the cameras start greening "
            f"together and A6 completes green-up a few days sooner. The size of that difference at 80% is similar to "
            f"the published leaf-out effect; under Assumption T1 this is a comparison of size, not a test. The two "
            f"cameras' daily NDVI correlates at {N.get('rho_all', float('nan')):.2f}, so weather and light largely "
            f"cancel in the difference.")


ROADMAP = [
    ("1", "Methods and seasonal baseline", "Ready now",
     "How to obtain a reliable seasonal record from tower snapshot hyperspectral cameras in a mature oak FACE forest.",
     "The QA protocol and its seven corrected error sources; simulations of what the sensor can and cannot measure; "
     "spring green-up dates; the two-week lag of chlorophyll behind leaf expansion; leaf-out detected from spectral "
     "shape alone.",
     "Nothing further. Can be drafted from this record.",
     "A methods paper, and a reusable processing pipeline for other camera towers."),
    ("2", "Elevated CO2 and spring green-up timing", "One input",
     "Does elevated CO2 change when the mature oak canopy greens up?",
     "The camera comparison is built and tested. With the measured noise, a shift of about 4 days between two views is "
     "detectable (simulation S4).",
     "Confirmation of which ring, and which treatment, each camera views. Ideally, one camera on an elevated and one on "
     "an ambient ring with similar viewing angles.",
     "A short paper on CO2 and spring phenology, or a section of route 1."),
    ("3", "Elevated CO2 across the whole season", "Needs a season of imagery",
     "Does elevated CO2 change the length of the canopy season, including autumn senescence?",
     "A spring-and-autumn curve model is implemented and tested on simulated seasons. A reference panel is now in view, "
     "so new imagery can be put in absolute reflectance.",
     "Continuous imagery from leaf-out to leaf-fall (one full season), with the reference panel in view.",
     "The main research paper: a full-season, sub-daily hyperspectral record of CO2 effects on a mature forest canopy, "
     "comparable with satellite data."),
    ("4", "Canopy chlorophyll and photosynthetic capacity", "Needs site measurements",
     "Does red-edge chlorophyll from the towers track photosynthetic capacity and canopy warming under elevated CO2?",
     "Red-edge indices are shown to be chlorophyll-driven (S1) and follow a clear seasonal course.",
     "Site leaf gas exchange or photosynthetic capacity, leaf nitrogen and chlorophyll samples, and canopy temperature "
     "for the same period.",
     "A joint paper linking canopy spectra to the site's physiology measurements."),
    ("5", "Early detection of oak powdery mildew", "Needs summer imagery and surveys",
     "Can the tower cameras detect powdery mildew on oak before it is visible?",
     "Simulation suggests red-band ratios respond at roughly 5–15% infected canopy (S5).",
     "Summer imagery and field infection surveys on the same trees.",
     "An applied paper on camera-based disease monitoring."),
]
