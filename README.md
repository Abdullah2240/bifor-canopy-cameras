# bifor-pipeline

Turns the BIFoR-FACE tower-camera frames into a checked seasonal canopy record.

- **Input:** `D:\BIFOR`, two imec 5×5 snapshot cameras (A5, A6). Valid record 10 Mar – 3 Jun 2026 (`[window]` in `bifor.toml`). The pipeline only reads from it.
- **Output:** Parquet tables and figures in `D:\BIFOR_work`; a self-contained dashboard bundle in `app/data`; an HTML report in `docs/`.

## Run

```
python -m pytest -q                              # tests
python -m bifor.cli scan                         # catalog of every frame (seconds)
python scripts/audit.py                          # data audit tables + contact sheets
python -m bifor.cli sample --every-days 14       # pilot subset
python -m bifor.cli process --use-sample         # pilot
python -m bifor.cli process --workers 6          # all frames; rerun to resume
python -m bifor.cli weather
python -m bifor.cli daily
python -m bifor.cli fit
python -m bifor.cli figures
python -m bifor.cli resolve-bands                # index_resolution.csv
python -m bifor.cli check-citations              # citations.csv (Crossref)
cd simulations && python s1_sensitivity.py       # ... through s7_qa_detection.py
python scripts/patterns.py && python scripts/focus_areas.py
python scripts/lane_a_prime.py                   # camera comparison, joint bootstrap
python scripts/export_app_data.py                # app/data bundle + summary.json
python scripts/build_professor_report.py         # docs/BIFoR_canopy_cameras_2026.html
streamlit run app/streamlit_app.py               # dashboard (reads app/data only)
streamlit run app/explorer_local.py              # frame inspector + QA labelling (needs D:)
```

## Dashboard deployment

Streamlit Community Cloud: entry file `app/streamlit_app.py`, Python 3.12, dependencies `app/requirements.txt`.
The app reads only `app/data`, so no raw data is needed on the server.

## Layout

| Path | What |
|---|---|
| `bifor.toml` | Paths, site, QA thresholds, version tags |
| `bifor/reader.py` | Any format to a 24-band corrected cube: demosaic, dark, saturation, m0 |
| `bifor/catalog.py` | Drive scan |
| `bifor/qa.py` | Sun position, masked pylon registration, light jump, sharpness, pass rule |
| `bifor/roi.py` | ROI polygons, pylon/sky exclusion, per-ROI band statistics |
| `bifor/indices.py` | Index registry, resolved by filter slot so both cameras use the same filters |
| `bifor/pipeline.py` | Stages, master darks, per camera-day parallel processing |
| `bifor/timeseries.py` | Daily medians, Hampel filter, logistic fit with block bootstrap, power |
| `bifor/figures.py` | Data figures |
| `simulations/` | S1–S7 |
| `app/` | Dashboard (`streamlit_app.py`, `content.py`), local explorer, data bundle |
| `scripts/` | Audit, patterns, camera comparison, export, report |
| `docs/` | Audit, architecture, results, assumptions and tenets, report |

## Conventions

- **Versioning:** every output row carries `code_v`, `roi_v` and `calib_v`. Bump `versions.code` to reprocess everything.
- **QA:** thresholds are applied when tables are read, so changing one needs no reprocessing.
- **Reflectance:** all values are relative reflectance. The cameras auto-expose and there was no panel in view.
- **Assumptions:** see `docs/assumptions_and_tenets.md`. Treatment labels (`[treatment]`) are a working assumption.
