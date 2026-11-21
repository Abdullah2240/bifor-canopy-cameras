# Architecture

## Data flow

```mermaid
flowchart LR
  subgraph SRC["Sources (read-only)"]
    RAW["D:\\BIFOR<br/>A5 + A6 frames<br/>png / tiff / cube+raw"]
    CAL["calib/A5,A6_calib.json<br/>imec m0 24x25"]
    WX["Open-Meteo archive<br/>hourly cloud, rain, radiation"]
    CR["Crossref API"]
  end
  RAW --> SCAN["scan<br/>catalog.py + pvlib sun"] --> CAT[("catalog.parquet")]
  CAT --> DARK["master darks<br/>darkest nights per camera+bit depth"] --> DK[("darks/*.npy")]
  CAT --> PROC
  DK --> PROC
  CAL --> PROC
  ROIS[("rois/*.geojson<br/>exclude + ref .npy")] --> PROC
  subgraph PROC["process_day (parallel, per camera-day)"]
    direction TB
    P1["pass 1: read once, 800 nm band,<br/>masked pylon registration"] --> MED["day-median offset"]
    MED --> P2["pass 2: demosaic, dark, saturation,<br/>m0 correction, shifted ROI masks"]
    P2 --> QAM["QA metrics<br/>sat, light ratio, sharpness, shift"]
    P2 --> SPEC["ROI spectra<br/>median p10 p90 n"]
    SPEC --> IDX["index registry<br/>resolved by filter slot"]
  end
  QAM --> FR[("frames/")]
  SPEC --> RS[("roi_spectra/")]
  IDX --> IN[("indices/")]
  WX --> WXT[("weather.parquet")]
  FR --> VIEW["frames_view<br/>light_jump + weather + clearness<br/>QA rule applied at read time"]
  WXT --> VIEW
  VIEW --> DAILY["daily: QA-pass frames<br/>within 2 h of solar noon<br/>median, IQR, n + Hampel"]
  IN --> DAILY
  DAILY --> DT[("daily.parquet")]
  DT --> FIT["spring logistic fit<br/>block bootstrap CIs"] --> PH[("phenology.parquet")]
  CR --> CIT["citation check"] --> CC[("citations.csv")]
  DT --> FIGS["figures + Streamlit app"]
  PH --> FIGS
  FR --> FIGS
  SIM["simulations S1-S7<br/>PROSAIL, noise, power"] --> FIGS
```

## Data model

```mermaid
erDiagram
  CATALOG ||--o{ FRAMES : "frame_id"
  FRAMES ||--o{ ROI_SPECTRA : "frame_id"
  FRAMES ||--o{ INDICES : "frame_id"
  INDICES }o--|| INDEX_REGISTRY : "index"
  INDICES }o--o{ DAILY : "aggregated"
  DAILY }o--|| PHENOLOGY : "camera, roi, index"
  WEATHER ||--o{ FRAMES : "hour"
  CATALOG {
    string frame_id PK
    string camera
    datetime ts_utc
    string format
    string read_path
    float sun_elev
    float min_from_noon
  }
  FRAMES {
    string frame_id PK
    int bit_depth
    float dn_mean
    float sat_frac
    float light_ratio
    float sharpness
    float shift_dy
    float shift_dx
    float day_shift_dy
    float day_shift_dx
    string error
    string code_v
    string roi_v
    string calib_v
  }
  ROI_SPECTRA {
    string frame_id FK
    string roi
    float band_nm
    float median
    float p10
    float p90
    int n_valid
  }
  INDICES {
    string frame_id FK
    string roi
    string index
    float value
  }
  DAILY {
    date date
    string camera
    string roi
    string index
    float median
    float q25
    float q75
    int n_frames
    bool outlier
  }
  PHENOLOGY {
    string camera
    string roi
    string index
    float sos20
    float mid50
    float mat80
    float mid50_lo
    float mid50_hi
    float rmse
  }
```

## Stage order and resume

```mermaid
flowchart TB
  A["bifor scan"] --> B["audit scripts"] --> C["bifor sample + process --use-sample (pilot)"]
  C --> D["fix + version bump"] --> E["bifor process (full, incremental)"]
  E --> F["bifor weather"] --> G["bifor daily"] --> H["bifor fit"] --> I["bifor figures"]
  E -. "crash or restart: rerun the same command, finished frames are skipped" .-> E
```

Every row carries `code_v`, `roi_v` and `calib_v`. Readers keep only rows that match the current config. Bumping `versions.code` in `bifor.toml` therefore reprocesses everything without deleting old outputs.
