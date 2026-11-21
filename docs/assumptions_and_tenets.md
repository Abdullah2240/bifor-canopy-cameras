# Assumptions and tenets

This is the list of things the BIFoR camera analysis takes as given. Whenever the accuracy or legitimacy of the work is questioned, come back here first. Each tenet has a basis, a confidence level, what depends on it, and how to swap it if better information arrives. Change log at the bottom.

Confidence:
- **fact:** measured or confirmed.
- **guidance:** stated by the site lead, taken as authoritative.
- **assumption:** a working guess, used until replaced.

---

## T1. Treatment labels: A6 = eCO2, A5 = ambient (assumption)
- **What we know:** both cameras view FACE rings fitted with CO2 pylons. Nothing in the data or from the team identifies which ring receives elevated CO2.
- **Basis for the guess:** a meta-analysis of 624 records from 39 studies found that eCO2 brings leaf-out forward by 3.00 days on average, and plant maturity by 4.09 days (Gao et al. 2026). On the 10 Mar – 3 Jun record both cameras reach NDVI half-way on 3 May, but A6 completes its green-up sooner (80% reached 3 days before A5; fitted mid-point earlier, though the A5 fit is unreliable). So A6 is labelled eCO2.
- **Supporting context at BIFoR itself:**
  - Overall leaf reflectance of the 180-year-old *Q. robur* is unchanged under eCO2, but PSRI is higher, which points to a pigment-ratio shift (Jones et al. 2025).
  - eCO2 raises oak canopy temperature, a mechanism that could advance spring development (Hagan Brown et al. 2025).
- **Limit (must appear wherever T1 is used):** the label was picked from the observed result. The A6 − A5 difference therefore **cannot be presented as evidence** of an eCO2 effect. It can only be compared with the published effect size, and the comparison is either consistent or not consistent.
- **Second limit:** the cameras view at different angles (A5 oblique, A6 looking down). Viewing angle can shift green-up dates by itself, so the camera difference is treatment and angle combined.
- **Depends on it:** the eCO2 comparison (Lane A′) and every "eCO2 vs ambient" label in the app and report.
- **To swap:** change `[treatment]` in `bifor.toml`. The app and report read the labels from there.

## T2. Valid record: 10 Mar 2026 10:31 UTC to 3 Jun 2026 06:17 UTC (guidance + fact)
- **Basis:** the site lead states that data before March 2026 is bad (D9, D11). 10 March is the first 16-bit cube file (D12). No footage exists after 3 June (D3).
- **Depends on it:** every result.
- **To swap:** change `[window] start_utc` in `bifor.toml` and rerun.

## T3. Stands are comparable (guidance)
- **Basis:** both cameras view the same species (*Q. robur* canopy) in the same forest (D2).
- **Depends on it:** treating the A5–A6 difference as treatment plus angle, rather than different trees.

## T4. No camera changes between March and June (assumption)
- **Basis:** nothing is known either way (D7). The 1–2 px drift is corrected as normal mount movement.
- **Depends on it:** a continuous series with no event breaks.

## T5. Relative reflectance only (fact)
- **Basis:** the cameras auto-expose and no reference panel was in view during the record. A panel was installed afterwards (D8).
- **Consequence:** use ratios and normalised indices only. No absolute reflectance, no comparison with satellite products.

## T6. Imagery only, no ground truth (guidance)
- **Basis:** ground measurements come only after the imagery proves useful on its own (D10).
- **Consequence:** results are validated internally: the two cameras agree, the indices agree, and the index-free spectral clustering agrees. There is no ground check.

## T7. Timestamps are UTC (fact)
- **Basis:** frames continue through 01:00–02:00 on 29 Mar 2026, an hour that does not exist in UK local time.

---

## How the final outputs use these tenets
- The app and the report state T1–T7 on a dedicated page or section.
- Findings are written up as observed results plus methods.
- The closing section lists **open questions and curiosities** to take further, not conclusions.

## References
- Gao, Y., Ma, L., Zhao, G., et al. (2026). Elevated CO2 advances vegetation phenology. *Functional Ecology*, 40(9), 2946–2958. https://doi.org/10.1111/1365-2435.70402
- Jones, A. L., Gardner, A., Hayes, F., Pfrang, C., & Jeffers, E. S. (2025). The effect of elevated CO2 on hyperspectral leaf reflectance in mature trees. *Trees*, 39(4). https://doi.org/10.1007/s00468-025-02650-w
- Hagan Brown, W., Gloor, E., Fyfe, R., MacKenzie, A. R., Harper, N. J., Ganderton, P., et al. (2025). Elevated CO2 increases the canopy temperature of mature *Quercus robur* (pedunculate oak). *Global Change Biology*, 31, e70565. https://doi.org/10.1111/gcb.70565

## Change log
- 2026-11-21: T1 basis restated on the 10 Mar window: same half-way date, A6 finishes sooner.
- 2026-11-14: created. T1 from D1; T2 from D3, D9, D11 and D12; T3 from D2; T4 from D7; T5 from D8; T6 from D10.
