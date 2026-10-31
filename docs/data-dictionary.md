# Data Dictionary

| Standard field | Unit | Source field | Meaning | Transformation |
|---|---:|---|---|---|
| `timestamp` | UTC timestamp | `utc_timestamp` | Observation time | Parsed as UTC and sorted ascending |
| `total_load_mw` | MW | `DE_load_actual_entsoe_transparency` | Germany actual electricity load | Numeric coercion; no imputation during normalization |
| `solar_generation_mw` | MW | `DE_solar_generation_actual` | Germany actual solar generation | Numeric coercion; no imputation during normalization |
| `wind_generation_mw` | MW | `DE_wind_generation_actual` | Germany actual wind generation | Numeric coercion; no imputation during normalization |
| `net_load_mw` | MW | Derived | Load not covered by selected solar and wind generation | `total_load_mw - solar_generation_mw - wind_generation_mw` |

## Interpretation notes

- The selected fields describe system-level Germany time series, not a single building or local feeder.
- Missing source values are retained as missing and reported by the quality checker; they are not silently interpolated.
- Negative values are reported as suspicious observations, not automatically deleted.
- UTC is used for modeling to avoid daylight-saving-time ambiguity. Local-time charts can be added later with an explicit timezone conversion.
