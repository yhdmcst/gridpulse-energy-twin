# Data Source Decision

## Selected source

GridPulse uses the Open Power System Data (OPSD) time-series release:

- URL: https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
- Release path: `time_series/2020-10-06`
- Region: Germany (`DE`)
- Frequency: 60 minutes
- Time basis: source includes UTC and CET/CEST timestamps; GridPulse stores UTC for reproducibility
- Selected fields: `DE_load_actual_entsoe_transparency`, `DE_solar_generation_actual`, `DE_wind_generation_actual`
- License and citation: verify the release metadata and license terms before public submission; the source and access path are recorded here for reproducibility.

## Why this source was selected

It contains one region with actual load, solar generation, and wind generation in a single hourly table. The fields can be aligned without joining separate providers, and the release is downloadable from a stable public OPSD endpoint.

## Deliberately excluded fields

Forecast, capacity, profile, price, and distribution-grid subregion fields are excluded from the normalized modeling table.

## Reproduction

From the repository root:

```powershell
python scripts/build_dataset.py
python scripts/quality_report.py
python scripts/eda_report.py
```

The raw file is saved under `data/raw/`, the normalized table under `data/processed/`, and the quality/EDA outputs under `outputs/`.
