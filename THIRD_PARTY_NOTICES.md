# Third-Party Notices

This file records external material used by GridPulse. It is intentionally separate from the original project implementation.

## Lucide icons

- Bundled version: Lucide 0.468.0, locally vendored in `docs/showcase/lucide.min.js`.
- Source: https://github.com/lucide-icons/lucide
- License: ISC; complete upstream license and notices are retained in `docs/showcase/LUCIDE-LICENSE.txt`.
- Use: playback, chapter, theme, and navigation controls in the static HTML/SVG presentation.
- The presentation has no runtime dependency on a public icon CDN.

## SmartBuild-AI

- Historical local reference: `references/SmartBuild-AI/` in the original workspace; not bundled in this public release
- Source: https://github.com/Mahmoud-Faraj/SmartBuild-AI
- License: MIT, as provided in that repository's `LICENSE` file
- Intended use: engineering reference for modular organization, chronological evaluation, residual screening, tests, and cautious dashboard wording
- Not used as: GridPulse data, reported results, submission prose, UI copy, or copied implementation

## Competition materials

- Source: https://www.aicomp.cn/tracks/4726.html
- Relevant material: 2026 AIC AI+Energy algorithm theme, Technology Innovation Group rules and report outline
- Intended use: competition requirements and evaluation alignment

## Dataset records

The separately locked 2022 temporal replication uses the same Energy-Charts.info / Fraunhofer ISE public_power API under its declared CC BY 4.0 license. Source URL, download timestamp, provider metadata and raw/normalized SHA-256 are retained in `outputs/validation/independent_2022_data_quality.json`; methods and full results appear in `docs/later-year-validation.md`. It is a separate historical snapshot and does not overwrite the 2021 release.

### Open Power System Data - Time series

- Canonical URL: https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
- Dataset DOI: https://doi.org/10.25832/time_series/2020-10-06
- Publisher citation: Open Power System Data. 2020. *Data Package Time series*. Version 2020-10-06.
- Access date: 2026-09-29
- Region and period used: Germany, 2015-01-01 through 2020-09-30, UTC-normalized
- Frequency and unit: hourly; load and generation fields in MW
- Fields used: `DE_load_actual_entsoe_transparency`, `DE_solar_generation_actual`, `DE_wind_generation_actual`
- Transformation: source field renaming, numeric coercion, timestamp sorting, removal of one all-empty placeholder row, and derived `net_load_mw`
- Terms note: retain the dataset's published attribution and verify the release's current data-use terms again before final public submission; the OPSD website states its website text is CC BY 4.0, while the underlying primary-data terms may vary by source.

## Additional temporal holdout data

The 2021 German temporal validation uses Energy-Charts.info (Fraunhofer ISE), public power API: https://api.energy-charts.info/v2/public_power . The API declares CC BY 4.0 with attribution to Energy-Charts.info. Selected fields are Load, Solar, Wind onshore and Wind offshore; GridPulse aggregates complete quarter-hour MW intervals to hourly means and computes net load. This transformation is separate from the original OPSD dataset. The source URL, license statement, dates and checksums are retained in `outputs/validation/independent_2021_data_quality.json`. Historical data can be revised by the provider. See `docs/independent-validation.md` for measurement-time and reuse boundaries.
