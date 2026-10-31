# Baseline Results

The seasonal-naive baseline was executed on `data/processed/net_load_hourly.csv` using the protocol in `day2-baseline-protocol.md`.

## Frozen split

- Training period: through `2019-01-12 19:00 UTC`, 35,121 evaluable rows.
- Validation period: `2019-01-12 20:00 UTC` through `2019-08-09 20:00 UTC`, 5,017 rows.
- Untouched test period: `2019-08-09 21:00 UTC` through `2020-09-30 23:00 UTC`, 10,035 rows.
- Total evaluable rows: 50,173. The remaining normalized rows lack either the current target or the exact 24-hour lag target.

## Metrics from the executed run

| Split | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|
| Validation | 8,061.04 | 10,470.61 | 28.44 | 25.26 |
| Test | 8,760.67 | 11,465.88 | 34.67 | 30.08 |

The machine-readable record is `outputs/metrics/seasonal_naive_metrics.json`; test predictions are in `outputs/predictions/seasonal_naive_test.csv`.

## Interpretation boundary

These numbers establish a reproducible reference line only. They do not prove that a future model will generalize, and they do not imply any operational or economic benefit. Test error is higher than validation error; robustness diagnostics are documented in `rolling-backtest.md`.
