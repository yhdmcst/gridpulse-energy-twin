# Forecasting Baseline Protocol

## Target

The target is `net_load_mw` in the normalized Germany OPSD table.

## Baseline

The baseline prediction for timestamp `t` is the observed net load at `t - 24 hours`, joined by UTC timestamp:

```text
prediction(t) = net_load(t - 24 hours)
```

This is a seasonal-naive baseline for an hourly series with a daily cycle. It provides a reproducible reference for the model comparison.

## Missing-value policy

The source table is kept as an hourly timeline. An evaluation row is valid only when both the current target and the target exactly 24 hours earlier are present. Missing rows are excluded from the metric calculation and counted in the experiment metadata; they are not interpolated for this baseline.

## Chronological split

The evaluable rows are split once, in timestamp order:

- first 70%: training period, used only to establish the protocol;
- next 10%: validation period for model selection and threshold calibration;
- final 20%: untouched test period, used for the reported baseline result.

The seasonal-naive model has no fitted parameters, but the same time boundaries are retained for model comparisons.

## Metrics

- MAE in MW;
- RMSE in MW;
- MAPE in percent, excluding zero actual targets;
- sMAPE in percent as a complementary scale-free metric.

Run from the project root:

```powershell
python scripts/run_baseline.py
```

Outputs:

- `outputs/metrics/seasonal_naive_metrics.json`
- `outputs/predictions/seasonal_naive_test.csv`
