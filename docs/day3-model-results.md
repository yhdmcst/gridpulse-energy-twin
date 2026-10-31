# Machine-Learning Forecast Results

## Forecasting boundary

The forecasting experiment is a **one-step-ahead causal forecast**: the model predicts net load at time `t` using calendar features and measurements available no later than `t - 1 hour`. It is not yet a direct 24-hour-ahead forecast, because no future weather or renewable-generation forecast has been introduced. The report and dashboard distinguish these forecast horizons.

## Feature policy

All observed quantities are shifted before use. The feature set includes:

- hourly, weekday, weekend, and month seasonality;
- net-load lags at 1, 24, and 168 hours;
- rolling 24-hour net-load mean and standard deviation computed from prior observations only;
- solar and wind lags at 1 and 24 hours.

## Frozen comparison window

The validation boundary (`2019-01-12 20:00 UTC`) and test boundary (`2019-08-09 21:00 UTC`) exactly match the baseline evaluation. Feature availability reduces training rows to 34,875 but leaves all 5,017 validation and 10,035 test rows available.

## Executed test results

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | sMAPE (%) | MAE improvement vs. seasonal naive |
|---|---:|---:|---:|---:|---:|
| Seasonal naive 24-hour | 8,760.67 | 11,465.88 | 34.67 | 30.08 | Reference |
| Random Forest | 1,126.88 | 1,501.19 | 4.69 | 4.42 | 87.14% |
| Histogram Gradient Boosting | 1,019.87 | 1,354.19 | 4.54 | 4.12 | 88.36% |

Histogram Gradient Boosting is the selected model based on validation error; it also has the lowest test error among these models under the frozen protocol. Its validation permutation-importance output ranks `net_load_lag_1h` first, followed by time-of-day terms and solar lag features. This is an association-based diagnostic, not a causal explanation.

## Calibrated prediction interval

For each model, a symmetric interval is calibrated only from the validation absolute residuals. The reported radius is the 90th percentile of validation absolute error, then applied unchanged to the frozen test predictions. This is a simple empirical uncertainty interval, not a probabilistic guarantee. The generated prediction CSVs include lower and upper bounds, while `day3_model_metrics.json` reports validation/test coverage and mean width.

## Renewable-feature ablation

Removing solar and wind lag features from the selected model increases test MAE from 1,019.87 MW to 1,120.04 MW, a deterioration of 100.17 MW. This provides initial, dataset-specific evidence that prior renewable observations add predictive information beyond the net-load lag and calendar features.

## Artifacts

- Metrics: `outputs/models/day3_model_metrics.json`
- Test predictions: `outputs/models/*_test_predictions.csv`
- Validation permutation importance: `outputs/models/*_validation_permutation_importance.csv`
- Saved models: `outputs/models/*.joblib`

## Scope and limitations

The evaluation uses one Germany-wide historical data release and has no real fault labels. Before a 24-hour product claim, the project needs an explicit multi-horizon strategy and inputs that would actually be known at issuance time (for example, weather forecasts or recursively generated estimates). Risk thresholds must be calibrated on the existing validation period only.
