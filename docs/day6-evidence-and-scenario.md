# Forecast Ablations and Storage Planning Results

## Fixed experimental boundary

All forecast, interval, alert, and storage-policy evaluation uses the frozen baseline test start: `2019-08-09 21:00 UTC`. The original experiment freezes thresholds, model configurations, interval radii and policy weights. The later supplemental analysis was designed after original test results were known; it is exploratory even though its calibration uses only validation data. No supplemental winner replaces the original result.

## Feature ablations on the frozen test period

| Variant | Test MAE (MW) | Change vs. full model |
|---|---:|---:|
| Full histogram gradient boosting | 1,019.87 | Reference |
| Without renewable lags | 1,120.04 | +100.17 |
| Without rolling statistics | 1,035.03 | +15.16 |
| Calendar only | 9,160.43 | +8,140.56 |

Recent net-load history carries most predictive information in this dataset; renewable lags add measurable value; rolling statistics add a smaller positive contribution. These are dataset-specific predictive observations, not universal causal claims.

## Direct 1--6 Hour Planning Forecasts

The risk-aware storage policy uses six direct models. For a target at `t+h`, the `h`-hour model uses observations no later than `t`; all six models retain validation-calibrated 90% empirical intervals. This creates a causally available intraday planning window without claiming a direct 24-hour forecast.

| Horizon | Test MAE (MW) | Test interval coverage (%) |
|---|---:|---:|
| 1 hour | 1,019.87 | 86.92 |
| 2 hours | 1,898.34 | 87.07 |
| 3 hours | 2,650.71 | 86.50 |
| 4 hours | 3,333.81 | 86.37 |
| 5 hours | 3,934.67 | 86.87 |
| 6 hours | 4,420.12 | 85.65 |

Longer horizons are less accurate in this public-data prototype without forecast-time weather or renewable-production input. The original policy takes the maximum interval radius across horizons. Because those radii are fixed and the six-hour radius exceeds the normalization scale, its uncertainty contribution is constant and saturated. Time variation in the original reserve is mainly from forecast severity.

## Same-Information Fixed-Reserve Ablation and Risk-Aware Policy

The original one-step fixed policy is retained only as a legacy reference. The primary ablation runs both policies through the same six-hour direct forecasts and rolling linear min-max problem. The fixed control keeps 30% SOC; the new control starts at 15%, adds up to 20 percentage points for forecast severity and up to 15 points for interval width, and caps at 60%. Both execute only the current first action before forecasts and reserve targets are recomputed.

Both policies use initial SOC of 50%, 92% charge/discharge efficiency, maximum discharge of 2,500 MW, maximum charge of 1,250 MW, and a capacity/24-hour conservative discharge limit. Neither represents a real dispatch system.

## Executed Frozen-Test Comparison

| Capacity | Same-information fixed P95 (MW) | Risk-aware P95 (MW) | Difference (MW) | Fixed / risk-aware equivalent full cycles |
|---|---:|---:|---:|---:|
| 10,000 MWh | 54,200.8 | 54,188.3 | -12.5 | 0.78 / 0.84 |
| 20,000 MWh | 54,185.0 | 54,150.2 | -34.9 | 0.81 / 0.86 |
| 30,000 MWh | 54,177.4 | 54,077.7 | -99.7 | 0.84 / 0.89 |

The risk-aware policy solves optimally for 100% of evaluated planning windows. It lowers controlled P95 at all three capacities and retains a slightly higher minimum SOC than the fixed 30% reserve. Its theoretical throughput is also slightly higher, so the evidence supports a P95 and reserve tradeoff rather than a universal cycle-reduction claim. Neither direct-forecast policy reduces the single highest actual test-period net load; both have a full-period maximum reduction of zero.

## Interpretation Boundary

Supplemental component and power experiments, including terminal SOC and reserve shortfalls, are detailed in `docs/risk-aware-control.md` and `outputs/scenarios/reserve_ablation_metrics.json`. The reserve floor is softened to available SOC, and no terminal-SOC equality is enforced. These limits prevent long-run energy or economic conclusions from short-run P95 differences.

Separate daily energy-restoration results are in `outputs/scenarios/daily_energy_balance_metrics.json`. Over 417 complete matched days, both balanced policies end at their initial 50% SOC and respect the common hard 15% floor. Dynamic P95 is 15.09 MW lower and throughput 59,996.65 MWh lower than fixed, but fixed improves the highest peak while dynamic does not. Balanced mode caps SOC at 50% and permits charging at any hour. Recovery load and losses count. Independent daily resets, changed charging permissions, advisory reserve targets and post-test exploratory design limit interpretation; these results do not replace the original continuous backtest.

The outputs are theoretical historical scenarios only. They do not demonstrate actual cost saving, carbon reduction, battery lifetime, grid reliability improvement, feasible market operation, or deployed storage control. The simulation omits network limits, battery degradation, price signals, reserve obligations, weather forecasts, and operator approval.

## Outputs

- Direct forecast implementation: `src/multihorizon.py`
- Direct forecast runner: `scripts/train_multihorizon.py`
- Storage policies: `src/scenarios.py`
- Scenario runner: `scripts/run_storage_scenario.py`
- Detailed formulation: `docs/risk-aware-control.md`
- Metrics: `outputs/models/multihorizon_metrics.json`, `outputs/scenarios/storage_scenario_metrics.json`
