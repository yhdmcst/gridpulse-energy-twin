from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts import predict_periods
from src.multihorizon import build_planning_panel
from src.scenarios import (
    simulate_fixed_reserve_receding_dispatch,
    simulate_risk_aware_dispatch,
    simulate_storage_dispatch,
)


CAPACITIES_MWH = (10000.0, 20000.0, 30000.0)


def comparison(fixed: dict, optimized: dict) -> dict[str, float]:
    return {
        "optimized_minus_fixed_max_reduction_mw": float(
            optimized["theoretical_max_reduction_mw"] - fixed["theoretical_max_reduction_mw"]
        ),
        "optimized_minus_fixed_controlled_p95_mw": float(
            optimized["controlled_p95_net_load_mw"] - fixed["controlled_p95_net_load_mw"]
        ),
        "optimized_minus_fixed_throughput_mwh": float(
            optimized["throughput_mwh"] - fixed["throughput_mwh"]
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs/scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((root / "outputs/alerts/alert_evaluation.json").read_text(encoding="utf-8"))["thresholds"]
    validation, _ = predict_periods(
        root / "data/processed/net_load_hourly.csv", root / "outputs/models/hist_gradient_boosting.joblib"
    )
    charge_threshold = float(validation["predicted_net_load_mw"].quantile(0.25))
    emergency_threshold = float(validation["predicted_net_load_mw"].quantile(0.99))
    fixed_predictions = pd.read_csv(
        root / "outputs/models/hist_gradient_boosting_test_predictions.csv", parse_dates=["timestamp"]
    ).rename(columns={"hist_gradient_boosting_prediction_mw": "predicted_net_load_mw"})
    direct_predictions = pd.read_csv(
        root / "outputs/models/direct_horizon_test_predictions.csv", parse_dates=["issued_timestamp", "timestamp"]
    )
    planning_panel = build_planning_panel(direct_predictions, planning_window_hours=6)

    fixed_sensitivity: dict[str, dict] = {}
    direct_fixed_sensitivity: dict[str, dict] = {}
    optimized_sensitivity: dict[str, dict] = {}
    legacy_comparisons: dict[str, dict] = {}
    fair_comparisons: dict[str, dict] = {}
    for capacity_mwh in CAPACITIES_MWH:
        label = f"storage_{int(capacity_mwh)}mwh"
        fixed_result, fixed_metrics = simulate_storage_dispatch(
            fixed_predictions,
            peak_threshold_mw=float(thresholds["peak_prediction_threshold_mw"]),
            capacity_mwh=capacity_mwh,
            charge_threshold_mw=charge_threshold,
            emergency_threshold_mw=emergency_threshold,
            reserve_fraction=0.30,
            energy_smoothing_hours=24.0,
        )
        direct_fixed_result, direct_fixed_metrics = simulate_fixed_reserve_receding_dispatch(
            planning_panel,
            peak_threshold_mw=float(thresholds["peak_prediction_threshold_mw"]),
            capacity_mwh=capacity_mwh,
            fixed_reserve_fraction=0.30,
            charge_threshold_mw=charge_threshold,
            energy_smoothing_hours=24.0,
        )
        optimized_result, optimized_metrics = simulate_risk_aware_dispatch(
            planning_panel,
            peak_threshold_mw=float(thresholds["peak_prediction_threshold_mw"]),
            capacity_mwh=capacity_mwh,
            charge_threshold_mw=charge_threshold,
            energy_smoothing_hours=24.0,
        )
        fixed_metrics["throughput_mwh"] = float(fixed_result["storage_dispatch_mw"].abs().sum())
        fixed_metrics["equivalent_full_cycles"] = float(fixed_result["storage_dispatch_mw"].abs().sum() / (2 * capacity_mwh))
        direct_fixed_metrics["throughput_mwh"] = float(direct_fixed_result["storage_dispatch_mw"].abs().sum())
        direct_fixed_metrics["equivalent_full_cycles"] = float(
            direct_fixed_result["storage_dispatch_mw"].abs().sum() / (2 * capacity_mwh)
        )
        fixed_sensitivity[label] = fixed_metrics
        direct_fixed_sensitivity[label] = direct_fixed_metrics
        optimized_sensitivity[label] = optimized_metrics
        legacy_comparisons[label] = comparison(fixed_metrics, optimized_metrics)
        fair_comparisons[label] = comparison(direct_fixed_metrics, optimized_metrics)
        fixed_result.to_csv(output_dir / f"{label}_fixed_timeseries.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        direct_fixed_result.to_csv(
            output_dir / f"{label}_direct_fixed_reserve_timeseries.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
        )
        optimized_result.to_csv(output_dir / f"{label}_risk_aware_timeseries.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    report = {
        "scenario_type": "Theoretical storage backtest; not deployment evidence, savings claim, or dispatch authorization.",
        "time_resolution": "1 hour",
        "decision_information": "Legacy fixed policy uses one-step point forecasts. Fair storage ablation uses the same direct 1--6 hour forecasts and rolling optimizer for both policies; only the SOC reserve rule differs.",
        "fixed_policy": {
            "name": "fixed_reserve_rule_baseline",
            "assumptions": [
                "Initial state of charge is 50% of nominal capacity.",
                "Maximum discharge power is 2,500 MW; maximum charge power is 1,250 MW.",
                "A capacity/24-hour effective discharge limit is used as a conservative daily energy-spreading proxy.",
                "Charge and discharge efficiency are each assumed to be 92%.",
                "Thirty percent of capacity is reserved during ordinary peak alerts.",
            ],
            "sensitivity": fixed_sensitivity,
        },
        "risk_aware_optimization": {
            "name": "uncertainty_aware_receding_horizon_linear_program",
            "planning_window_hours": 6,
            "objective": "Minimize the maximum direct-forecast upper net load across the next six hours, with a small throughput penalty.",
            "constraints": [
                "Hourly charge/discharge power bounds and storage capacity bounds.",
                "SOC dynamics with 92% charge and discharge efficiency.",
                "Charging only when the one-hour lower forecast is below the validation-calibrated low-load threshold.",
                "Discharging only when the horizon upper forecast reaches the validation-calibrated peak threshold.",
                "Dynamic reserve target = base reserve + forecast-severity contribution + interval-width contribution, capped at 60% of capacity.",
                "Only the first action is executed before forecasts and reserve targets are recomputed.",
            ],
            "sensitivity": optimized_sensitivity,
        },
        "direct_fixed_reserve_baseline": {
            "name": "fixed_30pct_reserve_receding_horizon_linear_program",
            "purpose": "Fair ablation baseline: identical direct forecasts, rolling objective, power, SOC and efficiency constraints; only the reserve rule is fixed at 30%.",
            "sensitivity": direct_fixed_sensitivity,
        },
        "comparison": {
            "legacy_one_step_fixed_vs_risk_aware": legacy_comparisons,
            "fair_same_information_fixed_reserve_vs_risk_aware": fair_comparisons,
        },
        "calibration": {
            "charge_forecast_threshold_mw": charge_threshold,
            "fixed_emergency_forecast_threshold_mw": emergency_threshold,
            "peak_threshold_mw": float(thresholds["peak_prediction_threshold_mw"]),
            "calibration_period": "2019-01-12 20:00 UTC to 2019-08-09 20:00 UTC",
        },
        "limitations": [
            "No network constraints, battery degradation, market prices, reserve obligations, or real dispatch authorization are modeled.",
            "Results depend on historical public data and direct forecast accuracy; they do not demonstrate field reliability or realized savings.",
        ],
    }
    (output_dir / "storage_scenario_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
