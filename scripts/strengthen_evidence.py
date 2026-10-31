"""Supplemental baselines and validation-only hourly interval / reserve ablations.

Frozen published storage results are not overwritten. These experiments do not
select a replacement policy by test performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import regression_metrics
from src.features import model_frame
from src.models import frozen_split_for_features
from src.multihorizon import FEATURE_COLUMNS, VALIDATION_START, TEST_START, build_direct_horizon_frame, build_planning_panel
from src.scenarios import simulate_fixed_reserve_receding_dispatch, simulate_risk_aware_dispatch


def main():
    raw = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv", parse_dates=["timestamp"])
    _, _, test, _ = frozen_split_for_features(raw, model_frame(raw))
    predictions = pd.read_csv(ROOT / "outputs/models/hist_gradient_boosting_test_predictions.csv", parse_dates=["timestamp"])
    test = test.merge(predictions[["timestamp", "hist_gradient_boosting_prediction_mw"]], on="timestamp", validate="one_to_one")
    baseline = {
        "protocol": "All models use the same frozen eligible test timestamps; no policy or model is selected using these test metrics.",
        "test_rows": len(test),
        "metrics": {name: regression_metrics(test.net_load_mw, test[column]) for name, column in {
            "persistence_1h": "net_load_lag_1h", "seasonal_24h": "net_load_lag_24h", "seasonal_168h": "net_load_lag_168h", "hist_gradient_boosting": "hist_gradient_boosting_prediction_mw"}.items()},
    }
    (ROOT / "outputs/models/reference_baseline_metrics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print("Shared-timestamp baselines completed", flush=True)

    direct = pd.read_csv(ROOT / "outputs/models/direct_horizon_test_predictions.csv", parse_dates=["issued_timestamp", "timestamp"])
    calibrated = direct.copy()
    hourly_radii = {}
    validation_max_radii = []
    intervals = {}
    for h in range(1, 7):
        frame = build_direct_horizon_frame(raw, h)
        validation = frame.loc[(frame.timestamp >= VALIDATION_START) & (frame.timestamp < TEST_START)].copy()
        artifact = joblib.load(ROOT / f"outputs/models/direct_horizon_{h}h.joblib")
        validation["error"] = np.abs(validation.net_load_mw - artifact["model"].predict(validation[list(FEATURE_COLUMNS)]))
        radii = validation.groupby(validation.timestamp.dt.hour)["error"].quantile(0.9)
        hourly_radii[str(h)] = {str(hour): float(radius) for hour, radius in radii.items()}
        validation_max_radii.extend(radii.tolist())
        mask = calibrated.horizon_hours.eq(h)
        row_radii = calibrated.loc[mask, "timestamp"].dt.hour.map(radii)
        calibrated.loc[mask, "lower_mw"] = calibrated.loc[mask, "prediction_mw"] - row_radii
        calibrated.loc[mask, "upper_mw"] = calibrated.loc[mask, "prediction_mw"] + row_radii
        subset = calibrated.loc[mask]
        intervals[str(h)] = {"test_coverage_pct": float(subset.net_load_mw.between(subset.lower_mw, subset.upper_mw).mean() * 100), "mean_width_mw": float((subset.upper_mw - subset.lower_mw).mean()), "hourly_radius_min_mw": float(radii.min()), "hourly_radius_max_mw": float(radii.max())}
    scale = max(validation_max_radii) * 1.10
    panel = build_planning_panel(calibrated, 6)
    metrics = json.loads((ROOT / "outputs/scenarios/storage_scenario_metrics.json").read_text())
    calibration = metrics["calibration"]
    options = {
        "fixed_30pct": None,
        "severity_only": {"uncertainty_weight": 0.0},
        "uncertainty_only": {"severity_weight": 0.0},
        "combined": {},
        "combined_half_power": {},
        "combined_double_charge_power": {},
    }
    results = {}
    for name, reserve in options.items():
        kwargs = dict(peak_threshold_mw=calibration["peak_threshold_mw"], capacity_mwh=30000., charge_threshold_mw=calibration["charge_forecast_threshold_mw"], max_power_mw=625. if name == "combined_half_power" else 2500., charge_power_mw=2500. if name == "combined_double_charge_power" else 1250.)
        if reserve is None:
            trace, summary = simulate_fixed_reserve_receding_dispatch(panel, **kwargs)
        else:
            trace, summary = simulate_risk_aware_dispatch(panel, reserve_options={"uncertainty_scale_mw": scale, **reserve}, **kwargs)
        results[name] = summary
        print(f"Supplemental reserve experiment completed: {name}", flush=True)
    report = {
        "protocol": "Exploratory, predeclared component/power comparisons on the frozen test period. Validation-only hourly calibration; no claim of deployment or test-independent policy selection. Published original storage results remain separate.",
        "hourly_interval_calibration": hourly_radii,
        "interval_evaluation": intervals,
        "uncertainty_scale_mw": scale,
        "uncertainty_scale_policy": "1.10 times the maximum validation hourly radius, fixed before test application",
        "test_planning_rows": len(panel),
        "experiments": results,
        "limitations": ["Terminal SOC is reported but not constrained to equal initial SOC.", "Reserve target is softened to available SOC; shortfall hours are reported.", "Only one region and period; no economic or reliability conclusion."],
    }
    (ROOT / "outputs/scenarios/reserve_ablation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
