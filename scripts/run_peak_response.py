"""Exploratory peak-response ablation with validation-only trigger calibration.

Separate files preserve all published results. No threshold grid is searched on
the test peak; both single changes and the combined policy are reported.
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
from src.balanced_storage import simulate_balanced_storage
from src.multihorizon import (FEATURE_COLUMNS, VALIDATION_START, TEST_START,
                             build_direct_horizon_frame, build_planning_panel)


def interval_panel(direct: pd.DataFrame, supplemental: dict) -> pd.DataFrame:
    data = direct.copy()
    for h in range(1, 7):
        mask = data.horizon_hours.eq(h)
        radii = {int(k): v for k, v in supplemental["hourly_interval_calibration"][str(h)].items()}
        radius = data.loc[mask, "timestamp"].dt.hour.map(radii)
        data.loc[mask, "lower_mw"] = data.loc[mask, "prediction_mw"] - radius
        data.loc[mask, "upper_mw"] = data.loc[mask, "prediction_mw"] + radius
    return build_planning_panel(data, 6)


def validation_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in range(1, 7):
        frame = build_direct_horizon_frame(raw, h)
        valid = frame.loc[(frame.timestamp >= VALIDATION_START) & (frame.timestamp < TEST_START)]
        model = joblib.load(ROOT / f"outputs/models/direct_horizon_{h}h.joblib")["model"]
        rows.append(pd.DataFrame({
            "issued_timestamp": valid.issued_timestamp, "timestamp": valid.timestamp,
            "horizon_hours": h, "net_load_mw": valid.net_load_mw,
            "prediction_mw": model.predict(valid[list(FEATURE_COLUMNS)]),
            "lower_mw": 0., "upper_mw": 0.,
        }))
    return pd.concat(rows, ignore_index=True)


def evidence(trace: pd.DataFrame, summary: dict) -> dict:
    peak = trace.loc[trace.net_load_mw.idxmax()]
    controlled_peak = trace.loc[trace.controlled_net_load_mw.idxmax()]
    summary["max_reduction_mw"] = summary["baseline_peak_mw"] - summary["controlled_peak_mw"]
    summary["peak_event"] = {
        key: (value.isoformat() if isinstance(value, pd.Timestamp) else value.item() if isinstance(value, np.generic) else value)
        for key, value in peak.items()
    }
    summary["controlled_peak_timestamp"] = controlled_peak.timestamp.isoformat()
    # The complete maximum is primary; top-20 hours provide broader tail evidence.
    top = trace.nlargest(20, "net_load_mw")
    summary["top20_mean_reduction_mw"] = float((top.net_load_mw - top.controlled_net_load_mw).mean())
    summary["top20_improved_hours"] = int((top.controlled_net_load_mw < top.net_load_mw - 1e-6).sum())
    summary["load_increase_hours"] = int((trace.controlled_net_load_mw > trace.net_load_mw + 1e-6).sum())
    return summary


def main() -> None:
    out = ROOT / "outputs/scenarios"
    supplemental = json.loads((out / "reserve_ablation_metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((out / "daily_energy_balance_metrics.json").read_text(encoding="utf-8"))
    ordinary = json.loads((out / "storage_scenario_metrics.json").read_text(encoding="utf-8"))["calibration"]
    raw = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv", parse_dates=["timestamp"])
    validation = validation_predictions(raw)
    # Conventional upper-tail trigger, fixed from validation predictions only.
    threshold = float(validation.loc[validation.horizon_hours.eq(1), "prediction_mw"].quantile(.99))
    print(f"Validation-only q99 release threshold: {threshold:.2f} MW", flush=True)
    policies = {
        "fixed_upper": {"fixed_reserve_fraction": .30},
        "dynamic_upper": {},
        "dynamic_release_only": {"release_threshold_mw": threshold},
        "dynamic_point_only": {"planning_objective": "point"},
        "dynamic_peak_response": {"planning_objective": "point", "release_threshold_mw": threshold},
        "fixed_peak_response": {"fixed_reserve_fraction": .30, "planning_objective": "point", "release_threshold_mw": threshold},
    }
    test = pd.read_csv(ROOT / "outputs/models/direct_horizon_test_predictions.csv", parse_dates=["issued_timestamp", "timestamp"])
    periods = {}
    for period, direct in (("validation", validation), ("test", test)):
        panel = interval_panel(direct, supplemental)
        results = {}
        for name, options in policies.items():
            if period == "test" and name in {"fixed_upper", "dynamic_upper"}:
                key = "fixed_balanced" if name == "fixed_upper" else "dynamic_balanced"
                summary = dict(baseline["experiments"][key])
                trace = pd.read_csv(ROOT / summary["timeseries"], parse_dates=["timestamp"])
            else:
                trace, summary = simulate_balanced_storage(panel, ordinary["peak_threshold_mw"],
                    ordinary["charge_forecast_threshold_mw"], supplemental["uncertainty_scale_mw"], **options)
                path = out / f"peak_{period}_{name}_timeseries.csv"
                trace.to_csv(path, index=False)
                summary["timeseries"] = path.relative_to(ROOT).as_posix()
            results[name] = evidence(trace, summary)
            print(f"{period}/{name}: maximum={summary['controlled_peak_mw']:.2f}, P95={summary['controlled_p95_mw']:.2f}, terminal error={summary['max_abs_terminal_error_mwh']:.6f}", flush=True)
        periods[period] = {"experiments": results, "parameters": baseline["parameters"]}
    report = {
        "schema_version": "gridpulse-peak-response.v1",
        "calibration": {"release_threshold_mw": threshold, "quantile": .99,
            "signal": "Direct one-hour point forecasts in validation only", "validation_start": VALIDATION_START.isoformat(),
            "validation_end_exclusive": TEST_START.isoformat(), "validation_prediction_rows": int(validation.horizon_hours.eq(1).sum())},
        "policy": "Point forecasts locate the planned peak; interval upper bands still determine dynamic advisory reserves and discharge gates. A window point forecast reaching validation q99 releases the advisory floor to the shared hard minimum. Deadline reachability, power, losses and daily terminal equality remain enforced.",
        "periods": periods,
        "limitations": ["Designed after inspecting the previous test peak. All test results remain exploratory, even though the trigger is calibrated on validation only.",
            "Validation results reuse validation-calibrated intervals and trigger; they are internal diagnostics, not independent confirmation.",
            "Point-objective planning gives up the original upper-band minimax objective and may trade other outcomes for peak response.",
            "Released advisory reserves do not imply compliance with contractual backup obligations. The 15% operational floor is a modeling assumption.",
            "Independent UTC days, 50% SOC cap, unrestricted recovery charging, no network/prices/degradation; no deployment or reliability guarantee."],
    }
    (out / "peak_response_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
