"""Matched daily 2x2 experiment: reserve rule versus terminal energy accounting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.balanced_storage import simulate_balanced_storage
from src.multihorizon import build_planning_panel


def main() -> None:
    supplemental = json.loads((ROOT / "outputs/scenarios/reserve_ablation_metrics.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "outputs/scenarios/storage_scenario_metrics.json").read_text(encoding="utf-8"))["calibration"]
    direct = pd.read_csv(ROOT / "outputs/models/direct_horizon_test_predictions.csv", parse_dates=["issued_timestamp", "timestamp"])
    for horizon in range(1, 7):
        mask = direct.horizon_hours.eq(horizon)
        radii = {int(hour): radius for hour, radius in supplemental["hourly_interval_calibration"][str(horizon)].items()}
        radius = direct.loc[mask, "timestamp"].dt.hour.map(radii)
        direct.loc[mask, "lower_mw"] = direct.loc[mask, "prediction_mw"] - radius
        direct.loc[mask, "upper_mw"] = direct.loc[mask, "prediction_mw"] + radius
    panel = build_planning_panel(direct, 6)
    output = ROOT / "outputs/scenarios"
    experiments = {}
    for terminal in (False, True):
        for rule in ("fixed", "dynamic"):
            key = f"{rule}_{'balanced' if terminal else 'open'}"
            trace, summary = simulate_balanced_storage(
                panel, calibration["peak_threshold_mw"],
                calibration["charge_forecast_threshold_mw"], supplemental["uncertainty_scale_mw"],
                fixed_reserve_fraction=.30 if rule == "fixed" else None,
                daily_terminal=terminal,
            )
            path = output / f"daily_{key}_timeseries.csv"
            trace.to_csv(path, index=False)
            summary["timeseries"] = path.relative_to(ROOT).as_posix()
            experiments[key] = summary
            print(f"Completed {key}: P95={summary['controlled_p95_mw']:.2f}, terminal error={summary['max_abs_terminal_error_mwh']:.6f}", flush=True)
    comparisons = {}
    for protocol in ("open", "balanced"):
        fixed, dynamic = experiments[f"fixed_{protocol}"], experiments[f"dynamic_{protocol}"]
        comparisons[protocol] = {
            "dynamic_minus_fixed_p95_mw": dynamic["controlled_p95_mw"] - fixed["controlled_p95_mw"],
            "dynamic_minus_fixed_throughput_mwh": dynamic["throughput_mwh"] - fixed["throughput_mwh"],
            "dynamic_minus_fixed_initial_energy_consumption_mwh": dynamic["net_initial_energy_consumed_mwh"] - fixed["net_initial_energy_consumed_mwh"],
        }
    report = {
        "schema_version": "gridpulse-daily-energy-accounting.v1",
        "protocol": "Exploratory matched 2x2 comparisons on all complete UTC test days. Same direct forecasts, validation-hour radii, power, hard 15% SOC floor and daily 50% initial SOC. No policy chosen by test outcomes.",
        "balanced_protocol": "Every complete UTC day ends at 50% SOC. SOC capped at 50% in balanced mode. Charging at any hour is allowed to ensure restoration; all added load and losses count in metrics. Reserve targets remain advisory above the common hard floor.",
        "open_protocol": "Independent daily episodes start at 50% SOC without terminal restoration; low-load charging only. Initial energy consumed is explicitly reported. This daily reset is an experimental control, not a sustainable operation.",
        "information_boundary": "Only each row's issued direct forecasts, the known UTC deadline and rated power enter planning. Actual net load is used only for ex-post metrics.",
        "parameters": {"capacity_mwh": 30000., "charge_power_mw": 1250., "discharge_power_mw": 1250., "efficiency": .92, "hard_min_fraction": .15, "initial_soc_fraction": .50},
        "experiments": experiments, "comparisons": comparisons,
        "limitations": ["Exploratory analysis after original test outcomes were known.", "Terminal comparison also changes charging permission and SOC upper bound; do not attribute the entire difference solely to initial energy.", "Dynamic reserve targets remain advisory, distinct from the shared hard 15% SOC floor.", "No prices, degradation, network or deployment evidence.", "Incomplete UTC days are excluded identically; independent day episodes are not a continuous multi-day deployment."],
    }
    (output / "daily_energy_balance_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
