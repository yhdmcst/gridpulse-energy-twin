"""Exploratory forecast-to-dispatch comparison on the already known test period.

Never overwrites frozen models, calibrations or earlier experiment outputs.
All methods share the controller and use the same validation calibration rules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.balanced_storage import complete_utc_days, simulate_balanced_storage
from src.trajectory_audit import audited_summary
from src.optimality import StorageModel


def simple_panel(panel, raw, method):
    result = panel.copy()
    lookup = raw.set_index("timestamp").net_load_mw
    for h in range(1, 7):
        # Panel timestamp is the first target hour. Issuance is one hour earlier.
        # Persistence uses that latest observation; seasonal uses target minus 24h.
        source = result.timestamp - pd.Timedelta(hours=1 if method == "persistence" else 25-h)
        result[f"prediction_h{h}_mw"] = source.map(lookup).to_numpy()
    return result


def calibrate(validation, test):
    result = test.copy()
    radii = {}
    maximum = 0.
    for h in range(1, 7):
        actual = validation[f"actual_h{h}_mw"]
        errors = (actual-validation[f"prediction_h{h}_mw"]).abs()
        target_hour = (validation.timestamp+pd.Timedelta(hours=h-1)).dt.hour
        by_hour = errors.groupby(target_hour).quantile(.9)
        if len(by_hour) != 24 or not np.isfinite(by_hour).all():
            raise ValueError("All target hours require finite validation calibration.")
        radii[str(h)] = {str(k): float(v) for k, v in by_hour.items()}
        maximum = max(maximum, float(by_hour.max()))
        radius = (test.timestamp+pd.Timedelta(hours=h-1)).dt.hour.map(by_hour)
        result[f"lower_h{h}_mw"] = result[f"prediction_h{h}_mw"]-radius
        result[f"upper_h{h}_mw"] = result[f"prediction_h{h}_mw"]+radius
    values = validation.prediction_h1_mw
    return result, {"peak_threshold_mw": float(values.quantile(.95)),
                    "charge_threshold_mw": float(values.quantile(.25)),
                    "release_threshold_mw": float(values.quantile(.99)),
                    "uncertainty_scale_mw": max(maximum*1.1, 1.)}, radii


def main():
    output = ROOT/"outputs/experiments/forecast_dispatch"
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT/"data/processed/net_load_hourly.csv", parse_dates=["timestamp"])
    lookup = raw.set_index("timestamp").net_load_mw
    panels = {period: pd.read_csv(ROOT/f"outputs/development/{name}_panel.csv", parse_dates=["timestamp"])
              for period, name in (("validation", "validation"), ("test", "known_test"))}
    methods = {}
    for method in ("ai", "persistence", "seasonal_24h"):
        methods[method] = {p: panel.copy() if method == "ai" else simple_panel(panel, raw, method)
                           for p, panel in panels.items()}
    for period in panels:
        common = pd.Index(panels[period].timestamp)
        for pair in methods.values():
            frame = pair[period]
            for h in range(1, 7):
                frame[f"actual_h{h}_mw"] = (frame.timestamp+pd.Timedelta(hours=h-1)).map(lookup).to_numpy()
            cols = [f"{prefix}_h{h}_mw" for h in range(1, 7) for prefix in ("prediction", "actual")]
            common = common.intersection(frame.loc[np.isfinite(frame[cols]).all(axis=1), "timestamp"])
        kept, _ = complete_utc_days(panels[period].loc[panels[period].timestamp.isin(common)])
        for pair in methods.values():
            pair[period] = pair[period].loc[pair[period].timestamp.isin(kept.timestamp)].reset_index(drop=True)
    report = {"schema_version": "gridpulse-forecast-dispatch.v1",
        "status": "Exploratory comparison on an already inspected test period; no independent confirmation or policy selection.",
        "calibration_rule": "Common complete validation days; each method uses target-hour/horizon absolute-error q90 bands, prediction q95/q25/q99 gates and 1.1 times its maximum validation band radius.",
        "controller": "Same dynamic reserve point-objective controller, advisory release, hard SOC floor and daily restoration for every method.",
        "sample": {p: {"rows": len(next(iter(methods.values()))[p]),
                        "days": len(next(iter(methods.values()))[p])//24} for p in panels},
        "physical_model": {"capacity_mwh": 30000., "charge_power_mw": 1250., "discharge_power_mw": 1250.,
                           "efficiency": .92, "hard_min_fraction": .15, "daily_terminal": True},
        "methods": {}}
    for method, pair in methods.items():
        panel, cal, radii = calibrate(pair["validation"], pair["test"])
        started = time.perf_counter()
        trace, summary = simulate_balanced_storage(panel, **cal, planning_objective="point")
        summary["simulation_wall_seconds"] = time.perf_counter()-started
        physical = audited_summary(trace, StorageModel())
        if not physical["physical_audit_passed"]:
            raise RuntimeError("Forecast comparison produced an invalid physical trajectory.")
        summary["mean_daily_peak_mw"] = physical["mean_daily_peak_mw"]
        summary["forecast_h1_mae_mw"] = float((panel.net_load_mw-panel.prediction_h1_mw).abs().mean())
        path = output/f"{method}_timeseries.csv"
        trace.to_csv(path, index=False)
        summary["timeseries"] = path.relative_to(ROOT).as_posix()
        report["methods"][method] = {"calibration": cal, "hourly_radii": radii, "metrics": summary}
        print(f"{method}: peak={summary['controlled_peak_mw']:.2f}, P95={summary['controlled_p95_mw']:.2f}, throughput={summary['throughput_mwh']:.2f}", flush=True)
    (output/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
