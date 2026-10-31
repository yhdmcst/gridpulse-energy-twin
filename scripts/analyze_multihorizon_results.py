"""Readable aggregates and failure diagnostics; never modify frozen v3 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v3"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = read(OUT / "metrics.json")
    calibration = read(OUT / "calibration.json")
    keys = ("forecast_h1_mae_mw", "baseline_peak_mw", "controlled_peak_mw",
            "controlled_p95_mw", "mean_daily_peak_mw", "throughput_mwh",
            "conversion_loss_mwh", "recovery_charge_mwh", "max_abs_terminal_error_mwh")
    samples = read(OUT / "samples.json")
    result = {"seconds": report["seconds"], "families": {}, "stress": {}, "examples": {},
              "forecast_diagnostics_on_dispatch_rows": {}}
    for period, methods in report["methods"].items():
        result["families"][period] = {}
        result["forecast_diagnostics_on_dispatch_rows"][period] = {}
        for family in ("persistence", "hgb", "data", "physics"):
            names = [family] if family in ("hgb", "persistence") else [f"{family}_seed{s}" for s in (42, 43, 44)]
            predictions = []
            for name in names:
                frame = pd.read_csv(OUT / "predictions" / name / f"{period}.csv",
                                    parse_dates=["timestamp"])
                frame = frame.loc[frame.timestamp.between(
                    pd.Timestamp(samples[period]["start"]), pd.Timestamp(samples[period]["end"]))]
                assert len(frame) == samples[period]["rows"]
                predictions.append(frame)
            diagnostics = {
                "net_load_mae_by_horizon_mw": [
                    float(np.mean([(frame[f"actual_h{h}_mw"] - frame[f"prediction_h{h}_mw"]).abs().mean()
                                   for frame in predictions])) for h in range(1, 7)],
            }
            if family != "hgb":
                diagnostics["mean_accounting_residual_six_horizons_mw"] = float(np.mean([
                    (frame[f"predicted_h{h}_net_load_mw"] - frame[f"predicted_h{h}_total_load_mw"]
                     + frame[f"predicted_h{h}_solar_generation_mw"]
                     + frame[f"predicted_h{h}_wind_generation_mw"]).abs().mean()
                    for frame in predictions for h in range(1, 7)]))
            result["forecast_diagnostics_on_dispatch_rows"][period][family] = diagnostics
            group = {}
            for window in (1, 6):
                items = [methods[f"{name}_w{window}"] for name in names]
                values = {key: float(np.mean([item[key] for item in items])) for key in keys}
                high_load_charge, count = [], []
                for item in items:
                    trace = pd.read_csv(OUT / item["trajectory"])
                    mask = trace.net_load_mw.ge(calibration["gates"]["peak_threshold_mw"])
                    high_load_charge.append(trace.loc[mask, "charge_mw"].sum())
                    count.append((mask & trace.charge_mw.gt(1e-6)).sum())
                values["actual_high_load_charging_mwh"] = float(np.mean(high_load_charge))
                values["actual_high_load_charging_hours"] = float(np.mean(count))
                group[f"w{window}"] = values
            group["w6_minus_w1"] = {key: group["w6"][key] - group["w1"][key] for key in group["w1"]}
            result["families"][period][family] = group
        for window in (1, 6):
            result["families"][period][f"data_minus_hgb_w{window}"] = report["comparisons"][period][f"data_minus_hgb_w{window}"]
            result["families"][period][f"physics_minus_data_w{window}"] = report["comparisons"][period][f"physics_minus_data_w{window}"]
        daily_delta = []
        for seed in (42, 43, 44):
            a = pd.read_csv(OUT / methods[f"data_seed{seed}_w6"]["trajectory"], parse_dates=["timestamp"])
            b = pd.read_csv(OUT / methods[f"data_seed{seed}_w1"]["trajectory"], parse_dates=["timestamp"])
            daily_delta.append((
                a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                - b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy())
        values = np.mean(daily_delta, axis=0)
        rng = np.random.default_rng(20261026)
        bootstrap = []
        for _ in range(2000):
            starts = rng.integers(0, len(values), size=(len(values) + 6) // 7)
            indices = ((starts[:, None] + np.arange(7)) % len(values)).ravel()[:len(values)]
            bootstrap.append(values[indices].mean())
        result["families"][period]["data_w6_minus_w1_daily_peak_interval"] = {
            "estimate_mw": float(values.mean()),
            "interval_95_mw": np.quantile(bootstrap, [.025, .975]).tolist(),
            "days": len(values), "scope": "Conditional fixed-seed mean, seven-day circular blocks.",
        }
        result["examples"][period] = {}
        for window in (1, 6):
            trace = pd.read_csv(OUT / methods[f"data_seed42_w{window}"]["trajectory"], parse_dates=["timestamp"])
            at_peak = trace.loc[trace.controlled_net_load_mw.idxmax()]
            result["examples"][period][f"maximum_w{window}"] = {
                key: str(at_peak[key]) if key == "timestamp" else float(at_peak[key])
                for key in ("timestamp", "net_load_mw", "charge_mw", "discharge_mw",
                            "soc_before_mwh", "required_next_soc_mwh", "controlled_net_load_mw")
            }
            for stamp in ("2019-11-20T16:00:00Z", "2022-01-10T16:00:00Z"):
                example = trace.loc[trace.timestamp.eq(pd.Timestamp(stamp))]
                if len(example):
                    row = example.iloc[0]
                    result["examples"][period][f"{stamp}_w{window}"] = {
                        key: float(row[key]) for key in (
                            "net_load_mw", "prediction_h1_mw", "charge_mw", "discharge_mw",
                            "soc_before_mwh", "required_next_soc_mwh", "controlled_net_load_mw")
                    }
    for period, scenarios in report["stress"].items():
        result["stress"][period] = {}
        for name, arms in scenarios.items():
            result["stress"][period][name] = {
                "replay_terminal_error_mwh": arms["replay"]["max_abs_terminal_error_mwh"],
                "replan_terminal_error_mwh": arms["replan"]["max_abs_terminal_error_mwh"],
                "replan_minus_replay": {key: arms["replan"][key] - arms["replay"][key] for key in (
                    "controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
                    "throughput_mwh", "conversion_loss_mwh")},
            }
    path = ROOT / "outputs/fusion_v3_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "analysis": str(path), "seconds": result["seconds"],
        "forecast_diagnostics_on_dispatch_rows": result["forecast_diagnostics_on_dispatch_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
