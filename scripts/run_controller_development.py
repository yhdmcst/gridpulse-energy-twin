"""Prespecified post-hoc controller comparisons; no new holdout claim."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_peak_response import interval_panel, validation_predictions
from src.balanced_storage import complete_utc_days
from src.efficient_storage import ControlConfig, simulate, summarize
from src.optimality import StorageModel

OUT = ROOT/"outputs/development"
TOLERANCES = (0.,100.,250.,500.)


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration():
    original = read("outputs/scenarios/storage_scenario_metrics.json")["calibration"]
    supplemental = read("outputs/scenarios/reserve_ablation_metrics.json")
    return {"peak_threshold_mw":original["peak_threshold_mw"],
        "uncertainty_scale_mw":supplemental["uncertainty_scale_mw"],
        "release_threshold_mw":read("outputs/scenarios/peak_response_metrics.json")["calibration"]["release_threshold_mw"]}


def panels():
    supplemental = read("outputs/scenarios/reserve_ablation_metrics.json")
    raw = pd.read_csv(ROOT/"data/processed/net_load_hourly.csv",parse_dates=["timestamp"])
    result = {}
    for period,direct in (
        ("validation",validation_predictions(raw)),
        ("known_test",pd.read_csv(ROOT/"outputs/models/direct_horizon_test_predictions.csv",parse_dates=["timestamp","issued_timestamp"])),
        ("known_2021",pd.read_csv(ROOT/"outputs/validation/independent_2021_predictions.csv",parse_dates=["timestamp","issued_timestamp"]))):
        panel = interval_panel(direct,supplemental)
        if period == "known_2021":
            panel = panel.loc[panel.timestamp.ge(pd.Timestamp("2021-01-01T00:00Z")) & panel.timestamp.lt(pd.Timestamp("2022-01-01T00:00Z"))]
        selected,omitted = complete_utc_days(panel)
        result[period] = selected
        selected.to_csv(OUT/f"{period}_panel.csv",index=False)
        print(f"Prepared {period}: {len(selected)} rows; omitted partial days: {omitted}",flush=True)
    return result


def baseline(period):
    if period == "known_2021":
        return "outputs/validation/independent_2021_dynamic_peak_response_timeseries.csv"
    return f"outputs/scenarios/peak_{'test' if period == 'known_test' else 'validation'}_dynamic_peak_response_timeseries.csv"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cal = calibration()
    protocol = {"schema_version":"gridpulse-controller-development.v1",
        "status":"Post-hoc development on already-known years; not independent confirmation.",
        "grid_peak_tolerances_mw":TOLERANCES,"model":asdict(StorageModel()),"calibration":cal,
        "selection_rule":"On original validation only: peak no worse than frozen new dynamic +1 MW and P95 no worse than +50 MW; select lowest throughput, then smallest tolerance. If none passes, retain frozen controller.",
        "selection_peak_allowance_mw":1.,"selection_p95_allowance_mw":50.,
        "scope":"Daily SOC restoration, unchanged physical model, point forecasts and original advisory reserve/release/gates.",
        "development_inputs_sha256":{name:sha(ROOT/name) for name in (
            "src/efficient_storage.py","scripts/run_controller_development.py",
            "outputs/scenarios/peak_response_metrics.json","outputs/validation/independent_2021_protocol.json")}}
    # Written before this grid is evaluated, but known results mean this is
    # transparent development bookkeeping, never untouched preregistration.
    write(OUT/"controller_protocol.json",protocol)
    all_panels = panels()
    report = {"schema_version":"gridpulse-controller-development-results.v1","status":protocol["status"],
        "protocol_sha256":sha(OUT/"controller_protocol.json"),"model":protocol["model"],"periods":{}}
    for period,panel in all_panels.items():
        reference_path = baseline(period)
        reference = pd.read_csv(ROOT/reference_path,parse_dates=["timestamp"])
        if not pd.to_datetime(reference.timestamp,utc=True).equals(pd.to_datetime(panel.timestamp,utc=True)) or not (reference.net_load_mw.to_numpy() == panel.net_load_mw.to_numpy()).all():
            raise ValueError("Development and reference samples differ.")
        results = {"frozen_reference":{**summarize(reference,StorageModel()),"timeseries":reference_path}}
        for tol in TOLERANCES:
            trace,summary = simulate(panel,**cal,config=ControlConfig(tol))
            name = f"two_stage_{int(tol)}mw"
            path = OUT/f"{period}_{name}.csv"
            trace.to_csv(path,index=False)
            results[name] = {**summary,"peak_tolerance_mw":tol,"timeseries":path.relative_to(ROOT).as_posix()}
            print(f"{period}/{name}: peak={summary['controlled_peak_mw']:.2f}, P95={summary['controlled_p95_mw']:.2f}, throughput={summary['throughput_mwh']:.2f}",flush=True)
        report["periods"][period] = {"panel":(OUT/f"{period}_panel.csv").relative_to(ROOT).as_posix(),"experiments":results}
        if period == "validation":
            ref = results["frozen_reference"]
            eligible = [name for name,item in results.items() if name != "frozen_reference"
                and item["controlled_peak_mw"] <= ref["controlled_peak_mw"]+1.
                and item["controlled_p95_mw"] <= ref["controlled_p95_mw"]+50.]
            report["validation_eligible"] = eligible
            report["selected_policy"] = (min(eligible,key=lambda name:(results[name]["throughput_mwh"],results[name]["peak_tolerance_mw"]))
                                         if eligible else "frozen_reference")
            write(OUT/"controller_selection.json",{"selected_policy":report["selected_policy"],
                "eligible":eligible,"selection_period":"original validation only",
                "protocol_sha256":report["protocol_sha256"],"status":protocol["status"]})
    for period,item in report["periods"].items():
        a,b = (item["experiments"][key] for key in ("frozen_reference",report["selected_policy"]))
        item["selected_minus_reference"] = {metric:b[metric]-a[metric] for metric in (
            "controlled_peak_mw","controlled_p95_mw","mean_daily_peak_mw","throughput_mwh","conversion_loss_mwh")}
    write(OUT/"controller_metrics.json",report)
    print(json.dumps({"selected_policy":report["selected_policy"],"deltas":{p:v["selected_minus_reference"] for p,v in report["periods"].items()}},indent=2))


if __name__ == "__main__":
    main()
