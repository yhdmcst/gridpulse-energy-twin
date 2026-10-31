"""Recompute metrics and selection from delivered development trajectories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_controller_development import read,sha
from src.trajectory_audit import audited_summary
from src.optimality import StorageModel
from src.numerics import same_loads


def compare(trace,saved,model,daily=True):
    actual = audited_summary(trace,model,daily_terminal=daily)
    if not actual["physical_audit_passed"]:
        return False
    for k,v in actual.items():
        if v is None:
            if saved[k] is not None:
                return False
        elif isinstance(v,(int,float)):
            if not np.isfinite(float(saved[k])) or abs(float(v)-float(saved[k])) > 1e-5:
                return False
        elif v != saved[k]:
            return False
    return True


def audit():
    report = read("outputs/development/controller_metrics.json")
    protocol = read("outputs/development/controller_protocol.json")
    if sha(ROOT/"outputs/development/controller_protocol.json") != report["protocol_sha256"]:
        return False
    if any(sha(ROOT/path) != checksum for path,checksum in protocol["development_inputs_sha256"].items()):
        return False
    model = StorageModel(**report["model"])
    for period,item in report["periods"].items():
        panel = pd.read_csv(ROOT/item["panel"],parse_dates=["timestamp"])
        for name,exp in item["experiments"].items():
            trace = pd.read_csv(ROOT/exp["timeseries"],parse_dates=["timestamp"])
            if (not pd.to_datetime(trace.timestamp,utc=True).equals(pd.to_datetime(panel.timestamp,utc=True))
                    or not same_loads(trace.net_load_mw,panel.net_load_mw) or not compare(trace,exp,model)):
                return False
    valid = report["periods"]["validation"]["experiments"]
    ref = valid["frozen_reference"]
    eligible = [name for name,item in valid.items() if name != "frozen_reference"
        and item["controlled_peak_mw"] <= ref["controlled_peak_mw"]+protocol["selection_peak_allowance_mw"]
        and item["controlled_p95_mw"] <= ref["controlled_p95_mw"]+protocol["selection_p95_allowance_mw"]]
    selected = min(eligible,key=lambda n:(valid[n]["throughput_mwh"],valid[n]["peak_tolerance_mw"])) if eligible else "frozen_reference"
    if eligible != report["validation_eligible"] or selected != report["selected_policy"]:
        return False
    stress = read("outputs/development/operating_stress_metrics.json")
    stress_protocol = read("outputs/development/operating_stress_protocol.json")
    if stress["protocol"] != stress_protocol or set(stress["episodes"]) != set(stress_protocol["weeks"]):
        return False
    panel2021 = pd.read_csv(ROOT/"outputs/development/known_2021_panel.csv",parse_dates=["timestamp"])
    for date,week in stress["episodes"].items():
        if set(week) != set(stress_protocol["scenarios"]):
            return False
        for scenario,item in week.items():
            if set(item["experiments"]) != set(stress_protocol["policies"]):
                return False
            for exp in item["experiments"].values():
                if exp["status"] == "solved":
                    trace = pd.read_csv(ROOT/exp["timeseries"],parse_dates=["timestamp"])
                    expected = pd.date_range(pd.Timestamp(date,tz="UTC"),periods=168,freq="h")
                    if not np.array_equal(pd.to_datetime(trace.timestamp,utc=True),expected):
                        return False
                    truth = panel2021.set_index("timestamp").loc[expected,"net_load_mw"]
                    if not same_loads(trace.net_load_mw,truth):
                        return False
                    if not compare(trace,exp,StorageModel(**item["model"]),scenario != "continuous_week"):
                        return False
    from scripts.audit_energy_and_intervals import energy_metrics,interval_score
    energy = read("outputs/development/renewable_alignment_metrics.json")
    for period,experiments in energy["periods"].items():
        raw = pd.read_csv(ROOT/("data/processed/independent_de_2021_hourly.csv" if period == "known_2021" else "data/processed/net_load_hourly.csv"))
        for policy,saved in experiments.items():
            trace = pd.read_csv(ROOT/read("outputs/development/controller_metrics.json")["periods"][period]["experiments"][policy]["timeseries"])
            actual = energy_metrics(trace,raw,energy["high_renewable_share_threshold"])
            if any(not np.isfinite(saved[k]) or abs(actual[k]-saved[k]) > 1e-5 for k in actual):
                return False
    intervals = pd.read_csv(ROOT/"outputs/development/interval_diagnostics.csv")
    report = read("outputs/development/interval_audit_metrics.json")
    for row in report["groups"]:
        stamps = pd.to_datetime(intervals.timestamp,utc=True)
        g = intervals.loc[intervals.horizon_hours.eq(row["horizon_hours"])&stamps.dt.quarter.eq(row["quarter"])]
        if row["scope"] == "high_load":
            g = g.loc[g.net_load_mw.ge(report["high_load_threshold_mw"])]
        if len(g) != row["rows"]:
            return False
        for name in ("fixed","rolling"):
            values = [row[f"{name}_coverage_pct"], row[f"{name}_mean_width_mw"], row[f"{name}_mean_interval_score_mw"]]
            if not np.isfinite(values).all() or not np.isfinite(g[["net_load_mw", "prediction_mw", f"{name}_radius_mw"]].to_numpy(float)).all():
                return False
            if abs(100*g[f"{name}_covered"].mean()-row[f"{name}_coverage_pct"]) > 1e-5:
                return False
            if (abs(2*g[f"{name}_radius_mw"].mean()-row[f"{name}_mean_width_mw"]) > 1e-5
                    or abs(interval_score(g.net_load_mw-g.prediction_mw,g[f"{name}_radius_mw"])-row[f"{name}_mean_interval_score_mw"]) > 1e-5):
                return False
    return True


if __name__ == "__main__":
    passed = audit()
    print(json.dumps({"development_audit_passed":passed}))
    raise SystemExit(0 if passed else 1)
