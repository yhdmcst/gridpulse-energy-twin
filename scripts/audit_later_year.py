"""Recompute the locked 2022 replication from portable predictions and traces."""
from __future__ import annotations

from datetime import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_later_year import DATASET, END, OUT, PROTOCOL, START, read, sha, verify_frozen
from src.baseline import regression_metrics
from src.independent_validation import paired_block_summary
from src.optimality import StorageModel
from src.trajectory_audit import audited_summary


def close(actual, expected):
    return np.isfinite([actual, expected]).all() and abs(float(actual)-float(expected)) <= 1e-5


def audit():
    protocol = verify_frozen()
    report = read(OUT/"independent_2022_metrics.json")
    quality = read(OUT/"independent_2022_data_quality.json")
    if (report["protocol_sha256"] != sha(PROTOCOL) or report["dataset_sha256"] != sha(DATASET)
            or quality["protocol_sha256"] != sha(PROTOCOL) or quality["dataset_sha256"] != sha(DATASET)
            or datetime.fromisoformat(protocol["frozen_at_utc"]) >= datetime.fromisoformat(quality["downloaded_at_utc"])):
        return False
    raw = pd.read_csv(DATASET, parse_dates=["timestamp"]).set_index("timestamp")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        return False
    if not np.allclose(raw.net_load_mw, raw.total_load_mw-raw.solar_generation_mw-raw.wind_generation_mw, atol=1e-6, equal_nan=True):
        return False
    prediction = pd.read_csv(OUT/"independent_2022_predictions.csv", parse_dates=["timestamp", "issued_timestamp"])
    if set(prediction.horizon_hours) != set(range(1, 7)):
        return False
    supplemental = read(ROOT/"outputs/scenarios/reserve_ablation_metrics.json")
    for h, group in prediction.groupby("horizon_hours"):
        if group.timestamp.duplicated().any() or not group.timestamp.is_monotonic_increasing:
            return False
        if not (group.timestamp-group.issued_timestamp).eq(pd.Timedelta(hours=int(h)-1)).all():
            return False
        values = group[["net_load_mw", "prediction_mw", "lower_mw", "upper_mw"]].to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(group.net_load_mw, raw.loc[group.timestamp, "net_load_mw"]):
            return False
        radii = {int(k): v for k, v in supplemental["hourly_interval_calibration"][str(h)].items()}
        radius = group.timestamp.dt.hour.map(radii).to_numpy()
        if not np.allclose(group.lower_mw, group.prediction_mw-radius) or not np.allclose(group.upper_mw, group.prediction_mw+radius):
            return False
        valid = group.loc[group.timestamp.ge(START) & group.timestamp.lt(END)]
        expected = report["forecast_metrics"][str(h)]
        measured = regression_metrics(valid.net_load_mw, valid.prediction_mw)
        measured["rows"] = len(valid)
        measured["interval_coverage_pct"] = 100*np.mean((valid.net_load_mw >= valid.lower_mw) & (valid.net_load_mw <= valid.upper_mw))
        latest = raw.loc[valid.timestamp-pd.Timedelta(hours=int(h)), "net_load_mw"].to_numpy()
        measured["persistence_mae_mw"] = np.mean(np.abs(valid.net_load_mw.to_numpy()-latest))
        if any(not close(value, expected[key]) for key, value in measured.items()):
            return False
    if set(report["experiments"]) != set(protocol["policies"]):
        return False
    traces = {}
    reference = None
    for name, summary in report["experiments"].items():
        trace = pd.read_csv(ROOT/summary["timeseries"], parse_dates=["timestamp"])
        physical = audited_summary(trace, StorageModel())
        if not physical["physical_audit_passed"]:
            return False
        for key, value in physical.items():
            saved = summary["physical_audit"][key]
            if value is None:
                if saved is not None:
                    return False
            elif isinstance(value, (int, float)):
                if not close(value, saved):
                    return False
            elif value != saved:
                return False
        if any(not close(physical[k], summary[k]) for k in ("rows", "days", "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")):
            return False
        if reference is None:
            reference = trace.timestamp
        if (not trace.timestamp.equals(reference) or not np.allclose(trace.net_load_mw, raw.loc[trace.timestamp, "net_load_mw"])
                or not trace.timestamp.ge(START).all() or not trace.timestamp.lt(END).all()):
            return False
        for q, group in trace.groupby(trace.timestamp.dt.quarter):
            saved = report["quarters"][name][str(q)]
            measured = {"days": len(group)//24, "controlled_peak_mw": group.controlled_net_load_mw.max(),
                "controlled_p95_mw": group.controlled_net_load_mw.quantile(.95),
                "throughput_mwh": (group.charge_mw+group.discharge_mw).sum()}
            if any(not close(value, saved[key]) for key, value in measured.items()):
                return False
        traces[name] = trace
    observed = pd.DatetimeIndex(reference.dt.floor("D").unique())
    omitted = pd.date_range(START, END, freq="D", inclusive="left").difference(observed)
    if report["shared_days"] != len(observed) or report["omitted_days"] != [d.isoformat() for d in omitted]:
        return False
    for old in ("dynamic_upper", "fixed_peak_response", "dynamic_point_only"):
        new = "dynamic_peak_response"
        a, b = report["experiments"][old], report["experiments"][new]
        saved = report["comparisons"][f"{new}_versus_{old}"]
        for field, key, sign in (("controlled_peak_mw", "full_period_peak_reduction_mw", 1),
            ("controlled_p95_mw", "p95_reduction_mw", 1), ("throughput_mwh", "throughput_increase_mwh", -1),
            ("conversion_loss_mwh", "loss_increase_mwh", -1)):
            if not close(sign*(a[field]-b[field]), saved[key]):
                return False
    primary = report["comparisons"]["dynamic_peak_response_versus_dynamic_upper"]
    recalculated = paired_block_summary(traces["dynamic_upper"], traces["dynamic_peak_response"], seed=20221010)
    for field in ("mean_daily_peak_reduction_mw", "p95_reduction_mw"):
        if any(not close(v, primary["uncertainty"][field][key]) for key, v in recalculated[field].items()):
            return False
    return report["primary_observed_success"] == (primary["full_period_peak_reduction_mw"] > 1e-6)


if __name__ == "__main__":
    passed = audit()
    print(json.dumps({"later_year_audit_passed": passed}))
    raise SystemExit(0 if passed else 1)
