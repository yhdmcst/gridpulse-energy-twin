"""A separately locked 2022 replication; never modify the 2021 protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import regression_metrics
from src.balanced_storage import complete_utc_days, simulate_balanced_storage
from src.downloads import atomic_download
from src.independent_validation import normalize_public_power, paired_block_summary
from src.multihorizon import FEATURE_COLUMNS, build_direct_horizon_frame
from src.optimality import StorageModel
from src.trajectory_audit import audited_summary
from scripts.run_peak_response import evidence, interval_panel

OUT = ROOT / "outputs/validation"
PROTOCOL = OUT / "independent_2022_protocol.json"
RAW = ROOT / "data/raw/energy_charts_de_2022.json"
DATASET = ROOT / "data/processed/independent_de_2022_hourly.csv"
START, END = pd.Timestamp("2022-01-01T00:00Z"), pd.Timestamp("2023-01-01T00:00Z")
WARMUP, INPUT_END = START-pd.Timedelta(days=8), END+pd.Timedelta(hours=6)
URL = "https://api.energy-charts.info/v2/public_power?country=de&start=2021-12-24T00:00Z&end=2023-01-01T06:00Z"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if any(p.exists() for p in (PROTOCOL, RAW, DATASET, OUT/"independent_2022_metrics.json")):
        raise FileExistsError("2022 protocol or observations exist; refusing retrospective registration.")
    original = read(OUT/"independent_2021_protocol.json")
    for path, digest in original["frozen_sha256"].items():
        if sha(ROOT/path) != digest:
            raise ValueError("2021 frozen input changed: " + path)
    files = list(original["frozen_sha256"]) + ["outputs/validation/independent_2021_protocol.json",
        "src/downloads.py", "src/trajectory_audit.py", "scripts/validate_later_year.py"]
    release = original["release_threshold_mw"]
    policies = {"fixed_upper": {"fixed_reserve_fraction": .30}, "dynamic_upper": {},
        "dynamic_release_only": {"release_threshold_mw": release},
        "dynamic_point_only": {"planning_objective": "point"},
        "dynamic_peak_response": {"planning_objective": "point", "release_threshold_mw": release},
        "fixed_peak_response": {"fixed_reserve_fraction": .30, "planning_objective": "point", "release_threshold_mw": release}}
    write(PROTOCOL, {"schema_version": "gridpulse-later-year-protocol.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "source_url": URL,
        "source": "Energy-Charts.info, Fraunhofer ISE, CC BY 4.0",
        "period": {"start": START.isoformat(), "end_exclusive": END.isoformat(), "warmup_start": WARMUP.isoformat()},
        "parameters": original["parameters"], "policies": policies,
        "clock_semantics": original["clock_semantics"],
        "primary_endpoint": "Full-year maximum reduction: dynamic_peak_response versus dynamic_upper. Report zero or negative results as failure.",
        "secondary_endpoints": ["mean daily peak", "P95", "all quarters", "throughput", "conversion loss", "physical constraints", "six forecast horizons and coverage"],
        "uncertainty": {"block_days": [7, 3, 14], "repetitions": 2000, "seed": 20221010},
        "rules": ["2022 data are not yet downloaded or evaluated in this project. All six original policies are locked.",
            "2021 and prior post-hoc development results are known; this validates the unchanged original policies only.",
            "No retraining, recalibration, rescaling or selection on 2022 results. New development candidates are excluded.",
            "Evaluate on shared complete UTC days; never impute incomplete source hours.",
            "Disclose all quarters and negative results. Bootstrap cannot infer unseen extreme maxima.",
            "Same-country revised historical simulation with zero extra publication delay, not live deployment."],
        "frozen_sha256": {name: sha(ROOT/name) for name in files}})
    print("2022 protocol locked before data download.", flush=True)


def verify_frozen():
    protocol = read(PROTOCOL)
    if protocol["source_url"] != URL or protocol["period"]["start"] != START.isoformat():
        raise ValueError("Unexpected holdout protocol.")
    for path, expected in protocol["frozen_sha256"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("Frozen replication input changed: " + path)
    return protocol


def validate_payload(path):
    payload = read(path)
    hourly, quality = normalize_public_power(payload, WARMUP, INPUT_END)
    values = hourly.drop(columns="timestamp").to_numpy(float)
    if np.isinf(values).any() or any(quality["negative_source_values"].values()):
        raise ValueError("Infinite or negative source power values are invalid.")
    evaluation = hourly.loc[hourly.timestamp.ge(START) & hourly.timestamp.lt(END)]
    if evaluation.net_load_mw.notna().sum() < 24:
        raise ValueError("Source returned no usable holdout day.")
    return hourly, quality


def download():
    protocol = verify_frozen()
    if not RAW.exists():
        atomic_download(URL, RAW, validator=validate_payload)
    hourly, quality = validate_payload(RAW)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(DATASET, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    payload = read(RAW)
    write(OUT/"independent_2022_data_quality.json", {**quality, "protocol_sha256": sha(PROTOCOL),
        "raw_sha256": sha(RAW), "dataset_sha256": sha(DATASET), "source_url": protocol["source_url"],
        "license": payload.get("license"), "provider_generated_at": payload.get("generated_at"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"2022 snapshot normalized: {len(hourly)} hours.", flush=True)


def evaluate():
    protocol = verify_frozen()
    quality = read(OUT/"independent_2022_data_quality.json")
    if quality["dataset_sha256"] != sha(DATASET) or quality["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Holdout snapshot lineage failed.")
    raw = pd.read_csv(DATASET, parse_dates=["timestamp"])
    supplemental = read(ROOT/"outputs/scenarios/reserve_ablation_metrics.json")
    calibration = read(ROOT/"outputs/scenarios/storage_scenario_metrics.json")["calibration"]
    forecasts, direct = {}, []
    for h in range(1, 7):
        frame = build_direct_horizon_frame(raw, h)
        selected = frame.loc[frame.timestamp.ge(START) & frame.timestamp.lt(INPUT_END)]
        model = joblib.load(ROOT/f"outputs/models/direct_horizon_{h}h.joblib")["model"]
        predicted = model.predict(selected[list(FEATURE_COLUMNS)])
        radii = {int(k): v for k, v in supplemental["hourly_interval_calibration"][str(h)].items()}
        radius = selected.timestamp.dt.hour.map(radii).to_numpy()
        truth, valid = selected.net_load_mw.to_numpy(), selected.timestamp.lt(END).to_numpy()
        forecasts[str(h)] = {**regression_metrics(pd.Series(truth[valid]), pd.Series(predicted[valid])),
            "rows": int(valid.sum()), "interval_coverage_pct": float(100*np.mean(np.abs(truth[valid]-predicted[valid]) <= radius[valid])),
            "persistence_mae_mw": float(np.mean(np.abs(truth[valid]-selected.net_load_latest_available_mw.to_numpy()[valid])))}
        direct.append(pd.DataFrame({"timestamp": selected.timestamp,
            "issued_timestamp": selected.issued_timestamp+pd.Timedelta(hours=1), "horizon_hours": h,
            "net_load_mw": truth, "prediction_mw": predicted, "lower_mw": predicted-radius, "upper_mw": predicted+radius}))
    predictions = pd.concat(direct, ignore_index=True)
    predictions.to_csv(OUT/"independent_2022_predictions.csv", index=False)
    panel = interval_panel(predictions, supplemental)
    panel = panel.loc[panel.timestamp.ge(START) & panel.timestamp.lt(END)]
    panel, _ = complete_utc_days(panel)
    expected_days = pd.date_range(START, END, freq="D", inclusive="left")
    included_days = pd.DatetimeIndex(panel.timestamp.dt.floor("D").unique())
    omitted = expected_days.difference(included_days)
    traces, experiments, quarters = {}, {}, {}
    for name, options in protocol["policies"].items():
        trace, summary = simulate_balanced_storage(panel, calibration["peak_threshold_mw"],
            calibration["charge_forecast_threshold_mw"], supplemental["uncertainty_scale_mw"],
            **protocol["parameters"], **options)
        physical = audited_summary(trace, StorageModel())
        if not physical["physical_audit_passed"]:
            raise ValueError("Holdout physical trajectory audit failed.")
        path = OUT/f"independent_2022_{name}_timeseries.csv"
        trace.to_csv(path, index=False)
        summary.update({"timeseries": path.relative_to(ROOT).as_posix(), "physical_audit": physical})
        experiments[name], traces[name] = evidence(trace, summary), trace
        quarters[name] = {str(q): {"days": len(t)//24, "controlled_peak_mw": float(t.controlled_net_load_mw.max()),
            "controlled_p95_mw": float(t.controlled_net_load_mw.quantile(.95)),
            "throughput_mwh": float((t.charge_mw+t.discharge_mw).sum())} for q, t in trace.groupby(trace.timestamp.dt.quarter)}
        print(f"2022/{name}: peak={summary['controlled_peak_mw']:.2f}, P95={summary['controlled_p95_mw']:.2f}, days={summary['days']}", flush=True)
    comparisons = {}
    for old in ("dynamic_upper", "fixed_peak_response", "dynamic_point_only"):
        new = "dynamic_peak_response"
        a, b = experiments[old], experiments[new]
        comparisons[f"{new}_versus_{old}"] = {"full_period_peak_reduction_mw": a["controlled_peak_mw"]-b["controlled_peak_mw"],
            "p95_reduction_mw": a["controlled_p95_mw"]-b["controlled_p95_mw"],
            "throughput_increase_mwh": b["throughput_mwh"]-a["throughput_mwh"],
            "loss_increase_mwh": b["conversion_loss_mwh"]-a["conversion_loss_mwh"],
            "uncertainty": paired_block_summary(traces[old], traces[new], seed=20221010)}
    write(OUT/"independent_2022_metrics.json", {"schema_version": "gridpulse-later-year-results.v1",
        "protocol_sha256": sha(PROTOCOL), "dataset_sha256": sha(DATASET), "freeze_verified": True,
        "period": protocol["period"], "source": protocol["source"], "parameters": protocol["parameters"],
        "shared_days": len(included_days), "omitted_days": [d.isoformat() for d in omitted],
        "forecast_metrics": forecasts, "experiments": experiments, "quarters": quarters, "comparisons": comparisons,
        "bootstrap_block_sensitivity": {str(block): paired_block_summary(traces["dynamic_upper"], traces["dynamic_peak_response"],
            block_days=block, seed=20221010) for block in (3, 14)},
        "primary_observed_success": comparisons["dynamic_peak_response_versus_dynamic_upper"]["full_period_peak_reduction_mw"] > 1e-6,
        "limitations": protocol["rules"]})
    print("2022 replication complete; no selection or retuning.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "download", "evaluate"))
    args = parser.parse_args()
    {"freeze": freeze, "download": download, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
