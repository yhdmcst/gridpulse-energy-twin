"""Freeze a temporal holdout protocol BEFORE downloading evaluation observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.balanced_storage import simulate_balanced_storage
from src.multihorizon import FEATURE_COLUMNS, build_direct_horizon_frame
from src.baseline import regression_metrics
from src.independent_validation import normalize_public_power, paired_block_summary
from scripts.run_peak_response import evidence, interval_panel

OUT = ROOT / "outputs/validation"
PROTOCOL = OUT / "independent_2021_protocol.json"
RAW = ROOT / "data/raw/energy_charts_de_2021.json"
DATASET = ROOT / "data/processed/independent_de_2021_hourly.csv"
START = pd.Timestamp("2021-01-01T00:00Z")
END = pd.Timestamp("2022-01-01T00:00Z")
WARMUP = pd.Timestamp("2020-12-24T00:00Z")
INPUT_END = pd.Timestamp("2022-01-01T06:00Z")
URL = "https://api.energy-charts.info/v2/public_power?country=de&start=2020-12-24T00:00Z&end=2022-01-01T06:00Z"
FROZEN_FILES = [*(f"outputs/models/direct_horizon_{h}h.joblib" for h in range(1, 7)),
    "outputs/scenarios/reserve_ablation_metrics.json", "outputs/scenarios/storage_scenario_metrics.json",
    "outputs/scenarios/peak_response_metrics.json", "src/scenarios.py", "src/balanced_storage.py",
    "src/multihorizon.py", "src/independent_validation.py", "scripts/run_independent_validation.py",
    "scripts/run_peak_response.py"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze() -> None:
    if PROTOCOL.exists() or RAW.exists():
        raise FileExistsError("Protocol or evaluation data already exists; refusing retrospective registration.")
    peak = json.loads((ROOT / "outputs/scenarios/peak_response_metrics.json").read_text(encoding="utf-8"))
    protocol = {"schema_version": "gridpulse-independent-protocol.v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start": START.isoformat(), "end_exclusive": END.isoformat(), "warmup_start": WARMUP.isoformat()},
        "source_url": URL, "source": "Energy-Charts.info, Fraunhofer ISE, CC BY 4.0",
        "data_boundary": "Later-year same-country temporal holdout with a different data provider. Revised historical observations, not contemporaneous availability.",
        "clock_semantics": "Provider timestamps label interval starts. Action at t controls [t,t+1h). Latest feature observation labels t-1h and finishes at t. Forecast horizon h predicts the interval ending h hours after issuance. No additional publication latency is assumed.",
        "release_threshold_mw": peak["calibration"]["release_threshold_mw"],
        "policies": ["fixed_upper", "dynamic_upper", "dynamic_release_only", "dynamic_point_only", "dynamic_peak_response", "fixed_peak_response"],
        "parameters": {"capacity_mwh": 30000., "charge_power_mw": 1250., "discharge_power_mw": 1250., "efficiency": .92,
                       "hard_min_fraction": .15, "daily_terminal": True},
        "primary_endpoint": "Full-year controlled maximum of dynamic_peak_response versus dynamic_upper; positive reduction is observed success, zero/negative is reported as no improvement.",
        "secondary_endpoints": ["P95", "quarter-specific maximum and P95", "top20 high-load-hour reduction", "throughput", "conversion losses", "soft-reserve shortfalls", "hard SOC and daily terminal equality", "prediction MAE and interval coverage"],
        "uncertainty": {"method": "Paired circular moving-block bootstrap", "block_days": 7, "repetitions": 2000, "seed": 20211009,
                        "endpoints": ["mean daily peak reduction", "P95 reduction"], "sensitivity_block_days": [3, 14]},
        "rules": ["No retraining, recalibration, threshold search or load rescaling on holdout.", "All six policies share complete UTC days.",
                  "Never impute missing quarter hours; report omitted days.", "Report all quarters and configurations, including negative findings.",
                  "Primary full-period maximum is descriptive; bootstrap intervals do not predict unseen extreme maxima.",
                  "Retain existing exploratory findings separately; confirmation is limited to frozen historical same-country simulation."],
        "frozen_sha256": {name: digest(ROOT / name) for name in FROZEN_FILES}}
    write_json(PROTOCOL, protocol)
    print("Protocol frozen before holdout download: " + str(PROTOCOL), flush=True)


def verify_frozen() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for name, expected in protocol["frozen_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    return protocol


def download() -> None:
    verify_frozen()
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                with urlopen(URL, timeout=120) as response:
                    content = response.read()
                json.loads(content)
                RAW.write_bytes(content)
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == 4:
                    raise
                wait = int(exc.headers.get("Retry-After", "30"))
                print(f"Rate limit: retry in {wait}s", flush=True)
                time.sleep(wait)
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    hourly, quality = normalize_public_power(payload, WARMUP, INPUT_END)
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(DATASET, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    write_json(OUT / "independent_2021_data_quality.json", {**quality, "source_url": URL, "raw_sha256": digest(RAW),
        "dataset_sha256": digest(DATASET), "license": payload.get("license"), "provider_generated_at": payload.get("generated_at"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "source_fields": list(FIELDS_FOR_DOC),
        "evaluation_start": START.isoformat(), "evaluation_end_exclusive": END.isoformat()})
    print(f"Downloaded and normalized {len(hourly)} hourly rows", flush=True)


FIELDS_FOR_DOC = ("load", "solar", "wind_onshore", "wind_offshore")


def evaluate() -> None:
    protocol = verify_frozen()
    supplemental = json.loads((ROOT / "outputs/scenarios/reserve_ablation_metrics.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "outputs/scenarios/storage_scenario_metrics.json").read_text(encoding="utf-8"))["calibration"]
    raw = pd.read_csv(DATASET, parse_dates=["timestamp"])
    direct, forecasts = [], {}
    for h in range(1, 7):
        frame = build_direct_horizon_frame(raw, h)
        selected = frame.loc[frame.timestamp.ge(START) & frame.timestamp.lt(END + pd.Timedelta(hours=6))]
        model = joblib.load(ROOT / f"outputs/models/direct_horizon_{h}h.joblib")["model"]
        predicted = model.predict(selected[list(FEATURE_COLUMNS)])
        # A source observation labelled s covers [s,s+1h) and is known only at
        # its end. Re-label issuance accordingly; never treat it as known at s.
        direct.append(pd.DataFrame({"timestamp": selected.timestamp, "issued_timestamp": selected.issued_timestamp + pd.Timedelta(hours=1),
            "horizon_hours": h, "net_load_mw": selected.net_load_mw, "prediction_mw": predicted, "lower_mw": 0., "upper_mw": 0.}))
        valid = selected.timestamp.lt(END)
        radii = {int(k): v for k,v in supplemental["hourly_interval_calibration"][str(h)].items()}
        radius = selected.timestamp.dt.hour.map(radii).to_numpy()
        truth = selected.net_load_mw.to_numpy()
        forecasts[str(h)] = {**regression_metrics(pd.Series(truth[valid]), pd.Series(predicted[valid])),
            "rows": int(valid.sum()), "interval_coverage_pct": float(np.mean(np.abs(truth[valid]-predicted[valid]) <= radius[valid]) * 100),
            "persistence_mae_mw": float(np.mean(np.abs(truth[valid] - selected.net_load_latest_available_mw.to_numpy()[valid])))}
    predictions = pd.concat(direct, ignore_index=True)
    predictions.to_csv(OUT / "independent_2021_predictions.csv", index=False)
    panel = interval_panel(predictions, supplemental)
    panel = panel.loc[panel.timestamp.ge(START) & panel.timestamp.lt(END)]
    threshold = protocol["release_threshold_mw"]
    policies = {"fixed_upper": {"fixed_reserve_fraction": .30}, "dynamic_upper": {},
        "dynamic_release_only": {"release_threshold_mw": threshold}, "dynamic_point_only": {"planning_objective": "point"},
        "dynamic_peak_response": {"planning_objective": "point", "release_threshold_mw": threshold},
        "fixed_peak_response": {"fixed_reserve_fraction": .30, "planning_objective": "point", "release_threshold_mw": threshold}}
    traces, experiments, quarters = {}, {}, {}
    for name, options in policies.items():
        trace, summary = simulate_balanced_storage(panel, calibration["peak_threshold_mw"], calibration["charge_forecast_threshold_mw"],
            supplemental["uncertainty_scale_mw"], **protocol["parameters"], **options)
        path = OUT / f"independent_2021_{name}_timeseries.csv"
        trace.to_csv(path, index=False)
        summary["timeseries"] = path.relative_to(ROOT).as_posix()
        experiments[name] = evidence(trace, summary)
        traces[name] = trace
        quarters[name] = {str(q): {"days": len(t)//24, "baseline_peak_mw": float(t.net_load_mw.max()),
            "controlled_peak_mw": float(t.controlled_net_load_mw.max()), "controlled_p95_mw": float(t.controlled_net_load_mw.quantile(.95)),
            "throughput_mwh": float((t.charge_mw + t.discharge_mw).sum())} for q,t in trace.groupby(trace.timestamp.dt.quarter)}
        print(f"2021/{name}: maximum={summary['controlled_peak_mw']:.2f}, P95={summary['controlled_p95_mw']:.2f}, days={summary['days']}", flush=True)
    comparisons = {}
    for old, new in (("dynamic_upper", "dynamic_peak_response"), ("fixed_peak_response", "dynamic_peak_response"),
                     ("dynamic_point_only", "dynamic_peak_response")):
        a,b = experiments[old],experiments[new]
        comparisons[f"{new}_versus_{old}"] = {"full_period_peak_reduction_mw": a["controlled_peak_mw"]-b["controlled_peak_mw"],
            "p95_reduction_mw": a["controlled_p95_mw"]-b["controlled_p95_mw"], "throughput_increase_mwh": b["throughput_mwh"]-a["throughput_mwh"],
            "loss_increase_mwh": b["conversion_loss_mwh"]-a["conversion_loss_mwh"],
            "uncertainty": paired_block_summary(traces[old],traces[new])}
    primary = comparisons["dynamic_peak_response_versus_dynamic_upper"]
    sensitivity = {str(block): paired_block_summary(traces["dynamic_upper"],traces["dynamic_peak_response"], block_days=block)
                   for block in protocol["uncertainty"]["sensitivity_block_days"]}
    original = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv", usecols=["timestamp"])
    overlap = set(original.timestamp).intersection(set(raw.loc[raw.timestamp.ge(START)&raw.timestamp.lt(END),"timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")))
    report = {"schema_version": "gridpulse-independent-validation.v1", "protocol_sha256": digest(PROTOCOL),
        "dataset_sha256": digest(DATASET), "freeze_verified": True, "old_dataset_overlap_hours": len(overlap),
        "period": protocol["period"], "source": protocol["source"], "forecast_metrics": forecasts,
        "experiments": experiments, "parameters": protocol["parameters"], "quarters": quarters, "comparisons": comparisons,
        "bootstrap_block_sensitivity": sensitivity, "primary_observed_success": primary["full_period_peak_reduction_mw"] > 1e-6,
        "limitations": ["Frozen later-year same-country historical simulation, not prospective deployment or cross-country validation.",
            "Provider differs from original OPSD. Definitions and revisions may differ; no rescaling or recalibration was performed.",
            "Provider interval-start timestamps are accounted for: issuance is the end of the latest observed hour. Additional publication delays and historical data vintages are not reconstructed.",
            "Seven-day circular bootstrap assumes enough local stationarity; 3/14-day sensitivity is supplied and extreme maximum inference is excluded.",
            "Hard SOC floor, independent UTC days, 50% SOC cap, unrestricted recovery charging and omitted network/price/degradation assumptions still apply."]}
    write_json(OUT / "independent_2021_metrics.json", report)
    print("Independent validation complete; all six configurations retained.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "download", "evaluate"))
    stage = parser.parse_args().stage
    {"freeze": freeze, "download": download, "evaluate": evaluate}[stage]()


if __name__ == "__main__":
    main()
