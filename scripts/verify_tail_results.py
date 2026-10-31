"""Independent NumPy/CSV v4 selection, realized-risk and physical audit."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_dispatch_results import check_trace

OUT = ROOT / "outputs/fusion_v4"
SOURCE = ROOT / "outputs/fusion_v3"
METHODS = ("hgb", "data_seed42", "data_seed43", "data_seed44")
CANDIDATES = (
    "peak_dynamic", "peak_physical", "tail_dynamic_0", "tail_dynamic_250",
    "tail_physical_0", "tail_physical_250",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_check(delta, reported):
    daily = np.asarray(delta, float)
    rng = np.random.default_rng(20261026)
    boot = []
    for _ in range(2000):
        starts = rng.integers(0, len(daily), size=(len(daily) + 6) // 7)
        indices = ((starts[:, None] + np.arange(7)) % len(daily)).ravel()[:len(daily)]
        boot.append(daily[indices].mean())
    assert abs(daily.mean() - reported["estimate_mw"]) < 1e-5
    assert np.allclose(np.quantile(boot, [.025, .975]),
                       [reported["lower_95_mw"], reported["upper_95_mw"]], atol=1e-5)


def verify_selection(validation, receipt):
    eligible, decisions = [], {}
    for control in CANDIDATES:
        guards, p95s, losses, deltas = {}, [], [], []
        for seed in (42, 43, 44):
            a, b = validation[control][f"data_seed{seed}"], validation["legacy"][f"data_seed{seed}"]
            guards[str(seed)] = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"]),
                "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                "throughput": bool(a["throughput_mwh"] <= 1.1 * b["throughput_mwh"] + 1e-6),
                "loss": bool(a["conversion_loss_mwh"] <= 1.1 * b["conversion_loss_mwh"] + 1e-6),
                "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
            }
            p95s.append(a["controlled_p95_mw"])
            losses.append(a["conversion_loss_mwh"])
            deltas.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
        is_eligible = bool(all(all(g.values()) for g in guards.values()) and np.mean(deltas) <= -1.)
        if is_eligible:
            eligible.append(control)
        expected = receipt["candidates"][control]
        assert expected["per_seed_guards"] == guards
        assert expected["eligible"] == is_eligible
        assert abs(expected["mean_p95_delta_mw"] - np.mean(deltas)) < 1e-5
        assert abs(expected["mean_p95_mw"] - np.mean(p95s)) < 1e-5
        assert abs(expected["mean_loss_mwh"] - np.mean(losses)) < 1e-5
        decisions[control] = (float(np.mean(p95s)), float(np.mean(losses)), control)
    selected = min(eligible, key=lambda name: decisions[name]) if eligible else "legacy"
    assert receipt["selected_policy"] == selected and receipt["fallback"] == (not eligible)
    assert receipt["selection_predictors"] == ["data_seed42", "data_seed43", "data_seed44"]
    return selected


def risk_metrics(trace, metrics, gates, tail):
    actual, controlled = trace.net_load_mw, trace.controlled_net_load_mw
    expected = {
        "baseline_p95_mw": float(actual.quantile(.95)),
        "baseline_mean_excess_mw": float(np.maximum(actual - tail, 0.).mean()),
        "controlled_mean_excess_mw": float(np.maximum(controlled - tail, 0.).mean()),
        "actual_high_load_charging_mwh": float(trace.loc[actual >= gates["peak_threshold_mw"], "charge_mw"].sum()),
        "actual_high_load_charging_hours": int(((actual >= gates["peak_threshold_mw"]) & trace.charge_mw.gt(1e-6)).sum()),
    }
    for key, value in expected.items():
        assert abs(value - metrics[key]) < 1e-5, key
    if "recovery_charge_mwh" in trace:
        assert abs(trace.recovery_charge_mwh.sum() - metrics["recovery_charge_mwh"]) < 1e-5
    else:
        assert metrics["recovery_charge_mwh"] is None


def check_forecast_and_reserve(trace, frame, cal, gates, model, control):
    forecast = frame.set_index("timestamp").loc[trace.timestamp]
    assert np.allclose(trace.net_load_mw, forecast.actual_h1_mw, rtol=0, atol=1e-6)
    assert np.allclose(trace.prediction_h1_mw, forecast.prediction_h1_mw, rtol=0, atol=1e-6)
    radii = trace.timestamp.dt.hour.map({int(k): v for k, v in cal["radii"]["1"].items()})
    assert np.allclose(trace.upper_h1_mw, trace.prediction_h1_mw + radii)
    recovery = np.where(trace.prediction_h1_mw - radii > gates["charge_threshold_mw"],
                        trace.charge_mw, 0.).sum()
    assert abs(recovery - trace.recovery_charge_mwh.sum()) < 1e-5
    if control == "legacy":
        return
    point = forecast[[f"prediction_h{h}_mw" for h in range(1, 7)]].to_numpy()
    radius = np.column_stack([
        (trace.timestamp + pd.Timedelta(hours=h - 1)).dt.hour.map({
            int(k): v for k, v in cal["radii"][str(h)].items()}).to_numpy() for h in range(1, 7)
    ])
    available = np.arange(6)[None, :] < (24 - trace.timestamp.dt.hour.to_numpy())[:, None]
    upper_max = np.max(np.where(available, point + radius, -np.inf), axis=1)
    point_max = np.max(np.where(available, point, -np.inf), axis=1)
    radius_max = np.max(np.where(available, radius, -np.inf), axis=1)
    severity = np.clip((upper_max - gates["peak_threshold_mw"]) / max(.1 * gates["peak_threshold_mw"], 1.), 0., 1.)
    uncertainty = np.clip(radius_max / cal["uncertainty_scale_mw"], 0., 1.)
    fraction = np.clip(.15 + .2 * severity + .15 * uncertainty, 0., .6)
    target = (np.full(len(trace), model["minimum_soc_mwh"]) if "physical" in control
              else fraction * model["capacity_mwh"])
    released = point_max >= gates["release_threshold_mw"]
    floor = np.where(released, model["minimum_soc_mwh"],
                     np.maximum(model["minimum_soc_mwh"], np.minimum(target, trace.soc_before_mwh)))
    expected_next = np.maximum(
        floor, model["initial_soc_mwh"]
        - (23 - trace.timestamp.dt.hour) * model["charge_power_mw"] * model["efficiency"])
    assert np.allclose(trace.reserve_target_mwh, target)
    assert np.array_equal(trace.reserve_release_active, released)
    assert np.allclose(trace.required_next_soc_mwh, expected_next)
    assert np.allclose(trace.advisory_floor_mwh, floor)
    tolerance = 250. if control.endswith("_250") else 0.
    assert np.allclose(trace.window_peak_cap_mw, trace.stage1_peak_mw + tolerance)
    assert (trace.planned_max_mw <= trace.window_peak_cap_mw + 1e-5).all()
    assert (trace.prediction_h1_mw + trace.charge_mw - trace.discharge_mw
            <= trace.planned_max_mw + 1e-5).all()


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected, relative
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for name, expected in manifest["validation_input_sha256"].items():
        assert sha(SOURCE / name) == expected
    for name, expected in read(OUT / "evaluation_inputs.json").items():
        assert sha(SOURCE / name) == expected
    assert sha(SOURCE / "calibration.json") == manifest["source_calibration_sha256"]
    assert sha(SOURCE / "samples.json") == manifest["source_samples_sha256"]
    assert read(OUT / "validation_failures.json") == read(OUT / "evaluation_failures.json") == []
    assert read(OUT / "status.json")["status"] == "complete"
    calibration, samples = read(OUT / "calibration.json"), read(OUT / "samples.json")
    original_cal = read(SOURCE / "calibration.json")
    assert {k: v for k, v in calibration.items() if k != "tail_threshold_mw"} == original_cal
    tail, gates = calibration["tail_threshold_mw"], calibration["gates"]
    inputs = {}
    for period in ("validation", "test", "2021", "2022"):
        inputs[period] = {}
        for method in METHODS:
            frame = pd.read_csv(SOURCE / "predictions" / method / f"{period}.csv",
                                parse_dates=["timestamp"])
            frame = frame.loc[frame.timestamp.between(
                pd.Timestamp(samples[period]["start"]), pd.Timestamp(samples[period]["end"]))]
            assert len(frame) == samples[period]["rows"]
            if period == "validation":
                assert frame.timestamp.min() >= pd.Timestamp("2019-01-12T20:00:00Z")
                assert frame.timestamp.max() + pd.Timedelta(hours=5) < pd.Timestamp("2019-08-09T21:00:00Z")
                assert abs(frame.actual_h1_mw.quantile(.9) - tail) < 1e-6
            inputs[period][method] = frame
    receipt = read(OUT / "selection.json")
    assert receipt["validation_results_sha256"] == sha(OUT / "validation_results.json")
    assert receipt["protocol_sha256"] == sha(ROOT / "experiments/fusion_v4/protocol.md")
    assert receipt["validation_input_sha256"] == manifest["validation_input_sha256"]
    evaluation_start = read(OUT / "evaluation_start.json")
    assert evaluation_start["selection_sha256"] == sha(OUT / "selection.json")
    assert datetime.fromisoformat(receipt["frozen_utc"]) <= datetime.fromisoformat(evaluation_start["started_utc"])
    validation = read(OUT / "validation_results.json")
    report = read(OUT / "metrics.json")
    groups = {"validation": validation, **report["methods"]}
    frames, total = {}, 0
    original_report = read(SOURCE / "metrics.json")
    for period, controls in groups.items():
        assert set(controls) == {"legacy", *CANDIDATES}
        frames[period] = {}
        for control, methods in controls.items():
            assert set(methods) == set(METHODS)
            frames[period][control] = {}
            for method, metrics in methods.items():
                trace = check_trace(OUT / metrics["trajectory"], metrics,
                                    terminal=True, continuous=False)["frame"]
                assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                risk_metrics(trace, metrics, gates, tail)
                check_forecast_and_reserve(trace, inputs[period][method], calibration["methods"][method],
                                          gates, metrics["physical_parameters"], control)
                mae = np.abs(trace.net_load_mw - trace.prediction_h1_mw).mean()
                assert abs(mae - metrics["forecast_h1_mae_mw"]) < 1e-6
                assert len(trace) == samples[period]["rows"]
                if period != "validation" and control == "legacy":
                    original = original_report["methods"][period][f"{method}_w6"]
                    assert sha(OUT / metrics["trajectory"]) == sha(SOURCE / original["trajectory"])
                frames[period][control][method] = trace
                total += 1
    selected = verify_selection(validation, receipt)
    assert report["selected_policy"] == selected
    assert report["selection_sha256"] == sha(OUT / "selection.json")
    interval_count = 0
    for period, controls in report["comparisons"]["all_paired"].items():
        for control, methods in controls.items():
            for method, comparison in methods.items():
                a, b = report["methods"][period][control][method], report["methods"][period]["legacy"][method]
                for key, value in comparison["differences"].items():
                    assert abs(a[key] - b[key] - value) < 1e-5
                fa, fb = frames[period][control][method], frames[period]["legacy"][method]
                da, db = fa.timestamp.dt.floor("D"), fb.timestamp.dt.floor("D")
                peak_delta = (fa.groupby(da).controlled_net_load_mw.max()
                              - fb.groupby(db).controlled_net_load_mw.max()).to_numpy()
                excess_delta = (np.maximum(fa.controlled_net_load_mw - tail, 0.).groupby(da).mean()
                                - np.maximum(fb.controlled_net_load_mw - tail, 0.).groupby(db).mean()).to_numpy()
                interval_check(peak_delta, comparison["daily_peak_interval"])
                interval_check(excess_delta, comparison["daily_excess_interval"])
                interval_count += 2
    for period, comparison in report["comparisons"]["selected_data_seed_mean"].items():
        peaks, excess = [], []
        for seed in (42, 43, 44):
            method = f"data_seed{seed}"
            fa, fb = frames[period][selected][method], frames[period]["legacy"][method]
            da, db = fa.timestamp.dt.floor("D"), fb.timestamp.dt.floor("D")
            peaks.append((fa.groupby(da).controlled_net_load_mw.max()
                          - fb.groupby(db).controlled_net_load_mw.max()).to_numpy())
            excess.append((np.maximum(fa.controlled_net_load_mw - tail, 0.).groupby(da).mean()
                           - np.maximum(fb.controlled_net_load_mw - tail, 0.).groupby(db).mean()).to_numpy())
        for key, value in comparison["fixed_seed_mean_differences"].items():
            delta = [report["methods"][period][selected][f"data_seed{s}"][key]
                     - report["methods"][period]["legacy"][f"data_seed{s}"][key] for s in (42, 43, 44)]
            assert abs(np.mean(delta) - value) < 1e-5
        interval_check(np.mean(peaks, axis=0), comparison["daily_peak_interval"])
        interval_check(np.mean(excess, axis=0), comparison["daily_excess_interval"])
        interval_count += 2
    stress_count = 0
    for period, scenarios in report["stress"].items():
        for scenario, arms in scenarios.items():
            pair = {}
            for arm, metrics in arms.items():
                trace = check_trace(OUT / metrics["trajectory"], metrics,
                                    terminal=arm == "replan", continuous=True)["frame"]
                assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                risk_metrics(trace, metrics, gates, tail)
                pair[arm] = trace
                stress_count += 1
            assert pair["replay"].timestamp.equals(pair["replan"].timestamp)
            assert np.allclose(pair["replay"].net_load_mw, pair["replan"].net_load_mw)
            assert arms["replay"]["physical_parameters"] == arms["replan"]["physical_parameters"]
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    result = {
        "passed": True, "artifact_hashes_verified": len(hashes),
        "validation_trajectories": 28, "nominal_trajectories": total - 28,
        "stress_trajectories": stress_count, "selection_rule_recomputed": True,
        "selected_policy": selected, "legacy_byte_identical_traces": 12,
        "paired_block_intervals_recomputed": interval_count,
        "input_forecast_files": len(METHODS) * 4,
        "boundary": "CSV/selection/physics audit, not real-world or untouched-year validation.",
    }
    (ROOT / "outputs/fusion_v4_local_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
