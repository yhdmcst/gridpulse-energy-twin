"""Independent local audit of v5 causal reference constraints and admission."""
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
from scripts.verify_tail_results import interval_check, risk_metrics

OUT = ROOT / "outputs/fusion_v5"
PREDICTIONS = ROOT / "outputs/fusion_v3"
REFERENCE = ROOT / "outputs/fusion_v4"
CANDIDATES = ("tail_unanchored", "tail_debt", "tail_guard", "peak_debt", "peak_guard")
METHODS = ("hgb", "data_seed42", "data_seed43", "data_seed44")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_check(validation, receipt):
    eligible, ordering = [], {}
    for name in CANDIDATES:
        all_guards, p95, losses, deltas = {}, [], [], []
        for seed in (42, 43, 44):
            a, b = validation[name][f"data_seed{seed}"], validation["legacy"][f"data_seed{seed}"]
            all_guards[str(seed)] = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"]),
                "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                "throughput": bool(a["throughput_mwh"] <= b["throughput_mwh"] * 1.1 + 1e-6),
                "loss": bool(a["conversion_loss_mwh"] <= b["conversion_loss_mwh"] * 1.1 + 1e-6),
                "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
            }
            p95.append(a["controlled_p95_mw"])
            losses.append(a["conversion_loss_mwh"])
            deltas.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
        passed = bool(np.mean(deltas) <= -1. and all(all(g.values()) for g in all_guards.values()))
        reported = receipt["candidates"][name]
        assert reported["per_seed_guards"] == all_guards and reported["eligible"] == passed
        for key, value in (("mean_p95_mw", np.mean(p95)), ("mean_loss_mwh", np.mean(losses)),
                           ("mean_p95_delta_mw", np.mean(deltas))):
            assert abs(reported[key] - value) < 1e-5
        if passed:
            eligible.append(name)
        ordering[name] = (float(np.mean(p95)), float(np.mean(losses)), name)
    selected = min(eligible, key=lambda n: ordering[n]) if eligible else "legacy"
    assert receipt["selected_policy"] == selected and receipt["fallback"] == (not eligible)
    assert receipt["selection_predictors"] == ["data_seed42", "data_seed43", "data_seed44"]
    return selected


def aligned_forecast(trace, frame, cal, gates, metrics):
    forecast = frame.set_index("timestamp").loc[trace.timestamp]
    assert np.allclose(trace.net_load_mw, forecast.actual_h1_mw, rtol=0, atol=1e-6)
    assert np.allclose(trace.prediction_h1_mw, forecast.prediction_h1_mw, rtol=0, atol=1e-6)
    radii = trace.timestamp.dt.hour.map({int(k): v for k, v in cal["radii"]["1"].items()})
    assert np.allclose(trace.upper_h1_mw, trace.prediction_h1_mw + radii)
    recovery = np.where(trace.prediction_h1_mw - radii > gates["charge_threshold_mw"],
                        trace.charge_mw, 0.).sum()
    assert abs(recovery - metrics["recovery_charge_mwh"]) < 1e-5
    assert abs((trace.net_load_mw - trace.prediction_h1_mw).abs().mean()
               - metrics["forecast_h1_mae_mw"]) < 1e-5
    return forecast


def governor_check(trace, reference, forecast, cal, gates, tail, metrics, name, model):
    assert trace.timestamp.equals(reference.timestamp)
    for target, source in (
        ("reference_soc_before_mwh", "soc_before_mwh"),
        ("reference_soc_after_mwh", "storage_soc_mwh"),
        ("reference_charge_mw", "charge_mw"), ("reference_discharge_mw", "discharge_mw"),
    ):
        assert np.allclose(trace[target], reference[source], rtol=0, atol=1e-6)
    debt = np.maximum(reference.storage_soc_mwh - trace.storage_soc_mwh, 0.)
    assert np.allclose(trace.reference_debt_mwh, debt)
    assert abs(float(debt.max()) - metrics["max_reference_debt_mwh"]) < 1e-5
    assert int((debt > 1e-5).sum()) == metrics["debt_hours"]
    anchored = name != "tail_unanchored"
    guard = name in ("tail_guard", "peak_guard")
    if anchored:
        assert debt.max() <= 1e-5
        assert (trace.soc_before_mwh >= reference.soc_before_mwh - 1e-5).all()
    point = forecast[[f"prediction_h{h}_mw" for h in range(1, 7)]].to_numpy()
    radius = np.column_stack([
        (trace.timestamp + pd.Timedelta(hours=h - 1)).dt.hour.map(
            {int(k): v for k, v in cal["radii"][str(h)].items()}).to_numpy()
        for h in range(1, 7)
    ])
    mask = np.arange(6)[None, :] < (24 - trace.timestamp.dt.hour.to_numpy())[:, None]
    point_max = np.max(np.where(mask, point, -np.inf), axis=1)
    upper_max = np.max(np.where(mask, point + radius, -np.inf), axis=1)
    radius_max = np.max(np.where(mask, radius, -np.inf), axis=1)
    fraction = np.clip(
        .15 + .2 * np.clip((upper_max - gates["peak_threshold_mw"])
                          / max(.1 * gates["peak_threshold_mw"], 1.), 0., 1.)
        + .15 * np.clip(radius_max / cal["uncertainty_scale_mw"], 0., 1.), 0., .6)
    target = fraction * model["capacity_mwh"]
    released = point_max >= gates["release_threshold_mw"]
    floor = np.where(released, model["minimum_soc_mwh"],
                     np.maximum(model["minimum_soc_mwh"], np.minimum(target, trace.soc_before_mwh)))
    required = np.maximum(
        floor, model["initial_soc_mwh"]
        - (23 - trace.timestamp.dt.hour) * model["charge_power_mw"] * model["efficiency"])
    if anchored:
        required = np.maximum(required, reference.storage_soc_mwh)
    assert np.allclose(trace.reserve_target_mwh, target)
    assert np.array_equal(trace.reserve_release_active, released)
    assert np.allclose(trace.advisory_floor_mwh, floor)
    assert np.allclose(trace.required_next_soc_mwh, required)
    assert (trace.storage_soc_mwh >= required - 1e-5).all()
    protected_count = 0
    if guard:
        net_ref = reference.charge_mw - reference.discharge_mw
        cap = np.maximum(net_ref, tail - trace.upper_h1_mw)
        violation = np.maximum(trace.charge_mw - trace.discharge_mw - cap, 0.)
        assert np.allclose(trace.headroom_cap_mw, cap)
        assert violation.max() <= 1e-5
        assert abs(metrics["max_headroom_violation_mw"] - violation.max()) < 1e-5
        protected = trace.upper_h1_mw + net_ref >= tail
        assert (trace.loc[protected, "controlled_net_load_mw"].to_numpy()
                <= reference.loc[protected, "controlled_net_load_mw"].to_numpy() + 1e-5).all()
        protected_count = int(protected.sum())
    else:
        assert trace.headroom_cap_mw.isna().all()
        assert metrics["max_headroom_violation_mw"] is None
    return protected_count


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected, relative
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in manifest["validation_prediction_sha256"].items():
        assert sha(PREDICTIONS / relative) == expected
    for relative, expected in manifest["validation_reference_sha256"].items():
        assert sha(REFERENCE / relative) == expected
    inputs = read(OUT / "evaluation_inputs.json")
    for relative, expected in inputs["predictions"].items():
        assert sha(PREDICTIONS / relative) == expected
    for relative, expected in inputs["reference"].items():
        assert sha(REFERENCE / relative) == expected
    assert sha(REFERENCE / "calibration.json") == manifest["source_calibration_sha256"]
    assert sha(REFERENCE / "samples.json") == manifest["source_samples_sha256"]
    calibration, samples = read(OUT / "calibration.json"), read(OUT / "samples.json")
    assert calibration == read(REFERENCE / "calibration.json")
    assert samples == read(REFERENCE / "samples.json")
    tail, gates = calibration["tail_threshold_mw"], calibration["gates"]
    assert read(OUT / "validation_failures.json") == read(OUT / "evaluation_failures.json") == []
    assert read(OUT / "status.json")["status"] == "complete"
    selection = read(OUT / "selection.json")
    assert sha(OUT / "validation_results.json") == selection["validation_results_sha256"]
    assert sha(ROOT / "experiments/fusion_v5/protocol.md") == selection["protocol_sha256"]
    assert selection["validation_prediction_sha256"] == manifest["validation_prediction_sha256"]
    assert selection["validation_reference_sha256"] == manifest["validation_reference_sha256"]
    started = read(OUT / "evaluation_start.json")
    assert started["selection_sha256"] == sha(OUT / "selection.json")
    assert datetime.fromisoformat(selection["frozen_utc"]) <= datetime.fromisoformat(started["started_utc"])
    validation, report = read(OUT / "validation_results.json"), read(OUT / "metrics.json")
    predictions, traces, total, protected_count = {}, {}, 0, 0
    groups = {"validation": validation, **report["methods"]}
    ref_validation, ref_results = read(REFERENCE / "validation_results.json"), read(REFERENCE / "metrics.json")
    for period, controls in groups.items():
        assert set(controls) == {"legacy", *CANDIDATES}
        predictions[period], traces[period] = {}, {}
        for method in METHODS:
            frame = pd.read_csv(PREDICTIONS / "predictions" / method / f"{period}.csv",
                                parse_dates=["timestamp"])
            frame = frame.loc[frame.timestamp.between(
                pd.Timestamp(samples[period]["start"]), pd.Timestamp(samples[period]["end"]))]
            assert len(frame) == samples[period]["rows"]
            if period == "validation":
                assert frame.timestamp.min() >= pd.Timestamp("2019-01-12T20:00:00Z")
                assert frame.timestamp.max() + pd.Timedelta(hours=5) < pd.Timestamp("2019-08-09T21:00:00Z")
                assert abs(frame.actual_h1_mw.quantile(.9) - tail) < 1e-6
            predictions[period][method] = frame
        for name in ("legacy", *CANDIDATES):
            methods = controls[name]
            assert set(methods) == set(METHODS)
            traces[period][name] = {}
            for method, metrics in methods.items():
                trace = check_trace(OUT / metrics["trajectory"], metrics,
                                    terminal=True, continuous=False)["frame"]
                assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                risk_metrics(trace, metrics, gates, tail)
                forecast = aligned_forecast(trace, predictions[period][method],
                                            calibration["methods"][method], gates, metrics)
                if name == "legacy":
                    old = (ref_validation["legacy"][method] if period == "validation"
                           else ref_results["methods"][period]["legacy"][method])
                    assert sha(OUT / metrics["trajectory"]) == sha(REFERENCE / old["trajectory"])
                else:
                    protected_count += governor_check(
                        trace, traces[period]["legacy"][method], forecast,
                        calibration["methods"][method], gates, tail, metrics, name,
                        metrics["physical_parameters"])
                traces[period][name][method] = trace
                total += 1
    selected = selection_check(validation, selection)
    assert report["selected_policy"] == selected and report["selection_sha256"] == sha(OUT / "selection.json")
    intervals = 0
    for period, controls in report["comparisons"]["all_paired"].items():
        for name, methods in controls.items():
            for method, comparison in methods.items():
                a, b = report["methods"][period][name][method], report["methods"][period]["legacy"][method]
                for key, expected in comparison["differences"].items():
                    assert abs(a[key] - b[key] - expected) < 1e-5
                fa, fb = traces[period][name][method], traces[period]["legacy"][method]
                da, db = fa.timestamp.dt.floor("D"), fb.timestamp.dt.floor("D")
                interval_check((fa.groupby(da).controlled_net_load_mw.max()
                                - fb.groupby(db).controlled_net_load_mw.max()).to_numpy(),
                               comparison["daily_peak_interval"])
                interval_check((np.maximum(fa.controlled_net_load_mw - tail, 0.).groupby(da).mean()
                                - np.maximum(fb.controlled_net_load_mw - tail, 0.).groupby(db).mean()).to_numpy(),
                               comparison["daily_excess_interval"])
                intervals += 2
    for period, comparison in report["comparisons"]["selected_data_seed_mean"].items():
        peak, excess = [], []
        for seed in (42, 43, 44):
            method = f"data_seed{seed}"
            a, b = traces[period][selected][method], traces[period]["legacy"][method]
            da, db = a.timestamp.dt.floor("D"), b.timestamp.dt.floor("D")
            peak.append((a.groupby(da).controlled_net_load_mw.max()
                         - b.groupby(db).controlled_net_load_mw.max()).to_numpy())
            excess.append((np.maximum(a.controlled_net_load_mw - tail, 0.).groupby(da).mean()
                           - np.maximum(b.controlled_net_load_mw - tail, 0.).groupby(db).mean()).to_numpy())
        for key, expected in comparison["fixed_seed_mean_differences"].items():
            values = [report["methods"][period][selected][f"data_seed{s}"][key]
                      - report["methods"][period]["legacy"][f"data_seed{s}"][key] for s in (42, 43, 44)]
            assert abs(np.mean(values) - expected) < 1e-5
        interval_check(np.mean(peak, axis=0), comparison["daily_peak_interval"])
        interval_check(np.mean(excess, axis=0), comparison["daily_excess_interval"])
        intervals += 2
    stress_count, reference_count = 0, 0
    for period, scenarios in report["stress"].items():
        for scenario, arms in scenarios.items():
            ref_m = report["changed_references"][period][scenario]
            ref = check_trace(OUT / ref_m["trajectory"], ref_m, terminal=True, continuous=True)["frame"]
            risk_metrics(ref, ref_m, gates, tail)
            reference_count += 1
            pair = {}
            for arm, metrics in arms.items():
                trace = check_trace(OUT / metrics["trajectory"], metrics,
                                    terminal=arm == "replan", continuous=True)["frame"]
                assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                risk_metrics(trace, metrics, gates, tail)
                if arm == "replan" and selected != "legacy":
                    forecast = aligned_forecast(trace, predictions[period]["data_seed42"],
                                                calibration["methods"]["data_seed42"], gates, metrics)
                    protected_count += governor_check(
                        trace, ref, forecast, calibration["methods"]["data_seed42"],
                        gates, tail, metrics, selected, metrics["physical_parameters"])
                pair[arm] = trace
                stress_count += 1
            assert pair["replay"].timestamp.equals(pair["replan"].timestamp)
            assert np.allclose(pair["replay"].net_load_mw, pair["replan"].net_load_mw)
            assert arms["replay"]["physical_parameters"] == arms["replan"]["physical_parameters"] == ref_m["physical_parameters"]
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    result = {
        "passed": True, "artifact_hashes_verified": len(hashes),
        "validation_trajectories": 24, "evaluation_trajectories": total - 24,
        "stress_trajectories": stress_count, "changed_reference_trajectories": reference_count,
        "byte_identical_legacy_references": 16, "selection_rule_recomputed": True,
        "selected_policy": selected, "paired_intervals": intervals,
        "protected_hour_checks": protected_count,
        "boundary": "Modeled reference constraints verified; calibrated bands do not guarantee actual future safety.",
    }
    (ROOT / "outputs/fusion_v5_local_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
