"""Independent NumPy audit of v6 timing, actual plant execution and UI decision."""
from __future__ import annotations

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

OUT = ROOT / "outputs/fusion_v6"
V3 = ROOT / "outputs/fusion_v3"
V5 = ROOT / "outputs/fusion_v5"
MODES = {"fresh6": (0, 6), "fresh5": (0, 5), "fresh3": (0, 3),
         "delayed1": (1, 5), "delayed3": (3, 3)}
MAIN = ("fresh6", "delayed1", "delayed3")
METHODS = ("data_seed42", "data_seed43", "data_seed44")
ERRORS = ("power_90pct", "power_70pct", "efficiency_090", "efficiency_085")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard_check(trace, reference, panel, cal, gates, tail, metrics):
    model = metrics["physical_parameters"]
    for target, source in (
        ("reference_soc_before_mwh", "soc_before_mwh"), ("reference_soc_after_mwh", "storage_soc_mwh"),
        ("reference_charge_mw", "charge_mw"), ("reference_discharge_mw", "discharge_mw"),
    ):
        assert np.allclose(trace[target], reference[source], atol=1e-6, rtol=0)
    assert (trace.soc_before_mwh >= reference.soc_before_mwh - 1e-5).all()
    debt = np.maximum(reference.storage_soc_mwh - trace.storage_soc_mwh, 0.)
    assert debt.max() <= 1e-5 and np.allclose(trace.reference_debt_mwh, debt)
    assert abs(metrics["max_reference_debt_mwh"] - debt.max()) < 1e-5
    cap = np.maximum(reference.charge_mw - reference.discharge_mw, tail - trace.upper_h1_mw)
    violation = np.maximum(trace.charge_mw - trace.discharge_mw - cap, 0.)
    assert np.allclose(trace.headroom_cap_mw, cap) and violation.max() <= 1e-5
    assert abs(metrics["max_headroom_violation_mw"] - violation.max()) < 1e-5
    horizons = int(panel.forecast_hours.iloc[0])
    point = panel[[f"prediction_h{h}_mw" for h in range(1, horizons + 1)]].to_numpy()
    upper = panel[[f"upper_h{h}_mw" for h in range(1, horizons + 1)]].to_numpy()
    mask = np.arange(horizons)[None, :] < (24 - panel.timestamp.dt.hour.to_numpy())[:, None]
    pmax = np.max(np.where(mask, point, -np.inf), axis=1)
    umax = np.max(np.where(mask, upper, -np.inf), axis=1)
    rmax = np.max(np.where(mask, upper - point, -np.inf), axis=1)
    fraction = np.clip(
        .15 + .2 * np.clip((umax - gates["peak_threshold_mw"]) / max(.1 * gates["peak_threshold_mw"], 1.), 0., 1.)
        + .15 * np.clip(rmax / cal["uncertainty_scale_mw"], 0., 1.), 0., .6)
    target = fraction * model["capacity_mwh"]
    released = pmax >= gates["release_threshold_mw"]
    floor = np.where(released, model["minimum_soc_mwh"],
                     np.maximum(model["minimum_soc_mwh"], np.minimum(target, trace.soc_before_mwh)))
    required = np.maximum(
        np.maximum(floor, model["initial_soc_mwh"]
                   - (23 - trace.timestamp.dt.hour) * model["charge_power_mw"] * model["efficiency"]),
        reference.storage_soc_mwh)
    assert np.allclose(trace.reserve_target_mwh, target)
    assert np.array_equal(trace.reserve_release_active, released)
    assert np.allclose(trace.advisory_floor_mwh, floor)
    assert np.allclose(trace.required_next_soc_mwh, required)
    assert (trace.storage_soc_mwh >= required - 1e-5).all()


def decision_check(delays, mismatches, reported):
    failed, guards, errors = [], {}, {}
    for year in ("2021", "2022"):
        guards[year], errors[year] = {}, {}
        for mode in MAIN:
            details, deltas = {}, []
            for seed in (42, 43, 44):
                pair = delays[year][mode][f"data_seed{seed}"]
                a, b = pair["tail_guard"], pair["legacy"]
                checks = {
                    "finite": bool(np.isfinite([m[k] for m in (a, b) for k in (
                        "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh",
                        "conversion_loss_mwh", "max_abs_terminal_error_mwh")]).all()),
                    "physical": bool(a["physical_audit_passed"] and b["physical_audit_passed"]),
                    "terminal": bool(a["terminal_requirement_passed"] and b["terminal_requirement_passed"]),
                    "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                    "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
                    "throughput": bool(a["throughput_mwh"] <= 1.1 * b["throughput_mwh"] + 1e-6),
                    "loss": bool(a["conversion_loss_mwh"] <= 1.1 * b["conversion_loss_mwh"] + 1e-6),
                }
                details[str(seed)] = checks
                deltas.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
                for key, passed in checks.items():
                    if not passed:
                        failed.append(f"delay/{year}/{mode}/seed{seed}/{key}")
            mean_delta = sum(deltas) / 3
            if not np.isfinite(mean_delta) or mean_delta > -1.:
                failed.append(f"delay/{year}/{mode}/mean_p95")
            guards[year][mode] = {"per_seed": details, "mean_p95_delta_mw": mean_delta}
    for year, modes in mismatches.items():
        for mode, cases in modes.items():
            errors[year][mode] = {}
            for case, pair in cases.items():
                a = pair["tail_guard"]
                checks = {
                    "physical": bool(a["physical_audit_passed"]),
                    "terminal": bool(a["terminal_requirement_passed"]
                                     and np.isfinite(a["max_abs_terminal_error_mwh"])
                                     and a["max_abs_terminal_error_mwh"] <= 1e-5),
                }
                errors[year][mode][case] = checks
                for key, passed in checks.items():
                    if not passed:
                        failed.append(f"mismatch/{year}/{mode}/{case}/{key}")
    assert reported["delay_guards"] == guards
    assert reported["mismatch_guards"] == errors
    assert reported["failed_conditions"] == failed
    assert reported["default_realtime_demo_admitted"] == (not failed)
    return not failed


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected, relative
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in manifest["input_forecast_sha256"].items():
        assert sha(V3 / relative) == expected
    for name, expected in manifest["data_sha256"].items():
        assert sha(ROOT / "data/processed" / name) == expected
    for relative, expected in read(OUT / "v5_reference_hashes.json").items():
        assert sha(V5 / relative) == expected
    assert sha(V5 / "calibration.json") == manifest["calibration_sha256"]
    assert sha(V5 / "selection.json") == manifest["v5_selection_sha256"]
    assert sha(V5 / "metrics.json") == manifest["v5_metrics_sha256"]
    assert read(V5 / "selection.json")["selected_policy"] == "tail_guard"
    assert read(OUT / "delay_failures.json") == read(OUT / "known_failures.json") == []
    assert read(OUT / "status.json")["status"] == "complete"
    calibration, samples, report = read(OUT / "calibration.json"), read(OUT / "samples.json"), read(OUT / "metrics.json")
    assert calibration == read(V5 / "calibration.json")
    tail, gates = calibration["tail_threshold_mw"], calibration["gates"]
    panels, truth, forecast_files = {}, {}, 0
    metadata = ["timestamp", "source_first_target_timestamp", "issued_timestamp", "available_timestamp"]
    for year in ("2021", "2022"):
        truth[year] = pd.read_csv(ROOT / f"data/processed/independent_de_{year}_hourly.csv",
                                  parse_dates=["timestamp"]).set_index("timestamp")
        panels[year] = {}
        common_times = None
        for method in METHODS:
            source = pd.read_csv(V3 / "predictions" / method / f"{year}.csv",
                                 parse_dates=["timestamp", "issued_timestamp"]).set_index("timestamp")
            panels[year][method] = {}
            forecast_files += 1
            for mode, (delay, horizon) in MODES.items():
                panel = pd.read_csv(OUT / "panels" / year / method / f"{mode}.csv", parse_dates=metadata)
                times = pd.DatetimeIndex(panel.timestamp)
                assert len(panel) == samples[year]["rows"]
                assert times.is_monotonic_increasing and not times.has_duplicates
                if common_times is None:
                    common_times = times
                assert times.equals(common_times)
                assert panel.observation_delay_hours.eq(delay).all() and panel.forecast_hours.eq(horizon).all()
                origins = times - pd.Timedelta(hours=delay)
                assert pd.DatetimeIndex(panel.source_first_target_timestamp).equals(origins)
                assert pd.DatetimeIndex(panel.issued_timestamp).equals(times - pd.Timedelta(hours=delay + 1))
                assert pd.DatetimeIndex(panel.available_timestamp).equals(times)
                old = source.loc[origins]
                assert np.allclose(panel.net_load_mw, truth[year].loc[times].net_load_mw, rtol=0, atol=1e-6)
                for h in range(1, horizon + 1):
                    expected = old[f"prediction_h{delay + h}_mw"].to_numpy()
                    hours = (panel.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
                    radius = hours.map({int(k): v for k, v in calibration["methods"][method]["radii"][str(delay + h)].items()})
                    assert np.allclose(panel[f"prediction_h{h}_mw"], expected, atol=1e-6, rtol=0)
                    assert np.allclose(panel[f"lower_h{h}_mw"], expected - radius)
                    assert np.allclose(panel[f"upper_h{h}_mw"], expected + radius)
                panels[year][method][mode] = panel
        # Independently reconstruct the common timestamps before the complete-day filter.
        original_times = source.index
        common = original_times
        for delay, _ in MODES.values():
            common = common.intersection(original_times + pd.Timedelta(hours=delay))
        common = common[common.isin(original_times)]
        counts = pd.Series(1, index=common).groupby(common.floor("D")).sum()
        full_days = counts.loc[counts.eq(24)].index
        assert common_times.equals(common[common.floor("D").isin(full_days)])
    trace_frames, delay_count, known_count, mismatch_count = {}, 0, 0, 0
    old_metrics, replication = read(V5 / "metrics.json"), read(OUT / "v5_replication.json")
    for stage, groups in (("delay", report["delay"]), ("known", report["known_parameters"])):
        trace_frames[stage] = {}
        for year, modes in groups.items():
            trace_frames[stage][year] = {}
            for mode, cases in modes.items():
                trace_frames[stage][year][mode] = {}
                for key, pair in cases.items():
                    method = key if stage == "delay" else "data_seed42"
                    panel = panels[year][method][mode]
                    trace_frames[stage][year][mode][key] = {}
                    for policy in ("legacy", "tail_guard"):
                        metrics = pair[policy]
                        trace = check_trace(OUT / metrics["trajectory"], metrics,
                                            terminal=True, continuous=True)["frame"]
                        assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                        risk_metrics(trace, metrics, gates, tail)
                        assert pd.DatetimeIndex(trace.timestamp).equals(pd.DatetimeIndex(panel.timestamp))
                        assert np.allclose(trace.net_load_mw, panel.net_load_mw, rtol=0, atol=1e-6)
                        for column in metadata[1:]:
                            assert pd.DatetimeIndex(pd.to_datetime(trace[column], utc=True)).equals(
                                pd.DatetimeIndex(panel[column]))
                        for column in ("prediction_h1_mw", "upper_h1_mw", "lower_h1_mw"):
                            assert np.allclose(trace[column], panel[column])
                        mae = float(np.abs(panel.net_load_mw - panel.prediction_h1_mw).mean())
                        coverage = float((panel.net_load_mw.ge(panel.lower_h1_mw)
                                          & panel.net_load_mw.le(panel.upper_h1_mw)).mean() * 100)
                        assert abs(mae - metrics["forecast_h1_mae_mw"]) < 1e-5
                        assert abs(coverage - metrics["interval_coverage_pct"]) < 1e-5
                        assert metrics["terminal_failure_days"] == 0
                        if policy == "tail_guard":
                            guard_check(
                                trace, trace_frames[stage][year][mode][key]["legacy"], panel,
                                calibration["methods"][method], gates, tail, metrics)
                        if stage == "delay":
                            assert metrics["physical_parameters"] == manifest["nominal_model"]
                            delay_count += 1
                            if mode == "fresh6":
                                old = old_metrics["methods"][year][policy][method]
                                old_trace = pd.read_csv(V5 / old["trajectory"],
                                                        parse_dates=["timestamp"]).set_index("timestamp").loc[trace.timestamp]
                                cols = ["charge_mw", "discharge_mw", "soc_before_mwh",
                                        "storage_soc_mwh", "controlled_net_load_mw"]
                                difference = float(np.abs(trace[cols].to_numpy() - old_trace[cols].to_numpy()).max())
                                assert difference <= 1e-5
                                assert replication[year][method][policy]["passed"]
                                assert abs(difference - replication[year][method][policy]["max_absolute_action_state_load_difference"]) < 1e-8
                        else:
                            assert metrics["physical_parameters"] == manifest["error_models"][key]
                            known_count += 1
                        trace_frames[stage][year][mode][key][policy] = trace
    for year, modes in report["mismatch"].items():
        for mode, cases in modes.items():
            for case, pair in cases.items():
                actual = manifest["error_models"][case]
                for policy, metrics in pair.items():
                    trace = check_trace(OUT / metrics["trajectory"], metrics,
                                        terminal=False, continuous=True)["frame"]
                    assert metrics["physical_parameters"] == actual
                    assert metrics["assumed_parameters"] == manifest["nominal_model"]
                    assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
                    risk_metrics(trace, metrics, gates, tail)
                    request = trace_frames["delay"][year][mode]["data_seed42"][policy]
                    source_m = report["delay"][year][mode]["data_seed42"][policy]
                    assert metrics["requested_trajectory"] == source_m["trajectory"]
                    assert metrics["requested_trajectory_sha256"] == source_m["trajectory_sha256"]
                    assert trace.timestamp.equals(request.timestamp)
                    assert np.allclose(trace.net_load_mw, request.net_load_mw)
                    assert np.allclose(trace.requested_charge_mw, request.charge_mw)
                    assert np.allclose(trace.requested_discharge_mw, request.discharge_mw)
                    eta, before = actual["efficiency"], trace.soc_before_mwh.to_numpy()
                    charge = np.minimum(np.minimum(request.charge_mw, actual["charge_power_mw"]),
                                        np.maximum((actual["maximum_soc_mwh"] - before) / eta, 0.))
                    discharge = np.minimum(np.minimum(request.discharge_mw, actual["discharge_power_mw"]),
                                           np.maximum((before - actual["minimum_soc_mwh"]) * eta, 0.))
                    assert np.allclose(trace.charge_mw, charge, rtol=0, atol=1e-5)
                    assert np.allclose(trace.discharge_mw, discharge, rtol=0, atol=1e-5)
                    assert abs(float((request.charge_mw - trace.charge_mw).sum())
                               - metrics["unexecuted_charge_mwh"]) < 1e-5
                    assert abs(float((request.discharge_mw - trace.discharge_mw).sum())
                               - metrics["unexecuted_discharge_mwh"]) < 1e-5
                    last = trace.timestamp.dt.hour.eq(23)
                    failures = int((trace.loc[last, "storage_soc_mwh"] - 15000.).abs().gt(1e-5).sum())
                    assert failures == metrics["terminal_failure_days"]
                    assert int(last.sum()) == metrics["terminal_days"]
                    mismatch_count += 1
    comparisons, interval_count = report["comparisons"], 0
    by_path = {}
    for section in ("delay", "mismatch", "known_parameters"):
        for modes in report[section].values():
            for cases in modes.values():
                for pair in cases.values():
                    for metrics in pair.values():
                        by_path[metrics["trajectory"]] = metrics
    for section in comparisons.values():
        for year in section.values():
            for cases in year.values():
                for comparison in cases.values():
                    a, b = by_path[comparison["left_trajectory"]], by_path[comparison["right_trajectory"]]
                    for key, expected in comparison["differences"].items():
                        assert abs(a[key] - b[key] - expected) < 1e-5
                    fa = pd.read_csv(OUT / a["trajectory"], parse_dates=["timestamp"])
                    fb = pd.read_csv(OUT / b["trajectory"], parse_dates=["timestamp"])
                    assert fa.timestamp.equals(fb.timestamp) and np.allclose(fa.net_load_mw, fb.net_load_mw)
                    daily = (fa.groupby(fa.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                             - fb.groupby(fb.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy()
                    interval_check(daily, comparison["daily_peak_interval"])
                    interval_count += 1
    decision = read(OUT / "demo_decision.json")
    assert report["demo_decision"] == decision
    admitted = decision_check(report["delay"], report["mismatch"], decision)
    assert read(OUT / "status.json")["default_realtime_demo_admitted"] == admitted
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    result = {
        "passed": True, "artifact_hashes_verified": len(hashes),
        "source_forecast_files": forecast_files, "delayed_panels_reconstructed": 30,
        "delay_trajectories": delay_count, "mismatch_trajectories": mismatch_count,
        "known_parameter_trajectories": known_count, "fresh6_replication_checks": 12,
        "paired_intervals_recomputed": interval_count,
        "default_realtime_demo_admitted": admitted,
        "demo_decision_recomputed": True,
        "boundary": "Fixed-request mismatch audit; no closed-loop state-feedback or system identification claim.",
    }
    (ROOT / "outputs/fusion_v6_local_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
