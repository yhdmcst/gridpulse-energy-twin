"""Independent v8 CSV audit: information, conservative requests, actuator and retention."""
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
from scripts.verify_feedback_results import actuator_check, check_packet_time, schedule, terminal_check
from scripts.verify_tail_results import interval_check, risk_metrics

OUT, V3, V6, V7 = (ROOT / f"outputs/{n}" for n in ("fusion_v8", "fusion_v3", "fusion_v6", "fusion_v7"))
POLICIES = ("envelope_reference", "envelope_tail")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(metrics, gates, tail):
    frame = check_trace(OUT / metrics["trajectory"], metrics, terminal=False, continuous=True)["frame"]
    assert metrics["physical_audit_passed"]
    risk_metrics(frame, metrics, gates, tail)
    terminal_check(frame, metrics)
    return frame


def check_controller(trace, reference, base, stale, metrics, model, envelope, event, policy, gates, tail):
    assert trace.timestamp.equals(base.timestamp)
    assert np.allclose(trace.net_load_mw, base.net_load_mw, rtol=0, atol=1e-6)
    actuator_check(trace, metrics["physical_parameters"])
    assert np.allclose(trace.measured_soc_mwh, trace.soc_before_mwh, rtol=0, atol=1e-8)
    for target, source in (
        ("reference_soc_before_mwh", "soc_before_mwh"), ("reference_soc_after_mwh", "storage_soc_mwh"),
        ("reference_charge_mw", "charge_mw"), ("reference_discharge_mw", "discharge_mw"),
        ("reference_solver_failed", "solver_failed")):
        assert np.allclose(trace[target], reference[source], atol=1e-6, rtol=0)
    expired, outage = schedule(trace, event)
    observed = check_packet_time(trace, base, stale, expired)
    assert np.allclose(trace.observation_age_hours,
                       (trace.timestamp - observed) / pd.Timedelta(hours=1))
    lower = np.maximum(model["minimum_soc_mwh"], model["initial_soc_mwh"]
                       - (23 - trace.timestamp.dt.hour) * envelope["charge_power_mw"] * envelope["efficiency"])
    needed = np.maximum((lower - trace.measured_soc_mwh) / envelope["efficiency"], 0.)
    fallback_c = np.minimum(needed, envelope["charge_power_mw"])
    natural = trace.primary_solver_failed.to_numpy(bool)
    natural_name = "reference_solver_failure" if policy == "envelope_reference" else "primary_solver_failure"
    ref_failed = reference.solver_failed.to_numpy(bool) & ~expired & ~outage if policy == "envelope_tail" else np.zeros(len(trace), bool)
    expected = np.where(expired, "forecast_expired", np.where(outage, "solver_unavailable",
                         np.where(ref_failed, "reference_solver_failure",
                                  np.where(natural, natural_name, "ok"))))
    assert np.array_equal(trace.gate_reason, expected)
    attempted = ~(expired | outage | ref_failed)
    assert np.array_equal(trace.primary_solver_attempted, attempted)
    assert not (natural & ~attempted).any()
    assert trace.loc[natural, "primary_solver_error"].astype("string").fillna("").str.len().gt(0).all()
    fallback = trace.gate_reason.ne("ok")
    assert np.array_equal(trace.fallback_active, fallback)
    assert trace.loc[fallback, "primary_discharge_mw"].eq(0.).all()
    assert np.allclose(trace.loc[fallback, "primary_charge_mw"], fallback_c[fallback])
    c, d = trace.primary_charge_mw.copy(), trace.primary_discharge_mw.copy()
    below = trace.measured_soc_mwh < lower
    c = np.where(below, np.maximum(c, fallback_c), c)
    d = np.where(below, 0., np.minimum(d, np.maximum((trace.measured_soc_mwh - lower) * envelope["efficiency"], 0.)))
    assert np.allclose(trace.requested_charge_mw, c, rtol=0, atol=1e-5)
    assert np.allclose(trace.requested_discharge_mw, d, rtol=0, atol=1e-5)
    intervene = ((np.abs(c - trace.primary_charge_mw) > 1e-6)
                 | (np.abs(d - trace.primary_discharge_mw) > 1e-6))
    assert np.array_equal(trace.supervisor_intervened, intervene)
    worst = np.minimum(model["maximum_soc_mwh"], trace.measured_soc_mwh
                       + envelope["efficiency"] * np.minimum(c, envelope["charge_power_mw"])
                       - d / envelope["efficiency"])
    assert np.allclose(trace.safe_floor_next_mwh, lower)
    assert np.allclose(trace.worst_case_next_soc_mwh, worst)
    assert np.array_equal(trace.recovery_unreachable, needed.gt(envelope["charge_power_mw"] + 1e-5))
    assert np.allclose(trace.actual_floor_violation_mwh, np.maximum(lower - trace.storage_soc_mwh, 0.))
    assert np.allclose(trace.predicted_next_soc_mwh,
                       trace.measured_soc_mwh + model["efficiency"] * c - d / model["efficiency"])
    if policy == "envelope_tail":
        cap = np.maximum(reference.charge_mw - reference.discharge_mw, tail - base.upper_h1_mw)
        ok = ~fallback
        assert np.allclose(trace.loc[ok, "headroom_cap_mw"], cap[ok])
        assert trace.loc[~ok, "headroom_cap_mw"].isna().all()
        assert (trace.loc[ok, "primary_charge_mw"] - trace.loc[ok, "primary_discharge_mw"] <= cap[ok] + 1e-5).all()
        assert np.allclose(trace.loc[ok, "post_supervisor_headroom_excess_mw"],
                           np.maximum(c - d - cap, 0.)[ok])
    else:
        assert trace.headroom_cap_mw.isna().all()
    counters = {
        "fallback_hours": int(fallback.sum()),
        "primary_solver_attempts": int(attempted.sum()),
        "primary_solver_failure_hours": int(natural.sum()),
        "reference_solver_failure_hours": int(reference.solver_failed.sum()),
        "expiry_rejections": int(expired.sum()), "outage_fallback_hours": int(outage.sum()),
        "supervisor_interventions": int(intervene.sum()),
        "recovery_unreachable_hours": int(trace.recovery_unreachable.sum()),
        "gated_optimizer_calls": int(((expired | outage) & attempted).sum()),
        "actual_clipping_hours": int(
            ((trace.requested_charge_mw - trace.charge_mw).abs().gt(1e-5)
             | (trace.requested_discharge_mw - trace.discharge_mw).abs().gt(1e-5)).sum()),
    }
    for key, value in counters.items():
        assert metrics[key] == value, (key, value, metrics[key])
    floating = {
        "max_actual_floor_violation_mwh": trace.actual_floor_violation_mwh.max(),
        "max_worst_case_floor_violation_mwh": np.maximum(lower - worst, 0.).max(),
        "unexecuted_charge_mwh": (trace.requested_charge_mw - trace.charge_mw).sum(),
        "unexecuted_discharge_mwh": (trace.requested_discharge_mw - trace.discharge_mw).sum(),
        "max_actual_reference_debt_mwh": np.maximum(reference.storage_soc_mwh - trace.storage_soc_mwh, 0.).max(),
    }
    for key, value in floating.items():
        assert abs(metrics[key] - value) < 1e-5
    assert np.allclose(trace.actual_reference_debt_mwh,
                       np.maximum(reference.storage_soc_mwh - trace.storage_soc_mwh, 0.))


def selection_check(validation, selection):
    eligible, order = [], {}
    for policy in POLICIES:
        guards, deltas, p95, losses = {}, [], [], []
        for case in ("nominal", "combined90"):
            a, b = validation[policy][case], validation["baseline_v7"][case]
            guards[case] = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"] and a["max_abs_terminal_error_mwh"] <= 1e-5),
                "recovery": bool(max(a["max_actual_floor_violation_mwh"], a["max_worst_case_floor_violation_mwh"]) <= 1e-5),
                "gates": a["gated_optimizer_calls"] == 0,
                "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
                "throughput": bool(a["throughput_mwh"] <= b["throughput_mwh"] * 1.1 + 1e-6),
                "loss": bool(a["conversion_loss_mwh"] <= b["conversion_loss_mwh"] * 1.1 + 1e-6),
            }
            deltas.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
            p95.append(a["controlled_p95_mw"])
            losses.append(a["conversion_loss_mwh"])
        m = selection["candidates"][policy]
        is_eligible = bool(np.mean(deltas) <= -1. and all(all(g.values()) for g in guards.values()))
        assert m["scenario_guards"] == guards and m["eligible"] == is_eligible
        for key, value in (("mean_p95_delta_mw", np.mean(deltas)),
                           ("mean_p95_mw", np.mean(p95)), ("mean_loss_mwh", np.mean(losses))):
            assert abs(m[key] - value) < 1e-5
        if is_eligible:
            eligible.append(policy)
        order[policy] = (float(np.mean(p95)), float(np.mean(losses)), policy)
    selected = min(eligible, key=lambda p: order[p]) if eligible else "baseline_v7"
    assert selection["selected_policy"] == selected and selection["fallback"] == (not eligible)
    return selected


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(V3 / "predictions/data_seed42/validation.csv") == manifest["validation_prediction_sha256"]
    assert sha(V7 / "calibration.json") == manifest["source_calibration_sha256"]
    calibration = read(OUT / "calibration.json")
    assert calibration == read(V7 / "calibration.json")
    assert read(OUT / "validation_failures.json") == read(OUT / "evaluation_failures.json") == []
    report, validation, selection = read(OUT / "metrics.json"), read(OUT / "validation_results.json"), read(OUT / "selection.json")
    assert report["validation"] == validation
    assert selection["validation_results_sha256"] == sha(OUT / "validation_results.json")
    assert selection["validation_prediction_sha256"] == manifest["validation_prediction_sha256"]
    assert selection["protocol_sha256"] == sha(ROOT / "experiments/fusion_v8/protocol.md")
    started = read(OUT / "evaluation_start.json")
    assert started["selection_sha256"] == sha(OUT / "selection.json")
    assert datetime.fromisoformat(selection["frozen_utc"]) <= datetime.fromisoformat(started["started_utc"])
    for relative, expected in read(OUT / "evaluation_inputs.json").items():
        prefix, path = relative.split("/", 1)
        assert sha((V6 if prefix == "v6" else V7) / path) == expected
    dates = ["timestamp", "source_first_target_timestamp", "issued_timestamp", "available_timestamp"]
    base = pd.read_csv(OUT / "panels/validation/fresh6.csv", parse_dates=dates)
    stale = pd.read_csv(OUT / "panels/validation/delayed3.csv", parse_dates=dates)
    source = pd.read_csv(V3 / "predictions/data_seed42/validation.csv",
                         parse_dates=["timestamp", "issued_timestamp"]).set_index("timestamp")
    times = pd.DatetimeIndex(base.timestamp)
    assert times.min() >= pd.Timestamp("2019-01-12T20:00:00Z")
    assert times.max() + pd.Timedelta(hours=5) < pd.Timestamp("2019-08-09T21:00:00Z")
    for panel, delay, horizon in ((base, 0, 6), (stale, 3, 3)):
        assert panel.timestamp.equals(base.timestamp)
        old = source.loc[times - pd.Timedelta(hours=delay)]
        assert pd.DatetimeIndex(panel.issued_timestamp).equals(times - pd.Timedelta(hours=delay + 1))
        assert np.allclose(panel.net_load_mw, source.loc[times].actual_h1_mw)
        for h in range(1, horizon + 1):
            expected = old[f"prediction_h{delay + h}_mw"].to_numpy()
            hours = (panel.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
            radius = hours.map({int(k): v for k, v in calibration["methods"]["data_seed42"]["radii"][str(delay + h)].items()})
            assert np.allclose(panel[f"prediction_h{h}_mw"], expected)
            assert np.allclose(panel[f"upper_h{h}_mw"], expected + radius)
            assert np.allclose(panel[f"lower_h{h}_mw"], expected - radius)
    gates, tail = calibration["gates"], calibration["tail_threshold_mw"]
    envelope, lower = manifest["recovery_envelope"], manifest["planning_model"]
    assert lower["charge_power_mw"] == lower["discharge_power_mw"] == envelope["charge_power_mw"] == 875.
    assert lower["efficiency"] == envelope["efficiency"] == .85
    models, all_metrics = read(V7 / "manifest.json")["actual_models"], {}
    ref_meta = read(OUT / "validation_references.json")
    references, ref_count, val_count, eval_count, reused = {}, 0, 0, 0, 0
    for label, met in ref_meta.items():
        ref = load_trace(met, gates, tail)
        assumed = manifest["nominal_model"] if label == "nominal" else lower
        assert met["physical_parameters"] == assumed
        actuator_check(ref, assumed)
        assert ref.timestamp.equals(base.timestamp)
        references[label] = ref
        ref_count += 1
        all_metrics[met["trajectory"]] = met
    for policy, scenarios in validation.items():
        for case, metrics in scenarios.items():
            trace = load_trace(metrics, gates, tail)
            assert metrics["physical_parameters"] == models[case]
            model = manifest["nominal_model"] if policy == "baseline_v7" else lower
            assert metrics["assumed_parameters"] == model
            check_controller(trace, references["nominal" if policy == "baseline_v7" else "conservative"],
                             base, stale, metrics, model, envelope, "healthy",
                             "envelope_tail" if policy == "baseline_v7" else policy, gates, tail)
            all_metrics[metrics["trajectory"]] = metrics
            val_count += 1
    selected = selection_check(validation, selection)
    assert report["selected_policy"] == selected
    refs, panels = {}, {}
    for year in ("2021", "2022"):
        panels[year] = {mode: pd.read_csv(V6 / "panels" / year / "data_seed42" / f"{mode}.csv", parse_dates=dates)
                        for mode in ("fresh6", "delayed1", "delayed3")}
        refs[year] = {}
        for mode, events in report["references"][year].items():
            refs[year][mode] = {}
            for event, metrics in events.items():
                ref = load_trace(metrics, gates, tail)
                assert metrics["physical_parameters"] == lower
                actuator_check(ref, lower)
                assert ref.timestamp.equals(panels[year][mode].timestamp)
                expired, outage = schedule(ref, event)
                check_packet_time(ref, panels[year][mode], panels[year]["delayed3"], expired)
                assert not ref.loc[expired | outage, "solver_attempted"].any()
                refs[year][mode][event] = ref
                all_metrics[metrics["trajectory"]] = metrics
                ref_count += 1
    in_range = []
    for year, modes in report["methods"].items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, policies in events.items():
                    assert set(policies) == {"baseline_v7", *POLICIES}
                    for policy, metrics in policies.items():
                        trace = load_trace(metrics, gates, tail)
                        assert trace.timestamp.equals(panels[year][mode].timestamp)
                        assert metrics["physical_parameters"] == models[case]
                        if policy == "baseline_v7":
                            assert sha(OUT / metrics["trajectory"]) == sha(V7 / metrics["source_trajectory"]) == metrics["source_sha256"]
                            reused += 1
                        else:
                            assert metrics["assumed_parameters"] == lower
                            assert metrics["recovery_envelope"] == envelope
                            check_controller(trace, refs[year][mode][event],
                                             panels[year][mode], panels[year]["delayed3"],
                                             metrics, lower, envelope, event, policy, gates, tail)
                            inside = models[case]["charge_power_mw"] >= envelope["charge_power_mw"] and models[case]["efficiency"] >= envelope["efficiency"]
                            assert metrics["plant_within_declared_envelope"] == inside
                            if inside:
                                in_range.append(bool(
                                    metrics["terminal_requirement_passed"]
                                    and metrics["physical_audit_passed"] and metrics["gated_optimizer_calls"] == 0
                                    and metrics["max_actual_floor_violation_mwh"] <= 1e-5
                                    and metrics["max_worst_case_floor_violation_mwh"] <= 1e-5))
                            eval_count += 1
                        all_metrics[metrics["trajectory"]] = metrics
    intervals = 0
    for contrast in report["comparisons"].values():
        a, b = all_metrics[contrast["left"]], all_metrics[contrast["right"]]
        for key, expected in contrast["differences"].items():
            assert abs(a[key] - b[key] - expected) < 1e-5
        fa = pd.read_csv(OUT / a["trajectory"], parse_dates=["timestamp"])
        fb = pd.read_csv(OUT / b["trajectory"], parse_dates=["timestamp"])
        assert fa.timestamp.equals(fb.timestamp) and np.allclose(fa.net_load_mw, fb.net_load_mw)
        daily = (fa.groupby(fa.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                 - fb.groupby(fb.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy()
        interval_check(daily, contrast["daily_peak_interval"])
        intervals += 1
    decision = read(OUT / "decision.json")
    assert decision == report["decision"]
    assert decision["selected_policy"] == selected and decision["selection_sha256"] == sha(OUT / "selection.json")
    assert decision["new_in_envelope_runs"] == len(in_range)
    assert decision["new_in_envelope_recovery_passed"] == bool(in_range and all(in_range))
    assert decision["realtime_default_admitted"] is False
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    result = {
        "passed": True, "artifact_hashes_verified": len(hashes), "validation_traces": val_count,
        "new_evaluation_traces": eval_count, "modeled_references": ref_count,
        "byte_identical_v7_baselines": reused, "paired_intervals": intervals,
        "selected_policy": selected, "in_envelope_recovery_passed": bool(in_range and all(in_range)),
        "realtime_default_admitted": False,
        "scope": "Offline conservative-pipeline ablation, fixed seed and inspected years.",
    }
    (ROOT / "outputs/fusion_v8_local_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
