"""Independent NumPy/CSV audit of plant feedback, TTL/outage gates and supervisor."""
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

OUT = ROOT / "outputs/fusion_v7"
V6 = ROOT / "outputs/fusion_v6"
DATES = ("timestamp", "source_first_target_timestamp", "issued_timestamp", "available_timestamp")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actuator_check(trace, model):
    before = trace.soc_before_mwh.to_numpy()
    req_c, req_d = trace.requested_charge_mw.to_numpy(), trace.requested_discharge_mw.to_numpy()
    eta = model["efficiency"]
    assert np.minimum(req_c, req_d).max() <= 1e-6
    assert min(req_c.min(), req_d.min()) >= -1e-6
    c = np.minimum(np.minimum(req_c, model["charge_power_mw"]),
                   np.maximum((model["maximum_soc_mwh"] - before) / eta, 0.))
    d = np.minimum(np.minimum(req_d, model["discharge_power_mw"]),
                   np.maximum((before - model["minimum_soc_mwh"]) * eta, 0.))
    assert np.allclose(trace.charge_mw, c, rtol=0, atol=1e-5)
    assert np.allclose(trace.discharge_mw, d, rtol=0, atol=1e-5)
    assert np.allclose(trace.storage_soc_mwh, before + eta * c - d / eta, rtol=0, atol=1e-5)


def schedule(trace, event):
    index = np.arange(len(trace))
    chosen = (index // 24) % 7 == 0
    hour = trace.timestamp.dt.hour
    stale = chosen & hour.isin([14, 15, 16, 17]).to_numpy() if event in ("stale_burst", "combined_fault") else np.zeros(len(trace), bool)
    outage = chosen & hour.eq(10).to_numpy() if event in ("solver_outage", "combined_fault") else np.zeros(len(trace), bool)
    assert np.array_equal(trace.injected_stale, stale)
    assert np.array_equal(trace.injected_outage, outage)
    return stale, outage


def check_packet_time(trace, base, stale_panel, stale):
    expected_observed = base.issued_timestamp.where(~stale, stale_panel.issued_timestamp)
    expected_source = base.source_first_target_timestamp.where(~stale, stale_panel.source_first_target_timestamp)
    expected_available = base.available_timestamp.where(~stale, stale_panel.available_timestamp)
    for field, expected in (("observed_timestamp", expected_observed), ("source_timestamp", expected_source),
                            ("available_timestamp", expected_available)):
        assert pd.DatetimeIndex(pd.to_datetime(trace[field], utc=True)).equals(pd.DatetimeIndex(expected))
    return expected_observed


def terminal_check(trace, metrics):
    model = metrics["physical_parameters"]
    assert abs(trace.soc_before_mwh.iloc[0] - model["initial_soc_mwh"]) < 1e-5
    assert np.allclose(trace.soc_before_mwh.iloc[1:], trace.storage_soc_mwh.iloc[:-1], atol=1e-5, rtol=0)
    last = trace.timestamp.dt.hour.eq(23)
    errors = (trace.loc[last, "storage_soc_mwh"] - model["initial_soc_mwh"]).abs()
    assert metrics["terminal_failure_days"] == int(errors.gt(1e-5).sum())
    assert metrics["terminal_days"] == int(last.sum())


def continuous_trace(metrics, gates, tail):
    trace = check_trace(OUT / metrics["trajectory"], metrics, terminal=False, continuous=True)["frame"]
    assert metrics["physical_audit_passed"]
    risk_metrics(trace, metrics, gates, tail)
    terminal_check(trace, metrics)
    return trace


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected, relative
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in manifest["input_panel_sha256"].items():
        assert sha(V6 / relative) == expected
    for relative, expected in read(OUT / "source_open_hashes.json").items():
        assert sha(V6 / relative) == expected
    for file, key in (("metrics.json", "source_metrics_sha256"), ("calibration.json", "source_calibration_sha256"),
                      ("samples.json", "source_samples_sha256")):
        assert sha(V6 / file) == manifest[key]
    assert read(OUT / "failures.json") == [] and read(OUT / "status.json")["status"] == "complete"
    calibration, samples, report = read(OUT / "calibration.json"), read(OUT / "samples.json"), read(OUT / "metrics.json")
    assert calibration == read(V6 / "calibration.json") and samples == read(V6 / "samples.json")
    tail, gates = calibration["tail_threshold_mw"], calibration["gates"]
    assumed, envelope = manifest["assumed_model"], manifest["recovery_envelope"]
    panels, refs, all_metrics = {}, {}, {}
    ref_count, closed_count, open_count, fault_hours = 0, 0, 0, 0
    for year in ("2021", "2022"):
        panels[year], refs[year] = {}, {}
        for mode in ("fresh6", "delayed1", "delayed3"):
            panels[year][mode] = pd.read_csv(
                V6 / "panels" / year / "data_seed42" / f"{mode}.csv", parse_dates=list(DATES))
        for mode, events in report["references"][year].items():
            refs[year][mode] = {}
            for event, metrics in events.items():
                ref = continuous_trace(metrics, gates, tail)
                assert metrics["physical_parameters"] == assumed
                actuator_check(ref, assumed)
                assert len(ref) == samples[year]["rows"]
                base, old = panels[year][mode], panels[year]["delayed3"]
                assert ref.timestamp.equals(base.timestamp)
                stale, outage = schedule(ref, event)
                check_packet_time(ref, base, old, stale)
                expected = np.where(stale, "forecast_expired", np.where(outage, "solver_unavailable", "ok"))
                assert np.array_equal(ref.loc[stale | outage, "gate_reason"], expected[stale | outage])
                assert not ref.loc[stale | outage, "solver_attempted"].any()
                assert not ref.loc[stale | outage, "solver_failed"].any()
                assert np.array_equal(ref.solver_attempted.to_numpy(), ~(stale | outage))
                fallback = ref.gate_reason.ne("ok")
                floor = np.maximum(assumed["minimum_soc_mwh"], assumed["initial_soc_mwh"]
                                   - (23 - ref.timestamp.dt.hour) * envelope["charge_power_mw"] * envelope["efficiency"])
                charge = np.minimum(np.maximum((floor - ref.soc_before_mwh) / envelope["efficiency"], 0.),
                                    envelope["charge_power_mw"])
                assert np.allclose(ref.loc[fallback, "requested_charge_mw"], charge[fallback], atol=1e-5, rtol=0)
                assert ref.loc[fallback, "requested_discharge_mw"].eq(0.).all()
                assert ref.loc[ref.solver_failed, "solver_error"].fillna("").str.len().gt(0).all()
                refs[year][mode][event] = ref
                all_metrics[metrics["trajectory"]] = metrics
                ref_count += 1
    replication = read(OUT / "nominal_replication.json")
    previous = read(V6 / "metrics.json")
    replication_count, supervised_checks = 0, []
    conditions = {}
    for year, modes in report["closed"].items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, variants in events.items():
                    for variant, metrics in variants.items():
                        trace = continuous_trace(metrics, gates, tail)
                        model = manifest["actual_models"][case]
                        assert metrics["physical_parameters"] == model
                        assert metrics["assumed_parameters"] == assumed
                        assert metrics["recovery_envelope"] == envelope
                        assert metrics["supervised"] == (variant == "supervised")
                        actuator_check(trace, model)
                        base, old = panels[year][mode], panels[year]["delayed3"]
                        reference = refs[year][mode][event]
                        assert trace.timestamp.equals(base.timestamp)
                        assert np.allclose(trace.net_load_mw, base.net_load_mw, atol=1e-6, rtol=0)
                        assert np.allclose(trace.measured_soc_mwh, trace.soc_before_mwh, atol=1e-8, rtol=0)
                        for target, source in (
                            ("reference_soc_before_mwh", "soc_before_mwh"),
                            ("reference_soc_after_mwh", "storage_soc_mwh"),
                            ("reference_charge_mw", "charge_mw"), ("reference_discharge_mw", "discharge_mw"),
                            ("reference_solver_failed", "solver_failed")):
                            assert np.allclose(trace[target], reference[source], atol=1e-6, rtol=0)
                        assert np.allclose(
                            trace.actual_reference_debt_mwh, np.maximum(reference.storage_soc_mwh - trace.storage_soc_mwh, 0.))
                        stale, outage = schedule(trace, event)
                        observed = check_packet_time(trace, base, old, stale)
                        age = (trace.timestamp - observed) / pd.Timedelta(hours=1)
                        assert np.allclose(trace.observation_age_hours, age)
                        assert age.loc[~stale].le(envelope["max_observation_age_hours"]).all()
                        assert age.loc[stale].gt(envelope["max_observation_age_hours"]).all()
                        reasons = trace.gate_reason
                        expected = np.where(stale, "forecast_expired",
                                            np.where(outage, "solver_unavailable",
                                                     np.where(reference.solver_failed, "reference_solver_failure", "ok")))
                        normal = expected == "ok"
                        actual_nonfailure = reasons.ne("primary_solver_failure").to_numpy()
                        assert np.array_equal(reasons.loc[actual_nonfailure], expected[actual_nonfailure])
                        assert np.array_equal(trace.primary_solver_attempted, normal)
                        assert np.array_equal(trace.primary_solver_failed, reasons.eq("primary_solver_failure"))
                        assert np.array_equal(trace.fallback_active, reasons.ne("ok"))
                        assert trace.loc[trace.primary_solver_failed, "primary_solver_error"].fillna("").str.len().gt(0).all()
                        fallback = trace.fallback_active
                        floor = np.maximum(assumed["minimum_soc_mwh"], assumed["initial_soc_mwh"]
                                           - (23 - trace.timestamp.dt.hour) * envelope["charge_power_mw"] * envelope["efficiency"])
                        needed = np.maximum((floor - trace.measured_soc_mwh) / envelope["efficiency"], 0.)
                        fallback_c = np.minimum(needed, envelope["charge_power_mw"])
                        assert np.allclose(trace.safe_floor_next_mwh, floor)
                        assert np.array_equal(trace.recovery_unreachable, needed.gt(envelope["charge_power_mw"] + 1e-5))
                        assert np.allclose(trace.loc[fallback, "primary_charge_mw"], fallback_c[fallback], atol=1e-5, rtol=0)
                        assert trace.loc[fallback, "primary_discharge_mw"].eq(0.).all()
                        c, d = trace.primary_charge_mw.copy(), trace.primary_discharge_mw.copy()
                        if variant == "supervised":
                            below = trace.measured_soc_mwh < floor
                            c = np.where(below, np.maximum(c, fallback_c), c)
                            d = np.where(below, 0., np.minimum(d, np.maximum(
                                (trace.measured_soc_mwh - floor) * envelope["efficiency"], 0.)))
                        assert np.allclose(trace.requested_charge_mw, c, atol=1e-5, rtol=0)
                        assert np.allclose(trace.requested_discharge_mw, d, atol=1e-5, rtol=0)
                        intervention = ((np.abs(c - trace.primary_charge_mw) > 1e-6)
                                        | (np.abs(d - trace.primary_discharge_mw) > 1e-6))
                        assert np.array_equal(trace.supervisor_intervened, intervention)
                        worst = np.minimum(
                            assumed["maximum_soc_mwh"], trace.measured_soc_mwh
                            + envelope["efficiency"] * np.minimum(c, envelope["charge_power_mw"])
                            - d / envelope["efficiency"])
                        assert np.allclose(trace.worst_case_next_soc_mwh, worst)
                        assert np.allclose(trace.actual_floor_violation_mwh, np.maximum(floor - trace.storage_soc_mwh, 0.))
                        assert np.allclose(
                            trace.predicted_next_soc_mwh,
                            trace.measured_soc_mwh + assumed["efficiency"] * c - d / assumed["efficiency"])
                        ok = reasons.eq("ok")
                        expected_cap = np.maximum(reference.charge_mw - reference.discharge_mw,
                                                  tail - base.upper_h1_mw)
                        assert np.allclose(trace.loc[ok, "headroom_cap_mw"], expected_cap[ok], atol=1e-5, rtol=0)
                        assert trace.loc[~ok, "headroom_cap_mw"].isna().all()
                        assert (trace.loc[ok, "primary_charge_mw"] - trace.loc[ok, "primary_discharge_mw"]
                                <= trace.loc[ok, "headroom_cap_mw"] + 1e-5).all()
                        headroom_excess = np.maximum(trace.requested_charge_mw - trace.requested_discharge_mw
                                                     - trace.headroom_cap_mw, 0.)
                        assert np.allclose(trace.loc[ok, "post_supervisor_headroom_excess_mw"],
                                           headroom_excess[ok], atol=1e-5, rtol=0)
                        counters = {
                            "fallback_hours": int(fallback.sum()),
                            "primary_solver_attempts": int(trace.primary_solver_attempted.sum()),
                            "primary_solver_failure_hours": int(trace.primary_solver_failed.sum()),
                            "reference_solver_failure_hours": int(reference.solver_failed.sum()),
                            "expiry_rejections": int(stale.sum()), "outage_fallback_hours": int(outage.sum()),
                            "supervisor_interventions": int(intervention.sum()),
                            "recovery_unreachable_hours": int(trace.recovery_unreachable.sum()),
                            "actual_clipping_hours": int(
                                ((trace.requested_charge_mw - trace.charge_mw).abs().gt(1e-5)
                                 | (trace.requested_discharge_mw - trace.discharge_mw).abs().gt(1e-5)).sum()),
                            "gated_optimizer_calls": int(((stale | outage) & trace.primary_solver_attempted).sum()),
                        }
                        for key, value in counters.items():
                            assert metrics[key] == value, (key, value, metrics[key])
                        floating = {
                            "max_actual_floor_violation_mwh": trace.actual_floor_violation_mwh.max(),
                            "max_worst_case_floor_violation_mwh": np.maximum(floor - worst, 0.).max(),
                            "unexecuted_charge_mwh": (trace.requested_charge_mw - trace.charge_mw).sum(),
                            "unexecuted_discharge_mwh": (trace.requested_discharge_mw - trace.discharge_mw).sum(),
                            "max_actual_reference_debt_mwh": trace.actual_reference_debt_mwh.max(),
                        }
                        for key, value in floating.items():
                            assert abs(metrics[key] - value) < 1e-5
                        inside = model["charge_power_mw"] >= envelope["charge_power_mw"] and model["efficiency"] >= envelope["efficiency"]
                        assert metrics["plant_within_declared_envelope"] == inside
                        cond_key = f"{year}/{mode}/{case}/{event}/{variant}"
                        conditions[cond_key] = {
                            "hard_physics": True, "no_optimizer_when_gated": counters["gated_optimizer_calls"] == 0,
                            "terminal": metrics["terminal_requirement_passed"],
                            "within_recovery_envelope": bool(inside),
                        }
                        if variant == "supervised" and inside:
                            supervised_checks.append(bool(
                                counters["gated_optimizer_calls"] == 0 and metrics["terminal_requirement_passed"]
                                and floating["max_actual_floor_violation_mwh"] <= 1e-5
                                and floating["max_worst_case_floor_violation_mwh"] <= 1e-5))
                        if case == "nominal" and variant == "feedback_only" and event == "healthy":
                            original = previous["delay"][year][mode]["data_seed42"]["tail_guard"]
                            old_trace = pd.read_csv(V6 / original["trajectory"])
                            fields = ["charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh", "controlled_net_load_mw"]
                            diff = float(np.abs(trace[fields].to_numpy() - old_trace[fields].to_numpy()).max())
                            assert diff <= 1e-5
                            assert abs(replication[year][mode]["max_difference_mw_mwh"] - diff) < 1e-8
                            assert replication[year][mode]["passed"]
                            replication_count += 1
                        fault_hours += int((stale | outage).sum())
                        all_metrics[metrics["trajectory"]] = metrics
                        closed_count += 1
    byte_reused = 0
    for year, modes in report["open"].items():
        for mode, cases in modes.items():
            for case, metrics in cases.items():
                trace = continuous_trace(metrics, gates, tail)
                assert metrics["physical_parameters"] == manifest["actual_models"][case]
                if metrics["byte_reused"]:
                    assert sha(OUT / metrics["trajectory"]) == sha(V6 / metrics["source_trajectory"])
                    byte_reused += 1
                else:
                    original = previous["delay"][year][mode]["data_seed42"]["tail_guard"]
                    request = pd.read_csv(V6 / original["trajectory"])
                    assert np.allclose(trace.requested_charge_mw, request.charge_mw)
                    assert np.allclose(trace.requested_discharge_mw, request.discharge_mw)
                    actuator_check(trace, metrics["physical_parameters"])
                all_metrics[metrics["trajectory"]] = metrics
                open_count += 1
    interval_count = 0
    for comparison in report["comparisons"].values():
        left, right = all_metrics[comparison["left_trajectory"]], all_metrics[comparison["right_trajectory"]]
        for key, expected in comparison["differences"].items():
            assert abs(left[key] - right[key] - expected) < 1e-5
        a = pd.read_csv(OUT / left["trajectory"], parse_dates=["timestamp"])
        b = pd.read_csv(OUT / right["trajectory"], parse_dates=["timestamp"])
        assert a.timestamp.equals(b.timestamp) and np.allclose(a.net_load_mw, b.net_load_mw)
        daily = (a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                 - b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy()
        interval_check(daily, comparison["daily_peak_interval"])
        interval_count += 1
    capability = read(OUT / "capability_checks.json")
    assert report["capability_checks"] == capability
    assert capability["conditions"] == conditions
    assert capability["nominal_replication_passed"] and capability["closed_physics_passed"]
    assert capability["gated_no_optimizer_passed"] == all(x["no_optimizer_when_gated"] for x in conditions.values())
    assert capability["supervised_in_envelope_recovery_passed"] == bool(supervised_checks and all(supervised_checks))
    assert capability["realtime_default_admitted"] is False
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    result = {
        "passed": True, "artifact_hashes_verified": len(hashes),
        "closed_loop_trajectories": closed_count, "open_loop_trajectories": open_count,
        "modeled_reference_trajectories": ref_count, "byte_reused_open_traces": byte_reused,
        "nominal_replication_checks": replication_count, "paired_intervals": interval_count,
        "gated_model_variant_hours_checked": fault_hours,
        "supervised_in_envelope_recovery_passed": capability["supervised_in_envelope_recovery_passed"],
        "realtime_default_admitted": False,
        "boundary": "Perfect simulator SOC and declared envelope, no real-device or unknown-bound guarantee.",
    }
    (ROOT / "outputs/fusion_v7_local_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
