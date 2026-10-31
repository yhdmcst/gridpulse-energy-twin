"""Offline hourly SOC feedback, conservative recovery and injected fault probes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from src.feedback_control import RecoveryEnvelope, make_reference, plant_models, simulate_feedback
from src.optimality import StorageModel
from src.robustness_control import replay_mismatch, annotate
from scripts.run_dispatch_experiment import daily_ci, sha, save
from scripts.run_tail_experiment import read, save_trace, utc_now

YEARS = ("2021", "2022")
MODES = ("fresh6", "delayed1")
CASES = ("nominal", "power_90pct", "power_70pct", "efficiency_090", "efficiency_085", "combined90")
VARIANTS = ("feedback_only", "supervised")
EVENTS = ("stale_burst", "solver_outage", "combined_fault")
MEASURES = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
            "throughput_mwh", "conversion_loss_mwh", "max_abs_terminal_error_mwh")


def job(args):
    year, mode, case, event, variant, base, old, reference, calibration, output = args
    out = Path(output)
    label = f"{year}_{mode}_{case}_{event}_{variant}"
    started = time.monotonic()
    try:
        trace, metrics = simulate_feedback(
            base, old, reference, event, plant_models()[case], variant == "supervised",
            calibration["methods"]["data_seed42"], calibration["gates"], calibration["tail_threshold_mw"])
        metrics.update({
            "condition": {"year": year, "mode": mode, "case": case, "event": event, "variant": variant},
            "seconds": time.monotonic() - started,
        })
        metrics = save_trace(out, trace, metrics, f"closed/{year}/{mode}/{case}/{event}/{variant}.csv")
        save(out / "jobs" / f"{label}.json", {"status": "complete", "metrics": metrics})
        print(f"{label}: terminal={metrics['max_abs_terminal_error_mwh']:.5f} "
              f"fallback={metrics['fallback_hours']} shield={metrics['supervisor_interventions']}", flush=True)
        return year, mode, case, event, variant, metrics
    except Exception:
        save(out / "jobs" / f"{label}.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


def reference_streams(out, panels, calibration):
    data, reports = {}, {}
    for year in YEARS:
        data[year], reports[year] = {}, {}
        for mode in MODES:
            data[year][mode], reports[year][mode] = {}, {}
            for event in ("healthy", *EVENTS) if mode == "fresh6" else ("healthy",):
                trace, metrics = make_reference(
                    panels[year][mode], panels[year]["delayed3"], event,
                    calibration["methods"]["data_seed42"], calibration["gates"],
                    calibration["tail_threshold_mw"])
                reports[year][mode][event] = save_trace(
                    out, trace, metrics, f"references/{year}_{mode}_{event}.csv")
                data[year][mode][event] = trace
    return data, reports


def open_comparators(out, source, previous, calibration):
    output, hashes = {}, {}
    for year in YEARS:
        output[year] = {}
        for mode in MODES:
            output[year][mode] = {}
            nominal = previous["delay"][year][mode]["data_seed42"]["tail_guard"]
            nominal_path = source / nominal["trajectory"]
            requests = pd.read_csv(nominal_path, parse_dates=["timestamp"])
            hashes[nominal["trajectory"]] = sha(nominal_path)
            for case in CASES:
                if case == "combined90":
                    trace, metrics = replay_mismatch(
                        requests, plant_models()[case], calibration["gates"], calibration["tail_threshold_mw"])
                    trace, metrics = annotate(trace, metrics)
                    metrics["requested_trajectory"] = nominal["trajectory"]
                    metrics["requested_trajectory_sha256"] = nominal["trajectory_sha256"]
                    metrics = save_trace(out, trace, metrics, f"open/{year}/{mode}/{case}.csv")
                    metrics["byte_reused"] = False
                else:
                    original = nominal if case == "nominal" else previous["mismatch"][year][mode][case]["tail_guard"]
                    path = source / original["trajectory"]
                    hashes[original["trajectory"]] = sha(path)
                    trace = pd.read_csv(path, parse_dates=["timestamp"])
                    metrics = save_trace(out, trace, dict(original), f"open/{year}/{mode}/{case}.csv", copy_source=path)
                    metrics["byte_reused"] = True
                    metrics["source_trajectory"] = original["trajectory"]
                output[year][mode][case] = metrics
    return output, hashes


def replication_check(out, results, source, previous):
    output = {}
    columns = ("charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh", "controlled_net_load_mw")
    for year in YEARS:
        output[year] = {}
        for mode in MODES:
            metrics = results[year][mode]["nominal"]["healthy"]["feedback_only"]
            original = previous["delay"][year][mode]["data_seed42"]["tail_guard"]
            a = pd.read_csv(out / metrics["trajectory"])
            b = pd.read_csv(source / original["trajectory"])
            difference = float(np.abs(a[list(columns)].to_numpy() - b[list(columns)].to_numpy()).max())
            if difference > 1e-5:
                raise RuntimeError("Nominal feedback-only did not reproduce frozen v6.")
            output[year][mode] = {"passed": True, "max_difference_mw_mwh": difference}
    return output


def contrast(out, left, right):
    a = pd.read_csv(out / left["trajectory"], parse_dates=["timestamp"])
    b = pd.read_csv(out / right["trajectory"], parse_dates=["timestamp"])
    if not a.timestamp.equals(b.timestamp) or not np.allclose(a.net_load_mw, b.net_load_mw):
        raise ValueError("Comparisons require identical hours and physical input loads.")
    daily = (a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
             - b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy()
    return {
        "left_trajectory": left["trajectory"], "right_trajectory": right["trajectory"],
        "differences": {k: left[k] - right[k] for k in MEASURES},
        "daily_peak_interval": daily_ci(daily),
    }


def comparisons(out, closed, opened):
    rows = {}
    for year in YEARS:
        for mode in MODES:
            for case in CASES:
                base = closed[year][mode][case]["healthy"]
                for variant in VARIANTS:
                    rows[f"{year}/{mode}/{case}/{variant}_minus_open"] = contrast(
                        out, base[variant], opened[year][mode][case])
                rows[f"{year}/{mode}/{case}/supervised_minus_feedback"] = contrast(
                    out, base["supervised"], base["feedback_only"])
        for event in EVENTS:
            for variant in VARIANTS:
                rows[f"{year}/fresh6/combined90/{event}/{variant}_minus_healthy"] = contrast(
                    out, closed[year]["fresh6"]["combined90"][event][variant],
                    closed[year]["fresh6"]["combined90"]["healthy"][variant])
    return rows


def capability_checks(results, replication):
    conditions = {}
    in_range = []
    for year, modes in results.items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, variants in events.items():
                    for variant, metrics in variants.items():
                        key = f"{year}/{mode}/{case}/{event}/{variant}"
                        gates = metrics["gated_optimizer_calls"] == 0
                        conditions[key] = {
                            "hard_physics": metrics["physical_audit_passed"],
                            "no_optimizer_when_gated": gates,
                            "terminal": metrics["terminal_requirement_passed"],
                            "within_recovery_envelope": metrics["plant_within_declared_envelope"],
                        }
                        if variant == "supervised" and metrics["plant_within_declared_envelope"]:
                            in_range.append(bool(gates and metrics["terminal_requirement_passed"]
                                                 and metrics["max_actual_floor_violation_mwh"] <= 1e-5
                                                 and metrics["max_worst_case_floor_violation_mwh"] <= 1e-5))
    return {
        "nominal_replication_passed": all(x["passed"] for modes in replication.values() for x in modes.values()),
        "closed_physics_passed": all(x["hard_physics"] for x in conditions.values()),
        "gated_no_optimizer_passed": all(x["no_optimizer_when_gated"] for x in conditions.values()),
        "supervised_in_envelope_recovery_passed": bool(in_range and all(in_range)),
        "conditions": conditions,
        "realtime_default_admitted": False,
        "ui_action": "offline_labelled_replay_only_no_ui_change",
        "boundary": "Perfect simulator SOC and supplied recovery bounds, not field qualification or online identification.",
    }


def figures(out, results, opened):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["P90", "P70", "eta90", "eta85", "combined"]
    cases = CASES[1:]
    for ax, year in zip(axes, YEARS):
        x = np.arange(len(cases))
        for j, (variant, color) in enumerate((("open", "#bb7a28"), ("feedback_only", "#2374ae"), ("supervised", "#18816c"))):
            vals = [opened[year]["fresh6"][case]["max_abs_terminal_error_mwh"] if variant == "open"
                    else results[year]["fresh6"][case]["healthy"][variant]["max_abs_terminal_error_mwh"]
                    for case in cases]
            ax.bar(x + (j - 1) * .25, vals, width=.25, label=variant, color=color)
        ax.set_xticks(x, labels)
        ax.set(title=year, ylabel="Maximum daily terminal error (MWh)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, year in zip(axes, YEARS):
        labels2 = ["healthy", *EVENTS]
        for variant, marker, color in (("feedback_only", "o", "#2374ae"), ("supervised", "s", "#18816c")):
            values = [results[year]["fresh6"]["combined90"][e][variant]["controlled_peak_mw"] for e in labels2]
            ax.plot(labels2, values, marker=marker, color=color, label=variant)
        ax.set(title=year, ylabel="Controlled maximum (MW)")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v7")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("At most3 CPU workers.")
    out, source = args.output, ROOT / "outputs/fusion_v6"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        calibration, samples, previous = read(source / "calibration.json"), read(source / "samples.json"), read(source / "metrics.json")
        panels, input_hashes = {}, {}
        dates = ("timestamp", "source_first_target_timestamp", "issued_timestamp", "available_timestamp")
        for year in YEARS:
            panels[year] = {}
            for mode in (*MODES, "delayed3"):
                path = source / "panels" / year / "data_seed42" / f"{mode}.csv"
                input_hashes[path.relative_to(source).as_posix()] = sha(path)
                panels[year][mode] = pd.read_csv(path, parse_dates=list(dates))
                if len(panels[year][mode]) != samples[year]["rows"]:
                    raise ValueError("The v6 common sample must remain unchanged.")
        paths = [
            Path(__file__), ROOT / "scripts/verify_feedback_results.py",
            ROOT / "scripts/run_tail_experiment.py", ROOT / "scripts/run_dispatch_experiment.py",
            ROOT / "scripts/verify_dispatch_results.py", ROOT / "scripts/verify_tail_results.py",
            *[ROOT / f"src/{n}.py" for n in (
                "feedback_control", "recovery_control", "robustness_control", "forecast_dispatch",
                "balanced_storage", "scenarios", "tail_control", "trajectory_audit", "efficient_storage",
                "optimality", "energy_twin", "numerics", "multihorizon_fusion", "physics_forecast")],
            ROOT / "tests/test_feedback_control.py", ROOT / "experiments/fusion_v7/protocol.md",
            ROOT / "experiments/fusion_v7/run_remote.sh",
        ]
        save(out / "manifest.json", {
            "created_utc": utc_now(), "gpu_used": False, "workers": args.workers,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "input_panel_sha256": input_hashes,
            "source_metrics_sha256": sha(source / "metrics.json"),
            "source_calibration_sha256": sha(source / "calibration.json"),
            "source_samples_sha256": sha(source / "samples.json"),
            "assumed_model": asdict(StorageModel()), "recovery_envelope": asdict(RecoveryEnvelope()),
            "actual_models": {k: asdict(m) for k, m in plant_models().items()},
            "boundary": "Offline state-feedback simulation; true plant parameters only in actuator, no real battery.",
        })
        save(out / "calibration.json", calibration)
        save(out / "samples.json", samples)
        print("V7: hourly plant SOC feedback plus auditable stale/outage recovery; offline only.", flush=True)
        refs, ref_metrics = reference_streams(out, panels, calibration)
        save(out / "references.json", ref_metrics)
        opened, hashes = open_comparators(out, source, previous, calibration)
        save(out / "source_open_hashes.json", hashes)
        jobs = []
        for year in YEARS:
            for mode in MODES:
                for case in CASES:
                    for variant in VARIANTS:
                        jobs.append((year, mode, case, "healthy", variant, panels[year][mode],
                                     panels[year]["delayed3"], refs[year][mode]["healthy"], calibration, str(out)))
            for event in EVENTS:
                for variant in VARIANTS:
                    jobs.append((year, "fresh6", "combined90", event, variant, panels[year]["fresh6"],
                                 panels[year]["delayed3"], refs[year]["fresh6"][event], calibration, str(out)))
            jobs.append((year, "fresh6", "outside", "healthy", "supervised", panels[year]["fresh6"],
                         panels[year]["delayed3"], refs[year]["fresh6"]["healthy"], calibration, str(out)))
        results, failures = {}, []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(job, args): args[:5] for args in jobs}
            for future in as_completed(futures):
                try:
                    year, mode, case, event, variant, metrics = future.result()
                    results.setdefault(year, {}).setdefault(mode, {}).setdefault(case, {}).setdefault(event, {})[variant] = metrics
                except Exception as exc:
                    failures.append({"job": list(futures[future]), "error": str(exc)})
        save(out / "failures.json", failures)
        if failures:
            raise RuntimeError("Failed feedback jobs retained; no complete study claimed.")
        replication = replication_check(out, results, source, previous)
        save(out / "nominal_replication.json", replication)
        capabilities = capability_checks(results, replication)
        save(out / "capability_checks.json", capabilities)
        report = {
            "schema": "gridpulse-feedback-fallback.v7", "status": "complete",
            "closed": results, "open": opened, "references": ref_metrics,
            "comparisons": comparisons(out, results, opened), "capability_checks": capabilities,
            "seconds": time.monotonic() - started,
            "boundary": "No winner selection; supervision may relax forecast guard and increase peaks/losses. "
                        "Offline replay only; perfect SOC and declared lower envelope.",
        }
        save(out / "metrics.json", report)
        rows = []
        for year, modes in results.items():
            for mode, cases in modes.items():
                for case, events in cases.items():
                    for event, variants in events.items():
                        for variant, m in variants.items():
                            rows.append({
                                "year": year, "mode": mode, "case": case, "event": event, "variant": variant,
                                **{k: m[k] for k in MEASURES}, **{k: m[k] for k in (
                                    "terminal_failure_days", "fallback_hours", "primary_solver_failure_hours",
                                    "expiry_rejections", "outage_fallback_hours", "supervisor_interventions",
                                    "recovery_unreachable_hours", "max_actual_floor_violation_mwh")},
                            })
        pd.DataFrame(rows).to_csv(out / "method_table.csv", index=False)
        figures(out, results, opened)
        save(out / "status.json", {
            "status": "complete", "closed_loop_traces": 62, "open_loop_traces": 24,
            "reference_traces": 10, "byte_reused_open_loop_traces": 20,
            "realtime_default_admitted": False,
        })
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V7_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
