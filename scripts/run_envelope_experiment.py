"""V8 isolated planning-envelope ablation with validation retention frozen first."""
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

from src.delayed_forecasts import common_panels
from src.envelope_planning import POLICIES, planning_model, simulate_policy
from src.envelope_selection import select_policy
from src.feedback_control import RecoveryEnvelope, make_reference, plant_models, simulate_feedback
from src.optimality import StorageModel
from src.tail_selection import validate_validation_range
from scripts.run_dispatch_experiment import daily_ci, save, sha
from scripts.run_tail_experiment import read, save_trace, utc_now

YEARS = ("2021", "2022")
MODES = ("fresh6", "delayed1")
CASES = ("nominal", "power_90pct", "power_70pct", "efficiency_090", "efficiency_085", "combined90")
EVENTS = ("stale_burst", "solver_outage", "combined_fault")
MEASURES = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
            "throughput_mwh", "conversion_loss_mwh", "max_abs_terminal_error_mwh")


def evaluate_job(args):
    stage, year, mode, case, event, policy, base, old, ref, calibration, output = args
    out = Path(output)
    label = f"{stage}_{year}_{mode}_{case}_{event}_{policy}"
    try:
        if policy == "baseline_v7":
            trace, metrics = simulate_feedback(
                base, old, ref, event, plant_models()[case], True,
                calibration["methods"]["data_seed42"], calibration["gates"], calibration["tail_threshold_mw"])
        else:
            trace, metrics = simulate_policy(
                base, old, ref, event, plant_models()[case], policy,
                calibration["methods"]["data_seed42"], calibration["gates"], calibration["tail_threshold_mw"])
        metrics["condition"] = {"stage": stage, "year": year, "mode": mode, "case": case,
                                "event": event, "policy": policy}
        metrics = save_trace(out, trace, metrics, f"{stage}/{year}/{mode}/{case}/{event}/{policy}.csv")
        save(out / "jobs" / f"{label}.json", {"status": "complete", "metrics": metrics})
        print(f"{label}: P95={metrics['controlled_p95_mw']:.3f} "
              f"max={metrics['controlled_peak_mw']:.3f} "
              f"terminal={metrics['max_abs_terminal_error_mwh']:.5f} "
              f"shield={metrics['supervisor_interventions']}", flush=True)
        return year, mode, case, event, policy, metrics
    except Exception:
        save(out / "jobs" / f"{label}.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


def run_jobs(jobs, out, workers, stage):
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(evaluate_job, args): args[:6] for args in jobs}
        for future in as_completed(futures):
            try:
                year, mode, case, event, policy, metrics = future.result()
                results.setdefault(year, {}).setdefault(mode, {}).setdefault(case, {}).setdefault(event, {})[policy] = metrics
            except Exception as exc:
                failures.append({"job": list(futures[future]), "error": str(exc)})
    save(out / f"{stage}_failures.json", failures)
    if failures:
        raise RuntimeError(f"{stage} failures retained; cannot report a complete study.")
    return results


def save_reference(out, base, old, event, assumed, cal, relative):
    trace, metrics = make_reference(
        base, old, event, cal["methods"]["data_seed42"], cal["gates"],
        cal["tail_threshold_mw"], assumed=assumed)
    metrics = save_trace(out, trace, metrics, relative)
    return trace, metrics


def compare(out, left, right):
    a = pd.read_csv(out / left["trajectory"], parse_dates=["timestamp"])
    b = pd.read_csv(out / right["trajectory"], parse_dates=["timestamp"])
    if not a.timestamp.equals(b.timestamp) or not np.allclose(a.net_load_mw, b.net_load_mw):
        raise ValueError("Paired conditions must have identical hours and loads.")
    daily = (a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
             - b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()).to_numpy()
    return {
        "left": left["trajectory"], "right": right["trajectory"],
        "differences": {k: left[k] - right[k] for k in MEASURES},
        "daily_peak_interval": daily_ci(daily),
    }


def figures(out, validation, selection, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, case in zip(axes, ("nominal", "combined90")):
        values = [validation[p][case]["controlled_p95_mw"]
                  - validation["baseline_v7"][case]["controlled_p95_mw"] for p in POLICIES]
        ax.bar(["reference", "tail"], values, color=["#2374ae", "#18816c"])
        ax.axhline(0., color="black", linewidth=.8)
        ax.set(title=case, ylabel="Validation P95 delta vs v7 (MW)")
    fig.suptitle(f"Validation retention: {selection['selected_policy']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, year in zip(axes, YEARS):
        cases = ("nominal", "combined90")
        for policy, marker, color in (
            ("envelope_reference", "o", "#2374ae"), ("envelope_tail", "s", "#18816c")):
            values = [results[year]["fresh6"][case]["healthy"][policy]["controlled_peak_mw"]
                      - results[year]["fresh6"][case]["healthy"]["baseline_v7"]["controlled_peak_mw"]
                      for case in cases]
            ax.plot(cases, values, marker=marker, label=policy, color=color)
        ax.axhline(0., color="black", linewidth=.8)
        ax.set(title=year, ylabel="Maximum delta vs v7 supervised (MW)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v8")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("At most3 CPU workers required.")
    out, v3, v6, v7 = args.output, ROOT / "outputs/fusion_v3", ROOT / "outputs/fusion_v6", ROOT / "outputs/fusion_v7"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        calibration = read(v7 / "calibration.json")
        path = v3 / "predictions/data_seed42/validation.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp", "issued_timestamp"])
        validate_validation_range(frame)
        vp, vsample = common_panels({"data_seed42": frame}, calibration)
        validation_panels = vp["data_seed42"]
        for mode in ("fresh6", "delayed3"):
            path_out = out / "panels/validation" / f"{mode}.csv"
            path_out.parent.mkdir(parents=True, exist_ok=True)
            validation_panels[mode].to_csv(path_out, index=False)
        code_paths = [
            Path(__file__), ROOT / "scripts/verify_envelope_results.py",
            ROOT / "scripts/run_tail_experiment.py", ROOT / "scripts/run_dispatch_experiment.py",
            *[ROOT / f"src/{n}.py" for n in (
                "envelope_planning", "envelope_selection", "feedback_control", "delayed_forecasts",
                "recovery_control", "robustness_control", "forecast_dispatch", "balanced_storage",
                "scenarios", "tail_control", "tail_selection", "trajectory_audit", "efficient_storage",
                "optimality", "energy_twin", "numerics", "multihorizon_fusion", "physics_forecast")],
            ROOT / "tests/test_envelope_planning.py", ROOT / "experiments/fusion_v8/protocol.md",
            ROOT / "experiments/fusion_v8/run_remote.sh",
        ]
        save(out / "manifest.json", {
            "created_utc": utc_now(), "gpu_used": False, "workers": args.workers,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in code_paths},
            "validation_prediction_sha256": sha(path),
            "source_calibration_sha256": sha(v7 / "calibration.json"),
            "nominal_model": asdict(StorageModel()), "planning_model": asdict(planning_model()),
            "recovery_envelope": asdict(RecoveryEnvelope()),
            "scope": "Planning pipeline ablation, fixed seed42 and inspected years; offline only.",
        })
        save(out / "calibration.json", calibration)
        save(out / "validation_sample.json", vsample)
        refs, ref_metrics = {}, {}
        for label, assumed in (("nominal", StorageModel()), ("conservative", planning_model())):
            refs[label], ref_metrics[label] = save_reference(
                out, validation_panels["fresh6"], validation_panels["delayed3"], "healthy",
                assumed, calibration, f"validation_references/{label}.csv")
        save(out / "validation_references.json", ref_metrics)
        print("V8: test planning-time envelope before known-year retention; no guard relaxation.", flush=True)
        jobs = [
            ("validation", "validation", "fresh6", case, "healthy", policy,
             validation_panels["fresh6"], validation_panels["delayed3"],
             refs["nominal" if policy == "baseline_v7" else "conservative"], calibration, str(out))
            for policy in ("baseline_v7", *POLICIES) for case in ("nominal", "combined90")
        ]
        raw = run_jobs(jobs, out, args.workers, "validation")["validation"]["fresh6"]
        validation = {p: {c: raw[c]["healthy"][p] for c in ("nominal", "combined90")}
                      for p in ("baseline_v7", *POLICIES)}
        save(out / "validation_results.json", validation)
        selection = select_policy(validation)
        selection.update({
            "frozen_utc": utc_now(), "validation_results_sha256": sha(out / "validation_results.json"),
            "protocol_sha256": sha(ROOT / "experiments/fusion_v8/protocol.md"),
            "validation_prediction_sha256": sha(path),
        })
        save(out / "selection.json", selection)
        save(out / "evaluation_start.json", {
            "started_utc": utc_now(), "selection_sha256": sha(out / "selection.json")})
        print(f"V8_SELECTION_FROZEN {selection['selected_policy']}", flush=True)
        # No evaluation panels or baseline outcome metrics are parsed before this receipt.
        prior = read(v7 / "metrics.json")
        samples = read(v7 / "samples.json")
        panels, eval_refs, reports, input_hashes = {}, {}, {}, {}
        dates = ["timestamp", "source_first_target_timestamp", "issued_timestamp", "available_timestamp"]
        for year in YEARS:
            panels[year], eval_refs[year], reports[year] = {}, {}, {}
            for mode in (*MODES, "delayed3"):
                p = v6 / "panels" / year / "data_seed42" / f"{mode}.csv"
                input_hashes[f"v6/{p.relative_to(v6).as_posix()}"] = sha(p)
                panels[year][mode] = pd.read_csv(p, parse_dates=dates)
                if len(panels[year][mode]) != samples[year]["rows"]:
                    raise ValueError("Fixed v6 sample changed.")
            for mode in MODES:
                eval_refs[year][mode], reports[year][mode] = {}, {}
                for event in ("healthy", *EVENTS) if mode == "fresh6" else ("healthy",):
                    ref, met = save_reference(
                        out, panels[year][mode], panels[year]["delayed3"], event,
                        planning_model(), calibration, f"references/{year}_{mode}_{event}.csv")
                    eval_refs[year][mode][event], reports[year][mode][event] = ref, met
        save(out / "samples.json", samples)
        save(out / "references.json", reports)
        results, jobs = {}, []
        for year in YEARS:
            results[year] = {}
            conditions = [(mode, case, "healthy") for mode in MODES for case in CASES]
            conditions += [("fresh6", "combined90", e) for e in EVENTS]
            conditions += [("fresh6", "outside", "healthy")]
            for mode, case, event in conditions:
                old = prior["closed"][year][mode][case][event]["supervised"]
                original = v7 / old["trajectory"]
                input_hashes[f"v7/{old['trajectory']}"] = sha(original)
                trace = pd.read_csv(original, parse_dates=["timestamp"])
                copied = save_trace(out, trace, dict(old),
                                    f"baseline/{year}/{mode}/{case}/{event}.csv", copy_source=original)
                copied["source_trajectory"] = old["trajectory"]
                copied["source_sha256"] = sha(original)
                results[year].setdefault(mode, {}).setdefault(case, {}).setdefault(event, {})["baseline_v7"] = copied
                for policy in POLICIES:
                    jobs.append(("evaluation", year, mode, case, event, policy,
                                 panels[year][mode], panels[year]["delayed3"],
                                 eval_refs[year][mode][event], calibration, str(out)))
        input_hashes["v7/metrics.json"] = sha(v7 / "metrics.json")
        save(out / "evaluation_inputs.json", input_hashes)
        new = run_jobs(jobs, out, args.workers, "evaluation")
        contrasts = {}
        for year, modes in new.items():
            for mode, cases in modes.items():
                for case, events in cases.items():
                    for event, policies in events.items():
                        base = results[year][mode][case][event]
                        base.update(policies)
                        for policy in POLICIES:
                            contrasts[f"{year}/{mode}/{case}/{event}/{policy}_minus_v7"] = compare(
                                out, base[policy], base["baseline_v7"])
                        contrasts[f"{year}/{mode}/{case}/{event}/tail_minus_reference"] = compare(
                            out, base["envelope_tail"], base["envelope_reference"])
        in_range = [m for modes in results.values() for cases in modes.values()
                    for events in cases.values() for policies in events.values()
                    for p, m in policies.items() if p in POLICIES and m["plant_within_declared_envelope"]]
        decision = {
            "selected_policy": selection["selected_policy"],
            "selection_sha256": sha(out / "selection.json"),
            "new_in_envelope_runs": len(in_range),
            "new_in_envelope_recovery_passed": bool(in_range and all(
                m["physical_audit_passed"] and m["terminal_requirement_passed"]
                and m["max_actual_floor_violation_mwh"] <= 1e-5
                and m["max_worst_case_floor_violation_mwh"] <= 1e-5
                and m["gated_optimizer_calls"] == 0 for m in in_range)),
            "realtime_default_admitted": False,
            "ui_action": "offline_only_no_ui_changes",
        }
        save(out / "decision.json", decision)
        report = {
            "schema": "gridpulse-planning-envelope.v8", "status": "complete",
            "validation": validation, "selected_policy": selection["selected_policy"],
            "methods": results, "references": reports, "comparisons": contrasts,
            "decision": decision, "seconds": time.monotonic() - started,
            "scope": "Conservative model and reference pipeline change together; no actual parameter identification.",
        }
        save(out / "metrics.json", report)
        rows = [{"year": year, "mode": mode, "case": case, "event": event, "policy": policy,
                 **{k: m[k] for k in MEASURES},
                 "supervisor_interventions": m["supervisor_interventions"],
                 "primary_solver_failures": m["primary_solver_failure_hours"]}
                for year, modes in results.items() for mode, cases in modes.items()
                for case, events in cases.items() for event, policies in events.items()
                for policy, m in policies.items()]
        pd.DataFrame(rows).to_csv(out / "method_table.csv", index=False)
        figures(out, validation, selection, results)
        save(out / "status.json", {
            "status": "complete", "validation_traces": 6, "validation_reference_traces": 2,
            "new_evaluation_traces": 64, "byte_reused_baseline_traces": 32, "reference_traces": 10,
            "selected_policy": selection["selected_policy"], "realtime_default_admitted": False})
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"})
        print("FUSION_V8_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
