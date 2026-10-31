"""CPU-only controller ablations with a validation freeze before known-year evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from src.multihorizon_fusion import shared_days, planning_panel
from src.forecast_dispatch import dispatch, replay
from src.optimality import StorageModel
from src.tail_control import POLICIES, augment_metrics, simulate
from src.tail_selection import CANDIDATES, SEEDS, select_policy, validate_validation_range
from scripts.run_dispatch_experiment import daily_ci, sha, save

PERIODS = ("test", "2021", "2022")
METHODS = ("hgb", *(f"data_seed{s}" for s in SEEDS))
CONTROLS = ("legacy", *CANDIDATES)
MEASURES = (
    "controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
    "controlled_mean_excess_mw", "throughput_mwh", "conversion_loss_mwh",
    "recovery_charge_mwh", "actual_high_load_charging_mwh",
)


def read(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_period(source, period, calibration, samples):
    frames, hashes = {}, {}
    for method in METHODS:
        path = source / "predictions" / method / f"{period}.csv"
        hashes[path.relative_to(source).as_posix()] = sha(path)
        frame = pd.read_csv(path, parse_dates=["timestamp", "issued_timestamp"])
        if period == "validation":
            validate_validation_range(frame)
        frames[method] = frame
    common, info = shared_days(frames)
    for key in ("rows", "days", "start", "end"):
        if info[key] != samples[period][key]:
            raise ValueError("The v3 complete-day sample must remain unchanged.")
    panels = {method: planning_panel(frame, calibration["methods"][method], 6)
              for method, frame in common.items()}
    return common, panels, hashes


def save_trace(out, trace, metrics, relative, copy_source=None):
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if copy_source is None:
        trace.to_csv(path, index=False)
    else:
        shutil.copyfile(copy_source, path)
    metrics = dict(metrics)
    metrics["trajectory"] = relative
    metrics["trajectory_sha256"] = sha(path)
    return metrics


def job(args):
    stage, period, control, method, panel, cal, gates, tail, output = args
    out = Path(output)
    status = out / "jobs" / stage / f"{period}_{control}_{method}.json"
    started = time.monotonic()
    try:
        if control == "legacy":
            trace, metrics = dispatch(panel, cal, gates)
            metrics = augment_metrics(trace, metrics, gates, tail)
        else:
            trace, metrics = simulate(panel, cal, gates, tail, POLICIES[control])
        metrics["seconds"] = time.monotonic() - started
        metrics = save_trace(out, trace, metrics, f"{stage}/{control}/{method}/{period}.csv")
        save(status, {"status": "complete", "metrics": metrics})
        return control, method, period, metrics
    except Exception:
        save(status, {"status": "failed", "traceback": traceback.format_exc()})
        raise


def run_jobs(jobs, out, workers, stage):
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(job, args): args[:4] for args in jobs}
        for future in as_completed(futures):
            try:
                control, method, period, metrics = future.result()
                results.setdefault(period, {}).setdefault(control, {})[method] = metrics
                print(f"{stage} {period} {control} {method}: "
                      f"P95={metrics['controlled_p95_mw']:.3f} peak={metrics['controlled_peak_mw']:.3f}",
                      flush=True)
            except Exception as exc:
                failures.append({"job": list(futures[future]), "error": str(exc)})
    save(out / f"{stage}_failures.json", failures)
    if failures:
        raise RuntimeError(f"{stage} failed jobs retained; cannot publish a complete comparison.")
    return results


def daily_measures(out, metrics, tail):
    frame = pd.read_csv(out / metrics["trajectory"], parse_dates=["timestamp"])
    dates = frame.timestamp.dt.floor("D")
    values = pd.DataFrame({
        "peak": frame.groupby(dates).controlled_net_load_mw.max(),
        "excess": np.maximum(frame.controlled_net_load_mw - tail, 0.).groupby(dates).mean(),
    })
    return values


def contrasts(out, results, selected, tail):
    pairs, selected_means = {}, {}
    for period in PERIODS:
        pairs[period] = {}
        for control in CANDIDATES:
            pairs[period][control] = {}
            for method in METHODS:
                a, b = results[period][control][method], results[period]["legacy"][method]
                da, db = daily_measures(out, a, tail), daily_measures(out, b, tail)
                if not da.index.equals(db.index):
                    raise ValueError("Paired timestamps differ.")
                pairs[period][control][method] = {
                    "differences": {key: a[key] - b[key] for key in MEASURES},
                    "daily_peak_interval": daily_ci((da.peak - db.peak).to_numpy()),
                    "daily_excess_interval": daily_ci((da.excess - db.excess).to_numpy()),
                }
        deltas, daily_peak, daily_excess = [], [], []
        for seed in SEEDS:
            method = f"data_seed{seed}"
            a, b = results[period][selected][method], results[period]["legacy"][method]
            deltas.append({key: a[key] - b[key] for key in MEASURES})
            da, db = daily_measures(out, a, tail), daily_measures(out, b, tail)
            daily_peak.append((da.peak - db.peak).to_numpy())
            daily_excess.append((da.excess - db.excess).to_numpy())
        selected_means[period] = {
            "fixed_seed_mean_differences": {key: float(np.mean([d[key] for d in deltas])) for key in MEASURES},
            "daily_peak_interval": daily_ci(np.mean(daily_peak, axis=0)),
            "daily_excess_interval": daily_ci(np.mean(daily_excess, axis=0)),
        }
    return {"all_paired": pairs, "selected_data_seed_mean": selected_means}


def stress_experiments(out, panels, calibration, tail, results, selected):
    method = "data_seed42"
    conditions = {
        "power_70pct": replace(StorageModel(), charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_085": replace(StorageModel(), efficiency=.85),
    }
    output = {}
    for period in ("2021", "2022"):
        original = pd.read_csv(out / results[period][selected][method]["trajectory"],
                               parse_dates=["timestamp"])
        output[period] = {}
        for condition, model in conditions.items():
            rtrace, rm = replay(original, model)
            rm = augment_metrics(rtrace, rm, calibration["gates"], tail)
            if selected == "legacy":
                ptrace, pm = dispatch(panels[period][method], calibration["methods"][method],
                                      calibration["gates"], model)
                pm = augment_metrics(ptrace, pm, calibration["gates"], tail)
            else:
                ptrace, pm = simulate(
                    panels[period][method], calibration["methods"][method], calibration["gates"],
                    tail, POLICIES[selected], model)
            if not rtrace.timestamp.equals(ptrace.timestamp):
                raise ValueError("Stress pair timestamps differ.")
            output[period][condition] = {
                arm: save_trace(out, trace, metrics, f"stress/{period}_{condition}_{arm}.csv")
                for arm, trace, metrics in (("replay", rtrace, rm), ("replan", ptrace, pm))
            }
            print(f"stress {period} {condition}: terminal replay={rm['max_abs_terminal_error_mwh']:.3f}"
                  f" replan={pm['max_abs_terminal_error_mwh']:.3f}", flush=True)
    return output


def figures(out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = report["selected_policy"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, period in zip(axes, PERIODS):
        values = [
            report["methods"][period][selected][m]["controlled_p95_mw"]
            - report["methods"][period]["legacy"][m]["controlled_p95_mw"] for m in METHODS]
        ax.bar(range(4), values, color=["#bd7221", "#2374ae", "#18816c", "#a13d68"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(4), ["HGB", "data42", "data43", "data44"], rotation=30)
        ax.set(title=period, ylabel="Selected minus legacy P95 (MW)")
    fig.suptitle(f"Validation selection: {selected}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in (
        (axes[0], "controlled_peak_mw", "Selected minus legacy max (MW)"),
        (axes[1], "conversion_loss_mwh", "Selected minus legacy loss (MWh)"),
    ):
        values = [report["comparisons"]["selected_data_seed_mean"][p]["fixed_seed_mean_differences"][key]
                  for p in PERIODS]
        ax.bar(PERIODS, values, color=["#2374ae", "#18816c", "#a13d68"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/fusion_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v4")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("One to three bounded CPU workers required.")
    out, source = args.output, args.input
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        calibration, samples = read(source / "calibration.json"), read(source / "samples.json")
        validation, vpanels, vhashes = load_period(source, "validation", calibration, samples)
        reference = validation["data_seed42"]
        tail = float(reference.actual_h1_mw.quantile(.9))
        files = [
            Path(__file__), ROOT / "scripts/verify_tail_results.py",
            ROOT / "scripts/run_dispatch_experiment.py",
            *[ROOT / f"src/{name}.py" for name in (
                "tail_control", "tail_selection", "multihorizon_fusion", "physics_forecast",
                "forecast_dispatch", "balanced_storage", "scenarios", "trajectory_audit",
                "efficient_storage", "optimality", "energy_twin", "numerics")],
            ROOT / "tests/test_tail_control.py",
            ROOT / "experiments/fusion_v4/protocol.md", ROOT / "experiments/fusion_v4/run_remote.sh",
        ]
        save(out / "manifest.json", {
            "created_utc": utc_now(), "workers": args.workers, "gpu_used": False,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
            "validation_input_sha256": vhashes,
            "source_calibration_sha256": sha(source / "calibration.json"),
            "source_samples_sha256": sha(source / "samples.json"),
            "policies": {name: asdict(policy) for name, policy in POLICIES.items()},
            "physical_model": asdict(StorageModel()), "tail_threshold_mw": tail,
            "boundary": "Exploratory previously viewed years; no evaluation-based controller selection.",
        })
        save(out / "calibration.json", {**calibration, "tail_threshold_mw": tail})
        save(out / "samples.json", {p: samples[p] for p in ("validation", *PERIODS)})
        print("V4: validate tail proxies and reserve ablations; freeze selection before year evaluation.",
              flush=True)
        validation_jobs = [
            ("validation", "validation", control, method, panel,
             calibration["methods"][method], calibration["gates"], tail, str(out))
            for control in CONTROLS for method, panel in vpanels.items()
        ]
        val = run_jobs(validation_jobs, out, args.workers, "validation")["validation"]
        save(out / "validation_results.json", val)
        selection = select_policy(val)
        selection.update({
            "frozen_utc": utc_now(), "validation_results_sha256": sha(out / "validation_results.json"),
            "validation_input_sha256": vhashes, "protocol_sha256": sha(ROOT / "experiments/fusion_v4/protocol.md"),
        })
        save(out / "selection.json", selection)
        selection_hash = sha(out / "selection.json")
        selected = selection["selected_policy"]
        print(f"VALIDATION_SELECTION_FROZEN {selected} sha256={selection_hash}", flush=True)
        save(out / "evaluation_start.json", {"started_utc": utc_now(), "selection_sha256": selection_hash})

        # No evaluation-period outcomes are parsed until the selection receipt exists.
        old_results = read(source / "metrics.json")
        results, all_panels, input_hashes, jobs = {}, {}, {"metrics.json": sha(source / "metrics.json")}, []
        for period in PERIODS:
            _, panels, hashes = load_period(source, period, calibration, samples)
            all_panels[period] = panels
            input_hashes.update(hashes)
            results[period] = {"legacy": {}}
            for method, panel in panels.items():
                previous = old_results["methods"][period][f"{method}_w6"]
                path = source / previous["trajectory"]
                trace = pd.read_csv(path, parse_dates=["timestamp"])
                if (not trace.timestamp.equals(panel.timestamp)
                        or not np.allclose(trace.net_load_mw, panel.net_load_mw)):
                    raise ValueError("Legacy replay sample differs from the matched six-hour panel.")
                input_hashes[previous["trajectory"]] = sha(path)
                metrics = augment_metrics(trace, previous, calibration["gates"], tail)
                results[period]["legacy"][method] = save_trace(
                    out, trace, metrics, f"evaluation/legacy/{method}/{period}.csv", copy_source=path)
                for control in CANDIDATES:
                    jobs.append((
                        "evaluation", period, control, method, panel,
                        calibration["methods"][method], calibration["gates"], tail, str(out)))
        save(out / "evaluation_inputs.json", input_hashes)
        new_results = run_jobs(jobs, out, args.workers, "evaluation")
        for period in PERIODS:
            results[period].update(new_results[period])
        comparisons = contrasts(out, results, selected, tail)
        stress = stress_experiments(out, all_panels, calibration, tail, results, selected)
        report = {
            "schema": "gridpulse-tail-control.v4", "status": "complete",
            "selected_policy": selected, "selection_sha256": selection_hash,
            "methods": results, "comparisons": comparisons, "stress": stress,
            "seconds": time.monotonic() - started,
            "boundary": "Fixed validation threshold proxy, not direct empirical P95/CVaR optimization. "
                        "Known parameter changes, no field safety or unseen-year confirmation.",
        }
        save(out / "metrics.json", report)
        rows = [{"period": period, "policy": control, "method": method,
                 "selected": control == selected, **{key: m[key] for key in MEASURES},
                 "terminal_error_mwh": m["max_abs_terminal_error_mwh"]}
                for period, controls in results.items() for control, methods in controls.items()
                for method, m in methods.items()]
        pd.DataFrame(rows).to_csv(out / "method_table.csv", index=False)
        figures(out, report)
        save(out / "status.json", {
            "status": "complete", "validation_traces": 28, "nominal_traces": 84,
            "reused_legacy_traces": 12, "stress_traces": 8, "selected_policy": selected,
        })
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V4_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
