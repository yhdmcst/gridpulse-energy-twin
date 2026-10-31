"""Reference-relative recovery experiment; only validation may select a policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
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

from src.forecast_dispatch import dispatch, replay
from src.optimality import StorageModel
from src.recovery_control import POLICIES, simulate, validate_reference
from src.recovery_selection import CANDIDATES, SEEDS, select_policy
from src.tail_control import augment_metrics
from scripts.run_tail_experiment import METHODS, PERIODS, MEASURES, load_period, read, save_trace, utc_now
from scripts.run_dispatch_experiment import daily_ci, sha, save


def attach_references(source, period, metadata, panels, out):
    refs, legacy, hashes = {}, {}, {}
    for method in METHODS:
        previous = metadata[method]
        path = source / previous["trajectory"]
        hashes[previous["trajectory"]] = sha(path)
        trace = pd.read_csv(path, parse_dates=["timestamp"])
        if (not trace.timestamp.equals(panels[method].timestamp)
                or not np.allclose(trace.net_load_mw, panels[method].net_load_mw)):
            raise ValueError("Reference actuals and timestamps must match the fixed predictor sample.")
        refs[method] = validate_reference(trace, panels[method].timestamp, StorageModel())
        metrics = dict(previous, max_reference_debt_mwh=0., debt_hours=0,
                       max_headroom_violation_mw=None, seconds=0., reference_reused=True)
        legacy[method] = save_trace(
            out, trace, metrics, f"{period}/legacy/{method}.csv", copy_source=path)
    return refs, legacy, hashes


def job(args):
    period, control, method, panel, reference, cal, gates, tail, output = args
    out = Path(output)
    status = out / "jobs" / f"{period}_{control}_{method}.json"
    started = time.monotonic()
    try:
        trace, metrics = simulate(panel, reference, cal, gates, tail, POLICIES[control])
        metrics["seconds"] = time.monotonic() - started
        metrics = save_trace(out, trace, metrics, f"{period}/{control}/{method}.csv")
        save(status, {"status": "complete", "metrics": metrics})
        return period, control, method, metrics
    except Exception:
        save(status, {"status": "failed", "traceback": traceback.format_exc()})
        raise


def run_jobs(jobs, out, workers, stage):
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(job, args): args[:3] for args in jobs}
        for future in as_completed(futures):
            try:
                period, control, method, metrics = future.result()
                results.setdefault(period, {}).setdefault(control, {})[method] = metrics
                print(f"{period} {control} {method}: P95={metrics['controlled_p95_mw']:.3f} "
                      f"peak={metrics['controlled_peak_mw']:.3f} "
                      f"debt={metrics['max_reference_debt_mwh']:.3f}", flush=True)
            except Exception as exc:
                failures.append({"job": list(futures[future]), "error": str(exc)})
    save(out / f"{stage}_failures.json", failures)
    if failures:
        raise RuntimeError(f"{stage} failed jobs preserved; cannot publish a complete study.")
    return results


def daily_values(out, metrics, tail):
    frame = pd.read_csv(out / metrics["trajectory"], parse_dates=["timestamp"])
    dates = frame.timestamp.dt.floor("D")
    return pd.DataFrame({
        "peak": frame.groupby(dates).controlled_net_load_mw.max(),
        "excess": np.maximum(frame.controlled_net_load_mw - tail, 0.).groupby(dates).mean(),
    })


def compare(out, results, selected, tail):
    all_pairs, aggregate = {}, {}
    for period in PERIODS:
        all_pairs[period] = {}
        for control in CANDIDATES:
            all_pairs[period][control] = {}
            for method in METHODS:
                a, b = results[period][control][method], results[period]["legacy"][method]
                da, db = daily_values(out, a, tail), daily_values(out, b, tail)
                if not da.index.equals(db.index):
                    raise ValueError("Daily contrasts must use matched days.")
                all_pairs[period][control][method] = {
                    "differences": {key: a[key] - b[key] for key in MEASURES},
                    "daily_peak_interval": daily_ci((da.peak - db.peak).to_numpy()),
                    "daily_excess_interval": daily_ci((da.excess - db.excess).to_numpy()),
                }
        deltas, peaks, excess = [], [], []
        for seed in SEEDS:
            method = f"data_seed{seed}"
            a, b = results[period][selected][method], results[period]["legacy"][method]
            deltas.append({key: a[key] - b[key] for key in MEASURES})
            da, db = daily_values(out, a, tail), daily_values(out, b, tail)
            peaks.append((da.peak - db.peak).to_numpy())
            excess.append((da.excess - db.excess).to_numpy())
        aggregate[period] = {
            "fixed_seed_mean_differences": {key: float(np.mean([d[key] for d in deltas])) for key in MEASURES},
            "daily_peak_interval": daily_ci(np.mean(peaks, axis=0)),
            "daily_excess_interval": daily_ci(np.mean(excess, axis=0)),
        }
    return {"all_paired": all_pairs, "selected_data_seed_mean": aggregate}


def stress_experiments(out, panels, calibration, tail, results, selected):
    output, changed_references = {}, {}
    models = {
        "power_70pct": replace(StorageModel(), charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_085": replace(StorageModel(), efficiency=.85),
    }
    method = "data_seed42"
    for period in ("2021", "2022"):
        original = pd.read_csv(out / results[period][selected][method]["trajectory"],
                               parse_dates=["timestamp"])
        output[period], changed_references[period] = {}, {}
        for scenario, model in models.items():
            panel, cal = panels[period][method], calibration["methods"][method]
            ref, ref_metrics = dispatch(panel, cal, calibration["gates"], model)
            ref_metrics = augment_metrics(ref, ref_metrics, calibration["gates"], tail)
            changed_references[period][scenario] = save_trace(
                out, ref, ref_metrics, f"changed_references/{period}_{scenario}.csv")
            rtrace, rm = replay(original, model)
            rm = augment_metrics(rtrace, rm, calibration["gates"], tail)
            if selected == "legacy":
                ptrace, pm = ref, ref_metrics
            else:
                ptrace, pm = simulate(
                    panel, ref, cal, calibration["gates"], tail, POLICIES[selected], model)
            if not rtrace.timestamp.equals(ptrace.timestamp):
                raise ValueError("Paired stress timestamps differ.")
            output[period][scenario] = {
                arm: save_trace(out, trace, metrics, f"stress/{period}_{scenario}_{arm}.csv")
                for arm, trace, metrics in (("replay", rtrace, rm), ("replan", ptrace, pm))
            }
            print(f"stress {period} {scenario}: replay_terminal={rm['max_abs_terminal_error_mwh']:.3f} "
                  f"replan_terminal={pm['max_abs_terminal_error_mwh']:.3f}", flush=True)
    return output, changed_references


def figures(out, report, validation, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["tail", "debt", "guard", "pkDebt", "pkGuard"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    p95 = [selection["candidates"][name]["mean_p95_delta_mw"] for name in CANDIDATES]
    peak = [max(
        validation[name][f"data_seed{s}"]["controlled_peak_mw"]
        - validation["legacy"][f"data_seed{s}"]["controlled_peak_mw"] for s in SEEDS)
            for name in CANDIDATES]
    colors = ["#18816c" if selection["candidates"][n]["eligible"] else "#b77b2d" for n in CANDIDATES]
    axes[0].bar(labels, p95, color=colors)
    axes[0].axhline(-1., color="black", linewidth=.8)
    axes[0].set_ylabel("Validation mean P95 delta (MW)")
    axes[1].bar(labels, peak, color=colors)
    axes[1].axhline(25., color="black", linewidth=.8)
    axes[1].set_ylabel("Worst-seed validation max delta (MW)")
    fig.suptitle(f"Fixed selection: {report['selected_policy']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, period in zip(axes, PERIODS):
        values = [float(np.mean([
            report["methods"][period][n][f"data_seed{s}"]["controlled_p95_mw"]
            - report["methods"][period]["legacy"][f"data_seed{s}"]["controlled_p95_mw"]
            for s in SEEDS])) for n in CANDIDATES]
        ax.bar(labels, values, color=["#b77b2d", "#2374ae", "#18816c", "#a13d68", "#657b83"])
        ax.axhline(0., color="black", linewidth=.8)
        ax.set(title=period, ylabel="Fixed-seed mean P95 delta (MW)")
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/fusion_v3")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/fusion_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v5")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("One to three CPU workers required.")
    out, source, ref_source = args.output, args.predictions, args.reference
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        calibration, samples = read(ref_source / "calibration.json"), read(ref_source / "samples.json")
        tail = calibration["tail_threshold_mw"]
        validation_frames, panels, prediction_hashes = load_period(source, "validation", calibration, samples)
        if abs(validation_frames["data_seed42"].actual_h1_mw.quantile(.9) - tail) > 1e-6:
            raise ValueError("The fixed validation tail threshold must remain unchanged.")
        prior_validation = read(ref_source / "validation_results.json")
        references, legacy, ref_hashes = attach_references(
            ref_source, "validation", prior_validation["legacy"], panels, out)
        paths = [
            Path(__file__), ROOT / "scripts/verify_recovery_results.py",
            ROOT / "scripts/verify_tail_results.py", ROOT / "scripts/verify_dispatch_results.py",
            ROOT / "scripts/run_tail_experiment.py", ROOT / "scripts/run_dispatch_experiment.py",
            *[ROOT / f"src/{name}.py" for name in (
                "recovery_control", "recovery_selection", "tail_control", "tail_selection",
                "forecast_dispatch", "balanced_storage", "multihorizon_fusion", "physics_forecast",
                "scenarios", "trajectory_audit", "efficient_storage", "optimality", "energy_twin", "numerics")],
            ROOT / "tests/test_recovery_control.py", ROOT / "experiments/fusion_v5/protocol.md",
            ROOT / "experiments/fusion_v5/run_remote.sh", ROOT / "docs/fusion-v4-evaluation.md",
        ]
        save(out / "manifest.json", {
            "created_utc": utc_now(), "gpu_used": False, "workers": args.workers,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "validation_prediction_sha256": prediction_hashes,
            "validation_reference_sha256": ref_hashes,
            "source_calibration_sha256": sha(ref_source / "calibration.json"),
            "source_samples_sha256": sha(ref_source / "samples.json"),
            "physical_model": asdict(StorageModel()),
            "policies": {name: asdict(p) for name, p in POLICIES.items()},
            "scope": "Exploratory reference-relative governor; no future reference rows in actions.",
        })
        save(out / "calibration.json", calibration)
        save(out / "samples.json", samples)
        print("V5: test debt and headroom governors; no relaxation of the v4 admission budgets.", flush=True)
        jobs = [
            ("validation", control, method, panel, references[method],
             calibration["methods"][method], calibration["gates"], tail, str(out))
            for control in CANDIDATES for method, panel in panels.items()
        ]
        val = {"legacy": legacy, **run_jobs(jobs, out, args.workers, "validation")["validation"]}
        save(out / "validation_results.json", val)
        selection = select_policy(val)
        selection.update({
            "frozen_utc": utc_now(), "validation_results_sha256": sha(out / "validation_results.json"),
            "protocol_sha256": sha(ROOT / "experiments/fusion_v5/protocol.md"),
            "validation_prediction_sha256": prediction_hashes, "validation_reference_sha256": ref_hashes,
        })
        save(out / "selection.json", selection)
        selected, selection_hash = selection["selected_policy"], sha(out / "selection.json")
        print(f"V5_SELECTION_FROZEN {selected} sha256={selection_hash}", flush=True)
        save(out / "evaluation_start.json", {"started_utc": utc_now(), "selection_sha256": selection_hash})

        # Evaluation reference rows and metrics are loaded only after freezing selection.
        old_results = read(ref_source / "metrics.json")
        eval_prediction_hashes, eval_reference_hashes = {}, {"metrics.json": sha(ref_source / "metrics.json")}
        results, all_panels, jobs = {}, {}, []
        for period in PERIODS:
            _, panels, hashes = load_period(source, period, calibration, samples)
            all_panels[period] = panels
            eval_prediction_hashes.update(hashes)
            references, legacy, hashes = attach_references(
                ref_source, period, old_results["methods"][period]["legacy"], panels, out)
            eval_reference_hashes.update(hashes)
            results[period] = {"legacy": legacy}
            for control in CANDIDATES:
                for method, panel in panels.items():
                    jobs.append((period, control, method, panel, references[method],
                                 calibration["methods"][method], calibration["gates"], tail, str(out)))
        save(out / "evaluation_inputs.json", {
            "predictions": eval_prediction_hashes, "reference": eval_reference_hashes})
        new = run_jobs(jobs, out, args.workers, "evaluation")
        for period in PERIODS:
            results[period].update(new[period])
        comparisons = compare(out, results, selected, tail)
        stress, changed_refs = stress_experiments(out, all_panels, calibration, tail, results, selected)
        report = {
            "schema": "gridpulse-recovery-control.v5", "status": "complete",
            "selected_policy": selected, "selection_sha256": selection_hash,
            "methods": results, "comparisons": comparisons, "stress": stress,
            "changed_references": changed_refs, "seconds": time.monotonic() - started,
            "boundary": "Zero reference debt is a modeled constraint; band headroom is not field safety. "
                        "Known parameter changes; all source years previously inspected.",
        }
        save(out / "metrics.json", report)
        rows = [{"period": period, "policy": control, "method": method, "selected": control == selected,
                 **{key: m[key] for key in MEASURES}, "max_reference_debt_mwh": m["max_reference_debt_mwh"],
                 "terminal_error_mwh": m["max_abs_terminal_error_mwh"]}
                for period, controls in results.items() for control, methods in controls.items()
                for method, m in methods.items()]
        pd.DataFrame(rows).to_csv(out / "method_table.csv", index=False)
        figures(out, report, val, selection)
        save(out / "status.json", {
            "status": "complete", "validation_traces": 24, "evaluation_traces": 72,
            "reused_legacy_traces": 16, "stress_traces": 8, "changed_reference_traces": 4,
            "selected_policy": selected,
        })
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V5_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
