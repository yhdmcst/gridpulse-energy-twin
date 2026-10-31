"""Frozen-prediction one-hour dispatch comparison and paired twin replanning."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
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

from src.forecast_dispatch import shared_complete_days, calibrate, make_panel, dispatch, replay
from src.optimality import StorageModel

PERIODS = ("test", "2021", "2022")
SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def method_paths() -> dict[str, str]:
    methods = {method: f"baseline/{method}" for method in ("persistence", "seasonal24", "hgb")}
    methods.update({
        f"{arm}_{variant}_seed{seed}": f"{arm}_seed{seed}/{variant}"
        for arm in ("data", "physics") for variant in ("raw", "projected") for seed in SEEDS
    })
    return methods


def load_inputs(source: Path) -> tuple[dict, dict, dict]:
    methods = method_paths()
    frames, samples, hashes = {}, {}, {}
    for period in ("validation", *PERIODS):
        raw = {}
        for method, relative in methods.items():
            path = source / relative / f"{period}.csv"
            hashes[path.relative_to(source).as_posix()] = sha(path)
            raw[method] = pd.read_csv(path, parse_dates=["timestamp"])
        frames[period], samples[period] = shared_complete_days(raw)
    return frames, samples, hashes


def evaluate_job(args: tuple) -> tuple[str, str, dict]:
    period, method, panel, calibration, gates, output = args
    root = Path(output)
    status = root / "jobs" / f"{period}_{method}.json"
    started = time.monotonic()
    try:
        trace, metrics = dispatch(panel, calibration, gates)
        destination = root / "traces" / method / f"{period}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(destination, index=False)
        metrics["seconds"] = time.monotonic() - started
        metrics["trajectory"] = destination.relative_to(root).as_posix()
        metrics["trajectory_sha256"] = sha(destination)
        save(status, {"status": "complete", "metrics": metrics})
        return period, method, metrics
    except Exception:
        save(status, {"status": "failed", "traceback": traceback.format_exc()})
        raise


def daily_ci(delta: np.ndarray) -> dict:
    delta = np.asarray(delta, float)
    rng = np.random.default_rng(20261026)
    samples = np.empty(2000)
    for i in range(2000):
        starts = rng.integers(0, len(delta), size=(len(delta) + 6) // 7)
        indices = ((starts[:, None] + np.arange(7)) % len(delta)).ravel()[:len(delta)]
        samples[i] = delta[indices].mean()
    return {"estimate_mw": float(delta.mean()),
            "lower_95_mw": float(np.quantile(samples, .025)),
            "upper_95_mw": float(np.quantile(samples, .975)),
            "days": len(delta), "block_days": 7, "replicates": 2000}


def daily_peaks(root: Path, item: dict) -> pd.Series:
    frame = pd.read_csv(root / item["trajectory"], parse_dates=["timestamp"])
    return frame.groupby(frame.timestamp.dt.floor("D")).controlled_net_load_mw.max()


def compare(results: dict, output: Path) -> dict:
    comparisons = {}
    families = (
        ("data_raw_minus_hgb", "data_raw", "hgb"),
        ("data_projected_minus_data_raw", "data_projected", "data_raw"),
        ("physics_raw_minus_data_raw", "physics_raw", "data_raw"),
        ("physics_projected_minus_data_projected", "physics_projected", "data_projected"),
    )
    measures = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
                "throughput_mwh", "conversion_loss_mwh", "forecast_h1_mae_mw")
    for period, methods in results.items():
        comparisons[period] = {}
        for label, left_family, right_family in families:
            deltas = []
            daily = []
            for seed in SEEDS:
                a = methods[f"{left_family}_seed{seed}"]
                b = methods[right_family if right_family == "hgb" else f"{right_family}_seed{seed}"]
                deltas.append({key: a[key] - b[key] for key in measures})
                left = daily_peaks(output, a)
                right = daily_peaks(output, b)
                if not left.index.equals(right.index):
                    raise ValueError("Daily contrast timestamps differ.")
                daily.append((left - right).to_numpy())
            comparisons[period][label] = {
                "seed_differences": {str(seed): value for seed, value in zip(SEEDS, deltas)},
                "mean_differences": {key: float(np.mean([x[key] for x in deltas])) for key in measures},
                "conditional_mean_daily_peak_interval": daily_ci(np.mean(daily, axis=0)),
            }
    return comparisons


def stress_experiments(panels: dict, calibration: dict, results: dict, output: Path) -> dict:
    method = "data_projected_seed42"
    nominal = StorageModel()
    scenarios = {
        "power_70pct": replace(nominal, charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_085": replace(nominal, efficiency=.85),
    }
    stress = {}
    for period in ("2021", "2022"):
        original = pd.read_csv(
            output / results[period][method]["trajectory"], parse_dates=["timestamp"])
        stress[period] = {}
        for scenario, model in scenarios.items():
            replay_trace, replay_metrics = replay(original, model)
            replan_trace, replan_metrics = dispatch(
                panels[period][method], calibration["methods"][method], calibration["gates"], model)
            if not replay_trace.timestamp.equals(replan_trace.timestamp):
                raise ValueError("Stress pair timestamps differ.")
            values = {}
            for arm, trace, metrics in (("replay", replay_trace, replay_metrics),
                                        ("replan", replan_trace, replan_metrics)):
                path = output / "stress" / f"{period}_{scenario}_{arm}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                trace.to_csv(path, index=False)
                metrics["trajectory"] = path.relative_to(output).as_posix()
                metrics["trajectory_sha256"] = sha(path)
                values[arm] = metrics
            stress[period][scenario] = values
            print(json.dumps({
                "stress": scenario, "period": period,
                "replay_terminal_error": replay_metrics["max_abs_terminal_error_mwh"],
                "replan_terminal_error": replan_metrics["max_abs_terminal_error_mwh"],
            }), flush=True)
    return stress


def figures(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = ("hgb", "data_raw", "data_projected", "physics_projected")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, period in zip(axes, PERIODS):
        values = []
        for family in families:
            metrics = report["methods"][period]
            values.append(metrics[family]["mean_daily_peak_mw"] if family == "hgb" else float(np.mean(
                [metrics[f"{family}_seed{s}"]["mean_daily_peak_mw"] for s in SEEDS])))
        ax.bar(range(4), values, color=["#d97706", "#2563eb", "#0f766e", "#9b277d"])
        ax.set_xticks(range(4), families, rotation=40, ha="right")
        ax.set(title=period, ylabel="Mean daily controlled peak (MW)")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, period in zip(axes, ("2021", "2022")):
        scenarios = report["stress"][period]
        x = np.arange(len(scenarios))
        for arm, offset, color in (("replay", -.18, "#d97706"), ("replan", .18, "#0f766e")):
            ax.bar(x + offset, [scenarios[k][arm]["max_abs_terminal_error_mwh"] for k in scenarios],
                   width=.36, color=color, label=arm)
        ax.set_xticks(x, list(scenarios))
        ax.set(title=period, ylabel="Maximum daily terminal SOC error (MWh)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "Figure_2.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/fusion_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v2")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use at most three bounded CPU workers.")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        frames, samples, hashes = load_inputs(args.input)
        calibration = calibrate(frames["validation"])
        code_paths = [
            Path(__file__), ROOT / "src/forecast_dispatch.py",
            ROOT / "src/balanced_storage.py", ROOT / "src/scenarios.py",
            ROOT / "src/trajectory_audit.py", ROOT / "src/efficient_storage.py",
            ROOT / "src/optimality.py", ROOT / "src/numerics.py", ROOT / "src/energy_twin.py",
            ROOT / "experiments/fusion_v2/protocol.md",
        ]
        save(args.output / "manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "input_sha256": hashes, "code_sha256": {
                p.relative_to(ROOT).as_posix(): sha(p) for p in code_paths},
            "workers": args.workers, "forecast_hours": 1,
            "gpu_used": False, "physical_model": asdict(StorageModel()),
            "scope": "Exploratory known-data forecast-plus-calibration pipeline comparison.",
        })
        save(args.output / "samples.json", samples)
        save(args.output / "calibration.json", calibration)
        panels = {
            period: {name: make_panel(frame, calibration["methods"][name])
                     for name, frame in frames[period].items()}
            for period in PERIODS
        }
        jobs = [
            (period, method, panel, calibration["methods"][method],
             calibration["gates"], str(args.output))
            for period, methods in panels.items() for method, panel in methods.items()
        ]
        results = {period: {} for period in PERIODS}
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_job, job): (job[0], job[1]) for job in jobs}
            for future in as_completed(futures):
                try:
                    period, method, metrics = future.result()
                    results[period][method] = metrics
                    print(json.dumps({
                        "period": period, "method": method,
                        "peak": metrics["controlled_peak_mw"],
                        "p95": metrics["controlled_p95_mw"],
                        "seconds": metrics["seconds"],
                    }), flush=True)
                except Exception as exc:
                    failures.append({"job": list(futures[future]), "error": str(exc)})
        save(args.output / "failures.json", failures)
        if failures:
            raise RuntimeError("Failed runs retained; cannot publish a complete comparison.")
        stress = stress_experiments(panels, calibration, results, args.output)
        contrasts = compare(results, args.output)
        report = {
            "schema": "gridpulse-fusion-dispatch.v2", "status": "complete",
            "forecast_hours": 1, "methods": results, "samples": samples,
            "calibration": calibration, "comparisons": contrasts, "stress": stress,
            "seconds": time.monotonic() - started,
            "boundary": "No model retraining, no untouched holdout, no financial or field-safety claim. "
                        "One-hour controller, not comparison with the original six-hour controller. "
                        "Stress replanning assumes actual parameter changes known at episode start.",
        }
        save(args.output / "metrics.json", report)
        rows = [
            {"period": period, "method": name, **{
                key: metrics[key] for key in (
                    "forecast_h1_mae_mw", "controlled_peak_mw", "controlled_p95_mw",
                    "mean_daily_peak_mw", "throughput_mwh", "conversion_loss_mwh",
                    "max_abs_terminal_error_mwh", "physical_audit_passed")
            }}
            for period, methods in results.items() for name, metrics in methods.items()
        ]
        pd.DataFrame(rows).to_csv(args.output / "method_table.csv", index=False)
        figures(report, args.output)
        save(args.output / "status.json", {"status": "complete", "jobs": len(jobs), "stress_traces": 8})
        save(args.output / "ARTIFACT_SHA256.json", {
            path.relative_to(args.output).as_posix(): sha(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V2_COMPLETE", flush=True)
    except Exception:
        save(args.output / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
