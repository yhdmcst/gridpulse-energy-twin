"""Frozen-policy delay and optimistic model-error stress audit, without UI edits."""
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

from src.delayed_forecasts import MODES, common_panels
from src.forecast_dispatch import dispatch
from src.optimality import StorageModel
from src.robustness_admission import MAIN_MODES, demo_admission
from src.robustness_control import annotate, error_models, replay_mismatch, simulate_guard
from src.tail_control import augment_metrics
from scripts.run_dispatch_experiment import daily_ci, save, sha
from scripts.run_tail_experiment import read, save_trace, utc_now

YEARS = ("2021", "2022")
METHODS = tuple(f"data_seed{s}" for s in (42, 43, 44))
MEASURES = (
    "controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw", "throughput_mwh",
    "conversion_loss_mwh", "max_abs_terminal_error_mwh", "controlled_mean_excess_mw",
)


def pair_job(args):
    stage, year, mode, method, case, panel, cal, gates, tail, model, output = args
    out = Path(output)
    status = out / "jobs" / stage / f"{year}_{mode}_{method}_{case}.json"
    started = time.monotonic()
    try:
        ref, lm = dispatch(panel, cal, gates, model)
        lm = augment_metrics(ref, lm, gates, tail)
        candidate, cm = simulate_guard(panel, ref, cal, gates, tail, model)
        pair = {}
        for policy, trace, metrics in (("legacy", ref, lm), ("tail_guard", candidate, cm)):
            trace, metrics = annotate(trace, metrics, panel)
            metrics["assumed_parameters"] = asdict(model)
            metrics["condition"] = {"stage": stage, "year": year, "mode": mode, "method": method, "case": case}
            suffix = method if stage == "delay" else case
            pair[policy] = save_trace(
                out, trace, metrics, f"{stage}/{year}/{mode}/{suffix}/{policy}.csv")
        save(status, {"status": "complete", "seconds": time.monotonic() - started, "pair": pair})
        print(f"{stage} {year} {mode} {method} {case}: "
              f"P95_delta={cm['controlled_p95_mw']-lm['controlled_p95_mw']:.3f} "
              f"peak_delta={cm['controlled_peak_mw']-lm['controlled_peak_mw']:.3f}", flush=True)
        return stage, year, mode, method, case, pair
    except Exception:
        save(status, {"status": "failed", "traceback": traceback.format_exc()})
        raise


def run_jobs(jobs, out, workers, stage):
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(pair_job, args): args[:5] for args in jobs}
        for future in as_completed(futures):
            try:
                _, year, mode, method, case, pair = future.result()
                key = method if stage == "delay" else case
                results.setdefault(year, {}).setdefault(mode, {})[key] = pair
            except Exception as exc:
                failures.append({"job": list(futures[future]), "error": str(exc)})
    save(out / f"{stage}_failures.json", failures)
    if failures:
        raise RuntimeError(f"{stage} failures preserved; no successful robustness claim.")
    return results


def check_replication(out, delay, previous, old_root):
    checks = {}
    columns = ["charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh",
               "controlled_net_load_mw"]
    hashes = {}
    for year in YEARS:
        checks[year] = {}
        for method in METHODS:
            checks[year][method] = {}
            for policy in ("legacy", "tail_guard"):
                item = delay[year]["fresh6"][method][policy]
                old = previous["methods"][year][policy][method]
                path = old_root / old["trajectory"]
                hashes[old["trajectory"]] = sha(path)
                a = pd.read_csv(out / item["trajectory"], parse_dates=["timestamp"])
                b = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").loc[a.timestamp]
                difference = float(np.abs(a[columns].to_numpy() - b[columns].to_numpy()).max())
                if difference > 1e-5:
                    raise RuntimeError("Fresh6 adapter did not reproduce the frozen v5 policy.")
                checks[year][method][policy] = {"passed": True, "max_absolute_action_state_load_difference": difference}
    return checks, hashes


def mismatch_probes(out, delay, panels, calibration):
    output = {}
    for year in YEARS:
        output[year] = {}
        for mode in MAIN_MODES:
            output[year][mode] = {}
            panel = panels[year]["data_seed42"][mode]
            for case, actual in error_models().items():
                output[year][mode][case] = {}
                for policy in ("legacy", "tail_guard"):
                    nominal = delay[year][mode]["data_seed42"][policy]
                    requested = pd.read_csv(out / nominal["trajectory"], parse_dates=["timestamp"])
                    trace, metrics = replay_mismatch(
                        requested, actual, calibration["gates"], calibration["tail_threshold_mw"])
                    trace, metrics = annotate(trace, metrics, panel)
                    metrics["requested_trajectory"] = nominal["trajectory"]
                    metrics["requested_trajectory_sha256"] = nominal["trajectory_sha256"]
                    metrics["condition"] = {"year": year, "mode": mode, "case": case, "policy": policy}
                    output[year][mode][case][policy] = save_trace(
                        out, trace, metrics, f"mismatch/{year}/{mode}/{case}/{policy}.csv")
                print(f"mismatch {year} {mode} {case}: "
                      f"guard_terminal={output[year][mode][case]['tail_guard']['max_abs_terminal_error_mwh']:.3f}",
                      flush=True)
    return output


def contrast(out, left, right):
    a = pd.read_csv(out / left["trajectory"], parse_dates=["timestamp"])
    b = pd.read_csv(out / right["trajectory"], parse_dates=["timestamp"])
    if not a.timestamp.equals(b.timestamp) or not np.allclose(a.net_load_mw, b.net_load_mw):
        raise ValueError("Every contrast must use identical actual loads and hours.")
    peak_a = a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
    peak_b = b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()
    return {
        "left_trajectory": left["trajectory"], "right_trajectory": right["trajectory"],
        "differences": {key: left[key] - right[key] for key in MEASURES},
        "daily_peak_interval": daily_ci((peak_a - peak_b).to_numpy()),
    }


def comparisons(out, delays, mismatch, known):
    result = {"delay_policy": {}, "freshness": {}, "horizon": {}, "mismatch_policy": {}, "known_policy": {}}
    for year in YEARS:
        for section in result:
            result[section][year] = {}
        for mode, methods in delays[year].items():
            result["delay_policy"][year][mode] = {
                method: contrast(out, pair["tail_guard"], pair["legacy"])
                for method, pair in methods.items()}
        for label, left_mode, right_mode in (
            ("delay1_minus_fresh5", "delayed1", "fresh5"),
            ("delay3_minus_fresh3", "delayed3", "fresh3"),
            ("fresh5_minus_fresh6", "fresh5", "fresh6"),
            ("fresh3_minus_fresh6", "fresh3", "fresh6"),
        ):
            section = "freshness" if label.startswith("delay") else "horizon"
            result[section][year][label] = {
                f"{method}/{policy}": contrast(
                    out, delays[year][left_mode][method][policy], delays[year][right_mode][method][policy])
                for method in METHODS for policy in ("legacy", "tail_guard")}
        for mode in MAIN_MODES:
            result["mismatch_policy"][year][mode] = {
                case: contrast(out, pair["tail_guard"], pair["legacy"])
                for case, pair in mismatch[year][mode].items()}
            result["known_policy"][year][mode] = {
                case: contrast(out, pair["tail_guard"], pair["legacy"])
                for case, pair in known[year][mode].items()}
    return result


def figures(out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(MODES)
    for ax, year in zip(axes, YEARS):
        values = [float(np.mean([
            report["delay"][year][mode][method]["tail_guard"]["controlled_p95_mw"]
            - report["delay"][year][mode][method]["legacy"]["controlled_p95_mw"]
            for method in METHODS])) for mode in names]
        ax.bar(names, values, color=["#18816c", "#2374ae", "#bb7a28", "#a63d67", "#657b83"])
        ax.axhline(0., color="black", linewidth=.8)
        ax.set(title=year, ylabel="Guard minus matched legacy P95 (MW)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cases = list(error_models())
    for ax, year in zip(axes, YEARS):
        x = np.arange(len(cases))
        for index, (mode, color) in enumerate(zip(MAIN_MODES, ("#18816c", "#2374ae", "#bb7a28"))):
            ax.bar(x + (index - 1) * .25, [
                report["mismatch"][year][mode][case]["tail_guard"]["max_abs_terminal_error_mwh"]
                for case in cases], width=.25, label=mode, color=color)
        ax.set_xticks(x, cases, rotation=30, ha="right")
        ax.set(title=year, ylabel="Mismatch max daily terminal error (MWh)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v6")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("One to three bounded CPU workers required.")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        v3, v5 = ROOT / "outputs/fusion_v3", ROOT / "outputs/fusion_v5"
        calibration = read(v5 / "calibration.json")
        if read(v5 / "selection.json")["selected_policy"] != "tail_guard":
            raise ValueError("This audit must hold the frozen v5 selection fixed.")
        panels, samples, input_hashes = {}, {}, {}
        for year in YEARS:
            frames = {}
            for method in METHODS:
                path = v3 / "predictions" / method / f"{year}.csv"
                frames[method] = pd.read_csv(path, parse_dates=["timestamp", "issued_timestamp"])
                input_hashes[path.relative_to(v3).as_posix()] = sha(path)
            panels[year], samples[year] = common_panels(frames, calibration)
        files = [
            Path(__file__), ROOT / "scripts/verify_robustness_results.py",
            ROOT / "scripts/verify_dispatch_results.py", ROOT / "scripts/verify_tail_results.py",
            ROOT / "scripts/run_tail_experiment.py", ROOT / "scripts/run_dispatch_experiment.py",
            *[ROOT / f"src/{name}.py" for name in (
                "delayed_forecasts", "robustness_control", "robustness_admission", "recovery_control",
                "forecast_dispatch", "balanced_storage", "scenarios", "tail_control", "multihorizon_fusion",
                "physics_forecast", "trajectory_audit", "efficient_storage", "optimality", "energy_twin", "numerics")],
            ROOT / "tests/test_robustness.py", ROOT / "experiments/fusion_v6/protocol.md",
            ROOT / "experiments/fusion_v6/run_remote.sh",
        ]
        save(out / "manifest.json", {
            "created_utc": utc_now(), "gpu_used": False, "workers": args.workers,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
            "input_forecast_sha256": input_hashes,
            "data_sha256": {
                f"independent_de_{year}_hourly.csv": sha(ROOT / f"data/processed/independent_de_{year}_hourly.csv")
                for year in YEARS},
            "calibration_sha256": sha(v5 / "calibration.json"),
            "v5_selection_sha256": sha(v5 / "selection.json"),
            "v5_metrics_sha256": sha(v5 / "metrics.json"),
            "modes": {k: {"delay": d, "horizon": h} for k, (d, h) in MODES.items()},
            "error_models": {k: asdict(m) for k, m in error_models().items()},
            "fixed_policy": "tail_guard", "nominal_model": asdict(StorageModel()),
            "boundary": "Inspected years; fixed-request mismatch, not closed-loop parameter adaptation.",
        })
        save(out / "calibration.json", calibration)
        save(out / "samples.json", samples)
        for year in YEARS:
            for method, modes in panels[year].items():
                for mode, panel in modes.items():
                    path = out / "panels" / year / method / f"{mode}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    panel.to_csv(path, index=False)
        print("V6: audit delayed information and wrong-parameter requests before demo integration.", flush=True)
        jobs = [
            ("delay", year, mode, method, "nominal", panel, calibration["methods"][method],
             calibration["gates"], calibration["tail_threshold_mw"], StorageModel(), str(out))
            for year in YEARS for method, modes in panels[year].items() for mode, panel in modes.items()
        ]
        delay = run_jobs(jobs, out, args.workers, "delay")
        replication, hashes = check_replication(out, delay, read(v5 / "metrics.json"), v5)
        save(out / "v5_replication.json", replication)
        save(out / "v5_reference_hashes.json", hashes)
        mismatch = mismatch_probes(out, delay, panels, calibration)
        jobs = [
            ("known", year, mode, "data_seed42", case, panels[year]["data_seed42"][mode],
             calibration["methods"]["data_seed42"], calibration["gates"], calibration["tail_threshold_mw"],
             actual, str(out))
            for year in YEARS for mode in MAIN_MODES for case, actual in error_models().items()
        ]
        known = run_jobs(jobs, out, args.workers, "known")
        decision = demo_admission(delay, mismatch, audit_ok=True)
        save(out / "demo_decision.json", decision)
        report = {
            "schema": "gridpulse-robustness.v6", "status": "complete",
            "delay": delay, "mismatch": mismatch, "known_parameters": known,
            "comparisons": comparisons(out, delay, mismatch, known),
            "demo_decision": decision, "seconds": time.monotonic() - started,
            "boundary": "Default UI decision is conditional on independent local audit. "
                        "No UI changes; no SOC-feedback or automatic parameter identification in mismatch replays.",
        }
        save(out / "metrics.json", report)
        rows = []
        for year, modes in delay.items():
            for mode, methods in modes.items():
                for method, pair in methods.items():
                    for policy, metrics in pair.items():
                        rows.append({"stage": "delay", "year": year, "mode": mode, "method": method,
                                     "case": "nominal", "policy": policy,
                                     **{key: metrics[key] for key in MEASURES},
                                     "terminal_failure_days": metrics["terminal_failure_days"]})
        for stage, groups in (("mismatch", mismatch), ("known", known)):
            for year, modes in groups.items():
                for mode, cases in modes.items():
                    for case, pair in cases.items():
                        for policy, metrics in pair.items():
                            rows.append({"stage": stage, "year": year, "mode": mode, "method": "data_seed42",
                                         "case": case, "policy": policy, **{key: metrics[key] for key in MEASURES},
                                         "terminal_failure_days": metrics["terminal_failure_days"]})
        pd.DataFrame(rows).to_csv(out / "method_table.csv", index=False)
        figures(out, report)
        save(out / "status.json", {
            "status": "complete", "delay_traces": 60, "mismatch_traces": 48, "known_parameter_traces": 48,
            "default_realtime_demo_admitted": decision["default_realtime_demo_admitted"],
        })
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V6_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
