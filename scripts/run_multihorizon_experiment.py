"""Bounded v3 experiment; retain old artifacts and every new failure."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.nn import functional as F

from src.multihorizon_fusion import (
    HORIZONS, SixHorizonMLP, arrays, build_frame, restrict_split, split_original,
    shared_days, calibrate_six, planning_panel,
)
from src.physics_forecast import TARGETS, POWER_SCALE, accounting_loss, project_accounting
from src.forecast_dispatch import dispatch, replay
from src.optimality import StorageModel
from src.causal_intervals import rolling_radii
from scripts.run_dispatch_experiment import evaluate_job, daily_ci, daily_peaks, sha, save

PERIODS = ("test", "2021", "2022")
SEEDS = (42, 43, 44)


def dataset():
    original, features = build_frame(pd.read_csv(ROOT / "data/processed/net_load_hourly.csv"))
    frames = split_original(original)
    exclusions = {"original_eligible_feature_rows": len(original)}
    for year in (2021, 2022):
        full, columns = build_frame(pd.read_csv(
            ROOT / f"data/processed/independent_de_{year}_hourly.csv"))
        if columns != features:
            raise ValueError("Inconsistent feature schemas.")
        frames[str(year)] = restrict_split(
            full, start=f"{year}-01-01T00:00:00Z", end=f"{year + 1}-01-01T00:00:00Z")
        exclusions[f"{year}_eligible_feature_rows"] = int(full.timestamp.dt.year.eq(year).sum())
    if any(f.empty for f in frames.values()):
        raise ValueError("Every split must have usable six-horizon labels.")
    return frames, features, exclusions


def forecast_metrics(y, prediction):
    metrics = {}
    for h in HORIZONS:
        actual, pred = y[:, h - 1], prediction[:, h - 1]
        error = actual[:, 3] - pred[:, 3]
        relation = pred[:, 3] - pred[:, 0] + pred[:, 1] + pred[:, 2]
        metrics[str(h)] = {
            "rows": len(actual), "mae_mw": float(np.abs(error).mean()),
            "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
            "accounting_residual_mae_mw": float(np.abs(relation).mean()),
            "negative_component_rows": int((pred[:, :3] < -1e-6).any(axis=1).sum()),
        }
    return metrics


def export_predictions(out, method, frames, predictions, joint=True):
    metrics = {}
    for period in ("validation", *PERIODS):
        frame = frames[period]
        prediction = predictions[period]
        y = frame[[f"target_h{h}_{c}" for h in HORIZONS for c in TARGETS]].to_numpy().reshape(-1, 6, 4)
        metrics[period] = {
            "raw": forecast_metrics(y, prediction), "joint_accounting_applicable": joint,
        }
        if joint:
            projected = project_accounting(prediction.reshape(-1, 4)).reshape(-1, 6, 4)
            metrics[period]["projected"] = forecast_metrics(y, projected)
        output = frame[["timestamp", "issued_timestamp"]].copy()
        for h in HORIZONS:
            output[f"actual_h{h}_mw"] = y[:, h - 1, 3]
            output[f"prediction_h{h}_mw"] = prediction[:, h - 1, 3]
            for i, c in enumerate(TARGETS):
                output[f"predicted_h{h}_{c}"] = prediction[:, h - 1, i]
        path = out / "predictions" / method / f"{period}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
    save(out / "forecasts" / f"{method}.json", metrics)
    return metrics


def train_neural(out, frames, features, seed, physics):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    method = f"{'physics' if physics else 'data'}_seed{seed}"
    dest = out / "models" / method
    dest.mkdir(parents=True)
    mean = frames["train"][features].mean().to_numpy()
    scale = frames["train"][features].std().to_numpy()
    scale[scale < 1e-8] = 1.
    tensors = {}
    for name, frame in frames.items():
        x, latest, y = arrays(frame, features, mean, scale)
        tensors[name] = (
            torch.tensor(x, dtype=torch.float32, device="cuda"),
            torch.tensor(latest / POWER_SCALE, dtype=torch.float32, device="cuda"),
            torch.tensor(y / POWER_SCALE, dtype=torch.float32, device="cuda"),
        )
    model = SixHorizonMLP(len(features)).cuda()
    initial_hash = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch = float("inf"), None, 0
    started = time.monotonic()
    history = []
    x, latest, y = tensors["train"]
    for epoch in range(1, 26):
        model.train()
        losses = []
        for index in torch.randperm(len(x), generator=order_rng).split(512):
            index = index.cuda()
            pred = model(x[index], latest[index])
            loss = F.mse_loss(pred[:, :, 3], y[index, :, 3]) + .25 * F.mse_loss(
                pred[:, :, :3], y[index, :, :3])
            if physics:
                loss = loss + .2 * accounting_loss(pred.reshape(-1, 4))
                loss = loss + .1 * torch.relu(-pred[:, :, :3]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vx, vl, vy = tensors["validation"]
            val = float((model(vx, vl)[:, :, 3] - vy[:, :, 3]).abs().mean().cpu()) * POWER_SCALE
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_mean_six_horizon_mae_mw": val, "best_epoch": best_epoch})
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint.")
    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state, "features": features, "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(), "power_scale": POWER_SCALE,
        "targets": list(TARGETS), "horizons": list(HORIZONS),
    }, dest / "model.pt")
    predictions = {}
    with torch.no_grad():
        for period in ("validation", *PERIODS):
            fx, fl, _ = tensors[period]
            predictions[period] = model(fx, fl).cpu().numpy().astype(float) * POWER_SCALE
    metrics = export_predictions(out, method, frames, predictions)
    save(dest / "training.json", history)
    metadata = {
        "seed": seed, "physics": physics, "initial_parameter_sha256": initial_hash,
        "best_epoch": best_epoch, "validation_mean_six_horizon_mae_mw": best,
        "seconds": time.monotonic() - started,
        "peak_cuda_allocation_mb": torch.cuda.max_memory_allocated() / 2**20,
        "model_sha256": sha(dest / "model.pt"),
    }
    save(dest / "manifest.json", metadata)
    # Reconstruct from the delivered checkpoint on CPU, not the in-memory GPU model.
    checkpoint = torch.load(dest / "model.pt", map_location="cpu", weights_only=False)
    cpu = SixHorizonMLP(len(features))
    cpu.load_state_dict(checkpoint["state_dict"])
    cpu.eval()
    cx, cl, _ = arrays(frames["test"], features,
                       np.asarray(checkpoint["feature_mean"]), np.asarray(checkpoint["feature_scale"]))
    with torch.no_grad():
        reproduced = cpu(torch.tensor(cx).float(), torch.tensor(cl / POWER_SCALE).float()).numpy()
    difference = float(np.max(np.abs(reproduced.astype(float) * POWER_SCALE - predictions["test"])))
    if difference > .1:
        raise RuntimeError("CPU checkpoint inference did not reproduce GPU predictions.")
    save(dest / "cpu_reload_check.json", {"passed": True, "max_absolute_difference_mw": difference})
    print(json.dumps({"trained": method, **metadata, "cpu_reload_max_error_mw": difference}), flush=True)
    del model, tensors
    torch.cuda.empty_cache()
    return method, metrics


def baselines(out, frames, features):
    models = []
    for h in HORIZONS:
        model = HistGradientBoostingRegressor(
            learning_rate=.08, max_iter=350, max_leaf_nodes=31,
            l2_regularization=1., random_state=20261026)
        model.fit(frames["train"][features], frames["train"][f"target_h{h}_net_load_mw"])
        models.append(model)
    path = out / "models/hgb_six.joblib"
    joblib.dump({"models": models, "features": features}, path)
    reports = {}
    for method in ("persistence", "hgb"):
        predictions = {}
        for period in ("validation", *PERIODS):
            frame = frames[period]
            latest = frame[[f"{c}_lag1" for c in TARGETS]].to_numpy(float)
            pred = np.repeat(latest[:, None, :], 6, axis=1)
            if method == "hgb":
                for h, model in enumerate(models):
                    pred[:, h, 3] = model.predict(frame[features])
            predictions[period] = pred
        reports[method] = export_predictions(out, method, frames, predictions, joint=method != "hgb")
        print(json.dumps({"baseline": method, "test_h1_mae_mw":
                         reports[method]["test"]["raw"]["1"]["mae_mw"]}), flush=True)
    return reports


def comparisons(results, out):
    measures = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
                "throughput_mwh", "conversion_loss_mwh", "recovery_charge_mwh")
    output = {}
    for period in PERIODS:
        methods = results[period]
        pairs = {f"{name}_w6_minus_w1": [(f"{name}_w6", f"{name}_w1")]
                 for name in ("persistence", "hgb", *(f"{a}_seed{s}" for a in ("data", "physics") for s in SEEDS))}
        for window in (1, 6):
            pairs[f"physics_minus_data_w{window}"] = [
                (f"physics_seed{s}_w{window}", f"data_seed{s}_w{window}") for s in SEEDS]
            pairs[f"data_minus_hgb_w{window}"] = [
                (f"data_seed{s}_w{window}", f"hgb_w{window}") for s in SEEDS]
        output[period] = {}
        for label, matched in pairs.items():
            deltas, daily = [], []
            for left, right in matched:
                a, b = methods[left], methods[right]
                deltas.append({key: a[key] - b[key] for key in measures})
                pa, pb = daily_peaks(out, a), daily_peaks(out, b)
                if not pa.index.equals(pb.index):
                    raise ValueError("Paired daily timestamps differ.")
                daily.append((pa - pb).to_numpy())
            output[period][label] = {
                "pairs": matched, "pair_differences": deltas,
                "mean_differences": {key: float(np.mean([d[key] for d in deltas])) for key in measures},
                "conditional_daily_peak_interval": daily_ci(np.mean(daily, axis=0)),
            }
    return output


def stress_experiments(out, frames, calibration, results):
    method = "data_seed42"
    models = {
        "power_70pct": replace(StorageModel(), charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_085": replace(StorageModel(), efficiency=.85),
    }
    output = {}
    for period in ("2021", "2022"):
        nominal = pd.read_csv(out / results[period][f"{method}_w6"]["trajectory"], parse_dates=["timestamp"])
        panel = planning_panel(frames[period][method], calibration["methods"][method], 6)
        output[period] = {}
        for name, model in models.items():
            rtrace, rmetrics = replay(nominal, model)
            ptrace, pmetrics = dispatch(panel, calibration["methods"][method], calibration["gates"], model)
            if not rtrace.timestamp.equals(ptrace.timestamp):
                raise ValueError("Stress pair timestamps differ.")
            output[period][name] = {}
            for arm, trace, metrics in (("replay", rtrace, rmetrics), ("replan", ptrace, pmetrics)):
                path = out / "stress" / f"{period}_{name}_{arm}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                trace.to_csv(path, index=False)
                metrics["trajectory"] = path.relative_to(out).as_posix()
                metrics["trajectory_sha256"] = sha(path)
                output[period][name][arm] = metrics
            print(json.dumps({"stress": name, "period": period,
                              "replay_terminal_error_mwh": rmetrics["max_abs_terminal_error_mwh"],
                              "replan_terminal_error_mwh": pmetrics["max_abs_terminal_error_mwh"]}), flush=True)
    return output


def figures(out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, period in zip(axes, PERIODS):
        for window, offset, color in ((1, -.18, "#2474b5"), (6, .18, "#18836b")):
            families = ("hgb", "data", "physics")
            values = []
            for family in families:
                names = [f"hgb_w{window}"] if family == "hgb" else [f"{family}_seed{s}_w{window}" for s in SEEDS]
                values.append(np.mean([report["methods"][period][n]["mean_daily_peak_mw"] for n in names]))
            ax.bar(np.arange(3) + offset, values, width=.36, color=color, label=f"{window}h")
        ax.set_xticks(range(3), families)
        ax.set(title=period, ylabel="Mean daily controlled peak (MW)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, period in zip(axes, PERIODS):
        groups = report["methods"][period]
        names = [f"data_seed42_w{w}" for w in (1, 6)]
        values = [groups[n]["controlled_peak_mw"] - groups[n]["baseline_peak_mw"] for n in names]
        ax.bar(["1h", "6h"], values, color=["#2474b5", "#18836b"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{period}: fixed seed42", ylabel="Maximum minus no-storage peak (MW)")
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v3")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("One to three CPU workers required.")
    if not torch.cuda.is_available():
        raise RuntimeError("One inspected free CUDA GPU is required.")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    started = time.monotonic()
    try:
        frames, features, exclusions = dataset()
        paths = [
            Path(__file__), ROOT / "scripts/run_dispatch_experiment.py",
            *[ROOT / f"src/{p}.py" for p in (
                "multihorizon_fusion", "physics_forecast", "causal_intervals",
                "forecast_dispatch", "balanced_storage", "scenarios", "energy_twin",
                "trajectory_audit", "efficient_storage", "optimality", "numerics")],
            ROOT / "experiments/fusion_v3/protocol.md",
            ROOT / "experiments/fusion_v3/run_remote.sh",
            ROOT / "scripts/verify_multihorizon_results.py",
        ]
        save(out / "manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "torch": torch.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "visible_gpu_indices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "data_sha256": {p.name: sha(p) for p in sorted((ROOT / "data/processed").glob("*.csv"))},
            "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            "features": features, "workers": args.workers, "physical_model": asdict(StorageModel()),
            "boundary": "Exploratory known years, idealized immediate hourly observation availability.",
        })
        save(out / "splits.json", {
            "feature_eligibility": exclusions,
            "splits": {name: {
                "rows": len(frame), "first_target": frame.timestamp.min().isoformat(),
                "last_first_target": frame.timestamp.max().isoformat(),
                "last_target": frame.last_target_timestamp.max().isoformat(),
                "timestamp_sha256": hashlib.sha256(frame.timestamp.astype(str).str.cat(sep="\n").encode()).hexdigest(),
            } for name, frame in frames.items()},
        })
        print("V3: test whether true six-hour planning fixes recovery peaks, retaining all costs.", flush=True)
        for seed in SEEDS:
            for physics in (False, True):
                train_neural(out, frames, features, seed, physics)
        baselines(out, frames, features)
        inputs, samples = {}, {}
        names = ("persistence", "hgb", *(f"{a}_seed{s}" for a in ("data", "physics") for s in SEEDS))
        for period in ("validation", *PERIODS):
            raw = {name: pd.read_csv(out / "predictions" / name / f"{period}.csv",
                                     parse_dates=["timestamp", "issued_timestamp"]) for name in names}
            inputs[period], samples[period] = shared_days(raw)
        calibration = calibrate_six(inputs["validation"])
        save(out / "samples.json", samples)
        save(out / "calibration.json", calibration)
        jobs = [
            (period, f"{name}_w{window}",
             planning_panel(inputs[period][name], calibration["methods"][name], window),
             calibration["methods"][name], calibration["gates"], str(out))
            for period in PERIODS for name in names for window in (1, 6)
        ]
        results, failures = {p: {} for p in PERIODS}, []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(evaluate_job, job): (job[0], job[1]) for job in jobs}
            for future in as_completed(futures):
                try:
                    period, method, metrics = future.result()
                    results[period][method] = metrics
                    print(json.dumps({"dispatch": method, "period": period,
                                      "peak_mw": metrics["controlled_peak_mw"]}), flush=True)
                except Exception as exc:
                    failures.append({"job": list(futures[future]), "error": str(exc)})
        save(out / "failures.json", failures)
        if failures:
            raise RuntimeError("Failed dispatch jobs preserved; no complete comparison allowed.")
        stress = stress_experiments(out, inputs, calibration, results)
        report = {
            "schema": "gridpulse-fusion-multihorizon.v3", "status": "complete",
            "methods": results, "stress": stress, "comparisons": comparisons(results, out),
            "seconds": time.monotonic() - started,
            "boundary": "Six-hour reserve sees future bands; not pure LP-lookahead isolation. "
                        "Known parameter changes in stress, no fault inference or field guarantee.",
        }
        save(out / "metrics.json", report)
        table = [{"period": p, "method": n, **{k: m[k] for k in (
            "forecast_h1_mae_mw", "controlled_peak_mw", "controlled_p95_mw",
            "mean_daily_peak_mw", "throughput_mwh", "conversion_loss_mwh",
            "recovery_charge_mwh", "max_abs_terminal_error_mwh", "physical_audit_passed")}}
                 for p, methods in results.items() for n, m in methods.items()]
        pd.DataFrame(table).to_csv(out / "method_table.csv", index=False)
        # Corrected helper must reproduce the already audited v1 intervals under this runtime.
        portability = {}
        for period in PERIODS:
            old = pd.read_csv(ROOT / f"outputs/fusion_v1/followup/interval_{period}.csv",
                              parse_dates=["timestamp"])
            radii = rolling_radii(old, float(old.radius_mw.iloc[0]))
            difference = float(np.max(np.abs(radii - old.rolling_radius_mw.to_numpy())))
            if difference > 1e-6:
                raise RuntimeError("Timestamp-safe rolling helper differs from frozen audited output.")
            portability[period] = {"max_difference_mw": difference, "rows": len(old)}
        save(out / "interval_portability.json", portability)
        figures(out, report)
        save(out / "status.json", {"status": "complete", "nominal_traces": 48, "stress_traces": 8})
        save(out / "ARTIFACT_SHA256.json", {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "ARTIFACT_SHA256.json"
        })
        print("FUSION_V3_COMPLETE", flush=True)
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
