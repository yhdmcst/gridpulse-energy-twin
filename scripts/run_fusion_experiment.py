"""Bounded paired GPU experiment and separate historical replay diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.nn import functional as F

from src.physics_forecast import (
    TARGETS, VALIDATION_START, TEST_START, POWER_SCALE,
    ResidualMLP, accounting_loss, causal_frame, project_accounting,
)
from src.energy_twin import TwinConfig, default_twin_scenarios, simulate_energy_twin
from src.twin_drift import EnergyTwinDriftMonitor


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")


def dataset() -> tuple[dict[str, pd.DataFrame], list[str]]:
    raw = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv")
    original, features = causal_frame(raw)
    frames = {
        "train": original.loc[original.timestamp < VALIDATION_START].copy(),
        "validation": original.loc[
            (original.timestamp >= VALIDATION_START) & (original.timestamp < TEST_START)].copy(),
        "test": original.loc[original.timestamp >= TEST_START].copy(),
    }
    for year in (2021, 2022):
        frame, columns = causal_frame(pd.read_csv(
            ROOT / f"data/processed/independent_de_{year}_hourly.csv"))
        if columns != features:
            raise ValueError("Feature contracts differ.")
        frames[str(year)] = frame.loc[frame.timestamp.dt.year.eq(year)].copy()
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Empty study split.")
    return frames, features


def provenance(args: argparse.Namespace, features: list[str], frames: dict) -> dict:
    import scipy
    import sklearn
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(args).items() if k not in {"output"}},
        "python": platform.python_version(),
        "packages": {"torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "scipy": scipy.__version__},
        "cuda": torch.version.cuda,
        "visible_gpu_indices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "data_sha256": {
            p.name: sha(p) for p in sorted((ROOT / "data/processed").glob("*.csv"))
        },
        "code_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [Path(__file__), ROOT / "src/physics_forecast.py",
                      ROOT / "src/energy_twin.py", ROOT / "src/twin_drift.py",
                      ROOT / "experiments/fusion_v1/protocol.md"]
        },
        "features": features,
        "splits": {
            name: {"rows": len(frame), "start": frame.timestamp.min().isoformat(),
                   "end": frame.timestamp.max().isoformat(),
                   "timestamp_sha256": hashlib.sha256(
                       frame.timestamp.astype(str).str.cat(sep="\n").encode()).hexdigest()}
            for name, frame in frames.items()
        },
        "boundary": "Exploratory known-data study; no new untouched holdout; no PV power labels.",
    }


def prediction_metrics(y: np.ndarray, prediction: np.ndarray, radius: float) -> dict:
    error = y[:, 3] - prediction[:, 3]
    abs_error = np.abs(error)
    relation = prediction[:, 3] - prediction[:, 0] + prediction[:, 1] + prediction[:, 2]
    return {
        "mae_mw": float(abs_error.mean()),
        "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error_mw": float(np.quantile(abs_error, .95)),
        "component_mae_mw": {
            name: float(np.mean(np.abs(y[:, i] - prediction[:, i])))
            for i, name in enumerate(TARGETS[:3])
        },
        "accounting_residual_mae_mw": float(np.abs(relation).mean()),
        "accounting_residual_max_mw": float(np.abs(relation).max()),
        "negative_component_rows": int((prediction[:, :3] < -1e-6).any(axis=1).sum()),
        "interval_radius_mw": float(radius),
        "interval_coverage_pct": float((abs_error <= radius).mean() * 100),
        "interval_score_mw": float((2 * radius + 20 * np.maximum(abs_error - radius, 0)).mean()),
        "rows": len(y),
    }


def write_predictions(out: Path, name: str, frame: pd.DataFrame,
                      predicted: np.ndarray, radius: float) -> dict:
    output = frame[["timestamp", *TARGETS]].copy()
    for i, column in enumerate(TARGETS):
        output[f"prediction_{column}"] = predicted[:, i]
    output["residual_mw"] = frame.net_load_mw.to_numpy() - predicted[:, 3]
    output["radius_mw"] = radius
    output.to_csv(out / f"{name}.csv", index=False)
    return prediction_metrics(frame[list(TARGETS)].to_numpy(float), predicted, radius)


def train(args: argparse.Namespace, out: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU experiment requires a working CUDA allocation.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frames, features = dataset()
    meta = provenance(args, features, frames)
    training = frames["train"]
    mean = training[features].mean().to_numpy()
    scale = training[features].std().to_numpy()
    scale[scale < 1e-8] = 1.
    latest_columns = [f"{column}_lag1" for column in TARGETS]
    tensors = {
        name: (
            torch.as_tensor(((frame[features].to_numpy() - mean) / scale),
                            dtype=torch.float32, device="cuda"),
            torch.as_tensor(frame[latest_columns].to_numpy() / POWER_SCALE,
                            dtype=torch.float32, device="cuda"),
            torch.as_tensor(frame[list(TARGETS)].to_numpy() / POWER_SCALE,
                            dtype=torch.float32, device="cuda"),
        )
        for name, frame in frames.items()
    }
    model = ResidualMLP(len(features)).cuda()
    meta["initial_parameter_sha256"] = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    save(out / "manifest.json", meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x, latest, y = tensors["train"]
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    best, best_epoch, best_state = float("inf"), 0, None
    start = time.monotonic()
    with (out / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(x), generator=rng)
            losses = []
            for indices in order.split(512):
                idx = indices.to("cuda")
                pred = model(x[idx], latest[idx])
                loss = F.mse_loss(pred[:, 3], y[idx, 3]) + .25 * F.mse_loss(pred[:, :3], y[idx, :3])
                if args.physics:
                    loss = loss + .2 * accounting_loss(pred) + .1 * torch.relu(-pred[:, :3]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                vx, vl, vy = tensors["validation"]
                val = float((model(vx, vl)[:, 3] - vy[:, 3]).abs().mean().cpu()) * POWER_SCALE
            if val < best:
                best, best_epoch = val, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "validation_mae_mw": val, "best_epoch": best_epoch}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint.")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "feature_mean": mean.tolist(),
                "feature_scale": scale.tolist(), "features": features,
                "targets": list(TARGETS), "power_scale": POWER_SCALE}, out / "model.pt")
    predictions = {}
    with torch.no_grad():
        for name, (fx, fl, _) in tensors.items():
            predictions[name] = model(fx, fl).cpu().numpy().astype(float) * POWER_SCALE
    report = {"seed": args.seed, "physics": args.physics, "best_epoch": best_epoch,
              "validation_mae_mw": best, "seconds": time.monotonic() - start,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
              "variants": {}}
    for variant in ("raw", "projected"):
        destination = out / variant
        destination.mkdir()
        predicted = {k: v if variant == "raw" else project_accounting(v)
                     for k, v in predictions.items()}
        radius = float(np.quantile(
            np.abs(frames["validation"].net_load_mw.to_numpy() - predicted["validation"][:, 3]), .9))
        report["variants"][variant] = {
            name: write_predictions(destination, name, frame, predicted[name], radius)
            for name, frame in frames.items()
        }
    save(out / "metrics.json", report)
    print(json.dumps({"complete": True, "seed": args.seed, "physics": args.physics,
                      "seconds": report["seconds"]}), flush=True)


def baseline(args: argparse.Namespace, out: Path) -> None:
    frames, features = dataset()
    save(out / "manifest.json", provenance(args, features, frames))
    model = HistGradientBoostingRegressor(
        learning_rate=.08, max_iter=350, max_leaf_nodes=31,
        l2_regularization=1., random_state=20261026,
    )
    model.fit(frames["train"][features], frames["train"].net_load_mw)
    methods = {}
    for method in ("persistence", "seasonal24", "hgb"):
        dest = out / method
        dest.mkdir()
        predictions = {}
        for name, frame in frames.items():
            p = frame[[f"{c}_lag{24 if method == 'seasonal24' else 1}" for c in TARGETS]].to_numpy(float)
            if method == "hgb":
                p[:, 3] = model.predict(frame[features])
            predictions[name] = p
        radius = float(np.quantile(
            np.abs(frames["validation"].net_load_mw.to_numpy() - predictions["validation"][:, 3]), .9))
        methods[method] = {
            name: write_predictions(dest, name, frame, predictions[name], radius)
            for name, frame in frames.items()
        }
        if method == "hgb":
            for m in methods[method].values():
                # HGB predicts net only; auxiliary persistence columns are not a joint model.
                m["joint_accounting_applicable"] = False
        print(method, methods[method]["test"]["mae_mw"], flush=True)
    save(out / "metrics.json", methods)


def diagnostics(args: argparse.Namespace, out: Path) -> None:
    summaries = {}
    for year in (2021, 2022):
        trace = pd.read_csv(
            ROOT / f"outputs/validation/independent_{year}_dynamic_peak_response_timeseries.csv")
        raw = pd.read_csv(ROOT / f"data/processed/independent_de_{year}_hourly.csv")
        for data in (trace, raw):
            data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
        trace = trace.merge(raw[["timestamp", "solar_generation_mw"]],
                            on="timestamp", validate="one_to_one", how="left")
        config = TwinConfig(maximum_soc_mwh=15000.)
        result = {}
        for key, scenario in default_twin_scenarios().items():
            simulation = simulate_energy_twin(trace, config, scenario)
            values = dict(simulation.metrics)
            values["maximum_daily_terminal_deviation_mwh"] = float(
                simulation.trace.loc[simulation.trace.timestamp.dt.hour.eq(23),
                                     "storage_soc_mwh"].sub(15000).abs().max())
            if not values["source_audit_passed"] or not values["physical_replay_passed"]:
                raise RuntimeError("Independent source/replay audit failed.")
            path = out / f"{year}_{key}.csv"
            simulation.trace.to_csv(path, index=False)
            values["trajectory_sha256"] = sha(path)
            result[key] = values
        summaries[str(year)] = result
    save(out / "twin_metrics.json", summaries)

    # Reference is original training, and probes use only already-observed residual windows.
    run = args.output / "data_seed42/raw"
    reference = pd.read_csv(run / "train.csv", parse_dates=["timestamp"])
    monitor = EnergyTwinDriftMonitor()
    monitor.set_baseline(reference[["residual_mw"]])
    rows = []
    for name in ("test", "2021", "2022"):
        frame = pd.read_csv(run / f"{name}.csv", parse_dates=["timestamp"])
        for start in range(0, len(frame) - 167, 168):
            window = frame.iloc[start:start + 168].copy()
            if not pd.DatetimeIndex(window.timestamp).equals(
                    pd.date_range(window.timestamp.iloc[0], periods=168, freq="h")):
                continue
            for kind, bias in (("unmodified", 0.), ("synthetic_bias_3000mw", 3000.)):
                sample = window[["timestamp", "residual_mw"]].copy()
                sample["residual_mw"] += bias
                report = monitor.analyze(sample)
                rows.append({"period": name, "start": sample.timestamp.iloc[0].isoformat(),
                             "kind": kind, "alarm": report.drift_detected,
                             "score": report.score})
    pd.DataFrame(rows).to_csv(out / "drift_windows.csv", index=False)
    rates = pd.DataFrame(rows).groupby(["period", "kind"]).alarm.agg(["mean", "count"]).reset_index()
    save(out / "drift_metrics.json", {
        "window_hours": 168, "records": rates.to_dict(orient="records"),
        "boundary": "Unmodified periods have no true drift labels; alarm rates are not field FPR. "
                    "Injected bias sensitivity is not real fault detection. No controller changes.",
    })
    print("replay_and_drift_complete", flush=True)


def paired_ci(rows: pd.DataFrame) -> dict:
    daily = rows.assign(day=rows.timestamp.dt.floor("D")).groupby("day").delta.mean().to_numpy()
    rng = np.random.default_rng(20261026)
    stats = []
    for _ in range(2000):
        starts = rng.integers(0, len(daily), size=(len(daily) + 6) // 7)
        indices = (starts[:, None] + np.arange(7)) % len(daily)
        stats.append(float(daily[indices.ravel()[:len(daily)]].mean()))
    return {"estimate_mw": float(rows.delta.mean()),
            "daily_block_interval_95_mw": np.quantile(stats, [.025, .975]).tolist(),
            "days": len(daily), "block_days": 7, "replicates": 2000}


def summary(args: argparse.Namespace, out: Path) -> None:
    runs = {}
    for arm in ("data", "physics"):
        for seed in (42, 43, 44):
            path = args.output / f"{arm}_seed{seed}"
            runs[path.name] = json.loads((path / "metrics.json").read_text())
    baselines = json.loads((args.output / "baseline/metrics.json").read_text())
    comparisons = {}
    for period in ("test", "2021", "2022"):
        rows = []
        for seed in (42, 43, 44):
            a = pd.read_csv(args.output / f"data_seed{seed}/raw/{period}.csv", parse_dates=["timestamp"])
            b = pd.read_csv(args.output / f"physics_seed{seed}/raw/{period}.csv", parse_dates=["timestamp"])
            if not a.timestamp.equals(b.timestamp) or not np.allclose(a.net_load_mw, b.net_load_mw):
                raise ValueError("Unmatched evaluation timestamps/targets.")
            rows.append(pd.DataFrame({
                "timestamp": a.timestamp,
                "delta": b.residual_mw.abs() - a.residual_mw.abs(),
            }))
        pooled = pd.concat(rows).groupby("timestamp").delta.mean().reset_index()
        comparisons[period] = paired_ci(pooled)
    save(out / "summary.json", {
        "runs": runs, "baselines": baselines,
        "physics_raw_minus_data_raw_mae": comparisons,
        "boundary": "Exploratory known data; accounting identity, not a new physics law. "
                    "Intervals conditional on observed days and the three fixed trained seed pairs. "
                    "Projection consistency is guaranteed by construction, not evidence of accuracy.",
    })
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["persistence", "hgb", "data_raw", "physics_raw", "physics_projected"]
    periods = ("test", "2021", "2022")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, period in zip(axes, periods):
        vals = [baselines[k][period]["mae_mw"] for k in methods[:2]]
        for arm, variant in (("data", "raw"), ("physics", "raw"), ("physics", "projected")):
            vals.append(float(np.mean([
                runs[f"{arm}_seed{s}"]["variants"][variant][period]["mae_mw"]
                for s in (42, 43, 44)])))
        ax.bar(range(5), vals, color=["#64748b", "#d97706", "#2563eb", "#0f766e", "#9b277d"])
        ax.set_xticks(range(5), methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(period)
        ax.set_ylabel("Net-load MAE (MW)")
    fig.tight_layout()
    fig.savefig(out / "Figure_1.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, color in (("data", "#2563eb"), ("physics", "#0f766e")):
        for seed in (42, 43, 44):
            log = pd.read_json(args.output / f"{arm}_seed{seed}/training.jsonl", lines=True)
            ax.plot(log.epoch, log.validation_mae_mw, color=color, alpha=.55,
                    label=arm if seed == 42 else None)
    ax.set(xlabel="Epoch", ylabel="Validation net-load MAE (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=160)
    plt.close(fig)
    print(json.dumps(comparisons), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["train", "baseline", "diagnostics", "summary"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_v1")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    label = f"{'physics' if args.physics else 'data'}_seed{args.seed}" if args.task == "train" else args.task
    out = args.output / label
    out.mkdir(parents=True, exist_ok=False)
    try:
        {"train": train, "baseline": baseline,
         "diagnostics": diagnostics, "summary": summary}[args.task](args, out)
        save(out / "status.json", {"status": "complete"})
    except Exception:
        save(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
