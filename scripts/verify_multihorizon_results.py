"""Independent local v3 CSV/hash/physics audit, without Torch or SciPy."""
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
from src.causal_intervals import rolling_radii

OUT = ROOT / "outputs/fusion_v3"
TARGETS = ("total_load_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes = read(OUT / "ARTIFACT_SHA256.json")
    for relative, expected in hashes.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and sha(path) == expected, relative
    manifest = read(OUT / "manifest.json")
    for relative, expected in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for name, expected in manifest["data_sha256"].items():
        assert sha(ROOT / "data/processed" / name) == expected, name
    assert read(OUT / "status.json")["status"] == "complete"
    assert read(OUT / "failures.json") == []
    split = read(OUT / "splits.json")["splits"]
    assert pd.Timestamp(split["train"]["last_target"]) < pd.Timestamp("2019-01-12T20:00Z")
    assert pd.Timestamp(split["validation"]["first_target"]) >= pd.Timestamp("2019-01-12T20:00Z")
    assert pd.Timestamp(split["validation"]["last_target"]) < pd.Timestamp("2019-08-09T21:00Z")
    for seed in (42, 43, 44):
        a = read(OUT / f"models/data_seed{seed}/manifest.json")
        b = read(OUT / f"models/physics_seed{seed}/manifest.json")
        assert a["initial_parameter_sha256"] == b["initial_parameter_sha256"]
        for arm in ("data", "physics"):
            dest = OUT / f"models/{arm}_seed{seed}"
            history = read(dest / "training.json")
            meta = read(dest / "manifest.json")
            best = min(history, key=lambda r: r["validation_mean_six_horizon_mae_mw"])
            assert len(history) == 25 and meta["best_epoch"] == best["epoch"]
            assert sha(dest / "model.pt") == meta["model_sha256"]
            assert read(dest / "cpu_reload_check.json")["max_absolute_difference_mw"] < .1
    sources = {}
    for period in ("validation", "test", "2021", "2022"):
        name = f"independent_de_{period}_hourly.csv" if period in ("2021", "2022") else "net_load_hourly.csv"
        source = pd.read_csv(ROOT / "data/processed" / name, parse_dates=["timestamp"])
        sources[period] = source.set_index("timestamp")
    predictions, metric_sets = {}, 0
    names = ("persistence", "hgb", *(f"{a}_seed{s}" for a in ("data", "physics") for s in (42, 43, 44)))
    for name in names:
        metadata = read(OUT / "forecasts" / f"{name}.json")
        predictions[name] = {}
        for period in ("validation", "test", "2021", "2022"):
            frame = pd.read_csv(OUT / "predictions" / name / f"{period}.csv",
                                parse_dates=["timestamp", "issued_timestamp"])
            times = pd.DatetimeIndex(frame.timestamp)
            assert times.is_monotonic_increasing and not times.has_duplicates
            assert pd.DatetimeIndex(frame.issued_timestamp).equals(times - pd.Timedelta(hours=1))
            assert len(frame) == split[period]["rows"]
            assert hashlib.sha256(frame.timestamp.astype(str).str.cat(sep="\n").encode()).hexdigest() == split[period]["timestamp_sha256"]
            if period == "validation":
                assert times.min() >= pd.Timestamp("2019-01-12T20:00Z")
                assert times.max() + pd.Timedelta(hours=5) < pd.Timestamp("2019-08-09T21:00Z")
            if period in ("2021", "2022"):
                assert (times.year == int(period)).all()
                assert (times + pd.Timedelta(hours=5)).max().year == int(period)
            for h in range(1, 7):
                truth = sources[period].reindex(times + pd.Timedelta(hours=h - 1))[list(TARGETS)].to_numpy(float)
                pred = frame[[f"predicted_h{h}_{c}" for c in TARGETS]].to_numpy(float)
                assert np.isfinite(truth).all() and np.isfinite(pred).all()
                assert np.allclose(frame[f"actual_h{h}_mw"], truth[:, 3], rtol=0, atol=1e-6)
                assert np.allclose(frame[f"prediction_h{h}_mw"], pred[:, 3], rtol=0, atol=1e-6)
                for variant in ("raw", "projected") if "projected" in metadata[period] else ("raw",):
                    p = pred.copy()
                    if variant == "projected":
                        error = p[:, 3] - p[:, 0] + p[:, 1] + p[:, 2]
                        p -= error[:, None] * np.array([-1., 1., 1., 1.])[None, :] / 4
                        p[:, :3] = np.maximum(p[:, :3], 0.)
                        p[:, 3] = p[:, 0] - p[:, 1] - p[:, 2]
                    error = truth[:, 3] - p[:, 3]
                    m = metadata[period][variant][str(h)]
                    assert abs(np.abs(error).mean() - m["mae_mw"]) < 1e-5
                    assert abs(np.sqrt(np.mean(error ** 2)) - m["rmse_mw"]) < 1e-5
                    accounting = p[:, 3] - p[:, 0] + p[:, 1] + p[:, 2]
                    assert abs(np.abs(accounting).mean() - m["accounting_residual_mae_mw"]) < 1e-5
                    assert int((p[:, :3] < -1e-6).any(axis=1).sum()) == m["negative_component_rows"]
                    metric_sets += 1
            predictions[name][period] = frame

    samples = read(OUT / "samples.json")
    calibration = read(OUT / "calibration.json")
    for name in names:
        frame = predictions[name]["validation"]
        start, end = pd.Timestamp(samples["validation"]["start"]), pd.Timestamp(samples["validation"]["end"])
        frame = frame.loc[frame.timestamp.between(start, end)]
        assert len(frame) == samples["validation"]["rows"]
        method = calibration["methods"][name]
        radii = []
        for h in range(1, 7):
            hours = (frame.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
            error = np.abs(frame[f"actual_h{h}_mw"] - frame[f"prediction_h{h}_mw"])
            q = error.groupby(hours).quantile(.9)
            counts = error.groupby(hours).size()
            for hour, value in q.items():
                assert abs(value - method["radii"][str(h)][str(hour)]) < 1e-5
                assert counts[hour] == method["counts"][str(h)][str(hour)]
                radii.append(value)
        assert abs(max(1., 1.1 * max(radii)) - method["uncertainty_scale_mw"]) < 1e-5
        for gate, q in (("peak_threshold_mw", .95), ("charge_threshold_mw", .25),
                        ("release_threshold_mw", .99)):
            assert abs(frame.actual_h1_mw.quantile(q) - calibration["gates"][gate]) < 1e-5
    report = read(OUT / "metrics.json")
    traces = {}
    nominal_count, stress_count = 0, 0
    for period, methods in report["methods"].items():
        traces[period] = {}
        reference = None
        for name, metrics in methods.items():
            trace = check_trace(OUT / metrics["trajectory"], metrics,
                                terminal=True, continuous=False)["frame"]
            assert np.isclose(trace.soc_before_mwh.iloc[0], 15000.)
            method, window = name.rsplit("_w", 1)
            forecast = predictions[method][period].set_index("timestamp").loc[trace.timestamp]
            assert np.allclose(trace.net_load_mw, forecast.actual_h1_mw)
            assert np.allclose(trace.prediction_h1_mw, forecast.prediction_h1_mw)
            assert len(trace) == samples[period]["rows"]
            if reference is None:
                reference = trace
            else:
                assert trace.timestamp.equals(reference.timestamp)
                assert np.allclose(trace.net_load_mw, reference.net_load_mw)
            radius = trace.timestamp.dt.hour.map({
                int(k): v for k, v in calibration["methods"][method]["radii"]["1"].items()})
            assert np.allclose(trace.upper_h1_mw, trace.prediction_h1_mw + radius)
            mae = np.abs(trace.net_load_mw - trace.prediction_h1_mw).mean()
            assert abs(mae - metrics["forecast_h1_mae_mw"]) < 1e-5
            recovery = np.where(trace.prediction_h1_mw - radius > calibration["gates"]["charge_threshold_mw"],
                                trace.charge_mw, 0.).sum()
            assert abs(recovery - metrics["recovery_charge_mwh"]) < 1e-5
            traces[period][name] = trace
            nominal_count += 1
        for method in names:
            a, b = traces[period][f"{method}_w1"], traces[period][f"{method}_w6"]
            assert np.allclose(a.prediction_h1_mw, b.prediction_h1_mw)
            assert np.allclose(a.upper_h1_mw, b.upper_h1_mw)
    for period, scenarios in report["stress"].items():
        for name, arms in scenarios.items():
            pair = {}
            for arm, metrics in arms.items():
                pair[arm] = check_trace(OUT / metrics["trajectory"], metrics,
                                        terminal=arm == "replan", continuous=True)["frame"]
                assert np.isclose(pair[arm].soc_before_mwh.iloc[0], 15000.)
                stress_count += 1
            assert pair["replay"].timestamp.equals(pair["replan"].timestamp)
            assert np.allclose(pair["replay"].net_load_mw, pair["replan"].net_load_mw)
            assert arms["replay"]["physical_parameters"] == arms["replan"]["physical_parameters"]
    contrast_count = 0
    for period, contrasts in report["comparisons"].items():
        for name, comparison in contrasts.items():
            delta = []
            for left, right in comparison["pairs"]:
                a, b = traces[period][left], traces[period][right]
                pa = a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                pb = b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                assert pa.index.equals(pb.index)
                delta.append((pa - pb).to_numpy())
            for key, estimate in comparison["mean_differences"].items():
                values = [report["methods"][period][a][key] - report["methods"][period][b][key]
                          for a, b in comparison["pairs"]]
                assert abs(np.mean(values) - estimate) < 1e-5
            daily = np.mean(delta, axis=0)
            rng = np.random.default_rng(20261026)
            boot = []
            for _ in range(2000):
                starts = rng.integers(0, len(daily), size=(len(daily) + 6) // 7)
                indices = ((starts[:, None] + np.arange(7)) % len(daily)).ravel()[:len(daily)]
                boot.append(daily[indices].mean())
            interval = comparison["conditional_daily_peak_interval"]
            assert abs(daily.mean() - interval["estimate_mw"]) < 1e-5
            assert np.allclose(np.quantile(boot, [.025, .975]),
                               [interval["lower_95_mw"], interval["upper_95_mw"]], atol=1e-5)
            contrast_count += 1
    for period in ("test", "2021", "2022"):
        frame = pd.read_csv(ROOT / f"outputs/fusion_v1/followup/interval_{period}.csv",
                            parse_dates=["timestamp"])
        fixed = rolling_radii(frame, frame.radius_mw.iloc[0])
        assert np.allclose(fixed, frame.rolling_radius_mw, rtol=0, atol=1e-6)
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    receipt = {
        "passed": True, "artifact_hashes": len(hashes), "forecast_files": len(names) * 4,
        "forecast_horizon_metric_sets": metric_sets, "validation_methods": len(names),
        "nominal_trajectories": nominal_count, "stress_trajectories": stress_count,
        "paired_block_intervals": contrast_count, "matched_seed_pairs": 3,
        "checkpoint_reload_receipts": 6, "portable_interval_periods": 3,
        "boundary": "Independent CSV arithmetic and provenance audit, not field validation.",
    }
    (ROOT / "outputs/fusion_v3_local_verification.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
