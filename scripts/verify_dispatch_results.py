"""Independent NumPy audit of v2 trajectory, calibration and paired contrasts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/fusion_v2"
SOURCE = ROOT / "outputs/fusion_v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_trace(path: Path, metrics: dict, *, terminal: bool, continuous: bool) -> dict:
    trace = pd.read_csv(path, parse_dates=["timestamp"])
    model = metrics["physical_parameters"]
    eta = model["efficiency"]
    required = ["net_load_mw", "controlled_net_load_mw", "charge_mw", "discharge_mw",
                "soc_before_mwh", "storage_soc_mwh"]
    assert not trace.empty and np.isfinite(trace[required].to_numpy()).all()
    timestamps = pd.DatetimeIndex(trace.timestamp)
    assert not timestamps.has_duplicates and timestamps.is_monotonic_increasing
    for day, group in trace.groupby(trace.timestamp.dt.floor("D")):
        assert pd.DatetimeIndex(group.timestamp).equals(pd.date_range(day, periods=24, freq="h"))
    c, d = trace.charge_mw.to_numpy(), trace.discharge_mw.to_numpy()
    before, after = trace.soc_before_mwh.to_numpy(), trace.storage_soc_mwh.to_numpy()
    assert c.min() >= -1e-6 and d.min() >= -1e-6
    assert c.max() <= model["charge_power_mw"] + 1e-6
    assert d.max() <= model["discharge_power_mw"] + 1e-6
    assert np.minimum(c, d).max() < 1e-6
    assert min(before.min(), after.min()) >= model["minimum_soc_mwh"] - 1e-6
    assert max(before.max(), after.max()) <= model["maximum_soc_mwh"] + 1e-6
    assert np.max(np.abs(after - before - eta * c + d / eta)) < 1e-5
    controlled = trace.net_load_mw.to_numpy() + c - d
    assert np.allclose(trace.controlled_net_load_mw, controlled, rtol=0, atol=1e-6)
    adjacent = timestamps[1:] - timestamps[:-1] == pd.Timedelta(hours=1)
    if continuous:
        assert adjacent.all()
    assert np.max(np.abs((before[1:] - after[:-1])[adjacent])) < 1e-5
    last = trace.timestamp.dt.hour.eq(23)
    terminal_error = float(np.max(np.abs(after[last] - model["initial_soc_mwh"])))
    if terminal:
        assert terminal_error < 1e-5
        assert np.allclose(before[trace.timestamp.dt.hour.eq(0)], model["initial_soc_mwh"], atol=1e-5)
    loss = float(((1 - eta) * c + (1 / eta - 1) * d).sum())
    measures = {
        "baseline_peak_mw": float(trace.net_load_mw.max()),
        "controlled_peak_mw": float(controlled.max()),
        "controlled_p95_mw": float(np.quantile(controlled, .95)),
        "mean_daily_peak_mw": float(
            pd.Series(controlled, index=timestamps).groupby(timestamps.floor("D")).max().mean()),
        "throughput_mwh": float((c + d).sum()),
        "conversion_loss_mwh": loss,
        "max_abs_terminal_error_mwh": terminal_error,
    }
    for key, value in measures.items():
        assert abs(value - metrics[key]) < 1e-5, (path, key, value, metrics[key])
    assert metrics["terminal_requirement_passed"] == (terminal_error <= 1e-5)
    assert sha(path) == metrics["trajectory_sha256"]
    return {"frame": trace, "metrics": measures}


def main():
    hashes = read(OUTPUT / "ARTIFACT_SHA256.json")
    for relative, value in hashes.items():
        path = (OUTPUT / relative).resolve()
        assert path.is_relative_to(OUTPUT.resolve()) and sha(path) == value
    manifest = read(OUTPUT / "manifest.json")
    for relative, value in manifest["code_sha256"].items():
        assert sha(ROOT / relative) == value, relative
    for relative, value in manifest["input_sha256"].items():
        assert sha(SOURCE / relative) == value, relative
    report = read(OUTPUT / "metrics.json")
    calibration = read(OUTPUT / "calibration.json")
    sample = read(OUTPUT / "samples.json")
    assert read(OUTPUT / "failures.json") == []
    method_paths = {method: f"baseline/{method}" for method in ("persistence", "seasonal24", "hgb")}
    method_paths.update({
        f"{arm}_{variant}_seed{seed}": f"{arm}_seed{seed}/{variant}"
        for arm in ("data", "physics") for variant in ("raw", "projected") for seed in (42, 43, 44)
    })
    reference_validation = None
    for method, relative in method_paths.items():
        frame = pd.read_csv(SOURCE / relative / "validation.csv", parse_dates=["timestamp"])
        grouped = frame.groupby(frame.timestamp.dt.floor("D"))
        keep = [g for _, g in grouped if len(g) == 24]
        frame = pd.concat(keep).reset_index(drop=True)
        assert frame.timestamp.max() < pd.Timestamp("2019-08-09T21:00:00Z")
        if reference_validation is None:
            reference_validation = frame
        else:
            assert frame.timestamp.equals(reference_validation.timestamp)
        errors = (frame.net_load_mw - frame.prediction_net_load_mw).abs()
        actual = errors.groupby(frame.timestamp.dt.hour).quantile(.9)
        for hour, radius in actual.items():
            assert abs(radius - calibration["methods"][method]["hourly_radius_mw"][str(hour)]) < 1e-6
    for gate, quantile in (("peak_threshold_mw", .95), ("charge_threshold_mw", .25),
                           ("release_threshold_mw", .99)):
        assert abs(calibration["gates"][gate] - reference_validation.net_load_mw.quantile(quantile)) < 1e-6
    traces, count = {}, 0
    for period, methods in report["methods"].items():
        traces[period] = {}
        reference = None
        for method, metrics in methods.items():
            item = check_trace(OUTPUT / metrics["trajectory"], metrics, terminal=True, continuous=False)
            frame = item["frame"]
            if reference is None:
                reference = frame
            else:
                assert frame.timestamp.equals(reference.timestamp)
                assert np.allclose(frame.net_load_mw, reference.net_load_mw)
            inputs = pd.read_csv(
                SOURCE / method_paths[method] / f"{period}.csv", parse_dates=["timestamp"]).set_index("timestamp")
            aligned = inputs.loc[frame.timestamp]
            assert np.allclose(frame.net_load_mw, aligned.net_load_mw)
            assert np.allclose(frame.prediction_h1_mw, aligned.prediction_net_load_mw)
            radii = frame.timestamp.dt.hour.map({
                int(k): v for k, v in calibration["methods"][method]["hourly_radius_mw"].items()})
            assert np.allclose(frame.upper_h1_mw, frame.prediction_h1_mw + radii)
            mae = np.mean(np.abs(frame.net_load_mw.to_numpy() - aligned.prediction_net_load_mw.to_numpy()))
            assert abs(mae - metrics["forecast_h1_mae_mw"]) < 1e-6
            assert len(frame) == sample[period]["rows"]
            traces[period][method] = frame
            count += 1
    stress_count = 0
    for period, scenarios in report["stress"].items():
        for scenario, arms in scenarios.items():
            pair = {}
            for arm, metrics in arms.items():
                pair[arm] = check_trace(
                    OUTPUT / metrics["trajectory"], metrics,
                    terminal=(arm == "replan"), continuous=True)["frame"]
                stress_count += 1
            assert pair["replay"].timestamp.equals(pair["replan"].timestamp)
            assert np.allclose(pair["replay"].net_load_mw, pair["replan"].net_load_mw)
            assert arms["replay"]["physical_parameters"] == arms["replan"]["physical_parameters"]

    families = (
        ("data_raw_minus_hgb", "data_raw", "hgb"),
        ("data_projected_minus_data_raw", "data_projected", "data_raw"),
        ("physics_raw_minus_data_raw", "physics_raw", "data_raw"),
        ("physics_projected_minus_data_projected", "physics_projected", "data_projected"),
    )
    for period, contrasts in report["comparisons"].items():
        for label, left, right in families:
            for key, estimate in contrasts[label]["mean_differences"].items():
                values = [
                    report["methods"][period][f"{left}_seed{seed}"][key]
                    - report["methods"][period][right if right == "hgb" else f"{right}_seed{seed}"][key]
                    for seed in (42, 43, 44)
                ]
                assert abs(np.mean(values) - estimate) < 1e-5
            daily = []
            for seed in (42, 43, 44):
                a = traces[period][f"{left}_seed{seed}"]
                b = traces[period][right if right == "hgb" else f"{right}_seed{seed}"]
                daily.append((
                    a.groupby(a.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                    - b.groupby(b.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                ).to_numpy())
            pooled = np.mean(daily, axis=0)
            rng = np.random.default_rng(20261026)
            estimates = []
            for _ in range(2000):
                starts = rng.integers(0, len(pooled), size=(len(pooled) + 6) // 7)
                idx = ((starts[:, None] + np.arange(7)) % len(pooled)).ravel()[:len(pooled)]
                estimates.append(pooled[idx].mean())
            interval = contrasts[label]["conditional_mean_daily_peak_interval"]
            assert abs(pooled.mean() - interval["estimate_mw"]) < 1e-5
            assert np.allclose(np.quantile(estimates, [.025, .975]),
                               [interval["lower_95_mw"], interval["upper_95_mw"]], atol=1e-5)
    from PIL import Image, ImageStat
    for name in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUTPUT / name) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    verification = {
        "passed": True, "hashes_verified": len(hashes),
        "input_prediction_files_verified": len(manifest["input_sha256"]),
        "nominal_trajectories_recomputed": count,
        "paired_stress_trajectories_recomputed": stress_count,
        "validation_calibrations_recomputed": len(method_paths),
        "paired_block_intervals_recomputed": 12,
        "figures_pixel_checked": 2,
        "boundary": "Independent arithmetic, calibration and conservation audit; no field-safety claim.",
    }
    (ROOT / "outputs/fusion_v2_local_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
