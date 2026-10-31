"""Independent CPU/NumPy audit of delivered artifacts; requires no torch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metric_file(path: Path, metrics: dict) -> None:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    assert len(frame) == metrics["rows"] and not frame.timestamp.duplicated().any()
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    error = frame.net_load_mw - frame.prediction_net_load_mw
    assert np.allclose(frame.residual_mw, error)
    assert abs(error.abs().mean() - metrics["mae_mw"]) < 1e-6
    assert abs(np.sqrt((error ** 2).mean()) - metrics["rmse_mw"]) < 1e-6
    assert abs(error.abs().quantile(.95) - metrics["p95_absolute_error_mw"]) < 1e-6
    radius = metrics["interval_radius_mw"]
    assert abs((error.abs() <= radius).mean() * 100 - metrics["interval_coverage_pct"]) < 1e-6
    assert abs(np.mean(2 * radius + 20 * np.maximum(error.abs() - radius, 0))
               - metrics["interval_score_mw"]) < 1e-6
    residual = (frame.prediction_net_load_mw - frame.prediction_total_load_mw
                + frame.prediction_solar_generation_mw + frame.prediction_wind_generation_mw)
    assert abs(residual.abs().mean() - metrics["accounting_residual_mae_mw"]) < 1e-6


def main() -> None:
    manifest = read(OUT / "ARTIFACT_SHA256.json")
    for relative, value in manifest.items():
        path = (OUT / relative).resolve()
        assert path.is_relative_to(OUT.resolve()) and checksum(path) == value
    summary = read(OUT / "summary/summary.json")
    metric_count, source_checks = 0, 0
    for arm in ("data", "physics"):
        for seed in (42, 43, 44):
            run = OUT / f"{arm}_seed{seed}"
            meta = read(run / "manifest.json")
            for relative, value in meta["code_sha256"].items():
                assert checksum(ROOT / relative) == value, relative
            for relative, value in meta["data_sha256"].items():
                assert checksum(ROOT / "data/processed" / relative) == value
            for variant in ("raw", "projected"):
                for period, metrics in summary["runs"][run.name]["variants"][variant].items():
                    check_metric_file(run / variant / f"{period}.csv", metrics)
                    metric_count += 1
    for seed in (42, 43, 44):
        a = read(OUT / f"data_seed{seed}/manifest.json")
        b = read(OUT / f"physics_seed{seed}/manifest.json")
        assert a["initial_parameter_sha256"] == b["initial_parameter_sha256"]
        assert a["splits"] == b["splits"] and a["data_sha256"] == b["data_sha256"]
    for method, values in summary["baselines"].items():
        for period, metrics in values.items():
            check_metric_file(OUT / "baseline" / method / f"{period}.csv", metrics)
            metric_count += 1

    # Verify prediction truths against the supplied data, not only duplicated CSVs.
    source = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv", parse_dates=["timestamp"])
    for period in ("train", "validation", "test", "2021", "2022"):
        if period in ("2021", "2022"):
            source = pd.read_csv(
                ROOT / f"data/processed/independent_de_{period}_hourly.csv", parse_dates=["timestamp"])
        pred = pd.read_csv(OUT / "data_seed42/raw" / f"{period}.csv", parse_dates=["timestamp"])
        actual = source.set_index("timestamp").loc[pred.timestamp]
        for column in ("total_load_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw"):
            assert np.allclose(pred[column], actual[column], rtol=0, atol=1e-6)
        source_checks += 1

    replay = read(OUT / "diagnostics/twin_metrics.json")
    for year, scenarios in replay.items():
        for key, m in scenarios.items():
            frame = pd.read_csv(OUT / "diagnostics" / f"{year}_{key}.csv")
            eta = m["effective_efficiency"]
            balance = (frame.storage_soc_mwh - frame.soc_before_mwh
                       - eta * frame.charge_mw + frame.discharge_mw / eta)
            assert balance.abs().max() < 1e-6
            assert (frame.storage_soc_mwh.between(4500 - 1e-6, 15000 + 1e-6)).all()
            assert np.allclose(frame.soc_before_mwh.iloc[1:], frame.storage_soc_mwh.iloc[:-1], atol=1e-6)
            assert np.allclose(frame.controlled_net_load_mw,
                               frame.net_load_mw + frame.charge_mw - frame.discharge_mw)
            loss = ((1 - eta) * frame.charge_mw + (1 / eta - 1) * frame.discharge_mw).sum()
            assert abs(loss - m["conversion_loss_mwh"]) < 1e-5
            assert abs(frame.controlled_net_load_mw.max() - m["controlled_peak_mw"]) < 1e-6
    followup = read(OUT / "followup/metrics.json")
    for relative, value in followup["provenance"]["code_sha256"].items():
        assert checksum(ROOT / relative) == value
    for period, metrics in followup["intervals"].items():
        frame = pd.read_csv(OUT / "followup" / f"interval_{period}.csv", parse_dates=["timestamp"])
        errors = frame.residual_mw.abs().to_numpy()
        # Pandas 3 can infer microsecond timestamps; the time window is nanoseconds.
        stamps = pd.DatetimeIndex(frame.timestamp).as_unit("ns").asi8
        expected = np.full(len(frame), frame.radius_mw.iloc[0])
        for i in range(168, len(frame)):
            start = np.searchsorted(stamps, stamps[i] - pd.Timedelta(hours=720).value)
            if i - start >= 168:
                expected[i] = np.quantile(errors[start:i], .9)
        assert np.allclose(frame.rolling_radius_mw, expected, atol=1e-6)
        for kind, radii in (("fixed", frame.radius_mw.to_numpy()), ("rolling", expected)):
            assert abs(np.mean(errors <= radii) * 100 - metrics[kind]["coverage_pct"]) < 1e-6

    from PIL import Image, ImageStat
    for file in ("Figure_1.png", "Figure_2.png"):
        with Image.open(OUT / "summary" / file) as image:
            assert min(image.size) > 100 and max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    report = {
        "passed": True, "artifact_hashes_verified": len(manifest),
        "prediction_metric_sets_recomputed": metric_count,
        "truth_source_splits_verified": source_checks,
        "matched_seed_initialization_and_splits": 3,
        "continuous_replay_trajectories_audited": 10,
        "causal_interval_periods_recomputed": 3,
        "figure_pixel_checks": 2,
        "boundary": "Independent arithmetic/hash/causality audit, not field validation. "
                    "Checkpoint inference replay was performed on remote CPU separately.",
    }
    output = ROOT / "outputs/fusion_v1_local_verification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
