"""Post-v1 validation-only calibration correction and causal interval diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.residual_drift import CalibratedResidualMonitor


def windows(frame: pd.DataFrame):
    for start in range(0, len(frame) - 167, 168):
        item = frame.iloc[start:start + 168]
        stamps = pd.DatetimeIndex(item.timestamp)
        if stamps.equals(pd.date_range(stamps[0], periods=168, freq="h")):
            yield item


def rolling_radii(frame: pd.DataFrame, initial_radius: float) -> np.ndarray:
    stamps = pd.DatetimeIndex(frame.timestamp).asi8
    errors = frame.residual_mw.abs().to_numpy()
    radii = np.full(len(frame), initial_radius)
    for i in range(len(frame)):
        start = np.searchsorted(stamps, stamps[i] - pd.Timedelta(hours=720).value)
        history = errors[start:i]  # Outcome of target hour i is not available.
        if len(history) >= 168:
            radii[i] = np.quantile(history, .9)
    return radii


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/fusion_v1")
    args = parser.parse_args()
    output = args.input / "followup"
    output.mkdir(exist_ok=False)
    source = args.input / "data_seed42/raw"
    frames = {name: pd.read_csv(source / f"{name}.csv", parse_dates=["timestamp"])
              for name in ("train", "validation", "test", "2021", "2022")}
    model = CalibratedResidualMonitor()
    calibration = model.fit(
        frames["train"].residual_mw.to_numpy(),
        [x.residual_mw.to_numpy() for x in windows(frames["validation"])],
    )
    rows, intervals = [], {}
    for name in ("test", "2021", "2022"):
        frame = frames[name]
        for window in windows(frame):
            for kind, bias in (("unmodified", 0.), ("synthetic_bias_3000mw", 3000.)):
                result = model.analyze(window.residual_mw.to_numpy() + bias)
                rows.append({"period": name, "start": window.timestamp.iloc[0].isoformat(),
                             "kind": kind, **result})
        adaptive = rolling_radii(frame, float(frame.radius_mw.iloc[0]))
        abs_error = frame.residual_mw.abs().to_numpy()
        item = {}
        for method, radius in (("fixed", frame.radius_mw.to_numpy()), ("rolling", adaptive)):
            item[method] = {
                "coverage_pct": float((abs_error <= radius).mean() * 100),
                "mean_width_mw": float(2 * radius.mean()),
                "interval_score_mw": float(np.mean(2 * radius + 20 * np.maximum(abs_error - radius, 0))),
            }
        export = frame[["timestamp", "residual_mw", "radius_mw"]].copy()
        export["rolling_radius_mw"] = adaptive
        export.to_csv(output / f"interval_{name}.csv", index=False)
        intervals[name] = item
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "calibrated_drift_windows.csv", index=False)
    report = {
        "calibration": calibration,
        "alarm_rates": detail.groupby(["period", "kind"]).alarm.agg(
            ["mean", "count"]).reset_index().to_dict(orient="records"),
        "intervals": intervals,
        "provenance": {
            "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), ROOT / "src/residual_drift.py",
                                      ROOT / "experiments/fusion_v1/followup_protocol.md"]},
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source.glob("*.csv")},
        },
        "boundary": "Post-v1 correction; seed42 fixed in advance, calibration on validation only. "
                    "Weekly historical alarm rate is unlabelled, not field FPR. "
                    "Rolling intervals use only prior observed outcomes; no controller update "
                    "or prediction retraining. No statistical coverage guarantee.",
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
