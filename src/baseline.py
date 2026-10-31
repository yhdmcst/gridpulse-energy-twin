from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "net_load_mw"
LAG_HOURS = 24


@dataclass(frozen=True)
class SplitInfo:
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_rows: int
    validation_rows: int
    test_rows: int


def load_evaluable_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    lookup = frame[["timestamp", TARGET]].rename(
        columns={"timestamp": "lag_timestamp", TARGET: "seasonal_naive_prediction_mw"}
    )
    frame["lag_timestamp"] = frame["timestamp"] - pd.Timedelta(hours=LAG_HOURS)
    frame = frame.merge(lookup, on="lag_timestamp", how="left")
    frame = frame.dropna(subset=[TARGET, "seasonal_naive_prediction_mw"]).reset_index(drop=True)
    frame["residual_mw"] = frame[TARGET] - frame["seasonal_naive_prediction_mw"]
    return frame


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    if len(frame) < 100:
        raise ValueError("At least 100 evaluable observations are required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave a test split")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()
    info = SplitInfo(
        train_end=str(train["timestamp"].iloc[-1]),
        validation_start=str(validation["timestamp"].iloc[0]),
        validation_end=str(validation["timestamp"].iloc[-1]),
        test_start=str(test["timestamp"].iloc[0]),
        test_end=str(test["timestamp"].iloc[-1]),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
    )
    return train, validation, test, info


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    nonzero = actual.to_numpy(dtype=float) != 0
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error[nonzero] / actual.to_numpy(dtype=float)[nonzero])) * 100),
        "smape_pct": float(
            np.mean(
                2 * np.abs(error) / (np.abs(actual.to_numpy(dtype=float)) + np.abs(predicted.to_numpy(dtype=float)))
            )
            * 100
        ),
    }


def interval_metrics(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    return {
        "coverage_pct": float(np.mean((actual_values >= lower_values) & (actual_values <= upper_values)) * 100),
        "mean_width_mw": float(np.mean(upper_values - lower_values)),
    }


def run_baseline(input_path: Path, prediction_path: Path, metrics_path: Path) -> dict:
    evaluable = load_evaluable_frame(input_path)
    train, validation, test, split = chronological_split(evaluable)
    metrics = {
        "model": "seasonal_naive_24h",
        "target": TARGET,
        "lag_hours": LAG_HOURS,
        "source_rows": int(pd.read_csv(input_path, usecols=["timestamp"]).shape[0]),
        "evaluable_rows": int(len(evaluable)),
        "split": asdict(split),
        "validation": regression_metrics(validation[TARGET], validation["seasonal_naive_prediction_mw"]),
        "test": regression_metrics(test[TARGET], test["seasonal_naive_prediction_mw"]),
        "protocol": [
            "Chronological 70/10/20 split over evaluable target/lag pairs.",
            "The previous UTC day's same hour is joined by timestamp, not by row offset.",
            "No random shuffling and no final-test-based threshold or model selection.",
            "MAPE excludes zero actual targets; sMAPE is reported as a complementary scale-free metric.",
        ],
    }
    output = test[["timestamp", TARGET, "seasonal_naive_prediction_mw", "residual_mw"]].copy()
    output["split"] = "test"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(prediction_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions/seasonal_naive_test.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics/seasonal_naive_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.input, args.predictions, args.metrics), indent=2))


if __name__ == "__main__":
    main()
