from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import regression_metrics
from src.features import BASE_FEATURES, TARGET, model_frame
from src.models import make_gradient_boosting


FOLD_STARTS = (
    "2018-01-01T00:00:00Z",
    "2018-07-01T00:00:00Z",
    "2019-01-01T00:00:00Z",
    "2019-07-01T00:00:00Z",
    "2020-01-01T00:00:00Z",
)
HORIZON_DAYS = 30


def seasonal_naive(frame: pd.DataFrame) -> pd.Series:
    lookup = frame[["timestamp", TARGET]].rename(columns={"timestamp": "lag_timestamp", TARGET: "prediction"})
    joined = frame.assign(lag_timestamp=frame["timestamp"] - pd.Timedelta(hours=24)).merge(
        lookup, on="lag_timestamp", how="left"
    )
    return joined["prediction"]


def run_rolling_backtest(input_path: Path, output_path: Path) -> dict:
    raw = pd.read_csv(input_path, parse_dates=["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    data = model_frame(raw)
    rows: list[dict] = []
    for start_text in FOLD_STARTS:
        start = pd.Timestamp(start_text)
        end = start + pd.Timedelta(days=HORIZON_DAYS)
        train = data.loc[data["timestamp"] < start]
        test = data.loc[(data["timestamp"] >= start) & (data["timestamp"] < end)].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"Fold has no data: {start} to {end}")
        model = make_gradient_boosting().fit(train[BASE_FEATURES], train[TARGET])
        prediction = model.predict(test[BASE_FEATURES])
        # Feature eligibility guarantees the timestamp-aligned day lag exists.
        naive = test["net_load_lag_24h"]
        rows.append(
            {
                "fold_start": str(start),
                "fold_end_exclusive": str(end),
                "train_start": str(train["timestamp"].iloc[0]),
                "train_end": str(train["timestamp"].iloc[-1]),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "baseline_rows": int(len(test)),
                "hist_gradient_boosting": regression_metrics(test[TARGET], pd.Series(prediction)),
                "seasonal_naive_24h": regression_metrics(
                    test[TARGET], naive
                ),
                "persistence_1h": regression_metrics(test[TARGET], test["net_load_lag_1h"]),
            }
        )
    model_mae = [row["hist_gradient_boosting"]["mae_mw"] for row in rows]
    naive_mae = [row["seasonal_naive_24h"]["mae_mw"] for row in rows]
    result = {
        "schema_version": "gridpulse-rolling-backtest.v1",
        "forecast_design": "one_step_ahead_causal",
        "horizon_days": HORIZON_DAYS,
        "fold_starts": list(FOLD_STARTS),
        "selection_policy": "Fold dates and horizon are fixed in source code before evaluation; no fold is selected by performance.",
        "folds": rows,
        "summary": {
            "hist_gradient_boosting_mae_mean_mw": float(pd.Series(model_mae).mean()),
            "hist_gradient_boosting_mae_std_mw": float(pd.Series(model_mae).std(ddof=1)),
            "seasonal_naive_mae_mean_mw": float(pd.Series(naive_mae).mean()),
            "seasonal_naive_mae_std_mw": float(pd.Series(naive_mae).std(ddof=1)),
            "folds_where_model_beats_baseline": int(sum(a < b for a, b in zip(model_mae, naive_mae))),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/rolling_backtest_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(run_rolling_backtest(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
