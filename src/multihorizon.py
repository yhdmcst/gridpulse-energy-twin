from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline import interval_metrics, regression_metrics
from src.models import make_gradient_boosting


TARGET = "net_load_mw"
HORIZONS = (1, 2, 3, 4, 5, 6)
VALIDATION_START = pd.Timestamp("2019-01-12 20:00:00+00:00")
TEST_START = pd.Timestamp("2019-08-09 21:00:00+00:00")
FEATURE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
    "net_load_latest_available_mw",
    "net_load_same_day_prior_mw",
    "net_load_same_week_prior_mw",
    "net_load_rolling_mean_24h",
    "net_load_rolling_std_24h",
    "solar_latest_available_mw",
    "solar_same_day_prior_mw",
    "wind_latest_available_mw",
    "wind_same_day_prior_mw",
)


def build_direct_horizon_frame(raw: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Build rows that predict target time t using only observations through t-h."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least 1")
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError("Direct forecasts require unique valid timestamps.")
    if not data["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Direct forecasts require a complete hourly grid, with missing values retained as NaN.")
    timestamp = data["timestamp"]
    hour = timestamp.dt.hour
    weekday = timestamp.dt.dayofweek
    month = timestamp.dt.month
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_of_week_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["is_weekend"] = (weekday >= 5).astype(int)
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    data["net_load_latest_available_mw"] = data[TARGET].shift(horizon_hours)
    data["net_load_same_day_prior_mw"] = data[TARGET].shift(horizon_hours + 23)
    data["net_load_same_week_prior_mw"] = data[TARGET].shift(horizon_hours + 167)
    historical_net_load = data[TARGET].shift(horizon_hours)
    data["net_load_rolling_mean_24h"] = historical_net_load.rolling(24, min_periods=24).mean()
    data["net_load_rolling_std_24h"] = historical_net_load.rolling(24, min_periods=24).std()
    data["solar_latest_available_mw"] = data["solar_generation_mw"].shift(horizon_hours)
    data["solar_same_day_prior_mw"] = data["solar_generation_mw"].shift(horizon_hours + 23)
    data["wind_latest_available_mw"] = data["wind_generation_mw"].shift(horizon_hours)
    data["wind_same_day_prior_mw"] = data["wind_generation_mw"].shift(horizon_hours + 23)
    data["issued_timestamp"] = data["timestamp"] - pd.Timedelta(hours=horizon_hours)
    return data.dropna(subset=[TARGET, *FEATURE_COLUMNS]).reset_index(drop=True)


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["timestamp"] < VALIDATION_START].copy()
    validation = frame.loc[(frame["timestamp"] >= VALIDATION_START) & (frame["timestamp"] < TEST_START)].copy()
    test = frame.loc[frame["timestamp"] >= TEST_START].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological direct-horizon split produced an empty partition")
    return train, validation, test


def train_direct_horizon_models(
    input_path: Path,
    output_dir: Path,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """Train direct, causally available forecasts for an intraday planning window."""
    raw = pd.read_csv(input_path, parse_dates=["timestamp"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    test_predictions: list[pd.DataFrame] = []
    for horizon in horizons:
        data = build_direct_horizon_frame(raw, horizon)
        train, validation, test = _split(data)
        model = make_gradient_boosting().fit(train[list(FEATURE_COLUMNS)], train[TARGET])
        validation_prediction = model.predict(validation[list(FEATURE_COLUMNS)])
        test_prediction = model.predict(test[list(FEATURE_COLUMNS)])
        radius = float(np.quantile(np.abs(validation[TARGET].to_numpy() - validation_prediction), 0.90))
        predictions = pd.DataFrame(
            {
                "issued_timestamp": test["issued_timestamp"].to_numpy(),
                "timestamp": test["timestamp"].to_numpy(),
                "horizon_hours": horizon,
                "net_load_mw": test[TARGET].to_numpy(),
                "prediction_mw": test_prediction,
                "lower_mw": test_prediction - radius,
                "upper_mw": test_prediction + radius,
            }
        )
        test_predictions.append(predictions)
        artifact_path = output_dir / f"direct_horizon_{horizon}h.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_columns": list(FEATURE_COLUMNS),
                "horizon_hours": horizon,
                "prediction_interval_radius_mw": radius,
                "issued_information_boundary": f"observations through target time minus {horizon} hours",
            },
            artifact_path,
        )
        summaries[f"horizon_{horizon}h"] = {
            "horizon_hours": horizon,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "validation": regression_metrics(validation[TARGET], pd.Series(validation_prediction)),
            "test": regression_metrics(test[TARGET], pd.Series(test_prediction)),
            "prediction_interval": {
                "calibration_quantile": 0.90,
                "radius_mw": radius,
                "validation": interval_metrics(
                    validation[TARGET],
                    pd.Series(validation_prediction - radius),
                    pd.Series(validation_prediction + radius),
                ),
                "test": interval_metrics(
                    test[TARGET],
                    pd.Series(test_prediction - radius),
                    pd.Series(test_prediction + radius),
                ),
                "method": "symmetric absolute-residual calibration on validation only",
            },
            "artifact": f"outputs/models/{artifact_path.name}",
        }
    prediction_path = output_dir / "direct_horizon_test_predictions.csv"
    pd.concat(test_predictions, ignore_index=True).to_csv(prediction_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    report = {
        "forecast_design": "direct_multi_horizon_causal",
        "planning_window_hours": max(horizons),
        "horizons": list(horizons),
        "target": TARGET,
        "feature_columns": list(FEATURE_COLUMNS),
        "information_policy": "Each h-hour model uses observations no later than h hours before its target timestamp.",
        "validation_start": str(VALIDATION_START),
        "test_start": str(TEST_START),
        "models": summaries,
        "test_predictions": f"outputs/models/{prediction_path.name}",
    }
    (output_dir / "multihorizon_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_planning_panel(predictions: pd.DataFrame, planning_window_hours: int) -> pd.DataFrame:
    """Align direct forecasts by issuance time for receding-horizon storage decisions."""
    required = {"issued_timestamp", "timestamp", "horizon_hours", "net_load_mw", "prediction_mw", "lower_mw", "upper_mw"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing direct-horizon columns: {sorted(missing)}")
    data = predictions.copy()
    data["issued_timestamp"] = pd.to_datetime(data["issued_timestamp"], utc=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.loc[data["horizon_hours"].between(1, planning_window_hours)].copy()
    rows: list[dict] = []
    for issued_timestamp, group in data.groupby("issued_timestamp", sort=True):
        by_horizon = group.set_index("horizon_hours")
        if not set(range(1, planning_window_hours + 1)).issubset(by_horizon.index):
            continue
        first = by_horizon.loc[1]
        row: dict[str, object] = {
            "decision_timestamp": issued_timestamp,
            "timestamp": first["timestamp"],
            "net_load_mw": float(first["net_load_mw"]),
        }
        for horizon in range(1, planning_window_hours + 1):
            forecast = by_horizon.loc[horizon]
            row[f"prediction_h{horizon}_mw"] = float(forecast["prediction_mw"])
            row[f"lower_h{horizon}_mw"] = float(forecast["lower_mw"])
            row[f"upper_h{horizon}_mw"] = float(forecast["upper_mw"])
        rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No complete direct-forecast windows were available for planning")
    return panel.sort_values("timestamp").reset_index(drop=True)
