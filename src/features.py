from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "net_load_mw"
BASE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
    "net_load_lag_1h",
    "net_load_lag_24h",
    "net_load_lag_168h",
    "net_load_rolling_mean_24h",
    "net_load_rolling_std_24h",
    "solar_lag_1h",
    "solar_lag_24h",
    "wind_lag_1h",
    "wind_lag_24h",
]
LAG_ONLY_FEATURES = [feature for feature in BASE_FEATURES if feature.startswith("net_load_") or feature in {"hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend", "month_sin", "month_cos"}]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal features: each observed-value feature is shifted before use."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError("Feature input requires unique, valid hourly timestamps.")
    if not data["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Feature input must retain a complete hourly grid; retain missing values as NaN.")
    timestamp = data["timestamp"]
    hour = timestamp.dt.hour
    day_of_week = timestamp.dt.dayofweek
    month = timestamp.dt.month
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    data["is_weekend"] = (day_of_week >= 5).astype(int)
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    data["net_load_lag_1h"] = data[TARGET].shift(1)
    data["net_load_lag_24h"] = data[TARGET].shift(24)
    data["net_load_lag_168h"] = data[TARGET].shift(168)
    shifted_target = data[TARGET].shift(1)
    data["net_load_rolling_mean_24h"] = shifted_target.rolling(24, min_periods=24).mean()
    data["net_load_rolling_std_24h"] = shifted_target.rolling(24, min_periods=24).std()
    data["solar_lag_1h"] = data["solar_generation_mw"].shift(1)
    data["solar_lag_24h"] = data["solar_generation_mw"].shift(24)
    data["wind_lag_1h"] = data["wind_generation_mw"].shift(1)
    data["wind_lag_24h"] = data["wind_generation_mw"].shift(24)
    return data


def model_frame(frame: pd.DataFrame, feature_columns: list[str] = BASE_FEATURES) -> pd.DataFrame:
    data = build_features(frame)
    return data.dropna(subset=[TARGET, *feature_columns]).reset_index(drop=True)
