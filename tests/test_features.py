import pandas as pd
import pytest

from src.features import build_features


def test_current_target_is_not_used_in_its_own_lag_features():
    timestamps = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            "timestamp": timestamps,
            "net_load_mw": range(200),
            "solar_generation_mw": range(200),
            "wind_generation_mw": range(200),
        }
    )
    changed = source.copy()
    changed.loc[180, "net_load_mw"] = 99999
    original_features = build_features(source)
    changed_features = build_features(changed)
    original_row = original_features.loc[original_features["timestamp"] == timestamps[180]].iloc[0]
    changed_row = changed_features.loc[changed_features["timestamp"] == timestamps[180]].iloc[0]
    assert original_row["net_load_lag_1h"] == changed_row["net_load_lag_1h"]
    assert original_row["net_load_rolling_mean_24h"] == changed_row["net_load_rolling_mean_24h"]


def test_missing_hour_is_not_silently_relabelled_as_one_hour_lag():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T02:00Z"]), "net_load_mw": [1, 2], "solar_generation_mw": [0, 0], "wind_generation_mw": [0, 0]})
    with pytest.raises(ValueError, match="complete hourly grid"):
        build_features(frame)
