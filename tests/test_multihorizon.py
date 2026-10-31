import pandas as pd

from src.multihorizon import build_direct_horizon_frame, build_planning_panel


def test_direct_horizon_features_use_only_available_history():
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=220, freq="h", tz="UTC"),
            "net_load_mw": range(220),
            "solar_generation_mw": range(220),
            "wind_generation_mw": range(220),
        }
    )
    frame = build_direct_horizon_frame(raw, horizon_hours=3)
    row = frame.loc[frame["timestamp"] == pd.Timestamp("2020-01-09 00:00:00+00:00")].iloc[0]
    assert row["net_load_latest_available_mw"] == 189
    assert row["issued_timestamp"] == pd.Timestamp("2020-01-08 21:00:00+00:00")


def test_planning_panel_requires_a_complete_horizon_window():
    issued = pd.Timestamp("2020-01-01 00:00:00+00:00")
    rows = []
    for horizon in range(1, 4):
        rows.append(
            {
                "issued_timestamp": issued,
                "timestamp": issued + pd.Timedelta(hours=horizon),
                "horizon_hours": horizon,
                "net_load_mw": 100.0,
                "prediction_mw": 100.0 + horizon,
                "lower_mw": 90.0,
                "upper_mw": 110.0,
            }
        )
    panel = build_planning_panel(pd.DataFrame(rows), planning_window_hours=3)
    assert len(panel) == 1
    assert panel.loc[0, "prediction_h3_mw"] == 103.0
