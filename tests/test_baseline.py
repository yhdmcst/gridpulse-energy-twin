import pandas as pd

from src.baseline import load_evaluable_frame, regression_metrics


def test_seasonal_join_uses_timestamp_not_row_offset(tmp_path):
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=50, freq="h", tz="UTC"),
            "net_load_mw": range(50),
        }
    )
    source.loc[24, "net_load_mw"] = None
    path = tmp_path / "data.csv"
    source.to_csv(path, index=False)

    frame = load_evaluable_frame(path)
    row = frame.loc[frame["timestamp"] == pd.Timestamp("2020-01-02 00:00", tz="UTC")]
    assert row.empty
    row = frame.loc[frame["timestamp"] == pd.Timestamp("2020-01-02 01:00", tz="UTC")]
    assert row.iloc[0]["seasonal_naive_prediction_mw"] == 1


def test_regression_metrics_are_deterministic():
    actual = pd.Series([10.0, 20.0])
    predicted = pd.Series([8.0, 24.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae_mw"] == 3.0
    assert metrics["rmse_mw"] == 3.1622776601683795
    assert metrics["mape_pct"] == 20.0
