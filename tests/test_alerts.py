import pandas as pd

from src.alerts import _sustained_mask, apply_alerts
from src.scenarios import dispatch_suggestions


def test_sustained_signal_is_prefix_invariant():
    values = pd.Series([200., 200., 200., 200., -200., -200., -200., 0.])
    full = _sustained_mask(values, 100, 3)
    assert full.tolist() == [False, False, True, True, False, False, True, False]
    for length in range(1, len(values) + 1):
        assert _sustained_mask(values.iloc[:length], 100, 3).tolist() == full.iloc[:length].tolist()


def test_sustained_signal_resets_at_missing_hour():
    timestamps = pd.Series(pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T01:00Z", "2020-01-01T03:00Z"]))
    assert not _sustained_mask(pd.Series([200., 200., 200.]), 100, 3, timestamps).any()


def test_interval_watch_has_a_label_and_suggestion_but_is_not_candidate_fault():
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=1, tz="UTC"), "predicted_net_load_mw": [90.], "prediction_upper_mw": [110.], "solar_generation_mw": [10.], "residual_mw": [0.]})
    result = apply_alerts(frame, {"peak_prediction_threshold_mw": 100., "solar_drop_threshold_mw_per_hour": -10., "absolute_residual_threshold_mw": 20., "sustained_hours": 3})
    assert result.loc[0, "risk_types"] == "interval_peak_watch"
    assert result.loc[0, "has_review_signal"]
    assert not result.loc[0, "has_candidate_risk"]
    assert dispatch_suggestions(result.loc[0, "risk_types"], 90, 0)
