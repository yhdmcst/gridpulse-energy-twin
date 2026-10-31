import numpy as np
import pandas as pd
import pytest

from src.independent_validation import normalize_public_power, paired_block_summary


def test_missing_quarter_hour_is_not_averaged_as_complete_hour():
    stamps = pd.date_range("2021-01-01", periods=8, freq="15min", tz="UTC")
    payload = {"unit": "MW", "interval_minutes": 15, "data": [
        {"timestamp": t.isoformat(), "values": {"load": 100, "solar": 10, "wind_onshore": 20, "wind_offshore": 5}}
        for i,t in enumerate(stamps) if i != 2]}
    frame, quality = normalize_public_power(payload, stamps[0], stamps[-1]+pd.Timedelta(minutes=15))
    assert pd.isna(frame.net_load_mw.iloc[0])
    assert frame.net_load_mw.iloc[1] == 65
    assert quality["missing_quarter_hours"] == 1


def test_paired_blocks_preserve_known_policy_difference_and_reject_mismatched_data():
    stamps = pd.date_range("2021-01-01", periods=24*21, freq="h", tz="UTC")
    old = pd.DataFrame({"timestamp": stamps, "net_load_mw": np.arange(len(stamps))+1000,
                        "controlled_net_load_mw": np.arange(len(stamps))+1000})
    new = old.copy()
    new.controlled_net_load_mw -= 10
    summary = paired_block_summary(old,new,repetitions=50)
    for endpoint in ("mean_daily_peak_reduction_mw", "p95_reduction_mw"):
        assert summary[endpoint]["estimate"] == pytest.approx(10)
        assert summary[endpoint]["lower_95"] == pytest.approx(10)
        assert summary[endpoint]["upper_95"] == pytest.approx(10)
    new.loc[0,"net_load_mw"] += 100
    with pytest.raises(ValueError,match="identical"):
        paired_block_summary(old,new,repetitions=10)
