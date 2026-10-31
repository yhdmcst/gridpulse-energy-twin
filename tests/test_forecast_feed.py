import numpy as np
import pandas as pd
import pytest

from src.forecast_feed import stress_feed


def panel():
    frame = pd.DataFrame({"timestamp":pd.date_range("2021-01-01",periods=30,freq="h",tz="UTC"),"net_load_mw":99999.})
    for h in range(1,7):
        frame[f"prediction_h{h}_mw"] = np.arange(30)*10+h
        frame[f"upper_h{h}_mw"] = frame[f"prediction_h{h}_mw"]+5.
    return frame


def test_lag_aligns_target_horizons_and_does_not_read_truth():
    frame = panel()
    a,_ = stress_feed(frame,delay_hours=1,allow_warmup=True)
    assert a.loc[20,"prediction_h1_mw"] == frame.loc[19,"prediction_h2_mw"]
    assert a.loc[20,"prediction_h6_mw"] == frame.loc[19,"prediction_h6_mw"]
    frame.net_load_mw = 0.
    b,_ = stress_feed(frame,delay_hours=1,allow_warmup=True)
    assert a.prediction_h1_mw.equals(b.prediction_h1_mw)


def test_outage_reuses_last_issue_then_recovers():
    frame = panel()
    a,s = stress_feed(frame,outage_rows=list(range(10,17)))
    assert a.loc[10,"prediction_h1_mw"] == frame.loc[9,"prediction_h2_mw"]
    assert a.loc[16,"prediction_h1_mw"] == frame.loc[9,"prediction_h6_mw"]
    assert a.loc[17,"prediction_h1_mw"] == frame.loc[17,"prediction_h1_mw"]
    assert s["stale_issue_hours"] == 7
    with pytest.raises(ValueError,match="Warmup"):
        stress_feed(frame,delay_hours=1)
