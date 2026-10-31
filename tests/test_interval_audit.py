import numpy as np
import pandas as pd

from src.interval_audit import rolling_radius


def test_rolling_interval_uses_only_residuals_observed_by_issue():
    stamps = pd.date_range("2021-01-01",periods=20,freq="h",tz="UTC")
    origins = stamps-pd.Timedelta(hours=2)
    errors = np.arange(20,dtype=float)
    a,counts = rolling_radius(stamps,errors,origins,np.full(20,50.),window_hours=20,minimum=3)
    changed = errors.copy()
    changed[8:] = 9999.
    b,_ = rolling_radius(stamps,changed,origins,np.full(20,50.),window_hours=20,minimum=3)
    assert np.array_equal(a[:11],b[:11])
    assert counts[10] == 8
    assert a[1] == 50.
