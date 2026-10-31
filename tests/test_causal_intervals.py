import numpy as np
import pandas as pd
import pytest

from src.causal_intervals import rolling_radii


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_window_is_720_hours_independent_of_datetime_unit(unit):
    times = pd.date_range("2021-01-01", periods=1000, freq="h", tz="UTC").as_unit(unit)
    frame = pd.DataFrame({"timestamp": times, "residual_mw": np.arange(1000.)})
    result = rolling_radii(frame, 1.)
    assert result[999] == pytest.approx(np.quantile(np.arange(279., 999.), .9))
    frame.loc[999, "residual_mw"] = 1e9
    assert np.allclose(result, rolling_radii(frame, 1.))
    assert result[167] == 1.


def test_missing_hours_are_elapsed_time_not_row_count():
    times = pd.date_range("2021-01-01", periods=1000, freq="2h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "residual_mw": np.arange(1000.)})
    assert rolling_radii(frame, 1.)[-1] == pytest.approx(np.quantile(np.arange(639., 999.), .9))
    frame.loc[0, "residual_mw"] = np.nan
    with pytest.raises(ValueError):
        rolling_radii(frame, 1.)
