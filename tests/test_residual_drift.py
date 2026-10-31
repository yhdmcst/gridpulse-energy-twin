import numpy as np
import pandas as pd
import pytest

from src.residual_drift import CalibratedResidualMonitor
from scripts.run_fusion_followup import rolling_radii


def test_calibration_is_required_and_bias_perturbation_is_detectable():
    monitor = CalibratedResidualMonitor()
    with pytest.raises(ValueError):
        monitor.analyze(np.zeros(168))
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 300, 10000)
    calibration = [rng.normal(0, 300, 168) for _ in range(25)]
    monitor.fit(baseline, calibration)
    assert not monitor.analyze(calibration[0])["alarm"]
    assert monitor.analyze(baseline[:168] + 3000)["alarm"]


def test_current_and_future_outcomes_never_change_prior_rolling_radii():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=260, freq="h", tz="UTC"),
        "residual_mw": np.arange(260, dtype=float),
    })
    a = rolling_radii(frame, 100.)
    frame.loc[180:, "residual_mw"] += 100000.
    b = rolling_radii(frame, 100.)
    assert np.array_equal(a[:181], b[:181])
    assert not np.array_equal(a[181:], b[181:])
