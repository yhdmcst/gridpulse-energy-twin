import pandas as pd

from src.twin_drift import EnergyTwinDriftMonitor


def frame(loads, residual=None):
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=len(loads), freq="h", tz="UTC"),
        "net_load_mw": loads,
    }
    if residual is not None:
        data["residual_mw"] = residual
    return pd.DataFrame(data)


def test_stable_window_does_not_trigger_drift():
    monitor = EnergyTwinDriftMonitor()
    baseline = frame([100.0] * 48)
    monitor.set_baseline(baseline)
    report = monitor.analyze(frame([100.0] * 24))
    assert not report.drift_detected
    assert report.score == 100.0
    assert not report.recalibration_recommended


def test_load_regime_change_triggers_conservative_recommendation():
    monitor = EnergyTwinDriftMonitor()
    monitor.set_baseline(frame([100.0] * 48))
    report = monitor.analyze(frame([150.0] * 24))
    assert report.drift_detected
    assert "load_regime" in report.drift_types
    assert report.recalibration_recommended
    assert report.urgency in {"low", "medium", "high"}


def test_residual_drift_triggers_interval_recalibration():
    monitor = EnergyTwinDriftMonitor()
    monitor.set_baseline(frame([100.0] * 48, residual=[0.0] * 48))
    report = monitor.analyze(frame([100.0] * 24, residual=[5000.0] * 24))
    assert report.drift_detected
    assert "forecast_model" in report.drift_types
    assert any("预测区间" in recommendation for recommendation in report.recommendations)
