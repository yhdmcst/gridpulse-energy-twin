from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.forecast_dispatch import (
    calibrate, shared_complete_days, make_panel, dispatch, replay, validate_prediction,
)
from src.optimality import StorageModel


def forecast(start="2019-02-01", days=2):
    stamps = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    values = np.where(stamps.hour < 12, 80., 200.)
    return pd.DataFrame({"timestamp": stamps, "net_load_mw": values,
                         "prediction_net_load_mw": values + 5.})


def small_model():
    return StorageModel(capacity_mwh=200., initial_soc_mwh=100.,
                        maximum_soc_mwh=100., minimum_soc_mwh=30.,
                        charge_power_mw=20., discharge_power_mw=20., efficiency=.92)


def calibration():
    return {"hourly_radius_mw": {str(h): 5. for h in range(24)},
            "uncertainty_scale_mw": 20.}


GATES = {"peak_threshold_mw": 150., "charge_threshold_mw": 90., "release_threshold_mw": 180.}


def test_only_validation_values_calibrate_gates_and_method_bands():
    a = forecast()
    b = a.copy()
    b.prediction_net_load_mw += 50.
    common, _ = shared_complete_days({"a": a, "b": b})
    result = calibrate(common)
    assert result["methods"]["a"]["hourly_radius_mw"]["0"] == 5.
    assert result["methods"]["b"]["hourly_radius_mw"]["0"] == 55.
    assert result["gates"]["peak_threshold_mw"] == a.net_load_mw.quantile(.95)
    with pytest.raises(ValueError, match="Calibration"):
        calibrate({"a": forecast("2021-01-01")})


def test_shared_day_selection_is_explicit_and_rejects_truth_mismatch():
    a = forecast()
    b = a.drop(index=10)
    common, info = shared_complete_days({"a": a, "b": b})
    assert info["days"] == 1 and len(info["omitted_partial_days"]) == 1
    assert common["a"].timestamp.equals(common["b"].timestamp)
    b = a.copy()
    b.loc[0, "net_load_mw"] += 1.
    with pytest.raises(ValueError, match="actual loads"):
        shared_complete_days({"a": a, "b": b})


def test_one_hour_actions_never_read_evaluation_truth():
    panel = make_panel(forecast("2021-01-01"), calibration())
    original, metrics = dispatch(panel, calibration(), GATES, small_model())
    panel["net_load_mw"] = np.arange(len(panel), dtype=float) * 10000
    changed, _ = dispatch(panel, calibration(), GATES, small_model())
    assert np.allclose(original.charge_mw, changed.charge_mw)
    assert np.allclose(original.discharge_mw, changed.discharge_mw)
    assert metrics["terminal_requirement_passed"]
    assert metrics["consecutive_hour_continuity_error_mwh"] < 1e-6


@pytest.mark.parametrize("options", [{"charge_power_mw": 14., "discharge_power_mw": 14.},
                                     {"efficiency": .85}])
def test_changed_parameters_replan_restores_energy_and_replay_reports_deficit(options):
    panel = make_panel(forecast("2021-01-01"), calibration())
    model = small_model()
    nominal, _ = dispatch(panel, calibration(), GATES, model)
    changed_model = replace(model, **options)
    replan, metrics = dispatch(panel, calibration(), GATES, changed_model)
    replayed, replay_metrics = replay(nominal, changed_model)
    assert metrics["terminal_requirement_passed"]
    assert metrics["physical_audit_passed"] and replay_metrics["physical_audit_passed"]
    assert replay_metrics["max_abs_terminal_error_mwh"] >= 0
    assert replan.timestamp.equals(replayed.timestamp)
    assert np.allclose(replan.net_load_mw, replayed.net_load_mw)


def test_missing_or_nonfinite_forecast_cannot_become_success():
    frame = forecast()
    frame.loc[0, "prediction_net_load_mw"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_prediction(frame)
