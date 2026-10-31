from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.delayed_forecasts import MODES, common_panels, delay_panel
from src.forecast_dispatch import dispatch
from src.optimality import StorageModel
from src.recovery_control import POLICIES, simulate
from src.robustness_admission import ERROR_CASES, MAIN_MODES, demo_admission
from src.robustness_control import annotate, replay_mismatch, simulate_guard

GATES = {"peak_threshold_mw": 150., "charge_threshold_mw": 90., "release_threshold_mw": 180.}


def calibration():
    return {"radii": {str(h): {str(hour): float(h) for hour in range(24)}
                      for h in range(1, 7)}, "uncertainty_scale_mw": 20.}


def forecasts():
    times = pd.date_range("2021-01-01", periods=120, freq="h", tz="UTC")
    result = pd.DataFrame({"timestamp": times, "issued_timestamp": times - pd.Timedelta(hours=1)})
    for h in range(1, 7):
        target = times + pd.Timedelta(hours=h - 1)
        actual = np.where(target.hour < 12, 80., 200.)
        result[f"actual_h{h}_mw"] = actual
        result[f"prediction_h{h}_mw"] = actual + np.arange(len(times)) * .001 + h * .01
    return result


def small_model():
    return StorageModel(capacity_mwh=200., initial_soc_mwh=100., maximum_soc_mwh=100.,
                        minimum_soc_mwh=30., charge_power_mw=20., discharge_power_mw=20.)


def panels():
    return common_panels({"data_seed42": forecasts()}, {"methods": {"data_seed42": calibration()}})[0]["data_seed42"]


@pytest.mark.parametrize("name", list(MODES))
def test_delayed_forecast_uses_original_lead_and_correct_information_time(name):
    frame = forecasts()
    d, n = MODES[name]
    result = delay_panel(frame, calibration(), d, n)
    target = pd.Timestamp("2021-01-02T12:00:00Z")
    row = result.loc[result.timestamp.eq(target)].iloc[0]
    origin = frame.loc[frame.timestamp.eq(target - pd.Timedelta(hours=d))].iloc[0]
    assert row.source_first_target_timestamp == target - pd.Timedelta(hours=d)
    assert row.issued_timestamp == target - pd.Timedelta(hours=d + 1)
    assert row.available_timestamp == target
    for h in range(1, n + 1):
        assert row[f"prediction_h{h}_mw"] == origin[f"prediction_h{h + d}_mw"]
        assert row[f"upper_h{h}_mw"] - row[f"prediction_h{h}_mw"] == pytest.approx(h + d)
    assert row.net_load_mw == frame.loc[frame.timestamp.eq(target), "actual_h1_mw"].iloc[0]


def test_missing_origin_is_excluded_without_interpolation_and_shared_days_are_explicit():
    frame = forecasts()
    missing = frame.drop(index=30)
    result = delay_panel(missing, calibration(), 3, 3)
    assert frame.timestamp.iloc[33] not in set(result.timestamp)
    prepared, sample = common_panels(
        {"a": frame, "b": frame.copy()}, {"methods": {"a": calibration(), "b": calibration()}})
    assert sample["days"] == 4
    for method in prepared.values():
        assert all(len(p) == 96 for p in method.values())
    with pytest.raises(ValueError):
        delay_panel(frame, calibration(), 3, 6)


def test_newly_issued_forecasts_cannot_influence_delayed_prefix():
    source = forecasts()
    old = delay_panel(source, calibration(), 3, 3)
    source.loc[36:, [f"prediction_h{h}_mw" for h in range(1, 7)]] += 50.
    new = delay_panel(source, calibration(), 3, 3)
    boundary = source.timestamp.iloc[39]
    cols = [f"prediction_h{h}_mw" for h in range(1, 4)]
    assert np.allclose(old.loc[old.timestamp < boundary, cols], new.loc[new.timestamp < boundary, cols])
    assert not np.allclose(old.loc[old.timestamp >= boundary, cols], new.loc[new.timestamp >= boundary, cols])


def test_fresh6_adapter_exactly_reproduces_frozen_v5_solver():
    p = panels()["fresh6"]
    model = small_model()
    reference, _ = dispatch(p, calibration(), GATES, model)
    old_panel = p.drop(columns=["source_first_target_timestamp", "available_timestamp",
                                "observation_delay_hours", "forecast_hours"])
    a, _ = simulate(old_panel, reference, calibration(), GATES, 120., POLICIES["tail_guard"], model)
    b, metrics = simulate_guard(p, reference, calibration(), GATES, 120., model)
    for column in ("charge_mw", "discharge_mw", "storage_soc_mwh", "controlled_net_load_mw"):
        assert np.allclose(a[column], b[column], rtol=0, atol=1e-8)
    assert metrics["terminal_requirement_passed"] and metrics["physical_audit_passed"]


@pytest.mark.parametrize("name", list(MODES))
def test_variable_horizon_actions_ignore_current_actual_load(name):
    p = panels()[name]
    model = small_model()
    ref, _ = dispatch(p, calibration(), GATES, model)
    a, _ = simulate_guard(p, ref, calibration(), GATES, 120., model)
    modified = p.copy()
    modified.net_load_mw = np.arange(len(p)) * 1e6
    b, metrics = simulate_guard(modified, ref, calibration(), GATES, 120., model)
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert np.allclose(a.discharge_mw, b.discharge_mw)
    assert metrics["terminal_requirement_passed"]


def test_mismatch_replay_does_not_reset_energy_or_hide_terminal_failure():
    p = panels()["fresh6"]
    model = small_model()
    reference, _ = dispatch(p, calibration(), GATES, model)
    nominal, _ = simulate_guard(p, reference, calibration(), GATES, 120., model)
    actual = replace(model, efficiency=.85)
    trace, metrics = replay_mismatch(nominal, actual, GATES, 120.)
    trace, metrics = annotate(trace, metrics)
    assert np.allclose(trace.soc_before_mwh.iloc[1:], trace.storage_soc_mwh.iloc[:-1])
    assert metrics["physical_audit_passed"]
    assert not metrics["terminal_requirement_passed"] and metrics["terminal_failure_days"] > 0
    assert metrics["max_abs_terminal_error_mwh"] > 0
    assert np.allclose(trace.requested_charge_mw, nominal.charge_mw)


def admission_fixture():
    legacy = {
        "controlled_peak_mw": 200., "controlled_p95_mw": 180., "throughput_mwh": 100.,
        "conversion_loss_mwh": 10., "max_abs_terminal_error_mwh": 0.,
        "physical_audit_passed": True, "terminal_requirement_passed": True,
    }
    candidate = dict(legacy, controlled_p95_mw=178.)
    delays = {year: {mode: {f"data_seed{s}": {"legacy": dict(legacy), "tail_guard": dict(candidate)}
                            for s in (42, 43, 44)} for mode in MAIN_MODES}
              for year in ("2021", "2022")}
    mismatch = {year: {mode: {case: {"legacy": dict(legacy), "tail_guard": dict(candidate)}
                             for case in ERROR_CASES} for mode in MAIN_MODES}
                for year in ("2021", "2022")}
    return delays, mismatch


def test_demo_admission_rejects_terminal_or_per_seed_regression_without_relaxation():
    delay, error = admission_fixture()
    assert demo_admission(delay, error, audit_ok=True)["default_realtime_demo_admitted"]
    changed = deepcopy(error)
    changed["2021"]["delayed1"]["efficiency_090"]["tail_guard"]["terminal_requirement_passed"] = False
    assert not demo_admission(delay, changed, audit_ok=True)["default_realtime_demo_admitted"]
    delay["2022"]["delayed3"]["data_seed43"]["tail_guard"]["controlled_peak_mw"] = 226.
    assert not demo_admission(delay, error, audit_ok=True)["default_realtime_demo_admitted"]
    with pytest.raises(ValueError):
        demo_admission({}, {}, audit_ok=True)
