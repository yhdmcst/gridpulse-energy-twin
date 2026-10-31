import numpy as np
import pandas as pd
import pytest

from src.balanced_storage import complete_utc_days, simulate_balanced_storage


def panel():
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC"), "net_load_mw": 110.0})
    for horizon in range(1, 7):
        frame[f"prediction_h{horizon}_mw"] = 110.0
        frame[f"lower_h{horizon}_mw"] = 105.0
        frame[f"upper_h{horizon}_mw"] = 115.0
    return frame


def run(frame, **options):
    return simulate_balanced_storage(frame, 100., 60., 30., capacity_mwh=200., charge_power_mw=20., discharge_power_mw=20., efficiency=.9, fixed_reserve_fraction=.30, **options)


def test_balanced_policy_restores_soc_and_counts_non_low_load_recharge():
    trace, metrics = run(panel())
    assert metrics["max_abs_terminal_error_mwh"] < 1e-7
    assert metrics["hard_soc_violation_hours"] == 0
    assert metrics["energy_balance_error_mwh"] < 1e-7
    assert metrics["discharge_energy_mwh"] > 0
    assert metrics["recovery_charge_mwh"] > 0
    assert metrics["charge_energy_mwh"] > metrics["discharge_energy_mwh"]
    assert metrics["charge_energy_mwh"] - metrics["discharge_energy_mwh"] == pytest.approx(metrics["conversion_loss_mwh"])
    assert np.allclose(trace.controlled_net_load_mw, trace.net_load_mw + trace.charge_mw - trace.discharge_mw)
    assert (np.minimum(trace.charge_mw, trace.discharge_mw) < 1e-7).all()


def test_control_does_not_read_future_actual_load():
    original, _ = run(panel())
    changed = panel()
    changed.loc[12:, "net_load_mw"] = 99999.
    result, _ = run(changed)
    assert np.allclose(original.charge_mw, result.charge_mw)
    assert np.allclose(original.discharge_mw, result.discharge_mw)


def test_open_control_reports_consumed_initial_energy():
    _, metrics = run(panel(), daily_terminal=False)
    assert metrics["net_initial_energy_consumed_mwh"] > 0
    assert metrics["max_abs_terminal_error_mwh"] > 0
    assert metrics["recovery_charge_mwh"] == 0


def test_incomplete_day_is_explicitly_excluded():
    first = panel()
    second = panel()
    second.timestamp += pd.Timedelta(days=1)
    selected, omitted = complete_utc_days(pd.concat([first.drop(index=12), second]))
    assert len(selected) == 24
    assert omitted == ["2020-01-01T00:00:00+00:00"]


def test_zero_charge_power_cannot_spend_terminal_energy():
    trace, metrics = simulate_balanced_storage(panel(), 100., 60., 30., capacity_mwh=200., charge_power_mw=0., discharge_power_mw=20.)
    assert metrics["discharge_energy_mwh"] == pytest.approx(0.)
    assert metrics["max_abs_terminal_error_mwh"] == pytest.approx(0.)


def extreme_panel():
    frame = panel()
    frame.loc[12, "net_load_mw"] = 200.
    frame.loc[12, ["prediction_h1_mw", "lower_h1_mw", "upper_h1_mw"]] = [200., 195., 205.]
    # A distant interval is wider despite a lower point forecast.
    for h in range(2, 7):
        frame.loc[12, f"upper_h{h}_mw"] = 310.
    return frame


def test_point_objective_does_not_mistake_distant_band_width_for_current_peak():
    frame = extreme_panel()
    upper, _ = run(frame, release_threshold_mw=150.)
    point, metrics = run(frame, release_threshold_mw=150., planning_objective="point")
    assert point.loc[12, "discharge_mw"] > upper.loc[12, "discharge_mw"]
    assert point.loc[12, "controlled_net_load_mw"] < upper.loc[12, "controlled_net_load_mw"]
    assert metrics["max_abs_terminal_error_mwh"] < 1e-7


def test_extreme_release_preserves_hard_floor_and_terminal_recovery():
    frame = extreme_panel()
    options = dict(capacity_mwh=200., charge_power_mw=20., discharge_power_mw=20.,
                   efficiency=.9, fixed_reserve_fraction=.49, planning_objective="point")
    normal, _ = simulate_balanced_storage(frame, 100., 60., 30., **options)
    released, metrics = simulate_balanced_storage(frame, 100., 60., 30., release_threshold_mw=150., **options)
    assert released.loc[12, "discharge_mw"] > normal.loc[12, "discharge_mw"]
    assert released.loc[12, "reserve_release_active"]
    assert metrics["hard_soc_violation_hours"] == 0
    assert metrics["max_abs_terminal_error_mwh"] < 1e-7
    assert metrics["net_initial_energy_consumed_mwh"] == pytest.approx(0.)


def test_extreme_policy_never_uses_actual_load_for_release_or_actions():
    original, _ = run(extreme_panel(), release_threshold_mw=150., planning_objective="point")
    frame = extreme_panel()
    frame["net_load_mw"] = np.arange(24) * 10000.
    result, _ = run(frame, release_threshold_mw=150., planning_objective="point")
    assert np.allclose(original.charge_mw, result.charge_mw)
    assert np.allclose(original.discharge_mw, result.discharge_mw)
    assert original.reserve_release_active.equals(result.reserve_release_active)


def test_extreme_release_cannot_bypass_zero_charge_terminal_budget():
    _, metrics = simulate_balanced_storage(extreme_panel(), 100., 60., 30., capacity_mwh=200.,
        charge_power_mw=0., discharge_power_mw=20., release_threshold_mw=150., planning_objective="point")
    assert metrics["discharge_energy_mwh"] == pytest.approx(0.)
    assert metrics["max_abs_terminal_error_mwh"] == pytest.approx(0.)
