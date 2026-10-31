import pandas as pd

from src.scenarios import (
    dynamic_reserve_fraction,
    simulate_fixed_reserve_receding_dispatch,
    simulate_risk_aware_dispatch,
    simulate_storage_dispatch,
)


def test_storage_state_stays_within_capacity():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC"),
            "net_load_mw": [20, 20, 100, 100, 100, 20, 20, 20],
            "predicted_net_load_mw": [20, 20, 100, 100, 100, 20, 20, 20],
        }
    )
    result, metrics = simulate_storage_dispatch(
        frame,
        peak_threshold_mw=80,
        capacity_mwh=100,
        max_power_mw=50,
        charge_power_mw=20,
        efficiency=0.9,
    )
    assert result["storage_soc_mwh"].between(0, 100).all()
    assert metrics["min_soc_mwh"] >= 0
    assert metrics["max_soc_mwh"] <= 100


def test_reserve_is_held_until_an_emergency_forecast():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"),
            "net_load_mw": [100, 130],
            "predicted_net_load_mw": [100, 130],
        }
    )
    result, _ = simulate_storage_dispatch(
        frame,
        peak_threshold_mw=80,
        emergency_threshold_mw=120,
        capacity_mwh=200,
        max_power_mw=50,
        charge_power_mw=0,
        efficiency=1.0,
        reserve_fraction=0.30,
    )
    assert result.loc[0, "storage_dispatch_mw"] == 20
    assert result.loc[1, "storage_dispatch_mw"] == 50


def test_dynamic_reserve_rises_with_upper_forecast_stress():
    low_risk = dynamic_reserve_fraction(
        upper_forecasts_mw=pd.array([70.0, 72.0], dtype=float).to_numpy(),
        interval_radii_mw=pd.array([2.0, 2.0], dtype=float).to_numpy(),
        peak_threshold_mw=100.0,
    )
    high_risk = dynamic_reserve_fraction(
        upper_forecasts_mw=pd.array([115.0, 120.0], dtype=float).to_numpy(),
        interval_radii_mw=pd.array([12.0, 12.0], dtype=float).to_numpy(),
        peak_threshold_mw=100.0,
    )
    assert high_risk > low_risk


def test_risk_aware_optimizer_respects_soc_and_returns_an_optimal_plan():
    frame = pd.DataFrame(
        {
            "decision_timestamp": pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC"),
            "timestamp": pd.date_range("2020-01-01 01:00", periods=4, freq="h", tz="UTC"),
            "net_load_mw": [105.0, 110.0, 40.0, 105.0],
        }
    )
    for horizon in range(1, 4):
        frame[f"prediction_h{horizon}_mw"] = [105.0, 110.0, 40.0, 105.0]
        frame[f"lower_h{horizon}_mw"] = [95.0, 100.0, 30.0, 95.0]
        frame[f"upper_h{horizon}_mw"] = [115.0, 120.0, 50.0, 115.0]
    result, metrics = simulate_risk_aware_dispatch(
        frame,
        peak_threshold_mw=100.0,
        capacity_mwh=200.0,
        max_power_mw=50.0,
        charge_power_mw=20.0,
        efficiency=0.9,
        charge_threshold_mw=60.0,
        energy_smoothing_hours=None,
    )
    assert result["storage_soc_mwh"].between(0, 200).all()
    assert (result["solver_status"] == "optimal").all()
    assert metrics["solver_optimal_rate_pct"] == 100.0


def test_fixed_reserve_receding_baseline_uses_constant_reserve_target():
    frame = pd.DataFrame(
        {
            "decision_timestamp": pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC"),
            "timestamp": pd.date_range("2020-01-01 01:00", periods=3, freq="h", tz="UTC"),
            "net_load_mw": [110.0, 40.0, 110.0],
        }
    )
    for horizon in range(1, 3):
        frame[f"prediction_h{horizon}_mw"] = [110.0, 40.0, 110.0]
        frame[f"lower_h{horizon}_mw"] = [100.0, 30.0, 100.0]
        frame[f"upper_h{horizon}_mw"] = [120.0, 50.0, 120.0]
    result, metrics = simulate_fixed_reserve_receding_dispatch(
        frame,
        peak_threshold_mw=100.0,
        capacity_mwh=200.0,
        fixed_reserve_fraction=0.30,
        max_power_mw=50.0,
        charge_power_mw=20.0,
        efficiency=1.0,
        charge_threshold_mw=60.0,
        energy_smoothing_hours=None,
    )
    assert (result["dynamic_reserve_target_mwh"] == 60.0).all()
    assert metrics["reserve_policy"] == "fixed_30%"
    assert metrics["solver_optimal_rate_pct"] == 100.0
