import numpy as np
import pandas as pd
import pytest

from src.energy_twin import (
    TwinConfig,
    default_twin_scenarios,
    simulate_energy_twin,
)


def trace() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "net_load_mw": [100.0, 110.0, 120.0, 90.0],
            "solar_generation_mw": [20.0, 10.0, 0.0, 30.0],
            "charge_mw": [10.0, 0.0, 0.0, 0.0],
            "discharge_mw": [0.0, 20.0, 30.0, 0.0],
            "soc_before_mwh": [100.0, 109.2, 87.4608695652, 54.8521739130],
            "storage_soc_mwh": [109.2, 87.4608695652, 54.8521739130, 54.8521739130],
            "reserve_target_mwh": [30.0, 30.0, 30.0, 30.0],
        }
    )


def config() -> TwinConfig:
    return TwinConfig(
        capacity_mwh=200.0,
        charge_power_mw=20.0,
        discharge_power_mw=30.0,
        efficiency=0.92,
        minimum_soc_mwh=30.0,
        maximum_soc_mwh=200.0,
    )


def test_nominal_replay_preserves_energy_balance_and_peak_metrics():
    result = simulate_energy_twin(trace(), config())
    assert result.metrics["physical_replay_passed"] is True
    assert result.metrics["health_status"] == "可信"
    assert result.metrics["max_source_physics_residual_mwh"] < 1e-8
    assert result.trace.iloc[1]["storage_soc_mwh"] == pytest.approx(87.4608695652)
    assert result.metrics["controlled_peak_mw"] == pytest.approx(110.0)


def test_counterfactual_pv_drop_increases_net_load_without_changing_soc():
    result = simulate_energy_twin(trace(), config(), default_twin_scenarios()["pv_drop_30pct"])
    assert result.trace.loc[0, "net_load_mw"] == pytest.approx(106.0)
    assert result.trace.loc[0, "storage_soc_mwh"] == pytest.approx(109.2)
    assert result.metrics["controlled_peak_mw"] > simulate_energy_twin(trace(), config()).metrics["controlled_peak_mw"]


def test_derating_clips_actions_and_keeps_soc_inside_bounds():
    result = simulate_energy_twin(
        trace(),
        config(),
        default_twin_scenarios()["storage_derate_30pct"],
    )
    assert result.metrics["clipped_discharge_hours"] == 1
    assert result.metrics["hard_bound_hours"] == 0
    assert result.trace["storage_soc_mwh"].between(30.0, 200.0).all()


def test_efficiency_and_delay_are_explicit_counterfactuals():
    scenarios = default_twin_scenarios()
    efficient = simulate_energy_twin(trace(), config(), scenarios["efficiency_085"])
    delayed = simulate_energy_twin(trace(), config(), scenarios["feed_delay_3h"])
    assert efficient.metrics["effective_efficiency"] == pytest.approx(0.85)
    assert delayed.trace.loc[:2, "charge_mw"].sum() == pytest.approx(0.0)
    assert delayed.trace.loc[3, "charge_mw"] == pytest.approx(10.0)


def test_invalid_or_noncontiguous_input_is_rejected():
    broken = trace().drop(index=2)
    with pytest.raises(ValueError, match="contiguous"):
        simulate_energy_twin(broken, config())
    broken = trace().copy()
    broken.loc[0, "net_load_mw"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        simulate_energy_twin(broken, config())
