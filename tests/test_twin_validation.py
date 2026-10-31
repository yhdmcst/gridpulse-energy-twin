from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.energy_twin import TwinConfig, TwinScenario, simulate_energy_twin
from src.twin_drift import DriftThresholds, EnergyTwinDriftMonitor


def trace():
    return pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=24, freq="h", tz="UTC"),
        "net_load_mw": 100., "charge_mw": 0., "discharge_mw": 0.,
        "soc_before_mwh": 15000., "storage_soc_mwh": 15000.,
    })


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_parameters_fail_closed(value):
    with pytest.raises(ValueError, match="finite"):
        simulate_energy_twin(trace(), replace(TwinConfig(), capacity_mwh=value))
    with pytest.raises(ValueError, match="finite"):
        simulate_energy_twin(trace(), TwinConfig(), TwinScenario("bad", "bad", net_load_bias_mw=value))


def test_missing_state_or_negative_actions_cannot_claim_valid_replay():
    with pytest.raises(ValueError, match="requires"):
        simulate_energy_twin(trace().drop(columns="soc_before_mwh"), TwinConfig())
    frame = trace()
    frame.loc[1, "charge_mw"] = -1.
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_energy_twin(frame, TwinConfig())


def test_source_continuity_and_terminal_deficit_remain_visible():
    frame = trace()
    frame.loc[10:, ["soc_before_mwh", "storage_soc_mwh"]] = 16000.
    result = simulate_energy_twin(frame, TwinConfig())
    assert not result.metrics["source_audit_passed"]
    assert result.metrics["max_source_continuity_error_mwh"] == 1000.
    assert result.metrics["health_score"] < 95


def test_feature_count_cannot_replace_sample_count():
    tiny = pd.DataFrame({"net_load_mw": [100.] * 8, "residual_mw": [0.] * 8,
                         "controlled_net_load_mw": [100.] * 8})
    with pytest.raises(ValueError, match="small"):
        EnergyTwinDriftMonitor().set_baseline(tiny)


def test_missing_values_and_disordered_windows_are_not_silently_dropped():
    frame = trace()
    frame.loc[2, "net_load_mw"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        EnergyTwinDriftMonitor().set_baseline(frame[["timestamp", "net_load_mw"]])
    with pytest.raises(ValueError, match="timestamps"):
        EnergyTwinDriftMonitor().set_baseline(trace().iloc[::-1])


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        EnergyTwinDriftMonitor(DriftThresholds(mean_relative_change=np.nan))


def test_consistent_energy_trace_still_fails_source_power_audit():
    frame = trace()
    frame.loc[0, "charge_mw"] = 1500.
    frame.loc[0, "storage_soc_mwh"] = 16380.
    frame.loc[1:, ["soc_before_mwh", "storage_soc_mwh"]] = 16380.
    result = simulate_energy_twin(frame, TwinConfig())
    assert result.metrics["max_source_physics_residual_mwh"] < 1e-6
    assert result.metrics["max_source_bound_error"] == 250.
    assert not result.metrics["source_audit_passed"]


def test_source_soc_bounds_and_load_accounting_are_audited():
    frame = trace()
    frame.loc[10:, ["soc_before_mwh", "storage_soc_mwh"]] = 31000.
    result = simulate_energy_twin(frame, TwinConfig())
    assert result.metrics["max_source_bound_error"] == 1000.
    assert not result.metrics["source_audit_passed"]
    frame = trace()
    frame["controlled_net_load_mw"] = 999.
    result = simulate_energy_twin(frame, TwinConfig())
    assert result.metrics["max_source_load_error_mw"] == 899.
    assert not result.metrics["source_audit_passed"]
