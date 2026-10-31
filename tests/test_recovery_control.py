from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.forecast_dispatch import dispatch
from src.optimality import StorageModel
from src.recovery_control import POLICIES, REFERENCE_COLUMNS, simulate, solve_window, validate_reference
from src.recovery_selection import CANDIDATES, SEEDS, select_policy

GATES = {"charge_threshold_mw": 90., "peak_threshold_mw": 150., "release_threshold_mw": 180.}
CALIBRATION = {"uncertainty_scale_mw": 20.}


def model():
    return StorageModel(capacity_mwh=200., initial_soc_mwh=100., maximum_soc_mwh=100.,
                        minimum_soc_mwh=30., charge_power_mw=20., discharge_power_mw=20.)


def panel():
    times = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    data = pd.DataFrame({
        "timestamp": times, "issued_timestamp": times - pd.Timedelta(hours=1),
        "net_load_mw": np.where(times.hour < 12, 80., 200.),
    })
    for h in range(1, 7):
        values = np.where((times + pd.Timedelta(hours=h - 1)).hour < 12, 80., 200.)
        data[f"prediction_h{h}_mw"] = values
        data[f"lower_h{h}_mw"] = values - 5.
        data[f"upper_h{h}_mw"] = values + 5.
    return data


def reference(data, physical=None):
    return dispatch(data, CALIBRATION, GATES, physical or model())[0]


@pytest.mark.parametrize("name", list(POLICIES))
def test_actions_ignore_actual_load_and_obey_reference_constraints(name):
    data = panel()
    ref = reference(data)
    a, metrics = simulate(data, ref, CALIBRATION, GATES, 120., POLICIES[name], model())
    data["net_load_mw"] = np.arange(len(data)) * 10000.
    b, _ = simulate(data, ref, CALIBRATION, GATES, 120., POLICIES[name], model())
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert np.allclose(a.discharge_mw, b.discharge_mw)
    assert metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]
    if POLICIES[name].debt:
        assert metrics["max_reference_debt_mwh"] <= 1e-5
        assert (a.storage_soc_mwh >= a.reference_soc_after_mwh - 1e-5).all()
    if POLICIES[name].headroom:
        expected = np.maximum(ref.charge_mw - ref.discharge_mw, 120. - data.upper_h1_mw)
        assert np.allclose(a.headroom_cap_mw, expected)
        assert (a.charge_mw - a.discharge_mw <= expected + 1e-5).all()
    assert np.allclose(a.soc_before_mwh.iloc[1:], a.storage_soc_mwh.iloc[:-1], atol=1e-5)


@pytest.mark.parametrize("name", list(POLICIES))
def test_future_issued_forecasts_and_reference_changes_cannot_change_prefix(name):
    data = panel()
    ref = reference(data)
    a, _ = simulate(data, ref, CALIBRATION, GATES, 120., POLICIES[name], model())
    modified = data.copy()
    for h in range(1, 7):
        for prefix in ("prediction", "lower", "upper"):
            modified.loc[13:, f"{prefix}_h{h}_mw"] += 20.
    new_ref = reference(modified)
    assert np.allclose(ref.charge_mw.iloc[:13], new_ref.charge_mw.iloc[:13])
    b, _ = simulate(modified, new_ref, CALIBRATION, GATES, 120., POLICIES[name], model())
    assert np.allclose(a.charge_mw.iloc[:13], b.charge_mw.iloc[:13])
    assert np.allclose(a.discharge_mw.iloc[:13], b.discharge_mw.iloc[:13])


@pytest.mark.parametrize("efficiency,power", [(.85, 20.), (.92, 14.)])
@pytest.mark.parametrize("name", ["tail_debt", "tail_guard", "peak_debt", "peak_guard"])
def test_actual_parameter_references_preserve_reachability(efficiency, power, name):
    physical = replace(model(), efficiency=efficiency, charge_power_mw=power, discharge_power_mw=power)
    data = panel()
    ref = reference(data, physical)
    _, metrics = simulate(data, ref, CALIBRATION, GATES, 120., POLICIES[name], physical)
    assert metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]
    assert metrics["max_reference_debt_mwh"] <= 1e-5


def test_reference_strips_outcomes_and_rejects_invalid_physics_or_alignment():
    data = panel()
    ref = reference(data)
    assert tuple(validate_reference(ref, data.timestamp, model()).columns) == REFERENCE_COLUMNS
    modified = ref.copy()
    modified.net_load_mw = 1e9
    modified.controlled_net_load_mw = -1e9
    a, _ = simulate(data, ref, CALIBRATION, GATES, 120., POLICIES["tail_guard"], model())
    b, _ = simulate(data, modified, CALIBRATION, GATES, 120., POLICIES["tail_guard"], model())
    assert np.allclose(a.charge_mw, b.charge_mw)
    with pytest.raises(ValueError):
        validate_reference(ref.iloc[1:], data.timestamp, model())
    modified.loc[0, "charge_mw"] += 1.
    with pytest.raises(ValueError, match="physical"):
        validate_reference(modified, data.timestamp, model())
    with pytest.raises(ValueError, match="physical"):
        validate_reference(ref, data.timestamp, replace(model(), efficiency=.85))


def test_higher_candidate_soc_can_clip_reference_charge_without_free_energy():
    physical = replace(model(), efficiency=1.)
    c, d, info = solve_window(
        [200.], [205.], 95., [100.], [100.], physical, GATES, 120.,
        POLICIES["tail_guard"], reference_after=100., reference_net=20.)
    assert c[0] == pytest.approx(5.) and d[0] == pytest.approx(0.)
    assert info["required_next_soc_mwh"] == 100.
    assert c[0] - d[0] <= info["headroom_cap_mw"]


@pytest.mark.parametrize("name", list(POLICIES))
def test_negative_net_forecasts_are_not_zero_clamped(name):
    physical = replace(model(), charge_power_mw=0., discharge_power_mw=0.)
    c, d, info = solve_window(
        [-50., -100.], [-45., -95.], 100., [100., 100.], [100., 100.],
        physical, GATES, 120., POLICIES[name], reference_after=100., reference_net=0.)
    assert info["planned_max_mw"] == pytest.approx(-50.)
    assert (c + d).max() == 0.


def test_admission_keeps_v4_per_seed_guards_and_hgb_is_not_used():
    base = {
        "controlled_peak_mw": 200., "controlled_p95_mw": 180.,
        "throughput_mwh": 100., "conversion_loss_mwh": 10.,
        "physical_audit_passed": True, "terminal_requirement_passed": True,
    }
    rows = {name: {f"data_seed{s}": dict(base) for s in SEEDS} for name in ("legacy", *CANDIDATES)}
    assert select_policy(rows)["selected_policy"] == "legacy"
    for seed in SEEDS:
        rows["tail_guard"][f"data_seed{seed}"]["controlled_p95_mw"] -= 2.
    assert select_policy(rows)["selected_policy"] == "tail_guard"
    rows["tail_guard"]["hgb"] = {"controlled_p95_mw": 1e9}
    assert select_policy(rows)["selected_policy"] == "tail_guard"
    rows["tail_guard"]["data_seed43"]["controlled_peak_mw"] += 25.1
    assert select_policy(rows)["selected_policy"] == "legacy"
