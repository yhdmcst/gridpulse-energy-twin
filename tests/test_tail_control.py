from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.optimality import StorageModel
from src.tail_control import POLICIES, TailPolicy, canonical_actions, simulate, solve_window
from src.tail_selection import CANDIDATES, SEEDS, select_policy, validate_validation_range

GATES = {"charge_threshold_mw": 90., "peak_threshold_mw": 150., "release_threshold_mw": 180.}
CALIBRATION = {"uncertainty_scale_mw": 20.}


def model():
    return StorageModel(capacity_mwh=200., initial_soc_mwh=100., maximum_soc_mwh=100.,
                        minimum_soc_mwh=30., charge_power_mw=20., discharge_power_mw=20.)


def panel(start="2021-01-01", days=2):
    times = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": times, "issued_timestamp": times - pd.Timedelta(hours=1),
        "net_load_mw": np.where(times.hour < 12, 80., 200.),
    })
    for h in range(1, 7):
        values = np.where((times + pd.Timedelta(hours=h - 1)).hour < 12, 80., 200.)
        frame[f"prediction_h{h}_mw"] = values
        frame[f"lower_h{h}_mw"] = values - 5.
        frame[f"upper_h{h}_mw"] = values + 5.
    return frame


@pytest.mark.parametrize("name", list(POLICIES))
def test_every_policy_is_truth_invariant_and_restores_energy(name):
    before = panel()
    a, ma = simulate(before, CALIBRATION, GATES, 120., POLICIES[name], model())
    before["net_load_mw"] = np.arange(len(before)) * 10000.
    b, mb = simulate(before, CALIBRATION, GATES, 120., POLICIES[name], model())
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert np.allclose(a.discharge_mw, b.discharge_mw)
    assert ma["physical_audit_passed"] and mb["terminal_requirement_passed"]
    assert np.allclose(a.soc_before_mwh.iloc[1:], a.storage_soc_mwh.iloc[:-1], atol=1e-5)
    assert (a.planned_max_mw <= a.window_peak_cap_mw + 1e-5).all()


@pytest.mark.parametrize("efficiency", [.85, .92, 1.])
def test_canonicalization_preserves_soc_and_reduces_load_and_cycles(efficiency):
    charge = np.array([10., 0., 10., 5.])
    discharge = np.array([10., 20., 0., 1.])
    c, d = canonical_actions(charge, discharge, efficiency)
    assert np.allclose(efficiency * c - d / efficiency,
                       efficiency * charge - discharge / efficiency)
    assert (c - d <= charge - discharge + 1e-6).all()
    assert (c + d <= charge + discharge + 1e-6).all()
    assert np.minimum(c, d).max() == 0.


@pytest.mark.parametrize("name", list(POLICIES))
def test_changed_physical_parameters_remain_reachable(name):
    changed = replace(model(), charge_power_mw=14., discharge_power_mw=14., efficiency=.85)
    trace, metrics = simulate(panel(), CALIBRATION, GATES, 120., POLICIES[name], changed)
    assert metrics["terminal_requirement_passed"] and metrics["physical_audit_passed"]
    assert trace.charge_mw.max() <= 14. + 1e-6


def test_negative_forecast_peak_is_not_clamped_to_zero():
    m = replace(model(), charge_power_mw=0., discharge_power_mw=0.)
    c, d, info = solve_window(
        [-50., -100.], [-40., -90.], 100., [100., 100.], [100., 100.],
        m, GATES, 120., POLICIES["tail_dynamic_0"])
    assert info["stage1_peak_mw"] == pytest.approx(-50.)
    assert info["planned_max_mw"] == pytest.approx(-50.)
    assert np.max(c + d) == 0.


def test_tail_refinement_does_not_worsen_fixed_cap_tail_objective():
    m = replace(model(), charge_power_mw=50., discharge_power_mw=50., efficiency=1.)
    point = np.array([200., 175., 100., 50.])
    gates = dict(GATES, peak_threshold_mw=0.)
    lower, upper = np.array([30., 30., 30., 100.]), np.full(4, 100.)
    cp, dp, _ = solve_window(point, point, 100., lower, upper, m, gates, 150., TailPolicy("peak", "physical", 25.))
    ct, dt, info = solve_window(point, point, 100., lower, upper, m, gates, 150., TailPolicy("tail", "physical", 25.))
    peak_objective = np.maximum(point + cp - dp - 150., 0.).mean() + .001 * (cp + dp).sum()
    tail_objective = np.maximum(point + ct - dt - 150., 0.).mean() + .001 * (ct + dt).sum()
    assert tail_objective <= peak_objective + 1e-6
    assert (point + ct - dt).max() <= info["window_peak_cap_mw"] + 1e-6


def test_invalid_or_partial_input_fails_closed():
    with pytest.raises(ValueError, match="complete"):
        simulate(panel().drop(index=5), CALIBRATION, GATES, 120., POLICIES["peak_dynamic"], model())
    changed = panel()
    changed.loc[0, "prediction_h6_mw"] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        simulate(changed, CALIBRATION, GATES, 120., POLICIES["peak_dynamic"], model())
    with pytest.raises(ValueError):
        simulate(panel().iloc[::-1], CALIBRATION, GATES, 120., POLICIES["peak_dynamic"], model())
    with pytest.raises(ValueError):
        simulate(panel(), CALIBRATION, GATES, 160., POLICIES["peak_dynamic"], model())


def selection_rows():
    baseline = {
        "controlled_peak_mw": 200., "controlled_p95_mw": 180.,
        "throughput_mwh": 100., "conversion_loss_mwh": 10.,
        "physical_audit_passed": True, "terminal_requirement_passed": True,
    }
    return {name: {f"data_seed{s}": dict(baseline) for s in SEEDS}
            for name in ("legacy", *CANDIDATES)}


def test_selection_uses_all_data_seeds_ignores_hgb_and_falls_back():
    rows = selection_rows()
    assert select_policy(rows)["selected_policy"] == "legacy"
    for seed in SEEDS:
        rows["tail_dynamic_0"][f"data_seed{seed}"]["controlled_p95_mw"] -= 5.
    expected = select_policy(rows)
    assert expected["selected_policy"] == "tail_dynamic_0"
    rows["tail_dynamic_0"]["hgb"] = {"controlled_p95_mw": 1e9}
    assert select_policy(rows) == expected
    rows["tail_dynamic_0"]["data_seed44"]["controlled_peak_mw"] += 26.
    assert select_policy(rows)["selected_policy"] == "legacy"


def test_validation_range_rejects_both_boundaries_and_sixth_target_crossing():
    validate_validation_range(panel("2019-02-01"))
    for start in ("2018-01-01", "2019-08-09", "2021-01-01"):
        with pytest.raises(ValueError, match="validation"):
            validate_validation_range(panel(start))
