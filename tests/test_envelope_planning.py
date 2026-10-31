from copy import deepcopy
from dataclasses import replace
import inspect
from unittest.mock import patch

import numpy as np
import pytest

from src.envelope_planning import POLICIES, planning_model, simulate_policy
from src.envelope_selection import select_policy
from src.feedback_control import make_reference, plant_models
from test_feedback_control import CAL, ENV, GATES, model, panels


def conservative_ref(p, event="healthy"):
    return make_reference(p["fresh6"], p["delayed3"], event, CAL, GATES, 120.,
                          assumed=planning_model(ENV), envelope=ENV)[0]


def test_conservative_model_depends_only_on_declared_bounds():
    m = planning_model(ENV)
    assert m.charge_power_mw == m.discharge_power_mw == ENV.charge_power_mw
    assert m.efficiency == ENV.efficiency
    assert m.initial_soc_mwh == 15000.
    assert "physical" not in inspect.signature(planning_model).parameters


@pytest.mark.parametrize("policy", POLICIES)
def test_truth_invariance_envelope_recovery_and_continuous_feedback(policy):
    # Keep small-test energy bounds while exercising the same conservative composition.
    p = panels()
    lower = replace(model(), charge_power_mw=14., discharge_power_mw=14., efficiency=.85)
    ref = make_reference(p["fresh6"], p["delayed3"], "healthy", CAL, GATES, 120.,
                         assumed=lower, envelope=ENV)[0]
    actual = replace(model(), efficiency=.90)
    with patch("src.envelope_planning.planning_model", return_value=lower):
        a, metrics = simulate_policy(p["fresh6"], p["delayed3"], ref, "healthy", actual,
                                     policy, CAL, GATES, 120., ENV)
        b0, b1 = p["fresh6"].copy(), p["delayed3"].copy()
        b0.net_load_mw = b1.net_load_mw = np.arange(len(b0)) * 10000.
        b, _ = simulate_policy(b0, b1, ref, "healthy", actual, policy, CAL, GATES, 120., ENV)
    assert metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]
    assert metrics["gated_optimizer_calls"] == 0
    assert metrics["supervisor_interventions"] == 0
    assert np.allclose(a.requested_charge_mw, b.requested_charge_mw)
    assert np.allclose(a.requested_discharge_mw, b.requested_discharge_mw)
    assert np.allclose(a.measured_soc_mwh, a.soc_before_mwh)
    assert np.allclose(a.soc_before_mwh.iloc[1:], a.storage_soc_mwh.iloc[:-1])
    assert (a.storage_soc_mwh >= a.predicted_next_soc_mwh - 1e-5).all()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("event", ["stale_burst", "solver_outage", "combined_fault"])
def test_injected_events_and_future_prefix_causality(policy, event):
    p = panels()
    lower = replace(model(), charge_power_mw=14., discharge_power_mw=14., efficiency=.85)
    ref = make_reference(p["fresh6"], p["delayed3"], event, CAL, GATES, 120., assumed=lower, envelope=ENV)[0]
    actual = replace(model(), charge_power_mw=18., discharge_power_mw=18., efficiency=.90)
    with patch("src.envelope_planning.planning_model", return_value=lower):
        a, metrics = simulate_policy(p["fresh6"], p["delayed3"], ref, event, actual, policy, CAL, GATES, 120., ENV)
        changed = {k: v.copy() for k, v in p.items()}
        for name in ("fresh6", "delayed3"):
            for col in changed[name]:
                if col.startswith(("prediction_h", "lower_h", "upper_h")):
                    changed[name].loc[48:, col] += 20.
        ref2 = make_reference(changed["fresh6"], changed["delayed3"], event, CAL, GATES, 120.,
                              assumed=lower, envelope=ENV)[0]
        b, _ = simulate_policy(changed["fresh6"], changed["delayed3"], ref2, event, actual,
                               policy, CAL, GATES, 120., ENV)
    assert metrics["terminal_requirement_passed"]
    assert metrics["gated_optimizer_calls"] == 0
    assert np.allclose(a.requested_charge_mw.iloc[:48], b.requested_charge_mw.iloc[:48])
    assert np.allclose(a.requested_discharge_mw.iloc[:48], b.requested_discharge_mw.iloc[:48])
    blocked = a.injected_stale | a.injected_outage
    assert not a.loc[blocked, "primary_solver_attempted"].any()
    assert a.loc[blocked, "requested_discharge_mw"].eq(0.).all()


def test_selection_guards_are_scenario_specific_and_can_fall_back():
    base = {
        "controlled_peak_mw": 200., "controlled_p95_mw": 180., "throughput_mwh": 100.,
        "conversion_loss_mwh": 10., "max_abs_terminal_error_mwh": 0.,
        "max_actual_floor_violation_mwh": 0., "max_worst_case_floor_violation_mwh": 0.,
        "gated_optimizer_calls": 0, "physical_audit_passed": True, "terminal_requirement_passed": True,
    }
    rows = {p: {c: dict(base) for c in ("nominal", "combined90")} for p in ("baseline_v7", *POLICIES)}
    assert select_policy(rows)["selected_policy"] == "baseline_v7"
    for c in ("nominal", "combined90"):
        rows["envelope_tail"][c]["controlled_p95_mw"] -= 5.
    assert select_policy(rows)["selected_policy"] == "envelope_tail"
    modified = deepcopy(rows)
    modified["envelope_tail"]["combined90"]["controlled_peak_mw"] += 26.
    assert select_policy(modified)["selected_policy"] == "baseline_v7"
    modified["envelope_tail"]["combined90"]["controlled_p95_mw"] = np.nan
    with pytest.raises(ValueError):
        select_policy(modified)
