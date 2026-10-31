from dataclasses import replace
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.delayed_forecasts import common_panels
from src.feedback_control import (
    Packet, RecoveryEnvelope, actuate, apply_supervisor, candidate_decision, event_flags,
    make_reference, packet_gate, recovery_request, reference_decision, simulate_feedback,
)
from src.forecast_dispatch import dispatch
from src.optimality import StorageModel
from src.robustness_control import simulate_guard

GATES = {"peak_threshold_mw": 150., "charge_threshold_mw": 90., "release_threshold_mw": 180.}
CAL = {"radii": {str(h): {str(hour): float(h) for hour in range(24)} for h in range(1, 7)},
       "uncertainty_scale_mw": 20.}
ENV = RecoveryEnvelope(charge_power_mw=14., efficiency=.85)


def model():
    return StorageModel(capacity_mwh=200., initial_soc_mwh=100., maximum_soc_mwh=100.,
                        minimum_soc_mwh=30., charge_power_mw=20., discharge_power_mw=20.)


def panels():
    times = pd.date_range("2021-01-01", periods=120, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "issued_timestamp": times - pd.Timedelta(hours=1)})
    for h in range(1, 7):
        actual = np.where((times + pd.Timedelta(hours=h - 1)).hour < 12, 80., 200.)
        frame[f"actual_h{h}_mw"] = actual
        frame[f"prediction_h{h}_mw"] = actual + h * .01
    return common_panels({"data_seed42": frame}, {"methods": {"data_seed42": CAL}})[0]["data_seed42"]


def reference(p, event="healthy"):
    return make_reference(p["fresh6"], p["delayed3"], event, CAL, GATES, 120.,
                          assumed=model(), envelope=ENV)[0]


def packet(hour=23, expired=False):
    t = pd.Timestamp("2021-01-02T00:00:00Z") + pd.Timedelta(hours=hour)
    origin = t - pd.Timedelta(hours=3 if expired else 0)
    return Packet(t, origin, origin - pd.Timedelta(hours=1), t,
                  np.array([200., 200., 200.]), np.array([195., 195., 195.]),
                  np.array([205., 205., 205.]))


def anchor():
    return SimpleNamespace(soc_before_mwh=90., storage_soc_mwh=90., charge_mw=0.,
                           discharge_mw=0., solver_failed=False)


def test_nominal_feedback_reproduces_frozen_window_controller():
    p = panels()
    ref = reference(p)
    old_ref, _ = dispatch(p["fresh6"], CAL, GATES, model())
    for column in ("charge_mw", "discharge_mw", "storage_soc_mwh"):
        assert np.allclose(ref[column], old_ref[column], rtol=0, atol=1e-7)
    old, _ = simulate_guard(p["fresh6"], old_ref, CAL, GATES, 120., model())
    new, metrics = simulate_feedback(
        p["fresh6"], p["delayed3"], ref, "healthy", model(), False, CAL, GATES, 120.,
        assumed=model(), envelope=ENV)
    for column in ("charge_mw", "discharge_mw", "storage_soc_mwh", "controlled_net_load_mw"):
        assert np.allclose(old[column], new[column], rtol=0, atol=1e-7)
    assert metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]


@pytest.mark.parametrize("supervised", [False, True])
def test_load_truth_does_not_enter_decisions(supervised):
    p = panels()
    ref = reference(p)
    physical = replace(model(), efficiency=.90)
    a, _ = simulate_feedback(p["fresh6"], p["delayed3"], ref, "healthy", physical, supervised,
                            CAL, GATES, 120., assumed=model(), envelope=ENV)
    base, old = p["fresh6"].copy(), p["delayed3"].copy()
    base.net_load_mw = old.net_load_mw = np.arange(len(base)) * 100000.
    b, _ = simulate_feedback(base, old, ref, "healthy", physical, supervised,
                            CAL, GATES, 120., assumed=model(), envelope=ENV)
    assert np.allclose(a.requested_charge_mw, b.requested_charge_mw)
    assert np.allclose(a.requested_discharge_mw, b.requested_discharge_mw)


@pytest.mark.parametrize("supervised", [False, True])
def test_unknown_true_parameters_do_not_enter_first_nonzero_request(supervised):
    assert "physical" not in inspect.signature(candidate_decision).parameters
    p, nominal = panels(), model()
    ref = reference(p)
    a, _ = simulate_feedback(p["fresh6"], p["delayed3"], ref, "healthy", nominal, supervised,
                            CAL, GATES, 120., assumed=nominal, envelope=ENV)
    b, _ = simulate_feedback(p["fresh6"], p["delayed3"], ref, "healthy",
                            replace(nominal, efficiency=.90), supervised, CAL, GATES, 120.,
                            assumed=nominal, envelope=ENV)
    first = np.flatnonzero((a.requested_charge_mw + a.requested_discharge_mw).to_numpy() > 1e-6)[0]
    assert np.allclose(a.requested_charge_mw.iloc[:first + 1], b.requested_charge_mw.iloc[:first + 1])
    assert np.allclose(a.requested_discharge_mw.iloc[:first + 1], b.requested_discharge_mw.iloc[:first + 1])
    assert not np.allclose(a.storage_soc_mwh, b.storage_soc_mwh)
    assert np.allclose(b.measured_soc_mwh, b.soc_before_mwh)
    assert np.allclose(b.soc_before_mwh.iloc[1:], b.storage_soc_mwh.iloc[:-1])


@pytest.mark.parametrize("supervised", [False, True])
def test_future_packet_changes_do_not_change_prior_reference_or_candidate(supervised):
    p = panels()
    ref = reference(p)
    a, _ = simulate_feedback(p["fresh6"], p["delayed3"], ref, "healthy", model(), supervised,
                            CAL, GATES, 120., assumed=model(), envelope=ENV)
    changed = {k: v.copy() for k, v in p.items()}
    for name in ("fresh6", "delayed3"):
        for column in changed[name]:
            if column.startswith(("prediction_h", "lower_h", "upper_h")):
                changed[name].loc[36:, column] += 30.
    new_ref = reference(changed)
    b, _ = simulate_feedback(changed["fresh6"], changed["delayed3"], new_ref, "healthy",
                            model(), supervised, CAL, GATES, 120., assumed=model(), envelope=ENV)
    assert np.allclose(ref.storage_soc_mwh.iloc[:36], new_ref.storage_soc_mwh.iloc[:36])
    assert np.allclose(a.requested_charge_mw.iloc[:36], b.requested_charge_mw.iloc[:36])
    assert np.allclose(a.requested_discharge_mw.iloc[:36], b.requested_discharge_mw.iloc[:36])


@pytest.mark.parametrize("expired,outage", [(True, False), (False, True), (True, True)])
def test_gates_bypass_both_optimizers_and_never_reuse_discharge(expired, outage):
    p = packet(expired=expired)
    with patch("src.feedback_control._solve_storage_window", side_effect=AssertionError("must not call")):
        c, d, info = reference_decision(p, 90., model(), CAL, GATES, ENV, outage)
    assert d == 0. and c > 0. and not info["solver_attempted"]
    with patch("src.feedback_control.solve_window", side_effect=AssertionError("must not call")):
        c, d, info = candidate_decision(p, 90., anchor(), model(), CAL, GATES, 120., ENV, False, outage)
    assert d == 0. and c > 0. and info["fallback_active"] and not info["primary_solver_attempted"]


def test_natural_solver_failure_is_logged_and_programming_error_is_not_hidden():
    with patch("src.feedback_control.solve_window", side_effect=RuntimeError("infeasible probe")):
        _, d, info = candidate_decision(packet(), 90., anchor(), model(), CAL, GATES, 120., ENV, False)
    assert d == 0. and info["primary_solver_failed"] and "infeasible probe" in info["primary_solver_error"]
    with patch("src.feedback_control.solve_window", side_effect=ValueError("programming probe")):
        with pytest.raises(ValueError, match="programming"):
            candidate_decision(packet(), 90., anchor(), model(), CAL, GATES, 120., ENV, False)


def test_soc_telemetry_changes_recovery_and_unreachable_recovery_is_explicit():
    c_low, _, low = recovery_request(80., packet().timestamp, model(), ENV)
    c_high, _, high = recovery_request(98., packet().timestamp, model(), ENV)
    assert c_low > c_high
    assert low["recovery_unreachable"] and not high["recovery_unreachable"]
    with pytest.raises(ValueError):
        recovery_request(np.nan, packet().timestamp, model(), ENV)


def test_supervisor_invariant_over_declared_plant_envelope():
    m = model()
    for hour in range(24):
        t = packet(hour).timestamp
        floor_before = max(m.minimum_soc_mwh, m.initial_soc_mwh - (24 - hour) * 14. * .85)
        for soc in (floor_before, (floor_before + 100.) / 2, 100.):
            for req_c, req_d in ((0., 0.), (5., 0.), (0., 20.)):
                c, d, info = apply_supervisor(soc, t, req_c, req_d, m, ENV)
                assert not info["recovery_unreachable"]
                assert info["worst_case_next_soc_mwh"] >= info["safe_floor_next_mwh"] - 1e-6
                for power, eta in ((14., .85), (18., .90), (20., .92)):
                    _, _, actual = actuate(soc, c, d, replace(m, charge_power_mw=power,
                                                             discharge_power_mw=power, efficiency=eta))
                    assert actual >= info["safe_floor_next_mwh"] - 1e-6


@pytest.mark.parametrize("event", ["stale_burst", "solver_outage", "combined_fault"])
def test_fault_schedule_and_supervised_continuity(event):
    p = panels()
    ref = reference(p, event)
    trace, metrics = simulate_feedback(
        p["fresh6"], p["delayed3"], ref, event,
        replace(model(), charge_power_mw=18., discharge_power_mw=18., efficiency=.90),
        True, CAL, GATES, 120., assumed=model(), envelope=ENV)
    assert metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]
    assert metrics["gated_optimizer_calls"] == 0
    assert metrics["max_actual_floor_violation_mwh"] <= 1e-5
    assert np.allclose(trace.soc_before_mwh.iloc[1:], trace.storage_soc_mwh.iloc[:-1])
    assert np.allclose(trace.measured_soc_mwh, trace.soc_before_mwh)
    for i, row in enumerate(trace.itertuples(index=False)):
        stale, outage = event_flags(i, row.timestamp, event)
        assert row.injected_stale == stale and row.injected_outage == outage
        if stale or outage:
            assert row.fallback_active and not row.primary_solver_attempted
            assert row.requested_discharge_mw == 0.


def test_outside_envelope_terminal_failure_remains_possible():
    c, d, info = apply_supervisor(90., packet().timestamp, 0., 0., model(), ENV)
    assert not info["recovery_unreachable"]
    _, _, actual = actuate(90., c, d, replace(model(), efficiency=.80))
    assert actual < 100. - 1e-5
    assert actual < info["worst_case_next_soc_mwh"]


def test_invalid_envelope_and_invalid_packet_are_rejected():
    with pytest.raises(ValueError):
        RecoveryEnvelope(charge_power_mw=30.).validate(model())
    p = replace(packet(), available_timestamp=packet().timestamp + pd.Timedelta(hours=1))
    assert packet_gate(p, ENV) == "future_packet"
    p = replace(packet(), point=np.array([np.nan, 200., 200.]))
    assert packet_gate(p, ENV) == "invalid_forecast"


def test_packet_cannot_arrive_before_its_observation_interval_finishes():
    p = packet()
    assert packet_gate(replace(p, available_timestamp=p.observed_timestamp), ENV) == "invalid_availability_time"
    assert packet_gate(replace(p, timestamp=p.timestamp + pd.Timedelta(minutes=30)), ENV) == "invalid_packet_time"


def test_recovery_deadline_uses_utc_and_requires_a_valid_hour():
    utc = packet().timestamp
    local = utc.tz_convert("Asia/Shanghai")
    assert recovery_request(98., local, model(), ENV) == recovery_request(98., utc, model(), ENV)
    for invalid in (pd.NaT, utc.tz_localize(None), utc + pd.Timedelta(minutes=30)):
        with pytest.raises(ValueError, match="decision time"):
            recovery_request(98., invalid, model(), ENV)
