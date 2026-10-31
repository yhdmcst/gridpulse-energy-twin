import copy

import numpy as np
import pandas as pd
import pytest

from src.optimality import (StorageModel, analytic_certificate, audit_trace,
    offline_benchmark, solve_day, verify_benchmark, verify_lp_certificate,
    exact_power_bound_witness,verify_exact_witness)


def data(load):
    return pd.DataFrame({"timestamp":pd.date_range("2021-01-01",periods=24,freq="h",tz="UTC"),
                         "net_load_mw":load})


def test_hand_solvable_peak_and_lexicographic_cost():
    model = StorageModel(capacity_mwh=100.,charge_power_mw=10.,discharge_power_mw=10.,
        efficiency=1.,minimum_soc_mwh=0.,initial_soc_mwh=50.,maximum_soc_mwh=50.)
    frame = data([20.]*12+[100.]+[20.]*11)
    trace,report = offline_benchmark(frame,model)
    assert report["period_optimum_peak_mw"] == pytest.approx(90.)
    assert report["minimum_throughput_at_optimum_peak_mwh"] == pytest.approx(20.)
    assert analytic_certificate(trace,model)["globally_optimal_within_tolerance"]
    assert verify_benchmark(frame,trace,report)
    tampered = copy.deepcopy(report)
    tampered["daily_certificates"][0]["minimum_peak"]["inequality_dual"][12] = 0.
    assert not verify_benchmark(frame,trace,tampered)


def test_flat_load_cannot_improve_peak_by_spending_daily_initial_energy():
    frame = data([100.]*24)
    trace,report = offline_benchmark(frame,StorageModel())
    assert report["period_optimum_peak_mw"] == pytest.approx(100.)
    assert report["minimum_throughput_at_optimum_peak_mwh"] == pytest.approx(0.)
    assert verify_benchmark(frame,trace,report)


def test_lossy_storage_cost_counts_recharge_and_preserves_physical_witness():
    model = StorageModel(capacity_mwh=100.,charge_power_mw=10.,discharge_power_mw=10.,
        efficiency=.8,minimum_soc_mwh=0.,initial_soc_mwh=50.,maximum_soc_mwh=50.)
    frame = data([20.]*12+[100.]+[20.]*11)
    trace,report = offline_benchmark(frame,model)
    assert report["period_optimum_peak_mw"] == pytest.approx(90.)
    assert report["minimum_throughput_at_optimum_peak_mwh"] == pytest.approx(10.+10./.8**2)
    assert audit_trace(trace,model)["passed"]
    trace.loc[1,"soc_before_mwh"] -= 1.
    assert not audit_trace(trace,model)["passed"]


def test_dual_certificate_tampering_and_incomplete_day_rejected():
    frame = data(np.arange(24)+100.)
    model = StorageModel()
    cert,_,_ = solve_day(frame.net_load_mw,model)
    assert verify_lp_certificate(frame.net_load_mw,model,cert)["passed"]
    cert["x"][0] = -1.
    assert not verify_lp_certificate(frame.net_load_mw,model,cert)["passed"]
    with pytest.raises(ValueError,match="24 hours"):
        offline_benchmark(frame.iloc[:-1],model)


def test_period_optimum_is_max_daily_optimum_cost_uses_period_cap():
    model = StorageModel(capacity_mwh=100.,charge_power_mw=10.,discharge_power_mw=10.,
        efficiency=1.,minimum_soc_mwh=0.,initial_soc_mwh=50.,maximum_soc_mwh=50.)
    one = data([20.]*12+[100.]+[20.]*11)
    two = data([20.]*12+[80.]+[20.]*11)
    two.timestamp += pd.Timedelta(days=1)
    frame = pd.concat([one,two],ignore_index=True)
    trace,report = offline_benchmark(frame,model)
    assert report["period_optimum_peak_mw"] == pytest.approx(90.)
    assert report["minimum_throughput_at_optimum_peak_mwh"] == pytest.approx(20.)
    assert trace.iloc[24:].discharge_mw.sum() == pytest.approx(0.)
    assert verify_benchmark(frame,trace,report)


def test_exact_rational_peak_throughput_and_tampered_witness():
    model = StorageModel(capacity_mwh=100.,charge_power_mw=10.,discharge_power_mw=10.,
        efficiency=.8,minimum_soc_mwh=0.,initial_soc_mwh=50.,maximum_soc_mwh=50.)
    frame = data([20.]*12+[100.]+[20.]*11)
    loads = [str(x) for x in frame.net_load_mw]
    cert = exact_power_bound_witness(frame.timestamp,loads,model)
    assert cert["passed"]
    assert cert["throughput_lower_bound_mwh"] == "205/8"
    assert verify_exact_witness(frame.timestamp,loads,model,cert)
    cert["actions"][0]["discharge_mw"] = "9"
    assert not verify_exact_witness(frame.timestamp,loads,model,cert)


def test_exact_bound_does_not_claim_attainment_when_terminal_recovery_impossible():
    frame = data([20.]*23+[100.])
    model = StorageModel(capacity_mwh=100.,charge_power_mw=10.,discharge_power_mw=10.,
        efficiency=1.,minimum_soc_mwh=0.,initial_soc_mwh=50.,maximum_soc_mwh=50.)
    cert = exact_power_bound_witness(frame.timestamp,[str(x) for x in frame.net_load_mw],model)
    assert not cert["passed"]
