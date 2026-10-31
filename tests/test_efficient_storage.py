import numpy as np
import pandas as pd
import pytest

from src.efficient_storage import ControlConfig, simulate, solve_window
from src.optimality import StorageModel


def model():
    return StorageModel(capacity_mwh=200.,charge_power_mw=20.,discharge_power_mw=20.,
        efficiency=.9,minimum_soc_mwh=30.,initial_soc_mwh=100.,maximum_soc_mwh=100.)


def panel(days=1):
    f = pd.DataFrame({"timestamp":pd.date_range("2020-01-01",periods=24*days,freq="h",tz="UTC"),"net_load_mw":110.})
    for h in range(1,7):
        f[f"prediction_h{h}_mw"] = 110.
        f[f"upper_h{h}_mw"] = 115.
    return f


def test_tolerance_reduces_throughput_with_explicit_peak_budget():
    m = model()
    p = np.array([200.,80.])
    args = (p,p+5.,100.,np.array([30.,100.]),np.array([100.,100.]),m,100.)
    c0,d0,i0 = solve_window(*args,0.)
    c1,d1,i1 = solve_window(*args,10.)
    assert i1["stage1_peak_mw"] == pytest.approx(i0["stage1_peak_mw"])
    assert i1["stage2_peak_mw"] <= i0["stage1_peak_mw"]+10.+1e-7
    assert (c1+d1).sum() < (c0+d0).sum()
    assert np.minimum(c1,d1).max() == pytest.approx(0.)


def test_new_control_is_causal_restores_soc_and_preserves_hard_constraints():
    f = panel()
    a,s = simulate(f,100.,30.,150.,model=model(),config=ControlConfig(5.))
    assert s["physical_audit_passed"]
    f.net_load_mw = np.arange(24)*10000.
    b,_ = simulate(f,100.,30.,150.,model=model(),config=ControlConfig(5.))
    assert np.allclose(a.charge_mw,b.charge_mw)
    assert np.allclose(a.discharge_mw,b.discharge_mw)


def test_continuous_control_carries_state_across_midnight():
    f = panel(2)
    a,s = simulate(f,100.,30.,150.,model=model(),config=ControlConfig(0.,False))
    assert s["physical_audit_passed"]
    assert a.loc[24,"soc_before_mwh"] == pytest.approx(a.loc[23,"storage_soc_mwh"])
    assert a.iloc[-1].storage_soc_mwh == pytest.approx(100.)


def test_zero_charge_power_cannot_spend_terminal_budget():
    m = StorageModel(charge_power_mw=0.)
    a,s = simulate(panel(),100.,30.,150.,model=m)
    assert a.discharge_mw.sum() == pytest.approx(0.)
    assert s["physical_audit_passed"]


def test_incomplete_day_and_invalid_budget_rejected():
    with pytest.raises(ValueError,match="complete UTC days"):
        simulate(panel().iloc[:-1],100.,30.,150.)
    with pytest.raises(ValueError,match="tolerance"):
        simulate(panel(),100.,30.,150.,config=ControlConfig(-1.))
