"""Post-hoc two-stage causal storage controller, separate from frozen policies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.balanced_storage import complete_utc_days
from src.optimality import StorageModel
from src.scenarios import dynamic_reserve_fraction


@dataclass(frozen=True)
class ControlConfig:
    peak_tolerance_mw: float = 0.
    daily_terminal: bool = True


def solve_window(point, upper, soc, lower_soc, upper_soc, model, threshold, peak_tolerance):
    """Min peak then min throughput at peak <= z_min + declared tolerance.

    Retain original forecast-based discharge gates. Removing simultaneous
    actions preserves SOC, reduces net load, and respects these zero/max gates.
    """
    n = len(point)
    arrays = [np.asarray(v,dtype=float) for v in (point,upper,lower_soc,upper_soc)]
    if n == 0 or any(v.shape != (n,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Finite matching window arrays required.")
    point,upper,lower_soc,upper_soc = arrays
    if not np.isfinite([soc,threshold,peak_tolerance]).all() or peak_tolerance < 0:
        raise ValueError("Finite state and nonnegative peak tolerance required.")
    model.validate()
    size = 2*n+1
    energy = np.zeros((n,size))
    energy[:,:n] = model.efficiency*np.tril(np.ones((n,n)))
    energy[:,n:2*n] = -np.tril(np.ones((n,n)))/model.efficiency
    peak = np.zeros((n,size))
    peak[:,:n],peak[:,n:2*n],peak[:,-1] = np.eye(n),-np.eye(n),-1.
    a = np.vstack([peak,energy,-energy])
    b = np.r_[-point,upper_soc-soc,soc-lower_soc]
    bounds = ([(0.,model.charge_power_mw)]*n
              +[(0.,model.discharge_power_mw if value >= threshold else 0.) for value in upper]
              +[(None,None)])
    obj = np.zeros(size)
    obj[-1] = 1.
    first = linprog(obj,A_ub=a,b_ub=b,bounds=bounds,method="highs")
    if not first.success:
        raise RuntimeError(f"Stage 1 infeasible: {first.message}")
    cap = float(first.fun)+peak_tolerance
    bounds[-1] = (None,cap)
    obj[:2*n],obj[-1] = 1.,0.
    second = linprog(obj,A_ub=a,b_ub=b,bounds=bounds,method="highs")
    if not second.success:
        raise RuntimeError(f"Stage 2 infeasible: {second.message}")
    increment = model.efficiency*second.x[:n]-second.x[n:2*n]/model.efficiency
    c,d = np.maximum(increment,0.)/model.efficiency,np.maximum(-increment,0.)*model.efficiency
    planned_soc = soc+np.cumsum(model.efficiency*c-d/model.efficiency)
    if (np.any(planned_soc < lower_soc-1e-5) or np.any(planned_soc > upper_soc+1e-5)
            or np.max(point+c-d) > cap+1e-5 or np.max(c) > model.charge_power_mw+1e-5
            or np.max(d) > model.discharge_power_mw+1e-5
            or np.any(d[upper < threshold] > 1e-5)):
        raise RuntimeError("Two-stage physical plan failed independent feasibility checks.")
    return c,d,{"stage1_peak_mw":float(first.fun),"stage2_peak_mw":float(np.max(point+c-d)),
                "peak_cap_mw":cap,"planned_throughput_mwh":float((c+d).sum())}


def simulate(panel: pd.DataFrame, peak_threshold_mw: float, uncertainty_scale_mw: float,
             release_threshold_mw: float, *, model=StorageModel(),config=ControlConfig()):
    model.validate()
    if not np.isfinite([peak_threshold_mw,uncertainty_scale_mw,release_threshold_mw,config.peak_tolerance_mw]).all():
        raise ValueError("Finite controller parameters required.")
    if uncertainty_scale_mw <= 0 or release_threshold_mw < peak_threshold_mw or config.peak_tolerance_mw < 0:
        raise ValueError("Invalid calibration or tolerance.")
    data,omitted = complete_utc_days(panel)
    if omitted:
        raise ValueError("Development comparisons require complete days; omissions must be prepared explicitly.")
    if not config.daily_terminal and not data.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Continuous mode requires a contiguous sequence.")
    horizons = sorted(int(c.removeprefix("prediction_h").removesuffix("_mw"))
                      for c in data if c.startswith("prediction_h") and c.endswith("_mw"))
    if not horizons or horizons != list(range(1,len(horizons)+1)):
        raise ValueError("Contiguous prediction horizons required.")
    cols = [f"{prefix}_h{h}_mw" for h in horizons for prefix in ("prediction","upper")]
    if not np.isfinite(data[cols+["net_load_mw"]].to_numpy(float)).all():
        raise ValueError("Finite forecasts and evaluation loads required.")
    if any((data[f"prediction_h{h}_mw"] > data[f"upper_h{h}_mw"]).any() for h in horizons):
        raise ValueError("Upper forecasts must bound point forecasts.")
    soc = model.initial_soc_mwh
    rows = []
    for i,row in enumerate(data.itertuples(index=False)):
        if config.daily_terminal and row.timestamp.hour == 0:
            soc = model.initial_soc_mwh
        remaining = 24-row.timestamp.hour if config.daily_terminal else len(data)-i
        steps = min(len(horizons),remaining)
        point = np.array([getattr(row,f"prediction_h{h}_mw") for h in range(1,steps+1)])
        upper = np.array([getattr(row,f"upper_h{h}_mw") for h in range(1,steps+1)])
        target = model.capacity_mwh*dynamic_reserve_fraction(upper,upper-point,peak_threshold_mw,
                    uncertainty_scale_mw=uncertainty_scale_mw)
        released = point.max() >= release_threshold_mw
        floor = model.minimum_soc_mwh if released else max(model.minimum_soc_mwh,min(target,soc))
        after = remaining-np.arange(1,steps+1)
        lb = np.maximum(floor,model.initial_soc_mwh-after*model.charge_power_mw*model.efficiency)
        ub = np.minimum(model.maximum_soc_mwh,model.initial_soc_mwh+after*model.discharge_power_mw/model.efficiency)
        c,d,info = solve_window(point,upper,soc,lb,ub,model,peak_threshold_mw,config.peak_tolerance_mw)
        before = soc
        soc += model.efficiency*c[0]-d[0]/model.efficiency
        rows.append({"timestamp":row.timestamp,"net_load_mw":row.net_load_mw,
            "charge_mw":float(c[0]),"discharge_mw":float(d[0]),"soc_before_mwh":before,
            "storage_soc_mwh":soc,"controlled_net_load_mw":row.net_load_mw+c[0]-d[0],
            "reserve_target_mwh":target,"reserve_release_active":bool(released),
            "prediction_h1_mw":point[0],"upper_h1_mw":upper[0],"solver_status":"two_stage_optimal",**info})
    trace = pd.DataFrame(rows)
    summary = summarize(trace,model,daily_terminal=config.daily_terminal)
    if not summary["physical_audit_passed"]:
        raise RuntimeError("Executed development trajectory is physically infeasible.")
    return trace,summary


def summarize(trace,model,*,daily_terminal=True):
    c,d = trace.charge_mw,trace.discharge_mw
    before,after = trace.soc_before_mwh,trace.storage_soc_mwh
    dates = pd.to_datetime(trace.timestamp,utc=True)
    reset = dates.dt.hour.eq(0).to_numpy() if daily_terminal else np.arange(len(trace)) == 0
    expected = np.where(reset,model.initial_soc_mwh,np.roll(after.to_numpy(),1))
    terminal = dates.dt.hour.eq(23).to_numpy() if daily_terminal else np.arange(len(trace)) == len(trace)-1
    energy = float((after-before-model.efficiency*c+d/model.efficiency).abs().max())
    terminal_error = float((after[terminal]-model.initial_soc_mwh).abs().max())
    continuity = float(np.max(np.abs(before-expected)))
    bounds = float(max(0.,model.minimum_soc_mwh-min(before.min(),after.min()),
        max(before.max(),after.max())-model.maximum_soc_mwh,-c.min(),-d.min(),
        c.max()-model.charge_power_mw,d.max()-model.discharge_power_mw,np.minimum(c,d).max()))
    load_error = float((trace.controlled_net_load_mw-trace.net_load_mw-c+d).abs().max())
    return {"rows":len(trace),"days":len(trace)//24,"daily_terminal":daily_terminal,
        "baseline_peak_mw":float(trace.net_load_mw.max()),"controlled_peak_mw":float(trace.controlled_net_load_mw.max()),
        "controlled_p95_mw":float(trace.controlled_net_load_mw.quantile(.95)),
        "mean_daily_peak_mw":float(trace.groupby(dates.dt.floor("D")).controlled_net_load_mw.max().mean()),
        "throughput_mwh":float((c+d).sum()),"conversion_loss_mwh":float(((1-model.efficiency)*c+(1/model.efficiency-1)*d).sum()),
        "energy_balance_error_mwh":energy,"continuity_error_mwh":continuity,"bound_error":bounds,
        "max_abs_terminal_error_mwh":terminal_error,"load_accounting_error_mw":load_error,
        "physical_audit_passed":max(energy,continuity,bounds,terminal_error,load_error) <= 1e-5,
        "reserve_target_shortfall_hours":int((after < trace.reserve_target_mwh-1e-5).sum()) if "reserve_target_mwh" in trace else None}
