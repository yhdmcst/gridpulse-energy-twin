"""Offline finite-instance certificates, deliberately separate from frozen controllers.

Only an upper bound on grid load is imposed. Simultaneous LP actions can thus
be removed preserving SOC while reducing grid load and throughput (see docs).
No forecasts, prices, reserve gates or claims about unseen futures are included.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.numerics import same_loads


@dataclass(frozen=True)
class StorageModel:
    capacity_mwh: float = 30000.0
    charge_power_mw: float = 1250.0
    discharge_power_mw: float = 1250.0
    efficiency: float = 0.92
    minimum_soc_mwh: float = 4500.0
    initial_soc_mwh: float = 15000.0
    maximum_soc_mwh: float = 15000.0

    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Finite physical parameters required.")
        if not (0 < self.efficiency <= 1 and self.capacity_mwh > 0
                and 0 <= self.minimum_soc_mwh <= self.initial_soc_mwh
                <= self.maximum_soc_mwh <= self.capacity_mwh
                and min(self.charge_power_mw, self.discharge_power_mw) >= 0):
            raise ValueError("Invalid physical storage parameters.")


def check_days(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
    if data.empty or data.timestamp.isna().any() or data.timestamp.duplicated().any():
        raise ValueError("Nonempty unique valid timestamps required.")
    if not data.timestamp.is_monotonic_increasing:
        raise ValueError("Timestamps must already be ordered.")
    if not np.isfinite(data.net_load_mw.to_numpy(float)).all():
        raise ValueError("Finite loads required.")
    for date, group in data.groupby(data.timestamp.dt.floor("D")):
        if len(group) != 24 or not group.timestamp.equals(
                pd.Series(pd.date_range(date, periods=24, freq="h"), index=group.index, name="timestamp")):
            raise ValueError("Every included UTC day must contain all 24 hours; no silent omissions.")
    return data


def audit_trace(frame: pd.DataFrame, model: StorageModel, tolerance: float = 1e-5) -> dict:
    model.validate()
    data = check_days(frame)
    cols = ["charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh", "controlled_net_load_mw"]
    if not np.isfinite(data[cols].to_numpy(float)).all():
        raise ValueError("Finite actions and SOC required.")
    c, d, before, after, controlled = [data[col].to_numpy(float) for col in cols]
    first = data.timestamp.dt.hour.eq(0).to_numpy()
    last = data.timestamp.dt.hour.eq(23).to_numpy()
    expected_before = np.where(first, model.initial_soc_mwh, np.roll(after, 1))
    residuals = {
        "charge_bound_mw": float(max(0., -c.min(), c.max() - model.charge_power_mw)),
        "discharge_bound_mw": float(max(0., -d.min(), d.max() - model.discharge_power_mw)),
        "soc_bound_mwh": float(max(0., model.minimum_soc_mwh - min(before.min(), after.min()),
                                     max(before.max(), after.max()) - model.maximum_soc_mwh)),
        "soc_continuity_mwh": float(np.max(np.abs(before - expected_before))),
        "energy_balance_mwh": float(np.max(np.abs(after - before - model.efficiency*c + d/model.efficiency))),
        "terminal_error_mwh": float(np.max(np.abs(after[last] - model.initial_soc_mwh))),
        "simultaneous_action_mw": float(np.maximum(0., np.minimum(c, d)).max()),
        "load_accounting_mw": float(np.max(np.abs(controlled - data.net_load_mw.to_numpy(float) - c + d))),
    }
    return {"passed": max(residuals.values()) <= tolerance, "tolerance": tolerance,
            "residuals": residuals, "days": int(first.sum()), "rows": len(data)}


def analytic_certificate(frame: pd.DataFrame, model: StorageModel) -> dict:
    audit = audit_trace(frame, model)
    lower = float(frame.net_load_mw.max() - model.discharge_power_mw)
    achieved = float(frame.controlled_net_load_mw.max())
    return {"method": "power_bound_and_feasible_attainment", "audit": audit,
            "lower_bound_mw": lower, "achieved_peak_mw": achieved,
            "absolute_gap_mw": achieved-lower,
            "globally_optimal_within_tolerance": bool(audit["passed"] and abs(achieved-lower) <= 1e-5),
            "scope": "This finite load sequence, stated physical model, full-period maximum objective only."}


def lp_problem(load: np.ndarray, model: StorageModel, *, peak_cap: float | None = None):
    """Hourly MW and MWh coincide only because dt=1 hour; n must be 24."""
    model.validate()
    load = np.asarray(load, dtype=float)
    if load.shape != (24,) or not np.isfinite(load).all():
        raise ValueError("One complete finite 24-hour load vector required.")
    n = len(load)
    size = 2*n + (peak_cap is None)
    objective = np.zeros(size)
    if peak_cap is None:
        objective[-1] = 1.
    else:
        if not np.isfinite(peak_cap):
            raise ValueError("Finite peak cap required.")
        objective[:] = 1.
    energy = np.zeros((n, size))
    energy[:, :n] = model.efficiency*np.tril(np.ones((n,n)))
    energy[:, n:2*n] = -np.tril(np.ones((n,n)))/model.efficiency
    peak = np.zeros((n, size))
    peak[:, :n], peak[:, n:2*n] = np.eye(n), -np.eye(n)
    if peak_cap is None:
        peak[:, -1] = -1.
    a = np.vstack([peak, energy, -energy])
    b = np.r_[-load if peak_cap is None else peak_cap-load,
              np.full(n, model.maximum_soc_mwh-model.initial_soc_mwh),
              np.full(n, model.initial_soc_mwh-model.minimum_soc_mwh)]
    eq = energy[-1:]
    rhs = np.array([0.])
    bounds = [(0.,model.charge_power_mw)]*n + [(0.,model.discharge_power_mw)]*n
    if peak_cap is None:
        bounds.append((None,None))
    return objective, a, b, eq, rhs, bounds


def verify_lp_certificate(load, model, certificate, *, peak_cap=None) -> dict:
    """Rebuild matrices and check a witness without invoking any optimizer."""
    objective,a,b,eq,rhs,bounds = lp_problem(load,model,peak_cap=peak_cap)
    x = np.asarray(certificate["x"], float)
    y, v, lo, hi = [np.asarray(certificate[key], float)
                    for key in ("inequality_dual", "equality_dual", "lower_dual", "upper_dual")]
    if (x.shape != objective.shape or y.shape != b.shape or v.shape != rhs.shape
            or lo.shape != x.shape or hi.shape != x.shape
            or not all(np.isfinite(z).all() for z in (x,y,v,lo,hi))):
        raise ValueError("Malformed primal-dual certificate.")
    lb = np.array([-np.inf if p[0] is None else p[0] for p in bounds])
    ub = np.array([np.inf if p[1] is None else p[1] for p in bounds])
    finite_l,finite_u = np.isfinite(lb),np.isfinite(ub)
    slack = b-a@x
    primal = float(objective@x)
    dual = float(b@y + rhs@v + lb[finite_l]@lo[finite_l] + ub[finite_u]@hi[finite_u])
    residuals = {
        "primal_feasibility": float(max(0., -slack.min(), np.abs(eq@x-rhs).max(),
                                       np.max(lb-x), np.max(x-ub))),
        "dual_sign": float(max(0., y.max(), -lo.min(), hi.max(),
                               np.abs(lo[~finite_l]).max(initial=0.), np.abs(hi[~finite_u]).max(initial=0.))),
        "stationarity": float(np.abs(objective-a.T@y-eq.T@v-lo-hi).max()),
        "complementarity": float(max(np.abs(slack*y).max(),
                     np.abs((x[finite_l]-lb[finite_l])*lo[finite_l]).max(initial=0.),
                     np.abs((ub[finite_u]-x[finite_u])*hi[finite_u]).max(initial=0.))),
        "duality_gap": abs(primal-dual),
        "recorded_objective_error": abs(primal-float(certificate["objective"])),
    }
    limits = {"primal_feasibility":1e-5,"dual_sign":1e-8,"stationarity":1e-8,
              "complementarity":1e-4,"duality_gap":1e-4,"recorded_objective_error":1e-5}
    return {"passed": all(residuals[k] <= limits[k] for k in limits),
            "residuals":residuals,"limits":limits,"primal_objective":primal,"dual_objective":dual}


def solve_day(load, model, *, peak_cap=None):
    problem = lp_problem(load,model,peak_cap=peak_cap)
    objective,a,b,eq,rhs,bounds = problem
    result = linprog(objective,A_ub=a,b_ub=b,A_eq=eq,b_eq=rhs,bounds=bounds,method="highs")
    if not result.success:
        raise RuntimeError(f"Offline benchmark failed: {result.message}")
    cert = {"x":result.x.tolist(),"objective":float(result.fun),
            "inequality_dual":result.ineqlin.marginals.tolist(),
            "equality_dual":result.eqlin.marginals.tolist(),
            "lower_dual":result.lower.marginals.tolist(),"upper_dual":result.upper.marginals.tolist()}
    verification = verify_lp_certificate(load,model,cert,peak_cap=peak_cap)
    if not verification["passed"]:
        raise RuntimeError(f"Unverified LP result: {verification}")
    # Preserve each SOC increment, remove simultaneous charging/discharging.
    increment = model.efficiency*result.x[:24]-result.x[24:48]/model.efficiency
    charge = np.maximum(increment,0.)/model.efficiency
    discharge = np.maximum(-increment,0.)*model.efficiency
    return cert,charge,discharge


def offline_benchmark(frame: pd.DataFrame, model: StorageModel) -> tuple[pd.DataFrame,dict]:
    data = check_days(frame)
    days = []
    groups = list(data.groupby(data.timestamp.dt.floor("D")))
    for date,group in groups:
        cert,_,_ = solve_day(group.net_load_mw.to_numpy(float),model)
        days.append({"date":date.isoformat(),"minimum_peak":cert})
    cap = max(day["minimum_peak"]["objective"] for day in days)
    traces = []
    for (_,group),day in zip(groups,days):
        cert,c,d = solve_day(group.net_load_mw.to_numpy(float),model,peak_cap=cap)
        day["minimum_throughput_at_period_peak"] = cert
        soc = model.initial_soc_mwh + np.cumsum(model.efficiency*c-d/model.efficiency)
        traces.append(pd.DataFrame({"timestamp":group.timestamp.to_numpy(),"net_load_mw":group.net_load_mw.to_numpy(),
            "charge_mw":c,"discharge_mw":d,"soc_before_mwh":np.r_[model.initial_soc_mwh,soc[:-1]],
            "storage_soc_mwh":soc,"controlled_net_load_mw":group.net_load_mw.to_numpy()+c-d}))
    trace = pd.concat(traces,ignore_index=True)
    audit = audit_trace(trace,model)
    if not audit["passed"] or trace.controlled_net_load_mw.max() > cap+1e-5:
        raise RuntimeError("Offline physical witness invalid.")
    return trace,{"model":asdict(model),"period_optimum_peak_mw":float(cap),
        "minimum_throughput_at_optimum_peak_mwh":float((trace.charge_mw+trace.discharge_mw).sum()),
        "oracle_p95_mw":float(trace.controlled_net_load_mw.quantile(.95)),
        "oracle_loss_mwh":float((trace.charge_mw-trace.discharge_mw).sum()),
        "physical_audit":audit,"daily_certificates":days}


def verify_benchmark(data: pd.DataFrame, trace: pd.DataFrame, report: dict) -> bool:
    """Independent recheck of decomposition, dual bounds and physical attainment."""
    model = StorageModel(**report["model"])
    data,trace = check_days(data),check_days(trace)
    if not data.timestamp.equals(trace.timestamp) or not same_loads(data.net_load_mw,trace.net_load_mw):
        return False
    groups = list(data.groupby(data.timestamp.dt.floor("D")))
    days = report["daily_certificates"]
    if len(days) != len(groups):
        return False
    cap = report["period_optimum_peak_mw"]
    objectives,total = [],0.
    for (date,group),day in zip(groups,days):
        if date.isoformat() != day["date"]:
            return False
        peak = verify_lp_certificate(group.net_load_mw,model,day["minimum_peak"])
        cost = verify_lp_certificate(group.net_load_mw,model,day["minimum_throughput_at_period_peak"],peak_cap=cap)
        if not peak["passed"] or not cost["passed"]:
            return False
        objectives.append(peak["primal_objective"])
        total += cost["primal_objective"]
    throughput = float((trace.charge_mw+trace.discharge_mw).sum())
    return bool(audit_trace(trace,model)["passed"] and abs(max(objectives)-cap) <= 1e-5
        and abs(trace.controlled_net_load_mw.max()-cap) <= 1e-5
        and abs(total-throughput) <= 1e-3
        and abs(throughput-report["minimum_throughput_at_optimum_peak_mwh"]) <= 1e-3
        and abs(trace.controlled_net_load_mw.quantile(.95)-report["oracle_p95_mw"]) <= 1e-5
        and abs((trace.charge_mw-trace.discharge_mw).sum()-report["oracle_loss_mwh"]) <= 1e-5)


def exact_power_bound_witness(timestamps, decimal_loads, model: StorageModel) -> dict:
    """Construct and verify an exact rational witness at the power lower bound.

    Greedy recharge is only a candidate construction, not a general optimizer.
    If it fails any constraint, no exact certificate is claimed.
    """
    model.validate()
    check_days(pd.DataFrame({"timestamp":timestamps,"net_load_mw":decimal_loads}))
    values = {key:Fraction(str(value)) for key,value in asdict(model).items()}
    loads = [Fraction(value) for value in decimal_loads]
    cap = max(loads)-values["discharge_power_mw"]
    eta = values["efficiency"]
    initial = values["initial_soc_mwh"]
    soc,total_d,total_c = initial,Fraction(0),Fraction(0)
    actions = []
    for i,(stamp,load) in enumerate(zip(pd.to_datetime(timestamps,utc=True),loads)):
        if stamp.hour == 0:
            soc = initial
        d = max(Fraction(0),load-cap)
        c = (min(values["charge_power_mw"],cap-load,(initial-soc)/eta)
             if d == 0 else Fraction(0))
        soc += eta*c-d/eta
        if (min(c,d) < 0 or c > values["charge_power_mw"] or d > values["discharge_power_mw"]
                or not values["minimum_soc_mwh"] <= soc <= values["maximum_soc_mwh"]
                or load+c-d > cap or (stamp.hour == 23 and soc != initial)):
            return {"passed":False,"reason":"Greedy exact witness infeasible; use verified LP bounds instead.","row":i}
        total_c += c
        total_d += d
        if c or d:
            actions.append({"row":i,"charge_mw":str(c),"discharge_mw":str(d)})
    required_d = sum((max(Fraction(0),load-cap) for load in loads),Fraction(0))
    throughput_lower = required_d*(1+1/eta**2)
    if total_c+total_d != throughput_lower:
        raise RuntimeError("Exact throughput lower bound not attained.")
    return {"passed":True,"arithmetic":"Python Fraction; exact decimal input strings, zero tolerance",
        "model":{key:str(value) for key,value in values.items()},
        "peak_lower_bound_mw":str(cap),"attained_peak_mw":str(cap),
        "required_discharge_mwh":str(required_d),
        "throughput_lower_bound_mwh":str(throughput_lower),"attained_throughput_mwh":str(total_c+total_d),
        "actions":actions,
        "scope":"Exact global peak and then throughput optimum for this finite physical offline model, not the causal controller."}


def verify_exact_witness(timestamps, decimal_loads, model, certificate) -> bool:
    """Check supplied sparse rational actions; no solver or greedy constructor."""
    model.validate()
    check_days(pd.DataFrame({"timestamp":timestamps,"net_load_mw":decimal_loads}))
    if certificate.get("passed") is not True:
        return False
    values = {key:Fraction(str(value)) for key,value in asdict(model).items()}
    if certificate["model"] != {key:str(value) for key,value in values.items()}:
        return False
    loads = [Fraction(value) for value in decimal_loads]
    cap = max(loads)-values["discharge_power_mw"]
    if Fraction(certificate["peak_lower_bound_mw"]) != cap or Fraction(certificate["attained_peak_mw"]) != cap:
        return False
    actions = {item["row"]:item for item in certificate["actions"]}
    if len(actions) != len(certificate["actions"]) or any(type(i) is not int or not 0 <= i < len(loads) for i in actions):
        return False
    eta,initial = values["efficiency"],values["initial_soc_mwh"]
    soc,total_c,total_d,attained = initial,Fraction(0),Fraction(0),None
    for i,(stamp,load) in enumerate(zip(pd.to_datetime(timestamps,utc=True),loads)):
        if stamp.hour == 0:
            soc = initial
        action = actions.get(i,{"charge_mw":"0","discharge_mw":"0"})
        c,d = Fraction(action["charge_mw"]),Fraction(action["discharge_mw"])
        soc += eta*c-d/eta
        controlled = load+c-d
        if (c < 0 or d < 0 or (c and d) or c > values["charge_power_mw"] or d > values["discharge_power_mw"]
                or not values["minimum_soc_mwh"] <= soc <= values["maximum_soc_mwh"]
                or controlled > cap or (stamp.hour == 23 and soc != initial)):
            return False
        attained = controlled if attained is None else max(attained,controlled)
        total_c,total_d = total_c+c,total_d+d
    required = sum((max(Fraction(0),load-cap) for load in loads),Fraction(0))
    lower = required*(1+1/eta**2)
    return bool(attained == cap and total_c+total_d == lower
                and Fraction(certificate["required_discharge_mwh"]) == required
                and Fraction(certificate["throughput_lower_bound_mwh"]) == lower
                and Fraction(certificate["attained_throughput_mwh"]) == lower)
