"""Isolated two-stage peak/tail controller; prior versioned policies stay frozen."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.balanced_storage import complete_utc_days
from src.forecast_dispatch import physical_summary
from src.optimality import StorageModel
from src.scenarios import dynamic_reserve_fraction


@dataclass(frozen=True)
class TailPolicy:
    objective: str = "tail"
    reserve: str = "dynamic"
    peak_tolerance_mw: float = 0.

    def validate(self):
        if (self.objective not in ("peak", "tail") or self.reserve not in ("dynamic", "physical")
                or not np.isfinite(self.peak_tolerance_mw) or self.peak_tolerance_mw < 0):
            raise ValueError("Invalid tail-control policy.")


POLICIES = {
    "peak_dynamic": TailPolicy("peak", "dynamic"),
    "peak_physical": TailPolicy("peak", "physical"),
    "tail_dynamic_0": TailPolicy("tail", "dynamic"),
    "tail_dynamic_250": TailPolicy("tail", "dynamic", 250.),
    "tail_physical_0": TailPolicy("tail", "physical"),
    "tail_physical_250": TailPolicy("tail", "physical", 250.),
}


def canonical_actions(charge, discharge, efficiency):
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must lie in (0,1].")
    c, d = np.asarray(charge, float), np.asarray(discharge, float)
    if (c.shape != d.shape or not np.isfinite(c).all() or not np.isfinite(d).all()
            or min(c.min(), d.min()) < -1e-6):
        raise ValueError("Finite matching nonnegative actions required.")
    increment = efficiency * c - d / efficiency
    return np.maximum(increment, 0.) / efficiency, np.maximum(-increment, 0.) * efficiency


def solve_window(point, upper, soc, lower_soc, upper_soc, model, gates, tail_threshold, policy):
    policy.validate()
    model.validate()
    arrays = [np.asarray(a, float) for a in (point, upper, lower_soc, upper_soc)]
    point, upper, lower_soc, upper_soc = arrays
    if point.ndim != 1:
        raise ValueError("One-dimensional planning arrays required.")
    n = len(point)
    if (n == 0 or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays)
            or not np.isfinite([soc, tail_threshold, gates["peak_threshold_mw"]]).all()
            or np.any(upper < point) or np.any(lower_soc > upper_soc)
            or np.any(lower_soc < model.minimum_soc_mwh - 1e-6)
            or np.any(upper_soc > model.maximum_soc_mwh + 1e-6)
            or not model.minimum_soc_mwh - 1e-6 <= soc <= model.maximum_soc_mwh + 1e-6):
        raise ValueError("Finite feasible matching planning arrays and state required.")
    size = 2 * n + 1
    energy = np.zeros((n, size))
    energy[:, :n] = model.efficiency * np.tril(np.ones((n, n)))
    energy[:, n:2 * n] = -np.tril(np.ones((n, n))) / model.efficiency
    peak = np.zeros((n, size))
    peak[:, :n], peak[:, n:2 * n], peak[:, -1] = np.eye(n), -np.eye(n), -1.
    a = np.vstack([peak, energy, -energy])
    b = np.r_[-point, upper_soc - soc, soc - lower_soc]
    bounds = (
        [(0., model.charge_power_mw)] * n
        + [(0., model.discharge_power_mw if u >= gates["peak_threshold_mw"] else 0.) for u in upper]
        + [(None, None)]
    )
    objective = np.zeros(size)
    objective[-1] = 1.
    first = linprog(objective, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    if not first.success:
        raise RuntimeError(f"Peak-stage LP failed: {first.message}")
    cap = float(first.fun) + policy.peak_tolerance_mw
    bounds[-1] = (None, cap)
    objective[-1] = 0.
    objective[:2 * n] = 1.
    if policy.objective == "tail":
        # One nonnegative slack per hour measures excess above a fixed validation threshold.
        extended = np.zeros((len(a) + n, size + n))
        extended[:len(a), :size] = a
        extended[len(a):, :n] = np.eye(n)
        extended[len(a):, n:2 * n] = -np.eye(n)
        extended[len(a):, size:] = -np.eye(n)
        a, b = extended, np.r_[b, tail_threshold - point]
        objective = np.r_[objective * .001, np.full(n, 1. / n)]
        bounds = bounds + [(0., None)] * n
    second = linprog(objective, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    if not second.success:
        raise RuntimeError(f"Refinement-stage LP failed: {second.message}")
    c, d = canonical_actions(second.x[:n], second.x[n:2 * n], model.efficiency)
    planned_soc = soc + np.cumsum(model.efficiency * c - d / model.efficiency)
    planned_load = point + c - d
    if (np.any(planned_soc < lower_soc - 1e-5) or np.any(planned_soc > upper_soc + 1e-5)
            or c.max() > model.charge_power_mw + 1e-5 or d.max() > model.discharge_power_mw + 1e-5
            or np.any(d[upper < gates["peak_threshold_mw"]] > 1e-5)
            or planned_load.max() > cap + 1e-5):
        raise RuntimeError("Canonical LP plan failed independent feasibility checks.")
    return c, d, {
        "stage1_peak_mw": float(first.fun), "window_peak_cap_mw": cap,
        "planned_max_mw": float(planned_load.max()),
        "planned_mean_excess_mw": float(np.maximum(planned_load - tail_threshold, 0.).mean()),
        "planned_throughput_mwh": float((c + d).sum()),
    }


def augment_metrics(trace, metrics, gates, tail_threshold):
    metrics = dict(metrics)
    baseline_excess = np.maximum(trace.net_load_mw - tail_threshold, 0.)
    controlled_excess = np.maximum(trace.controlled_net_load_mw - tail_threshold, 0.)
    high = trace.net_load_mw >= gates["peak_threshold_mw"]
    metrics.update({
        "baseline_p95_mw": float(trace.net_load_mw.quantile(.95)),
        "baseline_mean_excess_mw": float(baseline_excess.mean()),
        "controlled_mean_excess_mw": float(controlled_excess.mean()),
        "actual_high_load_charging_mwh": float(trace.loc[high, "charge_mw"].sum()),
        "actual_high_load_charging_hours": int((high & trace.charge_mw.gt(1e-6)).sum()),
        "recovery_charge_mwh": float(trace.recovery_charge_mwh.sum()) if "recovery_charge_mwh" in trace else None,
    })
    return metrics


def simulate(panel, calibration, gates, tail_threshold, policy, model=StorageModel()):
    policy.validate()
    model.validate()
    scalar = [tail_threshold, calibration["uncertainty_scale_mw"], *gates.values()]
    if (not np.isfinite(scalar).all() or calibration["uncertainty_scale_mw"] <= 0
            or not gates["charge_threshold_mw"] <= tail_threshold <= gates["peak_threshold_mw"] <= gates["release_threshold_mw"]):
        raise ValueError("Finite ordered validation gates and positive uncertainty scale required.")
    times = pd.DatetimeIndex(pd.to_datetime(panel.timestamp, utc=True))
    if times.has_duplicates or times.hasnans or not times.is_monotonic_increasing:
        raise ValueError("Ordered unique prediction timestamps required.")
    data, omitted = complete_utc_days(panel)
    if omitted or len(data) != len(panel):
        raise ValueError("Prepare complete UTC days explicitly; do not silently omit hours.")
    columns = [f"{p}_h{h}_mw" for h in range(1, 7) for p in ("prediction", "lower", "upper")]
    if not np.isfinite(data[columns + ["net_load_mw"]].to_numpy(float)).all():
        raise ValueError("Finite six-hour forecasts and evaluation values required.")
    for h in range(1, 7):
        if not ((data[f"lower_h{h}_mw"] <= data[f"prediction_h{h}_mw"]).all()
                and (data[f"prediction_h{h}_mw"] <= data[f"upper_h{h}_mw"]).all()):
            raise ValueError("Calibrated forecast bounds must bracket point forecasts.")
    if "issued_timestamp" in data:
        issued = pd.DatetimeIndex(pd.to_datetime(data.issued_timestamp, utc=True))
        if not issued.equals(pd.DatetimeIndex(data.timestamp) - pd.Timedelta(hours=1)):
            raise ValueError("Forecast issuance must be the latest observed hour.")
    rows = []
    for _, group in data.groupby(data.timestamp.dt.floor("D")):
        soc = model.initial_soc_mwh
        for row in group.itertuples(index=False):
            remaining = 24 - row.timestamp.hour
            steps = min(6, remaining)
            point = np.array([getattr(row, f"prediction_h{h}_mw") for h in range(1, steps + 1)])
            upper = np.array([getattr(row, f"upper_h{h}_mw") for h in range(1, steps + 1)])
            fraction = dynamic_reserve_fraction(
                upper, upper - point, gates["peak_threshold_mw"],
                uncertainty_scale_mw=calibration["uncertainty_scale_mw"])
            target = model.minimum_soc_mwh if policy.reserve == "physical" else fraction * model.capacity_mwh
            released = point.max() >= gates["release_threshold_mw"]
            floor = model.minimum_soc_mwh if released else max(model.minimum_soc_mwh, min(target, soc))
            after = remaining - np.arange(1, steps + 1)
            lower_soc = np.maximum(floor, model.initial_soc_mwh - after * model.charge_power_mw * model.efficiency)
            upper_soc = np.minimum(model.maximum_soc_mwh, model.initial_soc_mwh + after * model.discharge_power_mw / model.efficiency)
            c, d, info = solve_window(
                point, upper, soc, lower_soc, upper_soc, model, gates, tail_threshold, policy)
            charge, discharge = float(c[0]), float(d[0])
            before = soc
            soc += model.efficiency * charge - discharge / model.efficiency
            rows.append({
                "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
                "charge_mw": charge, "discharge_mw": discharge, "soc_before_mwh": before,
                "storage_soc_mwh": soc, "controlled_net_load_mw": row.net_load_mw + charge - discharge,
                "prediction_h1_mw": point[0], "upper_h1_mw": upper[0],
                "lower_h1_mw": row.lower_h1_mw, "window_max_prediction_mw": float(point.max()),
                "reserve_target_mwh": target, "advisory_floor_mwh": floor,
                "reserve_release_active": bool(released), "required_next_soc_mwh": lower_soc[0],
                "recovery_charge_mwh": charge if row.lower_h1_mw > gates["charge_threshold_mw"] else 0.,
                "solver_status": "two_stage_optimal", **info,
            })
    trace = pd.DataFrame(rows)
    metrics = physical_summary(trace, model, terminal=True)
    metrics.update({
        "physical_parameters": asdict(model), "policy_parameters": asdict(policy),
        "forecast_h1_mae_mw": float((trace.net_load_mw - trace.prediction_h1_mw).abs().mean()),
    })
    return trace, augment_metrics(trace, metrics, gates, tail_threshold)
