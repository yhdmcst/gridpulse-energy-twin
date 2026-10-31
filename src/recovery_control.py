"""Causal next-state debt governor around a frozen reference policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.balanced_storage import complete_utc_days
from src.forecast_dispatch import physical_summary
from src.optimality import StorageModel
from src.scenarios import dynamic_reserve_fraction
from src.tail_control import augment_metrics, canonical_actions


@dataclass(frozen=True)
class RecoveryPolicy:
    objective: str
    debt: bool
    headroom: bool

    def validate(self):
        if self.objective not in ("peak", "tail") or (self.headroom and not self.debt):
            raise ValueError("Headroom guard requires the zero-debt recovery policy.")


POLICIES = {
    "tail_unanchored": RecoveryPolicy("tail", False, False),
    "tail_debt": RecoveryPolicy("tail", True, False),
    "tail_guard": RecoveryPolicy("tail", True, True),
    "peak_debt": RecoveryPolicy("peak", True, False),
    "peak_guard": RecoveryPolicy("peak", True, True),
}
REFERENCE_COLUMNS = (
    "timestamp", "charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh",
)


def validate_reference(reference, timestamps, model):
    model.validate()
    if reference.empty or any(c not in reference for c in REFERENCE_COLUMNS):
        raise ValueError("Complete reference action/state columns required.")
    ref = reference[list(REFERENCE_COLUMNS)].copy().reset_index(drop=True)
    ref.timestamp = pd.to_datetime(ref.timestamp, utc=True)
    times = pd.DatetimeIndex(ref.timestamp)
    if (times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
            or not times.equals(pd.DatetimeIndex(timestamps))):
        raise ValueError("Reference must align exactly with ordered planning timestamps.")
    numeric = ref[list(REFERENCE_COLUMNS[1:])].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Finite reference states and actions required.")
    c, d, before, after = numeric.T
    balance = after - before - model.efficiency * c + d / model.efficiency
    bounds = max(0., -c.min(), -d.min(), c.max() - model.charge_power_mw,
                 d.max() - model.discharge_power_mw, np.minimum(c, d).max(),
                 model.minimum_soc_mwh - min(before.min(), after.min()),
                 max(before.max(), after.max()) - model.maximum_soc_mwh)
    if bounds > 1e-5 or np.abs(balance).max() > 1e-5:
        raise ValueError("Reference violates actual modeled physical parameters.")
    consecutive = times[1:] - times[:-1] == pd.Timedelta(hours=1)
    if consecutive.any() and np.abs((before[1:] - after[:-1])[consecutive]).max() > 1e-5:
        raise ValueError("Reference continuity violation.")
    first, last = times.hour == 0, times.hour == 23
    if (np.abs(before[first] - model.initial_soc_mwh).max() > 1e-5
            or np.abs(after[last] - model.initial_soc_mwh).max() > 1e-5):
        raise ValueError("Reference must satisfy daily initial/terminal equality.")
    return ref


def solve_window(point, upper, soc, lower_soc, upper_soc, model, gates, tail,
                 policy, reference_after, reference_net):
    model.validate()
    policy.validate()
    arrays = [np.asarray(v, float) for v in (point, upper, lower_soc, upper_soc)]
    point, upper, lower_soc, upper_soc = arrays
    if point.ndim != 1:
        raise ValueError("One-dimensional planning arrays required.")
    n = len(point)
    if (not n or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays)
            or not np.isfinite([soc, tail, reference_after, reference_net]).all()
            or np.any(upper < point) or np.any(lower_soc > upper_soc)
            or np.any(lower_soc < model.minimum_soc_mwh - 1e-6)
            or np.any(upper_soc > model.maximum_soc_mwh + 1e-6)
            or not model.minimum_soc_mwh - 1e-6 <= soc <= model.maximum_soc_mwh + 1e-6
            or not model.minimum_soc_mwh - 1e-6 <= reference_after <= model.maximum_soc_mwh + 1e-6):
        raise ValueError("Finite matching feasible planning arrays required.")
    lower_soc = lower_soc.copy()
    if policy.debt:
        lower_soc[0] = max(lower_soc[0], reference_after)
    if np.any(lower_soc > upper_soc + 1e-6):
        raise RuntimeError("Debt floor exceeds physical/terminal bounds.")
    extra = 1 if policy.objective == "peak" else n
    size = 2 * n + extra
    energy = np.zeros((n, size))
    energy[:, :n] = model.efficiency * np.tril(np.ones((n, n)))
    energy[:, n:2 * n] = -np.tril(np.ones((n, n))) / model.efficiency
    load = np.zeros((n, size))
    load[:, :n], load[:, n:2 * n] = np.eye(n), -np.eye(n)
    objective = np.zeros(size)
    objective[:2 * n] = .001
    if policy.objective == "peak":
        load[:, -1] = -1.
        limits = -point
        objective[-1] = 1.
        auxiliary_bounds = [(None, None)]
    else:
        load[:, 2 * n:] = -np.eye(n)
        limits = tail - point
        objective[2 * n:] = 1. / n
        auxiliary_bounds = [(0., None)] * n
    a, b = np.vstack([load, energy, -energy]), np.r_[limits, upper_soc - soc, soc - lower_soc]
    cap = max(reference_net, tail - upper[0]) if policy.headroom else None
    if cap is not None:
        row = np.zeros(size)
        row[0], row[n] = 1., -1.
        a, b = np.vstack([a, row]), np.r_[b, cap]
    bounds = (
        [(0., model.charge_power_mw)] * n
        + [(0., model.discharge_power_mw if u >= gates["peak_threshold_mw"] else 0.) for u in upper]
        + auxiliary_bounds
    )
    result = linprog(objective, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Recovery LP failed: {result.message}")
    c, d = canonical_actions(result.x[:n], result.x[n:2 * n], model.efficiency)
    planned_soc = soc + np.cumsum(model.efficiency * c - d / model.efficiency)
    if (np.any(planned_soc < lower_soc - 1e-5) or np.any(planned_soc > upper_soc + 1e-5)
            or c.max() > model.charge_power_mw + 1e-5 or d.max() > model.discharge_power_mw + 1e-5
            or np.any(d[upper < gates["peak_threshold_mw"]] > 1e-5)
            or (cap is not None and c[0] - d[0] > cap + 1e-5)):
        raise RuntimeError("Recovery plan failed independent bound checks.")
    return c, d, {
        "required_next_soc_mwh": float(lower_soc[0]),
        "headroom_cap_mw": cap,
        "planned_max_mw": float(np.max(point + c - d)),
        "planned_mean_excess_mw": float(np.maximum(point + c - d - tail, 0.).mean()),
    }


def simulate(panel, reference, calibration, gates, tail, policy, model=StorageModel()):
    model.validate()
    policy.validate()
    if (not np.isfinite([tail, calibration["uncertainty_scale_mw"], *gates.values()]).all()
            or calibration["uncertainty_scale_mw"] <= 0
            or not gates["charge_threshold_mw"] <= tail <= gates["peak_threshold_mw"] <= gates["release_threshold_mw"]):
        raise ValueError("Finite ordered calibration and gates required.")
    times = pd.DatetimeIndex(pd.to_datetime(panel.timestamp, utc=True))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Ordered unique planning timestamps required.")
    data, omitted = complete_utc_days(panel)
    if omitted or len(data) != len(panel):
        raise ValueError("Complete UTC planning days required without silent omissions.")
    columns = [f"{p}_h{h}_mw" for h in range(1, 7) for p in ("prediction", "lower", "upper")]
    if not np.isfinite(data[columns + ["net_load_mw"]].to_numpy(float)).all():
        raise ValueError("Finite forecasts and evaluation values required.")
    for h in range(1, 7):
        if not ((data[f"lower_h{h}_mw"] <= data[f"prediction_h{h}_mw"]).all()
                and (data[f"prediction_h{h}_mw"] <= data[f"upper_h{h}_mw"]).all()):
            raise ValueError("Forecast bands must bracket points.")
    if "issued_timestamp" in data:
        issued = pd.DatetimeIndex(pd.to_datetime(data.issued_timestamp, utc=True))
        if not issued.equals(pd.DatetimeIndex(data.timestamp) - pd.Timedelta(hours=1)):
            raise ValueError("Invalid information availability timestamp.")
    ref = validate_reference(reference, data.timestamp, model)
    rows, soc = [], model.initial_soc_mwh
    for row, anchor in zip(data.itertuples(index=False), ref.itertuples(index=False)):
        if row.timestamp.hour == 0:
            soc = model.initial_soc_mwh
        if policy.debt and soc < anchor.soc_before_mwh - 1e-5:
            raise RuntimeError("Executed candidate fell below the causal reference state.")
        remaining = 24 - row.timestamp.hour
        steps = min(6, remaining)
        point = np.array([getattr(row, f"prediction_h{h}_mw") for h in range(1, steps + 1)])
        upper = np.array([getattr(row, f"upper_h{h}_mw") for h in range(1, steps + 1)])
        fraction = dynamic_reserve_fraction(
            upper, upper - point, gates["peak_threshold_mw"],
            uncertainty_scale_mw=calibration["uncertainty_scale_mw"])
        target = fraction * model.capacity_mwh
        released = point.max() >= gates["release_threshold_mw"]
        floor = model.minimum_soc_mwh if released else max(model.minimum_soc_mwh, min(target, soc))
        after = remaining - np.arange(1, steps + 1)
        lower = np.maximum(floor, model.initial_soc_mwh - after * model.charge_power_mw * model.efficiency)
        ceiling = np.minimum(model.maximum_soc_mwh, model.initial_soc_mwh + after * model.discharge_power_mw / model.efficiency)
        ref_net = anchor.charge_mw - anchor.discharge_mw
        c, d, info = solve_window(
            point, upper, soc, lower, ceiling, model, gates, tail, policy,
            anchor.storage_soc_mwh, ref_net)
        before, charge, discharge = soc, float(c[0]), float(d[0])
        soc += model.efficiency * charge - discharge / model.efficiency
        rows.append({
            "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
            "charge_mw": charge, "discharge_mw": discharge, "soc_before_mwh": before,
            "storage_soc_mwh": soc, "controlled_net_load_mw": row.net_load_mw + charge - discharge,
            "prediction_h1_mw": point[0], "upper_h1_mw": upper[0], "lower_h1_mw": row.lower_h1_mw,
            "reserve_target_mwh": target, "advisory_floor_mwh": floor, "reserve_release_active": bool(released),
            "reference_soc_before_mwh": anchor.soc_before_mwh,
            "reference_soc_after_mwh": anchor.storage_soc_mwh,
            "reference_charge_mw": anchor.charge_mw, "reference_discharge_mw": anchor.discharge_mw,
            "reference_debt_mwh": max(0., anchor.storage_soc_mwh - soc),
            "recovery_charge_mwh": charge if row.lower_h1_mw > gates["charge_threshold_mw"] else 0.,
            "solver_status": "recovery_optimal", **info,
        })
    trace = pd.DataFrame(rows)
    metrics = physical_summary(trace, model, terminal=True)
    metrics.update({
        "physical_parameters": asdict(model), "policy_parameters": asdict(policy),
        "forecast_h1_mae_mw": float((trace.net_load_mw - trace.prediction_h1_mw).abs().mean()),
        "max_reference_debt_mwh": float(trace.reference_debt_mwh.max()),
        "debt_hours": int(trace.reference_debt_mwh.gt(1e-5).sum()),
        "max_headroom_violation_mw": float(np.maximum(
            trace.charge_mw - trace.discharge_mw - trace.headroom_cap_mw, 0.).max()) if policy.headroom else None,
    })
    if policy.debt and metrics["max_reference_debt_mwh"] > 1e-5:
        raise RuntimeError("Reference debt constraint failed.")
    if policy.headroom and metrics["max_headroom_violation_mw"] > 1e-5:
        raise RuntimeError("Headroom constraint failed.")
    return trace, augment_metrics(trace, metrics, gates, tail)
