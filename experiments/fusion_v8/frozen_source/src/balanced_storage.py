"""Causal daily energy accounting with reachable end-of-day SOC constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scenarios import _solve_storage_window, dynamic_reserve_fraction


def complete_utc_days(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep complete contiguous UTC days; never pretend missing hours took no energy."""
    data = panel.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data.timestamp.isna().any() or data.timestamp.duplicated().any():
        raise ValueError("Planning timestamps must be valid and unique.")
    kept, omitted = [], []
    for date, group in data.groupby(data.timestamp.dt.floor("D")):
        expected = pd.date_range(date, periods=24, freq="h")
        if len(group) == 24 and np.array_equal(group.timestamp.to_numpy(), expected.to_numpy()):
            kept.append(group)
        else:
            omitted.append(date.isoformat())
    if not kept:
        raise ValueError("No complete UTC days available.")
    return pd.concat(kept, ignore_index=True), omitted


def simulate_balanced_storage(
    panel: pd.DataFrame,
    peak_threshold_mw: float,
    charge_threshold_mw: float,
    uncertainty_scale_mw: float,
    *,
    capacity_mwh: float = 30000.0,
    charge_power_mw: float = 1250.0,
    discharge_power_mw: float = 1250.0,
    efficiency: float = 0.92,
    fixed_reserve_fraction: float | None = None,
    hard_min_fraction: float = 0.15,
    daily_terminal: bool = True,
    planning_objective: str = "upper",
    release_threshold_mw: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Daily independent comparisons, with initial SOC 50% and optional equality.

    Both policies share a constant physical SOC minimum. Dynamic reserve remains
    an advisory target. Terminal mode allows recovery charging at any hour and
    counts that load; ordinary charging retains the validation low-load gate.
    Future actual net load never enters optimization. A missing day starts a new
    independent episode and is not silently bridged.
    """
    if not (0 < efficiency <= 1) or capacity_mwh <= 0:
        raise ValueError("Positive capacity and efficiency in (0, 1] required.")
    if charge_power_mw < 0 or discharge_power_mw < 0 or uncertainty_scale_mw <= 0:
        raise ValueError("Nonnegative powers and positive uncertainty scale required.")
    if not 0 <= hard_min_fraction <= 0.5:
        raise ValueError("Hard minimum must lie in [0, 0.5].")
    if fixed_reserve_fraction is not None and not 0 <= fixed_reserve_fraction <= 0.5:
        raise ValueError("Fixed reserve must lie in [0, 0.5].")
    if planning_objective not in {"upper", "point"}:
        raise ValueError("Planning objective must be upper or point.")
    if release_threshold_mw is not None and (not np.isfinite(release_threshold_mw) or release_threshold_mw < peak_threshold_mw):
        raise ValueError("Finite release threshold must be at least the ordinary peak threshold.")
    data, omitted = complete_utc_days(panel)
    horizons = sorted(int(c.removeprefix("upper_h").removesuffix("_mw"))
                      for c in data if c.startswith("upper_h") and c.endswith("_mw"))
    if not horizons or horizons != list(range(1, len(horizons) + 1)):
        raise ValueError("Contiguous forecast horizons required.")
    columns = [f"{prefix}_h{h}_mw" for h in horizons for prefix in ("prediction", "lower", "upper")]
    if not np.isfinite(data[columns + ["net_load_mw"]].to_numpy(dtype=float)).all():
        raise ValueError("Forecasts and evaluation values must be finite.")
    if not all((data[f"lower_h{h}_mw"] <= data[f"prediction_h{h}_mw"]).all()
               and (data[f"prediction_h{h}_mw"] <= data[f"upper_h{h}_mw"]).all() for h in horizons):
        raise ValueError("Forecast lower <= prediction <= upper required.")
    initial = capacity_mwh * 0.5
    hard_min = capacity_mwh * hard_min_fraction
    rows, daily = [], []
    for date, group in data.groupby(data.timestamp.dt.floor("D")):
        soc = initial
        for row in group.itertuples(index=False):
            remaining = 24 - row.timestamp.hour
            steps = min(len(horizons), remaining)
            upper = np.array([getattr(row, f"upper_h{h}_mw") for h in range(1, steps + 1)])
            lower = np.array([getattr(row, f"lower_h{h}_mw") for h in range(1, steps + 1)])
            point = np.array([getattr(row, f"prediction_h{h}_mw") for h in range(1, steps + 1)])
            fraction = (fixed_reserve_fraction if fixed_reserve_fraction is not None else
                        dynamic_reserve_fraction(upper, upper - point, peak_threshold_mw,
                                                 uncertainty_scale_mw=uncertainty_scale_mw))
            target = fraction * capacity_mwh
            release_active = release_threshold_mw is not None and float(point.max()) >= release_threshold_mw
            # Extreme forecast stress releases advisory energy, never the
            # physical SOC minimum or the terminal-reachability requirement.
            floor = hard_min if release_active else max(hard_min, min(target, soc))
            lower_bounds = np.full(steps, floor)
            # Balanced episodes cannot stockpile above the initial daily budget.
            # This ensures future all-low-load windows can always restore equality.
            upper_bounds = np.full(steps, initial if daily_terminal else capacity_mwh)
            if daily_terminal:
                # Known calendar and rated power determine reachability, without
                # future observations. Tighten bounds as the daily deadline nears.
                after = remaining - np.arange(1, steps + 1)
                lower_bounds = np.maximum(lower_bounds, initial - after * charge_power_mw * efficiency)
                upper_bounds = np.minimum(upper_bounds, initial + after * discharge_power_mw / efficiency)
            discharge, charge, status = _solve_storage_window(
                upper, lower, soc, capacity_mwh, peak_threshold_mw,
                charge_threshold_mw, floor, discharge_power_mw, charge_power_mw,
                efficiency, lower_bounds, upper_bounds,
                allow_recovery_charge=daily_terminal, return_plan=True,
                objective_forecasts_mw=point if planning_objective == "point" else None,
            )
            d, c = float(discharge[0]), float(charge[0])
            if min(c, d) > 1e-6:
                raise RuntimeError("Simultaneous charge and discharge requires a discrete model.")
            previous = soc
            soc = previous + efficiency * c - d / efficiency
            if soc < hard_min - 1e-5 or soc > capacity_mwh + 1e-5:
                raise RuntimeError("Solver action violated the hard SOC bound.")
            rows.append({"timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
                         "charge_mw": c, "discharge_mw": d, "soc_before_mwh": previous,
                         "storage_soc_mwh": soc, "reserve_target_mwh": target,
                         "hard_min_soc_mwh": hard_min, "required_next_soc_mwh": lower_bounds[0],
                         "prediction_h1_mw": point[0], "upper_h1_mw": upper[0],
                         "window_max_prediction_mw": float(point.max()),
                         "advisory_floor_mwh": floor, "reserve_release_active": release_active,
                         "controlled_net_load_mw": row.net_load_mw + c - d,
                         "recovery_charge_mwh": c if lower[0] > charge_threshold_mw else 0.0,
                         "solver_status": status})
        daily.append({"date": date.isoformat(), "initial_soc_mwh": initial, "final_soc_mwh": soc,
                      "terminal_error_mwh": soc - initial})
    trace = pd.DataFrame(rows)
    accounting = trace.storage_soc_mwh - trace.soc_before_mwh - efficiency * trace.charge_mw + trace.discharge_mw / efficiency
    summary = {
        "days": len(daily), "rows": len(trace), "capacity_mwh": capacity_mwh,
        "daily_terminal": daily_terminal, "hard_min_fraction": hard_min_fraction,
        "planning_objective": planning_objective, "release_threshold_mw": release_threshold_mw,
        "reserve_release_hours": int(trace.reserve_release_active.sum()),
        "initial_soc_per_day_mwh": initial,
        "mean_final_soc_mwh": float(np.mean([d["final_soc_mwh"] for d in daily])),
        "max_abs_terminal_error_mwh": float(max(abs(d["terminal_error_mwh"]) for d in daily)),
        "net_initial_energy_consumed_mwh": float(sum(initial - d["final_soc_mwh"] for d in daily)),
        "energy_balance_error_mwh": float(accounting.abs().max()),
        "hard_soc_violation_hours": int((trace.storage_soc_mwh < hard_min - 1e-5).sum()),
        "reserve_target_shortfall_hours": int((trace.storage_soc_mwh < trace.reserve_target_mwh - 1e-5).sum()),
        "baseline_p95_mw": float(trace.net_load_mw.quantile(.95)),
        "controlled_p95_mw": float(trace.controlled_net_load_mw.quantile(.95)),
        "baseline_peak_mw": float(trace.net_load_mw.max()),
        "controlled_peak_mw": float(trace.controlled_net_load_mw.max()),
        "charge_energy_mwh": float(trace.charge_mw.sum()),
        "discharge_energy_mwh": float(trace.discharge_mw.sum()),
        "conversion_loss_mwh": float(((1 - efficiency) * trace.charge_mw + (1 / efficiency - 1) * trace.discharge_mw).sum()),
        "throughput_mwh": float((trace.charge_mw + trace.discharge_mw).sum()),
        "recovery_charge_mwh": float(trace.recovery_charge_mwh.sum()),
        "solver_optimal_rate_pct": 100.0,
        "omitted_partial_days": omitted, "daily_soc": daily,
    }
    return trace, summary
