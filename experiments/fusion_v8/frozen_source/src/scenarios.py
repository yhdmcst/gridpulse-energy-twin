from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def dispatch_suggestions(risk_types: str, predicted_net_load_mw: float, residual_mw: float) -> list[dict[str, str]]:
    """Return auditable suggestions; these are not control commands."""
    suggestions: list[dict[str, str]] = []
    active = set(filter(None, str(risk_types if pd.notna(risk_types) else "interval_peak_watch").split("|")))
    if "interval_peak_watch" in active:
        suggestions.append({"action": "关注预测区间上界", "reason": "点预测尚未达到峰值阈值，但预测上界已达到阈值。", "constraint": "仅作为人工复核信号，结合后续观测确认，不直接下发控制指令。"})
    if "peak_load" in active:
        suggestions.append(
            {
                "action": "评估削峰资源预备状态",
                "reason": f"预测净负荷为 {predicted_net_load_mw:,.0f} MW，达到峰值风险阈值。",
                "constraint": "需在获得设备容量、荷电状态、用户负荷可调性和调度授权后执行。",
            }
        )
    if "solar_drop" in active:
        suggestions.append(
            {
                "action": "复核短时备用与储能响应窗口",
                "reason": "光伏出力出现快速下降，净负荷爬坡压力可能增大。",
                "constraint": "该回测信号使用实际观测光伏值；实际应用须替换为可用的预测或实时量测。",
            }
        )
    if "sustained_deviation" in active:
        direction = "高于" if residual_mw >= 0 else "低于"
        suggestions.append(
            {
                "action": "核验数据质量与运行状态",
                "reason": f"实际净负荷连续{direction}模型预期，残差为 {residual_mw:,.0f} MW。",
                "constraint": "候选偏差不等同于设备故障；需结合运行日志和人工复核。",
            }
        )
    return suggestions


def flatten_suggestions(alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in alerts.itertuples(index=False):
        for suggestion in dispatch_suggestions(row.risk_types, row.predicted_net_load_mw, row.residual_mw):
            rows.append(
                {
                    "时间 (UTC)": row.timestamp,
                    "风险类型": row.risk_types,
                    "建议动作": suggestion["action"],
                    "依据": suggestion["reason"],
                    "适用约束": suggestion["constraint"],
                }
            )
    return pd.DataFrame(rows)


def simulate_storage_dispatch(
    frame: pd.DataFrame,
    peak_threshold_mw: float,
    capacity_mwh: float,
    max_power_mw: float = 2500.0,
    charge_power_mw: float = 1250.0,
    efficiency: float = 0.92,
    charge_threshold_mw: float | None = None,
    emergency_threshold_mw: float | None = None,
    reserve_fraction: float = 0.30,
    energy_smoothing_hours: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a reserve-aware, theoretical battery peak-shaving scenario at hourly resolution."""
    result = frame.copy().sort_values("timestamp").reset_index(drop=True)
    low_charge_threshold = float(
        result["predicted_net_load_mw"].quantile(0.25) if charge_threshold_mw is None else charge_threshold_mw
    )
    emergency_threshold = float(
        peak_threshold_mw if emergency_threshold_mw is None else emergency_threshold_mw
    )
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1).")
    reserve_mwh = capacity_mwh * reserve_fraction
    effective_max_power_mw = min(
        max_power_mw,
        capacity_mwh / energy_smoothing_hours if energy_smoothing_hours else max_power_mw,
    )
    soc_mwh = capacity_mwh * 0.5
    dispatches = []
    controlled = []
    soc_values = []
    for row in result.itertuples(index=False):
        dispatch_mw = 0.0
        if row.predicted_net_load_mw >= peak_threshold_mw:
            requested = min(effective_max_power_mw, max(0.0, row.predicted_net_load_mw - peak_threshold_mw))
            available_mwh = soc_mwh if row.predicted_net_load_mw >= emergency_threshold else max(0.0, soc_mwh - reserve_mwh)
            dispatch_mw = min(requested, available_mwh * efficiency)
            soc_mwh -= dispatch_mw / efficiency
        elif row.predicted_net_load_mw <= low_charge_threshold:
            charge = min(charge_power_mw, (capacity_mwh - soc_mwh) / efficiency)
            dispatch_mw = -charge
            soc_mwh += charge * efficiency
        dispatches.append(dispatch_mw)
        controlled.append(row.net_load_mw - dispatch_mw)
        soc_values.append(soc_mwh)
    result["storage_dispatch_mw"] = dispatches
    result["storage_soc_mwh"] = soc_values
    result["controlled_net_load_mw"] = controlled
    triggered = result["predicted_net_load_mw"] >= peak_threshold_mw
    triggered_baseline = result.loc[triggered, "net_load_mw"]
    triggered_controlled = result.loc[triggered, "controlled_net_load_mw"]
    metrics = {
        "capacity_mwh": capacity_mwh,
        "max_discharge_power_mw": max_power_mw,
        "effective_discharge_power_mw": effective_max_power_mw,
        "energy_smoothing_hours": energy_smoothing_hours,
        "charge_power_mw": charge_power_mw,
        "round_trip_efficiency_proxy": efficiency**2,
        "reserve_fraction": reserve_fraction,
        "reserve_mwh": reserve_mwh,
        "baseline_max_net_load_mw": float(result["net_load_mw"].max()),
        "controlled_max_net_load_mw": float(result["controlled_net_load_mw"].max()),
        "theoretical_max_reduction_mw": float(result["net_load_mw"].max() - result["controlled_net_load_mw"].max()),
        "forecast_trigger_hours": int(triggered.sum()),
        "forecast_trigger_window_baseline_max_mw": float(triggered_baseline.max()),
        "forecast_trigger_window_controlled_max_mw": float(triggered_controlled.max()),
        "forecast_trigger_window_max_reduction_mw": float(triggered_baseline.max() - triggered_controlled.max()),
        "baseline_p95_net_load_mw": float(result["net_load_mw"].quantile(0.95)),
        "controlled_p95_net_load_mw": float(result["controlled_net_load_mw"].quantile(0.95)),
        "discharge_hours": int((result["storage_dispatch_mw"] > 0).sum()),
        "charge_hours": int((result["storage_dispatch_mw"] < 0).sum()),
        "min_soc_mwh": float(result["storage_soc_mwh"].min()),
        "max_soc_mwh": float(result["storage_soc_mwh"].max()),
        "forecast_peak_threshold_mw": peak_threshold_mw,
        "forecast_emergency_threshold_mw": emergency_threshold,
        "low_charge_forecast_threshold_mw": low_charge_threshold,
    }
    return result, metrics


def dynamic_reserve_fraction(
    upper_forecasts_mw: np.ndarray,
    interval_radii_mw: np.ndarray,
    peak_threshold_mw: float,
    base_fraction: float = 0.15,
    severity_weight: float = 0.20,
    uncertainty_weight: float = 0.15,
    uncertainty_scale_mw: float | None = None,
) -> float:
    """Set a transparent SOC reserve target from forecast stress and interval width."""
    peak_scale = max(0.10 * peak_threshold_mw, 1.0)
    severity = float(np.clip((np.max(upper_forecasts_mw) - peak_threshold_mw) / peak_scale, 0.0, 1.0))
    uncertainty = float(np.clip(np.max(interval_radii_mw) / (uncertainty_scale_mw or peak_scale), 0.0, 1.0))
    return float(np.clip(base_fraction + severity_weight * severity + uncertainty_weight * uncertainty, 0.0, 0.60))


def _solve_storage_window(
    upper_forecasts_mw: np.ndarray,
    lower_forecasts_mw: np.ndarray,
    soc_mwh: float,
    capacity_mwh: float,
    peak_threshold_mw: float,
    charge_threshold_mw: float,
    reserve_floor_mwh: float,
    max_discharge_power_mw: float,
    max_charge_power_mw: float,
    efficiency: float,
    soc_lower_bounds_mwh: np.ndarray | None = None,
    soc_upper_bounds_mwh: np.ndarray | None = None,
    allow_recovery_charge: bool = False,
    return_plan: bool = False,
    objective_forecasts_mw: np.ndarray | None = None,
) -> tuple[float, float, str] | tuple[np.ndarray, np.ndarray, str]:
    """Solve a min-max forecast peak LP, optionally returning the full plan.

    Objective forecasts may differ from the upper band used for action gates.
    The default preserves the original upper-band objective and constraints.
    """
    steps = len(upper_forecasts_mw)
    objective_forecasts = upper_forecasts_mw if objective_forecasts_mw is None else np.asarray(objective_forecasts_mw, dtype=float)
    if len(objective_forecasts) != steps or not np.isfinite(objective_forecasts).all():
        raise ValueError("Finite objective forecasts must match the planning window.")
    charge_start = 0
    discharge_start = steps
    peak_index = 2 * steps
    objective = np.zeros(2 * steps + 1)
    objective[peak_index] = 1.0
    # A small throughput penalty makes equivalent peak solutions less cycle-intensive.
    objective[: 2 * steps] = 0.001
    bounds = []
    for lower in lower_forecasts_mw:
        bounds.append((0.0, max_charge_power_mw if allow_recovery_charge or lower <= charge_threshold_mw else 0.0))
    for upper in upper_forecasts_mw:
        bounds.append((0.0, max_discharge_power_mw if upper >= peak_threshold_mw else 0.0))
    bounds.append((0.0, None))

    inequalities: list[np.ndarray] = []
    limits: list[float] = []
    for step, forecast in enumerate(objective_forecasts):
        row = np.zeros(2 * steps + 1)
        row[charge_start + step] = 1.0
        row[discharge_start + step] = -1.0
        row[peak_index] = -1.0
        inequalities.append(row)
        limits.append(-float(forecast))
    for step in range(steps):
        upper_soc = np.zeros(2 * steps + 1)
        lower_soc = np.zeros(2 * steps + 1)
        upper_soc[charge_start : charge_start + step + 1] = efficiency
        upper_soc[discharge_start : discharge_start + step + 1] = -1.0 / efficiency
        lower_soc[charge_start : charge_start + step + 1] = -efficiency
        lower_soc[discharge_start : discharge_start + step + 1] = 1.0 / efficiency
        inequalities.extend((upper_soc, lower_soc))
        upper_limit = capacity_mwh if soc_upper_bounds_mwh is None else soc_upper_bounds_mwh[step]
        lower_limit = reserve_floor_mwh if soc_lower_bounds_mwh is None else soc_lower_bounds_mwh[step]
        limits.extend((upper_limit - soc_mwh, soc_mwh - lower_limit))
    solution = linprog(
        objective,
        A_ub=np.vstack(inequalities),
        b_ub=np.asarray(limits),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        if return_plan:
            raise RuntimeError(f"Storage window infeasible: {solution.message}")
        return 0.0, 0.0, f"fallback:{solution.status}"
    if return_plan:
        return solution.x[discharge_start:discharge_start + steps], solution.x[:steps], "optimal"
    return float(solution.x[discharge_start]), float(solution.x[charge_start]), "optimal"


def _simulate_receding_horizon_dispatch(
    planning_panel: pd.DataFrame,
    peak_threshold_mw: float,
    capacity_mwh: float,
    max_power_mw: float = 2500.0,
    charge_power_mw: float = 1250.0,
    efficiency: float = 0.92,
    charge_threshold_mw: float | None = None,
    energy_smoothing_hours: float | None = 24.0,
    fixed_reserve_fraction: float | None = None,
    reserve_options: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run a direct-forecast rolling optimizer with either dynamic or fixed reserve."""
    result = planning_panel.copy().sort_values("timestamp").reset_index(drop=True)
    horizon_columns = sorted(
        int(column.removeprefix("prediction_h").removesuffix("_mw"))
        for column in result.columns
        if column.startswith("prediction_h") and column.endswith("_mw")
    )
    if not horizon_columns or horizon_columns != list(range(1, len(horizon_columns) + 1)):
        raise ValueError("Planning panel must contain contiguous prediction_h1_mw through prediction_hN_mw columns")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    if capacity_mwh <= 0 or max_power_mw < 0 or charge_power_mw < 0:
        raise ValueError("Capacity must be positive and power limits nonnegative.")
    if energy_smoothing_hours is not None and energy_smoothing_hours <= 0:
        raise ValueError("energy_smoothing_hours must be positive or None.")
    if charge_threshold_mw is None:
        raise ValueError("Supply a validation-calibrated charge threshold; test-period quantiles are not allowed.")
    if fixed_reserve_fraction is not None and not 0 <= fixed_reserve_fraction < 1:
        raise ValueError("fixed_reserve_fraction must be in [0, 1).")
    effective_max_power_mw = min(
        max_power_mw,
        capacity_mwh / energy_smoothing_hours if energy_smoothing_hours else max_power_mw,
    )
    charge_limit = float(
        charge_threshold_mw
    )
    soc_mwh = capacity_mwh * 0.5
    dispatches: list[float] = []
    soc_values: list[float] = []
    reserve_targets: list[float] = []
    reserve_floors: list[float] = []
    solver_status: list[str] = []
    for row in result.itertuples(index=False):
        upper = np.asarray([getattr(row, f"upper_h{hour}_mw") for hour in horizon_columns], dtype=float)
        lower = np.asarray([getattr(row, f"lower_h{hour}_mw") for hour in horizon_columns], dtype=float)
        prediction = np.asarray([getattr(row, f"prediction_h{hour}_mw") for hour in horizon_columns], dtype=float)
        reserve_fraction = (
            fixed_reserve_fraction
            if fixed_reserve_fraction is not None
            else dynamic_reserve_fraction(upper, upper - prediction, peak_threshold_mw, **(reserve_options or {}))
        )
        reserve_target = capacity_mwh * reserve_fraction
        reserve_floor = min(reserve_target, soc_mwh)
        discharge, charge, status = _solve_storage_window(
            upper_forecasts_mw=upper,
            lower_forecasts_mw=lower,
            soc_mwh=soc_mwh,
            capacity_mwh=capacity_mwh,
            peak_threshold_mw=peak_threshold_mw,
            charge_threshold_mw=charge_limit,
            reserve_floor_mwh=reserve_floor,
            max_discharge_power_mw=effective_max_power_mw,
            max_charge_power_mw=charge_power_mw,
            efficiency=efficiency,
        )
        dispatch = discharge - charge
        soc_mwh = float(np.clip(soc_mwh + charge * efficiency - discharge / efficiency, 0.0, capacity_mwh))
        dispatches.append(dispatch)
        soc_values.append(soc_mwh)
        reserve_targets.append(reserve_target)
        reserve_floors.append(reserve_floor)
        solver_status.append(status)
    result["storage_dispatch_mw"] = dispatches
    result["storage_soc_mwh"] = soc_values
    result["dynamic_reserve_target_mwh"] = reserve_targets
    result["dynamic_reserve_floor_mwh"] = reserve_floors
    result["solver_status"] = solver_status
    result["controlled_net_load_mw"] = result["net_load_mw"] - result["storage_dispatch_mw"]
    upper_h1 = result["upper_h1_mw"]
    metrics: dict[str, float | int | str] = {
        "policy": (
            "fixed_reserve_receding_horizon_linear_program"
            if fixed_reserve_fraction is not None
            else "uncertainty_aware_receding_horizon_linear_program"
        ),
        "reserve_policy": (
            f"fixed_{fixed_reserve_fraction:.0%}"
            if fixed_reserve_fraction is not None
            else "forecast_interval_dynamic"
        ),
        "planning_window_hours": len(horizon_columns),
        "capacity_mwh": capacity_mwh,
        "max_discharge_power_mw": max_power_mw,
        "effective_discharge_power_mw": effective_max_power_mw,
        "charge_power_mw": charge_power_mw,
        "round_trip_efficiency_proxy": efficiency**2,
        "forecast_peak_threshold_mw": peak_threshold_mw,
        "charge_forecast_threshold_mw": charge_limit,
        "baseline_max_net_load_mw": float(result["net_load_mw"].max()),
        "controlled_max_net_load_mw": float(result["controlled_net_load_mw"].max()),
        "theoretical_max_reduction_mw": float(result["net_load_mw"].max() - result["controlled_net_load_mw"].max()),
        "baseline_p95_net_load_mw": float(result["net_load_mw"].quantile(0.95)),
        "controlled_p95_net_load_mw": float(result["controlled_net_load_mw"].quantile(0.95)),
        "discharge_hours": int((result["storage_dispatch_mw"] > 0).sum()),
        "charge_hours": int((result["storage_dispatch_mw"] < 0).sum()),
        "throughput_mwh": float(result["storage_dispatch_mw"].abs().sum()),
        "equivalent_full_cycles": float(result["storage_dispatch_mw"].abs().sum() / (2 * capacity_mwh)),
        "min_soc_mwh": float(result["storage_soc_mwh"].min()),
        "max_soc_mwh": float(result["storage_soc_mwh"].max()),
        "final_soc_mwh": float(soc_mwh),
        "initial_soc_mwh": capacity_mwh * 0.5,
        "reserve_target_shortfall_hours": int(sum(soc < target - 1e-6 for soc, target in zip(soc_values, reserve_targets))),
        "mean_reserve_target_mwh": float(np.mean(reserve_targets)),
        "max_reserve_target_mwh": float(np.max(reserve_targets)),
        "reserve_limited_hours": int(sum(target > floor + 1e-6 for target, floor in zip(reserve_targets, reserve_floors))),
        "forecast_upper_peak_window_hours": int((upper_h1 >= peak_threshold_mw).sum()),
        "solver_optimal_rate_pct": float(np.mean([status == "optimal" for status in solver_status]) * 100),
    }
    return result, metrics


def simulate_risk_aware_dispatch(
    planning_panel: pd.DataFrame,
    peak_threshold_mw: float,
    capacity_mwh: float,
    max_power_mw: float = 2500.0,
    charge_power_mw: float = 1250.0,
    efficiency: float = 0.92,
    charge_threshold_mw: float | None = None,
    energy_smoothing_hours: float | None = 24.0,
    reserve_options: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run interval-driven, dynamic-reserve rolling planning on direct forecasts."""
    return _simulate_receding_horizon_dispatch(
        planning_panel,
        peak_threshold_mw=peak_threshold_mw,
        capacity_mwh=capacity_mwh,
        max_power_mw=max_power_mw,
        charge_power_mw=charge_power_mw,
        efficiency=efficiency,
        charge_threshold_mw=charge_threshold_mw,
        energy_smoothing_hours=energy_smoothing_hours,
        reserve_options=reserve_options,
    )


def simulate_fixed_reserve_receding_dispatch(
    planning_panel: pd.DataFrame,
    peak_threshold_mw: float,
    capacity_mwh: float,
    fixed_reserve_fraction: float = 0.30,
    max_power_mw: float = 2500.0,
    charge_power_mw: float = 1250.0,
    efficiency: float = 0.92,
    charge_threshold_mw: float | None = None,
    energy_smoothing_hours: float | None = 24.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Use the same direct forecasts and optimizer with a constant SOC reserve.

    This is the fairness baseline for the dynamic-reserve ablation: the only
    intended difference is how the reserve floor is selected.
    """
    return _simulate_receding_horizon_dispatch(
        planning_panel,
        peak_threshold_mw=peak_threshold_mw,
        capacity_mwh=capacity_mwh,
        max_power_mw=max_power_mw,
        charge_power_mw=charge_power_mw,
        efficiency=efficiency,
        charge_threshold_mw=charge_threshold_mw,
        energy_smoothing_hours=energy_smoothing_hours,
        fixed_reserve_fraction=fixed_reserve_fraction,
    )
