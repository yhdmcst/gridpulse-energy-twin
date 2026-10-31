"""Physics-constrained energy digital twin for storage trajectory replay.

The twin mirrors the state variables that are already present in GridPulse:
net load, charge/discharge power, SOC, efficiency, reserve floor, and hourly
time. It is intentionally a replay and counterfactual layer, not a live
dispatch controller or a claim of a complete grid twin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TwinConfig:
    """Physical limits used by the replay twin."""

    capacity_mwh: float = 30000.0
    charge_power_mw: float = 1250.0
    discharge_power_mw: float = 1250.0
    efficiency: float = 0.92
    minimum_soc_mwh: float = 4500.0
    maximum_soc_mwh: float = 30000.0
    tolerance_mwh: float = 1e-6

    def validate(self) -> None:
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("Twin parameters must be finite.")
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh must be positive.")
        if self.charge_power_mw < 0 or self.discharge_power_mw < 0:
            raise ValueError("Power limits must be nonnegative.")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if not 0 <= self.minimum_soc_mwh <= self.maximum_soc_mwh <= self.capacity_mwh:
            raise ValueError("SOC bounds must be ordered within capacity.")
        if self.tolerance_mwh <= 0:
            raise ValueError("tolerance_mwh must be positive.")


@dataclass(frozen=True)
class TwinScenario:
    """A transparent counterfactual perturbation applied to a stored trace."""

    key: str
    label: str
    solar_drop_fraction: float = 0.0
    solar_drop_mw: float = 0.0
    net_load_bias_mw: float = 0.0
    efficiency: float | None = None
    power_scale: float = 1.0
    action_delay_hours: int = 0

    def validate(self) -> None:
        values = [self.solar_drop_fraction, self.solar_drop_mw,
                  self.net_load_bias_mw, self.power_scale]
        if self.efficiency is not None:
            values.append(self.efficiency)
        if not np.isfinite(values).all():
            raise ValueError("Scenario parameters must be finite.")
        if not 0 <= self.solar_drop_fraction <= 1:
            raise ValueError("solar_drop_fraction must be in [0, 1].")
        if self.solar_drop_mw < 0:
            raise ValueError("solar_drop_mw must be nonnegative.")
        if self.efficiency is not None and not 0 < self.efficiency <= 1:
            raise ValueError("Scenario efficiency must be in (0, 1].")
        if not 0 < self.power_scale <= 1:
            raise ValueError("power_scale must be in (0, 1].")
        if (isinstance(self.action_delay_hours, bool)
                or not isinstance(self.action_delay_hours, int)
                or self.action_delay_hours < 0):
            raise ValueError("action_delay_hours must be a nonnegative integer.")


@dataclass(frozen=True)
class TwinResult:
    """Counterfactual trajectory plus auditable summary metrics."""

    trace: pd.DataFrame
    metrics: dict[str, float | int | str | bool]


def default_twin_scenarios() -> dict[str, TwinScenario]:
    """Return the scenarios exposed by the dashboard."""

    return {
        "nominal": TwinScenario("nominal", "名义情景"),
        "pv_drop_30pct": TwinScenario(
            "pv_drop_30pct",
            "光伏骤降 30%（无光伏字段时使用 +1,500 MW 代理）",
            solar_drop_fraction=0.30,
            solar_drop_mw=1500.0,
        ),
        "storage_derate_30pct": TwinScenario(
            "storage_derate_30pct",
            "储能可用功率下降 30%",
            power_scale=0.70,
        ),
        "efficiency_085": TwinScenario(
            "efficiency_085",
            "充放电效率降至 0.85",
            efficiency=0.85,
        ),
        "feed_delay_3h": TwinScenario(
            "feed_delay_3h",
            "控制动作延迟 3 小时",
            action_delay_hours=3,
        ),
    }


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Twin input trace cannot be empty.")
    if "timestamp" not in frame or "net_load_mw" not in frame:
        raise ValueError("Twin input requires timestamp and net_load_mw.")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    if result["timestamp"].isna().any() or result["timestamp"].duplicated().any():
        raise ValueError("Twin timestamps must be valid and unique.")
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("Twin timestamps must be ordered.")
    result = result.reset_index(drop=True)
    if not result["timestamp"].eq(result["timestamp"].dt.floor("h")).all():
        raise ValueError("Twin timestamps must align to UTC hours.")
    expected = pd.date_range(result["timestamp"].iloc[0], periods=len(result), freq="h")
    if not pd.DatetimeIndex(result["timestamp"]).equals(expected):
        raise ValueError("Twin input must be a contiguous hourly trace.")
    numeric = [column for column in (
        "net_load_mw",
        "solar_generation_mw",
        "charge_mw",
        "discharge_mw",
        "soc_before_mwh",
        "storage_soc_mwh",
    ) if column in result]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Twin numeric inputs must be finite.")
    for column in ("charge_mw", "discharge_mw", "soc_before_mwh", "storage_soc_mwh"):
        if column not in result:
            raise ValueError(f"Twin input requires {column}.")
    if (result[["charge_mw", "discharge_mw"]] < 0).any().any():
        raise ValueError("Source powers must be nonnegative.")
    if "solar_generation_mw" in result and result["solar_generation_mw"].lt(0).any():
        raise ValueError("Solar generation must be nonnegative.")
    return result


def _source_physics_residual(frame: pd.DataFrame, efficiency: float) -> pd.Series:
    if not {"storage_soc_mwh", "soc_before_mwh"}.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return (
        frame["storage_soc_mwh"]
        - frame["soc_before_mwh"]
        - efficiency * frame["charge_mw"]
        + frame["discharge_mw"] / efficiency
    )


def simulate_energy_twin(
    frame: pd.DataFrame,
    config: TwinConfig,
    scenario: TwinScenario | None = None,
) -> TwinResult:
    """Replay a trajectory under explicit energy-balance constraints.

    The function applies a scenario to the observed net load and the stored
    actions, then recomputes SOC and controlled net load from the physical
    equations. It never uses future observations to alter an action.
    """

    config.validate()
    scenario = scenario or TwinScenario("nominal", "名义情景")
    scenario.validate()
    data = _validate_frame(frame)
    efficiency = config.efficiency if scenario.efficiency is None else scenario.efficiency
    max_charge = config.charge_power_mw * scenario.power_scale
    max_discharge = config.discharge_power_mw * scenario.power_scale

    net_load = data["net_load_mw"].to_numpy(dtype=float).copy()
    if "solar_generation_mw" in data and scenario.solar_drop_fraction:
        net_load += data["solar_generation_mw"].to_numpy(dtype=float) * scenario.solar_drop_fraction
    elif scenario.solar_drop_fraction:
        net_load += scenario.solar_drop_mw
    net_load += scenario.net_load_bias_mw

    source_charge = data["charge_mw"].to_numpy(dtype=float)
    source_discharge = data["discharge_mw"].to_numpy(dtype=float)
    if scenario.action_delay_hours:
        charge = np.roll(source_charge, scenario.action_delay_hours)
        discharge = np.roll(source_discharge, scenario.action_delay_hours)
        charge[:scenario.action_delay_hours] = 0.0
        discharge[:scenario.action_delay_hours] = 0.0
    else:
        charge = source_charge.copy()
        discharge = source_discharge.copy()

    initial_soc = (
        float(data["soc_before_mwh"].iloc[0])
        if "soc_before_mwh" in data
        else min(config.maximum_soc_mwh, config.capacity_mwh * 0.5)
    )
    if not config.minimum_soc_mwh - config.tolerance_mwh <= initial_soc <= config.maximum_soc_mwh + config.tolerance_mwh:
        raise ValueError("Initial SOC is outside the twin bounds.")

    rows: list[dict[str, float | str | bool]] = []
    soc = initial_soc
    simultaneous_hours = 0
    clipped_charge_hours = 0
    clipped_discharge_hours = 0
    hard_bound_hours = 0

    for index, row in data.iterrows():
        requested_charge = max(0.0, min(float(charge[index]), max_charge))
        requested_discharge = max(0.0, min(float(discharge[index]), max_discharge))
        if charge[index] > max_charge + config.tolerance_mwh:
            clipped_charge_hours += 1
        if discharge[index] > max_discharge + config.tolerance_mwh:
            clipped_discharge_hours += 1
        simultaneous = requested_charge > config.tolerance_mwh and requested_discharge > config.tolerance_mwh
        if simultaneous:
            simultaneous_hours += 1
            shared = min(requested_charge, requested_discharge)
            requested_charge -= shared
            requested_discharge -= shared

        charge_limit = max(0.0, (config.maximum_soc_mwh - soc) / efficiency)
        discharge_limit = max(0.0, (soc - config.minimum_soc_mwh) * efficiency)
        actual_charge = min(requested_charge, charge_limit)
        actual_discharge = min(requested_discharge, discharge_limit)
        next_soc = soc + efficiency * actual_charge - actual_discharge / efficiency
        if next_soc < config.minimum_soc_mwh - config.tolerance_mwh or next_soc > config.maximum_soc_mwh + config.tolerance_mwh:
            hard_bound_hours += 1
        next_soc = float(np.clip(next_soc, config.minimum_soc_mwh, config.maximum_soc_mwh))
        controlled = float(net_load[index] + actual_charge - actual_discharge)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "source_net_load_mw": float(row["net_load_mw"]),
                "net_load_mw": float(net_load[index]),
                "source_charge_mw": float(source_charge[index]),
                "source_discharge_mw": float(source_discharge[index]),
                "charge_mw": float(actual_charge),
                "discharge_mw": float(actual_discharge),
                "soc_before_mwh": float(soc),
                "storage_soc_mwh": next_soc,
                "controlled_net_load_mw": controlled,
                "reserve_target_mwh": float(row.get("reserve_target_mwh", np.nan)),
                "scenario_name": scenario.key,
                "simultaneous_action": simultaneous,
            }
        )
        soc = next_soc

    trace = pd.DataFrame(rows)
    source_residual = _source_physics_residual(data, config.efficiency)
    finite_source_residual = source_residual.dropna()
    max_source_residual = (
        float(finite_source_residual.abs().max()) if not finite_source_residual.empty else 0.0
    )
    health_score = 100.0
    if max_source_residual > config.tolerance_mwh:
        health_score -= min(40.0, 10.0 + 10.0 * max_source_residual)
    if hard_bound_hours:
        health_score -= 40.0
    if simultaneous_hours:
        health_score -= min(20.0, float(simultaneous_hours))
    health_score = float(np.clip(health_score, 0.0, 100.0))

    metrics: dict[str, float | int | str | bool] = {
        "scenario_name": scenario.key,
        "scenario_label": scenario.label,
        "rows": len(trace),
        "start": trace["timestamp"].iloc[0].isoformat(),
        "end": trace["timestamp"].iloc[-1].isoformat(),
        "effective_efficiency": efficiency,
        "power_scale": scenario.power_scale,
        "action_delay_hours": scenario.action_delay_hours,
        "baseline_peak_mw": float(trace["net_load_mw"].max()),
        "controlled_peak_mw": float(trace["controlled_net_load_mw"].max()),
        "baseline_p95_mw": float(trace["net_load_mw"].quantile(0.95)),
        "controlled_p95_mw": float(trace["controlled_net_load_mw"].quantile(0.95)),
        "min_soc_mwh": float(min(initial_soc, trace["storage_soc_mwh"].min())),
        "max_soc_mwh": float(trace["storage_soc_mwh"].max()),
        "throughput_mwh": float((trace["charge_mw"] + trace["discharge_mw"]).sum()),
        "conversion_loss_mwh": float(
            ((1.0 - efficiency) * trace["charge_mw"]
             + (1.0 / efficiency - 1.0) * trace["discharge_mw"]).sum()
        ),
        "max_source_physics_residual_mwh": max_source_residual,
        "clipped_charge_hours": clipped_charge_hours,
        "clipped_discharge_hours": clipped_discharge_hours,
        "simultaneous_action_hours": simultaneous_hours,
        "hard_bound_hours": hard_bound_hours,
        "health_score": health_score,
        "health_status": "可信" if health_score >= 95.0 else "需要复核",
        "physical_replay_passed": bool(
            hard_bound_hours == 0
            and np.isfinite(
                trace[["storage_soc_mwh", "controlled_net_load_mw"]].to_numpy(dtype=float)
            ).all()
        ),
    }
    source_continuity = (
        data["soc_before_mwh"].iloc[1:].to_numpy()
        - data["storage_soc_mwh"].iloc[:-1].to_numpy()
    )
    replay_balance = (
        trace["storage_soc_mwh"] - trace["soc_before_mwh"]
        - efficiency * trace["charge_mw"] + trace["discharge_mw"] / efficiency
    )
    metrics.update({
        "max_source_continuity_error_mwh": float(np.max(np.abs(source_continuity)))
        if len(source_continuity) else 0.0,
        "replay_energy_error_mwh": float(replay_balance.abs().max()),
        "initial_soc_mwh": initial_soc,
        "final_soc_mwh": float(soc),
        "terminal_deviation_mwh": float(soc - initial_soc),
        "unexecuted_charge_mwh": float((charge - trace["charge_mw"].to_numpy()).sum()),
        "unexecuted_discharge_mwh": float((discharge - trace["discharge_mw"].to_numpy()).sum()),
        "soc_limited_hours": int(
            ((trace["charge_mw"].to_numpy() < np.minimum(charge, max_charge) - 1e-6)
             | (trace["discharge_mw"].to_numpy() < np.minimum(discharge, max_discharge) - 1e-6)).sum()
        ),
        "source_audit_passed": bool(
            max_source_residual <= config.tolerance_mwh
            and (not len(source_continuity)
                 or np.max(np.abs(source_continuity)) <= config.tolerance_mwh)
        ),
    })
    if not metrics["source_audit_passed"]:
        metrics["health_score"] = min(float(metrics["health_score"]), 60.0)
        metrics["health_status"] = "需要复核"
    metrics["physical_replay_passed"] = bool(
        metrics["physical_replay_passed"]
        and metrics["replay_energy_error_mwh"] <= config.tolerance_mwh
    )
    return TwinResult(trace=trace, metrics=metrics)
