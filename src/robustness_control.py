"""Variable-horizon adapter around the frozen v5 LP and continuous plant replay."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.balanced_storage import complete_utc_days
from src.forecast_dispatch import physical_summary, replay
from src.optimality import StorageModel
from src.recovery_control import POLICIES, solve_window, validate_reference
from src.scenarios import dynamic_reserve_fraction
from src.tail_control import augment_metrics


def error_models():
    nominal = StorageModel()
    return {
        "power_90pct": replace(nominal, charge_power_mw=1125., discharge_power_mw=1125.),
        "power_70pct": replace(nominal, charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_090": replace(nominal, efficiency=.90),
        "efficiency_085": replace(nominal, efficiency=.85),
    }


def simulate_guard(panel, reference, calibration, gates, tail, model=StorageModel()):
    model.validate()
    if (not np.isfinite([tail, calibration["uncertainty_scale_mw"], *gates.values()]).all()
            or calibration["uncertainty_scale_mw"] <= 0
            or not gates["charge_threshold_mw"] <= tail <= gates["peak_threshold_mw"] <= gates["release_threshold_mw"]):
        raise ValueError("Finite ordered calibration required.")
    times = pd.DatetimeIndex(pd.to_datetime(panel.timestamp, utc=True))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Ordered unique planning times required.")
    data, omitted = complete_utc_days(panel)
    if omitted or len(data) != len(panel):
        raise ValueError("Prepare complete UTC days explicitly.")
    horizons = sorted(int(c.removeprefix("prediction_h").removesuffix("_mw"))
                      for c in data if c.startswith("prediction_h") and c.endswith("_mw"))
    if not horizons or horizons != list(range(1, len(horizons) + 1)):
        raise ValueError("Contiguous available horizons required.")
    cols = [f"{prefix}_h{h}_mw" for h in horizons for prefix in ("prediction", "lower", "upper")]
    if not np.isfinite(data[cols + ["net_load_mw"]].to_numpy(float)).all():
        raise ValueError("Finite forecasts and actuals required.")
    for h in horizons:
        if not ((data[f"lower_h{h}_mw"] <= data[f"prediction_h{h}_mw"]).all()
                and (data[f"prediction_h{h}_mw"] <= data[f"upper_h{h}_mw"]).all()):
            raise ValueError("Forecast intervals must bracket points.")
    if not data.available_timestamp.equals(data.timestamp):
        raise ValueError("Only forecasts available by the current hour may be used.")
    expected_issued = data.timestamp - pd.to_timedelta(data.observation_delay_hours + 1, unit="h")
    if not data.issued_timestamp.equals(expected_issued):
        raise ValueError("Invalid delayed information boundary.")
    ref = validate_reference(reference, data.timestamp, model)
    rows, soc = [], model.initial_soc_mwh
    for row, anchor in zip(data.itertuples(index=False), ref.itertuples(index=False)):
        if row.timestamp.hour == 0:
            soc = model.initial_soc_mwh
        if soc < anchor.soc_before_mwh - 1e-5:
            raise RuntimeError("Candidate state fell below available reference.")
        remaining = 24 - row.timestamp.hour
        steps = min(len(horizons), remaining)
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
        ceiling = np.minimum(model.maximum_soc_mwh, model.initial_soc_mwh
                             + after * model.discharge_power_mw / model.efficiency)
        c, d, info = solve_window(
            point, upper, soc, lower, ceiling, model, gates, tail, POLICIES["tail_guard"],
            anchor.storage_soc_mwh, anchor.charge_mw - anchor.discharge_mw)
        before, charge, discharge = soc, float(c[0]), float(d[0])
        soc += model.efficiency * charge - discharge / model.efficiency
        rows.append({
            "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
            "charge_mw": charge, "discharge_mw": discharge, "soc_before_mwh": before,
            "storage_soc_mwh": soc, "controlled_net_load_mw": row.net_load_mw + charge - discharge,
            "prediction_h1_mw": point[0], "upper_h1_mw": upper[0], "lower_h1_mw": row.lower_h1_mw,
            "reserve_target_mwh": target, "advisory_floor_mwh": floor, "reserve_release_active": bool(released),
            "reference_soc_before_mwh": anchor.soc_before_mwh, "reference_soc_after_mwh": anchor.storage_soc_mwh,
            "reference_charge_mw": anchor.charge_mw, "reference_discharge_mw": anchor.discharge_mw,
            "reference_debt_mwh": max(0., anchor.storage_soc_mwh - soc),
            "recovery_charge_mwh": charge if row.lower_h1_mw > gates["charge_threshold_mw"] else 0.,
            "solver_status": "recovery_optimal", **info,
        })
    trace = pd.DataFrame(rows)
    metrics = physical_summary(trace, model, terminal=True)
    metrics.update({
        "physical_parameters": asdict(model),
        "forecast_h1_mae_mw": float((trace.net_load_mw - trace.prediction_h1_mw).abs().mean()),
        "max_reference_debt_mwh": float(trace.reference_debt_mwh.max()),
        "max_headroom_violation_mw": float(np.maximum(
            trace.charge_mw - trace.discharge_mw - trace.headroom_cap_mw, 0.).max()),
    })
    if max(metrics["max_reference_debt_mwh"], metrics["max_headroom_violation_mw"]) > 1e-5:
        raise RuntimeError("Frozen guard constraints failed.")
    return trace, augment_metrics(trace, metrics, gates, tail)


def replay_mismatch(requests, actual_model, gates, tail):
    trace, metrics = replay(requests, actual_model)
    if not np.isclose(trace.soc_before_mwh.iloc[0], actual_model.initial_soc_mwh, rtol=0, atol=1e-5):
        raise RuntimeError("Unexpected replay initial energy.")
    if not trace.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Mismatch replay requires a continuous episode, no omitted-day bridges.")
    # Expose requests independently of the simulator's legacy source field names.
    for field in ("charge_mw", "discharge_mw"):
        trace[f"requested_{field}"] = requests[field].to_numpy(float)
    metrics = augment_metrics(trace, metrics, gates, tail)
    metrics["assumed_parameters"] = asdict(StorageModel())
    return trace, metrics


def annotate(trace, metrics, panel=None):
    trace, metrics = trace.copy(), dict(metrics)
    last = trace.timestamp.dt.hour.eq(23)
    initial = metrics["physical_parameters"]["initial_soc_mwh"]
    deviation = (trace.loc[last, "storage_soc_mwh"] - initial).abs()
    metrics["terminal_failure_days"] = int(deviation.gt(1e-5).sum())
    metrics["terminal_days"] = int(last.sum())
    if panel is not None:
        for column in ("source_first_target_timestamp", "issued_timestamp", "available_timestamp",
                       "observation_delay_hours", "forecast_hours"):
            trace[column] = panel[column].to_numpy()
        for column in ("prediction_h1_mw", "lower_h1_mw", "upper_h1_mw"):
            if column not in trace:
                trace[column] = panel[column].to_numpy()
        metrics["forecast_h1_mae_mw"] = float((panel.net_load_mw - panel.prediction_h1_mw).abs().mean())
        metrics["interval_coverage_pct"] = float(
            (panel.net_load_mw.ge(panel.lower_h1_mw) & panel.net_load_mw.le(panel.upper_h1_mw)).mean() * 100)
        metrics["observation_delay_hours"] = int(panel.observation_delay_hours.iloc[0])
        metrics["forecast_hours"] = int(panel.forecast_hours.iloc[0])
    return trace, metrics
