"""One-hour forecast-to-dispatch adapters; the existing LP remains the controller."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.balanced_storage import complete_utc_days, simulate_balanced_storage
from src.energy_twin import TwinConfig, TwinScenario, simulate_energy_twin
from src.optimality import StorageModel
from src.trajectory_audit import audited_summary

TEST_START = pd.Timestamp("2019-08-09T21:00:00Z")


def validate_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "net_load_mw", "prediction_net_load_mw"]
    if frame.empty or any(column not in frame for column in required):
        raise ValueError("Nonempty one-hour predictions and required columns mandatory.")
    data = frame[required].copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    stamps = pd.DatetimeIndex(data.timestamp)
    if (stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing
            or not stamps.equals(stamps.floor("h"))):
        raise ValueError("Unique ordered UTC-hour prediction timestamps required.")
    if not np.isfinite(data[required[1:]].to_numpy(float)).all():
        raise ValueError("Prediction and actual values must be finite.")
    return data


def shared_complete_days(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    if not frames:
        raise ValueError("At least one method required.")
    prepared = {name: validate_prediction(frame) for name, frame in frames.items()}
    common = None
    for frame in prepared.values():
        stamps = pd.Index(frame.timestamp)
        common = stamps if common is None else common.intersection(stamps)
    first = next(iter(prepared.values()))
    kept, omitted = complete_utc_days(first.loc[first.timestamp.isin(common)])
    output = {}
    for name, frame in prepared.items():
        result = frame.loc[frame.timestamp.isin(kept.timestamp)].reset_index(drop=True)
        if (not result.timestamp.equals(kept.timestamp)
                or not np.allclose(result.net_load_mw, kept.net_load_mw, rtol=0, atol=1e-6)):
            raise ValueError("Methods must share identical actual loads and timestamps.")
        output[name] = result
    return output, {
        "rows": len(kept), "days": len(kept) // 24,
        "start": kept.timestamp.min().isoformat(), "end": kept.timestamp.max().isoformat(),
        "omitted_partial_days": omitted,
        "excluded_rows_by_method": {name: len(frame) - len(kept) for name, frame in prepared.items()},
    }


def calibrate(validation: dict[str, pd.DataFrame]) -> dict:
    reference = next(iter(validation.values()))
    if reference.timestamp.max() >= TEST_START:
        raise ValueError("Calibration may not use test or later-year observations.")
    gates = {
        "peak_threshold_mw": float(reference.net_load_mw.quantile(.95)),
        "charge_threshold_mw": float(reference.net_load_mw.quantile(.25)),
        "release_threshold_mw": float(reference.net_load_mw.quantile(.99)),
    }
    methods = {}
    for name, frame in validation.items():
        if not frame.timestamp.equals(reference.timestamp):
            raise ValueError("Calibration must use shared validation timestamps.")
        errors = (frame.net_load_mw - frame.prediction_net_load_mw).abs()
        radii = errors.groupby(frame.timestamp.dt.hour).quantile(.9)
        counts = errors.groupby(frame.timestamp.dt.hour).size()
        if set(radii.index) != set(range(24)) or not np.isfinite(radii).all():
            raise ValueError("Every target hour requires finite validation calibration.")
        methods[name] = {
            "hourly_radius_mw": {str(k): float(v) for k, v in radii.items()},
            "hourly_sample_counts": {str(k): int(v) for k, v in counts.items()},
            "uncertainty_scale_mw": max(1., float(radii.max()) * 1.1),
        }
    return {"gates": gates, "methods": methods, "validation_rows": len(reference)}


def make_panel(frame: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    data = validate_prediction(frame)
    radii = {int(k): v for k, v in calibration["hourly_radius_mw"].items()}
    radius = data.timestamp.dt.hour.map(radii)
    if radius.isna().any() or not np.isfinite(radius).all() or radius.lt(0).any():
        raise ValueError("Invalid or missing calibrated radius.")
    data["prediction_h1_mw"] = data.prediction_net_load_mw
    data["lower_h1_mw"] = data.prediction_net_load_mw - radius
    data["upper_h1_mw"] = data.prediction_net_load_mw + radius
    return data


def physical_summary(trace: pd.DataFrame, model: StorageModel, *, terminal: bool) -> dict:
    audit = audited_summary(trace, model, daily_terminal=True)
    # Replay is not required to restore SOC; report that failure, never reset it away.
    if not terminal:
        audit["physical_audit_passed"] = max(
            audit[key] for key in ("energy_balance_error_mwh", "continuity_error_mwh",
                                  "bound_error", "load_accounting_error_mw")) <= 1e-5
        # Existing audit expects daily initial SOC. Replay carries state across days.
        continuity = (trace.soc_before_mwh.iloc[1:].to_numpy()
                      - trace.storage_soc_mwh.iloc[:-1].to_numpy())
        audit["continuity_error_mwh"] = float(np.abs(continuity).max()) if len(continuity) else 0.
        audit["physical_audit_passed"] = max(
            audit[key] for key in ("energy_balance_error_mwh", "continuity_error_mwh",
                                  "bound_error", "load_accounting_error_mw")) <= 1e-5
    else:
        consecutive = trace.timestamp.diff().eq(pd.Timedelta(hours=1)).to_numpy()[1:]
        delta = trace.soc_before_mwh.iloc[1:].to_numpy() - trace.storage_soc_mwh.iloc[:-1].to_numpy()
        audit["consecutive_hour_continuity_error_mwh"] = float(
            np.max(np.abs(delta[consecutive]))) if consecutive.any() else 0.
        audit["physical_audit_passed"] = bool(
            audit["physical_audit_passed"]
            and audit["consecutive_hour_continuity_error_mwh"] <= 1e-5
        )
    audit["terminal_requirement_passed"] = bool(audit["max_abs_terminal_error_mwh"] <= 1e-5)
    audit["physical_audit_passed"] = bool(audit["physical_audit_passed"])
    if not audit["physical_audit_passed"] or (terminal and not audit["terminal_requirement_passed"]):
        raise RuntimeError("Trajectory failed independent physical/terminal checks.")
    return audit


def dispatch(panel: pd.DataFrame, calibration: dict, gates: dict,
             model: StorageModel | None = None) -> tuple[pd.DataFrame, dict]:
    model = model or StorageModel()
    model.validate()
    if model.maximum_soc_mwh != model.initial_soc_mwh or model.initial_soc_mwh != model.capacity_mwh * .5:
        raise ValueError("This experiment uses a common 50% initial/operational ceiling.")
    trace, metrics = simulate_balanced_storage(
        panel, **gates, uncertainty_scale_mw=calibration["uncertainty_scale_mw"],
        capacity_mwh=model.capacity_mwh, charge_power_mw=model.charge_power_mw,
        discharge_power_mw=model.discharge_power_mw, efficiency=model.efficiency,
        hard_min_fraction=model.minimum_soc_mwh / model.capacity_mwh,
        daily_terminal=True, planning_objective="point",
    )
    audit = physical_summary(trace, model, terminal=True)
    metrics = {k: v for k, v in metrics.items() if k != "daily_soc"}
    metrics.update(audit)
    metrics["forecast_h1_mae_mw"] = float(
        (panel.net_load_mw - panel.prediction_h1_mw).abs().mean())
    metrics["physical_parameters"] = asdict(model)
    return trace, metrics


def replay(nominal: pd.DataFrame, changed: StorageModel) -> tuple[pd.DataFrame, dict]:
    config = TwinConfig(
        capacity_mwh=changed.capacity_mwh, charge_power_mw=changed.charge_power_mw,
        discharge_power_mw=changed.discharge_power_mw, efficiency=.92,
        minimum_soc_mwh=changed.minimum_soc_mwh, maximum_soc_mwh=changed.maximum_soc_mwh,
    )
    result = simulate_energy_twin(
        nominal, config, TwinScenario("changed_physics_replay", "Replay", efficiency=changed.efficiency))
    metrics = dict(result.metrics)
    metrics.update(physical_summary(result.trace, changed, terminal=False))
    metrics["physical_parameters"] = asdict(changed)
    return result.trace, metrics
