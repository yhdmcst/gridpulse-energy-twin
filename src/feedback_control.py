"""Offline hourly plant feedback, fixed-model planning and explicit recovery fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from src.forecast_dispatch import physical_summary
from src.optimality import StorageModel
from src.recovery_control import POLICIES, solve_window
from src.scenarios import _solve_storage_window, dynamic_reserve_fraction
from src.tail_control import augment_metrics


@dataclass(frozen=True)
class RecoveryEnvelope:
    charge_power_mw: float = 875.
    efficiency: float = .85
    max_observation_age_hours: float = 2.

    def validate(self, assumed):
        if (not np.isfinite(list(asdict(self).values())).all()
                or not 0 < self.charge_power_mw <= assumed.charge_power_mw
                or not 0 < self.efficiency <= assumed.efficiency
                or self.max_observation_age_hours < 1):
            raise ValueError("A finite declared lower recovery envelope is required.")


@dataclass(frozen=True)
class Packet:
    timestamp: pd.Timestamp
    source_timestamp: pd.Timestamp
    observed_timestamp: pd.Timestamp
    available_timestamp: pd.Timestamp
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def plant_models():
    m = StorageModel()
    return {
        "nominal": m,
        "power_90pct": replace(m, charge_power_mw=1125., discharge_power_mw=1125.),
        "power_70pct": replace(m, charge_power_mw=875., discharge_power_mw=875.),
        "efficiency_090": replace(m, efficiency=.90),
        "efficiency_085": replace(m, efficiency=.85),
        "combined90": replace(m, charge_power_mw=1125., discharge_power_mw=1125., efficiency=.90),
        "outside": replace(m, charge_power_mw=750., discharge_power_mw=750., efficiency=.80),
    }


def packet_from_row(row):
    n = int(row.forecast_hours)
    return Packet(
        pd.Timestamp(row.timestamp), pd.Timestamp(row.source_first_target_timestamp),
        pd.Timestamp(row.issued_timestamp), pd.Timestamp(row.available_timestamp),
        np.array([getattr(row, f"prediction_h{h}_mw") for h in range(1, n + 1)], float),
        np.array([getattr(row, f"lower_h{h}_mw") for h in range(1, n + 1)], float),
        np.array([getattr(row, f"upper_h{h}_mw") for h in range(1, n + 1)], float),
    )


def packet_gate(packet, envelope):
    stamps = (packet.timestamp, packet.source_timestamp, packet.observed_timestamp, packet.available_timestamp)
    if any(pd.isna(t) or t.tzinfo is None for t in stamps):
        return "invalid_packet_time"
    if any(t.tz_convert("UTC") != t.tz_convert("UTC").floor("h") for t in stamps):
        return "invalid_packet_time"
    if packet.available_timestamp > packet.timestamp or packet.source_timestamp > packet.timestamp:
        return "future_packet"
    if packet.observed_timestamp != packet.source_timestamp - pd.Timedelta(hours=1):
        return "invalid_observation_time"
    if packet.available_timestamp < packet.source_timestamp:
        return "invalid_availability_time"
    age = (packet.timestamp - packet.observed_timestamp) / pd.Timedelta(hours=1)
    if age < 1:
        return "invalid_observation_age"
    if age > envelope.max_observation_age_hours:
        return "forecast_expired"
    n = len(packet.point)
    if (not 1 <= n <= 6 or any(a.shape != (n,) or not np.isfinite(a).all()
                              for a in (packet.point, packet.lower, packet.upper))
            or np.any(packet.lower > packet.point) or np.any(packet.point > packet.upper)):
        return "invalid_forecast"
    return "ok"


def recovery_floor(timestamp, assumed, envelope):
    timestamp = pd.Timestamp(timestamp)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("A valid timezone-aware hourly decision time is required.")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError("A valid timezone-aware hourly decision time is required.")
    return max(assumed.minimum_soc_mwh, assumed.initial_soc_mwh
               - (23 - timestamp.hour) * envelope.charge_power_mw * envelope.efficiency)


def recovery_request(soc, timestamp, assumed, envelope):
    assumed.validate()
    envelope.validate(assumed)
    if not np.isfinite(soc) or not assumed.minimum_soc_mwh - 1e-5 <= soc <= assumed.maximum_soc_mwh + 1e-5:
        raise ValueError("Current finite in-range SOC telemetry required.")
    floor = recovery_floor(timestamp, assumed, envelope)
    needed = max(0., (floor - soc) / envelope.efficiency)
    return min(needed, envelope.charge_power_mw), 0., {
        "safe_floor_next_mwh": floor,
        "recovery_unreachable": bool(needed > envelope.charge_power_mw + 1e-5),
    }


def apply_supervisor(soc, timestamp, charge, discharge, assumed, envelope):
    fallback_c, _, info = recovery_request(soc, timestamp, assumed, envelope)
    c, d = charge, discharge
    if soc < info["safe_floor_next_mwh"]:
        c, d = max(charge, fallback_c), 0.
    else:
        d = min(discharge, max(0., (soc - info["safe_floor_next_mwh"]) * envelope.efficiency))
    worst = min(assumed.maximum_soc_mwh, soc + envelope.efficiency * min(c, envelope.charge_power_mw)
                - d / envelope.efficiency)
    info.update({
        "supervisor_intervened": bool(max(abs(c - charge), abs(d - discharge)) > 1e-6),
        "worst_case_next_soc_mwh": float(worst),
    })
    return float(c), float(d), info


def actuate(soc, charge, discharge, physical):
    physical.validate()
    if (not np.isfinite([soc, charge, discharge]).all() or min(charge, discharge) < -1e-6
            or min(charge, discharge) > 1e-6
            or not physical.minimum_soc_mwh - 1e-5 <= soc <= physical.maximum_soc_mwh + 1e-5):
        raise ValueError("Valid measured state and nonsimultaneous nonnegative request required.")
    c = min(max(0., charge), physical.charge_power_mw,
            max(0., (physical.maximum_soc_mwh - soc) / physical.efficiency))
    d = min(max(0., discharge), physical.discharge_power_mw,
            max(0., (soc - physical.minimum_soc_mwh) * physical.efficiency))
    after = soc + physical.efficiency * c - d / physical.efficiency
    if not physical.minimum_soc_mwh - 1e-5 <= after <= physical.maximum_soc_mwh + 1e-5:
        raise RuntimeError("Physical execution violated a hard SOC boundary.")
    return float(c), float(d), float(after)


def window_bounds(packet, soc, assumed, calibration, gates):
    n = min(len(packet.point), 24 - packet.timestamp.hour)
    point, upper, lower = packet.point[:n], packet.upper[:n], packet.lower[:n]
    fraction = dynamic_reserve_fraction(
        upper, upper - point, gates["peak_threshold_mw"],
        uncertainty_scale_mw=calibration["uncertainty_scale_mw"])
    target = fraction * assumed.capacity_mwh
    released = point.max() >= gates["release_threshold_mw"]
    floor = assumed.minimum_soc_mwh if released else max(assumed.minimum_soc_mwh, min(target, soc))
    remaining_after = 24 - packet.timestamp.hour - np.arange(1, n + 1)
    lb = np.maximum(floor, assumed.initial_soc_mwh
                    - remaining_after * assumed.charge_power_mw * assumed.efficiency)
    ub = np.minimum(assumed.maximum_soc_mwh, assumed.initial_soc_mwh
                    + remaining_after * assumed.discharge_power_mw / assumed.efficiency)
    return point, upper, lower, floor, lb, ub


def reference_decision(packet, soc, assumed, calibration, gates, envelope, outage=False):
    gate = packet_gate(packet, envelope)
    reason = gate if gate != "ok" else ("solver_unavailable" if outage else "ok")
    attempted, failed, error = False, False, ""
    if reason == "ok":
        attempted = True
        point, upper, lower, floor, lb, ub = window_bounds(packet, soc, assumed, calibration, gates)
        try:
            d, c, _ = _solve_storage_window(
                upper, lower, soc, assumed.capacity_mwh, gates["peak_threshold_mw"],
                gates["charge_threshold_mw"], floor, assumed.discharge_power_mw,
                assumed.charge_power_mw, assumed.efficiency, lb, ub,
                allow_recovery_charge=True, return_plan=True, objective_forecasts_mw=point)
            charge, discharge = float(c[0]), float(d[0])
        except RuntimeError as exc:
            failed, error, reason = True, str(exc), "reference_solver_failure"
    if reason != "ok":
        charge, discharge, _ = recovery_request(soc, packet.timestamp, assumed, envelope)
    return charge, discharge, {
        "gate_reason": reason, "solver_attempted": attempted, "solver_failed": failed, "solver_error": error,
    }


def candidate_decision(packet, measured_soc, reference_row, assumed, calibration, gates,
                       tail, envelope, supervised, outage=False):
    assumed.validate()
    envelope.validate(assumed)
    if not np.isfinite(measured_soc) or not assumed.minimum_soc_mwh - 1e-5 <= measured_soc <= assumed.maximum_soc_mwh + 1e-5:
        raise ValueError("Current finite plant SOC required.")
    gate = packet_gate(packet, envelope)
    reason = gate if gate != "ok" else ("solver_unavailable" if outage else "ok")
    if reason == "ok" and reference_row.solver_failed:
        reason = "reference_solver_failure"
    attempted, failed, error, cap = False, False, "", None
    if reason == "ok":
        attempted = True
        point, upper, _, _, lb, ub = window_bounds(packet, measured_soc, assumed, calibration, gates)
        try:
            c, d, info = solve_window(
                point, upper, measured_soc, lb, ub, assumed, gates, tail, POLICIES["tail_guard"],
                reference_row.storage_soc_mwh, reference_row.charge_mw - reference_row.discharge_mw)
            charge, discharge, cap = float(c[0]), float(d[0]), info["headroom_cap_mw"]
        except RuntimeError as exc:
            failed, error, reason = True, str(exc), "primary_solver_failure"
    if reason != "ok":
        charge, discharge, _ = recovery_request(measured_soc, packet.timestamp, assumed, envelope)
    primary_charge, primary_discharge = charge, discharge
    _, _, recovery = recovery_request(measured_soc, packet.timestamp, assumed, envelope)
    worst = min(assumed.maximum_soc_mwh, measured_soc
                + envelope.efficiency * min(charge, envelope.charge_power_mw) - discharge / envelope.efficiency)
    recovery.update({"supervisor_intervened": False, "worst_case_next_soc_mwh": float(worst)})
    if supervised:
        charge, discharge, recovery = apply_supervisor(
            measured_soc, packet.timestamp, charge, discharge, assumed, envelope)
    return charge, discharge, {
        "gate_reason": reason, "primary_solver_attempted": attempted, "primary_solver_failed": failed,
        "primary_solver_error": error, "fallback_active": reason != "ok",
        "primary_charge_mw": primary_charge, "primary_discharge_mw": primary_discharge,
        "headroom_cap_mw": cap,
        "post_supervisor_headroom_excess_mw": max(0., charge - discharge - cap) if cap is not None else None,
        **recovery,
    }


def event_flags(index, timestamp, event):
    if event not in ("healthy", "stale_burst", "solver_outage", "combined_fault"):
        raise ValueError("Unknown declared event schedule.")
    selected_day = (index // 24) % 7 == 0
    stale = event in ("stale_burst", "combined_fault") and selected_day and timestamp.hour in (14, 15, 16, 17)
    outage = event in ("solver_outage", "combined_fault") and selected_day and timestamp.hour == 10
    return bool(stale), bool(outage)


def prepare_panels(base, stale):
    for frame in (base, stale):
        if not isinstance(frame.timestamp.dtype, pd.DatetimeTZDtype) or str(frame.timestamp.dt.tz) != "UTC":
            raise ValueError("Planning panels require timezone-aware UTC timestamps.")
        times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
        if (frame.empty or times.hasnans or times.has_duplicates
                or not times.equals(pd.date_range(times[0], periods=len(times), freq="h"))
                or times[0].hour != 0 or times[-1].hour != 23):
            raise ValueError("A continuous complete-day episode is required.")
    if (not base.timestamp.equals(stale.timestamp)
            or not np.allclose(base.net_load_mw, stale.net_load_mw, rtol=0, atol=1e-6)):
        raise ValueError("Healthy and stale packet streams must share scoring actuals and times.")
    return list(base.itertuples(index=False)), list(stale.itertuples(index=False))


def summarize_continuous(trace, physical, gates, tail):
    if not trace.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("No missing-hour episode bridges.")
    if abs(trace.soc_before_mwh.iloc[0] - physical.initial_soc_mwh) > 1e-5:
        raise RuntimeError("Unexpected initial energy.")
    metrics = physical_summary(trace, physical, terminal=False)
    metrics["physical_parameters"] = asdict(physical)
    metrics = augment_metrics(trace, metrics, gates, tail)
    last = trace.timestamp.dt.hour.eq(23)
    metrics["terminal_days"] = int(last.sum())
    metrics["terminal_failure_days"] = int(
        (trace.loc[last, "storage_soc_mwh"] - physical.initial_soc_mwh).abs().gt(1e-5).sum())
    return metrics


def make_reference(base, stale, event, calibration, gates, tail, assumed=StorageModel(),
                   envelope=RecoveryEnvelope()):
    assumed.validate()
    envelope.validate(assumed)
    base_rows, stale_rows = prepare_panels(base, stale)
    soc, rows = assumed.initial_soc_mwh, []
    for i, (row, stale_row) in enumerate(zip(base_rows, stale_rows)):
        expired, outage = event_flags(i, row.timestamp, event)
        packet = packet_from_row(stale_row if expired else row)
        req_c, req_d, status = reference_decision(
            packet, soc, assumed, calibration, gates, envelope, outage)
        c, d, after = actuate(soc, req_c, req_d, assumed)
        rows.append({
            "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
            "charge_mw": c, "discharge_mw": d, "soc_before_mwh": soc, "storage_soc_mwh": after,
            "controlled_net_load_mw": row.net_load_mw + c - d,
            "requested_charge_mw": req_c, "requested_discharge_mw": req_d,
            "observed_timestamp": packet.observed_timestamp, "available_timestamp": packet.available_timestamp,
            "source_timestamp": packet.source_timestamp, "injected_stale": expired, "injected_outage": outage,
            "recovery_charge_mwh": c if np.isfinite(packet.lower[0]) and packet.lower[0] > gates["charge_threshold_mw"] else 0.,
            **status,
        })
        soc = after
    trace = pd.DataFrame(rows)
    metrics = summarize_continuous(trace, assumed, gates, tail)
    metrics.update({"assumed_parameters": asdict(assumed), "recovery_envelope": asdict(envelope),
                    "event": event, "reference_only": True})
    return trace, metrics


def simulate_feedback(base, stale, reference, event, physical, supervised, calibration, gates, tail,
                      assumed=StorageModel(), envelope=RecoveryEnvelope()):
    physical.validate()
    assumed.validate()
    envelope.validate(assumed)
    if not (physical.capacity_mwh == assumed.capacity_mwh
            and physical.minimum_soc_mwh == assumed.minimum_soc_mwh
            and physical.maximum_soc_mwh == assumed.maximum_soc_mwh
            and physical.initial_soc_mwh == assumed.initial_soc_mwh):
        raise ValueError("This experiment fixes energy bounds; only power/efficiency may differ.")
    base_rows, stale_rows = prepare_panels(base, stale)
    if not reference.timestamp.equals(base.timestamp):
        raise ValueError("Causal reference timestamps must align.")
    # Validate the modeled reference continuously, without enforcing physical-plant daily resets.
    summarize_continuous(reference, assumed, gates, tail)
    soc, rows = physical.initial_soc_mwh, []
    for i, (row, old, anchor) in enumerate(zip(base_rows, stale_rows, reference.itertuples(index=False))):
        expired, outage = event_flags(i, row.timestamp, event)
        packet = packet_from_row(old if expired else row)
        req_c, req_d, status = candidate_decision(
            packet, soc, anchor, assumed, calibration, gates, tail, envelope, supervised, outage)
        c, d, after = actuate(soc, req_c, req_d, physical)
        floor = status["safe_floor_next_mwh"]
        rows.append({
            "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
            "charge_mw": c, "discharge_mw": d, "soc_before_mwh": soc, "storage_soc_mwh": after,
            "measured_soc_mwh": soc, "controlled_net_load_mw": row.net_load_mw + c - d,
            "requested_charge_mw": req_c, "requested_discharge_mw": req_d,
            "predicted_next_soc_mwh": soc + assumed.efficiency * req_c - req_d / assumed.efficiency,
            "reference_soc_before_mwh": anchor.soc_before_mwh,
            "reference_soc_after_mwh": anchor.storage_soc_mwh,
            "reference_charge_mw": anchor.charge_mw, "reference_discharge_mw": anchor.discharge_mw,
            "actual_reference_debt_mwh": max(0., anchor.storage_soc_mwh - after),
            "reference_solver_failed": anchor.solver_failed,
            "observed_timestamp": packet.observed_timestamp, "available_timestamp": packet.available_timestamp,
            "source_timestamp": packet.source_timestamp,
            "observation_age_hours": float((packet.timestamp - packet.observed_timestamp) / pd.Timedelta(hours=1)),
            "injected_stale": expired, "injected_outage": outage,
            "actual_floor_violation_mwh": max(0., floor - after),
            "recovery_charge_mwh": c if np.isfinite(packet.lower[0]) and packet.lower[0] > gates["charge_threshold_mw"] else 0.,
            **status,
        })
        soc = after
    trace = pd.DataFrame(rows)
    metrics = summarize_continuous(trace, physical, gates, tail)
    inside = physical.charge_power_mw >= envelope.charge_power_mw and physical.efficiency >= envelope.efficiency
    metrics.update({
        "assumed_parameters": asdict(assumed), "recovery_envelope": asdict(envelope),
        "supervised": bool(supervised), "plant_within_declared_envelope": bool(inside),
        "fallback_hours": int(trace.fallback_active.sum()),
        "primary_solver_attempts": int(trace.primary_solver_attempted.sum()),
        "primary_solver_failure_hours": int(trace.primary_solver_failed.sum()),
        "reference_solver_failure_hours": int(trace.reference_solver_failed.sum()),
        "expiry_rejections": int(trace.gate_reason.eq("forecast_expired").sum()),
        "outage_fallback_hours": int(trace.gate_reason.eq("solver_unavailable").sum()),
        "supervisor_interventions": int(trace.supervisor_intervened.sum()),
        "recovery_unreachable_hours": int(trace.recovery_unreachable.sum()),
        "max_actual_floor_violation_mwh": float(trace.actual_floor_violation_mwh.max()),
        "max_worst_case_floor_violation_mwh": float(
            np.maximum(trace.safe_floor_next_mwh - trace.worst_case_next_soc_mwh, 0.).max()),
        "actual_clipping_hours": int(
            ((trace.requested_charge_mw - trace.charge_mw).abs().gt(1e-5)
             | (trace.requested_discharge_mw - trace.discharge_mw).abs().gt(1e-5)).sum()),
        "unexecuted_charge_mwh": float((trace.requested_charge_mw - trace.charge_mw).sum()),
        "unexecuted_discharge_mwh": float((trace.requested_discharge_mw - trace.discharge_mw).sum()),
        "max_actual_reference_debt_mwh": float(trace.actual_reference_debt_mwh.max()),
        "gated_optimizer_calls": int(
            ((trace.injected_stale | trace.injected_outage) & trace.primary_solver_attempted).sum()),
    })
    return trace, metrics
