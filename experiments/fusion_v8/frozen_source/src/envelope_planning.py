"""Planning-time conservative model, using frozen v7 actuation and fallback rules."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from src.feedback_control import (
    RecoveryEnvelope, actuate, apply_supervisor, event_flags, packet_from_row,
    prepare_panels, reference_decision, simulate_feedback, summarize_continuous,
)
from src.optimality import StorageModel

POLICIES = ("envelope_reference", "envelope_tail")


def planning_model(envelope=RecoveryEnvelope()):
    model = StorageModel()
    envelope.validate(model)
    return replace(model, charge_power_mw=envelope.charge_power_mw,
                   discharge_power_mw=envelope.charge_power_mw, efficiency=envelope.efficiency)


def simulate_policy(base, stale, reference, event, physical, policy, calibration, gates, tail,
                    envelope=RecoveryEnvelope()):
    if policy not in POLICIES:
        raise ValueError("Unknown predeclared envelope planning policy.")
    assumed = planning_model(envelope)
    if policy == "envelope_tail":
        trace, metrics = simulate_feedback(
            base, stale, reference, event, physical, True, calibration, gates, tail,
            assumed=assumed, envelope=envelope)
    else:
        physical.validate()
        if any(getattr(physical, key) != getattr(assumed, key) for key in (
            "capacity_mwh", "minimum_soc_mwh", "maximum_soc_mwh", "initial_soc_mwh")):
            raise ValueError("Energy bounds must match the fixed planning contract.")
        rows_base, rows_stale = prepare_panels(base, stale)
        if not reference.timestamp.equals(base.timestamp):
            raise ValueError("Current causal reference rows must align.")
        summarize_continuous(reference, assumed, gates, tail)
        soc, rows = physical.initial_soc_mwh, []
        for i, (row, old, anchor) in enumerate(zip(
                rows_base, rows_stale, reference.itertuples(index=False))):
            expired, outage = event_flags(i, row.timestamp, event)
            packet = packet_from_row(old if expired else row)
            primary_c, primary_d, status = reference_decision(
                packet, soc, assumed, calibration, gates, envelope, outage)
            req_c, req_d, supervision = apply_supervisor(
                soc, packet.timestamp, primary_c, primary_d, assumed, envelope)
            c, d, after = actuate(soc, req_c, req_d, physical)
            rows.append({
                "timestamp": row.timestamp, "net_load_mw": row.net_load_mw,
                "charge_mw": c, "discharge_mw": d, "soc_before_mwh": soc,
                "storage_soc_mwh": after, "measured_soc_mwh": soc,
                "controlled_net_load_mw": row.net_load_mw + c - d,
                "requested_charge_mw": req_c, "requested_discharge_mw": req_d,
                "predicted_next_soc_mwh": soc + assumed.efficiency * req_c - req_d / assumed.efficiency,
                "reference_soc_before_mwh": anchor.soc_before_mwh,
                "reference_soc_after_mwh": anchor.storage_soc_mwh,
                "reference_charge_mw": anchor.charge_mw, "reference_discharge_mw": anchor.discharge_mw,
                "actual_reference_debt_mwh": max(0., anchor.storage_soc_mwh - after),
                "reference_solver_failed": anchor.solver_failed,
                "observed_timestamp": packet.observed_timestamp,
                "available_timestamp": packet.available_timestamp, "source_timestamp": packet.source_timestamp,
                "observation_age_hours": float((packet.timestamp - packet.observed_timestamp).total_seconds() / 3600.),
                "injected_stale": expired, "injected_outage": outage,
                "actual_floor_violation_mwh": max(0., supervision["safe_floor_next_mwh"] - after),
                "recovery_charge_mwh": c if np.isfinite(packet.lower[0]) and packet.lower[0] > gates["charge_threshold_mw"] else 0.,
                "gate_reason": status["gate_reason"],
                "primary_solver_attempted": status["solver_attempted"],
                "primary_solver_failed": status["solver_failed"],
                "primary_solver_error": status["solver_error"],
                "fallback_active": status["gate_reason"] != "ok",
                "primary_charge_mw": primary_c, "primary_discharge_mw": primary_d,
                "headroom_cap_mw": None, "post_supervisor_headroom_excess_mw": None,
                **supervision,
            })
            soc = after
        import pandas as pd
        trace = pd.DataFrame(rows)
        metrics = summarize_continuous(trace, physical, gates, tail)
        metrics.update({
            "assumed_parameters": asdict(assumed), "recovery_envelope": asdict(envelope),
            "supervised": True,
            "plant_within_declared_envelope": bool(
                physical.charge_power_mw >= envelope.charge_power_mw
                and physical.efficiency >= envelope.efficiency),
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
    metrics = dict(metrics, policy_id=policy, planning_pipeline="declared_conservative_envelope")
    return trace, metrics
