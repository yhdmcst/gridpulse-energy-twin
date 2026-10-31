"""Validation-only v8 retention, same peak/P95/cycling budgets on two fixed plants."""
from __future__ import annotations

import math

POLICIES = ("envelope_reference", "envelope_tail")
SCENARIOS = ("nominal", "combined90")


def select_policy(validation):
    decisions = {}
    for policy in ("baseline_v7", *POLICIES):
        for case in SCENARIOS:
            m = validation[policy][case]
            if not all(math.isfinite(m[k]) for k in (
                "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh",
                "conversion_loss_mwh", "max_abs_terminal_error_mwh",
                "max_actual_floor_violation_mwh", "max_worst_case_floor_violation_mwh")):
                raise ValueError("Finite complete validation metrics required.")
            if policy == "baseline_v7" and not (
                    m["physical_audit_passed"] and m["terminal_requirement_passed"]):
                raise ValueError("Invalid physical baseline.")
    for policy in POLICIES:
        per_case, delta, p95, loss = {}, [], [], []
        for case in SCENARIOS:
            a, b = validation[policy][case], validation["baseline_v7"][case]
            per_case[case] = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"] and a["max_abs_terminal_error_mwh"] <= 1e-5),
                "recovery": bool(max(a["max_actual_floor_violation_mwh"], a["max_worst_case_floor_violation_mwh"]) <= 1e-5),
                "gates": a["gated_optimizer_calls"] == 0,
                "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
                "throughput": bool(a["throughput_mwh"] <= b["throughput_mwh"] * 1.1 + 1e-6),
                "loss": bool(a["conversion_loss_mwh"] <= b["conversion_loss_mwh"] * 1.1 + 1e-6),
            }
            delta.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
            p95.append(a["controlled_p95_mw"])
            loss.append(a["conversion_loss_mwh"])
        decisions[policy] = {
            "scenario_guards": per_case, "mean_p95_delta_mw": sum(delta) / 2.,
            "mean_p95_mw": sum(p95) / 2., "mean_loss_mwh": sum(loss) / 2.,
            "eligible": bool(sum(delta) / 2. <= -1. and all(all(g.values()) for g in per_case.values())),
        }
    eligible = [p for p in POLICIES if decisions[p]["eligible"]]
    selected = min(eligible, key=lambda p: (
        decisions[p]["mean_p95_mw"], decisions[p]["mean_loss_mwh"], p)) if eligible else "baseline_v7"
    return {
        "selected_policy": selected, "fallback": not eligible, "candidates": decisions,
        "fixed_predictor": "data_seed42", "validation_scenarios": list(SCENARIOS),
        "scope": "Two fixed validation scenarios, not independent model seeds or sites.",
    }
