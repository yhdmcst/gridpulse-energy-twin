"""Descriptive v7 closed/open/fault costs; no post-evaluation policy selection."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v7"
KEYS = (
    "controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
    "throughput_mwh", "conversion_loss_mwh", "max_abs_terminal_error_mwh",
    "terminal_failure_days",
)
COUNTERS = (
    "fallback_hours", "primary_solver_attempts", "primary_solver_failure_hours",
    "reference_solver_failure_hours", "expiry_rejections", "outage_fallback_hours",
    "supervisor_interventions", "recovery_unreachable_hours", "actual_clipping_hours",
    "max_actual_reference_debt_mwh", "max_actual_floor_violation_mwh",
    "max_worst_case_floor_violation_mwh", "gated_optimizer_calls",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = read(OUT / "metrics.json")
    result = {
        "seconds": report["seconds"], "healthy": {}, "fault": {}, "outside": {},
        "capability_checks": report["capability_checks"],
        "scope": "Fixed seed42 and inspected years, exact simulator SOC, declared recovery envelope.",
    }
    for year, modes in report["closed"].items():
        result["healthy"][year], result["fault"][year] = {}, {}
        for mode, cases in modes.items():
            result["healthy"][year][mode] = {}
            for case, events in cases.items():
                if case == "outside":
                    m = events["healthy"]["supervised"]
                    result["outside"][year] = {k: m[k] for k in (*KEYS, *COUNTERS)}
                    continue
                variants = events["healthy"]
                opened = report["open"][year][mode][case]
                rows = {"open": {k: opened[k] for k in KEYS}}
                for variant, metrics in variants.items():
                    trace = pd.read_csv(OUT / metrics["trajectory"], usecols=[
                        "post_supervisor_headroom_excess_mw", "primary_charge_mw",
                        "requested_charge_mw", "primary_discharge_mw", "requested_discharge_mw",
                    ])
                    rows[variant] = {
                        **{k: metrics[k] for k in (*KEYS, *COUNTERS)},
                        "minus_open": {k: metrics[k] - opened[k] for k in KEYS},
                        "physical_audit_passed": metrics["physical_audit_passed"],
                        "headroom_override_hours": int(
                            trace.post_supervisor_headroom_excess_mw.gt(1e-5).sum()),
                        "maximum_headroom_override_mw": float(
                            trace.post_supervisor_headroom_excess_mw.max())
                        if trace.post_supervisor_headroom_excess_mw.notna().any() else 0.,
                        "supervisor_additional_requested_charge_mwh": float(
                            (trace.requested_charge_mw - trace.primary_charge_mw).sum()),
                        "supervisor_suppressed_requested_discharge_mwh": float(
                            (trace.primary_discharge_mw - trace.requested_discharge_mw).sum()),
                    }
                rows["supervised_minus_feedback"] = {
                    k: variants["supervised"][k] - variants["feedback_only"][k] for k in KEYS
                }
                result["healthy"][year][mode][case] = rows
                if mode == "fresh6" and case == "combined90":
                    for event, event_variants in events.items():
                        if event == "healthy":
                            continue
                        result["fault"][year][event] = {}
                        for variant, metrics in event_variants.items():
                            nominal = variants[variant]
                            result["fault"][year][event][variant] = {
                                **{k: metrics[k] for k in (*KEYS, *COUNTERS)},
                                "minus_healthy": {k: metrics[k] - nominal[k] for k in KEYS},
                            }
    summaries = []
    for year, modes in report["closed"].items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, variants in events.items():
                    for variant, metrics in variants.items():
                        summaries.append({
                            "year": year, "mode": mode, "case": case, "event": event,
                            "variant": variant, **{k: metrics[k] for k in (
                                "plant_within_declared_envelope", "terminal_requirement_passed",
                                "terminal_failure_days", "fallback_hours", "supervisor_interventions",
                                "max_abs_terminal_error_mwh", "gated_optimizer_calls")},
                        })
    result["counts"] = {
        "closed": len(summaries),
        "supervised_inside": sum(x["variant"] == "supervised" and x["plant_within_declared_envelope"] for x in summaries),
        "supervised_inside_terminal_passed": sum(
            x["variant"] == "supervised" and x["plant_within_declared_envelope"] and x["terminal_requirement_passed"]
            for x in summaries),
        "feedback_terminal_failed": sum(
            x["variant"] == "feedback_only" and not x["terminal_requirement_passed"] for x in summaries),
        "all_fallback_hours": sum(x["fallback_hours"] for x in summaries),
        "gated_optimizer_calls": sum(x["gated_optimizer_calls"] for x in summaries),
    }
    path = ROOT / "outputs/fusion_v7_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    bundle = ROOT / "outputs/fusion_v7_analysis_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    evidence = [
        OUT / "manifest.json", OUT / "ARTIFACT_SHA256.json", OUT / "metrics.json",
        OUT / "Figure_1.png", OUT / "Figure_2.png",
        ROOT / "outputs/fusion_v7_local_verification.json",
        ROOT / "outputs/fusion_v7_parser_compatibility.json",
        ROOT / "outputs/fusion_v7_regression_tests.txt",
        ROOT / "scripts/verify_feedback_results_local.py", Path(__file__), path,
    ]
    provenance = {
        "purpose": "Derived v7 evidence only; no frozen experimental artifact changes.",
        "fixed_seed": 42, "source_years": [2021, 2022], "hours_per_trace": 8712,
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence
        },
        "statistical_unit": "Paired UTC days within fixed-model inspected-year episodes, not independent sites.",
    }
    (bundle / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({
        "analysis": str(path), "seconds": result["seconds"], "counts": result["counts"],
        "healthy_fresh6_terminal_errors_mwh": {
            year: {case: {variant: values[variant]["max_abs_terminal_error_mwh"]
                          for variant in ("open", "feedback_only", "supervised")}
                   for case, values in modes["fresh6"].items()}
            for year, modes in result["healthy"].items()},
        "nominal_supervision_tradeoffs": {
            year: {
                "supervised_minus_feedback": modes["fresh6"]["nominal"]["supervised_minus_feedback"],
                "headroom_override_hours": modes["fresh6"]["nominal"]["supervised"]["headroom_override_hours"],
            } for year, modes in result["healthy"].items()},
        "capability_summary": {k: v for k, v in result["capability_checks"].items() if k != "conditions"},
    }, indent=2))


if __name__ == "__main__":
    main()
