"""Fixed-selection v8 planning-time tradeoffs, preserving failed/negative results."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v8"
KEYS = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw", "throughput_mwh",
        "conversion_loss_mwh", "max_abs_terminal_error_mwh", "terminal_failure_days",
        "supervisor_interventions", "primary_solver_failure_hours", "fallback_hours",
        "gated_optimizer_calls", "max_actual_floor_violation_mwh")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report, selection = read(OUT / "metrics.json"), read(OUT / "selection.json")
    result = {
        "selected_policy": selection["selected_policy"], "fallback": selection["fallback"],
        "seconds": report["seconds"], "validation": {}, "healthy": {}, "faults": {}, "outside": {},
        "later_protection_diagnostics": {},
        "decision": report["decision"],
        "scope": "One fixed seed, nominal/combined90 validation scenarios and inspected evaluation years.",
    }
    for p, cases in report["validation"].items():
        result["validation"][p] = {
            c: {"metrics": {k: m[k] for k in KEYS},
                "minus_v7": {k: m[k] - report["validation"]["baseline_v7"][c][k] for k in KEYS}}
            for c, m in cases.items()}
    items = []
    for year, modes in report["methods"].items():
        result["healthy"][year], result["faults"][year], result["outside"][year] = {}, {}, {}
        for mode, cases in modes.items():
            result["healthy"][year][mode] = {}
            for case, events in cases.items():
                for event, policies in events.items():
                    values = {}
                    for policy, m in policies.items():
                        values[policy] = {
                            "metrics": {k: m[k] for k in KEYS},
                            "minus_v7": {k: m[k] - policies["baseline_v7"][k] for k in KEYS},
                        }
                        if policy != "baseline_v7":
                            items.append(m)
                    if case == "outside":
                        result["outside"][year] = values
                    elif event != "healthy":
                        result["faults"][year][event] = values
                    else:
                        result["healthy"][year][mode][case] = values
    inside = [m for m in items if m["plant_within_declared_envelope"]]
    retained = [m for m in items if m.get("policy_id") == selection["selected_policy"]
                and m["plant_within_declared_envelope"]]
    result["counts"] = {
        "new_evaluation_traces": len(items), "inside_runs": len(inside),
        "inside_terminal_passed": sum(m["terminal_requirement_passed"] for m in inside),
        "inside_supervisor_interventions": sum(m["supervisor_interventions"] for m in inside),
        "inside_primary_solver_failure_hours": sum(m["primary_solver_failure_hours"] for m in inside),
        "retained_inside_runs": len(retained),
        "retained_inside_terminal_passed": sum(m["terminal_requirement_passed"] for m in retained),
        "retained_inside_supervisor_interventions": sum(m["supervisor_interventions"] for m in retained),
        "retained_inside_primary_solver_failure_hours": sum(m["primary_solver_failure_hours"] for m in retained),
        "gated_optimizer_calls": sum(m["gated_optimizer_calls"] for m in items),
    }
    for year, modes in report["methods"].items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, policies in events.items():
                    if case == "outside" or selection["selected_policy"] == "baseline_v7":
                        continue
                    a, b = policies[selection["selected_policy"]], policies["baseline_v7"]
                    checks = {
                        "terminal": bool(a["terminal_requirement_passed"]),
                        "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                        "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
                        "throughput": bool(a["throughput_mwh"] <= b["throughput_mwh"] * 1.1 + 1e-6),
                        "loss": bool(a["conversion_loss_mwh"] <= b["conversion_loss_mwh"] * 1.1 + 1e-6),
                    }
                    result["later_protection_diagnostics"][f"{year}/{mode}/{case}/{event}"] = checks
    result["counts"]["retained_later_conditions_passing_all_descriptive_budgets"] = sum(
        all(c.values()) for c in result["later_protection_diagnostics"].values())
    path = ROOT / "outputs/fusion_v8_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    bundle = ROOT / "outputs/fusion_v8_analysis_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    evidence = [
        OUT / "manifest.json", OUT / "ARTIFACT_SHA256.json", OUT / "metrics.json",
        OUT / "selection.json", OUT / "Figure_1.png", OUT / "Figure_2.png",
        ROOT / "outputs/fusion_v8_local_verification.json", ROOT / "outputs/fusion_v8_regression_tests.txt",
        Path(__file__), path,
    ]
    (bundle / "provenance.json").write_text(json.dumps({
        "purpose": "Derived fixed-selection v8 analysis; no frozen artifacts modified.",
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        "statistical_unit": "Paired complete UTC days, fixed seed42, inspected years.",
    }, indent=2), encoding="utf-8")
    selected = selection["selected_policy"]
    print(json.dumps({
        "analysis": str(path), "selected_policy": selected, "seconds": result["seconds"], "counts": result["counts"],
        "retained_fresh6_nominal_combined_delta": {
            y: {c: v[selected]["minus_v7"] if selected in v else v["baseline_v7"]["minus_v7"]
                for c, v in modes["fresh6"].items() if c in ("nominal", "combined90")}
            for y, modes in result["healthy"].items()},
        "outside_retained_terminal_error_mwh": {
            y: values.get(selected, values["baseline_v7"])["metrics"]["max_abs_terminal_error_mwh"]
            for y, values in result["outside"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
