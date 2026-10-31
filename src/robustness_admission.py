"""Fixed demo-admission rule, no controller selection or new tolerance search."""
from __future__ import annotations

import math

MAIN_MODES = ("fresh6", "delayed1", "delayed3")
ERROR_CASES = ("power_90pct", "power_70pct", "efficiency_090", "efficiency_085")


def demo_admission(delay_results, mismatch_results, *, audit_ok):
    failed = []
    guards = {}
    if set(delay_results) != {"2021", "2022"} or set(mismatch_results) != {"2021", "2022"}:
        raise ValueError("Both declared years are mandatory.")
    for year in ("2021", "2022"):
        if (not set(MAIN_MODES).issubset(delay_results[year])
                or set(mismatch_results[year]) != set(MAIN_MODES)
                or any(set(cases) != set(ERROR_CASES) for cases in mismatch_results[year].values())):
            raise ValueError("Every declared delay and mismatch condition is mandatory.")
    if not audit_ok:
        failed.append("integrity_or_replication")
    for year, modes in delay_results.items():
        guards[year] = {}
        for mode in MAIN_MODES:
            details, p95_delta = {}, []
            for seed in (42, 43, 44):
                pair = modes[mode][f"data_seed{seed}"]
                a, b = pair["tail_guard"], pair["legacy"]
                values = [m[key] for m in (a, b) for key in (
                    "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh",
                    "conversion_loss_mwh", "max_abs_terminal_error_mwh")]
                finite = all(math.isfinite(v) for v in values)
                checks = {
                    "finite": finite,
                    "physical": bool(a["physical_audit_passed"] and b["physical_audit_passed"]),
                    "terminal": bool(a["terminal_requirement_passed"] and b["terminal_requirement_passed"]),
                    "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                    "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
                    "throughput": bool(a["throughput_mwh"] <= 1.1 * b["throughput_mwh"] + 1e-6),
                    "loss": bool(a["conversion_loss_mwh"] <= 1.1 * b["conversion_loss_mwh"] + 1e-6),
                }
                details[str(seed)] = checks
                p95_delta.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
                for name, passed in checks.items():
                    if not passed:
                        failed.append(f"delay/{year}/{mode}/seed{seed}/{name}")
            mean_delta = sum(p95_delta) / 3
            if not math.isfinite(mean_delta) or mean_delta > -1.:
                failed.append(f"delay/{year}/{mode}/mean_p95")
            guards[year][mode] = {"per_seed": details, "mean_p95_delta_mw": mean_delta}
    mismatch_guards = {}
    for year, modes in mismatch_results.items():
        mismatch_guards[year] = {}
        for mode, cases in modes.items():
            mismatch_guards[year][mode] = {}
            for case, pair in cases.items():
                a = pair["tail_guard"]
                checks = {
                    "physical": bool(a["physical_audit_passed"]),
                    "terminal": bool(a["terminal_requirement_passed"]
                                     and math.isfinite(a["max_abs_terminal_error_mwh"])
                                     and a["max_abs_terminal_error_mwh"] <= 1e-5),
                }
                mismatch_guards[year][mode][case] = checks
                for name, passed in checks.items():
                    if not passed:
                        failed.append(f"mismatch/{year}/{mode}/{case}/{name}")
    return {
        "default_realtime_demo_admitted": not failed, "failed_conditions": failed,
        "delay_guards": guards, "mismatch_guards": mismatch_guards,
        "ui_action": "eligible_for_separate_review" if not failed else "defer_default_realtime_ui",
        "scope": "Fixed stress matrix only; offline results viewer is not live control.",
    }
