"""Summarize inspected-year robustness and fixed UI admission, without tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v6"
METHODS = ("data_seed42", "data_seed43", "data_seed44")
MAIN = ("fresh6", "delayed1", "delayed3")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = read(OUT / "metrics.json")
    result = {
        "seconds": report["seconds"], "delay": {}, "freshness": {}, "horizon": {},
        "mismatch": {}, "demo_decision": report["demo_decision"],
        "boundary": "Fixed models and inspected years; mismatch is fixed-request replay, not SOC-feedback adaptation.",
    }
    keys = ("baseline_peak_mw", "controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
            "throughput_mwh", "conversion_loss_mwh", "forecast_h1_mae_mw", "interval_coverage_pct",
            "max_abs_terminal_error_mwh")
    for year, modes in report["delay"].items():
        result["delay"][year] = {}
        result["freshness"][year] = {}
        result["horizon"][year] = {}
        for mode, methods in modes.items():
            metrics = {
                policy: {key: float(np.mean([methods[m][policy][key] for m in METHODS])) for key in keys}
                for policy in ("legacy", "tail_guard")}
            deltas = {key: metrics["tail_guard"][key] - metrics["legacy"][key] for key in keys}
            relative_fresh = {key: float(np.mean([
                methods[m]["tail_guard"][key] - modes["fresh6"][m]["tail_guard"][key] for m in METHODS]))
                for key in keys}
            metrics.update({
                "guard_minus_legacy": deltas,
                "guard_minus_fresh6_guard": relative_fresh,
                "seed_p95_deltas_mw": [methods[m]["tail_guard"]["controlled_p95_mw"]
                                      - methods[m]["legacy"]["controlled_p95_mw"] for m in METHODS],
                "seed_peak_deltas_mw": [methods[m]["tail_guard"]["controlled_peak_mw"]
                                       - methods[m]["legacy"]["controlled_peak_mw"] for m in METHODS],
                "throughput_change_pct": 100. * (metrics["tail_guard"]["throughput_mwh"]
                                                / metrics["legacy"]["throughput_mwh"] - 1.),
            })
            result["delay"][year][mode] = metrics
        for section in ("freshness", "horizon"):
            for label, contrasts in report["comparisons"][section][year].items():
                result[section][year][label] = {
                    key: float(np.mean([contrasts[f"{method}/tail_guard"]["differences"][key]
                                        for method in METHODS]))
                    for key in ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
                                "throughput_mwh", "conversion_loss_mwh")}
        result["mismatch"][year] = {}
        for mode, cases in report["mismatch"][year].items():
            result["mismatch"][year][mode] = {}
            for case, policies in cases.items():
                a, legacy = policies["tail_guard"], policies["legacy"]
                known = report["known_parameters"][year][mode][case]["tail_guard"]
                result["mismatch"][year][mode][case] = {
                    "guard_terminal_error_mwh": a["max_abs_terminal_error_mwh"],
                    "guard_terminal_failure_days": a["terminal_failure_days"],
                    "legacy_terminal_error_mwh": legacy["max_abs_terminal_error_mwh"],
                    "known_guard_terminal_error_mwh": known["max_abs_terminal_error_mwh"],
                    "unexecuted_charge_mwh": a["unexecuted_charge_mwh"],
                    "unexecuted_discharge_mwh": a["unexecuted_discharge_mwh"],
                    "guard_minus_legacy": {key: a[key] - legacy[key] for key in (
                        "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh",
                        "conversion_loss_mwh", "max_abs_terminal_error_mwh")},
                    "known_guard_minus_mismatch_guard": {key: known[key] - a[key] for key in (
                        "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh",
                        "conversion_loss_mwh", "max_abs_terminal_error_mwh")},
                }
    path = ROOT / "outputs/fusion_v6_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "analysis": str(path), "seconds": result["seconds"],
        "default_realtime_demo_admitted": result["demo_decision"]["default_realtime_demo_admitted"],
        "failed_conditions": result["demo_decision"]["failed_conditions"],
        "delay_summary": {year: {mode: {
            "guard_minus_legacy_p95_mw": x["guard_minus_legacy"]["controlled_p95_mw"],
            "seed_p95_deltas_mw": x["seed_p95_deltas_mw"],
            "guard_minus_legacy_peak_mw": x["guard_minus_legacy"]["controlled_peak_mw"],
            "seed_peak_deltas_mw": x["seed_peak_deltas_mw"],
            "guard_minus_fresh6_peak_mw": x["guard_minus_fresh6_guard"]["controlled_peak_mw"],
            "guard_minus_fresh6_p95_mw": x["guard_minus_fresh6_guard"]["controlled_p95_mw"],
            "throughput_change_pct": x["throughput_change_pct"],
        } for mode, x in modes.items()} for year, modes in result["delay"].items()},
        "freshness": result["freshness"], "horizon": result["horizon"],
        "mismatch_terminal_range_mwh": {
            year: {mode: {
                "min": min(x["guard_terminal_error_mwh"] for x in cases.values()),
                "max": max(x["guard_terminal_error_mwh"] for x in cases.values()),
                "max_known": max(x["known_guard_terminal_error_mwh"] for x in cases.values()),
            } for mode, cases in modes.items()} for year, modes in result["mismatch"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
