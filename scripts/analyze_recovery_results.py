"""Readable v5 aggregates without changing the validation-selected policy."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report, receipt = read(OUT / "metrics.json"), read(OUT / "selection.json")
    validation, v3 = read(OUT / "validation_results.json"), read(ROOT / "outputs/fusion_v3/metrics.json")
    selected = receipt["selected_policy"]
    seeds = (42, 43, 44)
    keys = (
        "baseline_peak_mw", "baseline_p95_mw", "controlled_peak_mw", "controlled_p95_mw",
        "mean_daily_peak_mw", "controlled_mean_excess_mw", "throughput_mwh",
        "conversion_loss_mwh", "recovery_charge_mwh", "actual_high_load_charging_mwh",
    )
    result = {
        "selected_policy": selected, "fallback": receipt["fallback"], "seconds": report["seconds"],
        "validation": {}, "evaluation": {}, "stress": {},
        "boundary": "Fixed-seed aggregates; no evaluation-year reselection or untouched-holdout claim.",
    }
    for control, decision in receipt["candidates"].items():
        per_seed = {}
        for seed in seeds:
            a, b = validation[control][f"data_seed{seed}"], validation["legacy"][f"data_seed{seed}"]
            per_seed[str(seed)] = {
                "p95_delta_mw": a["controlled_p95_mw"] - b["controlled_p95_mw"],
                "peak_delta_mw": a["controlled_peak_mw"] - b["controlled_peak_mw"],
                "throughput_delta_pct": 100. * (a["throughput_mwh"] / b["throughput_mwh"] - 1.),
                "loss_delta_mwh": a["conversion_loss_mwh"] - b["conversion_loss_mwh"],
                "debt_mwh": a["max_reference_debt_mwh"],
                "failed_guards": [name for name, passed in decision["per_seed_guards"][str(seed)].items()
                                  if not passed],
            }
        result["validation"][control] = {**decision, "per_seed_diagnostics": per_seed}
    for period, policies in report["methods"].items():
        data = {}
        for control, methods in policies.items():
            mean = {key: float(np.mean([methods[f"data_seed{s}"][key] for s in seeds])) for key in keys}
            delta = {key: float(np.mean([
                methods[f"data_seed{s}"][key] - policies["legacy"][f"data_seed{s}"][key] for s in seeds]))
                     for key in keys}
            data[control] = {
                "seed_mean": mean, "minus_legacy": delta,
                "worst_seed_debt_mwh": float(max(methods[f"data_seed{s}"]["max_reference_debt_mwh"] for s in seeds)),
            }
        old_onehour = {
            key: float(np.mean([v3["methods"][period][f"data_seed{s}_w1"][key] for s in seeds]))
            for key in ("controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")
        }
        result["evaluation"][period] = {
            "all_data_ablations": data,
            "hgb": {control: {key: methods["hgb"][key] for key in keys}
                    for control, methods in policies.items()},
            "selected_seed_metrics": {str(s): policies[selected][f"data_seed{s}"] for s in seeds},
            "selected_minus_onehour": {key: data[selected]["seed_mean"][key] - old_onehour[key] for key in old_onehour},
            "selected_intervals": report["comparisons"]["selected_data_seed_mean"][period],
        }
    for period, scenarios in report["stress"].items():
        result["stress"][period] = {}
        for scenario, arms in scenarios.items():
            result["stress"][period][scenario] = {
                "replay_terminal_error_mwh": arms["replay"]["max_abs_terminal_error_mwh"],
                "replan_terminal_error_mwh": arms["replan"]["max_abs_terminal_error_mwh"],
                "replan_minus_replay": {key: arms["replan"][key] - arms["replay"][key] for key in (
                    "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")},
            }
    path = ROOT / "outputs/fusion_v5_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "analysis": str(path), "selected_policy": selected, "fallback": receipt["fallback"],
        "seconds": report["seconds"],
        "validation_summary": {name: {
            "eligible": item["eligible"], "mean_p95_delta_mw": item["mean_p95_delta_mw"],
            "per_seed": item["per_seed_diagnostics"]} for name, item in result["validation"].items()},
        "selected_evaluation": {p: {
            "seed_mean": item["all_data_ablations"][selected]["seed_mean"],
            "minus_legacy": item["all_data_ablations"][selected]["minus_legacy"],
            "minus_onehour": item["selected_minus_onehour"],
        } for p, item in result["evaluation"].items()},
        "stress": result["stress"],
    }, indent=2))


if __name__ == "__main__":
    main()
