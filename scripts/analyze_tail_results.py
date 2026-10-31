"""Summarize fixed v4 selection and ablation tradeoffs without reselecting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fusion_v4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report, selection = read(OUT / "metrics.json"), read(OUT / "selection.json")
    validation, original = read(OUT / "validation_results.json"), read(ROOT / "outputs/fusion_v3/metrics.json")
    selected = selection["selected_policy"]
    controls = tuple(validation)
    seeds = (42, 43, 44)
    keys = (
        "baseline_peak_mw", "baseline_p95_mw", "controlled_peak_mw", "controlled_p95_mw",
        "mean_daily_peak_mw", "controlled_mean_excess_mw", "throughput_mwh",
        "conversion_loss_mwh", "recovery_charge_mwh", "actual_high_load_charging_mwh",
    )
    result = {
        "selected_policy": selected, "fallback": selection["fallback"],
        "seconds": report["seconds"], "validation": {}, "evaluation": {}, "stress": {},
        "selection_note": "Selection remains fixed; evaluation ablations are diagnostics only.",
    }
    for control in controls:
        if control == "legacy":
            continue
        p95_delta, peak_delta, loss_delta = [], [], []
        failed = {}
        for seed in seeds:
            a, b = validation[control][f"data_seed{seed}"], validation["legacy"][f"data_seed{seed}"]
            p95_delta.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
            peak_delta.append(a["controlled_peak_mw"] - b["controlled_peak_mw"])
            loss_delta.append(a["conversion_loss_mwh"] - b["conversion_loss_mwh"])
            guards = selection["candidates"][control]["per_seed_guards"][str(seed)]
            failed[str(seed)] = [key for key, passed in guards.items() if not passed]
        result["validation"][control] = {
            "mean_p95_delta_mw": float(np.mean(p95_delta)),
            "seed_p95_deltas_mw": p95_delta, "seed_peak_deltas_mw": peak_delta,
            "mean_loss_delta_mwh": float(np.mean(loss_delta)),
            "eligible": selection["candidates"][control]["eligible"], "failed_guards": failed,
        }
    for period, policies in report["methods"].items():
        data = {}
        for control, methods in policies.items():
            mean = {key: float(np.mean([methods[f"data_seed{s}"][key] for s in seeds])) for key in keys}
            baseline = policies["legacy"]
            delta = {key: float(np.mean([
                methods[f"data_seed{s}"][key] - baseline[f"data_seed{s}"][key] for s in seeds]))
                     for key in keys}
            data[control] = {"seed_mean": mean, "minus_legacy": delta}
        hgb = {control: {key: methods["hgb"][key] for key in keys}
               for control, methods in policies.items()}
        onehour = {
            key: float(np.mean([original["methods"][period][f"data_seed{s}_w1"][key] for s in seeds]))
            for key in ("controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")
        }
        result["evaluation"][period] = {
            "all_data_ablations": data, "hgb": hgb, "original_matched_onehour": onehour,
            "selected_minus_onehour": {key: data[selected]["seed_mean"][key] - onehour[key] for key in onehour},
            "selected_conditional_intervals": report["comparisons"]["selected_data_seed_mean"][period],
        }
    for period, scenarios in report["stress"].items():
        result["stress"][period] = {}
        for name, arms in scenarios.items():
            result["stress"][period][name] = {
                "replay_terminal_error_mwh": arms["replay"]["max_abs_terminal_error_mwh"],
                "replan_terminal_error_mwh": arms["replan"]["max_abs_terminal_error_mwh"],
                "replan_minus_replay": {key: arms["replan"][key] - arms["replay"][key] for key in (
                    "controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")},
            }
    path = ROOT / "outputs/fusion_v4_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "analysis": str(path), "seconds": result["seconds"],
        "selected_policy": selected, "fallback": result["fallback"], "validation": result["validation"],
        "evaluation_data_deltas": {
            p: {c: x["minus_legacy"] for c, x in v["all_data_ablations"].items()}
            for p, v in result["evaluation"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
