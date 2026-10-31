"""V5 controller IDs, with admission budgets unchanged from v4."""
from __future__ import annotations

import math

CANDIDATES = ("tail_unanchored", "tail_debt", "tail_guard", "peak_debt", "peak_guard")
SEEDS = (42, 43, 44)


def select_policy(validation):
    keys = ("controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")
    for name in ("legacy", *CANDIDATES):
        for seed in SEEDS:
            if not all(math.isfinite(validation[name][f"data_seed{seed}"][k]) for k in keys):
                raise ValueError("Finite selection metrics required.")
    decisions = {}
    for name in CANDIDATES:
        per_seed, p95, loss, delta = {}, [], [], []
        for seed in SEEDS:
            a, b = validation[name][f"data_seed{seed}"], validation["legacy"][f"data_seed{seed}"]
            if not b["physical_audit_passed"] or not b["terminal_requirement_passed"]:
                raise ValueError("Invalid legacy baseline.")
            per_seed[str(seed)] = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"]),
                "peak": bool(a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.),
                "throughput": bool(a["throughput_mwh"] <= 1.1 * b["throughput_mwh"] + 1e-6),
                "loss": bool(a["conversion_loss_mwh"] <= 1.1 * b["conversion_loss_mwh"] + 1e-6),
                "p95": bool(a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.),
            }
            p95.append(a["controlled_p95_mw"])
            loss.append(a["conversion_loss_mwh"])
            delta.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
        decisions[name] = {
            "per_seed_guards": per_seed, "mean_p95_mw": sum(p95) / 3,
            "mean_loss_mwh": sum(loss) / 3, "mean_p95_delta_mw": sum(delta) / 3,
            "eligible": bool(sum(delta) / 3 <= -1. and all(all(g.values()) for g in per_seed.values())),
        }
    eligible = [name for name in CANDIDATES if decisions[name]["eligible"]]
    selected = min(eligible, key=lambda name: (
        decisions[name]["mean_p95_mw"], decisions[name]["mean_loss_mwh"], name,
    )) if eligible else "legacy"
    return {
        "selected_policy": selected, "candidates": decisions, "fallback": not eligible,
        "selection_predictors": [f"data_seed{s}" for s in SEEDS],
        "rule": "Same v4 guards, minimum mean validation P95, loss then ID tie-break.",
    }
