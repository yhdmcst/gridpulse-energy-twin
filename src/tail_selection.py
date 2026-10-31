"""Validation-only controller selection, separate from solver and year evaluation."""
from __future__ import annotations

import math

import pandas as pd

SEEDS = (42, 43, 44)
CANDIDATES = (
    "peak_dynamic", "peak_physical", "tail_dynamic_0", "tail_dynamic_250",
    "tail_physical_0", "tail_physical_250",
)


def validate_validation_range(frame) -> None:
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
    if (frame.empty or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
            or not times.equals(times.floor("h"))
            or times.min() < pd.Timestamp("2019-01-12T20:00:00Z")
            or times.max() + pd.Timedelta(hours=5) >= pd.Timestamp("2019-08-09T21:00:00Z")):
        raise ValueError("All six calibration/selection targets must remain inside validation.")


def select_policy(validation: dict) -> dict:
    required = ("controlled_peak_mw", "controlled_p95_mw", "throughput_mwh", "conversion_loss_mwh")
    for policy in ("legacy", *CANDIDATES):
        for seed in SEEDS:
            metrics = validation[policy][f"data_seed{seed}"]
            if not all(math.isfinite(metrics[k]) for k in required):
                raise ValueError("Finite validation selection metrics required.")
    for seed in SEEDS:
        metrics = validation["legacy"][f"data_seed{seed}"]
        if not (metrics["physical_audit_passed"] and metrics["terminal_requirement_passed"]):
            raise ValueError("Legacy validation baseline must pass physical and terminal audits.")
    decisions = {}
    for policy in CANDIDATES:
        per_seed = {}
        p95, loss, delta = [], [], []
        for seed in SEEDS:
            a = validation[policy][f"data_seed{seed}"]
            b = validation["legacy"][f"data_seed{seed}"]
            guards = {
                "physical": bool(a["physical_audit_passed"]),
                "terminal": bool(a["terminal_requirement_passed"]),
                "peak": a["controlled_peak_mw"] <= b["controlled_peak_mw"] + 25.,
                "throughput": a["throughput_mwh"] <= 1.1 * b["throughput_mwh"] + 1e-6,
                "loss": a["conversion_loss_mwh"] <= 1.1 * b["conversion_loss_mwh"] + 1e-6,
                "p95": a["controlled_p95_mw"] <= b["controlled_p95_mw"] + 1.,
            }
            per_seed[str(seed)] = {k: bool(v) for k, v in guards.items()}
            p95.append(a["controlled_p95_mw"])
            loss.append(a["conversion_loss_mwh"])
            delta.append(a["controlled_p95_mw"] - b["controlled_p95_mw"])
        mean_delta = sum(delta) / len(delta)
        decisions[policy] = {
            "per_seed_guards": per_seed,
            "mean_p95_mw": sum(p95) / len(p95),
            "mean_loss_mwh": sum(loss) / len(loss),
            "mean_p95_delta_mw": mean_delta,
            "eligible": bool(mean_delta <= -1. and all(all(g.values()) for g in per_seed.values())),
        }
    eligible = [name for name in CANDIDATES if decisions[name]["eligible"]]
    selected = min(eligible, key=lambda name: (
        decisions[name]["mean_p95_mw"], decisions[name]["mean_loss_mwh"], name,
    )) if eligible else "legacy"
    return {
        "selected_policy": selected, "candidates": decisions,
        "selection_predictors": [f"data_seed{s}" for s in SEEDS],
        "fallback": not eligible,
        "rule": "Minimum mean validation P95 after per-seed guards; ties loss then policy ID.",
    }
