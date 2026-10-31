"""Evaluate v4 debt and coverage diagnostics without changing its frozen results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs/fusion_v4"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    calibration = read(V4 / "calibration.json")
    validation = read(V4 / "validation_results.json")
    tail = calibration["tail_threshold_mw"]
    result = {"validation_debt_diagnostics": {}, "scope": (
        "Descriptive diagnostic, not proof of SOC debt as the unique cause. "
        "Use validation rows only to motivate a bounded next experiment.")}
    for seed in (42, 43, 44):
        method = f"data_seed{seed}"
        legacy = pd.read_csv(V4 / validation["legacy"][method]["trajectory"], parse_dates=["timestamp"])
        candidate = pd.read_csv(V4 / validation["tail_dynamic_0"][method]["trajectory"],
                                parse_dates=["timestamp"])
        assert legacy.timestamp.equals(candidate.timestamp)
        debt = legacy.storage_soc_mwh - candidate.storage_soc_mwh
        radius = legacy.timestamp.dt.hour.map({
            int(k): v for k, v in calibration["methods"][method]["radii"]["1"].items()})
        high = legacy.net_load_mw.ge(legacy.net_load_mw.quantile(.99))
        reference_upper = legacy.prediction_h1_mw + radius + legacy.charge_mw - legacy.discharge_mw
        peak_index = candidate.controlled_net_load_mw.idxmax()
        result["validation_debt_diagnostics"][str(seed)] = {
            "hours": len(legacy),
            "candidate_below_reference_soc_pct": float((debt > 1e-5).mean() * 100),
            "maximum_soc_debt_mwh": float(max(0., debt.max())),
            "top_actual_1pct_hours": int(high.sum()),
            "top_actual_1pct_reference_upper_above_tail_pct": float(
                reference_upper.loc[high].ge(tail).mean() * 100),
            "candidate_peak_timestamp": candidate.timestamp.iloc[peak_index].isoformat(),
            "candidate_soc_before_at_peak_mwh": float(candidate.soc_before_mwh.iloc[peak_index]),
            "reference_soc_before_at_same_hour_mwh": float(legacy.soc_before_mwh.iloc[peak_index]),
            "candidate_minus_reference_load_at_same_hour_mw": float(
                candidate.controlled_net_load_mw.iloc[peak_index]
                - legacy.controlled_net_load_mw.iloc[peak_index]),
        }
    path = ROOT / "outputs/fusion_v4_recovery_review.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
