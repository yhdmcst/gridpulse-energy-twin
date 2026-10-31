import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compare_forecast_dispatch import simple_panel
from src.optimality import StorageModel
from src.trajectory_audit import audited_summary


def test_naive_forecasts_only_use_information_available_at_issuance():
    raw = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=72, freq="h", tz="UTC"),
                        "net_load_mw": np.arange(72, dtype=float)})
    panel = pd.DataFrame({"timestamp": [raw.timestamp.iloc[48]]})
    changed = raw.copy()
    changed.loc[48:, "net_load_mw"] += 1e6
    for method in ("persistence", "seasonal_24h"):
        before, after = simple_panel(panel, raw, method), simple_panel(panel, changed, method)
        pd.testing.assert_frame_equal(before, after)
        for h in range(1, 7):
            expected = 47. if method == "persistence" else 23.+h
            assert before[f"prediction_h{h}_mw"].iloc[0] == expected


def test_delivered_forecast_comparison_has_identical_samples_and_valid_metrics():
    root = Path(__file__).resolve().parents[1]
    path = root/"outputs/experiments/forecast_dispatch/metrics.json"
    if not path.is_file():
        pytest.skip("Optional exploratory experiment has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["methods"]) == {"ai", "persistence", "seasonal_24h"}
    reference = None
    for item in report["methods"].values():
        trace = pd.read_csv(root/item["metrics"]["timeseries"], parse_dates=["timestamp"])
        if reference is None:
            reference = trace
        pd.testing.assert_series_equal(trace.timestamp, reference.timestamp)
        np.testing.assert_allclose(trace.net_load_mw, reference.net_load_mw, rtol=0, atol=1e-9)
        actual = audited_summary(trace, StorageModel())
        assert actual["physical_audit_passed"]
        for key in ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw", "throughput_mwh", "conversion_loss_mwh"):
            assert actual[key] == pytest.approx(item["metrics"][key], rel=0, abs=1e-5)
        assert len(trace) == report["sample"]["test"]["rows"]
