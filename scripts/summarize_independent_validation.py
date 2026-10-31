"""Post-evaluation reporting checks, separate from the frozen primary protocol."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.independent_validation import paired_block_summary


def summarize() -> dict:
    report_path = ROOT / "outputs/validation/independent_2021_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    traces = {name: pd.read_csv(ROOT/item["timeseries"],parse_dates=["timestamp"])
              for name,item in report["experiments"].items()}
    old,new = traces["dynamic_upper"],traces["dynamic_peak_response"]
    daily_old = old.groupby(old.timestamp.dt.floor("D")).controlled_net_load_mw.max()
    daily_new = new.groupby(new.timestamp.dt.floor("D")).controlled_net_load_mw.max()
    delta = daily_old-daily_new
    # API schema probing fetched Jan 1--3 before registration. It informed only
    # timestamp/field compatibility. Excluding these days is a disclosed
    # post-hoc reporting sensitivity, never a replacement primary endpoint.
    clean = {name: trace.loc[trace.timestamp.ge(pd.Timestamp("2021-01-04T00:00Z"))].reset_index(drop=True)
             for name,trace in traces.items()}
    coverage = {}
    for name,trace in clean.items():
        coverage[name] = {"days":len(trace)//24, "rows":len(trace), "controlled_peak_mw":float(trace.controlled_net_load_mw.max()),
            "controlled_p95_mw":float(trace.controlled_net_load_mw.quantile(.95)),
            "throughput_mwh":float((trace.charge_mw+trace.discharge_mw).sum())}
    return {"schema_version":"gridpulse-independent-robustness.v1",
        "primary_report_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "status":"Post-hoc disclosure and sensitivity checks; primary frozen evaluation is unchanged.",
        "api_probe_disclosure":"Before full-year registration, API requests for Jan 1--3 inspected field names, units and timestamp semantics; only a first-row sample and schema were displayed. No trigger, model or policy was fitted to these probes. To avoid claiming every hour was untouched, all three dates are additionally excluded here.",
        "daily_peak_direction":{"improved_days":int(delta.gt(1e-6).sum()),"equal_days":int(delta.abs().le(1e-6).sum()),
                                "worsened_days":int(delta.lt(-1e-6).sum())},
        "probe_excluded_period":{"start":"2021-01-04T00:00:00+00:00","end_exclusive":"2022-01-01T00:00:00+00:00"},
        "probe_excluded_metrics":coverage,
        "probe_excluded_uncertainty":paired_block_summary(clean["dynamic_upper"],clean["dynamic_peak_response"]),
        "probe_excluded_peak_reduction_mw":coverage["dynamic_upper"]["controlled_peak_mw"]-coverage["dynamic_peak_response"]["controlled_peak_mw"]}


if __name__ == "__main__":
    result = summarize()
    (ROOT/"outputs/validation/independent_2021_robustness.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
