"""Read frozen experiment outputs without importing the experiment runtime."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
DEST = WORKSPACE / "records/report_revision/legacy_report_work"
SEEDS = (42, 43, 44)
SPLITS = ("test", "2021", "2022")
CASES = ("nominal", "power_90pct", "power_70pct",
         "efficiency_090", "efficiency_085", "combined90")
SOURCES: dict[str, str] = {}


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def export_table(name, rows):
    path = DEST / "图表数据" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    v1 = read("outputs/fusion_v1/summary/summary.json")
    v3 = read("outputs/fusion_v3_analysis.json")
    v5 = read("outputs/fusion_v5_analysis.json")
    v6 = read("outputs/fusion_v6_analysis.json")
    v7 = read("outputs/fusion_v7_analysis.json")
    v8 = read("outputs/fusion_v8_analysis.json")
    selection = read("outputs/fusion_v8/selection.json")
    verification = read("outputs/fusion_v8_local_verification.json")
    manifest = read("outputs/fusion_v8/manifest.json")
    read("outputs/fusion_v3/samples.json")
    read("outputs/fusion_v8/samples.json")
    for relative in ("outputs/fusion_v1/data_seed42/manifest.json",
                     "src/recovery_control.py", "src/feedback_control.py",
                     "docs/fusion-dispatch-report.md", "docs/fusion-tail-control-report.md"):
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    prediction = []
    for split in SPLITS:
        base = v1["baselines"]["hgb"][split]
        prediction.append(dict(split=split, method="HGB", seed="single",
                               mae_mw=base["mae_mw"],
                               accounting_mw=base["accounting_residual_mae_mw"],
                               coverage_pct=base["interval_coverage_pct"], rows=base["rows"]))
        for family, variant, name in (
            ("data", "raw", "Data MLP"), ("physics", "raw", "Consistency MLP"),
            ("physics", "projected", "Consistency + projection")):
            for seed in SEEDS:
                m = v1["runs"][f"{family}_seed{seed}"]["variants"][variant][split]
                prediction.append(dict(split=split, method=name, seed=seed,
                                       mae_mw=m["mae_mw"],
                                       accounting_mw=m["accounting_residual_mae_mw"],
                                       coverage_pct=m["interval_coverage_pct"], rows=m["rows"]))
    export_table("fig04_prediction", prediction)

    horizon = []
    tail = []
    for split in SPLITS:
        for window in ("w1", "w6"):
            m = v3["families"][split]["data"][window]
            horizon.append(dict(split=split, window=window, **m))
        m = v5["evaluation"][split]["all_data_ablations"]
        legacy, chosen = m["legacy"]["seed_mean"], m["tail_guard"]["seed_mean"]
        tail.append(dict(split=split,
                         p95_delta_mw=chosen["controlled_p95_mw"] - legacy["controlled_p95_mw"],
                         peak_delta_mw=chosen["controlled_peak_mw"] - legacy["controlled_peak_mw"],
                         throughput_delta_pct=100 * (chosen["throughput_mwh"] / legacy["throughput_mwh"] - 1),
                         loss_delta_mwh=chosen["conversion_loss_mwh"] - legacy["conversion_loss_mwh"]))
    export_table("fig05_horizon", horizon)
    export_table("fig05_tail", tail)

    delay = []
    mismatch = []
    for year in ("2021", "2022"):
        for mode in ("fresh6", "fresh5", "fresh3", "delayed1", "delayed3"):
            m = v6["delay"][year][mode]
            delay.append(dict(year=year, mode=mode,
                              p95_delta_mw=m["guard_minus_legacy"]["controlled_p95_mw"]))
        for mode, cases in v6["mismatch"][year].items():
            for case, m in cases.items():
                mismatch.append(dict(year=year, mode=mode, case=case,
                                     terminal_error_mwh=m["guard_terminal_error_mwh"],
                                     known_parameter_error_mwh=m["known_guard_terminal_error_mwh"]))
    export_table("fig06_delay", delay)
    export_table("fig06_parameter", mismatch)

    feedback = []
    for year, modes in v7["healthy"].items():
        for mode, cases in modes.items():
            for case, methods in cases.items():
                for method in ("open", "feedback_only", "supervised"):
                    m = methods[method]
                    feedback.append(dict(year=year, mode=mode, case=case, method=method,
                                         terminal_error_mwh=m["max_abs_terminal_error_mwh"],
                                         peak_mw=m["controlled_peak_mw"], p95_mw=m["controlled_p95_mw"]))
    export_table("fig07_feedback", feedback)

    envelope = []
    for year, modes in v8["healthy"].items():
        for mode, cases in modes.items():
            for case, methods in cases.items():
                key = f"{year}/{mode}/{case}/healthy"
                guards = v8["later_protection_diagnostics"][key]
                delta = methods["envelope_tail"]["minus_v7"]
                envelope.append(dict(year=year, mode=mode, case=case, event="healthy",
                                     **delta, **{f"guard_{k}": v for k, v in guards.items()},
                                     all_budgets_passed=all(guards.values())))
        for event, methods in v8["faults"][year].items():
            key = f"{year}/fresh6/combined90/{event}"
            guards = v8["later_protection_diagnostics"][key]
            envelope.append(dict(year=year, mode="fresh6", case="combined90", event=event,
                                 **methods["envelope_tail"]["minus_v7"],
                                 **{f"guard_{k}": v for k, v in guards.items()},
                                 all_budgets_passed=all(guards.values())))
    assert len(envelope) == 30
    assert sum(r["all_budgets_passed"] for r in envelope) == 11
    export_table("fig09_all_conditions", envelope)

    evidence = dict(prediction=prediction, horizon=horizon, tail=tail, delay=delay,
                    mismatch=mismatch, feedback=feedback, envelope=envelope,
                    v7=v7, v8=v8, selection=selection, verification=verification,
                    manifest=manifest)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "证据摘要.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def write_provenance():
    provenance = {
        "date": "2026-10-28",
        "operation": "Report-only synthesis; no new model training or dispatch experiments.",
        "source_root": "03_源代码",
        "sources_sha256": SOURCES,
        "aggregation": {
            "prediction": "Arithmetic mean of seeds 42/43/44; HGB is one deterministic baseline.",
            "horizon_and_tail": "Previously computed fixed-seed means, not pooled-hour maxima.",
            "envelope": "All 30 retained-candidate in-envelope evaluation conditions, fixed seed42.",
            "fault_replay": "First scheduled combined-fault UTC day in 2022, not best-case selection.",
        },
        "rendering": "Local Matplotlib; diagram geometry is manually laid out with equivalent Mermaid source.",
        "uncertainty": "No new confidence intervals or significance tests calculated for this report.",
        "limits": "Exploratory inspected years; exact simulator SOC; supplied lower envelope; offline only.",
    }
    (DEST / "图表溯源.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
