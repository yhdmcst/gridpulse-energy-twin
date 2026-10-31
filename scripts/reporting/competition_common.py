"""Versioned report destination and additional fixed-seed evidence extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import report_evidence as source

ROOT, WORKSPACE = source.ROOT, source.WORKSPACE
DEST = ROOT / "report_assets"
SKILL = Path.home() / ".codex/skills/nature-figure"
FIGURES, QA = DEST / "图表", DEST / "QA"


def collect():
    # Redirect only report products; source experiment paths remain unchanged.
    source.DEST = DEST
    evidence = source.collect()
    v1 = source.read("outputs/fusion_v1/summary/summary.json")
    v3 = source.read("outputs/fusion_v3/metrics.json")
    v5 = source.read("outputs/fusion_v5/metrics.json")
    evidence["v1"] = v1
    paired = []
    for split in source.SPLITS:
        for seed in source.SEEDS:
            a, b = (v3["methods"][split][f"data_seed{seed}_{w}"] for w in ("w1", "w6"))
            legacy = v5["methods"][split]["legacy"][f"data_seed{seed}"]
            tail = v5["methods"][split]["tail_guard"][f"data_seed{seed}"]
            paired.append(dict(
                split=split, seed=seed,
                horizon_peak_delta_mw=b["controlled_peak_mw"] - a["controlled_peak_mw"],
                horizon_p95_delta_mw=b["controlled_p95_mw"] - a["controlled_p95_mw"],
                tail_p95_delta_mw=tail["controlled_p95_mw"] - legacy["controlled_p95_mw"],
            ))
    source.export_table("fig05_paired_seeds", paired)
    evidence["dispatch_pairs"] = paired
    readme_sources = (
        "src/recovery_control.py", "src/envelope_planning.py", "src/physics_forecast.py",
        "src/fusion_evidence.py", "app.py",
        "experiments/fusion_v1/protocol.md", "experiments/fusion_v8/protocol.md",
    )
    for relative in readme_sources:
        source.SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    evidence["plotting_skill"] = "nature-figure 2.8.0; Python; competition-specific dimensions"
    (DEST / "证据摘要.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def write_provenance():
    source.write_provenance()
    path = DEST / "图表溯源.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["rendering"] = "Local Python/Matplotlib; editable SVG/PDF and PNG. No external image service."
    record["skill"] = {"name": "nature-figure", "version": "2.8.0", "backend": "python",
                       "target": "Competition report, not a Nature journal submission"}
    record["generator_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("competition_common.py", "competition_figures.py", "competition_report.py",
                     "competition_audit.py", "competition_template.md", "competition_metadata.json",
                     "build_report.py", "report_evidence.py")
    }
    record["official_template_sha256"] = hashlib.sha256(
        (WORKSPACE / "附件2AIC·AI能源技术报告参考大纲-科技创新组.pdf").read_bytes()).hexdigest()
    record["uncertainty"] = "Raw paired fixed seeds where available; no new CIs, p-values or independent-trial claims."
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def read_evidence():
    return json.loads((DEST / "证据摘要.json").read_text(encoding="utf-8"))
