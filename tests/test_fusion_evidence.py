import json
from pathlib import Path
import shutil

import pytest

from src.fusion_evidence import load_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_current_v8_catalog_matches_published_counts():
    evidence = load_evidence(ROOT)
    selected = evidence.catalog.loc[evidence.catalog.policy.eq("envelope_tail")]
    inside = selected.loc[selected.inside_envelope]
    assert len(evidence.catalog) == 96
    assert len(selected) == 32
    assert len(inside) == 30
    assert inside.recovery_passed.all()
    assert inside.all_budgets_passed.sum() == 11
    assert not selected.loc[~selected.inside_envelope, "recovery_passed"].any()


@pytest.mark.parametrize("year,mode,case,event", [
    ("2021", "fresh6", "nominal", "healthy"),
    ("2022", "delayed1", "efficiency_085", "healthy"),
    ("2022", "fresh6", "combined90", "combined_fault"),
    ("2022", "fresh6", "outside", "healthy"),
])
def test_current_v8_traces_have_complete_hours_and_checked_energy(year, mode, case, event):
    evidence = load_evidence(ROOT)
    metrics = evidence.metrics(year, mode, case, event)
    trace = evidence.trace(metrics)
    assert len(trace) == 8712
    assert trace.storage_soc_mwh.max() <= 15000 + 1e-5
    assert trace.timestamp.dt.hour.iloc[0] == 0
    assert trace.timestamp.dt.hour.iloc[-1] == 23


def test_modified_results_and_path_escapes_are_rejected(tmp_path):
    source = ROOT / "outputs/fusion_v8"
    target = tmp_path / "outputs/fusion_v8"
    target.mkdir(parents=True)
    for name in ("ARTIFACT_SHA256.json", "metrics.json", "selection.json",
                 "decision.json", "status.json"):
        shutil.copy2(source / name, target / name)
    evidence = load_evidence(tmp_path)
    with pytest.raises(ValueError, match="路径"):
        evidence.checked_bytes("../outside.csv")
    path = target / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "incomplete"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        load_evidence(tmp_path)
