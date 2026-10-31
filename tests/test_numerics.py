import copy
from pathlib import Path

import numpy as np
import pytest

from src.numerics import same_loads, same_numeric_evidence


def test_serialized_load_noise_is_allowed_but_changed_or_nonfinite_loads_are_not():
    assert same_loads([31545.124999999996], [31545.125])
    assert not same_loads([31545.125], [31545.126])
    assert not same_loads([1.0], [[1.0]])
    for value in (np.nan, np.inf, -np.inf):
        assert not same_loads([value], [value])


def test_nested_audit_keeps_counts_booleans_schema_and_material_values_strict():
    audit = {"passed": True, "rows": 8760, "residuals": {"load_mw": 9.890754881780595e-12}}
    saved = copy.deepcopy(audit)
    saved["residuals"]["load_mw"] = 1.0629719326971099e-11
    assert same_numeric_evidence(audit, saved)
    for key, value in (("passed", 1), ("rows", 8760.0), ("rows", 8759)):
        changed = {**saved, key: value}
        assert not same_numeric_evidence(audit, changed)
    assert not same_numeric_evidence(audit, {**saved, "extra": None})
    for value in (1e-5, np.nan, np.inf):
        changed = copy.deepcopy(saved)
        changed["residuals"]["load_mw"] = value
        assert not same_numeric_evidence(audit, changed)


def test_packaged_certificates_and_development_audit_survive_serialization():
    root = Path(__file__).resolve().parents[1]
    if not (root / "outputs/optimality/certificates.json").is_file():
        pytest.skip("Packaged evidence not present in a source-only checkout")
    from scripts.certify_optimality import verify
    from scripts.audit_development import audit
    assert verify(root)
    assert audit()
