"""Portable CSV parsing entry for the unchanged, hash-frozen v7 verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts import verify_feedback_results as frozen

READ_CSV = pd.read_csv
LOG_DTYPES = {"solver_error": "string", "primary_solver_error": "string"}


def typed_csv(*args, **kwargs):
    options = dict(kwargs)
    dtype = options.get("dtype")
    if dtype is not None and not isinstance(dtype, dict):
        raise ValueError("This verifier expects explicit per-column dtype mappings.")
    options["dtype"] = {**(dtype or {}), **LOG_DTYPES}
    return READ_CSV(*args, **options)


def main():
    # Parser configuration only: every original hash, arithmetic and constraint assertion stays intact.
    with patch.object(pd, "read_csv", typed_csv):
        frozen.main()
    receipt = {
        "passed": True, "pandas": pd.__version__,
        "frozen_verifier_sha256": hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
        "parser_entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "explicit_log_column_dtypes": LOG_DTYPES,
        "scope": "Avoid float inference for all-empty text logs; no numerical or admission assertions changed.",
    }
    (ROOT / "outputs/fusion_v7_parser_compatibility.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
