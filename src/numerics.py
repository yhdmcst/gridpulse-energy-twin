"""Absolute tolerances for re-reading serialized numerical evidence.

Hashes, timestamps, integer counts and rational certificates stay exact.
The 1e-9 tolerance is in the stored MW/MWh units, with no relative scaling.
"""
from __future__ import annotations

import math

import numpy as np

SERIALIZATION_ATOL = 1e-9


def same_loads(left, right) -> bool:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.allclose(a, b, rtol=0, atol=SERIALIZATION_ATOL))


def same_numeric_evidence(actual, saved) -> bool:
    """Compare structured audit output without tolerating schema/count changes."""
    if isinstance(actual, dict):
        return (isinstance(saved, dict) and actual.keys() == saved.keys()
                and all(same_numeric_evidence(v, saved[k]) for k, v in actual.items()))
    if isinstance(actual, list):
        return (isinstance(saved, list) and len(actual) == len(saved)
                and all(same_numeric_evidence(a, b) for a, b in zip(actual, saved)))
    if isinstance(actual, bool) or isinstance(saved, bool):
        return type(actual) is type(saved) and actual == saved
    if isinstance(actual, int):
        return type(actual) is type(saved) and actual == saved
    if isinstance(actual, float):
        return (isinstance(saved, (int, float)) and math.isfinite(actual)
                and math.isfinite(saved)
                and math.isclose(actual, saved, rel_tol=0, abs_tol=SERIALIZATION_ATOL))
    return type(actual) is type(saved) and actual == saved
