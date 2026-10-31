"""Timestamp-unit-independent residual windows; historical v1 code stays frozen."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_radii(frame: pd.DataFrame, initial_radius: float) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
    if (times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
            or not np.isfinite(initial_radius) or initial_radius < 0):
        raise ValueError("Ordered unique timestamps and nonnegative finite radius required.")
    errors = np.abs(frame.residual_mw.to_numpy(float))
    if not np.isfinite(errors).all():
        raise ValueError("Finite residuals required.")
    stamps = times.as_unit("ns").asi8
    result = np.full(len(frame), float(initial_radius))
    for i in range(len(frame)):
        left = np.searchsorted(stamps, stamps[i] - pd.Timedelta(hours=720).value)
        history = errors[left:i]
        if len(history) >= 168:
            result[i] = np.quantile(history, .9)
    return result
