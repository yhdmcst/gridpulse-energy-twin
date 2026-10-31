"""Causal rolling residual radii, exploratory alternative to frozen bands."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_radius(timestamps,residuals,issued,fallback,*,window_hours=720,minimum=168,quantile=.9):
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps,utc=True))
    origins = pd.DatetimeIndex(pd.to_datetime(issued,utc=True))
    residuals = np.asarray(residuals,float)
    fallback = np.asarray(fallback,float)
    if not stamps.is_monotonic_increasing or stamps.has_duplicates or len(origins) != len(stamps):
        raise ValueError("Ordered unique target times and matching issue times required.")
    if not np.isfinite(residuals).all() or not np.isfinite(fallback).all() or (fallback < 0).any():
        raise ValueError("Finite residuals and nonnegative fallback radii required.")
    if window_hours <= 0 or minimum < 1 or not 0 < quantile < 1:
        raise ValueError("Invalid rolling calibration settings.")
    # Truth labelled s covers [s,s+1h), only available at s+1h. At issue t
    # require s+1h <= t. searchsorted right includes exact available boundary.
    available = stamps+pd.Timedelta(hours=1)
    radii,counts = [],[]
    for i,origin in enumerate(origins):
        end = min(i,int(available.searchsorted(origin,side="right")))
        start = int(stamps.searchsorted(origin-pd.Timedelta(hours=window_hours),side="left"))
        history = np.abs(residuals[start:end])
        counts.append(len(history))
        radii.append(float(np.quantile(history,quantile)) if len(history) >= minimum else float(fallback[i]))
    return np.array(radii),np.array(counts)
