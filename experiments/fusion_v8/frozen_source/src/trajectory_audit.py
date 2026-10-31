"""Fail closed on invalid physical traces before recomputing their summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.efficient_storage import summarize


def audited_summary(trace, model, *, daily_terminal=True):
    model.validate()
    required = ("net_load_mw", "controlled_net_load_mw", "charge_mw", "discharge_mw",
                "soc_before_mwh", "storage_soc_mwh")
    if trace.empty or "timestamp" not in trace or any(c not in trace for c in required):
        raise ValueError("Nonempty trajectory and required physical columns are mandatory.")
    if not np.isfinite(trace[list(required)].to_numpy(float)).all():
        raise ValueError("Every physical trajectory value must be finite.")
    if "reserve_target_mwh" in trace and not np.isfinite(trace.reserve_target_mwh.to_numpy(float)).all():
        raise ValueError("Reserve targets must be finite.")
    stamps = pd.DatetimeIndex(pd.to_datetime(trace.timestamp, utc=True))
    if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("Unique ordered trajectory timestamps are mandatory.")
    if not stamps.equals(stamps.floor("h")):
        raise ValueError("Trajectory timestamps must align to UTC hours.")
    if daily_terminal:
        for _, group in trace.assign(_day=stamps.floor("D")).groupby("_day"):
            times = pd.DatetimeIndex(pd.to_datetime(group.timestamp, utc=True))
            if not times.equals(pd.date_range(times[0].floor("D"), periods=24, freq="h")):
                raise ValueError("Daily trajectories must contain every UTC hour exactly once.")
    elif not stamps.equals(pd.date_range(stamps[0], periods=len(stamps), freq="h")):
        raise ValueError("Continuous trajectories cannot bridge missing hours.")
    return summarize(trace, model, daily_terminal=daily_terminal)
