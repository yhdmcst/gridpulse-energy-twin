"""Stress forecast delivery without consulting actual evaluation loads."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stress_feed(panel,*,delay_hours=0,outage_rows=(),bias_mw=0.,allow_warmup=False):
    if type(delay_hours) is not int or delay_hours < 0 or not np.isfinite(bias_mw):
        raise ValueError("Nonnegative integer delay and finite forecast bias required.")
    data = panel.copy().reset_index(drop=True)
    stamps = pd.to_datetime(data.timestamp,utc=True)
    if not stamps.is_monotonic_increasing or stamps.duplicated().any():
        raise ValueError("Unique ordered forecast issuance times required.")
    blocked = set(outage_rows)
    if any(type(i) is not int or not 0 <= i < len(data) for i in blocked):
        raise ValueError("Outage rows must be valid integer positions.")
    n = max(int(c.removeprefix("prediction_h").removesuffix("_mw")) for c in data if c.startswith("prediction_h"))
    latest = None
    fallback_hours = stale_hours = 0
    for i,stamp in enumerate(stamps):
        # Latest delivered issue is old by delay_hours. During outage no new
        # issue is available. Do not shift columns blindly across missing dates.
        limit = stamp-pd.Timedelta(hours=delay_hours)
        eligible = np.flatnonzero(stamps.le(limit).to_numpy())
        candidate = int(eligible[-1]) if len(eligible) else None
        if i not in blocked and candidate is not None:
            latest = candidate
        if latest is None:
            if not allow_warmup:
                raise ValueError("Warmup forecast issuance required before a delayed episode.")
            # Leading delay hours use nominal issues, must be excluded from
            # evaluation. Caller supplies an entire previous warmup day.
            latest = i
        age = int((stamp-stamps.iloc[latest])/pd.Timedelta(hours=1))
        if age:
            stale_hours += 1
        fallback = False
        for h in range(1,n+1):
            target_h = age+h
            source_h = min(n,target_h)
            if target_h > n:
                fallback = True
            # Beyond published horizon carry the last point forecast and band.
            # No current/future actual load enters fallback decisions.
            for prefix in ("prediction","upper","lower"):
                key = f"{prefix}_h{h}_mw"
                if key in data:
                    data.loc[i,key] = float(panel.iloc[latest][f"{prefix}_h{source_h}_mw"])+bias_mw
        fallback_hours += fallback
    return data,{"stale_issue_hours":stale_hours,"beyond_horizon_fallback_hours":fallback_hours,
        "fallback":"Carry last published horizon point/band; never use evaluation actual loads."}
