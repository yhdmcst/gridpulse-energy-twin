"""Timestamp-aligned stale forecast availability; no forecast fabrication."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODES = {
    "fresh6": (0, 6), "fresh5": (0, 5), "fresh3": (0, 3),
    "delayed1": (1, 5), "delayed3": (3, 3),
}


def delay_panel(frame, calibration, delay, horizon):
    if (not isinstance(delay, int) or not isinstance(horizon, int)
            or delay < 0 or horizon < 1 or delay + horizon > 6):
        raise ValueError("Integer delay and available original forecast horizons required.")
    data = frame.copy()
    data.timestamp = pd.to_datetime(data.timestamp, utc=True)
    data.issued_timestamp = pd.to_datetime(data.issued_timestamp, utc=True)
    times = pd.DatetimeIndex(data.timestamp)
    if (data.empty or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing
            or not times.equals(times.floor("h"))
            or not pd.DatetimeIndex(data.issued_timestamp).equals(times - pd.Timedelta(hours=1))):
        raise ValueError("Unique ordered targets and latest-observation issuance required.")
    source = data.set_index("timestamp")
    origin_times = times - pd.Timedelta(hours=delay)
    keep = origin_times.isin(source.index)
    current = data.loc[keep].reset_index(drop=True)
    origin_times = origin_times[keep]
    old = source.loc[origin_times]
    panel = current[["timestamp"]].copy()
    panel["source_first_target_timestamp"] = origin_times
    panel["issued_timestamp"] = old.issued_timestamp.to_numpy()
    panel["available_timestamp"] = origin_times + pd.Timedelta(hours=delay)
    panel["observation_delay_hours"] = delay
    panel["forecast_hours"] = horizon
    panel["net_load_mw"] = current.actual_h1_mw.to_numpy(float)
    for h in range(1, horizon + 1):
        lead = delay + h
        target_hour = (panel.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
        radius = target_hour.map({int(k): v for k, v in calibration["radii"][str(lead)].items()})
        prediction = old[f"prediction_h{lead}_mw"].to_numpy(float)
        if (not np.isfinite(prediction).all() or radius.isna().any()
                or not np.isfinite(radius).all() or radius.lt(0).any()):
            raise ValueError("Finite forecast and original-lead calibration required.")
        panel[f"prediction_h{h}_mw"] = prediction
        panel[f"lower_h{h}_mw"] = prediction - radius
        panel[f"upper_h{h}_mw"] = prediction + radius
    if not np.isfinite(panel.net_load_mw).all():
        raise ValueError("Finite evaluation actual load required.")
    return panel


def common_panels(frames, calibration):
    from src.balanced_storage import complete_utc_days
    panels, common = {}, None
    for method, frame in frames.items():
        for name, (delay, horizon) in MODES.items():
            p = delay_panel(frame, calibration["methods"][method], delay, horizon)
            panels[(method, name)] = p
            index = pd.Index(p.timestamp)
            common = index if common is None else common.intersection(index)
    if common is None:
        raise ValueError("At least one method required.")
    first = next(iter(panels.values()))
    kept, omitted = complete_utc_days(first.loc[first.timestamp.isin(common)])
    prepared = {}
    excluded = {}
    for (method, name), panel in panels.items():
        p = panel.loc[panel.timestamp.isin(kept.timestamp)].reset_index(drop=True)
        if (not p.timestamp.equals(kept.timestamp)
                or not np.allclose(p.net_load_mw, kept.net_load_mw, rtol=0, atol=1e-6)):
            raise ValueError("All modes/seeds must share current actuals and hours.")
        prepared.setdefault(method, {})[name] = p
        excluded[f"{method}/{name}"] = len(panel) - len(p)
    return prepared, {
        "rows": len(kept), "days": len(kept) // 24,
        "start": kept.timestamp.min().isoformat(), "end": kept.timestamp.max().isoformat(),
        "omitted_partial_days": omitted, "excluded_rows": excluded,
    }
