"""Strict public-data normalization and paired temporal uncertainty summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


FIELDS = ("load", "solar", "wind_onshore", "wind_offshore")


def normalize_public_power(payload: dict, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    if payload.get("unit") != "MW" or payload.get("interval_minutes") != 15:
        raise ValueError("Expected MW values for 15-minute interval-start timestamps.")
    rows = [{"timestamp": row["timestamp"], **{k: row["values"].get(k) for k in FIELDS}}
            for row in payload["data"]]
    source = pd.DataFrame(rows)
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
    source = source.loc[source.timestamp.ge(start) & source.timestamp.lt(end)].sort_values("timestamp")
    if source.timestamp.duplicated().any():
        raise ValueError("Duplicate source timestamps must not be silently removed.")
    if not source.timestamp.dt.minute.isin([0, 15, 30, 45]).all() or not source.timestamp.dt.second.eq(0).all():
        raise ValueError("Source timestamps must align to quarter-hour boundaries.")
    for field in FIELDS:
        source[field] = pd.to_numeric(source[field], errors="raise")
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    aligned = source.set_index("timestamp").reindex(grid)
    counts = aligned.notna().resample("h").sum()
    means = aligned.resample("h").mean().where(counts.eq(4))
    hourly = pd.DataFrame({"timestamp": means.index, "total_load_mw": means.load.to_numpy(),
        "solar_generation_mw": means.solar.to_numpy(),
        "wind_generation_mw": (means.wind_onshore + means.wind_offshore).to_numpy()})
    hourly["net_load_mw"] = hourly.total_load_mw - hourly.solar_generation_mw - hourly.wind_generation_mw
    return hourly, {"expected_quarter_hours": len(grid), "received_quarter_hours": len(source),
        "missing_quarter_hours": int(aligned.isna().all(axis=1).sum()),
        "missing_by_source_field": aligned.isna().sum().astype(int).to_dict(),
        "incomplete_hours_by_field": counts.ne(4).sum().astype(int).to_dict(),
        "hourly_rows": len(hourly), "missing_hourly_net_load": int(hourly.net_load_mw.isna().sum()),
        "negative_source_values": {k: int((aligned[k] < 0).sum()) for k in FIELDS},
        "aggregation": "Mean of exactly four quarter-hour MW observations; incomplete hours remain NaN.",
        "descriptive_statistics": hourly.drop(columns="timestamp").describe().to_dict()}


def paired_block_summary(old: pd.DataFrame, new: pd.DataFrame, *, repetitions: int = 2000,
                         block_days: int = 7, seed: int = 20211009) -> dict:
    """Circular moving blocks, paired across strategies, preserving within-day hours.

    Positive deltas mean the new policy improves. Percentile intervals describe
    uncertainty conditional on the observed days; maxima of unseen extremes
    cannot be inferred from this resampling.
    """
    if not old.timestamp.equals(new.timestamp) or not np.allclose(old.net_load_mw, new.net_load_mw):
        raise ValueError("Paired uncertainty requires identical timestamps and actual loads.")
    if len(old) % 24 or not old.timestamp.dt.hour.to_numpy().reshape(-1, 24).tolist() == [list(range(24))] * (len(old) // 24):
        raise ValueError("Uncertainty input must consist of complete UTC days.")
    a = old.controlled_net_load_mw.to_numpy().reshape(-1, 24)
    b = new.controlled_net_load_mw.to_numpy().reshape(-1, 24)
    n = len(a)
    if n < block_days or repetitions < 1:
        raise ValueError("Insufficient days or repetitions.")
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, 2))
    for i in range(repetitions):
        starts = rng.integers(n, size=int(np.ceil(n / block_days)))
        indices = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
        samples[i] = [np.mean(a[indices].max(axis=1) - b[indices].max(axis=1)),
                      np.quantile(a[indices], .95) - np.quantile(b[indices], .95)]
    result = {"repetitions": repetitions, "block_days": block_days, "seed": seed, "days": n,
        "method": "Paired circular moving-block bootstrap of UTC days, percentile 95% intervals",
        "boundary": "Conditional on these observed days and frozen predictions; not an interval for unseen global extremes or field reliability."}
    for j, name in enumerate(("mean_daily_peak_reduction_mw", "p95_reduction_mw")):
        observed = float(np.mean(a.max(axis=1) - b.max(axis=1))) if j == 0 else float(np.quantile(a, .95) - np.quantile(b, .95))
        result[name] = {"estimate": observed, "lower_95": float(np.quantile(samples[:, j], .025)),
                        "upper_95": float(np.quantile(samples[:, j], .975))}
    return result
