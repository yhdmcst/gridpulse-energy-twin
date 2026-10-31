from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def contiguous_missing_hours(timestamps: pd.Series) -> dict:
    observed = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True).dropna().sort_values().unique())
    if len(observed) < 2:
        return {"missing_timestamps": 0, "max_consecutive_missing_hours": 0, "frequency_minutes": None}
    deltas = pd.Series(observed[1:] - observed[:-1]).dt.total_seconds().div(60)
    frequency = int(deltas.mode().iloc[0])
    missing = (deltas / frequency - 1).clip(lower=0).astype(int)
    return {
        "missing_timestamps": int(missing.sum()),
        "max_consecutive_missing_hours": int(missing.max()),
        "frequency_minutes": frequency,
    }


def run_quality_checks(path: Path) -> dict:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric = ["total_load_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw"]
    time = contiguous_missing_hours(frame["timestamp"])
    recomputed = frame["total_load_mw"] - frame["solar_generation_mw"] - frame["wind_generation_mw"]
    consistency = (frame["net_load_mw"] - recomputed).abs()
    values = {}
    for column in numeric:
        series = frame[column]
        values[column] = {
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "negative_count": int((series < 0).sum()),
            "zero_count": int((series == 0).sum()),
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        }
    result = {
        "file": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "time_coverage": {
            "start_utc": frame["timestamp"].min().isoformat(),
            "end_utc": frame["timestamp"].max().isoformat(),
            "duration_days": float((frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400),
            **time,
        },
        "duplicate_timestamp_count": int(frame["timestamp"].duplicated().sum()),
        "numeric_fields": values,
        "net_load_formula_max_abs_error": float(consistency.max(skipna=True)),
        "rows_with_any_missing_numeric": int(frame[numeric].isna().any(axis=1).sum()),
        "quality_notes": [
            "Negative values are reported as suspicious observations, not automatically deleted.",
            "Missing intervals and field gaps must be handled explicitly before modeling.",
            "All timestamps are normalized to UTC; local-time presentation is a later concern.",
        ],
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/data-quality-report.json"))
    args = parser.parse_args()
    result = run_quality_checks(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
