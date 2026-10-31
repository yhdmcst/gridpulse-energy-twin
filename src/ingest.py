from __future__ import annotations

import argparse
import hashlib
import csv
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
RAW_FILENAME = "opsd_time_series_60min_singleindex.csv"
REGION = "DE"
RAW_COLUMNS = {
    "utc_timestamp": "timestamp",
    "DE_load_actual_entsoe_transparency": "total_load_mw",
    "DE_solar_generation_actual": "solar_generation_mw",
    "DE_wind_generation_actual": "wind_generation_mw",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / RAW_FILENAME
    if force or not path.exists():
        from src.downloads import atomic_download
        atomic_download(SOURCE_URL,path,validator=validate_raw_header)
    validate_raw_header(path)
    return path


def validate_raw_header(path: Path) -> None:
    with path.open(encoding="utf-8-sig",newline="") as stream:
        header = next(csv.reader(stream),[])
    if not set(RAW_COLUMNS).issubset(header):
        raise ValueError("OPSD cache lacks required CSV fields; remove the invalid cache and download again.")


def build_standard_dataset(raw_path: Path, output_path: Path) -> dict:
    usecols = list(RAW_COLUMNS)
    frame = pd.read_csv(raw_path, usecols=usecols, parse_dates=["utc_timestamp"])
    frame = frame.rename(columns=RAW_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in RAW_COLUMNS.values():
        if column != "timestamp":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    source_numeric = ["total_load_mw", "solar_generation_mw", "wind_generation_mw"]
    all_missing_rows = int(frame[source_numeric].isna().all(axis=1).sum())
    frame = frame.loc[~frame[source_numeric].isna().all(axis=1)].copy()
    frame["net_load_mw"] = (
        frame["total_load_mw"] - frame["solar_generation_mw"] - frame["wind_generation_mw"]
    )
    frame.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return {
        "source_url": SOURCE_URL,
        "raw_file": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "region": REGION,
        "rows": len(frame),
        "all_source_fields_missing_rows_removed": all_missing_rows,
        "columns": list(frame.columns),
        "timestamp_start_utc": frame["timestamp"].min().isoformat(),
        "timestamp_end_utc": frame["timestamp"].max().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    raw_path = download_raw(args.raw_dir, args.force_download)
    args.processed.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_standard_dataset(raw_path, args.processed)
    print(metadata)


if __name__ == "__main__":
    main()
