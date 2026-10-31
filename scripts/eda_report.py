from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    input_path = Path("data/processed/net_load_hourly.csv")
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    fields = {
        "total_load_mw": "Total load",
        "solar_generation_mw": "Solar generation",
        "wind_generation_mw": "Wind generation",
        "net_load_mw": "Net load",
    }

    ax = frame[list(fields)].plot(figsize=(14, 6), linewidth=0.7)
    ax.set_title("Germany hourly load, renewable generation, and net load")
    ax.set_ylabel("MW")
    ax.set_xlabel("UTC")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "01_full_period_timeseries.png", dpi=150)
    plt.close(ax.figure)

    week = frame.loc["2019-06-01":"2019-06-07"]
    ax = week[list(fields)].plot(figsize=(14, 6), linewidth=1.2)
    ax.set_title("Representative week: hourly profiles")
    ax.set_ylabel("MW")
    ax.set_xlabel("UTC")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "02_representative_week.png", dpi=150)
    plt.close(ax.figure)

    hourly = frame.assign(hour=frame.index.hour).groupby("hour")["net_load_mw"].mean()
    ax = hourly.plot(kind="line", marker="o", figsize=(10, 5))
    ax.set_title("Average net load by UTC hour")
    ax.set_ylabel("MW")
    ax.set_xlabel("Hour of day")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "03_average_daily_net_load.png", dpi=150)
    plt.close(ax.figure)

    weekday = frame.assign(
        day_type=np.where(frame.index.dayofweek >= 5, "Weekend", "Weekday"),
        hour=frame.index.hour,
    )
    profile = weekday.groupby(["day_type", "hour"])["net_load_mw"].mean().unstack(0)
    ax = profile.plot(figsize=(10, 5), marker="o")
    ax.set_title("Weekday versus weekend average net load")
    ax.set_ylabel("MW")
    ax.set_xlabel("Hour of day")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "04_weekday_weekend_net_load.png", dpi=150)
    plt.close(ax.figure)

    ax = frame["net_load_mw"].dropna().plot(kind="hist", bins=50, figsize=(10, 5))
    ax.set_title("Net load distribution")
    ax.set_xlabel("MW")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "05_net_load_distribution.png", dpi=150)
    plt.close(ax.figure)

    ramps = frame[["solar_generation_mw", "wind_generation_mw"]].diff()
    ax = ramps.plot(kind="hist", bins=80, alpha=0.65, figsize=(10, 5))
    ax.set_title("Hourly renewable generation ramps")
    ax.set_xlabel("MW change from previous hour")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "06_renewable_ramp_distribution.png", dpi=150)
    plt.close(ax.figure)

    print(f"Wrote six figures to {output_dir}")


if __name__ == "__main__":
    main()
