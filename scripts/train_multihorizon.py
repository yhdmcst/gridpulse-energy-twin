from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multihorizon import train_direct_horizon_models


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = train_direct_horizon_models(
        root / "data/processed/net_load_hourly.csv",
        root / "outputs/models",
    )
    print(report)


if __name__ == "__main__":
    main()
