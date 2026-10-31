"""Versioned direct six-horizon features and matched planning adapters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.physics_forecast import TARGETS, VALIDATION_START, TEST_START, causal_frame

HORIZONS = tuple(range(1, 7))
LABELS = [f"target_h{h}_{c}" for h in HORIZONS for c in TARGETS]


def build_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame, features = causal_frame(raw)
    source = raw.copy()
    source["timestamp"] = pd.to_datetime(source.timestamp, utc=True)
    source = source.set_index("timestamp")
    additions = {}
    for h in HORIZONS:
        target = frame.timestamp + pd.Timedelta(hours=h - 1)
        truth = source.reindex(pd.DatetimeIndex(target))[list(TARGETS)].to_numpy(float)
        for i, column in enumerate(TARGETS):
            additions[f"target_h{h}_{column}"] = truth[:, i]
        if h > 1:
            for name, values, period in (
                ("hour", target.dt.hour, 24), ("weekday", target.dt.dayofweek, 7),
                ("month", target.dt.month - 1, 12),
            ):
                for kind, function in (("sin", np.sin), ("cos", np.cos)):
                    column = f"known_h{h}_{name}_{kind}"
                    additions[column] = function(2 * np.pi * values / period)
                    features.append(column)
    frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    frame["issued_timestamp"] = frame.timestamp - pd.Timedelta(hours=1)
    frame["last_target_timestamp"] = frame.timestamp + pd.Timedelta(hours=5)
    return frame, features


def restrict_split(frame: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    finite = np.isfinite(frame[LABELS].to_numpy(float)).all(axis=1)
    mask = pd.Series(finite, index=frame.index)
    if start is not None:
        mask &= frame.timestamp >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.last_target_timestamp < pd.Timestamp(end)
    return frame.loc[mask].reset_index(drop=True)


def split_original(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": restrict_split(frame, end=VALIDATION_START),
        "validation": restrict_split(frame, start=VALIDATION_START, end=TEST_START),
        "test": restrict_split(frame, start=TEST_START),
    }


def arrays(frame: pd.DataFrame, features: list[str], mean, scale):
    x = (frame[features].to_numpy(float) - mean) / scale
    latest = frame[[f"{c}_lag1" for c in TARGETS]].to_numpy(float)
    y = frame[LABELS].to_numpy(float).reshape(-1, 6, 4)
    return x, latest, y


class SixHorizonMLP(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 24),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
        return latest[:, None, :] + self.layers(x).reshape(-1, 6, 4)


def validate_predictions(frame: pd.DataFrame) -> None:
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
    if (frame.empty or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing or not times.equals(times.floor("h"))):
        raise ValueError("Nonempty unique ordered hourly prediction grid required.")
    issued = pd.DatetimeIndex(pd.to_datetime(frame.issued_timestamp, utc=True))
    if not issued.equals(times - pd.Timedelta(hours=1)):
        raise ValueError("Issuance must be the latest observed hour t-1.")
    columns = [f"{prefix}_h{h}_mw" for h in HORIZONS for prefix in ("actual", "prediction")]
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError("All six targets and predictions must be finite.")


def shared_days(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    from src.balanced_storage import complete_utc_days
    if not frames:
        raise ValueError("At least one predictor required.")
    common = None
    for frame in frames.values():
        validate_predictions(frame)
        times = pd.Index(frame.timestamp)
        common = times if common is None else common.intersection(times)
    first = next(iter(frames.values()))
    kept, omitted = complete_utc_days(first.loc[first.timestamp.isin(common)])
    output = {}
    truths = [f"actual_h{h}_mw" for h in HORIZONS]
    for name, frame in frames.items():
        sub = frame.loc[frame.timestamp.isin(kept.timestamp)].reset_index(drop=True)
        if (not sub.timestamp.equals(kept.timestamp)
                or not np.allclose(sub[truths], kept[truths], rtol=0, atol=1e-6)):
            raise ValueError("Matched target timestamps and actual loads required.")
        output[name] = sub
    return output, {
        "rows": len(kept), "days": len(kept) // 24,
        "start": kept.timestamp.min().isoformat(), "end": kept.timestamp.max().isoformat(),
        "omitted_partial_days": omitted,
        "excluded_prediction_rows": {k: len(v) - len(kept) for k, v in frames.items()},
    }


def calibrate_six(validation: dict[str, pd.DataFrame]) -> dict:
    reference = next(iter(validation.values()))
    for frame in validation.values():
        validate_predictions(frame)
        last_target = frame.timestamp + pd.Timedelta(hours=5)
        if frame.timestamp.min() < VALIDATION_START or last_target.max() >= TEST_START:
            raise ValueError("All calibration targets must lie inside the validation period.")
        if (not frame.timestamp.equals(reference.timestamp)
                or not np.allclose(frame.actual_h1_mw, reference.actual_h1_mw, rtol=0, atol=1e-6)):
            raise ValueError("Calibration must share actual values and timestamps.")
    methods = {}
    for name, frame in validation.items():
        radii, counts = {}, {}
        for h in HORIZONS:
            errors = (frame[f"actual_h{h}_mw"] - frame[f"prediction_h{h}_mw"]).abs()
            hours = (frame.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
            radius = errors.groupby(hours).quantile(.9)
            if set(radius.index) != set(range(24)):
                raise ValueError("Each horizon requires all 24 target hours.")
            radii[str(h)] = {str(k): float(v) for k, v in radius.items()}
            counts[str(h)] = {str(k): int(v) for k, v in errors.groupby(hours).size().items()}
        methods[name] = {
            "radii": radii, "counts": counts,
            "uncertainty_scale_mw": max(1., 1.1 * max(v for r in radii.values() for v in r.values())),
        }
    return {
        "gates": {
            "peak_threshold_mw": float(reference.actual_h1_mw.quantile(.95)),
            "charge_threshold_mw": float(reference.actual_h1_mw.quantile(.25)),
            "release_threshold_mw": float(reference.actual_h1_mw.quantile(.99)),
        },
        "methods": methods,
    }


def planning_panel(frame: pd.DataFrame, calibration: dict, hours: int) -> pd.DataFrame:
    validate_predictions(frame)
    if hours not in (1, 6):
        raise ValueError("This protocol compares exactly one versus six hours.")
    panel = frame[["timestamp", "issued_timestamp"]].copy()
    panel["net_load_mw"] = frame.actual_h1_mw
    for h in range(1, hours + 1):
        target_hour = (frame.timestamp + pd.Timedelta(hours=h - 1)).dt.hour
        radius = target_hour.map({int(k): v for k, v in calibration["radii"][str(h)].items()})
        if radius.isna().any() or not np.isfinite(radius).all() or radius.lt(0).any():
            raise ValueError("Finite nonnegative calibration required.")
        prediction = frame[f"prediction_h{h}_mw"]
        panel[f"prediction_h{h}_mw"] = prediction
        panel[f"lower_h{h}_mw"] = prediction - radius
        panel[f"upper_h{h}_mw"] = prediction + radius
    return panel
