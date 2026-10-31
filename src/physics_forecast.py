"""Causal paired MLP experiment with accounting regularization, not a PINN."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

TARGETS = ("total_load_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw")
VALIDATION_START = pd.Timestamp("2019-01-12T20:00:00Z")
TEST_START = pd.Timestamp("2019-08-09T21:00:00Z")
POWER_SCALE = 10000.0


def causal_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    stamps = pd.DatetimeIndex(data.timestamp)
    if (data.empty or stamps.hasnans or stamps.has_duplicates
            or not stamps.equals(pd.date_range(stamps[0], periods=len(data), freq="h"))):
        raise ValueError("Complete ordered hourly input required; retain missing values.")
    for column in TARGETS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    valid = data[list(TARGETS)].dropna()
    residual = valid.net_load_mw - valid.total_load_mw + valid.solar_generation_mw + valid.wind_generation_mw
    if not np.isfinite(valid.to_numpy()).all() or residual.abs().max() > 1e-4:
        raise ValueError("Finite targets and defined net-load accounting required.")
    features = []
    for column in TARGETS:
        for lag in (1, 2, 3, 24, 25, 168):
            name = f"{column}_lag{lag}"
            data[name] = data[column].shift(lag)
            features.append(name)
        for kind in ("mean", "std"):
            name = f"{column}_{kind}24"
            history = data[column].shift(1).rolling(24, min_periods=24)
            data[name] = history.mean() if kind == "mean" else history.std()
            features.append(name)
    for name, values, period in (
        ("hour", data.timestamp.dt.hour, 24),
        ("weekday", data.timestamp.dt.dayofweek, 7),
        ("month", data.timestamp.dt.month - 1, 12),
    ):
        for kind, function in (("sin", np.sin), ("cos", np.cos)):
            column = f"{name}_{kind}"
            data[column] = function(2 * np.pi * values / period)
            features.append(column)
    data = data.dropna(subset=[*TARGETS, *features]).reset_index(drop=True)
    if not np.isfinite(data[features].to_numpy()).all():
        raise ValueError("Nonfinite features.")
    return data, features


class ResidualMLP(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 4),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
        return latest + self.layers(x)


def accounting_loss(prediction: torch.Tensor) -> torch.Tensor:
    error = prediction[:, 3] - prediction[:, 0] + prediction[:, 1] + prediction[:, 2]
    return error.square().mean()


def project_accounting(prediction: np.ndarray) -> np.ndarray:
    data = np.asarray(prediction, dtype=float).copy()
    normal = np.array([-1., 1., 1., 1.])
    data -= (data @ normal)[:, None] * normal[None, :] / 4.
    data[:, :3] = np.maximum(data[:, :3], 0.)
    data[:, 3] = data[:, 0] - data[:, 1] - data[:, 2]
    return data
