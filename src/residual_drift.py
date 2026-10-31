"""Validation-calibrated residual-window diagnostic; no automatic model update."""
from __future__ import annotations

import math

import numpy as np


class CalibratedResidualMonitor:
    def __init__(self, alpha: float = .05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha
        self.reference: np.ndarray | None = None
        self.scale: float | None = None
        self.threshold: float | None = None

    @staticmethod
    def statistics(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("Finite residual window of at least two rows required.")
        return np.array([x.mean(), x.std(), np.abs(np.diff(x)).mean()])

    def score(self, values: np.ndarray) -> float:
        if self.reference is None or self.scale is None:
            raise ValueError("Reference not fitted.")
        current = self.statistics(values)
        return float(np.max(np.abs(current - self.reference)) / self.scale)

    def fit(self, training: np.ndarray, validation_windows: list[np.ndarray]) -> dict:
        if len(validation_windows) < 20:
            raise ValueError("At least 20 calibration windows required.")
        self.reference = self.statistics(training)
        self.scale = max(float(self.reference[1]), 1.)
        scores = np.array([self.score(x) for x in validation_windows])
        rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - self.alpha)))
        self.threshold = float(np.sort(scores)[rank - 1])
        return {"reference_statistics": self.reference.tolist(),
                "scale_mw": self.scale, "threshold": self.threshold,
                "alpha": self.alpha, "validation_windows": len(scores),
                "validation_alarm_rate": float((scores > self.threshold).mean())}

    def analyze(self, values: np.ndarray) -> dict:
        if self.threshold is None:
            raise ValueError("Calibrate before analyzing.")
        value = self.score(values)
        return {"score": value, "threshold": self.threshold,
                "alarm": value > self.threshold}
