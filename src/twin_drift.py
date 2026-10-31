"""Lightweight drift monitoring for the energy digital twin.

The monitor borrows the evidence-first lifecycle idea from industrial
monitoring systems: establish a baseline, classify changes, and recommend a
recalibration action without silently changing the frozen controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftThresholds:
    mean_relative_change: float = 0.25
    std_relative_change: float = 0.40
    ramp_relative_change: float = 0.50
    min_baseline_rows: int = 24
    min_current_rows: int = 6


@dataclass(frozen=True)
class DriftReport:
    drift_detected: bool
    drift_types: tuple[str, ...]
    score: float
    urgency: str
    evidence: tuple[str, ...]
    recommendations: tuple[str, ...]
    recalibration_recommended: bool
    baseline_stats: dict[str, dict[str, float]]
    current_stats: dict[str, dict[str, float]]


def _stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    if frame.empty:
        raise ValueError("A nonempty monitoring window is required.")
    if "timestamp" in frame:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        if (stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing
                or not stamps.equals(pd.date_range(stamps[0], periods=len(stamps), freq="h"))):
            raise ValueError("Monitoring requires ordered contiguous hourly timestamps.")
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Monitoring values must be finite; do not drop missing observations.")
        ramps = np.abs(np.diff(values))
        stats[column] = {
            "rows": float(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ramp_p95": float(np.quantile(ramps, 0.95)) if len(ramps) else 0.0,
        }
    return stats


class EnergyTwinDriftMonitor:
    """Compare a current twin window against a frozen baseline window."""

    FEATURE_GROUPS = {
        "load_regime": ("net_load_mw",),
        "renewable_regime": ("solar_generation_mw", "wind_generation_mw"),
        "forecast_model": ("residual_mw",),
        "controlled_regime": ("controlled_net_load_mw",),
    }

    def __init__(self, thresholds: DriftThresholds | None = None):
        self.thresholds = thresholds or DriftThresholds()
        if (not np.isfinite(list(vars(self.thresholds).values())).all()
                or min(vars(self.thresholds).values()) <= 0):
            raise ValueError("Positive finite drift thresholds required.")
        self._baseline: dict[str, dict[str, float]] | None = None

    def set_baseline(self, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Freeze baseline statistics without modifying the source frame."""

        columns = sorted({column for group in self.FEATURE_GROUPS.values() for column in group})
        stats = _stats(frame, columns)
        if not stats or len(frame) < self.thresholds.min_baseline_rows:
            raise ValueError("Baseline window is too small for drift monitoring.")
        self._baseline = stats
        return stats

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    def analyze(self, frame: pd.DataFrame) -> DriftReport:
        if self._baseline is None:
            raise ValueError("Set a baseline before analyzing drift.")
        columns = sorted({column for group in self.FEATURE_GROUPS.values() for column in group})
        current = _stats(frame, columns)
        if set(self._baseline).difference(current):
            raise ValueError("Current window is missing baseline features.")
        if len(frame) < self.thresholds.min_current_rows:
            raise ValueError("Current window is too small for drift monitoring.")

        drift_types: list[str] = []
        evidence: list[str] = []
        severity_values: list[float] = []
        baseline = self._baseline

        for group, group_columns in self.FEATURE_GROUPS.items():
            group_hits = []
            for column in group_columns:
                if column not in baseline or column not in current:
                    continue
                old = baseline[column]
                new = current[column]
                mean_delta = abs(new["mean"] - old["mean"]) / max(abs(old["mean"]), 1.0)
                std_delta = abs(new["std"] - old["std"]) / max(old["std"], 1.0)
                ramp_delta = abs(new["ramp_p95"] - old["ramp_p95"]) / max(old["ramp_p95"], 1.0)
                if mean_delta >= self.thresholds.mean_relative_change:
                    group_hits.append(f"{column} 均值变化 {mean_delta:.1%}")
                    severity_values.append(min(mean_delta, 1.0))
                if std_delta >= self.thresholds.std_relative_change:
                    group_hits.append(f"{column} 波动变化 {std_delta:.1%}")
                    severity_values.append(min(std_delta, 1.0))
                if ramp_delta >= self.thresholds.ramp_relative_change:
                    group_hits.append(f"{column} 爬坡 P95 变化 {ramp_delta:.1%}")
                    severity_values.append(min(ramp_delta, 1.0))
            if group_hits:
                drift_types.append(group)
                evidence.extend(group_hits)

        severity = float(np.mean(severity_values)) if severity_values else 0.0
        score = float(np.clip(100.0 - 70.0 * severity, 0.0, 100.0))
        detected = bool(drift_types)
        if not detected:
            urgency = "low"
        elif score < 55:
            urgency = "high"
        elif score < 75:
            urgency = "medium"
        else:
            urgency = "low"

        recommendations: list[str] = []
        if "forecast_model" in drift_types:
            recommendations.append("重新检查预测残差与区间覆盖，必要时重新校准预测区间。")
        if "renewable_regime" in drift_types:
            recommendations.append("检查新能源资源状态变化，优先使用保守储备并核对光伏/风电数据源。")
        if "load_regime" in drift_types:
            recommendations.append("检查净负荷分布和爬坡变化，不直接沿用原阈值作为现场阈值。")
        if "controlled_regime" in drift_types:
            recommendations.append("复核储能动作与 SOC 轨迹，确认策略变化来自输入而非状态镜像异常。")
        if not recommendations:
            recommendations.append("当前窗口与基线一致，继续积累观测后再更新基线。")

        return DriftReport(
            drift_detected=detected,
            drift_types=tuple(drift_types),
            score=score,
            urgency=urgency,
            evidence=tuple(evidence),
            recommendations=tuple(recommendations),
            recalibration_recommended=detected and ("forecast_model" in drift_types or score < 75.0),
            baseline_stats=baseline,
            current_stats=current,
        )
