"""Validate dashboard files with portable, actionable error messages."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


class ArtifactError(ValueError):
    pass


JSON_KEYS = {
    "day3_model_metrics.json": ("models",), "alert_evaluation.json": ("test_candidate_risk_rows",),
    "multihorizon_metrics.json": ("models", "planning_window_hours"),
    "storage_scenario_metrics.json": ("risk_aware_optimization", "direct_fixed_reserve_baseline"),
    "reference_baseline_metrics.json": ("test_rows", "metrics"),
    "reserve_ablation_metrics.json": ("experiments",),
    "daily_energy_balance_metrics.json": ("experiments", "comparisons"),
    "peak_response_metrics.json": ("calibration", "periods"),
    "independent_2021_metrics.json": ("experiments", "comparisons", "quarters"),
    "independent_2022_metrics.json": ("experiments", "comparisons", "quarters"),
    "certificates.json": ("cases",), "controller_metrics.json": ("periods",),
    "operating_stress_metrics.json": ("episodes",),
    "renewable_alignment_metrics.json": ("periods", "high_renewable_share_threshold"),
    "interval_audit_metrics.json": ("groups",),
}
CSV_COLUMNS = {
    "net_load_hourly.csv": ("timestamp", "total_load_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw"),
    "hist_gradient_boosting_test_predictions.csv": ("timestamp", "net_load_mw", "hist_gradient_boosting_prediction_mw", "hist_gradient_boosting_lower_mw", "hist_gradient_boosting_upper_mw", "residual_mw"),
    "test_review_queue.csv": ("timestamp", "risk_types", "risk_evidence", "predicted_net_load_mw", "prediction_upper_mw", "risk_score", "review_priority", "residual_mw"),
    "hist_gradient_boosting_validation_permutation_importance.csv": ("feature", "importance_mae_mw"),
}


def reject_nonfinite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            reject_nonfinite(item)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError("结果文件引用了项目目录之外的路径。")
    return path


def inspect_runtime(root, required):
    """Check file formats and trace references; return a cache invalidation key.

    This checks availability and format, not scientific correctness. The
    independent audits and package manifest remain the integrity authority.
    """
    root = Path(root)
    referenced = set(required)
    def references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "timeseries" and isinstance(item, str):
                    referenced.add(item)
                else:
                    references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)
    for relative in required:
        path = safe_path(root, relative)
        try:
            if path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                reject_nonfinite(value)
                if not isinstance(value, dict) or any(k not in value for k in JSON_KEYS.get(path.name, ())):
                    raise ValueError("结果结构缺少必需字段")
                for key in JSON_KEYS.get(path.name, ()):
                    if key in {"models", "metrics", "experiments", "comparisons", "periods", "cases", "episodes", "calibration"}:
                        if not isinstance(value[key], dict) or not value[key]:
                            raise ValueError("结果集合必须是非空对象")
                    elif key == "groups" and (not isinstance(value[key], list) or not value[key]):
                        raise ValueError("区间分组必须是非空列表")
                references(value)
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    header = next(csv.reader(stream), [])
                if any(c not in header for c in CSV_COLUMNS.get(path.name, ())):
                    raise ValueError("CSV 缺少必需字段")
        except (OSError, ValueError, UnicodeError, csv.Error) as exc:
            raise ArtifactError(f"{relative}：无法读取有效结果，请恢复原文件或按复现指南重新生成。") from exc
    stamps = []
    for relative in sorted(referenced):
        path = safe_path(root, relative)
        if not path.is_file():
            raise ArtifactError(f"{relative}：结果引用的轨迹文件缺失。请完整解压项目包。")
        stat = path.stat()
        stamps.append((relative, stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


def validate_frame(frame, name):
    """Validate parsed tables; source data may intentionally retain NaN power."""
    required = CSV_COLUMNS[name]
    if any(c not in frame for c in required):
        raise ArtifactError(f"{name}：CSV 缺少必需字段。")
    if name != "test_review_queue.csv" and frame.empty:
        raise ArtifactError(f"{name}：结果表不能为空。")
    if "timestamp" in frame:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
        if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
            raise ArtifactError(f"{name}：时间戳必须有效、唯一且递增。")
        frame["timestamp"] = stamps
    numeric = [c for c in required if c.endswith("_mw") or c == "risk_score"]
    if name != "net_load_hourly.csv" and not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ArtifactError(f"{name}：数值结果包含缺失值或非有限值。")
