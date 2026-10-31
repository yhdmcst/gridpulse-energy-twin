"""Validated, read-only access to the completed v8 experiment."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICY = "envelope_tail"
KEYS = ("year", "mode", "case", "event", "policy")
METRICS = ("controlled_peak_mw", "controlled_p95_mw", "mean_daily_peak_mw",
           "throughput_mwh", "conversion_loss_mwh", "max_abs_terminal_error_mwh")


@dataclass
class FusionEvidence:
    directory: Path
    hashes: dict
    report: dict
    catalog: pd.DataFrame

    def metrics(self, year: str, mode: str, case: str, event: str, policy=POLICY) -> dict:
        return self.report["methods"][str(year)][mode][case][event][policy]

    def checked_bytes(self, relative: str) -> bytes:
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
            raise ValueError(f"结果路径无效：{relative}")
        expected = self.hashes.get(relative)
        if not expected:
            raise ValueError(f"结果未列入冻结清单：{relative}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"结果哈希不匹配：{relative}")
        return data

    def trace(self, metrics: dict) -> pd.DataFrame:
        data = self.checked_bytes(metrics["trajectory"])
        if hashlib.sha256(data).hexdigest() != metrics["trajectory_sha256"]:
            raise ValueError("轨迹与指标记录不一致。")
        frame = pd.read_csv(io.BytesIO(data), parse_dates=["timestamp"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        columns = ["net_load_mw", "controlled_net_load_mw", "charge_mw",
                   "discharge_mw", "soc_before_mwh", "storage_soc_mwh"]
        if len(frame) != metrics["rows"] or not np.isfinite(frame[columns].to_numpy(float)).all():
            raise ValueError("轨迹行数或数值异常。")
        times = pd.DatetimeIndex(frame.timestamp)
        if not times.equals(pd.date_range(times[0], periods=len(times), freq="h")):
            raise ValueError("轨迹时间轴存在缺口。")
        model = metrics["physical_parameters"]
        eta = model["efficiency"]
        error = frame.storage_soc_mwh - frame.soc_before_mwh - eta * frame.charge_mw + frame.discharge_mw / eta
        continuity = frame.soc_before_mwh.iloc[1:].to_numpy() - frame.storage_soc_mwh.iloc[:-1].to_numpy()
        if error.abs().max() > 1e-5 or np.abs(continuity).max(initial=0) > 1e-5:
            raise ValueError("轨迹能量核算或状态连续性异常。")
        return frame


def load_evidence(source: Path) -> FusionEvidence:
    directory = source / "outputs/fusion_v8"
    hashes = json.loads((directory / "ARTIFACT_SHA256.json").read_text(encoding="utf-8"))
    evidence = FusionEvidence(directory, hashes, {}, pd.DataFrame())
    report = json.loads(evidence.checked_bytes("metrics.json"))
    decision = json.loads(evidence.checked_bytes("decision.json"))
    selection = json.loads(evidence.checked_bytes("selection.json"))
    status = json.loads(evidence.checked_bytes("status.json"))
    if not (
        report["status"] == status["status"] == "complete"
        and report["selected_policy"] == decision["selected_policy"] == selection["selected_policy"] == POLICY
        and report["decision"] == decision
        and hashlib.sha256(evidence.checked_bytes("selection.json")).hexdigest() == decision["selection_sha256"]
    ):
        raise ValueError("v8 完成状态或锁定选择不一致。")
    rows = []
    for year, modes in report["methods"].items():
        for mode, cases in modes.items():
            for case, events in cases.items():
                for event, policies in events.items():
                    baseline = policies["baseline_v7"]
                    for policy, metrics in policies.items():
                        condition = dict(zip(KEYS, (year, mode, case, event, policy)))
                        recorded = metrics["condition"]
                        if any(str(recorded[key]) != str(condition[key]) for key in KEYS[:-1]):
                            raise ValueError("实验条件标签与结果层级不一致。")
                        if (policy == "baseline_v7" and recorded.get("variant") != "supervised"
                                or policy != "baseline_v7" and recorded.get("policy") != policy):
                            raise ValueError("实验策略标签不一致。")
                        values = [metrics[key] for key in METRICS]
                        if not np.isfinite(values).all():
                            raise ValueError("实验汇总包含非有限值。")
                        rows.append({
                            **condition, **{key: metrics[key] for key in METRICS},
                            "inside_envelope": bool(metrics["plant_within_declared_envelope"]),
                            "recovery_passed": bool(metrics["terminal_requirement_passed"]),
                            "peak_delta_mw": metrics["controlled_peak_mw"] - baseline["controlled_peak_mw"],
                            "p95_delta_mw": metrics["controlled_p95_mw"] - baseline["controlled_p95_mw"],
                            "fallback_hours": metrics["fallback_hours"],
                            "all_budgets_passed": bool(
                                metrics["physical_audit_passed"]
                                and metrics["terminal_requirement_passed"]
                                and metrics["controlled_peak_mw"] <= baseline["controlled_peak_mw"] + 25
                                and metrics["controlled_p95_mw"] <= baseline["controlled_p95_mw"] + 1
                                and metrics["throughput_mwh"] <= baseline["throughput_mwh"] * 1.1 + 1e-6
                                and metrics["conversion_loss_mwh"] <= baseline["conversion_loss_mwh"] * 1.1 + 1e-6
                            ),
                        })
    evidence.report = report
    evidence.catalog = pd.DataFrame(rows)
    if evidence.catalog.duplicated(list(KEYS)).any():
        raise ValueError("实验条件重复。")
    return evidence
