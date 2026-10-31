"""Exercise the current v8 dashboard and its dependent condition selectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def check(root: Path) -> dict:
    sys.path.insert(0, str(root.resolve()))
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    errors, checks = [], []

    def clean(label):
        failures = [item.message for item in app.exception] + [item.value for item in app.error]
        errors.extend(f"{label}: {failure}" for failure in failures)
        if not failures:
            checks.append(label)
        return not failures

    def select(label, value):
        widget = next(item for item in app.selectbox if item.label == label)
        widget.set_value(value).run()
        return clean(f"{label}={value}")

    if not clean("v8_startup"):
        return {"passed": False, "checks": checks, "errors": errors}
    if app.title[0].value != "GridPulse V8":
        errors.append("Dashboard title is not V8.")
    if not any(item.label == "范围内电量恢复" and item.value == "30 / 30" for item in app.metric):
        errors.append("V8 recovery count is missing.")
    if not any(item.label == "全部性能预算通过" and item.value == "11 / 30" for item in app.metric):
        errors.append("V8 performance count is missing.")
    for label, value in (
        ("评价年份", "2021"), ("预测条件", "delayed1"),
        ("设备条件", "efficiency_085"), ("预测条件", "fresh6"),
        ("设备条件", "combined90"), ("故障条件", "combined_fault"),
    ):
        if not select(label, value):
            break
    if not errors:
        next(item for item in app.checkbox if item.label.startswith("仅显示回退")).check().run()
        clean("fault_log_filter")
        select("设备条件", "outside")
        if not app.warning:
            errors.append("Outside-envelope recovery failure is missing.")
        select("结果策略", "envelope_reference")
        select("结果策略", "baseline_v7")
        select("结果策略", "envelope_tail")
        next(item for item in app.checkbox if item.label == "叠加 V7 对照").uncheck().run()
        clean("comparison_toggle")
        dates = app.date_input[0]
        dates.set_value((dates.max, dates.max)).run()
        clean("last_evaluation_day")
    return {"passed": not errors, "checks": checks, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
