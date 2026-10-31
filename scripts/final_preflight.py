from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file() and relative.startswith(".codex-finalizer/"):
        path = ROOT.parent / "05_复现与合规/验证回执" / Path(relative).name
    return json.loads(path.read_text(encoding="utf-8"))


def audit_daily_energy(report: dict) -> bool:
    """Recompute conservation and matching from delivered hourly traces."""
    reference = None
    eta = report["parameters"]["efficiency"]
    for metrics in report["experiments"].values():
        trace = pd.read_csv(ROOT / metrics["timeseries"], parse_dates=["timestamp"])
        stamps = trace.timestamp.astype(str).tolist()
        if reference is None:
            reference = stamps
        if stamps != reference or len(trace) != metrics["days"] * 24:
            return False
        groups = trace.groupby(trace.timestamp.dt.floor("D"))
        if not all(len(day) == 24 and day.timestamp.dt.hour.tolist() == list(range(24)) for _, day in groups):
            return False
        energy = trace.soc_before_mwh + eta * trace.charge_mw - trace.discharge_mw / eta
        controlled = trace.net_load_mw + trace.charge_mw - trace.discharge_mw
        loss = ((1 - eta) * trace.charge_mw + (1 / eta - 1) * trace.discharge_mw).sum()
        if not (np.allclose(energy, trace.storage_soc_mwh, atol=1e-6, rtol=0)
                and np.allclose(controlled, trace.controlled_net_load_mw, atol=1e-6, rtol=0)
                and abs(loss - metrics["conversion_loss_mwh"]) < 1e-5
                and abs(controlled.quantile(.95) - metrics["controlled_p95_mw"]) < 1e-5
                and trace.charge_mw.between(-1e-6, report["parameters"]["charge_power_mw"] + 1e-6).all()
                and trace.discharge_mw.between(-1e-6, report["parameters"]["discharge_power_mw"] + 1e-6).all()
                and trace.storage_soc_mwh.ge(metrics["capacity_mwh"] * metrics["hard_min_fraction"] - 1e-6).all()
                and np.minimum(trace.charge_mw, trace.discharge_mw).le(1e-6).all()):
            return False
        for _, day in groups:
            if abs(day.soc_before_mwh.iloc[0] - metrics["initial_soc_per_day_mwh"]) > 1e-6:
                return False
            if not np.allclose(day.soc_before_mwh.iloc[1:], day.storage_soc_mwh.iloc[:-1], atol=1e-6, rtol=0):
                return False
            if metrics["daily_terminal"] and abs(day.storage_soc_mwh.iloc[-1] - metrics["initial_soc_per_day_mwh"]) > 1e-6:
                return False
        if metrics["daily_terminal"] and not trace.storage_soc_mwh.le(metrics["initial_soc_per_day_mwh"] + 1e-6).all():
            return False
    return True


def audit_peak_response(report: dict) -> bool:
    """Verify whole-period peak metrics and release flags from actual traces."""
    expected = {"fixed_upper", "dynamic_upper", "dynamic_release_only", "dynamic_point_only", "dynamic_peak_response", "fixed_peak_response"}
    if set(report["periods"]) != {"validation", "test"}:
        return False
    for period in report["periods"].values():
        if set(period["experiments"]) != expected or not audit_daily_energy(period):
            return False
        for metrics in period["experiments"].values():
            trace = pd.read_csv(ROOT / metrics["timeseries"], parse_dates=["timestamp"])
            top = trace.nlargest(20, "net_load_mw")
            measured = {
                "baseline_peak_mw": trace.net_load_mw.max(),
                "controlled_peak_mw": trace.controlled_net_load_mw.max(),
                "max_reduction_mw": trace.net_load_mw.max() - trace.controlled_net_load_mw.max(),
                "throughput_mwh": (trace.charge_mw + trace.discharge_mw).sum(),
                "top20_mean_reduction_mw": (top.net_load_mw - top.controlled_net_load_mw).mean(),
                "top20_improved_hours": (top.controlled_net_load_mw < top.net_load_mw - 1e-6).sum(),
                "reserve_target_shortfall_hours": (trace.storage_soc_mwh < trace.reserve_target_mwh - 1e-5).sum(),
            }
            if any(abs(value - metrics[name]) > 1e-5 for name, value in measured.items()):
                return False
            if "reserve_release_active" in trace:
                threshold = metrics["release_threshold_mw"]
                triggered = trace.window_max_prediction_mw.ge(threshold) if threshold is not None else pd.Series(False, index=trace.index)
                if not trace.reserve_release_active.eq(triggered).all():
                    return False
                if not np.allclose(trace.loc[triggered, "advisory_floor_mwh"],
                                   metrics["capacity_mwh"] * metrics["hard_min_fraction"], rtol=0, atol=1e-6):
                    return False
    return True


def audit_independent_validation(report: dict) -> bool:
    """Recompute the held-out maximum and costs; a negative finding is valid."""
    protocol_path = ROOT / "outputs/validation/independent_2021_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_path = ROOT / "data/processed/independent_de_2021_hourly.csv"
    if (checksum(protocol_path) != report["protocol_sha256"] or checksum(dataset_path) != report["dataset_sha256"]
            or not all(checksum(ROOT / path) == value for path,value in protocol["frozen_sha256"].items())
            or set(report["experiments"]) != set(protocol["policies"]) or not audit_daily_energy(report)):
        return False
    quality = read_json("outputs/validation/independent_2021_data_quality.json")
    if quality["dataset_sha256"] != checksum(dataset_path):
        return False
    raw = pd.read_csv(dataset_path, parse_dates=["timestamp"]).set_index("timestamp")
    start, end = pd.Timestamp(protocol["period"]["start"]), pd.Timestamp(protocol["period"]["end_exclusive"])
    expected = pd.date_range(start,end,freq="h",inclusive="left")
    from src.balanced_storage import complete_utc_days
    from src.multihorizon import build_direct_horizon_frame
    # Rebuild admissible windows from finite features, without predicting again.
    common = None
    for h in range(1,7):
        frame = build_direct_horizon_frame(raw.reset_index(),h)
        stamps = pd.Index(frame.timestamp - pd.Timedelta(hours=h-1))
        common = stamps if common is None else common.intersection(stamps)
    candidate = raw.reset_index().loc[lambda d: d.timestamp.isin(common)&d.timestamp.isin(expected)]
    complete, _ = complete_utc_days(candidate)
    traces = {}
    for name,item in report["experiments"].items():
        trace = pd.read_csv(ROOT / item["timeseries"],parse_dates=["timestamp"])
        if (not trace.timestamp.reset_index(drop=True).equals(complete.timestamp.reset_index(drop=True))
                or not np.allclose(trace.net_load_mw,raw.loc[trace.timestamp,"net_load_mw"],atol=1e-6,rtol=0)
                or not trace.timestamp.ge(start).all() or not trace.timestamp.lt(end).all()
                or abs(trace.controlled_net_load_mw.max()-item["controlled_peak_mw"]) > 1e-6
                or abs((trace.charge_mw+trace.discharge_mw).sum()-item["throughput_mwh"]) > 1e-5):
            return False
        release = trace.window_max_prediction_mw.ge(item["release_threshold_mw"]) if item["release_threshold_mw"] is not None else pd.Series(False,index=trace.index)
        if not trace.reserve_release_active.eq(release).all():
            return False
        for q,group in trace.groupby(trace.timestamp.dt.quarter):
            saved = report["quarters"][name][str(q)]
            if abs(group.controlled_net_load_mw.max()-saved["controlled_peak_mw"]) > 1e-6 or abs(group.controlled_net_load_mw.quantile(.95)-saved["controlled_p95_mw"]) > 1e-6:
                return False
        traces[name] = trace
    a,b = report["experiments"]["dynamic_upper"],report["experiments"]["dynamic_peak_response"]
    delta = a["controlled_peak_mw"]-b["controlled_peak_mw"]
    primary = report["comparisons"]["dynamic_peak_response_versus_dynamic_upper"]
    if abs(delta-primary["full_period_peak_reduction_mw"]) > 1e-6 or report["primary_observed_success"] != (delta>1e-6):
        return False
    uncertainty = primary["uncertainty"]
    old,new = traces["dynamic_upper"],traces["dynamic_peak_response"]
    robustness = read_json("outputs/validation/independent_2021_robustness.json")
    if robustness["primary_report_sha256"] != checksum(ROOT / "outputs/validation/independent_2021_metrics.json"):
        return False
    mean_delta = (old.groupby(old.timestamp.dt.floor("D")).controlled_net_load_mw.max()
                  -new.groupby(new.timestamp.dt.floor("D")).controlled_net_load_mw.max()).mean()
    if abs(mean_delta-uncertainty["mean_daily_peak_reduction_mw"]["estimate"]) > 1e-6:
        return False
    filtered_old = old.loc[old.timestamp.ge(pd.Timestamp("2021-01-04T00:00Z"))]
    filtered_new = new.loc[new.timestamp.ge(pd.Timestamp("2021-01-04T00:00Z"))]
    if abs(filtered_old.controlled_net_load_mw.max()-filtered_new.controlled_net_load_mw.max()-robustness["probe_excluded_peak_reduction_mw"]) > 1e-6:
        return False
    preds = pd.read_csv(ROOT / "outputs/validation/independent_2021_predictions.csv",parse_dates=["timestamp","issued_timestamp"])
    if not (preds.timestamp - preds.issued_timestamp).eq(pd.to_timedelta(preds.horizon_hours-1,unit="h")).all():
        return False
    frozen_at = pd.Timestamp(protocol["frozen_at_utc"])
    downloaded_at = pd.Timestamp(quality["downloaded_at_utc"])
    old_source = pd.read_csv(ROOT / "data/processed/net_load_hourly.csv",usecols=["timestamp"],parse_dates=["timestamp"])
    return bool(frozen_at < downloaded_at and old_source.timestamp.max() < start and report["old_dataset_overlap_hours"] == 0)


def legacy_v1_main() -> int:
    checks: list[dict[str, object]] = []

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    checks.append(
        {
            "name": "pytest",
            "passed": test.returncode == 0,
            "detail": (test.stdout + test.stderr).strip().splitlines()[-1:] or ["no output"],
        }
    )

    app_test = subprocess.run([sys.executable, "-B", "scripts/check_app.py"], cwd=ROOT, capture_output=True, text=True)
    checks.append({"name": "application-startup-and-date-interaction", "passed": app_test.returncode == 0})

    required = [
        "data/processed/net_load_hourly.csv",
        "outputs/metrics/data-quality-report.json",
        "outputs/models/day3_model_metrics.json",
        "outputs/models/multihorizon_metrics.json",
        "outputs/models/rolling_backtest_metrics.json",
        "outputs/models/reference_baseline_metrics.json",
        "outputs/alerts/alert_evaluation.json",
        "outputs/scenarios/storage_scenario_metrics.json",
        "outputs/scenarios/reserve_ablation_metrics.json",
        "outputs/scenarios/daily_energy_balance_metrics.json",
        "outputs/scenarios/peak_response_metrics.json",
        "outputs/validation/independent_2021_protocol.json",
        "outputs/validation/independent_2021_metrics.json",
        "outputs/validation/independent_2021_data_quality.json",
        "outputs/validation/independent_2021_robustness.json",
        "outputs/report/gridpulse-technical-report-v0.1.pdf",
        "outputs/presentation/gridpulse-defense-v9.pdf",
        "outputs/presentation/gridpulse-defense-v9.pptx",
        ".codex-finalizer/gridpulse-defense-v9.validation.json",
        ".codex-finalizer/gridpulse-anonymous-preflight.json",
    ]
    for relative in required:
        exists = (ROOT / relative).is_file() or (relative.startswith(".codex-finalizer/") and (ROOT.parent / "05_复现与合规/验证回执" / Path(relative).name).is_file())
        checks.append({"name": f"artifact:{relative}", "passed": exists})

    model = read_json("outputs/models/day3_model_metrics.json")
    multihorizon = read_json("outputs/models/multihorizon_metrics.json")
    scenario = read_json("outputs/scenarios/storage_scenario_metrics.json")
    baselines = read_json("outputs/models/reference_baseline_metrics.json")
    supplemental = read_json("outputs/scenarios/reserve_ablation_metrics.json")
    daily = read_json("outputs/scenarios/daily_energy_balance_metrics.json")
    checks.append({"name": "daily-matched-trace-energy-audit", "passed": audit_daily_energy(daily)})
    peak_response = read_json("outputs/scenarios/peak_response_metrics.json")
    checks.append({"name": "peak-response-energy-and-matched-sample-audit", "passed": audit_peak_response(peak_response)})
    independent = read_json("outputs/validation/independent_2021_metrics.json")
    checks.append({"name": "independent-frozen-protocol-and-results-audit", "passed": audit_independent_validation(independent)})
    from scripts.certify_optimality import verify as verify_optimality
    checks.append({"name":"finite-instance-exact-and-primal-dual-certificates", "passed":verify_optimality(ROOT)})
    from scripts.audit_development import audit as audit_development
    checks.append({"name":"controller-operating-interval-and-renewable-development-audit","passed":audit_development()})
    from scripts.audit_later_year import audit as audit_later_year
    checks.append({"name":"locked-2022-replication-and-trajectory-audit", "passed":audit_later_year()})
    clean = read_json("outputs/development/clean_environment_verification.json")
    checks.append({"name":"fresh-isolated-windows-environment","passed":clean["passed"]
        and clean["environment"]["isolated_venv"] and not clean["environment"]["system_site_packages"]
        and clean["requirements_sha256"] == hashlib.sha256((ROOT/"requirements.txt").read_bytes()).hexdigest()})
    rolling = read_json("outputs/models/rolling_backtest_metrics.json")
    receipt = read_json(".codex-finalizer/gridpulse-defense-v9.validation.json")
    anonymous = read_json(".codex-finalizer/gridpulse-anonymous-preflight.json")
    model_mae = model["models"]["hist_gradient_boosting"]["test"]["mae_mw"]
    checks.append({"name": "shared-timestamp-baselines", "passed": baselines.get("test_rows", 0) > 0 and abs(baselines["metrics"]["hist_gradient_boosting"]["mae_mw"] - model_mae) < 1e-6})
    checks.append({"name": "supplemental-soc-accounting", "passed": all(0 <= item["final_soc_mwh"] <= item["capacity_mwh"] and item["reserve_target_shortfall_hours"] >= 0 for item in supplemental["experiments"].values())})
    checks.append({"name": "rolling-identical-sample-count", "passed": all(fold["test_rows"] == fold["baseline_rows"] == 720 for fold in rolling["folds"])})
    checks.append(
        {
            "name": "model-metric-boundary",
            "passed": model.get("forecast_design") == "one_step_ahead_causal",
            "detail": model.get("forecast_design"),
        }
    )
    checks.append(
        {
            "name": "direct-multihorizon-boundary",
            "passed": multihorizon.get("forecast_design") == "direct_multi_horizon_causal"
            and multihorizon.get("horizons") == [1, 2, 3, 4, 5, 6],
            "detail": {
                "forecast_design": multihorizon.get("forecast_design"),
                "horizons": multihorizon.get("horizons"),
            },
        }
    )
    interval = model["models"]["hist_gradient_boosting"].get("prediction_interval", {})
    checks.append(
        {
            "name": "prediction-interval-calibration",
            "passed": 0 < interval.get("radius_mw", 0) and 0 <= interval.get("test", {}).get("coverage_pct", -1) <= 100,
            "detail": {
                "radius_mw": interval.get("radius_mw"),
                "test_coverage_pct": interval.get("test", {}).get("coverage_pct"),
            },
        }
    )
    checks.append(
        {
            "name": "storage-sensitivity",
            "passed": all(
                scenario["direct_fixed_reserve_baseline"]["sensitivity"][key]["reserve_policy"] == "fixed_30%"
                and scenario["risk_aware_optimization"]["sensitivity"][key]["solver_optimal_rate_pct"] == 100.0
                and scenario["risk_aware_optimization"]["sensitivity"][key]["min_soc_mwh"] >= 0
                for key in ("storage_10000mwh", "storage_20000mwh", "storage_30000mwh")
            ),
            "detail": {
                key: {
                    "fixed_reserve_policy": scenario["direct_fixed_reserve_baseline"]["sensitivity"][key]["reserve_policy"],
                    "fair_p95_delta_mw": scenario["comparison"]["fair_same_information_fixed_reserve_vs_risk_aware"][key]["optimized_minus_fixed_controlled_p95_mw"],
                    "risk_aware_solver_optimal_rate_pct": scenario["risk_aware_optimization"]["sensitivity"][key]["solver_optimal_rate_pct"],
                }
                for key in ("storage_10000mwh", "storage_20000mwh", "storage_30000mwh")
            },
        }
    )
    checks.append(
        {
            "name": "ppt-validation",
            "passed": receipt.get("packageIntegrity", {}).get("status") == "pass"
            and receipt.get("firstPartyImport", {}).get("passed") is True,
            "detail": {
                "package": receipt.get("packageIntegrity", {}).get("status"),
                "import_passed": receipt.get("firstPartyImport", {}).get("passed"),
            },
        }
    )
    checks.append(
        {
            "name": "anonymous-material-scan",
            "passed": anonymous.get("passed") is True,
            "detail": anonymous.get("manual_boundary"),
        }
    )

    pending = {
        "technical_report_pdf": not any((ROOT / "outputs").rglob("*.pdf")),
        "defense_ppt_pdf": not (ROOT / "outputs/presentation/gridpulse-defense-v9.pdf").is_file(),
        "demo_video": not any((ROOT / "outputs").glob("*.mp4")),
        "submission_upload": True,
        "anonymous_final_review": True,
    }
    report = {
        "schemaVersion": "gridpulse-final-preflight.v1",
        "passed": all(bool(item["passed"]) for item in checks),
        "core_metric": {"hist_gradient_boosting_test_mae_mw": model_mae},
        "checks": checks,
        "pending_external_or_manual_items": pending,
        "interpretation": "Core repository evidence is ready when passed=true; pending items require final competition packaging or human review.",
    }
    output = ROOT / ".codex-finalizer/gridpulse-final-preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def main() -> int:
    """Run current-source tests and V8 evidence/application checks."""
    import os
    from scripts.build_submission_package import write_manifest
    checks = []
    commands = (
        ("current-tests", ["-m", "pytest", "-p", "no:cacheprovider",
                           "--junitxml=outputs/review/pytest.xml"]),
        ("v8-independent-audit", ["scripts/verify_envelope_results.py"]),
        ("v8-application", ["scripts/check_app.py", "--output", "outputs/review/app_verification.json"]),
    )
    for name, args in commands:
        result = subprocess.run(
            [sys.executable, "-B", *args], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=240,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
        )
        checks.append({"name": name, "passed": result.returncode == 0,
                       "detail": (result.stdout if result.returncode == 0 else result.stderr)[-3500:]})
    pdf = ROOT.parent / "01_技术报告/GridPulse-技术报告.pdf"
    copies = sorted((ROOT.parent / "06_平台上传文件").glob("*-技术报告.pdf"))
    checks.append({
        "name": "current-report-copies",
        "passed": pdf.is_file() and len(copies) == 1
        and hashlib.sha256(pdf.read_bytes()).digest() == hashlib.sha256(copies[0].read_bytes()).digest(),
    })
    report = {"version": "fusion_v8", "passed": all(row["passed"] for row in checks),
              "checks": checks}
    output = ROOT / "outputs/review/final_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(ROOT.parent)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
