from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import BASE_FEATURES, model_frame

VALIDATION_START = pd.Timestamp("2019-01-12 20:00:00+00:00")
TEST_START = pd.Timestamp("2019-08-09 21:00:00+00:00")
SUSTAINED_HOURS = 3
RANDOM_STATE = 20260929


def predict_periods(data_path: Path, artifact_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(data_path, parse_dates=["timestamp"])
    data = model_frame(raw)
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    data["predicted_net_load_mw"] = model.predict(data[BASE_FEATURES])
    radius = float(artifact.get("prediction_interval_radius_mw", 0.0))
    data["prediction_lower_mw"] = data["predicted_net_load_mw"] - radius
    data["prediction_upper_mw"] = data["predicted_net_load_mw"] + radius
    data["prediction_interval_radius_mw"] = radius
    data["residual_mw"] = data["net_load_mw"] - data["predicted_net_load_mw"]
    validation = data.loc[(data["timestamp"] >= VALIDATION_START) & (data["timestamp"] < TEST_START)].copy()
    test = data.loc[data["timestamp"] >= TEST_START].copy()
    return validation, test


def calibrate_thresholds(validation: pd.DataFrame) -> dict[str, float | int]:
    validation = validation.sort_values("timestamp")
    contiguous = pd.to_datetime(validation["timestamp"], utc=True).diff().eq(pd.Timedelta(hours=1))
    solar_change = validation["solar_generation_mw"].diff().where(contiguous)
    return {
        "peak_prediction_threshold_mw": float(validation["predicted_net_load_mw"].quantile(0.95)),
        "solar_drop_threshold_mw_per_hour": float(solar_change.quantile(0.05)),
        "absolute_residual_threshold_mw": float(validation["residual_mw"].abs().quantile(0.975)),
        "sustained_hours": SUSTAINED_HOURS,
        "calibration_rows": int(len(validation)),
        "calibration_period_start": str(validation["timestamp"].min()),
        "calibration_period_end": str(validation["timestamp"].max()),
    }


def _sustained_mask(residual: pd.Series, threshold: float, hours: int, timestamps: pd.Series | None = None) -> pd.Series:
    """Confirm only after enough same-sign, contiguous observed hours."""
    exceedance = residual.abs() > threshold
    sign = np.sign(residual).fillna(0)
    breaks = exceedance.ne(exceedance.shift()) | sign.ne(sign.shift())
    if timestamps is not None:
        breaks |= pd.to_datetime(timestamps, utc=True).diff().ne(pd.Timedelta(hours=1))
    groups = breaks.cumsum()
    run_length = exceedance.astype(int).groupby(groups).cumsum()
    return exceedance & (run_length >= hours)


def apply_alerts(frame: pd.DataFrame, thresholds: dict[str, float | int]) -> pd.DataFrame:
    result = frame.copy().sort_values("timestamp").reset_index(drop=True)
    if "prediction_upper_mw" not in result:
        result["prediction_upper_mw"] = result["predicted_net_load_mw"]
    contiguous = pd.to_datetime(result["timestamp"], utc=True).diff().eq(pd.Timedelta(hours=1))
    result["solar_change_mw_per_hour"] = result["solar_generation_mw"].diff().where(contiguous)
    result["absolute_residual_mw"] = result["residual_mw"].abs()
    result["peak_load_risk"] = result["predicted_net_load_mw"] >= float(thresholds["peak_prediction_threshold_mw"])
    result["interval_peak_watch"] = result["prediction_upper_mw"] >= float(thresholds["peak_prediction_threshold_mw"])
    result["solar_drop_risk"] = result["solar_change_mw_per_hour"] <= float(thresholds["solar_drop_threshold_mw_per_hour"])
    result["sustained_deviation_risk"] = _sustained_mask(
        result["residual_mw"],
        float(thresholds["absolute_residual_threshold_mw"]),
        int(thresholds["sustained_hours"]),
        result["timestamp"],
    )
    result["risk_score"] = (
        result["peak_load_risk"].astype(float)
        + result["solar_drop_risk"].astype(float)
        + result["sustained_deviation_risk"].astype(float)
        + 0.5 * (result["interval_peak_watch"] & ~result["peak_load_risk"]).astype(float)
    )
    result["review_priority"] = np.select(
        [result["risk_score"] >= 2, result["risk_score"] > 0],
        ["high", "review"],
        default="normal",
    )
    risk_names = []
    evidence = []
    for row in result.itertuples(index=False):
        active = []
        details = []
        if row.peak_load_risk:
            active.append("peak_load")
            details.append(f"forecast={row.predicted_net_load_mw:.1f} MW >= peak threshold={thresholds['peak_prediction_threshold_mw']:.1f} MW")
        elif row.interval_peak_watch:
            active.append("interval_peak_watch")
            details.append(f"prediction upper bound={row.prediction_upper_mw:.1f} MW >= peak threshold={thresholds['peak_prediction_threshold_mw']:.1f} MW")
        if row.solar_drop_risk:
            active.append("solar_drop")
            details.append(f"solar change={row.solar_change_mw_per_hour:.1f} MW/h <= ramp threshold={thresholds['solar_drop_threshold_mw_per_hour']:.1f} MW/h")
        if row.sustained_deviation_risk:
            active.append("sustained_deviation")
            details.append(f"absolute residual={row.absolute_residual_mw:.1f} MW > residual threshold={thresholds['absolute_residual_threshold_mw']:.1f} MW for at least {thresholds['sustained_hours']} same-sign consecutive hours")
        risk_names.append("|".join(active))
        evidence.append("; ".join(details))
    result["risk_types"] = risk_names
    result["risk_evidence"] = evidence
    result["has_candidate_risk"] = result[["peak_load_risk", "solar_drop_risk", "sustained_deviation_risk"]].any(axis=1)
    result["has_review_signal"] = result["has_candidate_risk"] | result["interval_peak_watch"]
    return result


def _candidate_positions(length: int, count: int, event_length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    candidates = np.arange(2, length - event_length - 2)
    selected: list[int] = []
    for index in rng.permutation(candidates):
        if all(abs(int(index) - prior) >= event_length + 3 for prior in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return sorted(selected)


def synthetic_stress_test(test: pd.DataFrame, thresholds: dict[str, float | int]) -> dict:
    """Inject deterministic scenarios to test alert mechanics, not real fault detection."""
    test = test.reset_index(drop=True)
    event_length = int(thresholds["sustained_hours"])
    positions = _candidate_positions(len(test), 25, event_length, RANDOM_STATE)
    solar_drop = abs(float(thresholds["solar_drop_threshold_mw_per_hour"])) * 1.5
    solar_candidates = [
        index
        for index in range(1, len(test) - 1)
        if float(test.loc[index - 1, "solar_generation_mw"]) >= solar_drop + 100
    ]
    solar_positions = _candidate_positions_from_candidates(solar_candidates, 25, event_length, RANDOM_STATE + 1)
    scenario = test.copy().reset_index(drop=True)
    scenario["injected_peak"] = False
    scenario["injected_solar_drop"] = False
    scenario["injected_sustained_deviation"] = False
    peak_level = float(thresholds["peak_prediction_threshold_mw"]) * 1.08
    residual_shift = float(thresholds["absolute_residual_threshold_mw"]) * 1.5
    for start in positions:
        interval = list(range(start, start + event_length))
        scenario.loc[interval, "predicted_net_load_mw"] = peak_level
        scenario.loc[interval, "injected_peak"] = True
        scenario.loc[interval, "residual_mw"] = residual_shift
        scenario.loc[interval, "net_load_mw"] = peak_level + residual_shift
        scenario.loc[interval, "injected_sustained_deviation"] = True
    for start in solar_positions:
        scenario.loc[start, "solar_generation_mw"] = scenario.loc[start - 1, "solar_generation_mw"] - solar_drop
        scenario.loc[start, "injected_solar_drop"] = True
    flagged = apply_alerts(scenario, thresholds)
    checks = {
        "peak_load": (flagged["peak_load_risk"] & flagged["injected_peak"]).sum(),
        "solar_drop": (flagged["solar_drop_risk"] & flagged["injected_solar_drop"]).sum(),
        "sustained_deviation": (flagged["sustained_deviation_risk"] & flagged["injected_sustained_deviation"]).sum(),
    }
    totals = {
        "peak_load": int(flagged["injected_peak"].sum()),
        "solar_drop": int(flagged["injected_solar_drop"].sum()),
        "sustained_deviation": int(flagged["injected_sustained_deviation"].sum()),
    }
    return {
        "scenario": "Synthetic deterministic stress test; injected events are not historical fault labels.",
        "random_seed": RANDOM_STATE,
        "event_count": len(positions),
        "solar_drop_event_count": len(solar_positions),
        "event_length_hours": event_length,
        "injection_magnitudes": {
            "peak_prediction_level_mw": peak_level,
            "residual_shift_mw": residual_shift,
            "solar_drop_mw": solar_drop,
        },
        "injected_point_detection": {
            name: {"detected": int(checks[name]), "injected": totals[name], "rate": float(checks[name] / totals[name])}
            for name in totals
        },
        "sustained_event_detection": {
            "detected_events": int(sum(flagged.loc[start:start + event_length - 1, "sustained_deviation_risk"].any() for start in positions)),
            "injected_events": len(positions),
            "confirmation_delay_hours": event_length - 1,
            "interpretation": "Point rate is lower because earlier hours are not retroactively flagged; event detection is evaluated separately.",
        },
}


def _candidate_positions_from_candidates(candidates: list[int], count: int, event_length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for index in rng.permutation(candidates):
        if all(abs(int(index) - prior) >= event_length + 3 for prior in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return sorted(selected)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/hist_gradient_boosting.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alerts"))
    args = parser.parse_args()
    validation, test = predict_periods(args.data, args.model)
    thresholds = calibrate_thresholds(validation)
    alerts = apply_alerts(test, thresholds)
    stress = synthetic_stress_test(test, thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alerts.loc[alerts["has_candidate_risk"]].to_csv(
        args.output_dir / "test_candidate_risks.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    alerts.loc[alerts["has_review_signal"]].to_csv(
        args.output_dir / "test_review_queue.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    result = {
        "thresholds": thresholds,
        "test_rows": len(test),
        "test_candidate_risk_rows": int(alerts["has_candidate_risk"].sum()),
        "test_review_queue_rows": int(alerts["has_review_signal"].sum()),
        "interval_only_review_rows": int((alerts["interval_peak_watch"] & ~alerts["has_candidate_risk"]).sum()),
        "stress_test": stress,
    }
    (args.output_dir / "alert_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
