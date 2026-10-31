from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.baseline import chronological_split, interval_metrics, load_evaluable_frame, regression_metrics
from src.features import BASE_FEATURES, LAG_ONLY_FEATURES, TARGET, model_frame

RANDOM_STATE = 20260929
CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
]
WITHOUT_ROLLING_FEATURES = [feature for feature in BASE_FEATURES if "rolling" not in feature]


def make_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_features=0.8,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def make_gradient_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def frozen_split_for_features(raw: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Reuse Day 2 timestamp boundaries after feature rows become available."""
    _, _, _, day2_split = chronological_split(load_evaluable_frame_from_raw(raw))
    validation_start = pd.Timestamp(day2_split.validation_start)
    test_start = pd.Timestamp(day2_split.test_start)
    train = features.loc[features["timestamp"] < validation_start].copy()
    validation = features.loc[(features["timestamp"] >= validation_start) & (features["timestamp"] < test_start)].copy()
    test = features.loc[features["timestamp"] >= test_start].copy()
    split = {
        "day2_frozen_validation_start": str(validation_start),
        "day2_frozen_test_start": str(test_start),
        "train_rows_after_feature_filtering": len(train),
        "validation_rows_after_feature_filtering": len(validation),
        "test_rows_after_feature_filtering": len(test),
        "test_end": str(test["timestamp"].iloc[-1]),
    }
    return train, validation, test, split


def load_evaluable_frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the Day 2 seasonal-pair eligibility rule without re-reading disk."""
    frame = raw.copy().sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    lookup = frame[["timestamp", TARGET]].rename(columns={"timestamp": "lag_timestamp", TARGET: "lag_target"})
    frame["lag_timestamp"] = frame["timestamp"] - pd.Timedelta(hours=24)
    frame = frame.merge(lookup, on="lag_timestamp", how="left")
    return frame.dropna(subset=[TARGET, "lag_target"]).reset_index(drop=True)


def train_and_evaluate(input_path: Path, output_dir: Path) -> dict:
    raw = pd.read_csv(input_path, parse_dates=["timestamp"])
    data = model_frame(raw)
    train, validation, test, split = frozen_split_for_features(raw, data)
    result = {
        "target": TARGET,
        "forecast_design": "one_step_ahead_causal",
        "feature_columns": BASE_FEATURES,
        "split": split,
        "models": {},
        "ablation": {},
        "protocol": [
            "All observed target, solar, and wind features are shifted before calculation.",
            "Models predict the current net load using data available no later than the preceding hour.",
            "Chronological split uses the same 70/10/20 protocol as Day 2 after model-feature availability filtering.",
            "Validation is reserved for future model and threshold decisions; final test is reported without tuning against it.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = train[BASE_FEATURES], train[TARGET]
    x_validation, y_validation = validation[BASE_FEATURES], validation[TARGET]
    x_test, y_test = test[BASE_FEATURES], test[TARGET]

    for name, model in make_models().items():
        model.fit(x_train, y_train)
        validation_prediction = model.predict(x_validation)
        test_prediction = model.predict(x_test)
        calibration_abs_error = np.abs(y_validation.to_numpy() - validation_prediction)
        interval_radius = float(np.quantile(calibration_abs_error, 0.90))
        test_lower = test_prediction - interval_radius
        test_upper = test_prediction + interval_radius
        importance = permutation_importance(
            model,
            x_validation,
            y_validation,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importance_frame = pd.DataFrame(
            {"feature": BASE_FEATURES, "importance_mae_mw": importance.importances_mean}
        ).sort_values("importance_mae_mw", ascending=False)
        predictions = test[["timestamp", TARGET]].copy()
        predictions[f"{name}_prediction_mw"] = test_prediction
        predictions[f"{name}_lower_mw"] = test_lower
        predictions[f"{name}_upper_mw"] = test_upper
        predictions["residual_mw"] = y_test.to_numpy() - test_prediction
        predictions.to_csv(output_dir / f"{name}_test_predictions.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        importance_frame.to_csv(output_dir / f"{name}_validation_permutation_importance.csv", index=False)
        joblib.dump(
            {
                "model": model,
                "features": BASE_FEATURES,
                "split": split,
                "prediction_interval_radius_mw": interval_radius,
                "prediction_interval_calibration_quantile": 0.90,
            },
            output_dir / f"{name}.joblib",
        )
        result["models"][name] = {
            "validation": regression_metrics(y_validation, pd.Series(validation_prediction)),
            "test": regression_metrics(y_test, pd.Series(test_prediction)),
            "prediction_interval": {
                "calibration_quantile": 0.90,
                "radius_mw": interval_radius,
                "validation": interval_metrics(
                    y_validation,
                    pd.Series(validation_prediction - interval_radius),
                    pd.Series(validation_prediction + interval_radius),
                ),
                "test": interval_metrics(y_test, pd.Series(test_lower), pd.Series(test_upper)),
                "method": "symmetric absolute-residual calibration on validation only",
            },
            "artifact": f"outputs/models/{name}.joblib",
        }

    ablations = {
        "hist_gradient_boosting_without_renewable_lags": LAG_ONLY_FEATURES,
        "hist_gradient_boosting_without_rolling_statistics": WITHOUT_ROLLING_FEATURES,
        "hist_gradient_boosting_calendar_only": CALENDAR_FEATURES,
    }
    for name, feature_columns in ablations.items():
        ablation_model = make_gradient_boosting().fit(train[feature_columns], y_train)
        validation_prediction = ablation_model.predict(validation[feature_columns])
        test_prediction = ablation_model.predict(test[feature_columns])
        result["ablation"][name] = {
            "feature_columns": feature_columns,
            "validation": regression_metrics(y_validation, pd.Series(validation_prediction)),
            "test": regression_metrics(y_test, pd.Series(test_prediction)),
        }
    (output_dir / "day3_model_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/net_load_hourly.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
