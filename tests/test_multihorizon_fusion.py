import numpy as np
import pandas as pd
import pytest
import torch

from src.multihorizon_fusion import (
    LABELS, SixHorizonMLP, arrays, build_frame, restrict_split, split_original,
    calibrate_six, planning_panel, shared_days,
)
from src.physics_forecast import TARGETS, VALIDATION_START, TEST_START


def raw(start="2019-01-01", periods=1000):
    stamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": stamps, "total_load_mw": np.arange(periods) + 10000.,
        "solar_generation_mw": 500., "wind_generation_mw": 200.,
    })
    frame["net_load_mw"] = frame.total_load_mw - frame.solar_generation_mw - frame.wind_generation_mw
    return frame


def predictions(start="2019-02-01", days=2):
    times = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    out = pd.DataFrame({"timestamp": times, "issued_timestamp": times - pd.Timedelta(hours=1)})
    for h in range(1, 7):
        values = np.where((times + pd.Timedelta(hours=h - 1)).hour < 12, 80., 200.)
        out[f"actual_h{h}_mw"] = values
        out[f"prediction_h{h}_mw"] = values + h
    return out


def test_features_and_model_do_not_read_current_or_future_truth():
    source = raw()
    before, features = build_frame(source)
    source.loc[300:, ["total_load_mw", "net_load_mw"]] += 1e6
    after, new_features = build_frame(source)
    time = source.timestamp.iloc[300]
    a = before.loc[before.timestamp.eq(time)]
    b = after.loc[after.timestamp.eq(time)]
    assert features == new_features and len(features) == 68
    assert np.allclose(a[features], b[features])
    model = SixHorizonMLP(len(features)).eval()
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -.01, .01)
    ax, al, _ = arrays(a, features, 0., 1.)
    bx, bl, _ = arrays(b, features, 0., 1.)
    with torch.no_grad():
        assert torch.equal(model(torch.tensor(ax).float(), torch.tensor(al).float()),
                           model(torch.tensor(bx).float(), torch.tensor(bl).float()))
    assert not np.allclose(a[LABELS], b[LABELS])


def test_labels_align_by_elapsed_hour_not_cleaned_row_position():
    source = raw()
    source.loc[302, ["total_load_mw", "net_load_mw"]] = np.nan
    frame, _ = build_frame(source)
    row = frame.loc[frame.timestamp.eq(source.timestamp.iloc[300])].iloc[0]
    assert np.isnan(row.target_h3_net_load_mw)
    assert row.target_h4_net_load_mw == source.net_load_mw.iloc[303]
    assert row.timestamp not in set(restrict_split(frame).timestamp)


def test_split_embargo_covers_last_target_not_only_first_target():
    frame, _ = build_frame(raw(periods=6000))
    splits = split_original(frame)
    assert splits["train"].last_target_timestamp.max() < VALIDATION_START
    assert splits["validation"].timestamp.min() >= VALIDATION_START
    assert splits["validation"].last_target_timestamp.max() < TEST_START
    assert splits["test"].timestamp.min() >= TEST_START
    assert all(np.isfinite(f[LABELS]).all().all() for f in splits.values())


def test_calibration_bounds_and_matched_first_horizon():
    frame = predictions()
    cal = calibrate_six({"a": frame})
    assert cal["methods"]["a"]["radii"]["6"]["0"] == 6.
    for start in ("2018-01-01", "2019-08-09", "2021-01-01"):
        with pytest.raises(ValueError, match="validation"):
            calibrate_six({"a": predictions(start)})
    p1 = planning_panel(frame, cal["methods"]["a"], 1)
    p6 = planning_panel(frame, cal["methods"]["a"], 6)
    assert p1.equals(p6[p1.columns])
    changed = frame.copy()
    changed.loc[0, "actual_h3_mw"] += 1.
    with pytest.raises(ValueError, match="actual"):
        shared_days({"a": frame, "b": changed})


def test_six_hour_actions_never_read_any_evaluation_truth():
    from src.forecast_dispatch import dispatch
    from src.optimality import StorageModel
    cal = calibrate_six({"a": predictions()})
    original = predictions("2021-01-01")
    changed = original.copy()
    for h in range(1, 7):
        changed[f"actual_h{h}_mw"] += np.arange(len(changed)) * 100.
    model = StorageModel(capacity_mwh=200., initial_soc_mwh=100., maximum_soc_mwh=100.,
                         minimum_soc_mwh=30., charge_power_mw=20., discharge_power_mw=20.)
    gates = {"peak_threshold_mw": 150., "charge_threshold_mw": 90., "release_threshold_mw": 180.}
    a, ma = dispatch(planning_panel(original, cal["methods"]["a"], 6),
                     cal["methods"]["a"], gates, model)
    b, mb = dispatch(planning_panel(changed, cal["methods"]["a"], 6),
                     cal["methods"]["a"], gates, model)
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert np.allclose(a.discharge_mw, b.discharge_mw)
    assert ma["terminal_requirement_passed"] and mb["terminal_requirement_passed"]
