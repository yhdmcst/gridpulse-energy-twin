import numpy as np
import pandas as pd
import pytest
import torch

from src.physics_forecast import TARGETS, causal_frame, project_accounting, accounting_loss


def raw():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2018-01-01", periods=240, freq="h", tz="UTC"),
        "total_load_mw": np.arange(240) + 10000.,
        "solar_generation_mw": 500., "wind_generation_mw": 200.,
    })
    frame["net_load_mw"] = frame.total_load_mw - frame.solar_generation_mw - frame.wind_generation_mw
    return frame


def test_features_never_read_target_or_future_observations():
    before = raw()
    a, features = causal_frame(before)
    after = before.copy()
    after.loc[200:, ["total_load_mw", "net_load_mw"]] += 1000.
    b, _ = causal_frame(after)
    target = before.timestamp.iloc[200]
    assert np.allclose(a.loc[a.timestamp.eq(target), features], b.loc[b.timestamp.eq(target), features])
    assert a.loc[a.timestamp.eq(target), "total_load_mw_lag1"].iloc[0] == before.total_load_mw.iloc[199]


def test_projection_enforces_relation_but_allows_negative_net_load():
    prediction = np.array([[100., 200., 10., -110.], [-5., 1., 2., 4.]])
    out = project_accounting(prediction)
    assert (out[:, :3] >= 0).all()
    assert np.allclose(out[:, 3], out[:, 0] - out[:, 1] - out[:, 2])
    assert out[0, 3] < 0
    assert accounting_loss(torch.tensor(out)).item() == pytest.approx(0.)
    assert np.allclose(project_accounting(out), out)


def test_missing_and_unordered_data_do_not_become_clean_training_rows():
    data = raw()
    data.loc[190, "solar_generation_mw"] = np.nan
    data.loc[190, "net_load_mw"] = np.nan
    frame, _ = causal_frame(data)
    assert data.timestamp.iloc[191] not in set(frame.timestamp)
    with pytest.raises(ValueError):
        causal_frame(raw().iloc[::-1])
