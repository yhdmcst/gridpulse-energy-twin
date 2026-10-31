import json

import pytest

from src.artifacts import ArtifactError, inspect_runtime, validate_frame


def test_corrupt_or_wrong_schema_result_gives_filename(tmp_path):
    path = tmp_path / "controller_metrics.json"
    for payload in ("{", "{}", '{"periods":NaN}', '{"periods":[]}'):
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ArtifactError, match="controller_metrics.json"):
            inspect_runtime(tmp_path, [path.name])


def test_missing_referenced_trace_and_path_escape_are_rejected(tmp_path):
    path = tmp_path / "controller_metrics.json"
    path.write_text(json.dumps({"periods": {"timeseries": "missing.csv"}}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="missing.csv"):
        inspect_runtime(tmp_path, [path.name])
    path.write_text(json.dumps({"periods": {"timeseries": "../outside.csv"}}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="之外"):
        inspect_runtime(tmp_path, [path.name])


def test_bad_csv_header_and_changed_file_cache_key(tmp_path):
    path = tmp_path / "net_load_hourly.csv"
    path.write_text("timestamp,wrong\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="net_load_hourly.csv"):
        inspect_runtime(tmp_path, [path.name])
    path.write_text("timestamp,total_load_mw,solar_generation_mw,wind_generation_mw,net_load_mw\n", encoding="utf-8")
    before = inspect_runtime(tmp_path, [path.name])
    path.write_text(path.read_text() + "2022-01-01,10,1,2,7\n", encoding="utf-8")
    assert before != inspect_runtime(tmp_path, [path.name])


def test_parsed_csv_with_invalid_time_or_value_is_rejected():
    import pandas as pd
    frame = pd.DataFrame({"feature": ["lag"], "importance_mae_mw": [float("inf")]})
    with pytest.raises(ArtifactError, match="非有限值"):
        validate_frame(frame, "hist_gradient_boosting_validation_permutation_importance.csv")
    frame = pd.DataFrame({"timestamp": ["2022-01-01", "2022-01-01"], "total_load_mw": 10.,
                          "solar_generation_mw": 1., "wind_generation_mw": 2., "net_load_mw": 7.})
    with pytest.raises(ArtifactError, match="唯一"):
        validate_frame(frame, "net_load_hourly.csv")


def test_dashboard_displays_read_error_without_traceback(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    def broken(*args):
        raise ValueError("metrics.json：无法读取有效结果")
    monkeypatch.setattr("src.fusion_evidence.load_evidence", broken)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py")).run(timeout=45)
    assert not app.exception
    assert any("metrics.json" in item.value for item in app.error)
