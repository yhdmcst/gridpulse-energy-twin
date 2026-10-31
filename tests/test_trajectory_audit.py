import numpy as np
import pandas as pd
import pytest

from scripts.audit_development import compare
from src.optimality import StorageModel
from src.trajectory_audit import audited_summary


def idle_trace():
    return pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC"),
        "net_load_mw": 100., "controlled_net_load_mw": 100., "charge_mw": 0., "discharge_mw": 0.,
        "soc_before_mwh": 15000., "storage_soc_mwh": 15000.})


@pytest.mark.parametrize("column", ["charge_mw", "net_load_mw", "storage_soc_mwh"])
def test_nonfinite_physical_values_cannot_pass_audit(column):
    trace = idle_trace()
    trace.loc[5, column] = np.nan
    with pytest.raises(ValueError, match="finite"):
        audited_summary(trace, StorageModel())


def test_timestamp_gaps_and_duplicates_are_rejected():
    trace = idle_trace()
    with pytest.raises(ValueError, match="every UTC hour"):
        audited_summary(trace.drop(index=7), StorageModel())
    trace.loc[7, "timestamp"] = trace.loc[6, "timestamp"]
    with pytest.raises(ValueError, match="Unique"):
        audited_summary(trace, StorageModel())


def test_forged_nan_summary_is_not_accepted():
    trace = idle_trace()
    summary = audited_summary(trace, StorageModel())
    assert compare(trace, summary, StorageModel())
    summary["controlled_peak_mw"] = float("nan")
    assert not compare(trace, summary, StorageModel())
