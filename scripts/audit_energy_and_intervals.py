"""Post-hoc renewable alignment/ramp and conditional interval diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_controller_development import read,write,OUT
from src.interval_audit import rolling_radius


def energy_metrics(trace,data,threshold):
    trace.timestamp = pd.to_datetime(trace.timestamp,utc=True)
    data.timestamp = pd.to_datetime(data.timestamp,utc=True)
    aligned = trace.merge(data[["timestamp","total_load_mw","solar_generation_mw","wind_generation_mw"]],on="timestamp",validate="one_to_one")
    if len(aligned) != len(trace):
        raise ValueError("Energy attribution sample mismatch.")
    share = (aligned.solar_generation_mw+aligned.wind_generation_mw)/aligned.total_load_mw
    high = share >= threshold
    contiguous = aligned.timestamp.diff().eq(pd.Timedelta(hours=1))
    raw_ramp = aligned.net_load_mw.diff().where(contiguous).dropna()
    new_ramp = aligned.controlled_net_load_mw.diff().where(contiguous).dropna()
    charged = aligned.charge_mw.sum()
    return {"rows":len(aligned),"high_renewable_share_hours":int(high.sum()),
        "charge_in_high_renewable_hours_mwh":float(aligned.loc[high,"charge_mw"].sum()),
        "charge_in_high_renewable_hours_pct":float(100*aligned.loc[high,"charge_mw"].sum()/charged) if charged else 0.,
        "baseline_abs_ramp_p95_mw_per_hour":float(raw_ramp.abs().quantile(.95)),
        "controlled_abs_ramp_p95_mw_per_hour":float(new_ramp.abs().quantile(.95)),
        "baseline_max_up_ramp_mw_per_hour":float(raw_ramp.max()),
        "controlled_max_up_ramp_mw_per_hour":float(new_ramp.max()),
        "baseline_max_down_ramp_mw_per_hour":float(raw_ramp.min()),
        "controlled_max_down_ramp_mw_per_hour":float(new_ramp.min())}


def main():
    development = read("outputs/development/controller_metrics.json")
    original = pd.read_csv(ROOT/"data/processed/net_load_hourly.csv",parse_dates=["timestamp"])
    original.timestamp = pd.to_datetime(original.timestamp,utc=True)
    valid = original.loc[original.timestamp.ge(pd.Timestamp("2019-01-13T00:00Z"))&original.timestamp.lt(pd.Timestamp("2019-08-09T00:00Z"))]
    share_threshold = float(((valid.solar_generation_mw+valid.wind_generation_mw)/valid.total_load_mw).quantile(.9))
    later = pd.read_csv(ROOT/"data/processed/independent_de_2021_hourly.csv",parse_dates=["timestamp"])
    energy = {"status":"Post-hoc descriptive renewable alignment and ramp evaluation; not avoided curtailment or renewable-sourced energy attribution.",
        "high_renewable_share_threshold":share_threshold,"calibration_period":"original complete validation days q90 renewable/total load",
        "periods":{}}
    for period,item in development["periods"].items():
        raw = later if period == "known_2021" else original
        energy["periods"][period] = {name:energy_metrics(pd.read_csv(ROOT/exp["timeseries"]),raw,share_threshold)
            for name,exp in item["experiments"].items()}
    write(OUT/"renewable_alignment_metrics.json",energy)
    pred = pd.read_csv(ROOT/"outputs/validation/independent_2021_predictions.csv",parse_dates=["timestamp","issued_timestamp"])
    pred.timestamp = pd.to_datetime(pred.timestamp,utc=True)
    pred.issued_timestamp = pd.to_datetime(pred.issued_timestamp,utc=True)
    calibration = read("outputs/scenarios/reserve_ablation_metrics.json")["hourly_interval_calibration"]
    high_load = read("outputs/scenarios/storage_scenario_metrics.json")["calibration"]["peak_threshold_mw"]
    bands = []
    for h,g in pred.groupby("horizon_hours"):
        g = g.sort_values("timestamp").copy()
        fixed = g.timestamp.dt.hour.map({int(k):v for k,v in calibration[str(h)].items()}).to_numpy()
        residual = (g.net_load_mw-g.prediction_mw).to_numpy()
        adaptive,count = rolling_radius(g.timestamp,residual,g.issued_timestamp,fixed)
        g["fixed_radius_mw"],g["rolling_radius_mw"],g["rolling_history_rows"] = fixed,adaptive,count
        g["fixed_covered"],g["rolling_covered"] = np.abs(residual)<=fixed,np.abs(residual)<=adaptive
        bands.append(g.loc[g.timestamp.lt(pd.Timestamp("2022-01-01T00:00Z"))])
    joined = pd.concat(bands,ignore_index=True)
    joined.to_csv(OUT/"interval_diagnostics.csv",index=False)
    rows = []
    for (h,quarter),g in joined.groupby(["horizon_hours",joined.timestamp.dt.quarter]):
        for scope,mask in (("all",np.ones(len(g),bool)),("high_load",g.net_load_mw.ge(high_load))):
            sub = g.loc[mask]
            if not len(sub):
                continue
            rows.append({"horizon_hours":int(h),"quarter":int(quarter),"scope":scope,"rows":len(sub),
                "fixed_coverage_pct":float(sub.fixed_covered.mean()*100),"rolling_coverage_pct":float(sub.rolling_covered.mean()*100),
                "fixed_mean_width_mw":float(2*sub.fixed_radius_mw.mean()),"rolling_mean_width_mw":float(2*sub.rolling_radius_mw.mean()),
                "fixed_mean_interval_score_mw":interval_score(sub.net_load_mw-sub.prediction_mw,sub.fixed_radius_mw),
                "rolling_mean_interval_score_mw":interval_score(sub.net_load_mw-sub.prediction_mw,sub.rolling_radius_mw)})
    report = {"status":"Known-year post-hoc diagnostics, no controller feedback or independent confirmation.",
        "rolling_rule":"Prior observed absolute residuals over 720 hours, minimum 168 observations, q90. Truth is available only at target hour end; frozen hourly radius until enough history.",
        "high_load_threshold_mw":high_load,"groups":rows,
        "horizon_1_overall":{key:float(joined.loc[joined.horizon_hours.eq(1),key].mean()*100) for key in ("fixed_covered","rolling_covered")}}
    write(OUT/"interval_audit_metrics.json",report)
    print(report["horizon_1_overall"])


def interval_score(residual,radius):
    # Proper central-90% interval score: width + 2/alpha times misses.
    return float((2*radius+20*np.maximum(0.,np.abs(residual)-radius)).mean())


if __name__ == "__main__":
    main()
