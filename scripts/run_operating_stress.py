"""Fixed calendar episodes and scenarios, reported as post-hoc operating stress."""
from __future__ import annotations

from dataclasses import asdict,replace
import sys

import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_controller_development import calibration,read,write,OUT
from src.balanced_storage import simulate_balanced_storage
from src.efficient_storage import ControlConfig,simulate,summarize
from src.forecast_feed import stress_feed
from src.optimality import StorageModel


def main():
    protocol = {"status":"Post-hoc operating sensitivity on four fixed calendar weeks, not independent validation.",
        "weeks":["2021-01-08","2021-04-08","2021-07-08","2021-10-08"],
        "policies":["frozen_reference","two_stage_250mw_candidate"],
        "scenarios":["nominal","feed_delay_1h","feed_delay_3h","six_hour_outage",
            "forecast_bias_plus_1000mw","forecast_bias_minus_1000mw","power_70pct","efficiency_0_85","continuous_week"],
        "meaning":"Feed lag delays an already frozen forecast issue, not a retrained model with delayed observations. Continuity is within each week only. All weeks and scenarios fixed before stress evaluation."}
    write(OUT/"operating_stress_protocol.json",protocol)
    panel = pd.read_csv(OUT/"known_2021_panel.csv",parse_dates=["timestamp"])
    panel.timestamp = pd.to_datetime(panel.timestamp,utc=True)
    cal,model = calibration(),StorageModel()
    report = {"protocol":protocol,"episodes":{}}
    for date in protocol["weeks"]:
        start = pd.Timestamp(date,tz="UTC")
        source = panel.loc[panel.timestamp.ge(start-pd.Timedelta(days=1))&panel.timestamp.lt(start+pd.Timedelta(days=7))].reset_index(drop=True)
        outage = list(range(24+3*24+8,24+3*24+14))
        week = {}
        for scenario in protocol["scenarios"]:
            feed,diagnostics = stress_feed(source,delay_hours=1 if scenario == "feed_delay_1h" else 3 if scenario == "feed_delay_3h" else 0,
                outage_rows=outage if scenario == "six_hour_outage" else (),
                bias_mw=1000. if scenario == "forecast_bias_plus_1000mw" else -1000. if scenario == "forecast_bias_minus_1000mw" else 0.,allow_warmup=True)
            feed = feed.loc[feed.timestamp.ge(start)].reset_index(drop=True)
            physical = replace(model,charge_power_mw=875.,discharge_power_mw=875.) if scenario == "power_70pct" else replace(model,efficiency=.85) if scenario == "efficiency_0_85" else model
            continuous = scenario == "continuous_week"
            results = {}
            for policy in protocol["policies"]:
                if continuous and policy == "frozen_reference":
                    # Original daily controller has no equivalent continuous
                    # terminal model. Do not use daily resets as comparator.
                    results[policy] = {"status":"not_applicable","reason":"Frozen primary reference uses independent daily terminal equality."}
                    continue
                if policy == "frozen_reference":
                    trace,_ = simulate_balanced_storage(feed,cal["peak_threshold_mw"],0.,cal["uncertainty_scale_mw"],
                        capacity_mwh=physical.capacity_mwh,charge_power_mw=physical.charge_power_mw,
                        discharge_power_mw=physical.discharge_power_mw,efficiency=physical.efficiency,
                        planning_objective="point",release_threshold_mw=cal["release_threshold_mw"])
                    summary = summarize(trace,physical)
                else:
                    trace,summary = simulate(feed,**cal,model=physical,config=ControlConfig(250.,not continuous))
                path = OUT/f"stress_{date}_{scenario}_{policy}.csv"
                trace.to_csv(path,index=False)
                results[policy] = {"status":"solved",**summary,"timeseries":path.relative_to(ROOT).as_posix()}
            week[scenario] = {"model":asdict(physical),"feed":diagnostics,"experiments":results}
            print(f"{date}/{scenario} completed",flush=True)
        report["episodes"][date] = week
        write(OUT/"operating_stress_metrics.json",report)


if __name__ == "__main__":
    main()
