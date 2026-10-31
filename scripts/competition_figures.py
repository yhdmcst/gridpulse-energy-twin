from __future__ import annotations

from pathlib import Path

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"

# Match the Chinese font used in the report and deck so generated evidence is legible.
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def forecast_interval_figure() -> Path:
    predictions = pd.read_csv(
        ROOT / "outputs/models/hist_gradient_boosting_test_predictions.csv", parse_dates=["timestamp"]
    ).sort_values("timestamp")
    peak_index = predictions["net_load_mw"].idxmax()
    center = predictions.index.get_loc(peak_index)
    window = predictions.iloc[max(0, center - 84) : min(len(predictions), center + 85)]

    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(window["timestamp"], window["net_load_mw"], label="实际净负荷", color="#17302A", linewidth=1.5)
    axis.plot(
        window["timestamp"],
        window["hist_gradient_boosting_prediction_mw"],
        label="梯度提升预测",
        color="#0F766E",
        linewidth=1.3,
    )
    axis.fill_between(
        window["timestamp"],
        window["hist_gradient_boosting_lower_mw"],
        window["hist_gradient_boosting_upper_mw"],
        label="验证期校准经验区间",
        color="#7FC8B5",
        alpha=0.35,
    )
    axis.set_title("冻结测试期高负荷窗口的预测与经验区间")
    axis.set_ylabel("净负荷 (MW)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    path = FIGURES / "05_forecast_interval_peak_week.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def storage_ablation_figure() -> Path:
    fixed = pd.read_csv(
        ROOT / "outputs/scenarios/storage_30000mwh_direct_fixed_reserve_timeseries.csv", parse_dates=["timestamp"]
    )
    dynamic = pd.read_csv(
        ROOT / "outputs/scenarios/storage_30000mwh_risk_aware_timeseries.csv", parse_dates=["timestamp"]
    )
    merged = fixed.merge(dynamic, on="timestamp", suffixes=("_fixed", "_dynamic")).sort_values("timestamp")
    effect_index = (merged["controlled_net_load_mw_fixed"] - merged["controlled_net_load_mw_dynamic"]).idxmax()
    center = merged.index.get_loc(effect_index)
    window = merged.iloc[max(0, center - 48) : min(len(merged), center + 49)]

    fig, (net_axis, soc_axis) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    net_axis.plot(window["timestamp"], window["net_load_mw_fixed"], label="实际净负荷", color="#17302A", linewidth=1.4)
    net_axis.plot(
        window["timestamp"], window["controlled_net_load_mw_fixed"], label="固定 30% 储备", color="#E7A73B", linewidth=1.2
    )
    net_axis.plot(
        window["timestamp"], window["controlled_net_load_mw_dynamic"], label="风险感知动态储备", color="#0F766E", linewidth=1.3
    )
    net_axis.set_title("30 GWh 同信息储备策略对照窗口")
    net_axis.set_ylabel("受控净负荷 (MW)")
    net_axis.legend(frameon=False, ncol=3, loc="upper left")
    net_axis.grid(alpha=0.2)

    soc_axis.plot(window["timestamp"], window["storage_soc_mwh_fixed"], label="固定储备 SOC", color="#E7A73B", linewidth=1.2)
    soc_axis.plot(window["timestamp"], window["storage_soc_mwh_dynamic"], label="动态储备 SOC", color="#0F766E", linewidth=1.3)
    soc_axis.axhline(9000, color="#E7A73B", linestyle="--", linewidth=1, label="固定 30% SOC 下限")
    soc_axis.set_ylabel("SOC (MWh)")
    soc_axis.grid(alpha=0.2)
    soc_axis.legend(frameon=False, ncol=3, loc="upper left")
    soc_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    path = FIGURES / "06_same_information_storage_ablation_30gwh.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def peak_response_figure() -> Path:
    report = json.loads((ROOT / "outputs/scenarios/peak_response_metrics.json").read_text(encoding="utf-8"))
    experiments = report["periods"]["test"]["experiments"]
    peak_date = pd.Timestamp(experiments["dynamic_peak_response"]["peak_event"]["timestamp"]).date()
    fig, (net_axis, soc_axis) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for key, label, color in (("dynamic_upper", "原动态", "#B45309"),
                              ("dynamic_peak_response", "新动态组合", "#0F766E"),
                              ("fixed_peak_response", "新固定组合", "#567AB0")):
        trace = pd.read_csv(ROOT / experiments[key]["timeseries"], parse_dates=["timestamp"])
        day = trace.loc[trace.timestamp.dt.date.eq(peak_date)]
        if key == "dynamic_upper":
            net_axis.plot(day.timestamp, day.net_load_mw, label="实际净负荷", color="#17302A", linewidth=1.4)
        net_axis.plot(day.timestamp, day.controlled_net_load_mw, label=label, color=color, linewidth=1.2)
        soc_axis.plot(day.timestamp, day.storage_soc_mwh, label=label, color=color, linewidth=1.2)
    net_axis.set_title("测试期原始最高峰所在日：削峰、补电与储能电量")
    net_axis.set_ylabel("受控净负荷 (MW)")
    net_axis.legend(ncol=4, frameon=False)
    net_axis.grid(alpha=.2)
    soc_axis.axhline(4500, color="#B45309", linestyle="--", label="共同硬下限")
    soc_axis.axhline(15000, color="#50625E", linestyle=":", label="每日初末电量")
    soc_axis.set_ylabel("SOC (MWh)")
    soc_axis.legend(ncol=3, frameon=False)
    soc_axis.grid(alpha=.2)
    soc_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    path = FIGURES / "07_peak_response_daily_accounting.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(forecast_interval_figure())
    print(storage_ablation_figure())
    print(peak_response_figure())


if __name__ == "__main__":
    main()
