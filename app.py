from __future__ import annotations

import hashlib
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.fusion_evidence import POLICY, load_evidence

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="GridPulse V8", page_icon="GP", layout="wide")

CASE_NAMES = {
    "nominal": "名义设备 · 1,250 MW / 0.92",
    "power_90pct": "功率下降 10% · 1,125 MW",
    "power_70pct": "功率下降 30% · 875 MW",
    "efficiency_090": "单向效率 0.90",
    "efficiency_085": "单向效率 0.85",
    "combined90": "组合误差 · 1,125 MW / 0.90",
    "outside": "范围外 · 750 MW / 0.80",
}
EVENT_NAMES = {
    "healthy": "正常运行", "stale_burst": "预测包过期",
    "solver_outage": "求解服务中断", "combined_fault": "组合故障",
}
MODE_NAMES = {"fresh6": "新鲜预测 · 6 小时", "delayed1": "延迟 1 小时 · 原 h2-h6"}
POLICY_NAMES = {
    "envelope_tail": "V8 保守尾部规划",
    "envelope_reference": "V8 保守参考规划",
    "baseline_v7": "V7 反馈与恢复监督",
}
REASONS = {
    "ok": "正常求解", "forecast_expired": "预测过期",
    "solver_unavailable": "服务不可用", "primary_solver_failure": "主优化失败",
    "reference_solver_failure": "参考求解失败",
}


@st.cache_data
def read_evidence(fingerprint):
    return load_evidence(ROOT)


@st.cache_data
def read_trace(evidence, metrics, fingerprint):
    return evidence.trace(metrics)


def chart(frame, columns, colors, height=280, zero=False):
    series = frame[["timestamp", *columns]].rename(columns=columns).melt(
        id_vars="timestamp", var_name="指标", value_name="数值"
    )
    series["UTC"] = series.timestamp.dt.strftime("%Y-%m-%d %H:%M")
    visual = alt.Chart(series).mark_line(strokeWidth=1.6).encode(
        x=alt.X("timestamp:T", title="时间（UTC）", scale=alt.Scale(type="utc")),
        y=alt.Y("数值:Q", scale=alt.Scale(zero=zero), title=None),
        color=alt.Color("指标:N", scale=alt.Scale(domain=list(columns.values()), range=colors),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("UTC:N"), "指标:N", alt.Tooltip("数值:Q", format=",.2f")],
    ).properties(height=height)
    st.altair_chart(visual, width="stretch")


try:
    files = [ROOT / "outputs/fusion_v8" / name for name in (
        "ARTIFACT_SHA256.json", "metrics.json", "selection.json", "decision.json", "status.json"
    )]
    fingerprint = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in files)
    evidence = read_evidence(fingerprint)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"V8 结果读取失败：{exc}")
    st.stop()

st.title("GridPulse V8")
st.caption("能源数字孪生 · 保守规划包络 · 德国 2021 / 2022 小时数据 · 固定模型 seed42")

with st.sidebar:
    st.header("实验条件")
    years = sorted(evidence.report["methods"])
    year = st.selectbox("评价年份", years, index=len(years) - 1)
    modes = evidence.report["methods"][year]
    mode = st.selectbox("预测条件", list(modes), format_func=MODE_NAMES.get)
    cases = modes[mode]
    case = st.selectbox("设备条件", list(cases), format_func=lambda key: CASE_NAMES.get(key, key))
    events = cases[case]
    event = st.selectbox("故障条件", list(events), format_func=EVENT_NAMES.get)
    comparison = st.checkbox("叠加 V7 对照", value=True)
    st.divider()
    st.caption("规划：875 MW / 0.85\n\n容量：30 GWh\n\n运行电量：4.5–15 GWh\n\n日末目标：15 GWh")

try:
    metrics = evidence.metrics(year, mode, case, event)
    baseline_metrics = evidence.metrics(year, mode, case, event, "baseline_v7")
    trace_path = evidence.directory / metrics["trajectory"]
    trace = read_trace(evidence, metrics, (trace_path.stat().st_mtime_ns, trace_path.stat().st_size))
    baseline_path = evidence.directory / baseline_metrics["trajectory"]
    baseline = read_trace(evidence, baseline_metrics,
                          (baseline_path.stat().st_mtime_ns, baseline_path.stat().st_size))
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"轨迹校验失败：{exc}")
    st.stop()

candidate_rows = evidence.catalog.loc[evidence.catalog.policy.eq(POLICY)]
inside = candidate_rows.loc[candidate_rows.inside_envelope]
top = st.columns(4)
top[0].metric("范围内电量恢复", f"{int(inside.recovery_passed.sum())} / {len(inside)}")
top[1].metric("全部性能预算通过", f"{int(inside.all_budgets_passed.sum())} / {len(inside)}")
top[2].metric("评价时长 / 年", f"{metrics['rows']:,} h")
top[3].metric("当前日末最大偏差", f"{metrics['max_abs_terminal_error_mwh']:,.2f} MWh")

playback, results, method, audit = st.tabs(["闭环回放", "实验对照", "技术路线", "工件核对"])

with playback:
    st.subheader(f"{year} · {CASE_NAMES.get(case, case)} · {EVENT_NAMES[event]}")
    first, last = trace.timestamp.iloc[0].date(), trace.timestamp.iloc[-1].date()
    dates = st.date_input(
        "回放日期（UTC）",
        value=(first, min(first + pd.Timedelta(days=6), last)),
        min_value=first, max_value=last, key=f"dates_{year}_{mode}_{case}_{event}",
    )
    if len(dates) == 2:
        start, end = dates
        window = trace.loc[trace.timestamp.dt.date.between(start, end)].copy()
        reference = baseline.loc[baseline.timestamp.dt.date.between(start, end)]
        window["v7_controlled_mw"] = reference.controlled_net_load_mw.to_numpy()
        if not metrics["plant_within_declared_envelope"]:
            st.warning(f"范围外压力试验：最大日末欠账 {metrics['max_abs_terminal_error_mwh']:,.2f} MWh。")
        left, right = st.columns(2)
        with left:
            st.markdown("**净负荷 / MW**")
            columns = {"net_load_mw": "实际净负荷", "controlled_net_load_mw": "V8 受控净负荷"}
            palette = ["#90999f", "#007f75"]
            if comparison:
                columns["v7_controlled_mw"] = "V7 对照"
                palette.append("#c05a48")
            chart(window, columns, palette)
        with right:
            st.markdown("**储能电量 / MWh**")
            chart(window, {"storage_soc_mwh": "实际电量", "safe_floor_next_mwh": "恢复下界"},
                  ["#246bb2", "#c88b18"])
        st.markdown("**请求与实际功率 / MW**")
        window["requested_net_mw"] = window.requested_charge_mw - window.requested_discharge_mw
        window["actual_net_mw"] = window.charge_mw - window.discharge_mw
        chart(window, {"requested_net_mw": "请求功率", "actual_net_mw": "执行功率"},
              ["#95623b", "#007f75"], height=220, zero=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前时段回退", f"{int(window.fallback_active.sum())} h")
        c2.metric("当前时段监督干预", int(window.supervisor_intervened.sum()))
        c3.metric("全段最高点差（V8−V7）",
                  f"{metrics['controlled_peak_mw'] - baseline_metrics['controlled_peak_mw']:+,.2f} MW")
        c4.metric("全段 P95 差（V8−V7）",
                  f"{metrics['controlled_p95_mw'] - baseline_metrics['controlled_p95_mw']:+,.2f} MW")
        st.markdown("**逐小时运行日志**")
        faults_only = st.checkbox("仅显示回退或监督干预时段", key="faults_only")
        log = window.loc[window.fallback_active | window.supervisor_intervened] if faults_only else window
        log = log[["timestamp", "storage_soc_mwh", "safe_floor_next_mwh",
                   "requested_charge_mw", "requested_discharge_mw", "charge_mw", "discharge_mw",
                   "gate_reason", "primary_solver_attempted", "supervisor_intervened"]].copy()
        log["gate_reason"] = log.gate_reason.map(lambda value: REASONS.get(value, value))
        st.dataframe(log.rename(columns={
            "timestamp": "时间（UTC）", "storage_soc_mwh": "实际电量 / MWh",
            "safe_floor_next_mwh": "恢复下界 / MWh",
            "requested_charge_mw": "请求充电 / MW", "requested_discharge_mw": "请求放电 / MW",
            "charge_mw": "实际充电 / MW", "discharge_mw": "实际放电 / MW",
            "gate_reason": "运行状态", "primary_solver_attempted": "主优化调用",
            "supervisor_intervened": "监督干预",
        }), width="stretch", hide_index=True, height=300)
        st.download_button("导出当前轨迹", window.to_csv(index=False).encode("utf-8-sig"),
                           f"gridpulse_v8_{year}_{mode}_{case}_{event}.csv", "text/csv",
                           icon=":material/download:")
    else:
        st.info("选择回放结束日期。")

with results:
    st.subheader("V8 全条件结果")
    policy = st.selectbox("结果策略", [POLICY, "envelope_reference", "baseline_v7"],
                          format_func=POLICY_NAMES.get)
    table = evidence.catalog.loc[evidence.catalog.policy.eq(policy)].copy()
    table["case"] = table["case"].map(lambda value: CASE_NAMES.get(value, value))
    table["event"] = table["event"].map(EVENT_NAMES)
    st.dataframe(table.drop(columns="policy").rename(columns={
        "year": "年份", "mode": "预测条件", "case": "设备条件", "event": "故障条件",
        "controlled_peak_mw": "最高点 / MW", "controlled_p95_mw": "P95 / MW",
        "mean_daily_peak_mw": "平均日峰 / MW", "throughput_mwh": "吞吐 / MWh",
        "conversion_loss_mwh": "损耗 / MWh", "max_abs_terminal_error_mwh": "日末最大偏差 / MWh",
        "inside_envelope": "范围内", "recovery_passed": "电量恢复",
        "peak_delta_mw": "最高点差 / MW", "p95_delta_mw": "P95差 / MW",
        "fallback_hours": "回退小时", "all_budgets_passed": "全部预算通过",
    }), hide_index=True, width="stretch", height=500)
    st.image(str(evidence.directory / "Figure_2.png"), caption="同条件配对比较：V8 与 V7 的最高点、P95 变化")
    st.download_button("导出完整对照表", evidence.catalog.to_csv(index=False).encode("utf-8-sig"),
                       "gridpulse_v8_all_conditions.csv", "text/csv", icon=":material/download:")

with method:
    st.subheader("预测、规划、执行与状态反馈")
    st.dataframe(pd.DataFrame([
        ["数据", "德国聚合负荷、风电和光伏；UTC 小时对齐", "data/processed"],
        ["预测", "冻结 data_seed42；直接输出 h1–h6 及原始时距误差区间", "src/multihorizon_fusion.py"],
        ["规划", "875 MW / 0.85 保守模型；尾部超额与参考电量约束", "src/envelope_planning.py"],
        ["门控", "观测年龄上限 2 h；服务故障触发恢复或空闲", "src/feedback_control.py"],
        ["执行", "按场景实际功率、效率和剩余电量限幅", "src/feedback_control.py"],
        ["回传", "下一小时读取实际执行后的 SOC；整段跨日连续", "outputs/fusion_v8/evaluation"],
        ["核对", "逐小时能量平衡、恢复下界、物理边界和失败计数", "scripts/verify_envelope_results.py"],
    ], columns=["阶段", "实现", "代码与工件"]), hide_index=True, width="stretch")
    st.latex(r"E_{t+1}=E_t+\eta c_t-d_t/\eta")
    st.latex(r"E_{\mathrm{floor}}(k)=\max(4500,\;15000-(23-k)\times875\times0.85)")
    st.markdown("**实验设置**")
    st.write("准确即时 SOC 回传；功率、效率下界预先给定；每年 363 个完整 UTC 日。"
             "30 条范围内候选轨迹均恢复电量，11 条同时满足全部性能预算；750 MW / 0.80 场景保留恢复失败。")
    st.image(str(evidence.directory / "Figure_1.png"), caption="验证期锁定 envelope_tail，随后保持选择完成评价")

with audit:
    st.subheader("当前工件")
    st.dataframe(pd.DataFrame([
        ["实验版本", "fusion_v8"], ["完成状态", evidence.report["status"]],
        ["锁定候选", POLICY], ["固定模型", "data_seed42"],
        ["冻结工件数", str(len(evidence.hashes))],
        ["所选轨迹", metrics["trajectory"]],
        ["轨迹 SHA-256", hashlib.sha256(evidence.checked_bytes(metrics["trajectory"])).hexdigest()],
        ["实验用途", "离线仿真与轨迹回放"],
    ], columns=["项目", "值"]), hide_index=True, width="stretch")
    st.code(metrics["trajectory"], language="text")
