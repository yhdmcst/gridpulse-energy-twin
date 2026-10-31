"""Shared typography and frozen-result tokens for the current competition report."""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable

ABSTRACT = (
    "面向风光波动下的净负荷预测与储能电量恢复问题，本文提出GridPulse能源数字孪生离线原型，"
    "将因果多时距预测、滚动规划、物理执行仿真、每小时SOC回传与失效降级连接为可审计闭环。"
    "基于德国公开小时数据，使用同输入梯度提升基线、残差神经网络及一致性消融，区分预测能力与核算自洽性。"
    "三种子结果显示，数据网络相对匹配基线的MAE下降约14%至20%，而一致性损失的精度作用微小且不一致。"
    "调度实验进一步暴露长窗口最高点改善与P95退步并存，以及错误参数开环执行的电量欠账。"
    "在准确即时SOC和预先给定功率、效率下界条件下，保守规划候选的三十条范围内评价轨迹均恢复日末电量，"
    "但仅十一条同时通过原有性能预算，范围外仍存在失败。系统保存逐小时请求、实际动作、门控与求解日志，"
    "支持数据过期及服务中断时的恢复或空闲回退。"
    "该原型可用于策略比较、故障回放和运行审查；"
    "后续研究将结合独立地区数据、设备遥测误差和电网约束检验适用范围。"
)


def tokens(e):
    result = {"abstract": ABSTRACT}
    for split in ("test", "2021", "2022"):
        means = {}
        for name, method in (("hgb", "HGB"), ("data", "Data MLP"),
                             ("physics", "Consistency MLP"),
                             ("projected", "Consistency + projection")):
            values = [r["mae_mw"] for r in e["prediction"] if r["split"] == split and r["method"] == method]
            means[name] = sum(values) / len(values)
            result[f"{name}_{split}"] = f"{means[name]:.2f}"
        result[f"gain_{split}"] = f"{100 * (1 - means['data'] / means['hgb']):.2f}"
        result[f"physics_delta_{split}"] = f"{means['physics'] - means['data']:+.2f}"
        t = next(r for r in e["tail"] if r["split"] == split)
        result[f"tail_p95_{split}"] = f"{t['p95_delta_mw']:+.2f}"
        result[f"tail_through_{split}"] = f"{t['throughput_delta_pct']:+.2f}"
        result[f"tail_loss_{split}"] = f"{t['loss_delta_mwh']:+.2f}"
    for year in ("2021", "2022"):
        v7 = e["v7"]["healthy"][year]["fresh6"]["nominal"]["supervised_minus_feedback"]
        result[f"v7_peak_{year}"] = f"{v7['controlled_peak_mw']:+.2f}"
        result[f"v7_p95_{year}"] = f"{v7['controlled_p95_mw']:+.2f}"
        for case in ("nominal", "combined90"):
            d = e["v8"]["healthy"][year]["fresh6"][case]["envelope_tail"]["minus_v7"]
            result[f"v8_loss_{case}_{year}"] = f"{d['conversion_loss_mwh']:+.2f}"
        result[f"outside_{year}"] = f"{e['v8']['outside'][year]['envelope_tail']['metrics']['max_abs_terminal_error_mwh']:.2f}"
        result[f"v7_fallback_{year}"] = str(e["v8"]["faults"][year]["combined_fault"]["baseline_v7"]["metrics"]["fallback_hours"])
    result["v8_minutes"] = f"{e['v8']['seconds'] / 60:.2f}"
    result["v8_validation_p95"] = f"{e['selection']['candidates']['envelope_tail']['mean_p95_delta_mw']:+.2f}"
    result["budget_pass"] = str(e["v8"]["counts"]["retained_later_conditions_passing_all_descriptive_budgets"])
    result["hashes"] = str(e["verification"]["artifact_hashes_verified"])
    return result


def fonts():
    pdfmetrics.registerFont(TTFont("SimSunReport", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SimHeiReport", r"C:\Windows\Fonts\simhei.ttf"))
    pdfmetrics.registerFontFamily("SimSunReport", normal="SimSunReport", bold="SimHeiReport",
                                 italic="SimSunReport", boldItalic="SimHeiReport")


def inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='SimHeiReport'>\1</font>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


class Marker(Flowable):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.width = self.height = 0

    def draw(self):
        pass


if __name__ == "__main__":
    from competition_report import main
    main()
