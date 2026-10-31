"""Real-data competition figures with the nature-figure rendered QA gates."""
from __future__ import annotations

import json
import sys

from competition_common import DEST, FIGURES, QA, SKILL, WORKSPACE, collect, write_provenance

sys.path.insert(0, str(WORKSPACE / ".report-deps"))
sys.path.insert(0, str(SKILL / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.text import Text
import numpy as np
import report_figures as legacy
import report_evidence as source
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_figure_collisions import audit_pdf, write_overlay_pdf
from audit_pdf_text import audit_pdf as audit_text

TEAL, BLUE, RED, GRAY, AMBER = legacy.TEAL, legacy.BLUE, legacy.RED, legacy.GRAY, legacy.AMBER
SPLITS = ("test", "2021", "2022")
SEEDS = (42, 43, 44)
SUMMARY = {}


def save(fig, name):
    for text in fig.findobj(Text):
        if "\n" in text.get_text():
            text.set_linespacing(1.55)
    if name == "fig08_envelope_tradeoffs":
        for ax in fig.axes:
            ax.grid(False)
    multi = len(fig.axes) > 1
    if multi:
        for i, ax in enumerate(fig.axes):
            transform = ax.transAxes + ScaledTranslation(-16 / 72, 9 / 72, fig.dpi_scale_trans)
            ax.text(0, 1, chr(97 + i), transform=transform,
                    ha="left", va="bottom", fontweight="bold", fontsize=10, clip_on=False)
    fig.canvas.draw()
    if multi:
        positions = [ax.get_position().frozen() for ax in fig.axes]
        fig.set_layout_engine("none")
        if max(p.y0 for p in positions) - min(p.y0 for p in positions) < .05:
            left, right = min(p.x0 for p in positions), max(p.x1 for p in positions)
            bottom, top = max(p.y0 for p in positions), min(p.y1 for p in positions)
            width = min(p.width for p in positions)
            gap = (right - left - len(positions) * width) / (len(positions) - 1)
            for i, ax in enumerate(fig.axes):
                ax.set_position([left + i * (width + gap), bottom, width, top - bottom])
        else:
            left, right = max(p.x0 for p in positions), min(p.x1 for p in positions)
            bottom, top = min(p.y0 for p in positions), max(p.y1 for p in positions)
            height = min(p.height for p in positions)
            gap = (top - bottom - len(positions) * height) / (len(positions) - 1)
            for i, ax in enumerate(fig.axes):
                ax.set_position([left, top - height - i * (height + gap), right - left, height])
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=QA / f"{name}.alignment.json",
        overlay_svg=QA / f"{name}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=multi, strict=True,
    )
    for ext in ("svg", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300)
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    pdf = FIGURES / f"{name}.pdf"
    glyphs = audit_text(pdf.read_bytes(), minimum_pt=5)
    (QA / f"{name}.text-audit.json").write_text(json.dumps(glyphs, indent=2), encoding="utf-8")
    collisions = audit_pdf(pdf)
    (QA / f"{name}.collision-audit.json").write_text(
        json.dumps(collisions, ensure_ascii=False, indent=2), encoding="utf-8")
    if collisions["findings"]:
        write_overlay_pdf(pdf, QA / f"{name}.collision-overlay.pdf", collisions["findings"])
    SUMMARY[name] = {
        "alignment": alignment["verdict"], "glyph_min_pt": glyphs["minimum_found_pt"],
        "collision": collisions["verdict"], "collision_summary": collisions["summary"],
    }
    (QA / "figure_audit_summary.json").write_text(
        json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
    assert glyphs["auditable"] and not glyphs["below_minimum_count"], (name, glyphs)
    if collisions["summary"]["fail"]:
        raise RuntimeError(f"{name}: collision FAIL; inspect QA report")


def architecture():
    fig, ax = plt.subplots(figsize=(6.77, 3.65))
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    legacy.node(ax, .02, .69, .25, .23, "历史数据与时钟", "负荷 / 风 / 光\nUTC与观测可用时刻", BLUE)
    legacy.node(ax, .375, .69, .25, .23, "多时距预测", "h1-h6 + 验证期区间\n同输入、配对种子", BLUE)
    legacy.node(ax, .73, .69, .25, .23, "滚动窗口规划", "当前状态 + 保守下界\n参考约束 + 尾部目标", TEAL)
    legacy.node(ax, .73, .19, .25, .23, "门控与恢复监督", "数据过期 / 服务中断\n恢复充电或空闲", AMBER)
    legacy.node(ax, .375, .19, .25, .23, "物理执行仿真", "实际限幅 / 转换损耗\n请求与实际动作分离", TEAL)
    legacy.node(ax, .02, .19, .25, .23, "轨迹审计与回放", "状态连续 / 能量守恒\n失败与哈希留痕", GRAY)
    for a, b in (
        ((.27, .805), (.375, .805)), ((.625, .805), (.73, .805)),
        ((.855, .69), (.855, .42)), ((.73, .305), (.625, .305)),
        ((.375, .305), (.27, .305)),
    ):
        legacy.arrow(ax, a, b)
    ax.plot([.5, .5, .745], [.42, .55, .55], color=TEAL, lw=1.1)
    legacy.arrow(ax, (.745, .55), (.745, .69), color=TEAL)
    ax.text(.62, .595, "每小时仿真 SOC 回传", ha="center", fontsize=8, color=TEAL)
    ax.text(.5, .065, "控制器不读取真实功率/效率；本实验 SOC 准确、即时，充电资源持续可用。",
            ha="center", fontsize=7.5, color=RED)
    save(fig, "fig01_architecture")


def fallback():
    fig, ax = plt.subplots(figsize=(6.77, 3.65))
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.03)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    legacy.node(ax, .02, .70, .25, .22, "读取当前数据", "SOC / 时间戳\n预测值与上下界", BLUE)
    legacy.node(ax, .375, .70, .25, .22, "数据与服务门控", "年龄 ≤ 2 h\n时钟、区间及服务有效", AMBER)
    legacy.node(ax, .73, .70, .25, .22, "主优化", "仅可用信息\n保存真实失败状态", TEAL)
    legacy.node(ax, .375, .25, .25, .22, "确定性降级", "不沿用缓存放电\n按当前 SOC 恢复或空闲", AMBER)
    legacy.node(ax, .73, .25, .25, .22, "监督与实际执行", "恢复检查优先\n限幅并记录实际动作", TEAL)
    for a, b in (
        ((.27, .81), (.375, .81)), ((.625, .81), (.73, .81)),
        ((.5, .70), (.5, .47)), ((.855, .70), (.855, .47)),
        ((.73, .70), (.625, .47)), ((.625, .36), (.73, .36)),
    ):
        legacy.arrow(ax, a, b)
    ax.text(.55, .59, "拒绝", ha="left", fontsize=8, color=AMBER)
    ax.text(.76, .58, "失败", ha="center", fontsize=8, color=RED)
    ax.text(.15, .40, "门控拒绝时\n主优化调用 = 0", ha="center", fontsize=9, color=AMBER)
    ax.text(.5, .09, "SOC 无效时拒绝执行；服务中断不是设备停电，尚未验证设备拒动。",
            ha="center", fontsize=7.7, color=RED)
    save(fig, "fig02_fallback")


def paired_points(ax, groups, field, title):
    for i, split in enumerate(SPLITS):
        values = [r[field] for r in groups if r["split"] == split]
        for offset, value, marker in zip((-.14, 0, .14), values, ("o", "s", "^")):
            ax.scatter(i + offset, value, color=BLUE, marker=marker, s=26, zorder=4)
        ax.plot([i - .25, i + .25], [np.mean(values)] * 2, color=TEAL, lw=2)
    ax.axhline(0, color=GRAY, lw=.7)
    ax.set_xticks(range(3), ("原test", "2021", "2022"), fontsize=8)
    ax.set_title(title, fontsize=9.3)
    ax.margins(x=.2, y=.25)
    legacy.clean(ax)


def prediction(e):
    fig, axes = plt.subplots(1, 3, figsize=(6.77, 3.2), layout="constrained")
    ax = axes[0]
    for i, split in enumerate(SPLITS):
        baseline = next(r["mae_mw"] for r in e["prediction"] if r["split"] == split and r["method"] == "HGB")
        values = [r["mae_mw"] for r in e["prediction"] if r["split"] == split and r["method"] == "Data MLP"]
        ax.bar(i - .18, baseline, .32, color="#D7DDE1", edgecolor=GRAY, lw=.6)
        ax.bar(i + .18, np.mean(values), .32, color="#B8DAD7", edgecolor=TEAL, lw=.6)
        for offset, value, marker in zip((-.065, 0, .065), values, ("o", "s", "^")):
            ax.scatter(i + .18 + offset, value, color=TEAL, s=18, marker=marker, zorder=4)
    ax.set(ylim=(0, 1000), ylabel="净负荷 MAE / MW")
    ax.set_xticks(range(3), ("原test", "2021", "2022"), fontsize=8)
    ax.set_title("同输入 HGB 与数据 MLP", fontsize=9.3)
    legacy.clean(ax)
    paired = []
    for split in SPLITS:
        for seed in SEEDS:
            d = e["v1"]["runs"][f"data_seed{seed}"]["variants"]["raw"][split]
            p = e["v1"]["runs"][f"physics_seed{seed}"]["variants"]["raw"][split]
            paired.append(dict(split=split, seed=seed, mae_delta_mw=p["mae_mw"] - d["mae_mw"],
                               accounting_delta_mw=p["accounting_residual_mae_mw"] - d["accounting_residual_mae_mw"]))
    source.export_table("fig04_consistency_paired", paired)
    paired_points(axes[1], paired, "mae_delta_mw", "一致性损失的精度效应")
    axes[1].set_ylabel("一致性 MLP - 数据 MLP / MW")
    paired_points(axes[2], paired, "accounting_delta_mw", "核算残差的配对变化")
    axes[2].set_ylabel("核算残差 MAE 差 / MW")
    save(fig, "fig04_prediction")


def dispatch(e):
    fig, axes = plt.subplots(1, 3, figsize=(6.77, 3.2), layout="constrained")
    for ax, field, title in zip(axes,
                               ("horizon_peak_delta_mw", "horizon_p95_delta_mw", "tail_p95_delta_mw"),
                               ("六小时 - 一小时：最高点", "六小时 - 一小时：P95", "尾部保护 - 参考：P95")):
        paired_points(ax, e["dispatch_pairs"], field, title)
        ax.set_ylabel("配对差 / MW")
    save(fig, "fig05_dispatch_tradeoffs")


def robustness(e):
    fig, axes = plt.subplots(1, 2, figsize=(6.77, 3.2), layout="constrained")
    modes = ("fresh6", "fresh5", "fresh3", "delayed1", "delayed3")
    v6 = source.read("outputs/fusion_v6_analysis.json")
    for year, offset, color, marker in (("2021", -.12, BLUE, "o"), ("2022", .12, TEAL, "s")):
        for i, mode in enumerate(modes):
            values = v6["delay"][year][mode]["seed_p95_deltas_mw"]
            axes[0].scatter(i + offset + np.linspace(-.035, .035, 3), values,
                            marker=marker, color=color, s=18, label=year if i == 0 else None)
    axes[0].set_xticks(range(5), ("新6h", "新5h", "新3h", "延1h", "延3h"), fontsize=8)
    axes[0].set(ylabel="tail_guard - legacy，P95 / MW", title="旧预测包与短窗口分开评价")
    axes[0].axhline(0, color=GRAY, lw=.7)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.11), ncol=2, fontsize=8)
    cases = source.CASES[1:5]
    for year, offset, color, marker in (("2021", -.12, BLUE, "o"), ("2022", .12, TEAL, "s")):
        vals = [next(r["terminal_error_mwh"] for r in e["mismatch"]
                     if r["year"] == year and r["mode"] == "fresh6" and r["case"] == c) for c in cases]
        axes[1].scatter(np.arange(4) + offset, vals, color=color, marker=marker, s=25)
    axes[1].set_xticks(range(4), ("功率90%", "功率70%", "效率.90", "效率.85"), fontsize=7.8)
    axes[1].set(ylabel="最大日末偏差 / MWh", title="错误参数、固定请求回放", ylim=(0, 10500))
    for ax in axes:
        legacy.clean(ax)
    save(fig, "fig06_robustness")


def feedback(e):
    fig, axes = plt.subplots(1, 2, figsize=(6.77, 3.3), layout="constrained")
    rows = e["feedback"]
    keys = [(r["year"], r["mode"], r["case"]) for r in rows if r["method"] == "open"]
    for j, key in enumerate(keys):
        offset = (j / (len(keys) - 1) - .5) * .25
        values = [next(r["terminal_error_mwh"] for r in rows
                       if (r["year"], r["mode"], r["case"]) == key and r["method"] == m)
                  for m in ("open", "feedback_only", "supervised")]
        axes[0].plot(np.arange(3) + offset, values, color="#CBD3D6", lw=.5, zorder=1)
        for i, (color, marker) in enumerate(((GRAY, "o"), (BLUE, "s"), (TEAL, "^"))):
            axes[0].scatter(i + offset, values[i], color=color, marker=marker, s=16, zorder=3)
    axes[0].set_xticks(range(3), ("开环", "仅SOC反馈", "反馈+监督"), fontsize=8)
    axes[0].set(title="24个健康条件逐条配对", ylabel="最大日末偏差 / MWh", ylim=(-350, 10700))
    for offset, field, color, marker, label in (
        (-.12, "controlled_peak_mw", RED, "o", "最高点"),
        (.12, "controlled_p95_mw", AMBER, "s", "P95")):
        values = [e["v7"]["healthy"][y]["fresh6"]["nominal"]["supervised_minus_feedback"][field]
                  for y in ("2021", "2022")]
        axes[1].scatter(np.arange(2) + offset, values, color=color, marker=marker, s=32, label=label)
    axes[1].set_xticks(range(2), ("2021", "2022"))
    axes[1].set(title="名义条件：监督的性能代价", ylabel="监督 - 仅反馈 / MW", ylim=(-25, 430), xlim=(-.5, 1.5))
    axes[1].axhline(0, color=GRAY, lw=.7)
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.11), ncol=2, fontsize=8)
    for ax in axes:
        legacy.clean(ax)
    save(fig, "fig07_feedback")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    legacy.OUT, legacy.DEST = FIGURES, DEST
    legacy.save = save
    legacy.configure()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    e = collect()
    architecture()
    legacy.timeline()
    fallback()
    prediction(e)
    dispatch(e)
    robustness(e)
    feedback(e)
    legacy.envelope(e)
    legacy.all_conditions(e)
    legacy.fault_replay()
    write_provenance()
    print(json.dumps(SUMMARY, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
