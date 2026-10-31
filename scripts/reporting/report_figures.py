"""Publication-style plots and local editable diagrams for the revised report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from report_evidence import CASES, DEST, ROOT, SOURCES, SPLITS, WORKSPACE, collect, write_provenance

sys.path.insert(0, str(WORKSPACE / ".report-deps"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from matplotlib.text import Text

OUT = DEST / "图表"
TEAL, BLUE, AMBER, RED = "#087F8C", "#386CB0", "#9A650C", "#B44A48"
INK, GRAY, PALE = "#202A30", "#607079", "#EFF3F4"
GEOMETRY = {}


def configure():
    for f in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "SimHei", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BBC5CA", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": GRAY, "ytick.color": GRAY,
        "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipped = []
    inactive_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    inactive_ticks.update((id(tick.label1), id(tick.label2)))
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and id(text) not in inactive_ticks:
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                clipped.append(text.get_text())
    GEOMETRY[name] = {"text_outside_canvas": clipped, "size_inches": list(fig.get_size_inches())}
    if clipped:
        raise ValueError(f"Clipped figure labels in {name}: {clipped}")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def clean(ax, axis="y"):
    ax.grid(axis=axis, color="#E6EBEE", linewidth=.6)
    ax.set_axisbelow(True)


def node(ax, x, y, w, h, title, detail="", color=TEAL):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.1))
    ax.add_patch(Rectangle((x, y + h - .035), w, .035, facecolor=color, linewidth=0))
    ax.text(x + w / 2, y + h * .64, title, ha="center", va="center", fontsize=10, color=color)
    if detail:
        ax.text(x + w / 2, y + h * .29, detail, ha="center", va="center", fontsize=8.3, linespacing=1.5)


def arrow(ax, start, end, text="", color=GRAY, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.15, color=color, connectionstyle=connectionstyle))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .025,
                text, fontsize=8, color=color, ha="center", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=.02, right=.98, top=.96, bottom=.05)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    node(ax, .01, .67, .24, .25, "历史观测与数据时钟", "负荷 / 风 / 光\nUTC 对齐与可用性", BLUE)
    node(ax, .37, .67, .25, .25, "AI 多时距预测", "h1-h6 + 验证期区间\n原始 / 一致性消融", BLUE)
    node(ax, .75, .67, .24, .25, "滚动规划", "尾部目标 + 参考约束\n声明的保守恢复下界", TEAL)
    node(ax, .75, .18, .24, .25, "门控与恢复监督", "过期 / 不可用 / 失败\n恢复充电或空闲", AMBER)
    node(ax, .37, .18, .25, .25, "物理执行仿真", "限功率 / 效率 / SOC\n请求动作与实际动作", TEAL)
    node(ax, .01, .18, .24, .25, "逐小时审计与回放", "能量守恒 / 跨日连续\n失败记录 / 工件哈希", GRAY)
    arrow(ax, (.25, .795), (.37, .795))
    arrow(ax, (.62, .795), (.75, .795))
    arrow(ax, (.87, .67), (.87, .43), "首小时请求")
    arrow(ax, (.75, .305), (.62, .305), "动作")
    arrow(ax, (.37, .305), (.25, .305))
    arrow(ax, (.495, .43), (.75, .67), "每小时仿真真实 SOC", TEAL)
    ax.text(.5, .04, "信息边界：实际功率/效率只进入执行与核算，不输入控制器；SOC 无噪声、无延迟。",
            ha="center", fontsize=8.2, color=RED)
    save(fig, "fig01_architecture")
    (OUT / "fig01_architecture.mmd").write_text(
        'flowchart LR\n  A["历史观测与数据时钟"] --> B["AI多时距预测 h1-h6"]\n'
        '  B --> C["滚动规划：保守恢复下界"]\n  C --> D["门控与恢复监督"]\n'
        '  D --> E["物理执行仿真"]\n  E --> F["逐小时审计与回放"]\n'
        '  E -- "每小时仿真真实SOC" --> C\n', encoding="utf-8")


def flow():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.03)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    node(ax, .02, .68, .27, .24, "读取当前 SOC 与预测包", "校验时钟、区间与有限值", BLUE)
    node(ax, .38, .68, .26, .24, "数据与服务门控", "观测年龄 ≤ 2 h\n服务可用 / 参考有效", AMBER)
    node(ax, .73, .68, .25, .24, "主优化求解", "仅使用可用信息\n保留失败状态", TEAL)
    node(ax, .38, .25, .26, .24, "确定性降级", "停止沿用旧放电\n按当前 SOC 恢复或空闲", AMBER)
    node(ax, .73, .25, .25, .24, "监督与实际执行", "检查恢复下界\n物理限幅并记录", TEAL)
    arrow(ax, (.29, .8), (.38, .8))
    arrow(ax, (.64, .8), (.73, .8), "通过")
    arrow(ax, (.51, .68), (.51, .49), "拒绝")
    arrow(ax, (.85, .68), (.64, .49), "求解失败", RED)
    arrow(ax, (.85, .68), (.85, .49), "成功")
    arrow(ax, (.64, .37), (.73, .37))
    ax.text(.17, .4, "门控拒绝时\n优化器调用数 = 0", ha="center", fontsize=10, color=AMBER)
    ax.text(.5, .10, "无有效 SOC 遥测：当前实现拒绝执行；尚未实现带噪声/延迟 SOC 的安全估计。",
            ha="center", fontsize=8.5, color=RED)
    save(fig, "fig02_fallback")
    (OUT / "fig02_fallback.mmd").write_text(
        'flowchart TD\n A["读取SOC与预测包"] --> B{"数据、时钟与服务有效？"}\n'
        ' B -- "是" --> C["主优化"]\n B -- "否：不调用优化器" --> D["当前SOC恢复充电或空闲"]\n'
        ' C -- "失败" --> D\n C -- "成功" --> E["恢复监督与物理执行"]\n'
        ' D --> E\n E --> F["逐小时记录请求、实际动作与失败原因"]\n', encoding="utf-8")


def timeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.subplots_adjust(left=.03, right=.98, top=.94, bottom=.08)
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    bands = [(.1, 4.3, TEAL, "训练\n早于 2019-01-12 20:00 UTC"),
             (4.5, 2.6, BLUE, "验证\n截至 2019-08-09 20:00 UTC"),
             (7.2, 2.6, AMBER, "原 test\n随后至 OPSD 快照末尾")]
    for x, w, c, t in bands:
        ax.add_patch(Rectangle((x, 2.8), w, .8, facecolor=c, alpha=.11, edgecolor=c))
        ax.text(x + w / 2, 3.2, t, ha="center", va="center", fontsize=9)
    ax.text(.1, 2.45, "2021 / 2022：后续年份评价，但已多轮查看，不再是新融合的确认性留出集。",
            color=RED, fontsize=9)
    ax.text(.1, 1.85, "延迟对齐示例：当前 t，只收到起点 t-1 的旧包", fontsize=10, color=INK)
    labels = ("h1\n已过去", "h2\n对应当前", "h3", "h4", "h5", "h6")
    for i, label in enumerate(labels):
        x = 1.6 + i * 1.18
        ax.add_patch(Rectangle((x, .55), 1.02, .85, facecolor=PALE if i == 0 else "#E7F2F3",
                               edgecolor="#B5C7CC"))
        ax.text(x + .51, .97, label, ha="center", va="center", color=GRAY if i == 0 else TEAL)
    ax.text(.1, .12, "delayed1 保留旧包 h2-h6 的真实 5 个时距；误差区间仍按原始时距校准，不补造 h7。",
            fontsize=8.8)
    save(fig, "fig03_time_boundary")


def prediction(e):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharey=True, layout="constrained")
    methods = ("HGB", "Data MLP", "Consistency MLP", "Consistency + projection")
    labels = ("HGB", "数据\nMLP", "一致性\nMLP", "一致性\n+投影")
    for ax, split in zip(axes, SPLITS):
        for i, (method, color) in enumerate(zip(methods, (GRAY, BLUE, TEAL, AMBER))):
            values = [r["mae_mw"] for r in e["prediction"] if r["split"] == split and r["method"] == method]
            mean = np.mean(values)
            ax.bar(i, mean, color=color, alpha=.20, edgecolor=color, width=.66)
            offsets = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0]
            for x, value, mark in zip(offsets, values, ("o", "s", "^")):
                ax.scatter(i + x, value, color=color, marker=mark, s=18, zorder=4)
            ax.text(i, mean + 30, f"{mean:.1f}", ha="center", fontsize=8)
        ax.set_title("原 test" if split == "test" else split)
        ax.set_xticks(range(4), labels, fontsize=7.8)
        ax.set_ylim(0, 1030)
        clean(ax)
    axes[0].set_ylabel("净负荷 MAE / MW（越低越好）")
    save(fig, "fig04_prediction")


def horizon_tail(e):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), layout="constrained")
    metrics = ("controlled_peak_mw", "controlled_p95_mw")
    for ax, metric, title in zip(axes[:2], metrics, ("六小时减一小时：最高点", "六小时减一小时：P95")):
        vals = []
        for split in SPLITS:
            rows = {r["window"]: r for r in e["horizon"] if r["split"] == split}
            vals.append(rows["w6"][metric] - rows["w1"][metric])
        bars = ax.bar(range(3), vals, color=[TEAL if v < 0 else RED for v in vals], width=.6)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(3), ("test", "2021", "2022"))
        ax.axhline(0, color=GRAY, linewidth=.7)
        ax.margins(y=.25)
        ax.set_ylabel("差值 / MW")
        clean(ax)
    ax = axes[2]
    vals = [r["p95_delta_mw"] for r in e["tail"]]
    bars = ax.bar(range(3), vals, color=TEAL, alpha=.8, width=.6)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("tail_guard 减同窗 legacy：P95", fontsize=8.7)
    ax.set_xticks(range(3), ("test", "2021", "2022"))
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.margins(y=.28)
    ax.set_ylabel("差值 / MW")
    clean(ax)
    save(fig, "fig05_dispatch_tradeoffs")


def robustness(e):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    ax = axes[0]
    modes = ("fresh6", "fresh5", "fresh3", "delayed1", "delayed3")
    for year, color, marker in (("2021", BLUE, "o"), ("2022", TEAL, "s")):
        vals = [next(r["p95_delta_mw"] for r in e["delay"] if r["year"] == year and r["mode"] == m)
                for m in modes]
        ax.plot(range(5), vals, marker=marker, color=color, label=year)
    ax.set_xticks(range(5), ("新6h", "新5h", "新3h", "延1h", "延3h"))
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.set_title("同延迟条件：tail_guard 减 legacy")
    ax.set_ylabel("P95 差 / MW")
    ax.legend(fontsize=8)
    clean(ax)
    ax = axes[1]
    cases = CASES[1:5]
    for year, offset, color in (("2021", -.18, BLUE), ("2022", .18, TEAL)):
        vals = [next(r["terminal_error_mwh"] for r in e["mismatch"]
                     if r["year"] == year and r["mode"] == "fresh6" and r["case"] == c) for c in cases]
        ax.bar(np.arange(4) + offset, vals, width=.32, color=color, alpha=.8, label=year)
    ax.set_xticks(range(4), ("功率90%", "功率70%", "效率.90", "效率.85"), fontsize=8)
    ax.set_title("错误模型开环回放（fresh6）")
    ax.set_ylabel("最大日末电量偏差 / MWh")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    clean(ax)
    save(fig, "fig06_robustness")


def feedback(e):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    ax = axes[0]
    groups = (("open", "开环", GRAY), ("feedback_only", "仅SOC反馈", BLUE),
              ("supervised", "反馈+恢复监督", TEAL))
    for i, (method, label, color) in enumerate(groups):
        vals = [r["terminal_error_mwh"] for r in e["feedback"] if r["method"] == method]
        # Healthy conditions are paired; repeated zeros are offset deterministically.
        x = i + np.linspace(-.18, .18, len(vals))
        ax.scatter(x, vals, color=color, s=17, alpha=.8, marker=("o", "s", "^")[i])
    ax.set_xticks(range(3), [g[1] for g in groups], fontsize=8)
    ax.set_ylabel("最大日末电量偏差 / MWh")
    ax.set_title("全部24个健康条件，不含故障/范围外")
    ax.set_ylim(-300, 11000)
    clean(ax)
    ax = axes[1]
    x = np.arange(2)
    for offset, field, color, label in (
        (-.18, "controlled_peak_mw", RED, "最高点差"),
        (.18, "controlled_p95_mw", AMBER, "P95差")):
        vals = [e["v7"]["healthy"][y]["fresh6"]["nominal"]["supervised_minus_feedback"][field]
                for y in ("2021", "2022")]
        bars = ax.bar(x + offset, vals, width=.32, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x, ("2021", "2022"))
    ax.set_ylabel("监督减仅反馈 / MW")
    ax.set_title("名义 fresh6：恢复能力的性能代价")
    ax.set_ylim(0, 425)
    ax.legend(fontsize=8)
    clean(ax)
    save(fig, "fig07_feedback")


def envelope(e):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    labels, p95, peak = [], [], []
    for case in ("nominal", "combined90"):
        for year in ("2021", "2022"):
            d = e["v8"]["healthy"][year]["fresh6"][case]["envelope_tail"]["minus_v7"]
            labels.append(f"{'名义' if case == 'nominal' else '组合误差'}\n{year}")
            p95.append(d["controlled_p95_mw"])
            peak.append(d["controlled_peak_mw"])
    for ax, values, title in zip(axes, (p95, peak), ("P95：改善与退步并存", "最高点：四条件均退步")):
        bars = ax.bar(range(4), values, color=[TEAL if v <= 0 else RED for v in values], width=.62)
        ax.bar_label(bars, fmt="%+.1f", padding=3, fontsize=8)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("候选减第七轮监督 / MW")
        ax.axhline(0, color=GRAY, linewidth=.7)
        ax.margins(y=.32)
        clean(ax)
    save(fig, "fig08_envelope_tradeoffs")


def all_conditions(e):
    rows = e["envelope"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.0),
                             gridspec_kw={"width_ratios": (1.1, 1.2, .95)}, layout="constrained")
    labels = []
    for r in rows:
        case = dict(zip(CASES, ("名义", "功率90%", "功率70%", "效率.90", "效率.85", "组合90%")))[r["case"]]
        extra = "" if r["event"] == "healthy" else {"stale_burst": "/过期", "solver_outage": "/中断",
                                                  "combined_fault": "/组合故障"}[r["event"]]
        labels.append(f"{r['year'][-2:]} {'新6h' if r['mode'] == 'fresh6' else '延1h'} {case}{extra}")
    for ax, field, title in zip(axes[:2], ("controlled_peak_mw", "controlled_p95_mw"), ("最高点差 / MW", "P95差 / MW")):
        values = [r[field] for r in rows]
        ax.scatter(values, np.arange(30), c=[TEAL if v <= 0 else RED for v in values], s=16, marker="o")
        ax.axvline(0, color=GRAY, linewidth=.7)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(np.arange(30), labels if ax is axes[0] else [])
        ax.tick_params(axis="y", labelsize=7.6, length=0)
        ax.set_ylim(29.7, -.7)
        clean(ax, "x")
    matrix = np.array([[int(r[f"guard_{k}"]) for k in ("terminal", "peak", "p95", "throughput", "loss")]
                       for r in rows])
    axes[2].imshow(matrix, aspect="auto", interpolation="none",
                   cmap=matplotlib.colors.ListedColormap(["#ECC9C8", "#D2E9E5"]), vmin=0, vmax=1)
    axes[2].set_xticks(range(5), ("恢复", "峰值", "P95", "吞吐", "损耗"), rotation=90, fontsize=8)
    axes[2].set_yticks([])
    axes[2].set_title("预算通过 = ·  未过 = ×", fontsize=8)
    for row in range(30):
        for col in range(5):
            axes[2].text(col, row, "." if matrix[row, col] else "x",
                         va="center", ha="center", fontsize=8, color=TEAL if matrix[row, col] else RED)
    save(fig, "fig09_all_conditions")


def fault_replay():
    relative = "outputs/fusion_v8/evaluation/2022/fresh6/combined90/combined_fault/envelope_tail.csv"
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    day = frame.iloc[:24].copy()
    assert day["timestamp"].dt.date.nunique() == 1
    assert int(day["fallback_active"].sum()) == 5
    (DEST / "图表数据" / "fig10_fault_replay.csv").write_text(day.to_csv(index=False), encoding="utf-8-sig")
    x = day["timestamp"].dt.hour
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, layout="constrained")
    axes[0].plot(x, day["storage_soc_mwh"], color=TEAL, marker="o", markersize=3, label="实际电量")
    axes[0].plot(x, day["safe_floor_next_mwh"], color=AMBER, linestyle="--", label="下一步恢复下界")
    axes[0].set_ylabel("电量 / MWh")
    axes[0].set_ylim(4000, 15700)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=2, borderaxespad=0)
    axes[1].step(x, day["charge_mw"], where="mid", color=BLUE, label="实际充电")
    axes[1].step(x, -day["discharge_mw"], where="mid", color=TEAL, linestyle="--", label="实际放电(负)")
    axes[1].set_ylim(-1000, 1000)
    axes[1].set_ylabel("功率 / MW")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=2, borderaxespad=0)
    axes[2].step(x, day["primary_solver_attempted"].astype(int), where="mid", color=BLUE, label="主优化调用")
    axes[2].step(x, day["fallback_active"].astype(int), where="mid", color=RED, linestyle="--", label="降级")
    axes[2].set_yticks((0, 1))
    axes[2].set_ylim(-.15, 1.25)
    axes[2].set_ylabel("状态")
    axes[2].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=2, borderaxespad=0)
    axes[2].set_xlabel("2022-01-02 / UTC 小时（首个计划注入日，非最优日筛选）")
    axes[2].set_xticks(range(0, 24, 2))
    for ax in axes:
        ax.axvspan(9.5, 10.5, color=RED, alpha=.09)
        ax.axvspan(13.5, 17.5, color=AMBER, alpha=.12)
        ax.set_xlim(-.5, 23.5)
        clean(ax)
    save(fig, "fig10_fault_replay")


def main():
    configure()
    evidence = collect()
    architecture()
    flow()
    timeline()
    prediction(evidence)
    horizon_tail(evidence)
    robustness(evidence)
    feedback(evidence)
    envelope(evidence)
    all_conditions(evidence)
    fault_replay()
    (DEST / "图表几何核对.json").write_text(
        json.dumps(GEOMETRY, ensure_ascii=False, indent=2), encoding="utf-8")
    write_provenance()
    print(json.dumps({"figures": 10, "directory": str(OUT), "source_files": len(SOURCES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
