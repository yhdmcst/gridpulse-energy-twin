import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const tmpDir = path.join(workspaceDir, "tmp", "ppt-build");
const draftPath = path.join(tmpDir, "gridpulse-defense-v9-draft.pptx");
const font = "Microsoft YaHei";
const baseline = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/models/reference_baseline_metrics.json"), "utf8"));
const daily = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/scenarios/daily_energy_balance_metrics.json"), "utf8"));
const peakResponse = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/scenarios/peak_response_metrics.json"), "utf8"));
const independent = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/validation/independent_2021_metrics.json"), "utf8"));
const optimality = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/optimality/certificates.json"), "utf8"));
const development = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/development/controller_metrics.json"), "utf8"));
const intervalAudit = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/development/interval_audit_metrics.json"), "utf8"));
const later = JSON.parse(await fs.readFile(path.join(workspaceDir, "outputs/validation/independent_2022_metrics.json"), "utf8"));
function applyPresentationChartFont(chart, fontFamily) {
  for (const style of [chart.legend?.textStyle, chart.dataLabels?.textStyle, chart.xAxis?.textStyle, chart.yAxis?.textStyle]) {
    if (style) style.typeface = fontFamily;
  }
  return chart;
}
const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const C = { ink: "#17302A", teal: "#0F766E", mint: "#DFF4EC", amber: "#E7A73B", paper: "#FAFAF7", gray: "#50625E", line: "#C9D6D0", white: "#FFFFFF", red: "#B45309" };
function box(slide, text, left, top, width, height, size = 22, color = C.ink, bold = false) {
  const shape = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: size, color, bold, autoFit: "shrinkText", verticalAlignment: "middle" };
  return shape;
}
function title(slide, text, subtitle = "") {
  box(slide, text, 70, 43, 1140, 50, 32, C.ink, true);
  if (subtitle) box(slide, subtitle, 72, 94, 1120, 28, 15, C.gray);
  const rule = slide.shapes.add({ geometry: "rect", position: { left: 70, top: 126, width: 1140, height: 3 }, fill: C.teal, line: { fill: "none", width: 0 } });
  return rule;
}
function footer(slide, n) { box(slide, `GridPulse  |  ${n}`, 70, 681, 300, 20, 12, C.gray); }
function note(slide, text) { slide.speakerNotes.textFrame.setText(text); }
async function image(slide, relative, left, top, width, height, alt) {
  const bytes = await fs.readFile(path.join(workspaceDir, relative));
  slide.images.add({ blob: bytes, contentType: "image/png", alt, fit: "contain", position: { left, top, width, height } });
}
function metric(slide, label, value, left, top, width, accent = C.teal) {
  const rect = slide.shapes.add({ geometry: "roundRect", position: { left, top, width, height: 118 }, fill: C.white, line: { fill: C.line, width: 1 } });
  box(slide, value, left + 18, top + 18, width - 36, 44, 30, accent, true);
  box(slide, label, left + 18, top + 67, width - 36, 32, 15, C.gray);
  return rect;
}

// 1. Cover
{
  const s = ppt.slides.add(); s.background.fill = C.paper;
  s.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 1280, height: 720 }, fill: C.paper, line: { fill: "none", width: 0 } });
  s.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 22, height: 720 }, fill: C.teal, line: { fill: "none", width: 0 } });
  box(s, "GridPulse", 82, 152, 740, 78, 58, C.ink, true);
  box(s, "面向新能源波动的净负荷预测与风险感知储能规划", 86, 244, 740, 45, 25, C.teal, true);
  box(s, "科技创新组 | 公开数据回测原型", 88, 310, 520, 32, 19, C.gray);
  box(s, "数据、模型、风险规则与风险感知调控均可由脚本复现", 88, 510, 620, 34, 19, C.ink);
  await image(s, "outputs/figures/02_representative_week.png", 790, 160, 410, 360, "Representative week of load, wind, solar, and net load");
  note(s, "来源：GridPulse 输出图 outputs/figures/02_representative_week.png。项目范围：德国公开历史数据回测。");
}

// 2. Problem
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "工程问题", "新能源出力变化会放大净负荷爬坡与峰值风险");
  box(s, "净负荷 = 总负荷 - 光伏发电 - 风电发电", 92, 184, 1080, 70, 36, C.teal, true);
  box(s, "GridPulse 关注三个问题：净负荷何时抬升、可再生能源何时快速下降、预测偏差何时持续扩大", 92, 298, 1040, 75, 24, C.ink);
  box(s, "系统定位：为园区或区域级能源管理提供可审计的风险研判参考。", 92, 425, 1000, 38, 22, C.gray);
  box(s, "项目不声称实际部署、已实现节省或已确认故障。", 92, 495, 1000, 38, 19, C.red, true); footer(s, 2);
  note(s, "问题定义来自项目设计。净负荷公式使用公开 OPSD 字段构造。风险输出为候选运行风险。");
}

// 3. Data
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "数据与质量证据", "Open Power System Data 德国小时级负荷、光伏与风电数据");
  metric(s, "有效小时记录", "50,400", 72, 165, 235);
  metric(s, "数据覆盖", "2015-2020", 332, 165, 235);
  metric(s, "重复时间戳", "0", 592, 165, 235, C.amber);
  metric(s, "公式误差", "0 MW", 852, 165, 235, C.amber);
  await image(s, "outputs/figures/01_full_period_timeseries.png", 75, 325, 735, 295, "Full period time series");
  box(s, "质量检查", 860, 340, 210, 34, 25, C.ink, true);
  box(s, "时间频率：60 分钟\n缺失时间点：0\n负值：0\n净负荷缺失：105 行\n缺失值未静默填补", 860, 392, 300, 180, 19, C.ink);
  footer(s, 3); note(s, "来源：OPSD Time Series 2020-10-06，DOI 10.25832/time_series/2020-10-06；质量指标来自 outputs/metrics/data-quality-report.json。");
}

// 4. Forecast and planning method
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "因果预测与滚动规划", "从一步预测扩展为直接 1--6 小时规划窗口");
  const steps = [["历史观测", "负荷、光伏、风电"], ["直接预测", "1--6 小时独立模型\n仅使用发布时可得观测"], ["经验区间", "验证残差校准\n预测上界用于风险"], ["滚动决策", "动态 SOC 储备\n每小时重算\n执行第一步"]];
  steps.forEach((item, i) => { const x = 75 + i * 290; s.shapes.add({ geometry: "roundRect", position: { left: x, top: 250, width: 235, height: 190 }, fill: i === 2 ? C.mint : C.white, line: { fill: C.line, width: 1 } }); box(s, item[0], x + 20, 278, 195, 34, 23, C.teal, true); box(s, item[1], x + 20, 334, 195, 74, 18, C.ink); if (i < 3) box(s, "→", x + 244, 316, 36, 36, 28, C.amber, true); });
  box(s, "1--6 小时预测服务于理论调控；直接 24 小时预报仍需发布时可得的天气或新能源预测输入。", 92, 530, 1060, 44, 20, C.red, true); footer(s, 4);
  note(s, "方法来自 src/features.py、src/models.py、src/multihorizon.py 和 docs/risk-aware-control.md。每个时距模型只使用目标时刻前对应时距及更早的观测。");
}

// 5. Results
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "预测结果", "冻结测试期从 2019-08-09 21:00 UTC 开始");
  const chart = s.charts.add("bar", { position: { left: 72, top: 165, width: 520, height: 360 }, categories: ["上一小时", "24h参考", "168h参考", "梯度提升"], series: [{ name: "测试 MAE (MW)", values: ["persistence_1h", "seasonal_24h", "seasonal_168h", "hist_gradient_boosting"].map(key => Number(baseline.metrics[key].mae_mw.toFixed(2))), fill: C.teal }], barOptions: { direction: "column", grouping: "clustered" }, hasLegend: false, dataLabels: { showValue: true, position: "outEnd" } });
  applyPresentationChartFont(chart, font);
  await image(s, "outputs/figures/05_forecast_interval_peak_week.png", 640, 155, 550, 365, "Frozen-test forecast, actual net load and empirical interval");
  box(s, "梯度提升测试 MAE：1,019.87 MW", 90, 560, 470, 32, 22, C.teal, true);
  box(s, "相对上一小时持平基线，MAE 下降 57.20%。", 90, 605, 560, 28, 18, C.gray); footer(s, 5);
  note(s, "指标来源：outputs/models/reference_baseline_metrics.json，各模型对齐相同 10,035 个冻结测试时点。图来源：outputs/figures/05_forecast_interval_peak_week.png。补充基线不改变原模型选择。");
}

// 6. Alerts
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "候选运行风险筛查", "所有阈值仅在验证期校准，再冻结应用到测试期");
  const cards = [["峰值负荷", "预测净负荷 ≥ 55,406 MW"], ["光伏骤降", "小时变化 ≤ -4,243 MW/h"], ["持续偏差", "残差超过 2,937 MW\n且持续 3 小时"]];
  cards.forEach((item, i) => { const x = 78 + i * 382; s.shapes.add({ geometry: "roundRect", position: { left: x, top: 205, width: 320, height: 175 }, fill: C.white, line: { fill: C.line, width: 1 } }); box(s, item[0], x + 25, 238, 265, 34, 25, C.teal, true); box(s, item[1], x + 25, 295, 270, 55, 19, C.ink); });
  box(s, "测试期内 972 行至少触发一类候选风险。每个事件保留触发值、阈值与证据文本。", 90, 455, 1050, 44, 22, C.ink);
  box(s, "持续偏差：第三小时确认，延迟 2h。注入事件检出 25/25，点检出 25/75。", 90, 520, 1060, 38, 19, C.ink);
  box(s, "注入压力测试只验证人工场景下的规则响应，不代表真实故障检测准确率。", 90, 577, 1060, 38, 18, C.red, true); footer(s, 6);
  note(s, "来源：outputs/alerts/alert_evaluation.json 和 docs/day4-risk-alerts.md。风险不等同于确认故障。");
}

// 7. Ablation
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "消融证据", "历史净负荷和新能源滞后信息均带来可测量增益");
  const chart = s.charts.add("bar", { position: { left: 75, top: 165, width: 620, height: 390 }, categories: ["完整模型", "去新能源滞后", "去滚动统计", "仅日历"], series: [{ name: "测试 MAE (MW)", values: [1019.87, 1120.04, 1035.03, 9160.43], fill: C.amber }], barOptions: { direction: "column", grouping: "clustered" }, hasLegend: false, dataLabels: { showValue: true, position: "outEnd" } });
  applyPresentationChartFont(chart, font);
  box(s, "关键观察", 770, 180, 290, 32, 25, C.ink, true);
  box(s, "去除新能源滞后后\nMAE 增加 100.17 MW\n\n去除滚动统计后\nMAE 增加 15.16 MW\n\n仅日历模型显著退化", 770, 240, 340, 245, 22, C.ink);
  box(s, "结论限于当前数据集与冻结测试期。", 770, 540, 350, 28, 17, C.gray); footer(s, 7);
  note(s, "来源：outputs/models/day3_model_metrics.json，Day 6 消融。结果不构成因果或跨区域泛化结论。");
}

// 8. Risk-aware scenario
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "风险感知储能协同调控", "动态 SOC 储备将预测上界和区间宽度转为可解释约束");
  const table = s.tables.add({ rows: 4, columns: 5, left: 60, top: 175, width: 830, height: 230, values: [["容量", "固定储备 P95", "风险感知 P95", "P95 差值", "固定/风险感知循环"], ["10 GWh", "54,200.8", "54,188.3", "-12.5 MW", "0.78 / 0.84"], ["20 GWh", "54,185.0", "54,150.2", "-34.9 MW", "0.81 / 0.86"], ["30 GWh", "54,177.4", "54,077.7", "-99.7 MW", "0.84 / 0.89"]] });
  table.styleOptions = { headerRow: true, bandedRows: true }; for (let c = 0; c < 5; c++) table.getCell(0, c).fill = C.teal;
  table.cells.block({ row: 0, column: 0, rowCount: 4, columnCount: 5 }).assign({ textStyle: { typeface: font, color: C.ink, fontSize: 14 } });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 5 }).assign({ textStyle: { typeface: font, color: C.white, bold: true, fontSize: 14 } });
  box(s, "滚动优化", 940, 180, 200, 32, 25, C.ink, true);
  box(s, "目标：最小化未来 6h\n预测上界的最大净负荷\n\n约束：功率、SOC、效率、\n动态储备与低谷充电", 940, 238, 250, 180, 18, C.ink);
  box(s, "原版对照降低 P95，但吞吐略升，全期最高点未降低。原版不确定性项始终饱和。", 80, 460, 1100, 44, 20, C.red, true);
  box(s, "补充按小时校准：组合 vs 固定储备 P95 -65.85 MW，吞吐 +9,648.40 MWh。\n软储备目标未达 2,445 小时，期末 SOC 7,697.08 MWh（初始 15,000）。", 80, 529, 1100, 64, 19, C.ink);
  box(s, "补充分析为探索性实验。期末 SOC 未约束，不能据此推断长期收益或可靠性。", 80, 617, 1100, 30, 17, C.gray); footer(s, 8);
  note(s, "原版来源：outputs/scenarios/storage_scenario_metrics.json 的 fair_same_information_fixed_reserve_vs_risk_aware。补充来源：outputs/scenarios/reserve_ablation_metrics.json。补充校准只用验证数据，但设计发生在原测试结果已知后，不替代原冻结结果。两组共享预测和约束，仅储备规则不同。期末能量差异和软目标限制解释。");
}

// 9. Matched daily energy accounting
{
  const s = ppt.slides.add(); s.background.fill = C.paper;
  title(s, "原版每日能量恢复的公平性复核", "417 个完整 UTC 日，同一硬 SOC 下限，补电负荷与损耗全部计入");
  const fmt = value => value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const entries = [["fixed_open", "固定，未恢复"], ["dynamic_open", "动态，未恢复"], ["fixed_balanced", "固定，每日恢复"], ["dynamic_balanced", "动态，每日恢复"]];
  const table = s.tables.add({ rows: 5, columns: 4, left: 72, top: 165, width: 1136, height: 240,
    values: [["独立日策略", "P95 (MW)", "最高净负荷 (MW)", "吞吐 (MWh)"], ...entries.map(([key, label]) => {
      const item = daily.experiments[key];
      return [label, fmt(item.controlled_p95_mw), fmt(item.controlled_peak_mw), fmt(item.throughput_mwh)];
    })] });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < 4; c++) table.getCell(0, c).fill = C.teal;
  table.cells.block({ row: 0, column: 0, rowCount: 5, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.ink, fontSize: 17 } });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.white, bold: true, fontSize: 17 } });
  box(s, "每日恢复：动态 P95 低 15.09 MW，吞吐少 59,996.65 MWh，最高点仍未降低", 80, 436, 1120, 44, 21, C.teal, true);
  box(s, "初末 SOC 均为 15,000 MWh，硬下限 4,500 MWh，违规与日末误差均为 0。\n固定策略最高点更低。动态软目标仍有 17 小时未达，结论只支持部分指标取舍。", 80, 495, 1120, 65, 19, C.ink);
  box(s, "探索性日实验。恢复模式另限制 SOC 上限为 50%，允许任意小时补电。\n未恢复模式每天重置电量，仅为对照。模式差异不能只归因于日末等式。", 80, 590, 1120, 58, 17, C.gray);
  footer(s, 9);
  note(s, "来源：outputs/scenarios/daily_energy_balance_metrics.json 及 daily_*_timeseries.csv。四组相同 10,008 小时，排除首尾两个不完整日。参数：30 GWh，充放电均 1,250 MW，单向效率 0.92，共同硬下限 15%。恢复模式 SOC 上限 50%，日末 SOC=初始 SOC，全天可补电。动态与固定转换损耗分别为 29,187.52 和 34,178.57 MWh。规划只读取发布预测、已知日历和额定功率。属于原测试结果已知后的探索性分析，不能与原连续实验直接比较或宣称真实网络可行性。");
}

// 10. Extreme-peak response
{
  const s = ppt.slides.add(); s.background.fill = C.paper;
  title(s, "极端峰值响应与循环代价", "点预测定位峰值，验证期 q99 触发软储备释放，保持硬下限与每日恢复");
  const fmt = value => value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const entries = [["dynamic_upper", "原动态"], ["dynamic_release_only", "仅软储备释放"], ["dynamic_point_only", "仅点预测目标"], ["dynamic_peak_response", "新动态组合"], ["fixed_peak_response", "新固定组合"]];
  const table = s.tables.add({ rows: 6, columns: 4, left: 72, top: 165, width: 1136, height: 270,
    values: [["417 日测试配置", "最高净负荷 (MW)", "P95 (MW)", "吞吐 (MWh)"], ...entries.map(([key, label]) => {
      const item = peakResponse.periods.test.experiments[key];
      return [label, fmt(item.controlled_peak_mw), fmt(item.controlled_p95_mw), fmt(item.throughput_mwh)];
    })] });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < 4; c++) table.getCell(0, c).fill = C.teal;
  table.cells.block({ row: 0, column: 0, rowCount: 6, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.ink, fontSize: 17 } });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.white, bold: true, fontSize: 17 } });
  box(s, "新动态全期最高峰降低 645.85 MW，P95 比原动态低 170.10 MW", 80, 455, 1120, 40, 22, C.teal, true);
  box(s, "原峰小时放电 1,250 MW，最高点转移。吞吐与损耗增加，193 小时软目标未达。\n新固定最高峰相同、P95 更低，新动态吞吐少 200,541.46 MWh。", 80, 511, 1120, 68, 19, C.ink);
  box(s, "验证期内部诊断削峰 644.43 MW。设计发生在旧测试峰值已知后，仍为探索性证据。\n阈值仅用验证预测校准，硬 SOC 违规和最大日末误差为 0。", 80, 603, 1120, 54, 17, C.gray);
  footer(s, 10);
  note(s, "来源：outputs/scenarios/peak_response_metrics.json。完整六组含原固定的验证/测试数据及轨迹已保留。发布窗口点预测 >= 验证期一小时点预测 q99（63,183.08 MW）时释放软下限至硬下限 15%。点目标替代上界最小最大目标，区间仍决定储备与放电触发。日末恢复、50% SOC 上限和全天补电假设不变。新动态损耗 54,158.56 MWh、吞吐 651,031.03 MWh；原动态损耗 29,187.52 MWh、吞吐 350,858.30 MWh。全期最高点移至 2019-11-25 14:00 UTC。无独立确认、网络可行性、寿命或经济收益结论。");
}

// 11. Frozen later-year temporal holdout
{
  const s = ppt.slides.add(); s.background.fill = C.paper;
  title(s, "2021 年冻结时间留出验证", "Energy-Charts.info 德国全年 365 日，下载前锁定协议，模型与阈值不再调整");
  const fmt = value => value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const entries = [["fixed_upper", "原固定"], ["dynamic_upper", "原动态"], ["dynamic_release_only", "仅软储备释放"], ["dynamic_point_only", "仅点预测目标"], ["dynamic_peak_response", "新动态组合"], ["fixed_peak_response", "新固定组合"]];
  const table = s.tables.add({ rows: 7, columns: 4, left: 72, top: 165, width: 1136, height: 294,
    values: [["2021 冻结配置", "最高净负荷 (MW)", "P95 (MW)", "吞吐 (MWh)"], ...entries.map(([key,label]) => {
      const item = independent.experiments[key];
      return [label, fmt(item.controlled_peak_mw), fmt(item.controlled_p95_mw), fmt(item.throughput_mwh)];
    })] });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let c = 0; c < 4; c++) table.getCell(0,c).fill = C.teal;
  table.cells.block({ row: 0, column: 0, rowCount: 7, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.ink, fontSize: 17 } });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({ textStyle: { typeface: font, color: C.white, bold: true, fontSize: 17 } });
  const main = independent.comparisons.dynamic_peak_response_versus_dynamic_upper;
  const ci = main.uncertainty.mean_daily_peak_reduction_mw;
  box(s, `全年最高点降低 ${fmt(main.full_period_peak_reduction_mw)} MW，平均日最高点降低 ${fmt(ci.estimate)} MW`, 80, 480, 1120, 34, 21, C.teal, true);
  box(s, `平均日削峰 95% 区间 [${fmt(ci.lower_95)}, ${fmt(ci.upper_95)}] MW，7 天配对区块、2,000 次。\n全年 P95 升高 ${fmt(-main.p95_reduction_mw)} MW，Q3 最高点变差 230.18 MW，吞吐与损耗增加。`, 80, 529, 1120, 60, 18, C.ink);
  box(s, "证据支持限定的后续年份极端削峰，P95 优势未复现，新固定也达到相同最高点。\n同国修订历史数据，额外发布延迟、网络、价格和退化仍需核验。", 80, 610, 1120, 50, 17, C.gray);
  footer(s,11);
  note(s, "来源：Energy-Charts.info / Fraunhofer ISE public_power v2，CC BY 4.0，https://api.energy-charts.info/ 。协议 outputs/validation/independent_2021_protocol.json 在全年下载前冻结；完整六组输出 independent_2021_metrics.json，无原数据时间重叠。协议前短样本核查涉及 1 月 1--3 日，仅检查模式并展示首行，未预测/调控/拟合；排除三日的 362 日事后敏感性最大削峰仍为 1,250 MW，平均日削峰 237.30 MW，区间 [183.84,290.45]。全年原最大 70,449.90，新最大 69,199.90 MW，均在 2021-11-10 16:00 UTC。平均日削峰 7 天区块 95% 区间 [184.73,293.93]，3/14 天敏感性也为正；P95 削减区间 [-359.59,78.15] 覆盖零，不支持 P95 优势。改善/持平/变差日为 132/222/11。新动态相比原动态吞吐多 592,789.18 MWh、损耗多 49,313.48 MWh；软目标缺口 444 小时，硬下限和日末误差为零。新固定最大相同、P95 低 373.98 MW，吞吐多 260,877.83 MWh。源时间戳标记区间开始，发布按最新观测小时结束处理，额外发布延迟未知。仅为同国后续历史仿真，非现场/跨国验证，不支持经济收益或可靠性承诺。");
}

// 12. Finite-instance certificate
{
  const s = ppt.slides.add(); s.background.fill = C.paper;
  title(s, "有限实例的最优性证明", "固定物理模型，只优化全年最高点，随后最小化离线吞吐");
  const fmt = v => v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  box(s,"任何可行最高点 ≥ 原负荷最高点 − 最大放电功率",80,161,1120,47,27,C.teal,true);
  const table = s.tables.add({ rows:3,columns:4,left:72,top:235,width:1136,height:180,
    values:[["实例","物理最优最高点 MW","现有新动态差距 MW","离线最低吞吐 MWh"],
      ...[["independent_2021","2021 年"],["exploratory_test","旧探索性测试"]].map(([key,label])=>{
        const c=optimality.cases[key]; return [label,fmt(c.offline_benchmark.period_optimum_peak_mw),fmt(c.controller_peak_gap_mw),fmt(c.offline_benchmark.minimum_throughput_at_optimum_peak_mwh)];})]});
  table.styleOptions = {headerRow:true,bandedRows:true};
  table.cells.block({row:0,column:0,rowCount:3,columnCount:4}).assign({textStyle:{typeface:font,color:C.ink,fontSize:19}});
  for(let c=0;c<4;c++) table.getCell(0,c).fill=C.teal;
  table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:font,color:C.white,bold:true,fontSize:19}});
  box(s,"2021 年原最高点 70,449.90 − 1,250 = 69,199.90 MW，现有策略达到下界",80,447,1120,38,23,C.teal,true);
  box(s,"两组离线解均有精确有理数下界达到证明，1,564 个 LP 原始与对偶见证交叉核验。\n新固定策略也达到相同最高点，数学证书不证明动态储备独有优势。",80,506,1120,65,20,C.ink);
  box(s,"离线解预知全年真实负荷。最低吞吐、P95 和所有未来在线场景不能混为同一最优。",80,613,1120,36,18,C.red,true);
  footer(s,12); note(s,"来源：outputs/optimality/certificates.json、docs/global-optimality.md。物理约束：容量30000MWh，日初末SOC15000MWh，硬下限4500MWh，操作上限15000MWh，功率各1250MW，单向效率0.92。精确有理数证书针对CSV十进制原值，控制轨迹另按1e-5物理残差容差核验。离线调度去掉预测门槛和软储备，是更宽物理可行集。旧测试现有策略仍有604.15MW差距。标准线性规划和强对偶不作为原创算法主张。");
}

// 13. Transparent controller development
{
  const s=ppt.slides.add(); s.background.fill=C.paper;
  title(s,"循环代价与运行边界","新开发单独复核，原冻结证据保留");
  const fmt=v=>v.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
  const entries=[["frozen_reference","原冻结控制器"],["two_stage_0mw","两阶段 0 MW"],["two_stage_250mw","两阶段 250 MW"],["two_stage_500mw","两阶段 500 MW"]];
  const table=s.tables.add({rows:5,columns:4,left:72,top:165,width:1136,height:250,
    values:[["2021 已知年开发复核","最高点 MW","P95 MW","吞吐 MWh"],...entries.map(([key,label])=>{
      const c=development.periods.known_2021.experiments[key]; return [label,fmt(c.controlled_peak_mw),fmt(c.controlled_p95_mw),fmt(c.throughput_mwh)];})]});
  table.styleOptions={headerRow:true,bandedRows:true};
  table.cells.block({row:0,column:0,rowCount:5,columnCount:4}).assign({textStyle:{typeface:font,color:C.ink,fontSize:19}});
  for(let c=0;c<4;c++)table.getCell(0,c).fill=C.teal;
  table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:font,color:C.white,bold:true,fontSize:19}});
  box(s,"原验证期筛选未通过，保留原控制器。循环减少伴随最高点或 P95 变差。",80,443,1120,43,22,C.red,true);
  box(s,"四个固定季度周完成 68 组运行压力复核，含预测滞后、缺失和跨日运行。\n已求解轨迹物理约束核验通过，效果随运行条件变化。",80,506,1120,64,20,C.ink);
  box(s,`一小时区间覆盖率：固定 ${intervalAudit.horizon_1_overall.fixed_covered.toFixed(2)}%，滚动 ${intervalAudit.horizon_1_overall.rolling_covered.toFixed(2)}%。\n滚动校准仅用发布时已观测残差，尚未反馈控制器，属于已知年份事后分析。`,80,601,1120,61,18,C.gray);
  footer(s,13); note(s,"来源：outputs/development/controller_metrics.json、operating_stress_metrics.json、interval_audit_metrics.json。0/100/250/500MW窗口容许差先在原验证期比较。准入：峰值不高于原冻结新动态+1MW、P95不高于+50MW，合格配置按吞吐选择。四组均不合格，保留原控制器。2021年份早已公开，不再称独立验证。压力情景每季度固定7天，共68条可行轨迹；连续模式仅新版候选适用，原独立日对照不适用。覆盖率须结合宽度和区间评分解释，无概率或部署保证。");
}

// 14. Separately locked 2022 replication
{
  const s=ppt.slides.add(); s.background.fill=C.paper;
  title(s,"2022 年锁定复验与年度差异","下载前锁定原六组策略，365 个共享日，未重训或调参");
  const fmt=v=>v.toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
  const entries=[["fixed_upper","原固定"],["dynamic_upper","原动态"],["dynamic_release_only","仅储备释放"],
    ["dynamic_point_only","仅点预测"],["dynamic_peak_response","新动态组合"],["fixed_peak_response","新固定组合"]];
  const table=s.tables.add({rows:7,columns:3,left:72,top:165,width:1136,height:290,
    values:[["2022 冻结配置","最高净负荷 MW","P95 MW"],...entries.map(([key,label])=>{
      const c=later.experiments[key];return[label,fmt(c.controlled_peak_mw),fmt(c.controlled_p95_mw)];})]});
  table.styleOptions={headerRow:true,bandedRows:true};
  table.cells.block({row:0,column:0,rowCount:7,columnCount:3}).assign({textStyle:{typeface:font,color:C.ink,fontSize:18}});
  for(let c=0;c<3;c++)table.getCell(0,c).fill=C.teal;
  table.cells.block({row:0,column:0,rowCount:1,columnCount:3}).assign({textStyle:{typeface:font,color:C.white,bold:true,fontSize:18}});
  box(s,"全年最高点降低 1,250 MW，平均日降低 165.83 MW，区间 [106.43,227.73]",80,478,1120,39,21,C.teal,true);
  box(s,"全年 P95 低 145.49 MW，但改善区间覆盖零，四个季度 P95 均变差。\n相对原动态吞吐多 300,478.17 MWh，一小时区间覆盖率降至 82.03%。",80,535,1120,61,19,C.ink);
  box(s,"两年均复现极端削峰。尾部分位优势仍不稳定，循环代价与区间退化需保留。",80,617,1120,35,18,C.red,true);
  footer(s,14);note(s,"来源：outputs/validation/independent_2022_protocol.json、independent_2022_metrics.json 和 docs/later-year-validation.md。2021 及开发结果已知后单独锁定原六组策略，首次下载2022年度前保存模型、代码与阈值哈希。协议是本地记录，无外部时间戳见证。平均日7天配对区块95%区间[106.43,227.73]，P95削减区间[-238.78,291.67]覆盖零。季度P95均变差，全年分位不能按季度平均。新动态吞吐787162.14MWh、损耗65483.16MWh，软目标未达244小时。新固定最高点相同，新动态P95低29.20MW、吞吐少133032.13MWh，平均日最高点反而高61.89MW。一小时MAE1173.45MW，持平基线2465.62MW。无重训、重校准或候选选型。未新增2022精确最优性证书，仍为同国修订历史仿真。");
}

// 15. Product and next gate
{
  const s = ppt.slides.add(); s.background.fill = C.paper; title(s, "系统原型与下一研究关口", "Streamlit MVP 已串联数据、预测、风险证据和风险感知调控");
  box(s, "当前原型", 82, 175, 250, 35, 27, C.teal, true);
  box(s, "数据概览\n预测回测\n风险事件详情\n特征解释\n理论调度建议\n风险感知调控对照", 85, 225, 290, 220, 21, C.ink);
  box(s, "已验证证据", 440, 175, 250, 35, 27, C.teal, true);
  box(s, "滚动回测：5/5 窗口胜基线\nMAE 均值：837.30 MW\n测试区间覆盖率：86.92%\n区间预警进入人工复核队列", 440, 235, 360, 185, 21, C.ink);
  box(s, "下一关口", 850, 175, 250, 35, 27, C.teal, true);
  box(s, "构造真正可在发布时获取的 24 小时输入\n\n接入电价、退化和网络约束\n\n增加跨区域验证", 850, 235, 340, 185, 20, C.ink);
  box(s, "可复现路径", 82, 510, 210, 32, 23, C.ink, true);
  box(s, "下载数据 → 质量检查 → 基线 → 模型 → 区间 → 风险 → 情景 → 仪表盘", 82, 558, 1080, 34, 19, C.gray);
  box(s, "提交前：核查匿名要求、数据许可与每一项结果的脚本可再生性。", 82, 620, 1000, 28, 18, C.red, true); footer(s, 15);
  note(s, "系统功能来源：app.py；滚动回测来源：outputs/models/rolling_backtest_metrics.json；风险感知调控来源：outputs/scenarios/storage_scenario_metrics.json。下一关口为研究计划，尚未实施。");
}

await fs.mkdir(tmpDir, { recursive: true });
await (await PresentationFile.exportPptx(ppt)).save(draftPath);
console.log(draftPath);
