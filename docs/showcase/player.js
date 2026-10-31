/* The animation interpolates frozen hourly evidence; it never runs a controller. */
(() => {
  "use strict";
  const data = window.GRIDPULSE_TRACE;
  const $ = (id) => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const parameters = new URLSearchParams(location.search);
  let language = parameters.get("lang") === "en" ? "en" : "zh";
  let theme = parameters.get("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  let time = 0;
  let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  let speed = 1;
  let last = null;
  let activeChapter = -1;
  const duration = data.duration_seconds;
  const dictionary = {
    zh: {
      offline: "离线冻结证据", selectedDay: "选取的一日 · UTC", caseName: "组合误差 + 注入故障",
      forecast: "预测", planning: "规划", fallback: "回退", verify: "核对", netLoad: "净负荷 / GW",
      actual: "实际", controlled: "V8 执行", interval: "预测区间", soc: "储能电量 / GWh",
      energy: "实际电量", floor: "恢复下界", interpolation: "曲线来自逐小时工件；帧间插值仅用于动画显示。",
      state: "执行状态", socValue: "当前电量", power: "净请求功率", age: "观测年龄", model: "保守规划模型",
      solver: "主优化器调用", dayEnd: "该日末电量", notAll: "恢复 ≠ 全指标占优",
      metricRecovery: "范围内日末恢复", metricBudgets: "全部性能预算通过", metricArtifacts: "原始冻结工件",
      scope: "离线仿真中的一个选取日，不是屏幕录像或实机部署。完整评价包含退步及范围外恢复失败。",
      provenance: "数据溯源", readme: "项目 README", yes: "是", no: "否",
      gate: { ok: ["正常规划", "预测与服务可用，允许调用主优化器。"],
        solver_unavailable: ["求解服务中断", "服务不可用，门控后不调用主优化器；按当前 SOC 恢复或空闲。"],
        forecast_expired: ["预测包过期", "观测年龄 4 h，超过 2 h 门限；不复用旧放电请求。"] },
      chapters: [
        ["先预测，再决定。", "六时距预测与误差区间，为保守规划提供输入。", "六时距预测 · 原始时距误差区间"],
        ["把恢复，写进规划。", "同一保守模型用于滚动窗口和参考电量。", "875 MW / 0.85 · 规划与恢复资源相容"],
        ["信息失效，主动回退。", "服务中断和预测过期均保留显式门控。", "故障注入 · 门控期间主优化器不调用"],
        ["闭环完成，逐项核对。", "检查实际动作、连续电量与日末目标，而不只看预测误差。", "该日末 15 GWh · 恢复不等于性能全面占优"],
      ],
    },
    en: {
      offline: "FROZEN OFFLINE EVIDENCE", selectedDay: "SELECTED DAY · UTC", caseName: "Plant mismatch + injected faults",
      forecast: "Forecast", planning: "Plan", fallback: "Fallback", verify: "Verify", netLoad: "NET LOAD / GW",
      actual: "Actual", controlled: "V8 executed", interval: "Forecast band", soc: "STORAGE ENERGY / GWh",
      energy: "Actual energy", floor: "Recovery floor", interpolation: "Hourly evidence; frame interpolation is for animation only.",
      state: "EXECUTION STATE", socValue: "Stored energy", power: "Net request", age: "Observation age", model: "CONSERVATIVE MODEL",
      solver: "Primary optimizer called", dayEnd: "Terminal energy", notAll: "Recovery ≠ all-metric superiority",
      metricRecovery: "In-envelope recovery", metricBudgets: "All budgets passed", metricArtifacts: "Frozen artifacts",
      scope: "A selected day from an offline simulation, not a screen recording or deployed hardware. Full evaluation includes regressions and out-of-envelope failures.",
      provenance: "Provenance", readme: "Project README", yes: "Yes", no: "No",
      gate: { ok: ["Planning available", "Usable forecasts and service permit a primary optimizer call."],
        solver_unavailable: ["Solver unavailable", "The gate suppresses the optimizer; recovery or idle uses current SOC."],
        forecast_expired: ["Forecast expired", "Observation age is 4 h, beyond the 2 h limit. Old discharge requests are not reused."] },
      chapters: [
        ["Forecast before deciding.", "Six-horizon forecasts and uncertainty intervals feed conservative planning.", "Six horizons · lead-aligned uncertainty"],
        ["Plan for recovery.", "One conservative model governs the rolling horizon and reference energy.", "875 MW / 0.85 · recovery-aware planning"],
        ["Fail explicitly. Fall back.", "Service outages and stale forecasts trigger explicit gates.", "Injected faults · no gated optimizer calls"],
        ["Close the loop. Verify.", "Audit executed actions, continuous energy, and the terminal target.", "15 GWh at day end · not universal superiority"],
      ],
    },
  };
  function icons() { if (window.lucide) window.lucide.createIcons({ attrs: { "aria-hidden": "true" } }); }
  function element(tag, attributes, parent, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  const left = 60, right = 888, plotWidth = right - left;
  const x = (hour) => left + hour / 23 * plotWidth;
  const yLoad = (mw) => 283 - (mw / 1000 - 30) / 35 * 263;
  const ySoc = (mwh) => 159 - (mwh / 1000 - 4) / 12 * 139;
  function grid(parent, values, scale) {
    for (const value of values) {
      element("line", { x1: left, x2: right, y1: scale(value * 1000), y2: scale(value * 1000), class: "grid" }, parent);
      element("text", { x: 44, y: scale(value * 1000) + 4, "text-anchor": "end" }, parent, value);
    }
  }
  grid($("load-grid"), [30, 40, 50, 60], yLoad);
  grid($("soc-grid"), [5, 10, 15], ySoc);
  for (const hour of [0, 6, 12, 18, 23]) {
    element("text", { x: x(hour), y: 304, "text-anchor": "middle" }, $("load-hours"), `${String(hour).padStart(2, "0")}:00`);
    element("text", { x: x(hour), y: 184, "text-anchor": "middle" }, $("soc-hours"), `${String(hour).padStart(2, "0")}:00`);
  }
  for (const [start, end] of [[9.5, 10.5], [13.5, 17.5]]) {
    element("rect", { x: x(start), y: 20, width: x(end) - x(start), height: 263, class: "fault-region" }, $("fault-regions"));
  }
  function line(points, field, scale, until = 23) {
    const end = Math.min(23, Math.max(0, until));
    const rows = points.slice(0, Math.floor(end) + 1).map((point, index) => [index, point[field]]);
    if (end % 1 && end < 23) {
      const index = Math.floor(end);
      rows.push([end, points[index][field] + (points[index + 1][field] - points[index][field]) * (end % 1)]);
    }
    return rows.map(([hour, value], index) => `${index ? "L" : "M"}${x(hour).toFixed(2)},${scale(value).toFixed(2)}`).join(" ");
  }
  $("actual-path").setAttribute("d", line(data.points, "net_load_mw", yLoad));
  $("floor-path").setAttribute("d", line(data.points, "safe_floor_next_mwh", ySoc));
  function hourAt(position) {
    if (position < 7) return 9 * position / 7;
    if (position < 14) return 9 + (position - 7) / 7;
    if (position < 21) return 10 + (position - 14) / 7 * 7;
    return 17 + (position - 21) / 7 * 6;
  }
  function setLanguage(value) {
    language = value;
    document.documentElement.lang = value === "en" ? "en" : "zh-CN";
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = dictionary[value][node.dataset.i18n]; });
    $("lang-zh").setAttribute("aria-pressed", value === "zh");
    $("lang-en").setAttribute("aria-pressed", value === "en");
    activeChapter = -1;
    render(time);
  }
  function setTheme(value) {
    theme = value === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    $("theme").innerHTML = `<i data-lucide="${theme === "dark" ? "sun" : "moon"}"></i>`;
    icons();
  }
  function playbackIcon() {
    $("play").innerHTML = `<i data-lucide="${playing ? "pause" : "play"}"></i>`;
    $("play").title = playing ? "暂停 / Pause" : "播放 / Play";
    $("play").setAttribute("aria-label", $("play").title);
    icons();
  }
  function render(position) {
    time = Math.min(duration, Math.max(0, position));
    const hour = hourAt(time);
    const index = Math.min(23, Math.floor(hour + 1e-7));
    const point = data.points[index];
    const next = data.points[Math.min(23, index + 1)];
    const fraction = Math.min(1, hour - index);
    const interpolate = (field) => point[field] + (next[field] - point[field]) * fraction;
    const chapter = Math.min(3, Math.floor(time / 7));
    const words = dictionary[language];
    if (chapter !== activeChapter) {
      activeChapter = chapter;
      $("chapter-number").textContent = `0${chapter + 1} / 04`;
      $("chapter-title").textContent = words.chapters[chapter][0];
      $("chapter-description").textContent = words.chapters[chapter][1];
      $("chapter-caption").textContent = words.chapters[chapter][2];
      document.querySelectorAll("[data-chapter]").forEach((button, i) => {
        if (i === chapter) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      });
    }
    $("current-time").innerHTML = `${String(index).padStart(2, "0")}:00 <small>UTC</small>`;
    $("controlled-path").setAttribute("d", line(data.points, "controlled_net_load_mw", yLoad, hour));
    $("soc-path").setAttribute("d", line(data.points, "storage_soc_mwh", ySoc, hour));
    for (const id of ["load-cursor", "soc-cursor"]) {
      $(id).setAttribute("x1", x(hour)); $(id).setAttribute("x2", x(hour));
    }
    $("load-dot").setAttribute("cx", x(hour)); $("load-dot").setAttribute("cy", yLoad(interpolate("controlled_net_load_mw")));
    $("soc-dot").setAttribute("cx", x(hour)); $("soc-dot").setAttribute("cy", ySoc(interpolate("storage_soc_mwh")));
    const forecast = point.forecast.slice(0, 24 - index);
    const upper = forecast.map((f, i) => [x(index + i), yLoad(f.upper)]);
    const lower = forecast.map((f, i) => [x(index + i), yLoad(f.lower)]).reverse();
    $("forecast-band").setAttribute("d", [...upper, ...lower].map(([a, b], i) => `${i ? "L" : "M"}${a},${b}`).join(" ") + " Z");
    $("prediction-path").setAttribute("d", forecast.map((f, i) => `${i ? "L" : "M"}${x(index + i)},${yLoad(f.prediction)}`).join(" "));
    $("soc-value").textContent = (point.storage_soc_mwh / 1000).toFixed(2);
    const power = point.requested_charge_mw - point.requested_discharge_mw;
    $("power-value").textContent = `${power > 0 ? "+" : ""}${Math.round(power)}`;
    $("age-value").textContent = point.observation_age_hours.toFixed(0);
    const gate = words.gate[point.gate_reason] || words.gate.ok;
    $("gate-state").textContent = gate[0];
    $("gate-explanation").textContent = gate[1];
    const color = point.gate_reason === "ok" ? "var(--accent)" : "var(--red)";
    $("state-indicator").style.background = color;
    $("gate-state").style.color = color;
    $("solver-call").textContent = point.optimizer_called ? words.yes : words.no;
    $("solver-call").style.color = color;
    $("timeline").value = time;
    $("elapsed").textContent = `00:${String(Math.floor(time)).padStart(2, "0")}`;
    document.body.dataset.hour = String(index);
    document.body.dataset.chapter = String(chapter);
    document.body.dataset.playing = String(playing);
  }
  $("play").addEventListener("click", () => { playing = !playing; if (time >= duration) time = 0; playbackIcon(); render(time); });
  $("restart").addEventListener("click", () => { playing = true; playbackIcon(); render(0); });
  $("timeline").addEventListener("input", (event) => { render(Number(event.target.value)); });
  $("speed").addEventListener("change", (event) => { speed = Number(event.target.value); });
  $("lang-zh").addEventListener("click", () => setLanguage("zh"));
  $("lang-en").addEventListener("click", () => setLanguage("en"));
  $("theme").addEventListener("click", () => setTheme(theme === "dark" ? "light" : "dark"));
  $("fullscreen").addEventListener("click", async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { $("fullscreen").disabled = true; }
  });
  document.querySelectorAll("[data-chapter]").forEach((button) => {
    button.addEventListener("click", () => render(Number(button.dataset.chapter) * 7));
  });
  document.addEventListener("visibilitychange", () => { last = null; });
  function frame(now) {
    if (last !== null && playing && !document.hidden) {
      const step = Math.min(.15, (now - last) / 1000);
      render(time + step * speed);
      if (time >= duration) { playing = false; playbackIcon(); render(time); }
    }
    last = now;
    requestAnimationFrame(frame);
  }
  window.GridPulsePlayer = {
    seek: (value) => render(Number(value)),
    pause: () => { playing = false; playbackIcon(); render(time); },
    play: () => { if (time >= duration) time = 0; playing = true; playbackIcon(); render(time); },
    setLanguage, setTheme,
    state: () => ({ time, playing, speed, chapter: activeChapter, language, theme, hour: Number(document.body.dataset.hour) }),
    data,
  };
  setTheme(theme);
  setLanguage(language);
  playbackIcon();
  requestAnimationFrame(frame);
})();
