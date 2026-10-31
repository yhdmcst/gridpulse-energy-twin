# GridPulse V8 Motion Study

## 中文

直接在浏览器打开 [`index.html`](index.html) 即可播放，无需安装依赖、联网或启动服务器。
本演示使用 HTML、SVG 和 JavaScript，不是 MP4 视频，也不是 Streamlit 录屏。
仓库内另附[单文件版本](../../preview/GridPulse-SVG演示.html)和[全部展示预览](../../preview/index.html)，
无需额外收集项目外层文件。

- 时长 28 秒，包含预测、保守规划、回退、核对四章。
- 可播放/暂停、重播、拖动进度、切换章节与倍速。
- 支持中英文、深浅色和桌面/手机布局。
- 尊重系统减少动态效果设置：该偏好开启时默认暂停。
- 数据为 `2022-01-23` 的 `combined90 / fresh6 / combined_fault / envelope_tail` 24 条冻结小时记录。
- 该日被选中用于同时展示故障与非零储能动作，不是随机抽样或完整评价的替代。
- 曲线帧间插值只用于显示；数值读数以当前小时的原始记录为准。

预测区间来自 V6 固定模型的冻结预测面板，执行与电量来自 V8 冻结轨迹。
三份源文件的 SHA-256 保存在 [`provenance.json`](provenance.json)，完整来源仍可通过仓库内的证据包核对。
`trace-data.js` 为这三份工件中选取日期的公开摘录，不包含新训练或新求解结果。

GitHub 不在 README 内执行 HTML/JavaScript，因此 README 内展示 GIF。
直接点击仓库中的 HTML 文件会打开源码；下载后本地打开，或启用下面的 Pages 工作流。

## 可选 GitHub Pages

1. 将仓库正常上传 GitHub。
2. 在仓库 Settings → Pages 中将 Source 设为 GitHub Actions。
3. 在 Actions 中选择 **Publish Showcase**，手动点击 Run workflow。
4. 成功后使用部署步骤输出的真实地址，并将其填入 README 演示链接。

工作流只发布一个最小静态站点：根 README、README 图片、本演示目录及 `preview/`，
不会把 407 MiB 的恢复数据或全部研究证据发布为网页。
默认不会自动部署，也未在 README 预填虚构的站点地址。
站点根目录的 Markdown 在部分浏览器会显示为文本；完整文档以 GitHub 仓库阅读体验为准。

查询参数支持 `?lang=en`、`?theme=dark`，可组合使用。

## English

Open [`index.html`](index.html) directly in a browser. No server, dependencies, or internet access is required.
This is an HTML/SVG/JavaScript presentation, not an MP4 video or a Streamlit screen recording.
The [single-file edition](../../preview/GridPulse-SVG演示.html?lang=en) and
[preview collection](../../preview/index.html) are also included in this repository.

The 28-second player includes four chapters, pause/restart, seeking, speed selection, Chinese/English,
light/dark themes, and responsive layouts. Reduced-motion users start paused.

Its 24 hourly records come from a deliberately selected frozen day, **2022-01-23**.
The example shows both injected faults and nonzero storage actions; it is not an unbiased performance sample.
Curves interpolate for animation; numeric readouts use recorded hourly values.
Predictions come from the frozen V6 panels, and actions/SOC from the V8 trace.
[Provenance](provenance.json) retains the source hashes.

GitHub displays the GIF preview but does not execute the player inside README.
Download and open locally, or enable GitHub Actions as the Pages source and manually run **Publish Showcase**.
Use its real deployment URL only after a successful run.

## Files

| File | Role |
| --- | --- |
| `index.html` | Semantic player layout and SVG chart surfaces |
| `player.css` | Responsive light/dark styling |
| `player.js` | Deterministic playback and controls |
| `trace-data.js` | Frozen 24-hour data excerpt |
| `provenance.json` | Source hashes, conditions, and selection boundaries |
| `lucide.min.js` | Locally vendored Lucide 0.468.0 icons |
| `LUCIDE-LICENSE.txt` | Upstream ISC license and notices |
