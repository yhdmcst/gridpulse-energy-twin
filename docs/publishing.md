# GitHub 发布指南

应上传本文件所在仓库的根目录，而不是完整研究工作区。
公开仓库根目录应直接显示 `README.md`、`app.py`、`src/` 和 `artifacts/`。

## 上传前

1. 阅读 README 与第三方声明，确认自己有权公开代码、配图与随附的数据快照。
2. 执行 `python scripts/prepare_release.py`、相关回归测试与 V8 核算。
3. 检查 `git status --short`；`.venv/`、`outputs/`、恢复数据和本机设置不应进入暂存。
4. 保留 `.github/`、`.streamlit/`、`.gitignore`、`.gitattributes` 与 `artifacts/`。
5. 不要格式化 `experiments/fusion_v8/frozen_source/`，也不要手改冻结清单。
6. 将 `preview/` 一并上传，里面包含单文件 SVG 动画、双语 HTML 预览和预览截图；它们不是本地缓存。

升级版的全部交付都在当前仓库内：`README.md` / `README_EN.md`、
`assets/readme/`、`docs/showcase/` 与 `preview/`。
项目外层同名 HTML 只是便捷副本，不需要额外上传外层目录。

发布目录没有配置远程地址，也没有替你创建仓库或上传任何文件。

## 首次推送

先在自己的 GitHub 账户创建空仓库，建议命名为 `gridpulse-energy-twin`。
在本地仓库目录执行：

```bash
git init -b main
git add .
git status --short
git commit -m "Initial public release of GridPulse V8"
```

然后使用 GitHub 新仓库页面提供的实际地址设置远程并推送：

```bash
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

`<YOUR_REPOSITORY_URL>` 是说明性占位符，不是可直接使用的地址。
身份验证请使用自己的 SSH 或 Git 凭据管理器，不要把令牌写入脚本、README 或远程 URL。

## 仓库展示设置

- About 描述建议：`Offline energy digital twin with multi-horizon forecasting, conservative storage planning, and auditable V8 evidence.`
- Topics 建议：`energy`、`digital-twin`、`time-series`、`energy-storage`、`pytorch`、`streamlit`、`reproducible-research`。
- Social Preview 可使用 `assets/readme/cover.png`；它是概念视觉，不应标注为实际部署照片。
- Actions 成功运行后再添加对应状态徽章；当前 README 未预设任何虚假的通过状态。
- 交互式 SVG 动画可按 [演示发布说明](showcase/README.md)手动发布至 Pages；部署后再用真实站点地址更新演示入口。

## 后续维护

`artifacts/` 是本版本的可重放输入，不是可随意清理的缓存。
新研究应生成新的输出目录、版本清单和证据包，不要覆盖旧版本后沿用旧结论。
较大的新模型、视频或完整历史交付包宜另行制定分发方式，不要直接加入当前仓库。
