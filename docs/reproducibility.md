# 运行与复算指南

本指南针对独立 GitHub 发布版。所有命令均从仓库根目录执行，建议使用 Python 3.12。

## 1. 安装应用依赖

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`。
macOS / Linux 激活：`source .venv/bin/activate`。

```bash
python -m pip install -r requirements.txt
```

应用依赖沿用原项目经过本地验证的版本，并显式声明 Altair 与 Pillow。
另外固定 `starlette==0.49.3`：本次真实浏览器检查发现原环境的 Starlette 1.10.0
与 Streamlit 1.58.0 的媒体压缩中间件不兼容，会导致 PNG 请求返回 500。
发布版通过依赖约束解决，不修改原应用、第三方库源码或冻结结果。
神经网络训练与相关模块测试需要额外的 `requirements-training.txt`，界面回放不需要 PyTorch。

## 2. 准备证据与启动

```bash
python run.py
```

该入口先验证并恢复 `artifacts/` 中的证据，再启动原版 Streamlit 应用。
也可分步运行：

```bash
python scripts/prepare_release.py
python -m streamlit run app.py
```

默认地址为 `http://localhost:8501`。端口可通过 `python run.py --server.port 8502` 指定。
准备过程只使用 Python 标准库，不会联网。完整清单见
[`artifacts/manifest.json`](../artifacts/manifest.json)。
准备后约需 407 MiB 证据空间，另计虚拟环境。不要删除仓库中的原始 ZIP。

## 3. 三级核对

### 发布包完整性

```bash
python scripts/prepare_release.py --verify-only
```

检查所有压缩包及 250 个恢复文件的 SHA-256。重复准备会保留相同文件；
若本地证据不同则退出。确认已备份自己的修改后，才使用 `--repair` 恢复原始证据。

### 原实验独立复算

```bash
python scripts/verify_envelope_results.py
```

核对 204 个原始冻结工件、114 条轨迹与 96 组配对区间，并校验原源码、预测和基线来源。
默认回执写入 `outputs/review/fusion_v8_verification.json`。
这里包含复用的 V7 基线，不能把所有轨迹视为新的独立实验。

### 当前源码与界面

```bash
python -m pip install -r requirements-training.txt
python -m pytest
python scripts/check_app.py --output outputs/review/app_verification.json
```

`check_app.py` 覆盖年份、预测延迟、设备、故障切换，以及表格策略、日志过滤、范围外警示等状态。
完整测试依赖 PyTorch。两项前序可选实验测试在未提供相应历史工件时会按测试设计跳过；
不能将跳过记录描述为这些历史工件已经核验。

## 4. 原始环境与当前环境

V8 原实验的 Python、NumPy、Pandas、SciPy 版本以
[`evidence/v8/manifest.json`](../evidence/v8/manifest.json) 为准。
原神经网络训练环境为 PyTorch 2.3.0+cu121；当前完整测试使用 PyTorch 2.6.0。
当前测试通过不等于在原硬件上重新训练完成。

原实验源码位于 `experiments/fusion_v8/frozen_source/`，按原 manifest 校验，不进行格式化和换行归一化。
当前应用及校验修订与冻结源码分开保留。

## 5. 本发布版不包含的复现任务

- V1–V7 全部模型训练、轨迹和分析产物未全部打包。
- 论文图表、完整报告和答辩生成脚本保留为研究源码，但依赖前序工件、原目录布局、中文字体或本地审校工具。
- 远程实验脚本可能记录原机器解释器路径；这是历史执行记录，不是跨平台启动入口。
- 随附技术报告与图表可直接阅读，不需要先运行报告生成脚本。

不要直接运行会覆盖冻结输出的历史实验脚本；新实验应使用独立输出目录。
完整范围见 [release-scope.md](release-scope.md)。
