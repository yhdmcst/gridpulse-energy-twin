"""Portable Chinese chart fonts for browser and headless execution."""
from __future__ import annotations

import os

import matplotlib
from matplotlib import font_manager


def configure_chinese_font() -> str | None:
    path = os.environ.get("GRIDPULSE_FONT_PATH")
    if path:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        family = next((name for name in (
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC",
            "WenQuanYi Zen Hei", "Arial Unicode MS", "PingFang SC",
            "PingFang HK", "Songti SC", "STHeiti", "Heiti TC",
        ) if name in installed), None)
    if family:
        matplotlib.rcParams["font.sans-serif"] = [family]
        matplotlib.rcParams["axes.unicode_minus"] = False
    return family
