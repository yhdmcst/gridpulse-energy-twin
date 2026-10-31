"""Template, evidence, final-size typography and rendered-geometry verification."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
import sys
from pathlib import Path

from competition_common import DEST, FIGURES, QA, ROOT, SKILL, WORKSPACE, read_evidence
from competition_report import PDF, SOURCE, WIDTH

sys.path.insert(0, str(WORKSPACE / ".report-deps"))
sys.path.insert(0, str(SKILL / "scripts"))
import numpy as np
import pandas as pd
import pdfplumber
import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageStat
from pypdf import PdfReader
from audit_figure_collisions import (
    TextBox, TraceBox, audit_geometries, audit_pdf, extract_pdf_geometry, write_overlay_pdf,
)
from validate_figure import validate_source

EXPECTED = [
    "1. 项目背景与意义", "1.1 能源行业痛点分析", "1.2 人工智能技术应用价值", "1.3 项目目标与意义",
    "2. 技术方案与实现", "2.1 总体架构设计", "2.2 核心算法与模型", "2.3 数据处理与分析", "2.4 系统功能实现",
    "3. 实验结果与分析", "3.1 实验环境与设置", "3.2 评价指标", "3.3 结果展示与分析",
    "4. 创新点与应用前景", "4.1 主要创新点", "4.2 应用前景与预期效益", "5. 结论",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_metrics():
    base = ROOT / "outputs/fusion_v8"
    results = {}

    def walk(value):
        if isinstance(value, dict):
            if "trajectory" in value:
                results[value["trajectory"]] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(json.loads((base / "metrics.json").read_text(encoding="utf-8")))
    walk(json.loads((base / "validation_references.json").read_text(encoding="utf-8")))
    assert len(results) == 114
    hours, active_conditions = 0, []
    for relative, metrics in results.items():
        frame = pd.read_csv(base / relative)
        times = pd.to_datetime(frame.timestamp, utc=True)
        assert times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
        assert len(frame) == metrics["rows"]
        model = metrics["physical_parameters"]
        before = frame.soc_before_mwh.to_numpy(float)
        after = frame.storage_soc_mwh.to_numpy(float)
        charge = frame.charge_mw.to_numpy(float)
        discharge = frame.discharge_mw.to_numpy(float)
        eta = model["efficiency"]
        assert np.allclose(after - before, eta * charge - discharge / eta, rtol=0, atol=1e-5), relative
        assert np.allclose(before[1:], after[:-1], rtol=0, atol=1e-5), relative
        assert abs(before[0] - model["initial_soc_mwh"]) < 1e-5
        assert charge.min() >= -1e-5 and discharge.min() >= -1e-5
        assert charge.max() <= model["charge_power_mw"] + 1e-5
        assert discharge.max() <= model["discharge_power_mw"] + 1e-5
        assert np.minimum(charge, discharge).max() <= 1e-5
        assert min(before.min(), after.min()) >= model["minimum_soc_mwh"] - 1e-5
        assert max(before.max(), after.max()) <= model["maximum_soc_mwh"] + 1e-5
        actual = frame.net_load_mw.to_numpy(float)
        controlled = frame.controlled_net_load_mw.to_numpy(float)
        assert np.allclose(controlled, actual + charge - discharge, atol=1e-5, rtol=0)
        terminal = np.abs(after[times.dt.hour.eq(23)] - model["initial_soc_mwh"])
        values = dict(
            controlled_peak_mw=controlled.max(), controlled_p95_mw=np.quantile(controlled, .95),
            mean_daily_peak_mw=pd.Series(controlled).groupby(times.dt.floor("D")).max().mean(),
            throughput_mwh=(charge + discharge).sum(),
            conversion_loss_mwh=((1 - eta) * charge + (1 / eta - 1) * discharge).sum(),
            max_abs_terminal_error_mwh=terminal.max(),
        )
        for key, value in values.items():
            assert abs(value - metrics[key]) < 1e-5, (relative, key, value, metrics[key])
        if "primary_solver_attempted" in frame:
            gated = frame.gate_reason.isin(("forecast_expired", "solver_unavailable"))
            assert not frame.loc[gated, "primary_solver_attempted"].any()
            assert np.allclose(frame.measured_soc_mwh, before, atol=1e-8, rtol=0)
        condition = metrics.get("condition", {})
        if relative.startswith("evaluation/") and condition.get("policy") == "envelope_tail" and metrics["plant_within_declared_envelope"]:
            assert terminal.max() < 1e-5
            assert metrics["supervisor_interventions"] == 0
            assert metrics["primary_solver_failure_hours"] == 0
            active_conditions.append(relative)
        hours += len(frame)
    assert len(active_conditions) == 30
    return dict(trajectories=114, trajectory_hours=hours, candidate_inside_conditions=30,
                recomputed_fields=7, approach="Read-only CSV audit, independent of the main optimization runtime")


def main():
    source_text = SOURCE.read_text(encoding="utf-8")
    headings = re.findall(r"^#{2,3} (.+)$", source_text, re.MULTILINE)
    numbered = [h for h in headings if re.match(r"^\d", h)]
    assert numbered == EXPECTED, numbered
    assert all(h in headings for h in ("摘要", "参考文献", "附录"))
    assert "{{" not in source_text
    assert not re.search(r"^### (?:2\.[5-9]|3\.[4-9])(?: |$)", source_text, re.MULTILINE)
    assert not any(s in source_text for s in ("AmaxAmax", "218.192.", "C:\\Users", "指导老师："))
    abstract = source_text.split("## 摘要\n", 1)[1].split("关键词：", 1)[0].strip()
    assert 300 <= len(abstract) <= 500
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", abstract))
    assert 300 <= chinese_count <= 500

    hashes = json.loads((ROOT / "outputs/fusion_v8/ARTIFACT_SHA256.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        assert sha(ROOT / "outputs/fusion_v8" / relative) == expected, relative
    provenance = json.loads((DEST / "图表溯源.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["sources_sha256"].items():
        assert sha(ROOT / relative) == expected
    for name, expected in provenance["generator_sha256"].items():
        assert sha(Path(__file__).parent / name) == expected, name
    assert sha(WORKSPACE / "附件2AIC·AI能源技术报告参考大纲-科技创新组.pdf") == provenance["official_template_sha256"]
    numerical = trajectory_metrics()
    evidence = read_evidence()
    assert len(evidence["envelope"]) == 30
    assert sum(r["all_budgets_passed"] for r in evidence["envelope"]) == 11

    plotting = Path(__file__).with_name("competition_figures.py").read_text(encoding="utf-8")
    preflight = [asdict(row) for row in validate_source(plotting, "python")]
    assert not any(row["level"] == "FAIL" for row in preflight)
    (QA / "plot_source_preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    figures = json.loads((QA / "figure_audit_summary.json").read_text(encoding="utf-8"))
    assert len(figures) == 10
    assert all(row["collision"] == "PASS" for row in figures.values())
    assert all(row["alignment"] in ("PASS", "NOT APPLICABLE") for row in figures.values())
    collisions = audit_pdf(PDF)
    (QA / "report.raw-native-collision-audit.json").write_text(
        json.dumps(collisions, ensure_ascii=False, indent=2), encoding="utf-8")
    # A ReportLab paragraph can be one multiline texttrace. The skill's
    # line-to-trace union then inflates its first line across later baselines.
    # Keep that raw diagnostic, and audit the report with actual glyph-line
    # boxes; scientific figure audits above remain completely unmodified.
    geometries = extract_pdf_geometry(PDF)
    with pdfplumber.open(PDF) as document:
        for page, geometry in zip(document.pages, geometries):
            words = page.extract_words(x_tolerance=1, y_tolerance=1)
            geometry.texts = [
                TextBox(i, word["text"], (word["x0"], word["top"], word["x1"], word["bottom"]))
                for i, word in enumerate(words)
            ]
            geometry.traces = [TraceBox(t.index, t.text, t.bbox) for t in geometry.texts]
    line_collisions = audit_geometries(geometries)
    (QA / "report.line-collision-audit.json").write_text(
        json.dumps(line_collisions, ensure_ascii=False, indent=2), encoding="utf-8")
    if line_collisions["findings"]:
        write_overlay_pdf(PDF, QA / "report.line-collision-overlay.pdf", line_collisions["findings"])
    assert line_collisions["summary"]["fail"] == 0, "Inspect report.line-collision-audit.json"

    page_checks, placed_font_floor = [], 100.0
    with pdfplumber.open(PDF) as document:
        for number, page in enumerate(document.pages, 1):
            for char in page.chars:
                assert 0 <= char["x0"] <= char["x1"] <= page.width + .5
                assert 0 <= char["top"] <= char["bottom"] <= page.height + .5
            if number > 2:
                body = [c for c in page.chars if 72 < c["top"] < 765]
                allowed = (10.5, 12, 14)
                assert all(any(abs(c["size"] - size) < .05 for size in allowed) for c in body)
            page_checks.append(dict(page=number, characters=len(page.chars), images=len(page.images)))
        images = [image for page in document.pages for image in page.images]
        assert len(images) == 10
    for name, row in figures.items():
        with pymupdf.open(FIGURES / f"{name}.pdf") as figure:
            scale = min(1, WIDTH / figure[0].rect.width, 395 / figure[0].rect.height)
            placed_font_floor = min(placed_font_floor, scale * row["glyph_min_pt"])
    assert placed_font_floor >= 5
    assert PDF.stat().st_size < 10_000_000
    reader = PdfReader(PDF)
    font_names = set()
    for page in reader.pages:
        for font in page["/Resources"]["/Font"].get_object().values():
            font = font.get_object()
            name = str(font.get("/BaseFont"))
            font_names.add(name)
            if "SimSun" in name or "SimHei" in name:
                descriptor = font["/FontDescriptor"].get_object()
                assert "/FontFile2" in descriptor

    render = WORKSPACE / "tmp/pdfs/competition_template_review"
    render.mkdir(parents=True, exist_ok=True)
    thumbs = []
    with pymupdf.open(PDF) as document:
        assert len(document) == 21
        for number, page in enumerate(document, 1):
            pix = page.get_pixmap(dpi=120, alpha=False)
            path = render / f"page-{number:02d}.png"
            pix.save(path)
            with Image.open(path) as image:
                assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 4
                thumb = ImageOps.contain(image.convert("RGB"), (232, 329))
                tile = Image.new("RGB", (250, 355), "white")
                tile.paste(thumb, ((250 - thumb.width) // 2, 4))
                ImageDraw.Draw(tile).text((10, 337), f"{number:02d}", fill="#536168")
                thumbs.append(tile)
    montage = Image.new("RGB", (250 * 5, 355 * 5), "#E8EDEF")
    for i, thumb in enumerate(thumbs):
        montage.paste(thumb, ((i % 5) * 250, (i // 5) * 355))
    montage.save(DEST / "全页预览.jpg", quality=90)
    app_path = ROOT / "outputs/review/app_verification.json"
    app_check = json.loads(app_path.read_text(encoding="utf-8")) if app_path.exists() else {}
    browser_path = ROOT / "outputs/review/browser_verification.json"
    browser_check = json.loads(browser_path.read_text(encoding="utf-8")) if browser_path.exists() else {}
    metadata = json.loads((Path(__file__).parent / "competition_metadata.json").read_text(encoding="utf-8"))
    result = dict(
        automated_passed=True, pages=len(reader.pages), bytes=PDF.stat().st_size,
        official_numbered_headings=EXPECTED, official_heading_order_exact=True,
        abstract_characters=len(abstract), abstract_chinese_characters=chinese_count,
        body_font="SimSun12pt; single-line layout14.4pt", reference_font="SimSun10.5pt/20pt",
        effective_figure_glyph_floor_pt=placed_font_floor,
        source_preflight_failures=0,
        source_preflight_warnings=[row["check_id"] for row in preflight if row["level"] == "WARN"],
        figure_count=10, figure_collision_failures=0, figure_collision_warnings=0,
        all_comparable_panels_aligned=True,
        raw_texttrace_diagnostic=collisions["verdict"],
        native_trace_limit="Multiline ReportLab text traces are not line-level boxes; raw diagnostic retained.",
        full_report_line_geometry_verdict=line_collisions["verdict"],
        full_report_line_geometry_summary=line_collisions["summary"],
        frozen_v8_hashes_checked=len(hashes), frozen_v8_unchanged=True,
        read_only_numerical_audit=numerical, page_geometry=page_checks,
        render_directory=str(render), all_pages_rendered_nonblank=True,
        application_check_passed=app_check.get("passed", False),
        browser_check_passed=browser_check.get("passed", False),
        visual_review_method="Rendered-page geometry, embedded fonts, image placement and nonblank pixels.",
        pending_submission_fields=[key for key in ("team_name", "team_id") if not metadata.get(key)],
    )
    (QA / "submission_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "report_only_revision": True,
        "files_sha256": {
            str(path.relative_to(DEST)).replace("\\", "/"): sha(path)
            for path in sorted(DEST.rglob("*"))
            if path.is_file() and path.name != "delivery_manifest.json"
        },
        "authoritative_qa": "QA/submission_audit.json",
        "layout_checks": "Rendered pages, line geometry, embedded fonts and image placement.",
    }
    (DEST / "delivery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("page_geometry", "official_numbered_headings")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
