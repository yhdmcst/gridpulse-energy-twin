"""Read-only evidence verification plus PDF, raster and typography checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageOps, ImageStat
from pypdf import PdfReader

from report_evidence import DEST, ROOT, WORKSPACE

PDF = DEST / "GridPulse-能源数字孪生-技术报告-融合修订版.pdf"
RENDER = WORKSPACE / "tmp" / "pdfs" / "report_revision_20261028"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    evidence = json.loads((DEST / "证据摘要.json").read_text(encoding="utf-8"))
    provenance = json.loads((DEST / "图表溯源.json").read_text(encoding="utf-8"))
    source_hashes = provenance["sources_sha256"]
    for relative, expected in source_hashes.items():
        assert sha(ROOT / relative) == expected, relative
    artifact_dir = ROOT / "outputs/fusion_v8"
    artifacts = json.loads((artifact_dir / "ARTIFACT_SHA256.json").read_text(encoding="utf-8"))
    for relative, expected in artifacts.items():
        assert sha(artifact_dir / relative) == expected, relative

    assert len(evidence["envelope"]) == 30
    assert sum(row["all_budgets_passed"] for row in evidence["envelope"]) == 11
    assert evidence["v8"]["decision"]["realtime_default_admitted"] is False
    assert evidence["v8"]["counts"]["retained_inside_terminal_passed"] == 30
    assert evidence["v8"]["counts"]["retained_inside_supervisor_interventions"] == 0
    assert evidence["v8"]["counts"]["retained_inside_primary_solver_failure_hours"] == 0

    reader = PdfReader(PDF)
    assert len(reader.pages) == 20
    assert PDF.stat().st_size < 10_000_000
    fonts = {}
    for page in reader.pages:
        for font in page["/Resources"].get("/Font", {}).get_object().values():
            font = font.get_object()
            name = str(font.get("/BaseFont"))
            descriptor = font.get("/FontDescriptor")
            embedded = bool(descriptor and any(k in descriptor.get_object() for k in ("/FontFile", "/FontFile2", "/FontFile3")))
            fonts[name] = {"type": str(font.get("/Subtype")), "embedded": embedded}
    assert any("SimSun" in key for key in fonts), fonts
    assert all(value["embedded"] for key, value in fonts.items() if "SimSun" in key or "SimHei" in key)

    page_checks = []
    with pdfplumber.open(PDF) as document:
        fulltext = "\n".join(page.extract_text() or "" for page in document.pages)
        for number, page in enumerate(document.pages, 1):
            outside = [c["text"] for c in page.chars
                       if c["x0"] < 0 or c["x1"] > page.width + .5 or c["top"] < 0 or c["bottom"] > page.height]
            assert not outside, (number, outside)
            overlaps = []
            body = [c for c in page.chars if c["top"] > 53 and c["bottom"] < 793]
            for image in page.images:
                hits = [c for c in body if c["x0"] < image["x1"] - 1 and c["x1"] > image["x0"] + 1
                        and c["top"] < image["bottom"] - 1 and c["bottom"] > image["top"] + 1]
                overlaps.extend(c["text"] for c in hits)
            assert not overlaps, (number, overlaps)
            bodyfont = [c for c in body if "SimSun" in c["fontname"] and abs(c["size"] - 12) < .05]
            page_checks.append(dict(page=number, characters=len(page.chars),
                                    simsun_12pt_characters=len(bodyfont), image_count=len(page.images),
                                    text_within_page=True, image_text_overlap=False))
        assert "学校名称" not in fulltext and "AmaxAmax" not in fulltext
        assert "{{" not in fulltext and "\ufffd" not in fulltext
        assert all(len(page.extract_text() or "") > 100 for page in document.pages)

    rasters = []
    for path in sorted((DEST / "图表").glob("*.png")):
        with Image.open(path) as image:
            assert min(image.size) >= 900
            dpi = image.info.get("dpi", (0, 0))
            assert all(abs(value - 300) < 1 for value in dpi)
            std = max(ImageStat.Stat(image.convert("RGB")).stddev)
            assert std > 5
            rasters.append(dict(file=path.name, size=image.size, dpi=dpi, stddev=std))
        svg = path.with_suffix(".svg").read_text(encoding="utf-8")
        assert "<text" in svg, f"SVG should retain editable labels: {path}"
    assert len(rasters) == 10

    RENDER.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-r", "110", "-png", str(PDF), str(RENDER / "page")],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pages = sorted(RENDER.glob("page-*.png"))
    assert len(pages) == 20
    for path in pages:
        with Image.open(path) as image:
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 5
    thumbs = []
    for index, path in enumerate(pages, 1):
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (248, 351))
            tile = Image.new("RGB", (268, 377), "white")
            tile.paste(thumb, ((268 - thumb.width) // 2, 5))
            from PIL import ImageDraw
            ImageDraw.Draw(tile).text((12, 360), f"{index:02d}", fill="#607079")
            thumbs.append(tile)
    montage = Image.new("RGB", (268 * 5, 377 * 4), "#E5EAEC")
    for i, thumb in enumerate(thumbs):
        montage.paste(thumb, ((i % 5) * 268, (i // 5) * 377))
    montage.save(DEST / "全页排版预览.jpg", quality=90)
    result = dict(
        passed=True, pages=20, pdf_bytes=PDF.stat().st_size,
        report_source_hashes_checked=len(source_hashes), frozen_v8_artifact_hashes_checked=len(artifacts),
        frozen_v8_artifacts_unchanged=True, data_claim_assertions_passed=True,
        fonts=fonts, page_geometry=page_checks, figures=rasters,
        all_pages_rendered_nonblank=True, rendering_dpi=110, render_directory=str(RENDER),
        manual_visual_review="Not performed: current view_image tool reports unsupported image input.",
        remaining_actions=["Manual visual review", "Fill team name before submission", "Check current platform requirements"],
    )
    (DEST / "交付核对.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("page_geometry", "figures")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
