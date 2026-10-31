"""Official-outline report, with the template cover fields and CJK typography."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from competition_common import DEST, FIGURES, QA, read_evidence
import build_report as previous

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
PDF = DEST / "GridPulse-AI能源技术报告-模板对齐参赛审校版.pdf"
SOURCE = DEST / "GridPulse-AI能源技术报告-模板对齐参赛审校版.md"
METADATA = HERE / "competition_metadata.json"
LEFT, RIGHT, TOP, BOTTOM = 79.44, 79.30, 75.0, 78.0
WIDTH = A4[0] - LEFT - RIGHT
BLACK = colors.HexColor("#171C20")
GRAY = colors.HexColor("#536168")


def styles():
    base = dict(fontName="SimSunReport", textColor=BLACK, wordWrap="CJK",
                allowWidows=0, allowOrphans=0, splitLongWords=1)
    return dict(
        body=ParagraphStyle("Body", **base, fontSize=12, leading=14.4,
                            firstLineIndent=24, spaceAfter=7),
        h1=ParagraphStyle("H1", fontName="SimHeiReport", fontSize=14, leading=20,
                          spaceBefore=2, spaceAfter=10, keepWithNext=True, wordWrap="CJK"),
        h2=ParagraphStyle("H2", fontName="SimHeiReport", fontSize=12, leading=17,
                          spaceBefore=7, spaceAfter=7, keepWithNext=True, wordWrap="CJK"),
        h3=ParagraphStyle("H3", fontName="SimHeiReport", fontSize=12, leading=17,
                          spaceBefore=5, spaceAfter=7, keepWithNext=True, wordWrap="CJK"),
        abstract_title=ParagraphStyle("Abstract", fontName="SimHeiReport", fontSize=15,
                                      leading=22, alignment=1, spaceBefore=3, spaceAfter=18),
        cell=ParagraphStyle("Cell", **base, fontSize=12, leading=14.4),
        thead=ParagraphStyle("TableHead", fontName="SimHeiReport", fontSize=12,
                             leading=14.4, wordWrap="CJK", textColor=BLACK),
        caption=ParagraphStyle("Caption", **{**base, "textColor": GRAY},
                               fontSize=10.5, leading=14, spaceAfter=9),
        reference=ParagraphStyle("Reference", **base, fontSize=10.5, leading=20, spaceAfter=7),
        code=ParagraphStyle("Code", **base, fontSize=12, leading=14.4, spaceAfter=8,
                            leftIndent=4),
    )


def cover(canvas, metadata):
    height = A4[1]
    canvas.setFillColor(colors.black)
    canvas.setFont("SimHeiReport", 16)
    canvas.drawString(79.44, height - 87.4, "附件2")
    canvas.setFont("SimHeiReport", 22)
    for baseline, text in (
        (134, "2026 年第八届"), (164, "全球校园人工智能算法精英大赛"),
        (254, "算法主题赛"), (284, "AI+能源"), (314, "技术报告"),
    ):
        canvas.drawCentredString(A4[0] / 2, height - baseline, text)
    canvas.setFont("SimSunReport", 12)
    for baseline, label, value in (
        (447, "作品名称：", metadata["work_title"]),
        (477, "赛道组别：", "科技创新组"),
        (507, "团队名称：", metadata.get("team_name") or "________________"),
    ):
        canvas.drawString(186.12, height - baseline, label)
        canvas.drawString(251.5, height - baseline, value)
    canvas.setFont("SimSunReport", 15.95)
    canvas.drawCentredString(A4[0] / 2, height - 679, f"日期：{metadata['date']}")


class CompetitionDoc(BaseDocTemplate):
    def __init__(self, metadata):
        super().__init__(str(PDF), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
                         topMargin=TOP, bottomMargin=BOTTOM,
                         title="GridPulse能源数字孪生 - AI+能源技术报告",
                         author="GridPulse", subject="科技创新组；离线闭环仿真研究", pageCompression=1)
        self.metadata = metadata
        self.starts = []
        frame = Frame(LEFT, BOTTOM, self.width, self.height, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0, id="body")
        self.addPageTemplates(PageTemplate(id="official", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page == 1:
            cover(canvas, self.metadata)
        canvas.setFont("SimSunReport", 9)
        canvas.setFillColor(colors.black)
        canvas.drawRightString(516, A4[1] - 48.95, "第八届全球校园人工智能算法精英大赛·算法主题赛")
        canvas.setFont("SimSunReport", 12)
        canvas.drawCentredString(A4[0] / 2, 60.5, f"— {doc.page} —")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, previous.Marker):
            self.starts.append(dict(title=flowable.title, page=self.page))
            key = f"p{len(self.starts)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.title, key, level=0)


def table(lines, sty):
    raw = []
    for line in lines:
        row = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[-: ]+", c) for c in row):
            continue
        raw.append(row)
    count = len(raw[0])
    weights = (.40, .20, .20, .20) if count == 4 else (.30, .36, .34) if count == 3 else (.29, .71)
    rows = [[Paragraph(previous.inline(c), sty["thead"] if i == 0 else sty["cell"])
             for c in row] for i, row in enumerate(raw)]
    result = Table(rows, colWidths=[WIDTH * w for w in weights], repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F3")),
        ("LINEABOVE", (0, 0), (-1, 0), .7, BLACK),
        ("LINEBELOW", (0, 0), (-1, 0), .5, BLACK),
        ("LINEBELOW", (0, -1), (-1, -1), .7, BLACK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return KeepTogether([Spacer(1, 2), result, Spacer(1, 9)])


def parse(segment, sty):
    lines = segment.splitlines()
    story = []
    in_references = segment.lstrip().startswith("## 参考文献")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                assert pdfmetrics.stringWidth(lines[i], "SimSunReport", 12) <= WIDTH - 4, lines[i]
                code.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code), sty["code"]))
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(table(block, sty))
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.+)\]\((.+)\)", line)
            caption, filename = match.groups()
            with PILImage.open(DEST / filename) as image:
                w, h = image.size
            factor = min(WIDTH / w, 395 / h)
            story.append(KeepTogether([
                Image(str(DEST / filename), width=w * factor, height=h * factor),
                Spacer(1, 5), Paragraph(previous.inline(caption), sty["caption"]),
            ]))
        elif line.startswith("#### "):
            story.append(Paragraph(previous.inline(line[5:]), sty["h3"]))
        elif line.startswith("### "):
            story.append(Paragraph(previous.inline(line[4:]), sty["h2"]))
        elif line.startswith("## "):
            style = sty["abstract_title"] if line == "## 摘要" else sty["h1"]
            story.append(Paragraph(previous.inline(line[3:]), style))
        else:
            style = sty["reference"] if in_references and re.match(r"^\[\d+\]", line) else sty["body"]
            story.append(Paragraph(previous.inline(line), style))
        i += 1
    return story


def main():
    previous.fonts()
    evidence = read_evidence()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    replacement = previous.tokens(evidence)
    for method in ("persistence", "seasonal24"):
        for split in ("test", "2021", "2022"):
            replacement[f"{method}_{split}"] = f"{evidence['v1']['baselines'][method][split]['mae_mw']:.2f}"
    for arm, token in (("data", "account_data_test"), ("physics", "account_physics_test")):
        values = [evidence["v1"]["runs"][f"{arm}_seed{s}"]["variants"]["raw"]["test"]["accounting_residual_mae_mw"]
                  for s in (42, 43, 44)]
        replacement[token] = f"{sum(values) / 3:.2f}"
    raw = (HERE / "competition_template.md").read_text(encoding="utf-8")
    for key, value in replacement.items():
        raw = raw.replace("{{" + key + "}}", value)
    if metadata.get("team_name"):
        raw = raw.replace("团队名称：________________", f"团队名称：{metadata['team_name']}")
    assert "{{" not in raw
    SOURCE.write_text(raw, encoding="utf-8")
    parts = re.split(r"<!-- page: (.+?) -->", raw)
    story = [previous.Marker("封面"), Spacer(1, 1), PageBreak()]
    sty = styles()
    for i in range(1, len(parts), 2):
        story.extend([previous.Marker(parts[i]), *parse(parts[i + 1], sty)])
        if i + 2 < len(parts):
            story.append(PageBreak())
    doc = CompetitionDoc(metadata)
    doc.build(story)
    reader = PdfReader(PDF)
    result = dict(
        pdf=str(PDF), bytes=PDF.stat().st_size, pages=len(reader.pages),
        planned_sections=1 + (len(parts) - 1) // 2, section_starts=doc.starts,
        body_and_table_font="SimSun 12pt", body_leading_pt=14.4,
        reference_font="SimSun 10.5pt", reference_leading_pt=20,
        cover="Original template fields, text hierarchy, header and footer reconstructed",
        page_margins_pt=dict(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM),
        abstract_characters=len(previous.ABSTRACT),
        team_name_supplied=bool(metadata.get("team_name")),
        team_id_supplied=bool(metadata.get("team_id")),
        pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    )
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "report_build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert PDF.stat().st_size < 10_000_000
    if len(reader.pages) != result["planned_sections"]:
        raise RuntimeError("Unplanned page spill; inspect report_build.json before delivery.")


if __name__ == "__main__":
    main()
