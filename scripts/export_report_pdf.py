from __future__ import annotations

import re
import os
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager
from reportlab.platypus import (
    KeepTogether,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/technical-report-draft.md"
OUTPUT = ROOT / "outputs/report/gridpulse-technical-report-v0.1.pdf"
FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\simhei.ttf")


def register_fonts() -> None:
    regular = Path(os.environ.get("GRIDPULSE_FONT_PATH", str(FONT)))
    bold = Path(os.environ.get("GRIDPULSE_BOLD_FONT_PATH", str(FONT_BOLD)))
    if not regular.is_file():
        for family in ("Noto Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei"):
            try:
                regular = Path(font_manager.findfont(family, fallback_to_default=False))
                break
            except ValueError:
                continue
    if not regular.is_file():
        raise FileNotFoundError("Set GRIDPULSE_FONT_PATH to a licensed Chinese TrueType font; see docs/reproducibility.md.")
    if not bold.is_file():
        bold = regular
    pdfmetrics.registerFont(TTFont("NotoSC", str(regular)))
    pdfmetrics.registerFont(TTFont("NotoSC-Bold", str(bold)))
    pdfmetrics.registerFontFamily("NotoSC", normal="NotoSC", bold="NotoSC-Bold", italic="NotoSC", boldItalic="NotoSC-Bold")


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='NotoSC-Bold'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="NotoSC-Bold", fontSize=20, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#17302A"), spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="NotoSC-Bold", fontSize=15, leading=21, keepWithNext=True, textColor=colors.HexColor("#0F766E"), spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="NotoSC-Bold", fontSize=12, leading=17, keepWithNext=True, textColor=colors.HexColor("#17302A"), spaceBefore=9, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="NotoSC", fontSize=12, leading=14.4, wordWrap="CJK", allowWidows=0, allowOrphans=0, textColor=colors.HexColor("#24332F"), spaceAfter=6),
        "reference": ParagraphStyle("reference", parent=base["BodyText"], fontName="NotoSC", fontSize=10.5, leading=20, wordWrap="CJK", allowWidows=0, allowOrphans=0, spaceAfter=4),
        "table_body": ParagraphStyle("table_body", parent=base["BodyText"], fontName="NotoSC", fontSize=10.5, leading=14, wordWrap="CJK", textColor=colors.HexColor("#24332F"), spaceAfter=0),
        "table_header": ParagraphStyle("table_header", parent=base["BodyText"], fontName="NotoSC-Bold", fontSize=9.2, leading=14, textColor=colors.white, spaceAfter=0),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="NotoSC", fontSize=12, leading=14.4, wordWrap="CJK", leftIndent=13, firstLineIndent=-8, spaceAfter=3),
        "code": ParagraphStyle("code", parent=base["Code"], fontName="NotoSC", fontSize=8.5, leading=12, backColor=colors.HexColor("#F0F5F2"), borderPadding=6, spaceBefore=4, spaceAfter=7),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="NotoSC", fontSize=8, leading=11, textColor=colors.HexColor("#50625E"), spaceAfter=6),
    }


def parse_table(lines: list[str], styles):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        style = styles["table_header"] if not rows else styles["table_body"]
        rows.append([Paragraph(inline(cell), style) for cell in cells])
    if not rows:
        return Spacer(1, 1)
    widths = [((A4[0] - 35 * mm) / len(rows[0]))] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "NotoSC-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D6D0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FAFAF7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return KeepTogether([Spacer(1, 4), table, Spacer(1, 7)])


def build_story(styles):
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    in_references = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                story.append(Paragraph(inline("\n".join(code_lines)).replace("\n", "<br/>"), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            caption, relative_path = image_match.groups()
            image_path = SOURCE.parent / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Report figure is missing: {image_path}")
            source_width, source_height = ImageReader(str(image_path)).getSize()
            max_width = A4[0] - 36 * mm
            max_height = 98 * mm
            scale = min(max_width / source_width, max_height / source_height)
            story.append(Spacer(1, 4))
            story.append(Image(str(image_path), width=source_width * scale, height=source_height * scale, hAlign="CENTER"))
            story.append(Paragraph(inline(caption), styles["caption"]))
            story.append(Spacer(1, 4))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, styles))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"]))
        elif line.startswith("## "):
            in_references = line[3:] == "参考资料"
            story.append(Paragraph(inline(line[3:]), styles["h1"]))
        elif line.startswith("#### "):
            story.append(Paragraph(inline(line[5:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h2"]))
        elif line.startswith("- "):
            story.append(Paragraph("- " + inline(line[2:]), styles["reference"] if in_references else styles["bullet"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), styles["bullet"]))
        else:
            story.append(Paragraph(inline(line), styles["body"]))
        i += 1
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("NotoSC", 8)
    canvas.setFillColor(colors.HexColor("#50625E"))
    canvas.drawString(18 * mm, 12 * mm, "GridPulse | 技术报告 | 公开数据回测原型")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm, title="GridPulse 技术报告",
        author="GridPulse",
    )
    doc.build(build_story(make_styles()), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
