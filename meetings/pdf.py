import re
from io import BytesIO

from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import Meeting


NAVY = colors.HexColor("#07155f")
RED = colors.HexColor("#f04438")
MUTED = colors.HexColor("#5b6475")
LIGHT_LINE = colors.HexColor("#d9e0ec")
LIGHT_BG = colors.HexColor("#f7f9fd")
TEXT = colors.HexColor("#182033")


def build_pm_notes_pdf(meeting: Meeting) -> bytes:
    buffer = BytesIO()
    register_fonts()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=28 * mm,
        bottomMargin=30 * mm,
        title=f"{meeting.title or 'Meeting'} - Project Manager Notes",
        author="Bit68",
    )
    styles = build_styles()
    story = build_story(meeting, styles)
    doc.build(story, onFirstPage=lambda c, d: draw_page(c, d, meeting), onLaterPages=lambda c, d: draw_page(c, d, meeting))
    return buffer.getvalue()


def register_fonts() -> None:
    font_paths = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "PMBody"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "PMBodyBold"),
        ("C:/Windows/Fonts/arial.ttf", "PMBody"),
        ("C:/Windows/Fonts/arialbd.ttf", "PMBodyBold"),
    ]
    for path, name in font_paths:
        if name in pdfmetrics.getRegisteredFontNames():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path))
        except Exception:
            continue


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body_font = "PMBody" if "PMBody" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_font = "PMBodyBold" if "PMBodyBold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    return {
        "title": ParagraphStyle(
            "PMTitle",
            parent=sample["Title"],
            fontName=bold_font,
            fontSize=21,
            leading=26,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "PMSubtitle",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "section": ParagraphStyle(
            "PMSection",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "PMSubsection",
            parent=sample["Heading3"],
            fontName=bold_font,
            fontSize=11,
            leading=15,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "PMBody",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9.3,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "PMBullet",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9.1,
            leading=12.8,
            textColor=TEXT,
            leftIndent=12,
            firstLineIndent=-7,
            spaceAfter=3,
        ),
        "bullet2": ParagraphStyle(
            "PMBullet2",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.9,
            leading=12.4,
            textColor=TEXT,
            leftIndent=24,
            firstLineIndent=-7,
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "PMSmall",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "footer": ParagraphStyle(
            "PMFooter",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def build_story(meeting: Meeting, styles: dict[str, ParagraphStyle]) -> list:
    story = [
        Paragraph("Project Manager Meeting Notes", styles["title"]),
        Paragraph(escape_text(meeting.title or "Untitled meeting"), styles["subtitle"]),
    ]
    if not meeting.minutes_text.strip():
        story.append(Paragraph("No project manager notes have been generated yet.", styles["body"]))
        return story

    lines = meeting.minutes_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 3))
            continue
        if stripped == "Meeting Details:":
            details, consumed = collect_meeting_details(lines[index + 1 :])
            story.append(Paragraph("Meeting Details", styles["section"]))
            if details:
                story.append(build_details_table(details, styles))
            continue
        if stripped == "Discussion Points:":
            story.append(Paragraph("Discussion Points", styles["section"]))
            continue
        if stripped == "Attendees:":
            story.append(Paragraph("Attendees", styles["section"]))
            continue
        if is_detail_line(stripped):
            continue
        if is_bullet(line):
            level = bullet_level(line)
            text = normalize_bullet_text(stripped)
            bullet_style = styles["bullet2"] if level > 0 else styles["bullet"]
            story.append(Paragraph(f"&bull; {escape_text(text)}", bullet_style))
        elif is_heading(stripped):
            story.append(Paragraph(escape_text(stripped), styles["subsection"]))
        else:
            story.append(Paragraph(escape_text(stripped), styles["body"]))

    return story


def collect_meeting_details(lines: list[str]) -> tuple[list[tuple[str, str]], int]:
    details = []
    consumed = 0
    for line in lines:
        consumed += 1
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Attendees:" or stripped == "Discussion Points:":
            break
        if is_detail_line(stripped):
            label, value = stripped.split(":", 1)
            details.append((label, value.strip() or "Not specified"))
    return details, consumed


def build_details_table(details: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [
            Paragraph(f"<b>{escape_text(label)}</b>", styles["small"]),
            Paragraph(escape_text(value), styles["body"]),
        ]
        for label, value in details
    ]
    table = Table(rows, colWidths=[34 * mm, 112 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, LIGHT_LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, LIGHT_LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def draw_page(canvas, doc, meeting: Meeting) -> None:
    width, height = A4
    canvas.saveState()
    draw_wordmark(canvas, 22 * mm, height - 17 * mm)
    canvas.setStrokeColor(LIGHT_LINE)
    canvas.setLineWidth(0.6)
    canvas.line(22 * mm, height - 24 * mm, width - 22 * mm, height - 24 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 22 * mm, height - 16 * mm, "Project Manager Meeting Notes")

    footer_y = 16 * mm
    canvas.setStrokeColor(LIGHT_LINE)
    canvas.line(22 * mm, footer_y + 9 * mm, width - 22 * mm, footer_y + 9 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, footer_y, (meeting.title or "Meeting Notes")[:62])
    canvas.drawCentredString(width / 2, footer_y, f"Page {doc.page}")
    canvas.drawRightString(width - 22 * mm, footer_y, "www.bit68.com")
    canvas.drawRightString(
        width - 22 * mm,
        footer_y - 9,
        f"Generated: {timezone.localtime(timezone.now()).strftime('%d/%m/%Y')}",
    )
    canvas.restoreState()


def draw_wordmark(canvas, x: float, y: float) -> None:
    canvas.setFont("Helvetica", 18)
    canvas.setFillColor(colors.black)
    canvas.drawString(x, y, "Bit")
    canvas.setFillColor(RED)
    canvas.drawString(x + 26, y, "68")


def is_bullet(line: str) -> bool:
    return line.lstrip().startswith(("- ", "* "))


def bullet_level(line: str) -> int:
    return max(0, (len(line) - len(line.lstrip(" "))) // 2)


def normalize_bullet_text(stripped: str) -> str:
    return stripped[2:].strip() if stripped[:2] in {"- ", "* "} else stripped


def is_detail_line(stripped: str) -> bool:
    return bool(re.match(r"^(Date|Time|Location/Platform):", stripped))


def is_heading(stripped: str) -> bool:
    return (
        len(stripped) <= 90
        and not stripped.endswith(".")
        and not stripped.startswith("[")
        and ":" not in stripped[:24]
    )


def escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
