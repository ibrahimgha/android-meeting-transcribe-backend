import re
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import Meeting


BLACK = colors.black
HEADER_HEIGHT = 68
BODY_MARGIN_X = 37
TOP_MARGIN = 91
BOTTOM_MARGIN = 78
LOGO_DIR = Path(settings.BASE_DIR) / "meetings" / "static" / "meetings"
HEADER_LOGO_PATH = LOGO_DIR / "organization-logo.jpg"
FOOTER_LOGO_PATH = LOGO_DIR / "organization-logo-footer.jpg"
MUTED = colors.HexColor("#5b6475")
LIGHT_LINE = colors.HexColor("#d9e0ec")
LIGHT_BG = colors.HexColor("#f7f9fd")
TITLE_BAND = colors.HexColor("#d9d9d9")
TEXT = colors.HexColor("#182033")


def build_pm_notes_pdf(meeting: Meeting, *, minutes_text: str | None = None) -> bytes:
    buffer = BytesIO()
    register_fonts()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=BODY_MARGIN_X,
        leftMargin=BODY_MARGIN_X,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=f"{meeting.title or 'Meeting'} - Project Manager Notes",
        author=settings.PM_NOTES_PDF_AUTHOR,
    )
    styles = build_styles()
    story = build_story(meeting, styles, minutes_text=minutes_text)
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
            fontSize=14,
            leading=17,
            textColor=BLACK,
            alignment=TA_CENTER,
            spaceAfter=0,
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
            textColor=BLACK,
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
            textColor=BLACK,
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


def build_story(meeting: Meeting, styles: dict[str, ParagraphStyle], *, minutes_text: str | None = None) -> list:
    story = [
        build_title_band("Project Manager Meeting Notes", styles),
        Spacer(1, 14),
        Paragraph(escape_text(meeting.title or "Untitled meeting"), styles["subtitle"]),
    ]
    notes_text = meeting.minutes_text if minutes_text is None else minutes_text
    if not notes_text.strip():
        story.append(Paragraph("No project manager notes have been generated yet.", styles["body"]))
        return story

    lines = notes_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
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


def build_title_band(title: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [[Paragraph(escape_text(title), styles["title"])]],
        colWidths=[A4[0] - (BODY_MARGIN_X * 2)],
        rowHeights=[36],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TITLE_BAND),
                ("BOX", (0, 0), (-1, -1), 0, TITLE_BAND),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


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
    draw_header(canvas, doc)
    draw_footer(canvas, meeting)
    canvas.restoreState()


def draw_header(canvas, doc) -> None:
    width, height = A4
    canvas.setFillColor(colors.black)
    canvas.rect(0, height - HEADER_HEIGHT, width, HEADER_HEIGHT, fill=1, stroke=0)
    draw_logo_image(canvas, HEADER_LOGO_PATH, 34, height - 53, 88, 40)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 10)
    canvas.drawRightString(width - 32, height - 30, "Project Manager Meeting Notes")
    canvas.drawRightString(width - 32, height - 45, f"Page {doc.page}")


def draw_footer(canvas, meeting: Meeting) -> None:
    width, _ = A4
    issued_at = timezone.localtime(timezone.now()).strftime("%d/%m/%Y")
    meeting_date = timezone.localtime(meeting.started_at).strftime("%d/%m/%Y") if meeting.started_at else issued_at

    draw_logo_image(canvas, FOOTER_LOGO_PATH, 36, 35, 39, 18)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    if settings.PM_NOTES_PDF_FOOTER_TEXT:
        canvas.drawString(36, 20, settings.PM_NOTES_PDF_FOOTER_TEXT)

    canvas.drawRightString(width - 37, 45, f"Date issued: {issued_at}")
    canvas.drawRightString(width - 37, 32, f"Meeting date: {meeting_date}")
    canvas.drawRightString(width - 37, 19, f"Serial Number: {str(meeting.id).split('-')[0].upper()}")


def draw_logo_image(canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    if not path.exists():
        return
    canvas.drawImage(ImageReader(str(path)), x, y, width=width, height=height, preserveAspectRatio=True, mask="auto")


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
