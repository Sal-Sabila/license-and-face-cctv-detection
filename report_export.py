# ============================================================
# report_export.py
# Export Excel & PDF untuk Sistem Monitoring CCTV
# ============================================================

import io
import os
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import (
    Font,
    PatternFill,
    Border,
    Side,
    Alignment,
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.utils import ImageReader


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCEL_MIMETYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

PDF_MIMETYPE = "application/pdf"


# ============================================================
# WARNA EXCEL
# ============================================================

DARK = "1F2937"
HEADER = "2563EB"
LIGHT_BLUE = "DBEAFE"
LIGHT_GREEN = "DCFCE7"
LIGHT_YELLOW = "FEF3C7"
LIGHT_RED = "FEE2E2"
LIGHT_GRAY = "F3F4F6"
WHITE = "FFFFFF"
TEXT = "111827"
BORDER_COLOR = "D1D5DB"


# ============================================================
# HELPER UMUM
# ============================================================

def _value(data, *keys, default=""):
    """
    Mengambil nilai dari dictionary berdasarkan beberapa kemungkinan key.
    Membantu agar export tetap aman jika nama field sedikit berbeda.
    """
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def _safe_text(value):
    if value is None:
        return "-"
    return str(value)


def _safe_number(value, default=0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence(value):
    """
    Normalisasi confidence.

    Database bisa menyimpan:
    0.85
    atau
    85
    """
    value = _safe_number(value)

    if value > 1:
        value = value / 100

    return value


def _confidence_text(value):
    return f"{_confidence(value) * 100:.1f}%"


def _status(value):
    if value is None or value == "":
        return "Perlu dicek"

    text = str(value).strip()

    if text.lower() in [
        "success",
        "terbaca",
        "jelas",
        "valid",
        "berhasil",
        "detected",
    ]:
        return "Terbaca"

    if text.lower() in [
        "perlu dicek",
        "check",
        "need_check",
        "kurang jelas",
        "unclear",
        "gagal",
        "failed",
    ]:
        return "Perlu dicek"

    return text


def _format_datetime(value):
    if value is None or value == "":
        return "-"

    if isinstance(value, datetime):
        return value.strftime("%d-%m-%Y %H:%M:%S")

    return str(value)


def _resolve_image_path(path):
    """
    Mencari file gambar berdasarkan path yang disimpan database.
    """
    if not path:
        return None

    path = str(path).strip()

    if not path:
        return None

    # Absolute path
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Normalisasi slash
    normalized = path.replace("/", os.sep).lstrip("\\/")

    candidates = [
        os.path.join(BASE_DIR, normalized),
        os.path.join(BASE_DIR, "static", normalized),
        os.path.join(BASE_DIR, "captures", normalized),
        os.path.join(BASE_DIR, "static", "captures", normalized),
        os.path.join(BASE_DIR, "static", "captures", "plates", normalized),
        os.path.join(BASE_DIR, "static", "captures", "faces", normalized),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Jika hanya basename yang tersimpan
    basename = os.path.basename(normalized)

    if basename:
        search_dirs = [
            os.path.join(BASE_DIR, "captures"),
            os.path.join(BASE_DIR, "static"),
            os.path.join(BASE_DIR, "static", "captures"),
            os.path.join(BASE_DIR, "static", "captures", "plates"),
            os.path.join(BASE_DIR, "static", "captures", "faces"),
        ]

        for directory in search_dirs:
            candidate = os.path.join(directory, basename)

            if os.path.exists(candidate):
                return candidate

    return None


# ============================================================
# EXCEL HELPER
# ============================================================

def _apply_excel_header(ws, row, start_col, end_col):
    fill = PatternFill("solid", fgColor=HEADER)

    for col in range(start_col, end_col + 1):
        cell = ws.cell(row=row, column=col)

        cell.fill = fill
        cell.font = Font(
            color=WHITE,
            bold=True,
            size=11,
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        cell.border = Border(
            bottom=Side(
                style="thin",
                color=BORDER_COLOR,
            )
        )


def _style_excel_sheet(ws):
    ws.freeze_panes = "A4"
    ws.sheet_view.showGridLines = False

    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(
                name="Calibri",
                size=10,
                color=TEXT,
            )
            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

            cell.border = Border(
                left=Side(style="thin", color=BORDER_COLOR),
                right=Side(style="thin", color=BORDER_COLOR),
                top=Side(style="thin", color=BORDER_COLOR),
                bottom=Side(style="thin", color=BORDER_COLOR),
            )


def _add_excel_title(ws, title, subtitle, columns):
    ws.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=columns,
    )

    title_cell = ws.cell(row=1, column=1)
    title_cell.value = title
    title_cell.font = Font(
        size=16,
        bold=True,
        color=WHITE,
    )
    title_cell.fill = PatternFill(
        "solid",
        fgColor=DARK,
    )
    title_cell.alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    ws.row_dimensions[1].height = 28

    ws.merge_cells(
        start_row=2,
        start_column=1,
        end_row=2,
        end_column=columns,
    )

    subtitle_cell = ws.cell(row=2, column=1)
    subtitle_cell.value = subtitle
    subtitle_cell.font = Font(
        size=10,
        color=TEXT,
        italic=True,
    )
    subtitle_cell.fill = PatternFill(
        "solid",
        fgColor=LIGHT_GRAY,
    )
    subtitle_cell.alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    ws.row_dimensions[2].height = 22


def _set_widths(ws, widths):
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[
            get_column_letter(index)
        ].width = width


def _status_excel_fill(cell, status):
    text = str(status).lower()

    if "terbaca" in text or "jelas" in text or "berhasil" in text:
        cell.fill = PatternFill(
            "solid",
            fgColor=LIGHT_GREEN,
        )

    elif "dicek" in text or "unclear" in text or "gagal" in text:
        cell.fill = PatternFill(
            "solid",
            fgColor=LIGHT_YELLOW,
        )


def _confidence_excel_fill(cell, confidence):
    value = _confidence(confidence)

    if value >= 0.70:
        cell.fill = PatternFill(
            "solid",
            fgColor=LIGHT_GREEN,
        )
    else:
        cell.fill = PatternFill(
            "solid",
            fgColor=LIGHT_YELLOW,
        )


def _add_excel_table(ws, ref, name):
    try:
        table = Table(
            displayName=name,
            ref=ref,
        )

        style = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )

        table.tableStyleInfo = style
        ws.add_table(table)
    except Exception:
        pass


def _finalize_excel(wb):
    buffer = io.BytesIO()

    wb.save(buffer)

    buffer.seek(0)

    return buffer


# ============================================================
# 1. HASIL DETEKSI - EXCEL
# ============================================================

def build_detections_excel(items):
    wb = Workbook()

    ws = wb.active
    ws.title = "Hasil Deteksi"

    columns = 8

    _add_excel_title(
        ws,
        "HASIL DETEKSI CCTV",
        "Laporan hasil deteksi wajah dan plat nomor kendaraan",
        columns,
    )

    headers = [
        "No",
        "Target",
        "Jenis Deteksi",
        "Area CCTV",
        "Confidence",
        "Waktu Deteksi",
        "Status",
        "Keterangan",
    ]

    header_row = 4

    for col, value in enumerate(headers, start=1):
        ws.cell(
            row=header_row,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws,
        header_row,
        1,
        columns,
    )

    for index, item in enumerate(items or [], start=1):
        target = _value(
            item,
            "target",
            "plate_number",
            "plate",
            "object_type",
            "type",
            default="Objek",
        )

        detection_type = _value(
            item,
            "type",
            "object_type",
            "detection_type",
            default="Deteksi",
        )

        location = _value(
            item,
            "camera_name",
            "camera",
            "location",
            "area",
            default="-",
        )

        confidence = _value(
            item,
            "plate_confidence",
            "face_confidence",
            "person_confidence",
            "confidence",
            default=0,
        )

        detected_at = _value(
            item,
            "detected_at",
            "detection_time",
            "created_at",
            "timestamp",
            default="-",
        )

        status = _status(
            _value(
                item,
                "plate_status",
                "face_status",
                "status",
                default="Perlu dicek",
            )
        )

        note = _value(
            item,
            "description",
            "keterangan",
            "message",
            default="-",
        )

        values = [
            index,
            _safe_text(target),
            _safe_text(detection_type),
            _safe_text(location),
            _confidence_text(confidence),
            _format_datetime(detected_at),
            status,
            _safe_text(note),
        ]

        row_number = header_row + index

        for col, value in enumerate(values, start=1):
            ws.cell(
                row=row_number,
                column=col,
                value=value,
            )

        _confidence_excel_fill(
            ws.cell(row=row_number, column=5),
            confidence,
        )

        _status_excel_fill(
            ws.cell(row=row_number, column=7),
            status,
        )

        ws.row_dimensions[row_number].height = 30

    last_row = max(header_row + len(items or []), header_row + 1)

    _add_excel_table(
        ws,
        f"A4:H{last_row}",
        "DetectionsTable",
    )

    _set_widths(
        ws,
        [8, 20, 18, 25, 15, 22, 18, 30],
    )

    _style_excel_sheet(ws)

    ws.auto_filter.ref = f"A4:H{last_row}"

    ws.print_title_rows = "1:4"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True

    return _finalize_excel(wb)


# ============================================================
# 2. HASIL DETEKSI - PDF
# ============================================================

def build_detections_pdf(items):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DetectionTitle",
        parent=styles["Title"],
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "DetectionSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
    )

    cell_style = ParagraphStyle(
        "DetectionCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
    )

    story = []

    story.append(
        Paragraph(
            "HASIL DETEKSI CCTV",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Laporan hasil deteksi wajah dan plat nomor kendaraan",
            subtitle_style,
        )
    )

    table_data = [
        [
            Paragraph("<b>No</b>", cell_style),
            Paragraph("<b>Foto</b>", cell_style),
            Paragraph("<b>Target</b>", cell_style),
            Paragraph("<b>Jenis</b>", cell_style),
            Paragraph("<b>Area CCTV</b>", cell_style),
            Paragraph("<b>Confidence</b>", cell_style),
            Paragraph("<b>Waktu</b>", cell_style),
            Paragraph("<b>Status</b>", cell_style),
        ]
    ]

    image_rows = []

    for index, item in enumerate(items or [], start=1):
        target = _value(
            item,
            "target",
            "plate_number",
            "plate",
            "object_type",
            default="Objek",
        )

        detection_type = _value(
            item,
            "type",
            "object_type",
            "detection_type",
            default="Deteksi",
        )

        location = _value(
            item,
            "camera_name",
            "camera",
            "location",
            "area",
            default="-",
        )

        confidence = _value(
            item,
            "plate_confidence",
            "face_confidence",
            "person_confidence",
            "confidence",
            default=0,
        )

        detected_at = _value(
            item,
            "detected_at",
            "detection_time",
            "created_at",
            default="-",
        )

        status = _status(
            _value(
                item,
                "plate_status",
                "face_status",
                "status",
                default="Perlu dicek",
            )
        )

        image_path = _value(
            item,
            "image_path",
            "capture_path",
            "photo",
            "photo_path",
            "plate_image_path",
            "face_image_path",
            default="",
        )

        resolved = _resolve_image_path(image_path)

        photo = Paragraph(
            "Tidak ada foto",
            cell_style,
        )

        if resolved:
            try:
                reader = ImageReader(resolved)
                width, height = reader.getSize()

                max_width = 32 * mm
                max_height = 22 * mm

                ratio = min(
                    max_width / width,
                    max_height / height,
                )

                photo = Image(
                    resolved,
                    width=width * ratio,
                    height=height * ratio,
                )
            except Exception:
                pass

        table_data.append(
            [
                str(index),
                photo,
                Paragraph(_safe_text(target), cell_style),
                Paragraph(_safe_text(detection_type), cell_style),
                Paragraph(_safe_text(location), cell_style),
                Paragraph(
                    _confidence_text(confidence),
                    cell_style,
                ),
                Paragraph(
                    _format_datetime(detected_at),
                    cell_style,
                ),
                Paragraph(
                    status,
                    cell_style,
                ),
            ]
        )

        image_rows.append(
            {
                "row": len(table_data) - 1,
                "confidence": _confidence(confidence),
                "status": status,
            }
        )

    table = Table(
        table_data,
        colWidths=[
            10 * mm,
            38 * mm,
            30 * mm,
            25 * mm,
            40 * mm,
            27 * mm,
            37 * mm,
            28 * mm,
        ],
        repeatRows=1,
    )

    table_style = [
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#2563EB"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "ALIGN",
            (0, 0),
            (-1, 0),
            "CENTER",
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.HexColor("#D1D5DB"),
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor("#F9FAFB"),
            ],
        ),
        (
            "ALIGN",
            (0, 1),
            (0, -1),
            "CENTER",
        ),
    ]

    for info in image_rows:
        row = info["row"]

        if (
            "terbaca" in info["status"].lower()
            or info["confidence"] >= 0.70
        ):
            table_style.append(
                (
                    "BACKGROUND",
                    (5, row),
                    (5, row),
                    colors.HexColor("#DCFCE7"),
                )
            )
        else:
            table_style.append(
                (
                    "BACKGROUND",
                    (5, row),
                    (5, row),
                    colors.HexColor("#FEF3C7"),
                )
            )

        if "terbaca" in info["status"].lower():
            table_style.append(
                (
                    "BACKGROUND",
                    (7, row),
                    (7, row),
                    colors.HexColor("#DCFCE7"),
                )
            )
        else:
            table_style.append(
                (
                    "BACKGROUND",
                    (7, row),
                    (7, row),
                    colors.HexColor("#FEF3C7"),
                )
            )

    table.setStyle(TableStyle(table_style))

    story.append(table)

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# 3. RIWAYAT PLAT - EXCEL
# ============================================================

def build_plates_excel(items):
    wb = Workbook()

    ws = wb.active
    ws.title = "Riwayat Plat"

    columns = 7

    _add_excel_title(
        ws,
        "RIWAYAT PLAT NOMOR",
        "Riwayat hasil pembacaan plat kendaraan dari CCTV",
        columns,
    )

    headers = [
        "No",
        "Nomor Plat",
        "Area CCTV",
        "Confidence",
        "Waktu Deteksi",
        "Status",
        "Keterangan",
    ]

    for col, value in enumerate(headers, start=1):
        ws.cell(
            row=4,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws,
        4,
        1,
        columns,
    )

    for index, item in enumerate(items or [], start=1):
        plate = _value(
            item,
            "plate_number",
            "plate",
            "formatted_text",
            "text",
            default="-",
        )

        location = _value(
            item,
            "camera_name",
            "camera",
            "location",
            "area",
            default="-",
        )

        confidence = _value(
            item,
            "plate_confidence",
            "confidence",
            default=0,
        )

        detected_at = _value(
            item,
            "detected_at",
            "detection_time",
            "created_at",
            default="-",
        )

        status = _status(
            _value(
                item,
                "plate_status",
                "status",
                default="Perlu dicek",
            )
        )

        note = _value(
            item,
            "description",
            "keterangan",
            "message",
            default="-",
        )

        values = [
            index,
            _safe_text(plate),
            _safe_text(location),
            _confidence_text(confidence),
            _format_datetime(detected_at),
            status,
            _safe_text(note),
        ]

        row_number = 4 + index

        for col, value in enumerate(values, start=1):
            ws.cell(
                row=row_number,
                column=col,
                value=value,
            )

        _confidence_excel_fill(
            ws.cell(row=row_number, column=4),
            confidence,
        )

        _status_excel_fill(
            ws.cell(row=row_number, column=6),
            status,
        )

        ws.row_dimensions[row_number].height = 28

    last_row = max(5, 4 + len(items or []))

    _add_excel_table(
        ws,
        f"A4:G{last_row}",
        "PlateHistoryTable",
    )

    _set_widths(
        ws,
        [8, 20, 30, 16, 23, 18, 30],
    )

    _style_excel_sheet(ws)

    ws.auto_filter.ref = f"A4:G{last_row}"

    ws.print_title_rows = "1:4"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True

    return _finalize_excel(wb)


# ============================================================
# 4. RIWAYAT PLAT - PDF
# ============================================================

def build_plates_pdf(items):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PlateTitle",
        parent=styles["Title"],
        fontSize=17,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "PlateSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
    )

    cell_style = ParagraphStyle(
        "PlateCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    story = [
        Paragraph(
            "RIWAYAT PLAT NOMOR",
            title_style,
        ),
        Paragraph(
            "Riwayat hasil pembacaan plat kendaraan dari CCTV",
            subtitle_style,
        ),
    ]

    data = [
        [
            Paragraph("<b>No</b>", cell_style),
            Paragraph("<b>Foto Plat</b>", cell_style),
            Paragraph("<b>Nomor Plat</b>", cell_style),
            Paragraph("<b>Area CCTV</b>", cell_style),
            Paragraph("<b>Confidence</b>", cell_style),
            Paragraph("<b>Waktu Deteksi</b>", cell_style),
            Paragraph("<b>Status</b>", cell_style),
        ]
    ]

    rows_info = []

    for index, item in enumerate(items or [], start=1):
        plate = _value(
            item,
            "plate_number",
            "plate",
            "formatted_text",
            "text",
            default="-",
        )

        location = _value(
            item,
            "camera_name",
            "camera",
            "location",
            "area",
            default="-",
        )

        confidence = _value(
            item,
            "plate_confidence",
            "confidence",
            default=0,
        )

        detected_at = _value(
            item,
            "detected_at",
            "detection_time",
            "created_at",
            default="-",
        )

        status = _status(
            _value(
                item,
                "plate_status",
                "status",
                default="Perlu dicek",
            )
        )

        image_path = _value(
            item,
            "plate_image_path",
            "image_path",
            "crop_path",
            "crop",
            "photo",
            "photo_path",
            default="",
        )

        resolved = _resolve_image_path(image_path)

        photo = Paragraph(
            "Tidak ada foto",
            cell_style,
        )

        if resolved:
            try:
                reader = ImageReader(resolved)
                width, height = reader.getSize()

                max_width = 38 * mm
                max_height = 22 * mm

                ratio = min(
                    max_width / width,
                    max_height / height,
                )

                photo = Image(
                    resolved,
                    width=width * ratio,
                    height=height * ratio,
                )
            except Exception:
                pass

        data.append(
            [
                str(index),
                photo,
                Paragraph(
                    f"<b>{_safe_text(plate)}</b>",
                    cell_style,
                ),
                Paragraph(
                    _safe_text(location),
                    cell_style,
                ),
                Paragraph(
                    _confidence_text(confidence),
                    cell_style,
                ),
                Paragraph(
                    _format_datetime(detected_at),
                    cell_style,
                ),
                Paragraph(
                    status,
                    cell_style,
                ),
            ]
        )

        rows_info.append(
            {
                "row": len(data) - 1,
                "confidence": _confidence(confidence),
                "status": status,
            }
        )

    table = Table(
        data,
        colWidths=[
            10 * mm,
            45 * mm,
            35 * mm,
            45 * mm,
            28 * mm,
            42 * mm,
            30 * mm,
        ],
        repeatRows=1,
    )

    table_style = [
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#2563EB"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "ALIGN",
            (0, 0),
            (-1, 0),
            "CENTER",
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.HexColor("#D1D5DB"),
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor("#F9FAFB"),
            ],
        ),
        (
            "ALIGN",
            (0, 1),
            (0, -1),
            "CENTER",
        ),
    ]

    for info in rows_info:
        row = info["row"]

        if info["confidence"] >= 0.70:
            table_style.append(
                (
                    "BACKGROUND",
                    (4, row),
                    (4, row),
                    colors.HexColor("#DCFCE7"),
                )
            )
        else:
            table_style.append(
                (
                    "BACKGROUND",
                    (4, row),
                    (4, row),
                    colors.HexColor("#FEF3C7"),
                )
            )

        if "terbaca" in info["status"].lower():
            table_style.append(
                (
                    "BACKGROUND",
                    (6, row),
                    (6, row),
                    colors.HexColor("#DCFCE7"),
                )
            )
        else:
            table_style.append(
                (
                    "BACKGROUND",
                    (6, row),
                    (6, row),
                    colors.HexColor("#FEF3C7"),
                )
            )

    table.setStyle(TableStyle(table_style))

    story.append(table)

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# 5. REKAPITULASI - EXCEL
# ============================================================

def build_recap_excel(analytics):
    wb = Workbook()

    summary = analytics.get("summary", {}) if isinstance(analytics, dict) else {}
    daily = analytics.get("daily", []) if isinstance(analytics, dict) else []
    cameras = analytics.get("cameras", []) if isinstance(analytics, dict) else []

    # --------------------------------------------------------
    # SHEET 1 - RINGKASAN
    # --------------------------------------------------------

    ws = wb.active
    ws.title = "Ringkasan"

    _add_excel_title(
        ws,
        "REKAPITULASI MONITORING CCTV",
        "Ringkasan hasil pemantauan dan deteksi CCTV",
        4,
    )

    headers = [
        "Indikator",
        "Jumlah",
        "Keterangan",
        "Periode",
    ]

    for col, value in enumerate(headers, start=1):
        ws.cell(row=4, column=col, value=value)

    _apply_excel_header(ws, 4, 1, 4)

    period = _value(
        analytics,
        "period",
        default="Semua periode",
    )

    summary_rows = [
        [
            "Total Kendaraan",
            summary.get("vehicles", 0),
            "Objek kendaraan terdeteksi",
            period,
        ],
        [
            "Total Wajah/Orang",
            summary.get("people", 0),
            "Objek wajah/orang terdeteksi",
            period,
        ],
        [
            "Total Plat",
            summary.get("plates", 0),
            "Plat nomor yang terbaca",
            period,
        ],
        [
            "Kamera Aktif",
            summary.get("active_cameras", 0),
            "Kamera dengan status aktif",
            period,
        ],
        [
            "Total Kamera",
            summary.get("total_cameras", 0),
            "Seluruh kamera terdaftar",
            period,
        ],
    ]

    for row_index, row_data in enumerate(summary_rows, start=5):
        for col, value in enumerate(row_data, start=1):
            ws.cell(
                row=row_index,
                column=col,
                value=value,
            )

    _set_widths(
        ws,
        [25, 16, 40, 20],
    )

    _style_excel_sheet(ws)

    # --------------------------------------------------------
    # SHEET 2 - REKAP CCTV
    # --------------------------------------------------------

    ws_camera = wb.create_sheet("Rekap CCTV")

    _add_excel_title(
        ws_camera,
        "REKAPITULASI PER CCTV",
        "Ringkasan aktivitas deteksi berdasarkan lokasi kamera",
        6,
    )

    camera_headers = [
        "No",
        "CCTV / Lokasi",
        "Kendaraan",
        "Wajah/Orang",
        "Plat Unik",
        "Total Deteksi",
    ]

    for col, value in enumerate(camera_headers, start=1):
        ws_camera.cell(
            row=4,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws_camera,
        4,
        1,
        6,
    )

    for index, camera in enumerate(cameras or [], start=1):
        vehicles = _safe_number(
            _value(camera, "vehicles", default=0)
        )

        people = _safe_number(
            _value(camera, "people", default=0)
        )

        unique_plates = _safe_number(
            _value(camera, "unique_plates", default=0)
        )

        total = vehicles + people

        values = [
            index,
            _value(
                camera,
                "camera",
                "camera_name",
                "location",
                default="-",
            ),
            int(vehicles),
            int(people),
            int(unique_plates),
            int(total),
        ]

        row_number = 4 + index

        for col, value in enumerate(values, start=1):
            ws_camera.cell(
                row=row_number,
                column=col,
                value=value,
            )

    last_row = max(
        5,
        4 + len(cameras or []),
    )

    _add_excel_table(
        ws_camera,
        f"A4:F{last_row}",
        "RecapCameraTable",
    )

    _set_widths(
        ws_camera,
        [8, 30, 18, 18, 18, 20],
    )

    _style_excel_sheet(ws_camera)

    # --------------------------------------------------------
    # SHEET 3 - TOTAL HARIAN
    # --------------------------------------------------------

    ws_daily = wb.create_sheet("Total Harian")

    _add_excel_title(
        ws_daily,
        "TOTAL DETEKSI HARIAN",
        "Rekap jumlah kendaraan dan wajah/orang berdasarkan tanggal",
        4,
    )

    daily_headers = [
        "No",
        "Tanggal",
        "Kendaraan",
        "Wajah/Orang",
    ]

    for col, value in enumerate(daily_headers, start=1):
        ws_daily.cell(
            row=4,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws_daily,
        4,
        1,
        4,
    )

    for index, row in enumerate(daily or [], start=1):
        values = [
            index,
            _value(
                row,
                "date",
                "tanggal",
                default="-",
            ),
            _value(
                row,
                "vehicles",
                "vehicle",
                default=0,
            ),
            _value(
                row,
                "people",
                "persons",
                "faces",
                default=0,
            ),
        ]

        row_number = 4 + index

        for col, value in enumerate(values, start=1):
            ws_daily.cell(
                row=row_number,
                column=col,
                value=value,
            )

    last_row = max(
        5,
        4 + len(daily or []),
    )

    _add_excel_table(
        ws_daily,
        f"A4:D{last_row}",
        "DailyRecapTable",
    )

    _set_widths(
        ws_daily,
        [8, 22, 20, 20],
    )

    _style_excel_sheet(ws_daily)

    return _finalize_excel(wb)


# ============================================================
# 6. REKAPITULASI - PDF
# ============================================================

def build_recap_pdf(analytics):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RecapTitle",
        parent=styles["Title"],
        fontSize=17,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "RecapSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )

    section_style = ParagraphStyle(
        "RecapSection",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    cell_style = ParagraphStyle(
        "RecapCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    summary = analytics.get("summary", {})
    cameras = analytics.get("cameras", [])
    daily = analytics.get("daily", [])

    story = [
        Paragraph(
            "REKAPITULASI MONITORING CCTV",
            title_style,
        ),
        Paragraph(
            "Ringkasan hasil pemantauan dan deteksi CCTV",
            subtitle_style,
        ),
    ]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Ringkasan",
            section_style,
        )
    )

    summary_data = [
        ["Indikator", "Jumlah"],
        [
            "Total Kendaraan",
            str(summary.get("vehicles", 0)),
        ],
        [
            "Total Wajah/Orang",
            str(summary.get("people", 0)),
        ],
        [
            "Total Plat",
            str(summary.get("plates", 0)),
        ],
        [
            "Kamera Aktif",
            str(summary.get("active_cameras", 0)),
        ],
        [
            "Total Kamera",
            str(summary.get("total_cameras", 0)),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            100 * mm,
            45 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(summary_table)

    # --------------------------------------------------------
    # REKAP CCTV
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Rekapitulasi CCTV",
            section_style,
        )
    )

    camera_data = [
        [
            Paragraph("<b>No</b>", cell_style),
            Paragraph("<b>CCTV / Lokasi</b>", cell_style),
            Paragraph("<b>Kendaraan</b>", cell_style),
            Paragraph("<b>Wajah/Orang</b>", cell_style),
            Paragraph("<b>Plat Unik</b>", cell_style),
            Paragraph("<b>Total</b>", cell_style),
        ]
    ]

    for index, camera in enumerate(cameras or [], start=1):
        vehicles = int(
            _safe_number(
                _value(camera, "vehicles", default=0)
            )
        )

        people = int(
            _safe_number(
                _value(camera, "people", default=0)
            )
        )

        unique_plates = int(
            _safe_number(
                _value(camera, "unique_plates", default=0)
            )
        )

        camera_data.append(
            [
                str(index),
                _safe_text(
                    _value(
                        camera,
                        "camera",
                        "camera_name",
                        "location",
                        default="-",
                    )
                ),
                str(vehicles),
                str(people),
                str(unique_plates),
                str(vehicles + people),
            ]
        )

    camera_table = Table(
        camera_data,
        colWidths=[
            12 * mm,
            60 * mm,
            25 * mm,
            25 * mm,
            25 * mm,
            25 * mm,
        ],
        repeatRows=1,
    )

    camera_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(camera_table)

    # --------------------------------------------------------
    # TOTAL HARIAN
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Total Harian",
            section_style,
        )
    )

    daily_data = [
        [
            Paragraph("<b>No</b>", cell_style),
            Paragraph("<b>Tanggal</b>", cell_style),
            Paragraph("<b>Kendaraan</b>", cell_style),
            Paragraph("<b>Wajah/Orang</b>", cell_style),
        ]
    ]

    for index, row in enumerate(daily or [], start=1):
        daily_data.append(
            [
                str(index),
                _safe_text(
                    _value(
                        row,
                        "date",
                        "tanggal",
                        default="-",
                    )
                ),
                str(
                    _value(
                        row,
                        "vehicles",
                        "vehicle",
                        default=0,
                    )
                ),
                str(
                    _value(
                        row,
                        "people",
                        "persons",
                        "faces",
                        default=0,
                    )
                ),
            ]
        )

    daily_table = Table(
        daily_data,
        colWidths=[
            15 * mm,
            65 * mm,
            45 * mm,
            45 * mm,
        ],
        repeatRows=1,
    )

    daily_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(daily_table)

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# 7. STATISTIK - EXCEL
#    1 FILE = 2 SHEET
#    Sheet 1: Statistik
#    Sheet 2: Top 10
# ============================================================

def build_statistics_excel(analytics, period="today"):
    wb = Workbook()

    summary = analytics.get("summary", {})
    hourly = analytics.get("hourly", [])
    cameras = analytics.get("cameras", [])
    top_plates = analytics.get("top_plates", [])

    # --------------------------------------------------------
    # SHEET STATISTIK
    # --------------------------------------------------------

    ws = wb.active
    ws.title = "Statistik"

    _add_excel_title(
        ws,
        "STATISTIK MONITORING CCTV",
        f"Periode statistik: {period}",
        5,
    )

    headers = [
        "Indikator",
        "Jumlah",
        "Keterangan",
        "Periode",
        "Sumber",
    ]

    for col, value in enumerate(headers, start=1):
        ws.cell(row=4, column=col, value=value)

    _apply_excel_header(ws, 4, 1, 5)

    rows = [
        [
            "Total Kendaraan",
            summary.get("vehicles", 0),
            "Jumlah kendaraan terdeteksi",
            period,
            "Sistem CCTV",
        ],
        [
            "Total Wajah/Orang",
            summary.get("people", 0),
            "Jumlah wajah/orang terdeteksi",
            period,
            "Sistem CCTV",
        ],
        [
            "Total Plat",
            summary.get("plates", 0),
            "Jumlah plat terdeteksi",
            period,
            "OCR Plat",
        ],
        [
            "Kamera Aktif",
            summary.get("active_cameras", 0),
            "Kamera aktif",
            period,
            "Manajemen Kamera",
        ],
        [
            "Total Kamera",
            summary.get("total_cameras", 0),
            "Seluruh kamera",
            period,
            "Manajemen Kamera",
        ],
    ]

    for index, row in enumerate(rows, start=5):
        for col, value in enumerate(row, start=1):
            ws.cell(
                row=index,
                column=col,
                value=value,
            )

    _set_widths(
        ws,
        [25, 16, 40, 18, 25],
    )

    _style_excel_sheet(ws)

    # --------------------------------------------------------
    # JAM SIBUK
    # --------------------------------------------------------

    start_row = 12

    ws.cell(
        row=start_row,
        column=1,
        value="BEBAN LALU LINTAS PER JAM",
    )

    ws.merge_cells(
        start_row=start_row,
        start_column=1,
        end_row=start_row,
        end_column=3,
    )

    ws.cell(
        row=start_row,
        column=1,
    ).font = Font(
        bold=True,
        size=12,
        color=WHITE,
    )

    ws.cell(
        row=start_row,
        column=1,
    ).fill = PatternFill(
        "solid",
        fgColor=DARK,
    )

    hourly_header = start_row + 1

    hourly_headers = [
        "Jam",
        "Kendaraan",
        "Keterangan",
    ]

    for col, value in enumerate(hourly_headers, start=1):
        ws.cell(
            row=hourly_header,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws,
        hourly_header,
        1,
        3,
    )

    for index, hour in enumerate(hourly or [], start=1):
        vehicles = _value(
            hour,
            "vehicles",
            "count",
            "total",
            default=0,
        )

        label = _value(
            hour,
            "label",
            "hour",
            "jam",
            default="-",
        )

        values = [
            label,
            vehicles,
            "Aktivitas kendaraan",
        ]

        row_number = hourly_header + index

        for col, value in enumerate(values, start=1):
            ws.cell(
                row=row_number,
                column=col,
                value=value,
            )

    # --------------------------------------------------------
    # PERFORMA CCTV
    # --------------------------------------------------------

    camera_start = hourly_header + len(hourly or []) + 3

    ws.cell(
        row=camera_start,
        column=1,
        value="PERFORMA CCTV",
    )

    ws.merge_cells(
        start_row=camera_start,
        start_column=1,
        end_row=camera_start,
        end_column=5,
    )

    ws.cell(
        row=camera_start,
        column=1,
    ).font = Font(
        bold=True,
        size=12,
        color=WHITE,
    )

    ws.cell(
        row=camera_start,
        column=1,
    ).fill = PatternFill(
        "solid",
        fgColor=DARK,
    )

    camera_header = camera_start + 1

    camera_headers = [
        "No",
        "CCTV",
        "Kendaraan",
        "Wajah/Orang",
        "Plat Unik",
    ]

    for col, value in enumerate(camera_headers, start=1):
        ws.cell(
            row=camera_header,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws,
        camera_header,
        1,
        5,
    )

    for index, camera in enumerate(cameras or [], start=1):
        row_number = camera_header + index

        values = [
            index,
            _value(
                camera,
                "camera",
                "camera_name",
                "location",
                default="-",
            ),
            _value(camera, "vehicles", default=0),
            _value(camera, "people", default=0),
            _value(camera, "unique_plates", default=0),
        ]

        for col, value in enumerate(values, start=1):
            ws.cell(
                row=row_number,
                column=col,
                value=value,
            )

    _style_excel_sheet(ws)

    # --------------------------------------------------------
    # SHEET TOP 10
    # --------------------------------------------------------

    ws_top = wb.create_sheet("Top 10")

    _add_excel_title(
        ws_top,
        "TOP 10 PLAT TERBANYAK TERDETEKSI",
        f"Daftar plat berdasarkan jumlah kemunculan — periode {period}",
        6,
    )

    top_headers = [
        "Ranking",
        "Nomor Plat",
        "Jumlah Deteksi",
        "CCTV",
        "Terakhir Terlihat",
        "Confidence",
    ]

    for col, value in enumerate(top_headers, start=1):
        ws_top.cell(
            row=4,
            column=col,
            value=value,
        )

    _apply_excel_header(
        ws_top,
        4,
        1,
        6,
    )

    for index, plate in enumerate(
        (top_plates or [])[:10],
        start=1,
    ):
        values = [
            index,
            _value(
                plate,
                "plate",
                "plate_number",
                default="-",
            ),
            _value(
                plate,
                "count",
                "total",
                "frequency",
                default=0,
            ),
            _value(
                plate,
                "camera",
                "camera_name",
                "location",
                default="-",
            ),
            _format_datetime(
                _value(
                    plate,
                    "last_seen",
                    "detected_at",
                    default="-",
                )
            ),
            _confidence_text(
                _value(
                    plate,
                    "confidence",
                    "plate_confidence",
                    default=0,
                )
            ),
        ]

        row_number = 4 + index

        for col, value in enumerate(values, start=1):
            ws_top.cell(
                row=row_number,
                column=col,
                value=value,
            )

        _confidence_excel_fill(
            ws_top.cell(
                row=row_number,
                column=6,
            ),
            _value(
                plate,
                "confidence",
                "plate_confidence",
                default=0,
            ),
        )

    _set_widths(
        ws_top,
        [12, 22, 20, 30, 25, 18],
    )

    _style_excel_sheet(ws_top)

    return _finalize_excel(wb)


# ============================================================
# 8. STATISTIK - PDF
#    SATU DOKUMEN
#    Statistik utama
#    Beban lalu lintas
#    Jam sibuk
#    Top 10
# ============================================================

def build_statistics_pdf(analytics, period="today"):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "StatisticsTitle",
        parent=styles["Title"],
        fontSize=17,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "StatisticsSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )

    section_style = ParagraphStyle(
        "StatisticsSection",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    cell_style = ParagraphStyle(
        "StatisticsCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    summary = analytics.get("summary", {})
    hourly = analytics.get("hourly", [])
    cameras = analytics.get("cameras", [])
    top_plates = analytics.get("top_plates", [])

    story = [
        Paragraph(
            "STATISTIK MONITORING CCTV",
            title_style,
        ),
        Paragraph(
            f"Periode statistik: {period}",
            subtitle_style,
        ),
    ]

    # --------------------------------------------------------
    # STATISTIK UTAMA
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Statistik Utama",
            section_style,
        )
    )

    summary_data = [
        [
            "Indikator",
            "Jumlah",
        ],
        [
            "Total Kendaraan",
            str(summary.get("vehicles", 0)),
        ],
        [
            "Total Wajah/Orang",
            str(summary.get("people", 0)),
        ],
        [
            "Total Plat",
            str(summary.get("plates", 0)),
        ],
        [
            "Kamera Aktif",
            str(summary.get("active_cameras", 0)),
        ],
        [
            "Total Kamera",
            str(summary.get("total_cameras", 0)),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            105 * mm,
            50 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(summary_table)

    # --------------------------------------------------------
    # BEBAN LALU LINTAS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Beban Lalu Lintas per Jam",
            section_style,
        )
    )

    hourly_data = [
        [
            Paragraph("<b>Jam</b>", cell_style),
            Paragraph("<b>Jumlah Kendaraan</b>", cell_style),
            Paragraph("<b>Keterangan</b>", cell_style),
        ]
    ]

    for hour in hourly or []:
        hourly_data.append(
            [
                _safe_text(
                    _value(
                        hour,
                        "label",
                        "hour",
                        "jam",
                        default="-",
                    )
                ),
                str(
                    _value(
                        hour,
                        "vehicles",
                        "count",
                        "total",
                        default=0,
                    )
                ),
                "Aktivitas kendaraan",
            ]
        )

    hourly_table = Table(
        hourly_data,
        colWidths=[
            40 * mm,
            55 * mm,
            70 * mm,
        ],
        repeatRows=1,
    )

    hourly_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(hourly_table)

    # --------------------------------------------------------
    # PERFORMA CCTV
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Performa CCTV",
            section_style,
        )
    )

    camera_data = [
        [
            Paragraph("<b>No</b>", cell_style),
            Paragraph("<b>CCTV</b>", cell_style),
            Paragraph("<b>Kendaraan</b>", cell_style),
            Paragraph("<b>Wajah/Orang</b>", cell_style),
            Paragraph("<b>Plat Unik</b>", cell_style),
        ]
    ]

    for index, camera in enumerate(cameras or [], start=1):
        camera_data.append(
            [
                str(index),
                _safe_text(
                    _value(
                        camera,
                        "camera",
                        "camera_name",
                        "location",
                        default="-",
                    )
                ),
                str(
                    _value(
                        camera,
                        "vehicles",
                        default=0,
                    )
                ),
                str(
                    _value(
                        camera,
                        "people",
                        default=0,
                    )
                ),
                str(
                    _value(
                        camera,
                        "unique_plates",
                        default=0,
                    )
                ),
            ]
        )

    camera_table = Table(
        camera_data,
        colWidths=[
            15 * mm,
            60 * mm,
            35 * mm,
            35 * mm,
            35 * mm,
        ],
        repeatRows=1,
    )

    camera_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2563EB"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F9FAFB"),
                    ],
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(camera_table)

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Top 10 Plat Terbanyak Terdeteksi",
            section_style,
        )
    )

    top_data = [
        [
            Paragraph("<b>Ranking</b>", cell_style),
            Paragraph("<b>Nomor Plat</b>", cell_style),
            Paragraph("<b>Jumlah</b>", cell_style),
            Paragraph("<b>CCTV</b>", cell_style),
            Paragraph("<b>Terakhir Terlihat</b>", cell_style),
            Paragraph("<b>Confidence</b>", cell_style),
        ]
    ]

    for index, plate in enumerate(
        (top_plates or [])[:10],
        start=1,
    ):
        top_data.append(
            [
                str(index),
                _safe_text(
                    _value(
                        plate,
                        "plate",
                        "plate_number",
                        default="-",
                    )
                ),
                str(
                    _value(
                        plate,
                        "count",
                        "total",
                        "frequency",
                        default=0,
                    )
                ),
                _safe_text(
                    _value(
                        plate,
                        "camera",
                        "camera_name",
                        "location",
                        default="-",
                    )
                ),
                _format_datetime(
                    _value(
                        plate,
                        "last_seen",
                        "detected_at",
                        default="-",
                    )
                ),
                _confidence_text(
                    _value(
                        plate,
                        "confidence",
                        "plate_confidence",
                        default=0,
                    )
                ),
            ]
        )

    top_table = Table(
        top_data,
        colWidths=[
            18 * mm,
            32 * mm,
            20 * mm,
            35 * mm,
            40 * mm,
            28 * mm,
        ],
        repeatRows=1,
    )

    top_style = [
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#2563EB"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.4,
            colors.HexColor("#D1D5DB"),
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor("#F9FAFB"),
            ],
        ),
        (
            "ALIGN",
            (0, 0),
            (-1, -1),
            "CENTER",
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),
    ]

    top_table.setStyle(TableStyle(top_style))

    story.append(top_table)

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# 9. MONITORING CCTV - EXCEL
#    Data berasal dari db.get_all_cameras() -- fungsi yang sama
#    dipakai endpoint GET /cameras -- supaya data website, Excel,
#    dan PDF selalu konsisten. Tidak ada foto/preview kamera.
# ============================================================

def _camera_status(value):
    """Ambil label status kamera dari data backend, jangan bikin status baru."""
    if value is None or value == "":
        return "Nonaktif"

    if isinstance(value, str):
        text = value.strip()
        return text if text else "Nonaktif"

    return "Aktif" if value else "Nonaktif"


def _camera_status_excel_fill(cell, status):
    text = str(status).lower()

    if "non" in text:
        cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY)
    elif "error" in text:
        cell.fill = PatternFill("solid", fgColor=LIGHT_RED)
    elif "aktif" in text:
        cell.fill = PatternFill("solid", fgColor=LIGHT_GREEN)
    else:
        cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY)


def build_cameras_excel(cameras_db):
    wb = Workbook()

    ws = wb.active
    ws.title = "Monitoring CCTV"

    columns = 4

    _add_excel_title(
        ws,
        "LAPORAN MONITORING CCTV",
        f"Diekspor pada {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} "
        f"— Jumlah kamera: {len(cameras_db or [])}",
        columns,
    )

    headers = [
        "No",
        "Nama CCTV",
        "URL Stream",
        "Status",
    ]

    for col, value in enumerate(headers, start=1):
        ws.cell(row=4, column=col, value=value)

    _apply_excel_header(ws, 4, 1, columns)

    for index, camera in enumerate(cameras_db or [], start=1):
        name = _value(
            camera,
            "location",
            "name",
            "camera_name",
            default="-",
        )

        url = _value(
            camera,
            "stream_url",
            "url",
            default="-",
        )

        status = _camera_status(
            _value(
                camera,
                "status",
                "active",
                default=None,
            )
        )

        values = [
            index,
            _safe_text(name),
            _safe_text(url),
            status,
        ]

        row_number = 4 + index

        for col, value in enumerate(values, start=1):
            ws.cell(row=row_number, column=col, value=value)

        _camera_status_excel_fill(
            ws.cell(row=row_number, column=4),
            status,
        )

        ws.row_dimensions[row_number].height = 22

    last_row = max(5, 4 + len(cameras_db or []))

    _add_excel_table(
        ws,
        f"A4:D{last_row}",
        "CamerasTable",
    )

    _set_widths(ws, [8, 30, 60, 16])

    _style_excel_sheet(ws)

    ws.auto_filter.ref = f"A4:D{last_row}"

    return _finalize_excel(wb)


# ============================================================
# 10. MONITORING CCTV - PDF
# ============================================================

def build_cameras_pdf(cameras_db):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CamerasTitle",
        parent=styles["Title"],
        fontSize=17,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "CamerasSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )

    cell_style = ParagraphStyle(
        "CamerasCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    cameras_db = cameras_db or []

    story = [
        Paragraph("LAPORAN MONITORING CCTV", title_style),
        Paragraph(
            f"Tanggal/Waktu Laporan: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} "
            f"— Jumlah Kamera CCTV: {len(cameras_db)}",
            subtitle_style,
        ),
    ]

    if not cameras_db:
        story.append(
            Paragraph("Tidak ada data kamera CCTV.", styles["Normal"])
        )
    else:
        data = [
            [
                Paragraph("<b>No</b>", cell_style),
                Paragraph("<b>Nama CCTV</b>", cell_style),
                Paragraph("<b>URL Stream</b>", cell_style),
                Paragraph("<b>Status</b>", cell_style),
            ]
        ]

        rows_info = []

        for index, camera in enumerate(cameras_db, start=1):
            name = _value(
                camera,
                "location",
                "name",
                "camera_name",
                default="-",
            )

            url = _value(
                camera,
                "stream_url",
                "url",
                default="-",
            )

            status = _camera_status(
                _value(
                    camera,
                    "status",
                    "active",
                    default=None,
                )
            )

            data.append(
                [
                    str(index),
                    Paragraph(_safe_text(name), cell_style),
                    Paragraph(_safe_text(url), cell_style),
                    Paragraph(status, cell_style),
                ]
            )

            rows_info.append({"row": len(data) - 1, "status": status})

        table = Table(
            data,
            colWidths=[
                12 * mm,
                45 * mm,
                95 * mm,
                28 * mm,
            ],
            repeatRows=1,
        )

        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ALIGN", (3, 1), (3, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#F9FAFB")],
            ),
        ]

        for info in rows_info:
            row = info["row"]
            text = info["status"].lower()

            if "non" in text:
                fill = colors.HexColor("#F3F4F6")
            elif "error" in text:
                fill = colors.HexColor("#FEE2E2")
            elif "aktif" in text:
                fill = colors.HexColor("#DCFCE7")
            else:
                fill = colors.HexColor("#F3F4F6")

            table_style.append(
                ("BACKGROUND", (3, row), (3, row), fill)
            )

        table.setStyle(TableStyle(table_style))
        story.append(table)

    doc.build(story)

    buffer.seek(0)

    return buffer