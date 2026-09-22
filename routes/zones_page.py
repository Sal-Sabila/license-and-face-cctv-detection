from flask import Blueprint, render_template

zones_page_bp = Blueprint("zones_editor", __name__)


@zones_page_bp.route("/zones")
def zones_editor():
    return render_template(
        "index.html",
        page="zones",
        heading="Editor Zona Deteksi",
        subtitle="Atur zona deteksi MID & NEAR untuk setiap kamera CCTV."
    )