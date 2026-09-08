from flask import Blueprint, render_template


monitoring_bp = Blueprint("monitoring_page", __name__)


@monitoring_bp.route("/monitoring")
def monitoring():
    return render_template("index.html", page="monitoring", heading="Monitoring CCTV", subtitle="Kelola daftar kamera playlist yang digunakan sistem.")
