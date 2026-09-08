from flask import Blueprint, render_template


statistics_bp = Blueprint("statistics_page", __name__)


@statistics_bp.route("/statistics")
def statistics():
    return render_template("index.html", page="statistics", heading="Statistik", subtitle="Ringkasan performa deteksi CCTV.")
