from flask import Blueprint, render_template


history_bp = Blueprint("history_page", __name__)


@history_bp.route("/history")
def history():
    return render_template("index.html", page="history", heading="Riwayat Plat", subtitle="Riwayat pembacaan plat nomor dari kamera CCTV.")
