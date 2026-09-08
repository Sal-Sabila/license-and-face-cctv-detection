from flask import Blueprint, render_template


settings_bp = Blueprint("settings_page", __name__)


@settings_bp.route("/settings")
def settings():
    return render_template("index.html", page="settings", heading="Pengaturan", subtitle="Atur operator dan koneksi CCTV.")
