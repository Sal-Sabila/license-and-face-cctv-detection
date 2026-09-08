from flask import Blueprint, render_template


dashboard_bp = Blueprint("dashboard_page", __name__)


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
def dashboard():
    return render_template("index.html", page="dashboard", heading="Dashboard", subtitle="Monitoring wajah dan plat nomor secara realtime.")
