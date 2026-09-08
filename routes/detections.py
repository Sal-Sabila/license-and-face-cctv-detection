from flask import Blueprint, render_template


detections_bp = Blueprint("detections_page", __name__)


@detections_bp.route("/detections")
def detections():
    return render_template("index.html", page="detections", heading="Hasil Deteksi", subtitle="Hasil deteksi wajah dan plat nomor terbaru.")
