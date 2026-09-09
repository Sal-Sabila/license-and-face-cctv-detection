from flask import Flask
from routes.plate import plate_bp
from routes.dashboard import dashboard_bp
from routes.monitoring import monitoring_bp
from routes.detections import detections_bp
from routes.history import history_bp
from routes.statistics import statistics_bp
from routes.settings import settings_bp


app = Flask(__name__)


# ==============================
# REGISTER BLUEPRINT
# ==============================

app.register_blueprint(
    plate_bp,
    url_prefix="/api"
)
app.register_blueprint(dashboard_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(detections_bp)
app.register_blueprint(history_bp)
app.register_blueprint(statistics_bp)
app.register_blueprint(settings_bp)


# ==============================
# HALAMAN UTAMA
# ==============================

# ==============================
# HEALTH CHECK
# ==============================

@app.route("/health")
def health():
    return {
        "status": "ok",
        "service": "license-and-face-cctv-detection",
        "message": "Flask server berjalan"
    }


# ==============================
# START BACKGROUND CCTV DETECTOR
# ==============================

import os
from services.background_detector import BackgroundDetectionManager

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    try:
        BackgroundDetectionManager.get_instance().start()
    except Exception as e:
        print(f"[APP WARNING] Gagal memulai BackgroundDetectionManager: {e}")


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    # Jika dijalankan langsung tanpa reloader, pastikan service aktif
    if not app.debug:
        BackgroundDetectionManager.get_instance().start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )