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
# RUN SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )