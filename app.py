import os
import secrets
import time
from flask import Flask, jsonify, redirect, request, session, url_for
from routes.auth import auth_bp
from routes.plate import plate_bp
from routes.dashboard import dashboard_bp
from routes.monitoring import monitoring_bp
from routes.detections import detections_bp
from routes.history import history_bp
from routes.statistics import statistics_bp
from routes.recap import recap_bp
from routes.settings import settings_bp
from routes.zones import zones_bp
from routes.zones_page import zones_page_bp
from services.camera_worker_manager import CameraWorkerManager


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true",
    SESSION_IDLE_TIMEOUT_MINUTES=max(1, int(os.environ.get("SESSION_IDLE_TIMEOUT_MINUTES", "30"))),
)


@app.before_request
def require_login():
    if request.endpoint in {"auth.login", "auth.logout", "health", "static"}:
        return None
    if session.get("authenticated"):
        last_activity = session.get("last_activity")
        timeout_seconds = app.config["SESSION_IDLE_TIMEOUT_MINUTES"] * 60
        if isinstance(last_activity, (int, float)) and time.time() - last_activity < timeout_seconds:
            session["last_activity"] = time.time()
            return None
        session.clear()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required"}), 401
    return redirect(url_for("auth.login", next=request.full_path))


# ==============================
# REGISTER BLUEPRINT
# ==============================

app.register_blueprint(auth_bp)
app.register_blueprint(
    plate_bp,
    url_prefix="/api"
)
app.register_blueprint(dashboard_bp)
app.register_blueprint(monitoring_bp)
app.register_blueprint(detections_bp)
app.register_blueprint(history_bp)
app.register_blueprint(statistics_bp)
app.register_blueprint(recap_bp)   
app.register_blueprint(settings_bp)
app.register_blueprint(zones_bp)
app.register_blueprint(zones_page_bp)

# ==============================
# HALAMAN UTAMA
# ==============================

# ==============================
# START BACKGROUND DETECTION
# ==============================

if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    CameraWorkerManager.get_instance().start()

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

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
