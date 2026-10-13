import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager

def test_flask_integration():
    print("\n" + "=" * 60)
    print("TEST 3: Integrasi Flask & Jalur Deteksi Real-Time")
    print("=" * 60)

    client = app.test_client()

    # 1. Health check
    res_health = client.get("/health")
    print(f"[TEST FLASK] /health status_code: {res_health.status_code}, data: {res_health.get_json()}")
    assert res_health.status_code == 200
    assert res_health.get_json()["status"] == "ok"

    # Authenticate session
    with client.session_transaction() as sess:
        sess["authenticated"] = True
        sess["last_activity"] = time.time()

    # 2. Recorder status API
    res_status = client.get("/api/recorder/status")
    print(f"[TEST FLASK] /api/recorder/status status_code: {res_status.status_code}, data: {res_status.get_json()}")
    assert res_status.status_code == 200
    assert res_status.get_json()["success"] is True

    # 3. Recorder segments API
    res_segments = client.get("/api/recorder/segments")
    print(f"[TEST FLASK] /api/recorder/segments status_code: {res_segments.status_code}, data: {res_segments.get_json()}")
    assert res_segments.status_code == 200
    assert res_segments.get_json()["success"] is True

    # 4. Verifikasi CameraWorkerManager untuk real-time tetap ada dan berfungsi
    worker_mgr = CameraWorkerManager.get_instance()
    print(f"[TEST FLASK] CameraWorkerManager instance: {worker_mgr}, running: {worker_mgr.running}")
    assert worker_mgr is not None

    print("[TEST 3 BERHASIL]: Integrasi Flask dan kesiapan jalur realtime diverifikasi sukses!")
    return True

if __name__ == "__main__":
    test_flask_integration()
