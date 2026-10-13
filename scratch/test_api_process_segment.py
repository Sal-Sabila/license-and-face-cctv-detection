import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from services.camera_recorder_service import CameraRecorderService

def test_api_process_segment():
    print("=" * 60)
    print("TEST API: /api/recorder/process_segment")
    print("=" * 60)

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["authenticated"] = True
        sess["last_activity"] = time.time()

    svc = CameraRecorderService.get_instance()
    # Pastikan ada minimal 1 segmen selesai yang terdaftar
    # Kita bisa gunakan segmen yang baru saja dihasilkan
    segments = svc.get_completed_segments()
    print(f"Total completed segments di service: {len(segments)}")

    # 1. Coba request tanpa parameter (memproses segmen 'ready' pertama jika ada)
    res = client.post("/api/recorder/process_segment", json={})
    print(f"POST /api/recorder/process_segment (auto next): {res.status_code}, data: {res.get_json()}")

    # 2. Coba request dengan file in-flight fiktif (harus ditolak 400)
    res_fake = client.post("/api/recorder/process_segment", json={"filename": "non_existent_or_inflight.mp4"})
    print(f"POST /api/recorder/process_segment (in-flight reject): {res_fake.status_code}, data: {res_fake.get_json()}")
    assert res_fake.status_code == 400
    assert res_fake.get_json()["success"] is False

    print("=" * 60)
    print("API TEST SUCCESSFUL")
    print("=" * 60)

if __name__ == "__main__":
    test_api_process_segment()
