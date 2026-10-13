import os
import sys
import time
import json
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager
from app import app
import db

def test_tahap3():
    print("=" * 75)
    print("PENGUJIAN TAHAP 3: ANTREAN OTOMATIS & SINGLE WORKER HYBRID CCTV")
    print("=" * 75)

    test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_tahap3_segments")
    os.makedirs(test_dir, exist_ok=True)

    # -------------------------------------------------------------
    # PENGUJIAN 1: Antrean Otomatis & Pemrosesan Berurutan (Single Worker)
    # -------------------------------------------------------------
    print("\n[TEST 1] ANTREAN OTOMATIS & WORKER TUNGGAL SERIAL")
    recorder = CameraRecorderService(segments_dir=test_dir, segment_duration=10)
    recorder.camera_id = 1
    recorder.camera_name = "GSMasukViewLuar"
    sample_video = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "samples", "deteksi.mp4")
    recorder.stream_url = sample_video
    recorder.auto_process_enabled = True
    recorder.max_queue_size = 10

    # Start recorder & worker
    recorder.start(camera_id=1)
    print("    Recorder & Single Worker dimulai...")

    # Pantau perekaman & pemrosesan selama 55 detik (agar model loading & inferensi CPU tuntas)
    print("    Menunggu rekaman dan pemrosesan otomatis segmen selesai...")
    for i in range(55):
        time.sleep(1.0)
        status = recorder.get_status()
        b = status["backlog"]
        print(
            f"    T+{i+1:02d}s: status={status['status']}, "
            f"backlog=[total={b['total_recorded']}, pend={b['pending']}, proc={b['processing']}, comp={b['completed']}, fail={b['failed']}], "
            f"worker_busy={b['worker_is_busy']}"
        )
        # Berhenti jika minimal 1 atau 2 segmen sudah completed
        if b["completed"] >= 1 and b["total_recorded"] >= 2:
            print("    -> Berhasil mengamati pemrosesan serial oleh worker tunggal!")
            break

    status1 = recorder.get_status()
    backlog1 = status1["backlog"]
    print(f"\n    Ringkasan Backlog Test 1:")
    print(f"      Total recorded : {backlog1['total_recorded']}")
    print(f"      Completed      : {backlog1['completed']}")
    print(f"      Pending        : {backlog1['pending']}")
    print(f"      Failed         : {backlog1['failed']}")
    assert backlog1["completed"] >= 1, "Minimal 1 segmen harus sudah 'completed'"
    assert backlog1["processing"] in (0, 1), "Worker tunggal: processing tidak boleh lebih dari 1!"

    # -------------------------------------------------------------
    # PENGUJIAN 2: Penanganan Kegagalan (Fault Tolerance)
    # -------------------------------------------------------------
    print("\n[TEST 2] PENANGANAN KEGAGALAN TANPA MENGHENTIKAN SISTEM")
    # Buat file segmen korup fiktif
    corrupt_filename = "cam_1_corrupt_test_segment.mp4"
    corrupt_filepath = os.path.join(recorder.camera_dir, corrupt_filename)
    with open(corrupt_filepath, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPT_BYTES")

    corrupt_info = {
        "camera_id": 1,
        "camera_name": "GSMasukViewLuar",
        "filename": corrupt_filename,
        "file_path": corrupt_filepath,
        "start_time": 0.0,
        "end_time": 10.0,
        "duration": 10.0,
        "size_bytes": os.path.getsize(corrupt_filepath),
        "status": "pending",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with recorder._completed_segments_lock:
        recorder._segments_state[corrupt_filename] = corrupt_info
        recorder._completed_segments.append(corrupt_info)
        recorder._save_state()
        recorder._enqueue_segment(corrupt_info)

    print("    Segmen korup dimasukkan ke antrean. Menunggu worker menangani...")
    for _ in range(30):
        time.sleep(1.0)
        seg_state = recorder._segments_state.get(corrupt_filename, {})
        if seg_state.get("status") == "failed":
            print(f"    -> Berhasil: Status ditandai 'failed', error: {seg_state.get('error')[:60]}...")
            break

    assert recorder._segments_state.get(corrupt_filename, {}).get("status") == "failed", "Segmen korup harus ditandai 'failed'"
    assert recorder.is_running is True, "Recorder HARUS tetap berjalan normal saat terjadi kegagalan segmen!"
    print("    -> Recorder dan worker tetap aktif tanpa crash.")

    # -------------------------------------------------------------
    # PENGUJIAN 3: Simulasi Restart Aplikasi & Proteksi Pemrosesan Ganda
    # -------------------------------------------------------------
    print("\n[TEST 3] SIMULASI RESTART APLIKASI & CEK PEMROSESAN GANDA")
    # Hentikan instance saat ini
    recorder.stop()
    time.sleep(1.0)

    # Catat segmen yang sudah 'completed' sebelum restart
    completed_before = [s["filename"] for s in recorder.get_completed_segments(status="completed")]
    print(f"    Segmen completed sebelum restart: {completed_before}")
    assert len(completed_before) >= 1

    # Inisialisasi instance baru pada direktori yang sama (simulasi restart Flask)
    print("    Memulai ulang CameraRecorderService (simulasi restart)...")
    recorder_restart = CameraRecorderService(segments_dir=test_dir, segment_duration=10)
    recorder_restart.camera_id = 1
    recorder_restart.camera_dir = os.path.join(test_dir, "cam_1")
    recorder_restart.state_file = os.path.join(recorder_restart.camera_dir, "segment_queue_state.json")
    recorder_restart._load_state()

    # Periksa apakah riwayat segmen restored
    completed_after = [s["filename"] for s in recorder_restart.get_completed_segments(status="completed")]
    print(f"    Segmen completed setelah load state: {completed_after}")
    assert completed_before == completed_after, "Riwayat segmen completed harus persis sama setelah restart!"

    # Pastikan tidak ada pemrosesan ganda:
    # Segmen yang sudah 'completed' TIDAK dimasukkan ke processing_queue
    queued_filenames = [recorder_restart._processing_queue.queue[i]["filename"] for i in range(recorder_restart._processing_queue.qsize())]
    print(f"    File dalam processing_queue baru: {queued_filenames}")
    for fname in completed_before:
        assert fname not in queued_filenames, f"Segmen '{fname}' yang sudah selesai tidak boleh masuk antrean lagi!"
    print("    -> Proteksi pemrosesan ganda setelah restart TERVERIFIKASI SUKSES.")

    # -------------------------------------------------------------
    # PENGUJIAN 4: Verifikasi API Flask & Jalur Realtime
    # -------------------------------------------------------------
    print("\n[TEST 4] VERIFIKASI ENDPOINT FLASK & JALUR REALTIME")
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["authenticated"] = True
        sess["last_activity"] = time.time()

    # GET /api/recorder/status
    res_status = client.get("/api/recorder/status")
    assert res_status.status_code == 200
    st_json = res_status.get_json()["data"]
    print(f"    API /api/recorder/status: 200 OK")
    print(f"    Backlog API: {st_json.get('backlog')}")

    # GET /api/recorder/segments?status=completed
    res_completed = client.get("/api/recorder/segments?status=completed")
    assert res_completed.status_code == 200
    print(f"    API /api/recorder/segments?status=completed: total={res_completed.get_json()['total']}")

    # Verifikasi CameraWorkerManager realtime
    worker_mgr = CameraWorkerManager.get_instance()
    print(f"    CameraWorkerManager realtime running: {worker_mgr.running}")
    assert worker_mgr is not None

    print("\n" + "=" * 75)
    print("SEMUA PENGUJIAN TAHAP 3 BERHASIL!")
    print("=" * 75)
    return True

if __name__ == "__main__":
    test_tahap3()
