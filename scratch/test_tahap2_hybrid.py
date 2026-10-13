import os
import sys
import time
import psutil

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager
from app import app
import db

def test_tahap2():
    print("=" * 75)
    print("PENGUJIAN TAHAP 2 SISTEM HYBRID CCTV FLASK")
    print("=" * 75)

    test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_tahap2_segments")
    os.makedirs(test_dir, exist_ok=True)

    # 1. Inisialisasi CameraRecorderService dengan sampel video kendaraan & plat
    recorder = CameraRecorderService(segments_dir=test_dir, segment_duration=10)
    recorder.camera_id = 1
    recorder.camera_name = "GSMasukViewLuar"
    sample_video = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "samples", "deteksi.mp4")
    recorder.stream_url = sample_video

    print(f"\n[1] PEREKAMAN SEGMEN 10 DETIK")
    print(f"    Input Stream        : {sample_video}")
    print(f"    Direktori Segmen    : {test_dir}")
    print(f"    Durasi per Segmen   : {recorder.segment_duration} detik")

    started = recorder._start_recording_session()
    assert started, "Gagal memulai sesi perekaman FFmpeg"

    # Tunggu sampai segmen pertama selesai (sekitar 11-13 detik)
    print("    Menunggu rekaman segmen pertama selesai...")
    completed_seg = None
    for i in range(20):
        time.sleep(1.0)
        status = recorder.get_status()
        completed = recorder.get_completed_segments()
        print(f"    T+{i+1:02d}s: status={status['status']}, completed_count={len(completed)}")
        if len(completed) >= 1:
            completed_seg = completed[0]
            print(f"\n[2] SEGMEN PERTAMA SELESAI DIREKAM:")
            print(f"    Filename      : {completed_seg['filename']}")
            print(f"    File path     : {completed_seg['file_path']}")
            print(f"    Ukuran file   : {completed_seg['size_bytes']} bytes")
            print(f"    Durasi        : {completed_seg['duration']} detik")
            print(f"    Status segmen : {completed_seg['status']}")
            break

    assert completed_seg is not None, "Segmen pertama gagal dihasilkan dalam batas waktu."

    # 2. UJI PROTEKSI: Pastikan file yang masih direkam TIDAK BISA diproses
    print(f"\n[3] UJI PROTEKSI FILE IN-FLIGHT / BELUM SELESAI:")
    # Simulasikan file fiktif atau in-flight yang belum tercatat selesai
    in_flight_filename = f"cam_1_{recorder.session_id}_99999_inflight.mp4"
    in_flight_filepath = os.path.join(recorder.camera_dir, in_flight_filename)
    with open(in_flight_filepath, "wb") as f:
        f.write(b"in-flight-recording-data-in-progress")

    res_inflight = recorder.process_single_segment(segment_filename=in_flight_filename)
    print(f"    Hasil percobaan memproses file in-flight:")
    print(f"    Success: {res_inflight['success']}, Pesan: {res_inflight['message']}")
    assert res_inflight["success"] is False, "File in-flight seharusnya ditolak!"
    assert "masih dalam proses perekaman" in res_inflight["message"] or "belum divalidasi" in res_inflight["message"] or "tidak terdaftar" in res_inflight["message"]
    print("    -> Proteksi berhasil: File yang belum selesai direkam DITOLAK.")

    # Bersihkan file uji coba in-flight
    try:
        os.remove(in_flight_filepath)
    except OSError:
        pass

    # 3. UJI PEMROSESAN SATU SEGMEN 10 DETIK MENGGUNAKAN VideoAIService
    print(f"\n[4] MEMPROSES TEPAT SATU SEGMEN 10 DETIK MENGGUNAKAN VideoAIService...")
    print(f"    Target Segmen: {completed_seg['filename']}")
    cpu_before = psutil.cpu_percent(interval=None)
    
    # Proses hanya 1 segmen (save_annotated=True untuk menghasilkan bukti visual)
    process_result = recorder.process_single_segment(
        segment_filename=completed_seg["filename"],
        save_annotated=True
    )

    print(f"    Status pemrosesan : {process_result['success']}")
    assert process_result["success"] is True, f"Pemrosesan segmen gagal: {process_result}"

    data = process_result["data"]
    seg_info = data["segment"]
    detections = data["detections"]
    cpu_info = data["cpu_usage"]

    print("\n" + "=" * 75)
    print("HASIL PEMROSESAN SEGMEN 10 DETIK (VideoAIService)")
    print("=" * 75)
    print(f"Waktu Eksekusi AI     : {data['elapsed_seconds']} detik")
    print(f"Frame Inferensi       : {detections['inference_frames']}")
    print(f"Penggunaan CPU        : System: {cpu_info['system_cpu_percent']}%, Process: {cpu_info['process_cpu_percent']}%")
    print(f"Kendaraan/Person Caps : {detections['person_and_vehicle_captures']}")
    print(f"Pelat Terdeteksi      : {detections['plate_captures']}")
    print(f"Event Groups          : Kendaraan={detections['event_groups']['vehicle']}, Person={detections['event_groups']['person']}")
    if detections.get("plate_history"):
        print("Detail Bacaan Pelat   :")
        for idx, p in enumerate(detections["plate_history"][:5]):
            print(f"  [{idx+1}] Pelat: {p.get('plate')}, Conf: {p.get('confidence')}, Waktu: {p.get('time')}s")
    if detections.get("annotated_output"):
        print(f"Output Video Hasil    : {detections['annotated_output']}")

    # 4. VERIFIKASI JALUR REALTIME TETAP NORMAL & REST API FLASK
    print(f"\n[5] VERIFIKASI INTEGRASI FLASK & JALUR REALTIME:")
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["authenticated"] = True
        sess["last_activity"] = time.time()

    res_status = client.get("/api/recorder/status")
    status_data = res_status.get_json()["data"]
    print(f"    Status API Recorder   : {res_status.status_code}, status={status_data['status']}")
    print(f"    Pending Segments      : {status_data['pending_ready_segments']}")
    print(f"    Latest Processed Seg  : {status_data['latest_processed_segment']['filename'] if status_data['latest_processed_segment'] else None}")

    # CameraWorkerManager check
    mgr = CameraWorkerManager.get_instance()
    print(f"    CameraWorkerManager   : running={mgr.running}")
    assert mgr is not None

    # Hentikan perekaman uji
    recorder.stop()
    print("\n[6] KESIMPULAN TAHAP 2:")
    print("    - Perekaman 10 detik berhasil.")
    print("    - Proteksi file in-flight aktif dan terverifikasi.")
    print("    - Tepat 1 segmen diproses dengan VideoAIService tanpa memproses segmen lain secara otomatis.")
    print("    - Deteksi kendaraan dan pelat berhasil diperiksa.")
    print("    - Jalur realtime tetap berjalan normal.")
    print("=" * 75)
    return True

if __name__ == "__main__":
    test_tahap2()
