import os
import sys
import time
import cv2

# Pastikan path modul terbaca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_recorder_service import CameraRecorderService
from ffmpeg_stream_reader import find_ffmpeg_executable

def test_recorder_with_sample_video():
    print("=" * 60)
    print("TEST 1: Pengujian Perekaman & Segmentasi 10 Detik")
    print("=" * 60)

    test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_segments")
    os.makedirs(test_dir, exist_ok=True)

    recorder = CameraRecorderService(segments_dir=test_dir, segment_duration=10)
    recorder.camera_id = 99
    recorder.camera_name = "Kamera Uji Sampel"
    sample_video = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "samples", "deteksi.mp4")
    recorder.stream_url = sample_video

    print(f"[TEST] FFmpeg executable : {recorder.ffmpeg_path}")
    print(f"[TEST] Input video        : {sample_video}")
    print(f"[TEST] Target direktori   : {test_dir}")
    print(f"[TEST] Segment duration   : {recorder.segment_duration} detik")

    # Siapkan callback listener
    received_callbacks = []
    def on_segment(seg):
        received_callbacks.append(seg)
        print(f"[CALLBACK RECEIVED] Segmen siap: {seg['filename']} ({seg['duration']}s, {seg['size_bytes']} bytes)")

    recorder.register_completed_callback(on_segment)

    # Jalankan sesi perekaman
    started = recorder._start_recording_session()
    assert started, "Gagal memulai sesi perekaman FFmpeg"
    print("[TEST] Perekam berhasil dimulai, menunggu 24 detik untuk merekam 2 segmen 10 detik...")

    # Pantau selama 25 detik
    for i in range(25):
        time.sleep(1.0)
        status = recorder.get_status()
        completed = recorder.get_completed_segments()
        print(f"  Detik {i+1:02d}: status={status['status']}, completed_segments={len(completed)}")
        if len(completed) >= 2:
            print("[TEST] Berhasil mendapatkan minimal 2 segmen lengkap!")
            break

    # Cek antrean thread-safe (pop)
    popped1 = recorder.pop_next_completed_segment(timeout=1.0)
    print(f"[TEST] Pop queue segment 1: {popped1['filename'] if popped1 else None}")
    assert popped1 is not None, "Antrean segmen selesai kosong!"

    # Validasi bahwa file segmen benar-benar valid dan dapat dibuka OpenCV
    seg_file = popped1["file_path"]
    assert os.path.isfile(seg_file), f"File {seg_file} tidak ditemukan di disk!"
    assert popped1["size_bytes"] > 0, "Ukuran file segmen kosong!"

    cap = cv2.VideoCapture(seg_file)
    assert cap.isOpened(), f"OpenCV gagal membuka segmen {seg_file}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()

    print(f"[TEST VALIDASI SEGMEN]")
    print(f"  File: {seg_file}")
    print(f"  Total frames: {frame_count}, FPS: {fps:.2f}, Shape: {frame.shape if ret else 'None'}")
    assert ret, "Gagal membaca frame pertama dari segmen"
    assert frame_count > 0, "Segmen tidak memiliki frame"

    # Stop recorder
    recorder._terminate_ffmpeg()
    recorder.is_running = False
    print("[TEST 1 BERHASIL]: Perekaman dan segmentasi 10 detik valid dan terbaca sempurna!")
    return True

if __name__ == "__main__":
    test_recorder_with_sample_video()
