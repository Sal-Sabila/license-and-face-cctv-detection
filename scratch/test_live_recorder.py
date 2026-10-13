import os
import sys
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_recorder_service import CameraRecorderService
import db

def test_live_rtmp_recording():
    print("\n" + "=" * 60)
    print("TEST 2: Pengujian Perekaman Live CCTV RTMP (10 Detik Segmen)")
    print("=" * 60)

    test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_live_segments")
    os.makedirs(test_dir, exist_ok=True)

    recorder = CameraRecorderService(segments_dir=test_dir, segment_duration=10)
    # Gunakan kamera 6 (FM Noto McDonalds yang terbukti online)
    cameras = db.get_all_cameras()
    cam6 = next((c for c in cameras if c["camera_id"] == 6), None)
    assert cam6 is not None, "Kamera 6 tidak ditemukan di DB"

    print(f"[TEST] Kamera Target   : ID {cam6['camera_id']} ({cam6['location']})")
    print(f"[TEST] Stream URL      : {cam6['stream_url']}")

    # Jalankan recorder langsung dengan target camera 6
    started = recorder.start(camera_id=6)
    assert started, "Gagal memulai service recorder untuk kamera live"

    print("[TEST] Menunggu 23 detik untuk memastikan minimal 1 segmen 10 detik selesai direkam dari live CCTV...")
    completed_segment = None
    for i in range(25):
        time.sleep(1.0)
        st = recorder.get_status()
        segs = recorder.get_completed_segments()
        print(f"  Detik {i+1:02d}: status={st['status']}, process_active={st['process_active']}, total_completed={len(segs)}")
        if len(segs) >= 1:
            completed_segment = segs[0]
            print(f"[TEST] Segmen live berhasil direkam: {completed_segment['filename']}")
            break

    recorder.stop()

    assert completed_segment is not None, "Gagal mendapatkan segmen selesai dari live RTMP dalam 25 detik"
    seg_file = completed_segment["file_path"]
    assert os.path.isfile(seg_file), f"File {seg_file} tidak ditemukan di disk!"
    assert completed_segment["size_bytes"] > 100000, f"Ukuran file terlalu kecil: {completed_segment['size_bytes']} bytes"

    # Uji validasi OpenCV
    cap = cv2.VideoCapture(seg_file)
    assert cap.isOpened(), f"OpenCV gagal membuka live segment {seg_file}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    cap.release()

    print("[TEST HASIL LIVE CCTV]")
    print(f"  File path: {seg_file}")
    print(f"  Ukuran file: {completed_segment['size_bytes']} bytes")
    print(f"  Durasi segmen: {completed_segment['duration']} detik")
    print(f"  OpenCV read: frame_count={frame_count}, fps={fps:.2f}, frame_shape={frame.shape if ret else 'None'}")
    assert ret, "Gagal membaca frame pertama dari live segmen"

    print("[TEST 2 BERHASIL]: Perekaman stream live CCTV menghasilkan segmen 10 detik yang sempurna!")
    return True

if __name__ == "__main__":
    test_live_rtmp_recording()
