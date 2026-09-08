"""
test_stream.py
==============
Script pengujian koneksi stream CCTV menggunakan FFmpegStreamReader.
Jalankan ini untuk memastikan link stream CCTV aktif dan terbaca lancar
sebelum menjalankan pipeline deteksi orang / deteksi wajah.

Cara pakai:
    python test_stream.py
Tekan 'q' di jendela video untuk keluar.
"""

import time
import cv2
from ffmpeg_reader import FFmpegStreamReader

STREAM_URL = "rtmp://103.255.15.138:1935/live/GSKeluarViewLuar.stream"


def main():
    print("=" * 60)
    print("Test Koneksi Stream CCTV")
    print("=" * 60)
    print(f"Membuka stream: {STREAM_URL}")

    cap = FFmpegStreamReader(STREAM_URL, width=1920, height=1080)

    if not cap.isOpened():
        print("[GAGAL] Tidak dapat membuka stream. Cek link stream atau koneksi internet.")
        return

    print("[BERHASIL] Stream terhubung. Menampilkan preview...")

    window_name = "Test Stream - Tekan 'q' untuk keluar"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    frame_count = 0
    fail_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            if fail_count % 10 == 0:
                print(f"[WARNING] Gagal baca frame ({fail_count}x)")
            if fail_count > 30:
                print("[INFO] Mencoba reconnect stream...")
                cap.reconnect()
                fail_count = 0
            time.sleep(0.05)
            continue

        fail_count = 0
        frame_count += 1

        elapsed = time.time() - start
        if elapsed > 0 and frame_count % 30 == 0:
            fps = frame_count / elapsed
            print(f"Frame #{frame_count} | FPS rata-rata: {fps:.2f} | Resolusi: {frame.shape[1]}x{frame.shape[0]}")

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai. Total frame terbaca: {frame_count}")


if __name__ == "__main__":
    main()