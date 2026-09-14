"""
person_detection.py
===================
Deteksi & pelacakan orang (pedestrian) secara real-time pada 4 kamera CCTV
sekaligus menggunakan tata letak Grid 2x2, YOLOv8n, dan ByteTrack.

Fitur:
1. Menampilkan 4 feed CCTV live secara berjejer 2x2.
2. Setiap kamera memiliki pelacak (ByteTrack) dan state capture mandiri.
3. Penjadwalan inferensi cerdas (round-robin) agar CPU tetap ringan & preview smooth.
4. Auto-capture orang baru ke folder captures/ dengan nama kamera & track ID.
5. Indikator status live, jumlah orang di frame, dan total capture per kamera.

Penggunaan:
    python person_detection.py

Tekan 'q' atau tutup jendela untuk berhenti.
"""

import os
import sys
import time
import threading
import warnings
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Filter deprecation warning dari ByteTrack di terminal
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
torch.set_num_threads(2)
cv2.setNumThreads(2)

from ffmpeg_reader import FFmpegStreamReader


# =========================================
# PENGATURAN CCTV (4 KAMERA)
# =========================================
CAMERA_STREAMS = [
    {
        "id": 1,
        "name": "GSKeluarViewDalam",
        "url": "rtmp://103.255.15.138:1935/live/GSKeluarViewDalam.stream",
    },
    {
        "id": 2,
        "name": "GSKeluarViewLuar",
        "url": "rtmp://103.255.15.138:1935/live/GSKeluarViewLuar.stream",
    },
    {
        "id": 3,
        "name": "GSMasukViewDalam",
        "url": "rtmp://103.255.15.138:1935/live/GSMasukViewDalam.stream",
    },
    {
        "id": 4,
        "name": "GSMasukViewLuar",
        "url": "rtmp://103.255.15.138:1935/live/GSMasukViewLuar.stream",
    },
]

# ── Resolusi Grid 2x2 ──
# Tiap kamera berukuran 640x360 (16:9) -> Total grid 1280x720 (+ 38px master info bar)
TILE_WIDTH = 640
TILE_HEIGHT = 360
GRID_WIDTH = TILE_WIDTH * 2     # 1280 px
GRID_HEIGHT = TILE_HEIGHT * 2   # 720 px
HEADER_HEIGHT = 38              # baris info global di atas

# ── Path Model & Output ──
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

# ── Deteksi YOLO ──
CONFIDENCE_THRESHOLD = 0.30   # confidence minimum (dioptimalkan untuk pejalan kaki CCTV)
PERSON_CLASS_ID = 0            # class ID "person" di COCO
MIN_BBOX_AREA = 1200           # batas minimal luas bbox (dalam skala tile 640x360)
INFERENCE_IMGSZ = 480          # ukuran input gambar ke YOLO

# ── Tracking (ByteTrack) ──
LOST_TRACK_BUFFER = 120        # batas frame sebelum track dianggap hilang
FRAME_RATE = 25

# ── Capture ──
CAPTURE_ONCE_ONLY = True       # True: orang hanya di-capture 1 kali seumur hidup
CAPTURE_INTERVAL = 5.0         # interval jika CAPTURE_ONCE_ONLY = False
MIN_CROP_WIDTH = 25
MIN_CROP_HEIGHT = 45

# ── Reconnect ──
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 3


# =========================================
# FRAME GRABBER MULTI-STREAM
# =========================================
class FrameGrabber:
    """
    Membaca frame dari stream kamera di thread terpisah.
    Hanya menyimpan frame terbaru untuk mencegah lag / backlog.
    """

    def __init__(self, source, width=640, height=360, name="Camera"):
        self.source = source
        self.width = width
        self.height = height
        self.name = name

        self.cap = self._open_stream()
        self.frame = None
        self.running = self.cap is not None and self.cap.isOpened()
        self.lock = threading.Lock()
        self.fail_count = 0
        self.is_connected = self.running
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def _open_stream(self):
        """Membuka stream via FFmpegStreamReader atau cv2.VideoCapture."""
        if isinstance(self.source, str) and (
            self.source.startswith("rtmp://")
            or self.source.startswith("http://")
            or self.source.startswith("https://")
            or self.source.endswith(".stream")
            or self.source.endswith(".m3u8")
        ):
            return FFmpegStreamReader(self.source, width=self.width, height=self.height)

        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self._thread.start()
        return self

    def _reconnect(self):
        for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
            if self.cap is not None:
                self.cap.release()
            time.sleep(RECONNECT_DELAY)
            self.cap = self._open_stream()
            if self.cap is not None and self.cap.isOpened():
                self.fail_count = 0
                self.is_connected = True
                return True
        self.is_connected = False
        return False

    def _reader(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._reconnect():
                    self.running = False
                    break
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.fail_count += 1
                if self.fail_count > 90:
                    self.is_connected = False
                    if not self._reconnect():
                        self.running = False
                        break
                time.sleep(0.02)
                continue

            self.fail_count = 0
            self.is_connected = True
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()


# =========================================
# FUNGSI BANTU GAMBAR & SIMPAN CAPTURE
# =========================================
def save_crop(frame: np.ndarray, bbox, track_id: int, timestamp: datetime, camera_name: str):
    """
    Crop area badan orang dari frame dan simpan sebagai JPEG.
    Format: person_<camera_name>_<YYYYMMDD_HHMMSS_mmm>_track<ID>.jpg
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    clean_cam = camera_name.replace(" ", "_")
    filename = f"person_{clean_cam}_{ts_str}_track{track_id}.jpg"
    filepath = os.path.join(CAPTURE_DIR, filename)

    cv2.imwrite(filepath, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return filepath


def draw_camera_tile(tile: np.ndarray, tracked: sv.Detections,
                     capture_state: dict, cam_info: dict, total_captured: int):
    """
    Menggambar bounding box, label ID, dan header badge pada tiap tile CCTV.
    """
    # ── 1. Bounding Boxes ──
    for i in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
        track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
        conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

        state = capture_state.get(track_id)
        if state:
            if (time.time() - state["last_capture_time"]) < 1.5:
                color = (0, 255, 255)   # Kuning terang (baru di-capture)
                status_text = "CAPTURED!"
            else:
                color = (255, 200, 0)   # Biru muda (tersimpan)
                status_text = "saved"
        else:
            color = (0, 255, 0)         # Hijau (sedang dilacak)
            status_text = "tracking"

        # Bounding box
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"ID:{track_id} {conf:.0%} [{status_text}]"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lw, lh = label_size
        cv2.rectangle(tile, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, max(0, y1)), color, -1)
        cv2.putText(tile, label, (x1 + 2, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # ── 2. Top Banner Tile (Translucent Dark Header) ──
    banner_h = 28
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (tile.shape[1], banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, tile, 0.25, 0, tile)

    # Titik Status Hijau (LIVE)
    cv2.circle(tile, (12, 14), 5, (0, 255, 0), -1)

    # Nama Kamera
    cam_title = f"CAM {cam_info['id']}: {cam_info['name']}"
    cv2.putText(tile, cam_title, (24, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Statistik Kamera (Kanan)
    person_cnt = len(tracked)
    stats = f"Orang: {person_cnt}  |  Capture: {total_captured}"
    cv2.putText(tile, stats, (tile.shape[1] - 175, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 235), 1, cv2.LINE_AA)

    # Border pembatas tipis di sekeliling tile
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (50, 50, 50), 1)

    return tile


def create_placeholder_tile(cam_info: dict, message="Menghubungkan ke stream..."):
    """Membuat frame placeholder elegan jika feed CCTV belum tersedia / reconnect."""
    tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    tile[:] = (25, 25, 28)

    # Header bar
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH, 28), (15, 15, 18), -1)
    cv2.circle(tile, (12, 14), 5, (0, 165, 255), -1)  # Oranye (menunggu)
    cv2.putText(tile, f"CAM {cam_info['id']}: {cam_info['name']}", (24, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Pesan tengah
    text = f"[{cam_info['name']}]"
    cv2.putText(tile, text, (TILE_WIDTH // 2 - 120, TILE_HEIGHT // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(tile, message, (TILE_WIDTH // 2 - 130, TILE_HEIGHT // 2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1, cv2.LINE_AA)

    cv2.rectangle(tile, (0, 0), (TILE_WIDTH - 1, TILE_HEIGHT - 1), (50, 50, 50), 1)
    return tile


def draw_master_header(width: int, total_people: int, total_captures: int, fps: float):
    """Membuat baris header global di atas grid 2x2."""
    header = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
    header[:] = (18, 18, 22)

    # Judul Kiri
    cv2.putText(header, "CCTV AI MONITORING - DETEKSI ORANG (2x2 GRID)", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Indikator Tengah & Kanan
    stats = f"Total di Frame: {total_people}  |  Total Captured: {total_captures}  |  FPS: {fps:.1f}  |  [q] Keluar"
    cv2.putText(header, stats, (width - 560, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)

    # Garis pemisah bawah
    cv2.line(header, (0, HEADER_HEIGHT - 1), (width, HEADER_HEIGHT - 1), (60, 60, 70), 1)
    return header


# =========================================
# MAIN MULTI-STREAM PIPELINE
# =========================================
def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    print("=" * 65)
    print("Multi-CCTV Person Detection (4 Kamera Live - Grid 2x2)")
    print("=" * 65)

    # ── Load Model YOLOv8 ──
    print(f"[INFO] Memuat model YOLO dari: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model YOLO siap.")

    # ── Inisialisasi Tiap Kamera ──
    cameras = []
    print("[INFO] Membuka 4 stream CCTV di thread terpisah...")

    for info in CAMERA_STREAMS:
        print(f"  -> CAM {info['id']}: {info['name']} ({info['url']})")
        grabber = FrameGrabber(info["url"], width=TILE_WIDTH, height=TILE_HEIGHT, name=info["name"]).start()

        tracker = sv.ByteTrack(
            lost_track_buffer=LOST_TRACK_BUFFER,
            frame_rate=FRAME_RATE,
        )

        cameras.append({
            "info": info,
            "grabber": grabber,
            "tracker": tracker,
            "capture_state": {},
            "cached_tracked": sv.Detections.empty(),
            "total_captured": 0,
        })

    # ── Jendela OpenCV ──
    window_name = "Multi-CCTV Person Detection (2x2 Grid) - Tekan 'q' untuk keluar"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, GRID_WIDTH, GRID_HEIGHT + HEADER_HEIGHT)

    print("[INFO] Menunggu frame dari semua kamera...")
    time.sleep(1.0)

    total_loop_frames = 0
    start_time = time.time()
    active_cam_idx = 0  # Round-robin inferensi: 1 kamera per cycle loop

    try:
        while True:
            total_loop_frames += 1
            now = datetime.now()

            # Baca frame dari ke-4 kamera
            current_frames = []
            for cam in cameras:
                f = cam["grabber"].read()
                current_frames.append(f)

            # ── Deteksi YOLO (Round-Robin 1 Kamera per Ticks) ──
            # Menjamin display tetap smooth (~25 FPS) dan CPU hemat daya
            target_cam = cameras[active_cam_idx]
            target_frame = current_frames[active_cam_idx]

            if target_frame is not None:
                # Inference YOLO
                results = model(
                    target_frame,
                    imgsz=INFERENCE_IMGSZ,
                    classes=[PERSON_CLASS_ID],
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )[0]

                # Filter detections
                boxes = results.boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)

                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    valid = areas >= MIN_BBOX_AREA
                    xyxy = xyxy[valid]
                    confs = confs[valid]
                    class_ids = class_ids[valid]

                    detections = sv.Detections(
                        xyxy=xyxy.astype(np.float32),
                        confidence=confs.astype(np.float32),
                        class_id=class_ids,
                    )
                else:
                    detections = sv.Detections.empty()

                # Update tracker khusus kamera ini
                if len(detections) > 0:
                    tracked = target_cam["tracker"].update_with_detections(detections)
                else:
                    tracked = sv.Detections.empty()

                target_cam["cached_tracked"] = tracked

                # Cek dan simpan capture orang baru
                for i in range(len(tracked)):
                    track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    if track_id < 0:
                        continue

                    bbox = tracked.xyxy[i]
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    state = target_cam["capture_state"].get(track_id)
                    now_ts = time.time()

                    should_capture = (state is None) if CAPTURE_ONCE_ONLY else (
                        state is None or (now_ts - state["last_capture_time"]) >= CAPTURE_INTERVAL
                    )

                    if should_capture:
                        saved_path = save_crop(target_frame, bbox, track_id, now, target_cam["info"]["name"])
                        if saved_path:
                            target_cam["total_captured"] += 1
                            count = (state["capture_count"] + 1) if state else 1
                            target_cam["capture_state"][track_id] = {
                                "last_capture_time": now_ts,
                                "capture_count": count,
                            }
                            print(
                                f"[CAPTURE] [{target_cam['info']['name']}] ID={track_id} "
                                f"conf={conf:.0%} -> {os.path.basename(saved_path)}"
                            )

            # Geser giliran kamera berikutnya
            active_cam_idx = (active_cam_idx + 1) % len(cameras)

            # ── Render Ke-4 Tile CCTV ──
            rendered_tiles = []
            for i, cam in enumerate(cameras):
                f = current_frames[i]
                if f is not None:
                    tile_copy = f.copy()
                    tile = draw_camera_tile(
                        tile_copy,
                        cam["cached_tracked"],
                        cam["capture_state"],
                        cam["info"],
                        cam["total_captured"]
                    )
                else:
                    tile = create_placeholder_tile(cam["info"])
                rendered_tiles.append(tile)

            # Susun Grid 2x2:
            # [ Cam 1 ] [ Cam 2 ]
            # [ Cam 3 ] [ Cam 4 ]
            row_top = np.hstack([rendered_tiles[0], rendered_tiles[1]])
            row_bot = np.hstack([rendered_tiles[2], rendered_tiles[3]])
            grid_2x2 = np.vstack([row_top, row_bot])

            # Hitung statistik global
            elapsed = time.time() - start_time
            display_fps = total_loop_frames / elapsed if elapsed > 0 else 0
            total_people_in_view = sum(len(c["cached_tracked"]) for c in cameras)
            total_captures_all = sum(c["total_captured"] for c in cameras)

            # Master Header
            master_header = draw_master_header(GRID_WIDTH, total_people_in_view, total_captures_all, display_fps)

            # Gabungkan Header + Grid
            final_display = np.vstack([master_header, grid_2x2])

            # Tampilkan Preview
            cv2.imshow(window_name, final_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Dihentikan pengguna (tekan 'q').")
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[INFO] Jendela ditutup oleh pengguna.")
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan (Ctrl+C).")

    finally:
        print("\n[INFO] Menghentikan semua stream grabber...")
        for cam in cameras:
            cam["grabber"].stop()
        cv2.destroyAllWindows()

        print("=" * 65)
        print("RINGKASAN HASIL CAPTURE PER KAMERA:")
        total_all = 0
        for cam in cameras:
            cnt = cam["total_captured"]
            total_all += cnt
            print(f"  - {cam['info']['name']}: {cnt} orang")
        print(f"TOTAL SELURUH KAMERA: {total_all} orang")
        print(f"File foto tersimpan di: {CAPTURE_DIR}")
        print("=" * 65)


if __name__ == "__main__":
    main()
