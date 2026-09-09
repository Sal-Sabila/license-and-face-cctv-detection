import time
import threading
from ffmpeg_stream_reader import FFmpegStreamReader, normalize_stream_url
from services.stream_ai_service import StreamAIService
import db


class CameraWorker(threading.Thread):
    """Worker background untuk 1 kamera CCTV aktif.
    
    Tugas:
    - Membaca stream CCTV via FFmpegStreamReader
    - Menjalankan deteksi AI (YOLO + OCR + ByteTrack)
    - Menyimpan crop dan mencatat event ke database
    - Menyediakan frame terbaru untuk live stream dashboard
    """

    def __init__(self, camera_id: int, name: str, stream_url: str):
        super().__init__(name=f"Worker-CAM-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.name = name
        self.stream_url = stream_url
        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.consecutive_errors = 0
        self.reader = None

    def get_latest_frame(self):
        """Mengambil frame terbaru yang sudah ditangkap oleh worker."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """Menghentikan worker kamera."""
        self.running = False
        if self.reader is not None:
            try:
                self.reader.release()
            except Exception:
                pass

    def run(self):
        print(f"[BG CCTV] Memulai background detection untuk CAM {self.camera_id} ({self.name})...")
        norm_url = normalize_stream_url(self.stream_url)
        self.reader = None
        ai_service = StreamAIService.get_instance()

        while self.running:
            try:
                if self.reader is None:
                    self.reader = FFmpegStreamReader(norm_url, width=960, height=540)
                    time.sleep(0.5)

                ret, frame = self.reader.read()
                if not ret or frame is None:
                    self.consecutive_errors += 1
                    if self.consecutive_errors > 30:
                        print(f"[BG CCTV WARNING] CAM {self.camera_id} ({self.name}) stream terputus. Mencoba reconnect...")
                        try:
                            if self.reader is not None:
                                self.reader.release()
                        except Exception:
                            pass
                        self.reader = None
                        self.consecutive_errors = 0
                        # Tunggu 5 detik sebelum coba reconnect lagi
                        for _ in range(50):
                            if not self.running:
                                break
                            time.sleep(0.1)
                    else:
                        time.sleep(0.04)
                    continue

                self.consecutive_errors = 0

                # Update frame cache untuk pemantauan live di dashboard
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_time = time.time()

                # Jalankan proses AI (YOLO + OCR + ByteTrack + DB insert) di background
                try:
                    ai_service.process_frame(frame, draw_bbox=False, camera_id=self.camera_id)
                except Exception as ai_err:
                    pass

                # Delay kecil untuk kestabilan CPU (~25-30 FPS)
                time.sleep(0.03)

            except Exception as e:
                print(f"[BG CCTV ERROR] CAM {self.camera_id} ({self.name}) loop error: {e}")
                time.sleep(1.0)

        # Cleanup saat dihentikan
        if self.reader is not None:
            try:
                self.reader.release()
            except Exception:
                pass
        print(f"[BG CCTV] Background worker CAM {self.camera_id} ({self.name}) berhenti.")


class BackgroundDetectionManager:
    """Manajer orkestrasi deteksi CCTV background untuk seluruh kamera aktif."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.workers = {}  # camera_id -> CameraWorker
        self.manager_lock = threading.Lock()
        self.running = False
        self.sync_thread = None

    def start(self):
        """Memulai pengawasan dan deteksi background untuk semua CCTV aktif."""
        with self.manager_lock:
            if self.running:
                return
            self.running = True

            # Jalankan sync kamera pertama kali
            self._sync_cameras_internal()

            # Jalankan background poller untuk mendeteksi perubahan status di database setiap 8 detik
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True, name="CCTV-Sync-Watcher")
            self.sync_thread.start()
            print("[BG CCTV MANAGER] Service deteksi background aktif.")

    def stop(self):
        """Menghentikan seluruh background worker."""
        with self.manager_lock:
            self.running = False
            for cid, worker in list(self.workers.items()):
                try:
                    worker.stop()
                except Exception:
                    pass
            self.workers.clear()
            print("[BG CCTV MANAGER] Seluruh background worker dihentikan.")

    def sync_active_cameras(self):
        """Sinkronisasi kamera aktif (dipanggil saat kamera ditambah/diubah/dihapus via web)."""
        with self.manager_lock:
            self._sync_cameras_internal()

    def _sync_cameras_internal(self):
        """Logika internal untuk menyesuaikan worker dengan kamera aktif di database."""
        try:
            cameras = db.get_all_cameras()
            # Kamera yang berstatus aktif (status == 1) dan memiliki URL
            active_cams = {c["camera_id"]: c for c in cameras if c.get("status") == 1 and c.get("stream_url")}

            # 1. Hentikan worker untuk kamera yang dinonaktifkan atau dihapus atau URL-nya berubah
            for cid, worker in list(self.workers.items()):
                if cid not in active_cams or worker.stream_url != active_cams[cid]["stream_url"]:
                    print(f"[BG CCTV MANAGER] Menghentikan worker CAM {cid}...")
                    worker.stop()
                    del self.workers[cid]

            # 2. Jalankan worker baru untuk kamera aktif yang belum berjalan
            for cid, cam in active_cams.items():
                if cid not in self.workers or not self.workers[cid].is_alive():
                    worker = CameraWorker(
                        camera_id=cid,
                        name=cam.get("location") or f"Camera-{cid}",
                        stream_url=cam["stream_url"]
                    )
                    self.workers[cid] = worker
                    worker.start()

        except Exception as e:
            print(f"[BG CCTV MANAGER ERROR] Gagal sinkronisasi kamera: {e}")

    def _sync_loop(self):
        """Loop periodik untuk memeriksa database secara otomatis."""
        while self.running:
            for _ in range(80):
                if not self.running:
                    break
                time.sleep(0.1)
            if self.running:
                try:
                    with self.manager_lock:
                        self._sync_cameras_internal()
                except Exception:
                    pass

    def get_frame(self, camera_id: int):
        """Mengambil frame terbaru dari worker kamera aktif (jika ada)."""
        with self.manager_lock:
            worker = self.workers.get(int(camera_id))
            if worker:
                return worker.get_latest_frame()
        return None

    def is_camera_running(self, camera_id: int) -> bool:
        """Mengecek apakah worker kamera sedang aktif berjalan."""
        with self.manager_lock:
            worker = self.workers.get(int(camera_id))
            return worker is not None and worker.is_alive()
