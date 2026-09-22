import time
import threading

from ffmpeg_stream_reader import FFmpegStreamReader, normalize_stream_url
import db


# ============================================================
# KONFIGURASI
# ============================================================

# Seberapa sering supervisor cek status kamera di database
# (start worker baru kalau ada kamera di-"Aktifkan", stop worker
# kalau ada kamera di-"Nonaktifkan").
SUPERVISOR_POLL_INTERVAL = 5.0

# Resolusi frame yang dibaca untuk background detection. Tidak perlu
# resolusi penuh kamera -- ini cuma dipakai untuk deteksi AI, bukan
# untuk ditonton, jadi disamakan dengan resolusi yang dipakai
# generate_mjpeg_stream() di routes/plate.py.
WORKER_WIDTH = 960
WORKER_HEIGHT = 540

# Batas gagal baca frame berturut-turut sebelum worker mencoba
# membuka ulang koneksi RTMP dari awal (stream mungkin sempat putus).
MAX_CONSECUTIVE_FAILURES = 60


class _CameraWorker:
    """
    Satu instance = satu kamera yang sedang dipantau di background.
    Tugasnya cuma dua: baca frame dari RTMP terus-menerus, dan
    "suapin" tiap frame itu ke StreamAIService supaya AI (yang
    sudah berjalan di background thread-nya sendiri) selalu dapat
    pasokan frame -- lepas dari ada yang buka dashboard atau tidak.
    """

    def __init__(self, camera_id, stream_url, ai_service):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.ai_service = ai_service

        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"CameraWorker-{camera_id}",
            daemon=True,
        )

        self.last_frame = None
        self.last_frame_lock = threading.Lock()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def get_last_frame(self):
        with self.last_frame_lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def _run(self):
        print(f"[CAMERA WORKER] Kamera {self.camera_id}: mulai membaca stream (background)")

        norm_url = normalize_stream_url(self.stream_url)
        reader = FFmpegStreamReader(norm_url, width=WORKER_WIDTH, height=WORKER_HEIGHT)

        consecutive_failures = 0

        try:
            while not self.stop_event.is_set():

                ret, frame = reader.read()

                if not ret or frame is None:
                    consecutive_failures += 1

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(
                            f"[CAMERA WORKER] Kamera {self.camera_id}: "
                            f"gagal baca {consecutive_failures}x berturut-turut, "
                            f"mencoba buka ulang koneksi..."
                        )
                        reader.release()
                        time.sleep(1.0)
                        reader = FFmpegStreamReader(norm_url, width=WORKER_WIDTH, height=WORKER_HEIGHT)
                        consecutive_failures = 0

                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                with self.last_frame_lock:
                    self.last_frame = frame

                # draw_bbox=False -- worker ini tidak untuk ditonton,
                # jadi tidak perlu buang waktu menggambar bounding box.
                # process_frame() sendiri murah (cuma push ke queue),
                # AI berat-nya jalan di thread _ai_loop milik
                # StreamAIService, terpisah dari loop ini.
                self.ai_service.process_frame(
                    frame,
                    draw_bbox=False,
                    camera_id=self.camera_id,
                )

                # Batasi kecepatan baca supaya tidak membebani CPU
                # secara percuma -- AI sendiri sudah adaptif soal
                # seberapa sering dia benar-benar memproses frame.
                time.sleep(0.03)

        except Exception as exc:
            print(f"[CAMERA WORKER ERROR] Kamera {self.camera_id}: {exc}")

        finally:
            reader.release()
            print(f"[CAMERA WORKER] Kamera {self.camera_id}: berhenti")


class CameraWorkerManager:
    """
    Supervisor singleton yang menjaga: 1 worker background berjalan
    untuk setiap kamera yang status-nya Aktif di database, dan tidak
    ada worker untuk kamera yang Nonaktif.

    Ini yang membuat toggle "Aktifkan/Nonaktifkan" di halaman
    Monitoring CCTV benar-benar menyalakan/mematikan deteksi,
    tanpa perlu ada dashboard yang dibuka.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.workers = {}          # camera_id -> _CameraWorker
        self.workers_lock = threading.Lock()
        self.supervisor_thread = None
        self.running = False

    def start(self):
        """Mulai supervisor loop. Aman dipanggil lebih dari sekali."""
        if self.running:
            return

        self.running = True
        self.supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name="CameraWorkerSupervisor",
            daemon=True,
        )
        self.supervisor_thread.start()
        print("[CAMERA SUPERVISOR] Dimulai")

    def stop(self):
        self.running = False
        with self.workers_lock:
            camera_ids = list(self.workers.keys())
        for camera_id in camera_ids:
            self._stop_worker(camera_id)

    def get_worker(self, camera_id):
        with self.workers_lock:
            return self.workers.get(camera_id)

    # --------------------------------------------------------
    # SUPERVISOR LOOP
    # --------------------------------------------------------

    def _supervisor_loop(self):
        # Import di sini (bukan di top-level) supaya modul ini tidak
        # memaksa model AI ter-load sebelum benar-benar dibutuhkan --
        # StreamAIService baru dibuat saat get_instance() pertama
        # kali dipanggil, yaitu di sini.
        from services.stream_ai_service import StreamAIService

        ai_service = StreamAIService.get_instance()

        while self.running:
            try:
                self._sync_workers_with_db(ai_service)
            except Exception as exc:
                print(f"[CAMERA SUPERVISOR ERROR] {exc}")

            time.sleep(SUPERVISOR_POLL_INTERVAL)

    def _sync_workers_with_db(self, ai_service):
        cameras = db.get_all_cameras()

        # camera_id -> stream_url, hanya untuk kamera berstatus aktif
        active_cameras = {
            c["camera_id"]: c["stream_url"]
            for c in cameras
            if c.get("status") == 1 and c.get("stream_url")
        }

        with self.workers_lock:
            currently_running = set(self.workers.keys())

        # ---- Stop worker untuk kamera yang sudah tidak aktif ----
        for camera_id in currently_running - set(active_cameras.keys()):
            self._stop_worker(camera_id)

        # ---- Start worker untuk kamera yang baru jadi aktif ----
        for camera_id, stream_url in active_cameras.items():
            with self.workers_lock:
                existing = self.workers.get(camera_id)

            if existing is None:
                self._start_worker(camera_id, stream_url, ai_service)
                continue

            # Kalau URL kamera diubah (fitur Edit) sementara statusnya
            # tetap aktif, restart worker supaya pakai URL yang baru.
            if existing.stream_url != stream_url:
                print(f"[CAMERA SUPERVISOR] Kamera {camera_id}: URL berubah, restart worker")
                self._stop_worker(camera_id)
                self._start_worker(camera_id, stream_url, ai_service)

    def _start_worker(self, camera_id, stream_url, ai_service):
        worker = _CameraWorker(camera_id, stream_url, ai_service).start()
        with self.workers_lock:
            self.workers[camera_id] = worker
        print(f"[CAMERA SUPERVISOR] Kamera {camera_id}: worker dimulai ({stream_url})")

    def _stop_worker(self, camera_id):
        with self.workers_lock:
            worker = self.workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()
            print(f"[CAMERA SUPERVISOR] Kamera {camera_id}: worker dihentikan")