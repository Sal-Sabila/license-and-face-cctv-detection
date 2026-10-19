import os
import sys
import time
import threading
import cv2

import db
from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from ai.plate.validator import correct_and_validate_plate, is_valid_plate


class BackgroundPlateOCR:
    """
    Background worker service untuk memproses OCR plat nomor pada capture kendaraan
    yang sudah tersimpan di database (full_detection) namun belum memiliki plat (has_plate = 0).

    Berjalan sebagai daemon thread terpisah tanpa mengganggu realtime stream / detection worker.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(
        self,
        poll_interval: float = 10.0,
        batch_size: int = 10,
        max_retries: int = 3,
        item_delay: float = 0.2,
    ):
        self.poll_interval = float(poll_interval)
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.item_delay = float(item_delay)

        self.running = False
        self.thread = None
        self.stop_event = threading.Event()

        self.detector = None
        self.ocr = None

        self.total_processed = 0
        self.total_success = 0
        self.total_failed = 0

    def start(self):
        """Memulai background thread. Aman dipanggil berulang."""
        if self.running:
            return self

        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._worker_loop,
            name="BackgroundPlateOCRWorker",
            daemon=True,
        )
        self.thread.start()
        print("[BACKGROUND OCR] Service dimulai (daemon thread aktif)")
        return self

    def stop(self):
        """Menghentikan background thread secara aman."""
        if not self.running:
            return
        print("[BACKGROUND OCR] Menghentikan service...")
        self.running = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        print("[BACKGROUND OCR] Service dihentikan")

    def is_running(self) -> bool:
        return self.running and self.thread is not None and self.thread.is_alive()

    def get_stats(self) -> dict:
        return {
            "running": self.is_running(),
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "total_failed": self.total_failed,
        }

    def _init_models(self):
        """Inisialisasi PlateDetector & PlateOCR secara lazy di dalam thread worker."""
        if self.detector is None:
            print("[BACKGROUND OCR] Memuat model PlateDetector...")
            self.detector = PlateDetector()

        if self.ocr is None:
            print("[BACKGROUND OCR] Memuat model PlateOCR...")
            self.ocr = PlateOCR()

    def _worker_loop(self):
        print("[BACKGROUND OCR] Worker loop berjalan")

        # Inisialisasi model di awal worker loop
        try:
            self._init_models()
        except Exception as exc:
            print(f"[BACKGROUND OCR ERROR] Gagal inisialisasi model AI: {exc}")
            self.running = False
            return

        while self.running and not self.stop_event.is_set():
            try:
                processed_any = self._process_batch()
            except Exception as exc:
                print(f"[BACKGROUND OCR ERROR] Kesalahan pada siklus batch: {exc}")
                processed_any = False

            # Jika tidak ada yang diproses, tunggu poll_interval.
            # Jika ada yang diproses, tunggu lebih singkat agar antrian cepat selesai.
            wait_time = 2.0 if processed_any else self.poll_interval
            if self.stop_event.wait(timeout=wait_time):
                break

        print("[BACKGROUND OCR] Worker loop selesai")

    def _process_batch(self) -> bool:
        """
        Ambil batch kandidat dari DB dan proses satu per satu.
        Return True jika ada kandidat yang diproses.
        """
        candidates = db.get_unprocessed_plate_vehicles(
            limit=self.batch_size,
            max_retry=self.max_retries,
        )

        if not candidates:
            return False

        print(f"[BACKGROUND OCR] Menemukan {len(candidates)} kendaraan dalam antrian scan plat")

        for candidate in candidates:
            if self.stop_event.is_set():
                break

            self._process_single_vehicle(candidate)

            # Rate limiting antar item agar CPU/GPU tidak 100% saturasi
            if self.item_delay > 0:
                time.sleep(self.item_delay)

        return True

    def _process_single_vehicle(self, candidate: dict):
        detection_id = candidate.get("detection_id")
        camera_id = candidate.get("camera_id") or 1
        track_id = candidate.get("track_id")
        vehicle_type = candidate.get("vehicle_type") or "unknown"
        vehicle_image_rel = candidate.get("vehicle_image_path")
        event_key = candidate.get("event_key")

        self.total_processed += 1

        # 1. Validasi keberadaan file gambar lokal
        if not vehicle_image_rel:
            print(f"[BACKGROUND OCR] Kendaraan ID {detection_id}: vehicle_image_path kosong")
            db.increment_plate_retry(detection_id)
            self.total_failed += 1
            return

        abs_path = os.path.join(db.BASE_DIR, vehicle_image_rel.replace("/", os.sep))
        if not os.path.isfile(abs_path):
            print(f"[BACKGROUND OCR] Kendaraan ID {detection_id}: File gambar tidak ditemukan di disk ({abs_path})")
            db.increment_plate_retry(detection_id)
            self.total_failed += 1
            return

        # 2. Buka gambar kendaraan
        try:
            vehicle_img = cv2.imread(abs_path)
        except Exception as exc:
            print(f"[BACKGROUND OCR ERROR] Gagal membuka gambar {abs_path}: {exc}")
            vehicle_img = None

        if vehicle_img is None or vehicle_img.size == 0:
            print(f"[BACKGROUND OCR] Kendaraan ID {detection_id}: Gagal mendecode gambar ({abs_path})")
            db.increment_plate_retry(detection_id)
            self.total_failed += 1
            return

        # 3. Deteksi plat dari gambar kendaraan
        try:
            plates = self.detector.detect(vehicle_img)
        except Exception as exc:
            print(f"[BACKGROUND OCR ERROR] Kendaraan ID {detection_id}: Deteksi plat gagal: {exc}")
            plates = []

        if not plates:
            # Tidak ada plat terdeteksi pada capture ini
            db.increment_plate_retry(detection_id)
            self.total_failed += 1
            return

        # 4. Coba OCR pada setiap plat yang terdeteksi (diurutkan berdasarkan confidence deteksi)
        best_ocr_candidate = None

        for plate in plates:
            plate_crop = plate.get("crop")
            plate_conf = float(plate.get("confidence") or 0.0)

            if plate_crop is None or plate_crop.size == 0:
                continue

            try:
                ocr_result = self.ocr.read(plate_crop)
            except Exception as exc:
                print(f"[BACKGROUND OCR ERROR] Kendaraan ID {detection_id}: OCR read error: {exc}")
                continue

            if not ocr_result:
                continue

            raw_text = str(ocr_result.get("raw_text") or "").strip()
            corrected_text = str(ocr_result.get("text") or "").strip()
            ocr_conf = float(ocr_result.get("confidence") or 0.0)
            is_valid = bool(ocr_result.get("is_indonesia_pattern"))

            # Verifikasi tambahan dengan validator plat Indonesia
            if not is_valid and raw_text:
                val_res = correct_and_validate_plate(raw_text, ocr_conf)
                if val_res.get("valid"):
                    corrected_text = val_res.get("corrected") or corrected_text
                    is_valid = True

            # Kondisi diterima:
            # - Plat valid menurut pola Indonesia, ATAU
            # - Confidence OCR cukup tinggi (>= 0.45) dan memiliki teks
            if is_valid or (ocr_conf >= 0.45 and len(corrected_text) >= 3):
                best_ocr_candidate = {
                    "plate_crop": plate_crop,
                    "plate_conf": plate_conf,
                    "ocr_conf": ocr_conf,
                    "raw_text": raw_text or corrected_text,
                    "corrected_text": corrected_text,
                    "is_valid": is_valid,
                }
                # Jika valid dan confidence tinggi, cukup ambil yang terbaik ini
                if is_valid and ocr_conf >= 0.70:
                    break

        # 5. Simpan hasil ke database jika ditemukan plat yang terbaca
        if best_ocr_candidate is not None:
            # Pastikan event_key tersedia agar save_detection_event melakukan UPDATE bukan INSERT baru
            valid_event_key = event_key
            if not valid_event_key or not str(valid_event_key).strip():
                valid_event_key = db.ensure_vehicle_event_key(detection_id, camera_id)

            try:
                save_res = db.save_detection_event(
                    camera_id=camera_id,
                    plate_number=best_ocr_candidate["corrected_text"],
                    plate_crop=best_ocr_candidate["plate_crop"],
                    plate_conf=best_ocr_candidate["plate_conf"],
                    ocr_conf=best_ocr_candidate["ocr_conf"],
                    raw_ocr_text=best_ocr_candidate["raw_text"],
                    track_id=track_id,
                    object_type="vehicle",
                    vehicle_type=vehicle_type,
                    event_key=valid_event_key,
                    source_type="stream",
                )

                print(
                    f"[BACKGROUND OCR] ✅ BERHASIL! Kendaraan ID {detection_id}: "
                    f"Plat '{best_ocr_candidate['corrected_text']}' "
                    f"(OCR Conf: {best_ocr_candidate['ocr_conf']:.2f}, Det Conf: {best_ocr_candidate['plate_conf']:.2f})"
                )
                self.total_success += 1
                return
            except Exception as exc:
                print(f"[BACKGROUND OCR ERROR] Gagal menyimpan event plat ke DB (ID {detection_id}): {exc}")

        # Jika tidak ada OCR yang valid atau gagal simpan
        db.increment_plate_retry(detection_id)
        self.total_failed += 1
