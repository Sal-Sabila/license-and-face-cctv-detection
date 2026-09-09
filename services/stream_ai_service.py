import os
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
import db

# ============================================================
# KONFIGURASI AI STREAM PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plate", "license-plate-finetune-v2n.pt")


import threading

class StreamAIService:
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("[AI STREAM] Inisialisasi model YOLO dan OCR...")
        self.lock = threading.Lock()
        self.yolo_person = YOLO(PERSON_MODEL_PATH)
        self.plate_detector = PlateDetector(PLATE_MODEL_PATH)
        self.plate_ocr = PlateOCR()

        # ByteTrack per kamera: camera_id -> sv.ByteTrack instance
        self.trackers = {}

        # Cache untuk melacak track yang sudah di-capture (mencegah duplicate insert)
        # (camera_id, key) -> timestamp capture terakhir
        self.captured_tracks = {}
        self.last_ai_times = {}
        self.last_results = {}
        print("[AI STREAM] Pipeline AI siap digunakan!")

    def _get_tracker(self, camera_id):
        """Membuat atau mengambil instance ByteTrack terisolasi per kamera."""
        cam_key = int(camera_id) if camera_id else 1
        if cam_key not in self.trackers:
            self.trackers[cam_key] = sv.ByteTrack(
                track_activation_threshold=0.35,
                lost_track_buffer=60,
                minimum_matching_threshold=0.7,
                frame_rate=25
            )
        return self.trackers[cam_key]

    def _cleanup_old_tracks(self, current_time):
        """Menghapus cache track yang sudah lewat dari 30 detik."""
        expired = [k for k, ts in list(self.captured_tracks.items()) if current_time - ts > 30.0]
        for k in expired:
            self.captured_tracks.pop(k, None)

    def process_frame(self, frame, draw_bbox=True, camera_id=1):
        """
        Memproses 1 frame video:
        1. Menjalankan deteksi AI (YOLO + ByteTrack + OCR) pada interval ~0.2s
        2. Menyimpan hasil deteksi baru secara lokal dan mencatat ke database
        3. Menggambar bounding box bersih (hanya box & label, tanpa FPS) jika draw_bbox=True
        """
        if frame is None or frame.size == 0:
            return frame

        cam_key = int(camera_id) if camera_id else 1
        now = time.time()
        self._cleanup_old_tracks(now)

        h, w = frame.shape[:2]
        last_time = self.last_ai_times.get(cam_key, 0.0)

        # Jalankan AI inference setiap interval ~0.2s per kamera agar performa optimal
        if now - last_time >= 0.2:
            self.last_ai_times[cam_key] = now

            person_dets = []
            plate_dets = []

            # Ambil model lock agar thread-safe saat banyak kamera berjalan paralel
            with self.lock:
                # 1. Deteksi Orang / Kendaraan (Person 0, Car 2, Motorcycle 3)
                try:
                    p_results = self.yolo_person(frame, classes=[0, 2, 3], imgsz=416, verbose=False)[0]
                    if len(p_results.boxes) > 0:
                        sv_dets = sv.Detections.from_ultralytics(p_results)
                        tracker = self._get_tracker(cam_key)
                        tracked = tracker.update_with_detections(sv_dets)
                        for i in range(len(tracked)):
                            box = tracked.xyxy[i].astype(int)
                            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                            person_dets.append({"box": box, "track_id": tid, "conf": conf, "cls": cls_id})
                except Exception as e:
                    pass

                # 2. Deteksi Plat Nomor
                try:
                    plate_boxes = self.plate_detector.detect(frame)
                    for pb in plate_boxes:
                        bx = [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])]
                        p_conf = float(pb[4]) if len(pb) > 4 else 0.5

                        px1, py1, px2, py2 = max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])
                        plate_crop = frame[py1:py2, px1:px2]
                        plate_text = ""
                        ocr_conf = 0.0

                        if plate_crop.size > 0 and (px2 - px1) > 25 and (py2 - py1) > 12:
                            try:
                                plate_text, ocr_conf, _ = self.plate_ocr.read_plate(plate_crop)
                            except Exception:
                                pass

                        plate_dets.append({
                            "box": bx,
                            "conf": p_conf,
                            "text": plate_text,
                            "ocr_conf": ocr_conf,
                            "crop": plate_crop
                        })
                except Exception as e:
                    pass

            self.last_results[cam_key] = {"persons": person_dets, "plates": plate_dets}

            # 3. Simpan Deteksi Baru ke Database & Lokal
            self._save_new_events(frame, person_dets, plate_dets, cam_key, now)

        # 4. Gambar Bounding Box jika draw_bbox diaktifkan
        cam_results = self.last_results.get(cam_key, {"persons": [], "plates": []})
        if draw_bbox:
            output_frame = frame.copy()
            self._draw_clean_bboxes(output_frame, cam_results)
            return output_frame

        return frame

    def _save_new_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        """Menyimpan deteksi ke database MySQL dan file lokal (dengan filter cooldown)."""
        h, w = frame.shape[:2]

        # A. Cek Plat Nomor
        for p in plate_dets:
            p_text = p.get("text")
            p_crop = p.get("crop")
            p_conf = p.get("conf", 0.0)
            ocr_conf = p.get("ocr_conf", 0.0)

            # Jika plat terbaca atau confidence bagus
            if p_text or (p_crop is not None and p_conf >= 0.45):
                cache_key = f"cam{camera_id}_plate_{p_text or id(p_crop)}"
                if cache_key not in self.captured_tracks or (current_time - self.captured_tracks[cache_key] > 5.0):
                    self.captured_tracks[cache_key] = current_time

                    # Cari person/rider yang berdekatan dengan plat ini
                    associated_person_crop = None
                    associated_person_conf = 0.0
                    for per in person_dets:
                        px1, py1, px2, py2 = per["box"]
                        bx1, by1, bx2, by2 = p["box"]
                        # Jika plat berada di area bawah orang/motor
                        if px1 <= bx2 and px2 >= bx1 and py2 >= by1 - 50:
                            p_x1, p_y1 = max(0, px1), max(0, py1)
                            p_x2, p_y2 = min(w, px2), min(h, py2)
                            associated_person_crop = frame[p_y1:p_y2, p_x1:p_x2]
                            associated_person_conf = per["conf"]
                            break

                    try:
                        db.save_detection_event(
                            camera_id=camera_id,
                            plate_number=p_text if p_text else None,
                            plate_crop=p_crop,
                            plate_conf=p_conf,
                            ocr_conf=ocr_conf,
                            face_crop=associated_person_crop,
                            face_conf=associated_person_conf
                        )
                        print(f"[AI STREAM] Saved Plate Event -> {p_text or 'Plate'} (Cam {camera_id})")
                    except Exception as e:
                        print(f"[AI STREAM ERROR] Save plate failed: {e}")

        # B. Cek Pejalan Kaki (Person yang tidak berdekatan dengan plat)
        for per in person_dets:
            tid = per.get("track_id", -1)
            cls_id = per.get("cls", 0)

            # Hanya simpan pejalan kaki (class 0 = person) yang belum dicapture
            if cls_id == 0 and tid != -1:
                cache_key = f"cam{camera_id}_person_{tid}"
                if cache_key not in self.captured_tracks or (current_time - self.captured_tracks[cache_key] > 8.0):
                    self.captured_tracks[cache_key] = current_time

                    bx1, by1, bx2, by2 = per["box"]
                    x1, y1 = max(0, bx1), max(0, by1)
                    x2, y2 = min(w, bx2), min(h, by2)
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size > 0 and (x2 - x1) >= 30 and (y2 - y1) >= 60:
                        try:
                            db.save_detection_event(
                                camera_id=camera_id,
                                plate_number=None,
                                plate_crop=None,
                                face_crop=person_crop,
                                face_conf=per["conf"],
                                track_id=tid
                            )
                            print(f"[AI STREAM] Saved Pedestrian -> Track {tid} (Cam {camera_id})")
                        except Exception as e:
                            print(f"[AI STREAM ERROR] Save person failed: {e}")

    def _draw_clean_bboxes(self, frame, results):
        """Menggambar bounding box modern & minimalis (tanpa info FPS/debug berlebihan)."""
        # 1. Gambar Person / Pengendara (Amber / Vibrant Orange)
        for per in results.get("persons", []):
            x1, y1, x2, y2 = per["box"]
            cls_id = per.get("cls", 0)
            label = "Pengendara" if cls_id in (2, 3) else "Orang"
            color = (0, 165, 255) if cls_id == 0 else (0, 200, 255)

            # Rectangle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Badge Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            bg_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Gambar Plat Nomor (Emerald Green / Neon Cyan)
        for p in results.get("plates", []):
            x1, y1, x2, y2 = p["box"]
            plate_text = p.get("text")
            label = plate_text if plate_text else "Plat Nomor"
            color = (0, 230, 118) if plate_text else (255, 215, 0)

            # Rectangle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Badge Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            bg_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 12, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
