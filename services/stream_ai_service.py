import os
import time
import re

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
import db


# ============================================================
# KONFIGURASI PATH MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PERSON_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plate",
    "license-plate-finetune-v2n.pt"
)


# ============================================================
# KONFIGURASI PIPELINE
# ============================================================

# YOLO kendaraan
VEHICLE_CONFIDENCE = 0.35
VEHICLE_IMGSZ = 640

# Plate detector
# Nilai aktual juga dikontrol oleh PlateDetector.
PLATE_CONFIDENCE = 0.50

# Interval inference
AI_INTERVAL = 0.15

# OCR
OCR_SCALE = 3.0

# Minimum ukuran crop plat untuk OCR
MIN_OCR_WIDTH = 35
MIN_OCR_HEIGHT = 12

# Cooldown capture
PLATE_COOLDOWN = 5.0
PERSON_COOLDOWN = 8.0


# ============================================================
# STREAM AI SERVICE
# ============================================================

class StreamAIService:

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self):

        print("=" * 70)
        print("[AI STREAM] Inisialisasi AI pipeline...")
        print("=" * 70)

        # ----------------------------------------------------
        # MODEL PERSON / VEHICLE
        # ----------------------------------------------------

        print("[AI STREAM] Loading vehicle model:")
        print(f"             {PERSON_MODEL_PATH}")

        self.yolo_person = YOLO(
            PERSON_MODEL_PATH
        )

        # ----------------------------------------------------
        # MODEL PLAT
        # ----------------------------------------------------

        print("[AI STREAM] Loading plate model:")
        print(f"             {PLATE_MODEL_PATH}")

        self.plate_detector = PlateDetector(
            PLATE_MODEL_PATH,
            confidence=PLATE_CONFIDENCE,
            imgsz=640,
            device="cpu",
            max_det=10,
            iou=0.45,
            min_width=30,
            min_height=10
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        try:

            self.plate_ocr = PlateOCR()

            print("[AI STREAM] OCR berhasil dimuat")

        except Exception as e:

            self.plate_ocr = None

            print(
                "[AI STREAM WARNING] OCR tidak tersedia: "
                f"{e}"
            )

        # ----------------------------------------------------
        # BYTE TRACK
        # ----------------------------------------------------

        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=25
        )

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self.captured_tracks = {}

        self.last_ai_time = 0.0

        self.last_results = {
            "persons": [],
            "plates": []
        }

        print("=" * 70)
        print("[AI STREAM] Pipeline AI siap digunakan")
        print("[AI STREAM] Vehicle classes : person/car/motorcycle/bus")
        print("[AI STREAM] Plate ROI       : kendaraan")
        print("[AI STREAM] Plate confidence:", PLATE_CONFIDENCE)
        print("[AI STREAM] OCR scale       :", OCR_SCALE)
        print("=" * 70)

    # ========================================================
    # CLEANUP TRACK
    # ========================================================

    def _cleanup_old_tracks(self, current_time):

        expired = [
            tid
            for tid, timestamp in self.captured_tracks.items()
            if current_time - timestamp > 30.0
        ]

        for tid in expired:
            del self.captured_tracks[tid]

    # ========================================================
    # NORMALISASI OCR
    # ========================================================

    def _normalize_plate_text(self, text):

        if not text:
            return ""

        text = str(text).upper().strip()

        # Hilangkan karakter yang bukan huruf/angka
        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text
        )

        return text

    # ========================================================
    # VALIDASI FORMAT PLAT INDONESIA
    # ========================================================

    def _is_valid_indonesian_plate(self, text):

        if not text:
            return False

        text = self._normalize_plate_text(
            text
        )

        # Format dasar:
        #
        # B1234CD
        # BH1234A
        # H1234AB
        # AD1234XYZ
        #
        # 1-2 huruf depan
        # 1-4 angka
        # 0-3 huruf belakang

        pattern = (
            r"^[A-Z]{1,2}"
            r"[0-9]{1,4}"
            r"[A-Z]{0,3}$"
        )

        return bool(
            re.match(pattern, text)
        )

    # ========================================================
    # UPSCALE OCR
    # ========================================================

    def _prepare_ocr_crop(self, plate_crop):

        if plate_crop is None:
            return None

        if plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]

        if w < MIN_OCR_WIDTH:
            return None

        if h < MIN_OCR_HEIGHT:
            return None

        # Upscale 3x
        enlarged = cv2.resize(
            plate_crop,
            None,
            fx=OCR_SCALE,
            fy=OCR_SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        return enlarged

    # ========================================================
    # DETEKSI KENDARAAN
    # ========================================================

    def _detect_vehicles(self, frame):

        detections = []

        try:

            results = self.yolo_person(
                frame,
                classes=[
                    0,  # person
                    2,  # car
                    3,  # motorcycle
                    5   # bus
                ],
                imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONFIDENCE,
                verbose=False
            )[0]

            if results.boxes is None:
                return detections

            if len(results.boxes) == 0:
                return detections

            sv_dets = sv.Detections.from_ultralytics(
                results
            )

            tracked = (
                self.person_tracker
                .update_with_detections(sv_dets)
            )

            for i in range(len(tracked)):

                box = tracked.xyxy[i].astype(int)

                x1, y1, x2, y2 = box.tolist()

                tid = -1

                if tracked.tracker_id is not None:
                    tid = int(
                        tracked.tracker_id[i]
                    )

                conf = 0.5

                if tracked.confidence is not None:
                    conf = float(
                        tracked.confidence[i]
                    )

                cls_id = 0

                if tracked.class_id is not None:
                    cls_id = int(
                        tracked.class_id[i]
                    )

                detections.append({
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "track_id": tid,
                    "conf": conf,
                    "cls": cls_id
                })

        except Exception as e:

            print(
                "[AI STREAM ERROR] "
                f"Vehicle detection failed: {e}"
            )

        return detections

    # ========================================================
    # DETEKSI PLAT DI DALAM ROI KENDARAAN
    # ========================================================

    def _detect_plates_in_vehicles(
        self,
        frame,
        vehicle_dets
    ):

        h, w = frame.shape[:2]

        all_plates = []

        for vehicle in vehicle_dets:

            cls_id = vehicle.get(
                "cls",
                0
            )

            # Hanya kendaraan
            #
            # 2 = car
            # 3 = motorcycle
            # 5 = bus

            if cls_id not in (
                2,
                3,
                5
            ):
                continue

            vx1, vy1, vx2, vy2 = (
                vehicle["box"]
            )

            vx1 = max(
                0,
                min(int(vx1), w - 1)
            )

            vy1 = max(
                0,
                min(int(vy1), h - 1)
            )

            vx2 = max(
                0,
                min(int(vx2), w - 1)
            )

            vy2 = max(
                0,
                min(int(vy2), h - 1)
            )

            if vx2 <= vx1:
                continue

            if vy2 <= vy1:
                continue

            vehicle_width = vx2 - vx1
            vehicle_height = vy2 - vy1

            if vehicle_width < 50:
                continue

            if vehicle_height < 40:
                continue

            # ------------------------------------------------
            # MARGIN ROI
            # ------------------------------------------------

            pad_x = int(
                vehicle_width * 0.05
            )

            pad_y = int(
                vehicle_height * 0.08
            )

            rx1 = max(
                0,
                vx1 - pad_x
            )

            ry1 = max(
                0,
                vy1 - pad_y
            )

            rx2 = min(
                w,
                vx2 + pad_x
            )

            ry2 = min(
                h,
                vy2 + pad_y
            )

            vehicle_crop = frame[
                ry1:ry2,
                rx1:rx2
            ]

            if vehicle_crop.size == 0:
                continue

            # ------------------------------------------------
            # YOLO PLAT
            # HANYA DI DALAM KENDARAAN
            # ------------------------------------------------

            try:

                local_plates = (
                    self.plate_detector
                    .detect(vehicle_crop)
                )

            except Exception as e:

                print(
                    "[AI STREAM ERROR] "
                    f"Plate ROI failed: {e}"
                )

                continue

            # ------------------------------------------------
            # KONVERSI KOORDINAT ROI → FRAME
            # ------------------------------------------------

            for plate in local_plates:

                local_box = plate.get(
                    "bbox",
                    []
                )

                if len(local_box) != 4:
                    continue

                bx1, by1, bx2, by2 = [
                    int(v)
                    for v in local_box
                ]

                # ROI → frame
                bx1 += rx1
                by1 += ry1

                bx2 += rx1
                by2 += ry1

                bx1 = max(
                    0,
                    min(bx1, w - 1)
                )

                by1 = max(
                    0,
                    min(by1, h - 1)
                )

                bx2 = max(
                    0,
                    min(bx2, w - 1)
                )

                by2 = max(
                    0,
                    min(by2, h - 1)
                )

                if bx2 <= bx1:
                    continue

                if by2 <= by1:
                    continue

                # ------------------------------------------------
                # CROP DARI FRAME ASLI
                # ------------------------------------------------

                plate_crop = frame[
                    by1:by2,
                    bx1:bx2
                ]

                if plate_crop.size == 0:
                    continue

                # Hitung center global
                center_x = int(
                    (bx1 + bx2) / 2
                )

                center_y = int(
                    (by1 + by2) / 2
                )

                new_plate = dict(
                    plate
                )

                new_plate["bbox"] = [
                    bx1,
                    by1,
                    bx2,
                    by2
                ]

                new_plate["crop"] = (
                    plate_crop
                )

                new_plate["center"] = [
                    center_x,
                    center_y
                ]

                new_plate["vehicle_cls"] = (
                    cls_id
                )

                new_plate["vehicle_box"] = [
                    vx1,
                    vy1,
                    vx2,
                    vy2
                ]

                all_plates.append(
                    new_plate
                )

        # ====================================================
        # SORT CONFIDENCE
        # ====================================================

        all_plates.sort(
            key=lambda x: x.get(
                "confidence",
                0.0
            ),
            reverse=True
        )

        # ====================================================
        # REMOVE DUPLICATE
        # ====================================================

        final_plates = []

        for candidate in all_plates:

            cbox = candidate.get(
                "bbox",
                []
            )

            if len(cbox) != 4:
                continue

            cx = (
                cbox[0] + cbox[2]
            ) / 2

            cy = (
                cbox[1] + cbox[3]
            ) / 2

            duplicate = False

            for existing in final_plates:

                ebox = existing.get(
                    "bbox",
                    []
                )

                if len(ebox) != 4:
                    continue

                ecx = (
                    ebox[0] + ebox[2]
                ) / 2

                ecy = (
                    ebox[1] + ebox[3]
                ) / 2

                distance = (
                    (cx - ecx) ** 2
                    +
                    (cy - ecy) ** 2
                ) ** 0.5

                if distance < 15:
                    duplicate = True
                    break

            if not duplicate:
                final_plates.append(
                    candidate
                )

        return final_plates

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame,
        draw_bbox=True,
        camera_id=1
    ):

        if frame is None:
            return frame

        if frame.size == 0:
            return frame

        now = time.time()

        self._cleanup_old_tracks(
            now
        )

        h, w = frame.shape[:2]

        # ====================================================
        # AI INTERVAL
        # ====================================================

        if (
            now - self.last_ai_time
            >= AI_INTERVAL
        ):

            self.last_ai_time = now

            # =================================================
            # 1. DETEKSI PERSON + VEHICLE
            # =================================================

            person_dets = (
                self._detect_vehicles(
                    frame
                )
            )

            # =================================================
            # 2. DETEKSI PLAT
            # =================================================

            plate_dets = []

            try:

                plate_boxes = (
                    self._detect_plates_in_vehicles(
                        frame,
                        person_dets
                    )
                )

                for pb in plate_boxes:

                    bx = [
                        int(v)
                        for v in pb.get(
                            "bbox",
                            []
                        )
                    ]

                    if len(bx) != 4:
                        continue

                    p_conf = float(
                        pb.get(
                            "confidence",
                            0.0
                        )
                    )

                    # =================================================
                    # EXPAND CROP SEDIKIT
                    # =================================================

                    box_width = max(
                        1,
                        bx[2] - bx[0]
                    )

                    box_height = max(
                        1,
                        bx[3] - bx[1]
                    )

                    pad_x = max(
                        2,
                        int(
                            box_width * 0.08
                        )
                    )

                    pad_y = max(
                        2,
                        int(
                            box_height * 0.12
                        )
                    )

                    px1 = max(
                        0,
                        bx[0] - pad_x
                    )

                    py1 = max(
                        0,
                        bx[1] - pad_y
                    )

                    px2 = min(
                        w,
                        bx[2] + pad_x
                    )

                    py2 = min(
                        h,
                        bx[3] + pad_y
                    )

                    plate_crop = frame[
                        py1:py2,
                        px1:px2
                    ]

                    plate_text = ""
                    normalized_text = ""
                    ocr_conf = 0.0

                    # =================================================
                    # OCR
                    # =================================================

                    ocr_crop = (
                        self._prepare_ocr_crop(
                            plate_crop
                        )
                    )

                    if (
                        self.plate_ocr
                        is not None
                        and ocr_crop is not None
                    ):

                        try:

                            ocr_result = (
                                self.plate_ocr.read(
                                    ocr_crop
                                )
                            )

                            if isinstance(
                                ocr_result,
                                dict
                            ):

                                plate_text = (
                                    ocr_result.get(
                                        "formatted"
                                    )
                                    or
                                    ocr_result.get(
                                        "text"
                                    )
                                    or
                                    ""
                                ).strip()

                                ocr_conf = float(
                                    ocr_result.get(
                                        "confidence"
                                    )
                                    or 0.0
                                )

                            elif isinstance(
                                ocr_result,
                                (
                                    tuple,
                                    list
                                )
                            ):

                                if ocr_result:

                                    plate_text = (
                                        str(
                                            ocr_result[0]
                                            or ""
                                        ).strip()
                                    )

                                if len(
                                    ocr_result
                                ) > 1:

                                    ocr_conf = float(
                                        ocr_result[1]
                                        or 0.0
                                    )

                        except Exception as e:

                            print(
                                "[AI STREAM ERROR] "
                                f"OCR failed: {e}"
                            )

                    # =================================================
                    # NORMALISASI
                    # =================================================

                    normalized_text = (
                        self._normalize_plate_text(
                            plate_text
                        )
                    )

                    # =================================================
                    # VALIDASI FORMAT
                    # =================================================

                    valid_plate = (
                        self._is_valid_indonesian_plate(
                            normalized_text
                        )
                    )

                    if plate_text:

                        print(
                            "[AI STREAM] "
                            f"Plate candidate: "
                            f"{plate_text} "
                            f"-> {normalized_text} "
                            f"| YOLO={p_conf:.3f} "
                            f"| OCR={ocr_conf:.3f} "
                            f"| VALID={valid_plate}"
                        )

                    plate_dets.append({
                        "box": [
                            bx[0],
                            bx[1],
                            bx[2],
                            bx[3]
                        ],
                        "conf": p_conf,
                        "text": normalized_text,
                        "raw_text": plate_text,
                        "ocr_conf": ocr_conf,
                        "valid": valid_plate,
                        "crop": plate_crop
                    })

            except Exception as e:

                print(
                    "[AI STREAM ERROR] "
                    f"Plate detection failed: {e}"
                )

            # =================================================
            # SIMPAN HASIL TERAKHIR
            # =================================================

            self.last_results = {
                "persons": person_dets,
                "plates": plate_dets
            }

            # =================================================
            # DATABASE
            # =================================================

            self._save_new_events(
                frame,
                person_dets,
                plate_dets,
                camera_id,
                now
            )

        # ====================================================
        # DRAW
        # ====================================================

        if draw_bbox:

            output_frame = frame.copy()

            self._draw_clean_bboxes(
                output_frame,
                self.last_results
            )

            return output_frame

        return frame

    # ========================================================
    # SAVE EVENTS
    # ========================================================

    def _save_new_events(
        self,
        frame,
        person_dets,
        plate_dets,
        camera_id,
        current_time
    ):

        h, w = frame.shape[:2]

        # ====================================================
        # A. PLAT
        # ====================================================

        for p in plate_dets:

            p_text = (
                p.get("text")
                or ""
            )

            p_crop = p.get(
                "crop"
            )

            p_conf = float(
                p.get(
                    "conf",
                    0.0
                )
            )

            ocr_conf = float(
                p.get(
                    "ocr_conf",
                    0.0
                )
            )

            valid_plate = bool(
                p.get(
                    "valid",
                    False
                )
            )

            # ------------------------------------------------
            # JANGAN langsung anggap OCR sebagai plat
            # ------------------------------------------------

            if not valid_plate:

                # Tetap tampilkan di layar,
                # tetapi jangan simpan sebagai
                # nomor plat yang valid.
                #
                # Jika confidence YOLO cukup tinggi,
                # crop masih bisa disimpan untuk review.
                if p_conf < 0.60:
                    continue

                p_text_to_db = None

            else:

                p_text_to_db = p_text

            # ------------------------------------------------
            # CACHE KEY
            # ------------------------------------------------

            if p_text_to_db:

                cache_key = (
                    f"plate_{p_text_to_db}"
                )

            else:

                bx = p.get(
                    "box",
                    [0, 0, 0, 0]
                )

                center = (
                    f"{int((bx[0] + bx[2]) / 2)}_"
                    f"{int((bx[1] + bx[3]) / 2)}"
                )

                cache_key = (
                    f"plate_unknown_"
                    f"{camera_id}_"
                    f"{center}"
                )

            # ------------------------------------------------
            # COOLDOWN
            # ------------------------------------------------

            if (
                cache_key
                not in self.captured_tracks
                or
                (
                    current_time
                    -
                    self.captured_tracks[
                        cache_key
                    ]
                    >
                    PLATE_COOLDOWN
                )
            ):

                self.captured_tracks[
                    cache_key
                ] = current_time

                # ------------------------------------------------
                # CARI KENDARAAN TERKAIT
                # ------------------------------------------------

                associated_person_crop = None
                associated_person_conf = 0.0

                bx1, by1, bx2, by2 = (
                    p.get(
                        "box",
                        [0, 0, 0, 0]
                    )
                )

                for vehicle in person_dets:

                    cls_id = vehicle.get(
                        "cls",
                        0
                    )

                    if cls_id not in (
                        2,
                        3,
                        5
                    ):
                        continue

                    vx1, vy1, vx2, vy2 = (
                        vehicle["box"]
                    )

                    # Cek intersection
                    intersects = (
                        vx1 <= bx2
                        and
                        vx2 >= bx1
                        and
                        vy1 <= by2
                        and
                        vy2 >= by1
                    )

                    if not intersects:
                        continue

                    p_x1 = max(
                        0,
                        int(vx1)
                    )

                    p_y1 = max(
                        0,
                        int(vy1)
                    )

                    p_x2 = min(
                        w,
                        int(vx2)
                    )

                    p_y2 = min(
                        h,
                        int(vy2)
                    )

                    if (
                        p_x2 > p_x1
                        and
                        p_y2 > p_y1
                    ):

                        associated_person_crop = (
                            frame[
                                p_y1:p_y2,
                                p_x1:p_x2
                            ]
                        )

                        associated_person_conf = (
                            float(
                                vehicle.get(
                                    "conf",
                                    0.0
                                )
                            )
                        )

                    break

                # ------------------------------------------------
                # SAVE DATABASE
                # ------------------------------------------------

                try:

                    db.save_detection_event(
                        camera_id=camera_id,
                        plate_number=p_text_to_db,
                        plate_crop=p_crop,
                        plate_conf=p_conf,
                        ocr_conf=ocr_conf,
                        face_crop=associated_person_crop,
                        face_conf=associated_person_conf
                    )

                    if valid_plate:

                        print(
                            "[AI STREAM] "
                            f"Saved VALID Plate -> "
                            f"{p_text_to_db} "
                            f"(YOLO={p_conf:.3f}, "
                            f"OCR={ocr_conf:.3f}, "
                            f"Cam={camera_id})"
                        )

                    else:

                        print(
                            "[AI STREAM] "
                            f"Saved REVIEW Plate -> "
                            f"OCR={p_text or '-'} "
                            f"(YOLO={p_conf:.3f}, "
                            f"Cam={camera_id})"
                        )

                except Exception as e:

                    print(
                        "[AI STREAM ERROR] "
                        f"Save plate failed: {e}"
                    )

        # ====================================================
        # B. PERSON
        # ====================================================

        for per in person_dets:

            tid = per.get(
                "track_id",
                -1
            )

            cls_id = per.get(
                "cls",
                0
            )

            # Hanya person asli
            #
            # 0 = person
            if cls_id != 0:
                continue

            if tid == -1:
                continue

            cache_key = (
                f"person_{tid}"
            )

            if (
                cache_key
                not in self.captured_tracks
                or
                (
                    current_time
                    -
                    self.captured_tracks[
                        cache_key
                    ]
                    >
                    PERSON_COOLDOWN
                )
            ):

                self.captured_tracks[
                    cache_key
                ] = current_time

                bx1, by1, bx2, by2 = (
                    per["box"]
                )

                x1 = max(
                    0,
                    int(bx1)
                )

                y1 = max(
                    0,
                    int(by1)
                )

                x2 = min(
                    w,
                    int(bx2)
                )

                y2 = min(
                    h,
                    int(by2)
                )

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size == 0:
                    continue

                if (
                    x2 - x1
                    <
                    30
                ):
                    continue

                if (
                    y2 - y1
                    <
                    60
                ):
                    continue

                try:

                    db.save_detection_event(
                        camera_id=camera_id,
                        plate_number=None,
                        plate_crop=None,
                        face_crop=person_crop,
                        face_conf=per.get(
                            "conf",
                            0.0
                        ),
                        track_id=tid
                    )

                    print(
                        "[AI STREAM] "
                        f"Saved Pedestrian -> "
                        f"Track {tid} "
                        f"(Cam {camera_id})"
                    )

                except Exception as e:

                    print(
                        "[AI STREAM ERROR] "
                        f"Save person failed: {e}"
                    )

    # ========================================================
    # DRAW BOUNDING BOX
    # ========================================================

    def _draw_clean_bboxes(
        self,
        frame,
        results
    ):

        # ====================================================
        # PERSON / VEHICLE
        # ====================================================

        for per in results.get(
            "persons",
            []
        ):

            x1, y1, x2, y2 = (
                per["box"]
            )

            cls_id = per.get(
                "cls",
                0
            )

            if cls_id == 0:

                label = "Orang"

                color = (
                    0,
                    165,
                    255
                )

            elif cls_id == 2:

                label = "Mobil"

                color = (
                    0,
                    200,
                    255
                )

            elif cls_id == 3:

                label = "Motor"

                color = (
                    0,
                    200,
                    255
                )

            elif cls_id == 5:

                label = "Bus"

                color = (
                    0,
                    200,
                    255
                )

            else:

                label = "Objek"

                color = (
                    0,
                    200,
                    255
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            (
                tw,
                th
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                1
            )

            bg_y1 = max(
                0,
                y1 - th - 8
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    bg_y1
                ),
                (
                    x1 + tw + 10,
                    y1
                ),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 5,
                    y1 - 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (
                    255,
                    255,
                    255
                ),
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # PLAT
        # ====================================================

        for p in results.get(
            "plates",
            []
        ):

            x1, y1, x2, y2 = (
                p["box"]
            )

            plate_text = (
                p.get("text")
                or ""
            )

            raw_text = (
                p.get("raw_text")
                or ""
            )

            valid = bool(
                p.get(
                    "valid",
                    False
                )
            )

            p_conf = float(
                p.get(
                    "conf",
                    0.0
                )
            )

            ocr_conf = float(
                p.get(
                    "ocr_conf",
                    0.0
                )
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if valid and plate_text:

                label = (
                    f"{plate_text} "
                    f"{ocr_conf * 100:.0f}%"
                )

                # Hijau
                color = (
                    0,
                    230,
                    118
                )

            elif raw_text:

                label = (
                    f"Perlu Cek "
                    f"{p_conf * 100:.0f}%"
                )

                # Kuning
                color = (
                    0,
                    215,
                    255
                )

            else:

                label = (
                    f"Plat Nomor "
                    f"{p_conf * 100:.0f}%"
                )

                # Kuning
                color = (
                    0,
                    215,
                    255
                )

            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                color,
                2
            )

            (
                tw,
                th
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                2
            )

            bg_y1 = max(
                0,
                y1 - th - 8
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    bg_y1
                ),
                (
                    x1 + tw + 12,
                    y1
                ),
                color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 6,
                    y1 - 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    0,
                    0,
                    0
                ),
                2,
                cv2.LINE_AA
            )