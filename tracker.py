import collections
import cv2
import re
import threading
import time
from collections import Counter

# ✅ IMPORT VALIDATOR BARU
try:
    from ai.plate.validator import (
        correct_and_validate_plate,
        is_valid_plate,
        format_plate_for_display,
        normalize_plate_text,
        parse_plate,
        VALID_REGION_CODES,
    )
    HAS_PLATE_VALIDATOR = True
except ImportError:
    HAS_PLATE_VALIDATOR = False
    print("[TRACKER WARNING] ai.plate.validator tidak ditemukan. Pakai fallback regex.")
    VALID_REGION_CODES = set()


# FUNGSI IoU

def hitung_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# NORMALISASI TEKS — DENGAN KOREKSI OCR

def normalisasi_plat(text):
    """
    Normalisasi hasil OCR sebelum validasi/voting.

    Dilakukan:
      1. Uppercase
      2. Hapus karakter non-alphanumeric
      3. Ganti karakter OCR yang sering salah baca (O→0, I→1, S→5, dll)

    Catatan: Koreksi tidak agresif — hanya karakter yang sering salah.
    Koreksi kontekstual (berdasarkan posisi) dilakukan di validator.
    """
    if text is None:
        return ""

    text = str(text).upper().strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Hanya A-Z, 0-9, dan spasi
    text = re.sub(r"[^A-Z0-9 ]", "", text)

    # Gabungkan spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# VALIDASI FORMAT PLAT INDONESIA — DIPERKETAT

def validasi_format_plat(text):
    """
    Validasi format plat nomor Indonesia.

    Kalau validator tersedia, gunakan validasi berbasis kode wilayah.
    Fallback: regex sederhana.

    Contoh valid: B 1234 ABC, AB 1934 NY, A 1234 Z, BK 1234 XX
    Contoh tidak valid: IS1500IF, IU5TSR, I5001P (kode wilayah tidak ada)
    """
    if not text:
        return False

    if HAS_PLATE_VALIDATOR:
        # ✅ Pakai validator berbasis kode wilayah resmi
        result = correct_and_validate_plate(text)
        return result["valid"]

    # Fallback regex (kalau validator tidak ada)
    text = normalisasi_plat(text)
    if not text:
        return False

    compact = text.replace(" ", "")
    if len(compact) < 5 or len(compact) > 10:
        return False
    if not re.search(r"[A-Z]", compact):
        return False
    if not re.search(r"\d", compact):
        return False

    pattern = r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$"
    return bool(re.fullmatch(pattern, text))


def koreksi_plat(text):
    """
    Koreksi OCR + validasi + kembalikan teks final.

    Return (corrected_text, is_valid, penalty).
    - corrected_text: teks yang sudah dikoreksi
    - is_valid: apakah format valid
    - penalty: 0.0-1.0 (semakin besar = semakin banyak koreksi)
    """
    if not text:
        return "", False, 0.0

    if HAS_PLATE_VALIDATOR:
        result = correct_and_validate_plate(text)
        return result["corrected"], result["valid"], result["confidence_penalty"]

    # Fallback: hanya normalisasi
    normalized = normalize_plate_text(text)
    return normalized, validasi_format_plat(normalized), 0.0


# KELAS TRACK

class Track:
    def __init__(
        self,
        track_id,
        bbox,
        frame_number,
        detection_confidence=0.0,
        vehicle_track_id=None,
        vehicle_crop=None,
        vehicle_generation=1,
        vehicle_object_group_id=None,
        single_read_ocr_confidence=0.55,
        single_read_detection_confidence=0.20,
    ):
        self.id = track_id
        self.bbox = bbox
        self.last_seen_frame = frame_number
        self.detection_confidence = float(detection_confidence or 0.0)
        self.vehicle_track_id = vehicle_track_id
        self.vehicle_crop = vehicle_crop
        self.vehicle_generation = vehicle_generation
        self.vehicle_object_group_id = vehicle_object_group_id
        self.single_read_ocr_confidence = float(single_read_ocr_confidence)
        self.single_read_detection_confidence = float(single_read_detection_confidence)
        self.matches_since_last_ocr = 0
        self.ocr_attempts = 0
        self.max_ocr_attempts = 5
        self.ocr_pending = False
        self.ocr_queue_deferred = False
        self.best_frame_crop = None
        self.best_frame_crop_quality = -1.0
        self.best_frame_bbox = bbox
        self.best_frame_source = "frame_bbox"
        self.ocr_readings = []
        self.best_crop = None
        self.best_ocr_confidence = 0.0
        self.best_bbox = bbox
        self.finalized = False
        self.is_stable = False
        self.lock = threading.Lock()

    # TAMBAH OCR

    def tambah_bacaan_ocr(
        self,
        text,
        confidence,
        formatted=None,
        crop=None,
        bbox=None,
    ):
        """
        Tambah bacaan OCR. Text yang disimpan adalah hasil KOREKSI.
        """
        # ✅ Koreksi dulu sebelum disimpan
        corrected_text, is_valid, penalty = koreksi_plat(text)

        if not corrected_text:
            return

        confidence = float(confidence or 0.0)

        # ✅ Turunkan confidence proporsional dengan penalty koreksi
        adjusted_confidence = confidence * (1.0 - penalty * 0.5)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Format untuk display
        if is_valid:
            display_text = format_plate_for_display(corrected_text) if HAS_PLATE_VALIDATOR else corrected_text
        else:
            display_text = formatted or corrected_text

        with self.lock:
            self.ocr_readings.append({
                "text": corrected_text,           # ✅ simpan versi terkoreksi
                "raw_text": text,                 # ✅ simpan versi mentah untuk audit
                "confidence": adjusted_confidence,
                "original_confidence": confidence,
                "formatted": display_text,
                "is_valid": is_valid,
                "penalty": penalty,
            })

            # Simpan crop terbaik (hanya kalau lebih baik dari sebelumnya)
            if crop is not None and adjusted_confidence >= self.best_ocr_confidence:
                try:
                    self.best_crop = crop.copy()
                except Exception:
                    self.best_crop = crop

                self.best_ocr_confidence = adjusted_confidence

                if bbox is not None:
                    self.best_bbox = tuple(int(v) for v in bbox)

            valid_readings = [r for r in self.ocr_readings if r.get("is_valid")]
            valid_counts = Counter(r["text"] for r in valid_readings)
            if valid_counts:
                top_text, top_count = valid_counts.most_common(1)[0]
                top_readings = [r for r in valid_readings if r["text"] == top_text]
                self.is_stable = top_count >= 2 or (
                    len(valid_readings) == 1
                    and top_readings[0]["confidence"] >= self.single_read_ocr_confidence
                    and self.detection_confidence >= self.single_read_detection_confidence
                )

    # VOTING — PRIORITASKAN YANG VALID

    def hasil_voting(self):
        """
        Voting: pilih teks yang paling sering muncul.

        Kalau ada kandidat yang valid (kode wilayah Indonesia),
        prioritaskan yang valid. Kalau tidak ada, ambil yang paling sering.
        """
        with self.lock:
            readings = list(self.ocr_readings)
            is_stable = self.is_stable
            if not readings:
                return None

            # PILIH KANDIDAT TERBAIK

            # Filter: hanya readings yang valid
            valid_readings = [r for r in readings if r.get("is_valid")]

        # Kalau ada yang valid, pakai itu
        if valid_readings:
            counts = Counter(r["text"] for r in valid_readings)
            teks_terbanyak = max(
                counts,
                key=lambda text: (
                    counts[text],
                    sum(r["confidence"] for r in valid_readings if r["text"] == text)
                    / counts[text],
                    max(r["confidence"] for r in valid_readings if r["text"] == text),
                ),
            )
            jumlah_muncul = counts[teks_terbanyak]
            readings_menang = [r for r in valid_readings if r["text"] == teks_terbanyak]
        else:
            # Kalau tidak ada yang valid, fallback ke semua readings
            counts = Counter(r["text"] for r in readings)
            teks_terbanyak = max(
                counts,
                key=lambda text: (
                    counts[text],
                    sum(r["confidence"] for r in readings if r["text"] == text)
                    / counts[text],
                    max(r["confidence"] for r in readings if r["text"] == text),
                ),
            )
            jumlah_muncul = counts[teks_terbanyak]
            readings_menang = [r for r in readings if r["text"] == teks_terbanyak]

        # HITUNG CONFIDENCE

        confidence_rata2 = (
            sum(r["confidence"] for r in readings_menang)
            / max(len(readings_menang), 1)
        )

        bacaan_terbaik = max(readings_menang, key=lambda r: r["confidence"])

        return {
            "text": teks_terbanyak,
            "formatted": bacaan_terbaik.get("formatted", teks_terbanyak),
            "jumlah_muncul": jumlah_muncul,
            "total_bacaan": len(readings),
            "confidence_rata2": float(confidence_rata2),
            "confidence_best": float(max(r["confidence"] for r in readings_menang)),
            "is_valid": bool(bacaan_terbaik.get("is_valid", False)),
            "has_valid_candidate": bool(valid_readings),
            # Audit trail
            "raw_texts": [r.get("raw_text", "") for r in readings],
            "corrected_texts": [r["text"] for r in readings],
            "is_stable": is_stable,
        }


# PLATE TRACKER V2

class PlateTracker:
    def __init__(
        self,
        iou_threshold=0.3,
        max_frame_gap=30,
        ocr_every_n_matches=3,
        min_detection_confidence=0.20,
        min_ocr_confidence=0.45,
        min_final_confidence=0.45,
        min_consistent_reads=2,
        single_read_ocr_confidence=0.55,
        single_read_detection_confidence=0.20,
        max_history=5,
        min_ocr_width=30,
        min_ocr_height=10,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_frame_gap = int(max_frame_gap)
        self.ocr_every_n_matches = int(ocr_every_n_matches)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_ocr_confidence = float(min_ocr_confidence)
        self.min_final_confidence = float(min_final_confidence)
        self.min_consistent_reads = int(min_consistent_reads)
        self.single_read_ocr_confidence = float(single_read_ocr_confidence)
        self.single_read_detection_confidence = float(single_read_detection_confidence)
        self.max_history = int(max_history)
        self.min_ocr_width = max(1, int(min_ocr_width))
        self.min_ocr_height = max(1, int(min_ocr_height))

        self.tracks = {}
        self.next_track_id = 1
        self.frame_number = 0
        self.hasil_final = []
        self.history = []
        self.latest_capture = None
        self.latest_finished_capture = None
        self.finished_captures = collections.deque(maxlen=100)

    def consume_finished_captures(self):
        results = list(self.finished_captures)
        self.finished_captures.clear()
        self.latest_finished_capture = None
        return results

    @staticmethod
    def _get_ocr_crop(detection, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop_width = x2 - x1
        crop_height = y2 - y1
        detected_crop = detection.get("crop")
        if (
            detected_crop is not None
            and hasattr(detected_crop, "shape")
            and getattr(detected_crop, "size", 0) > 0
            and len(detected_crop.shape) >= 2
            and int(crop_width * 0.75) <= detected_crop.shape[1] <= int(crop_width * 1.5)
            and int(crop_height * 0.75) <= detected_crop.shape[0] <= int(crop_height * 1.5)
        ):
            crop = detected_crop
            source = "detector"
        else:
            pad_x = max(2, int(crop_width * 0.08))
            pad_y = max(2, int(crop_height * 0.08))
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(frame.shape[1], x2 + pad_x)
            cy2 = min(frame.shape[0], y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            source = "frame_bbox"

        if (
            crop is None
            or not hasattr(crop, "shape")
            or getattr(crop, "size", 0) == 0
            or len(crop.shape) < 2
            or crop.shape[1] < 1
            or crop.shape[0] < 1
        ):
            return None, source
        return crop, source

    @staticmethod
    def _ocr_crop_quality(crop):
        height, width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        resolution_score = min(1.0, (width * height) / float(192 * 64))
        focus_score = min(1.0, sharpness / 150.0)
        return 0.55 * resolution_score + 0.45 * focus_score

    def _remember_best_ocr_crop(self, track, crop, bbox, source):
        if crop.shape[1] < self.min_ocr_width or crop.shape[0] < self.min_ocr_height:
            return
        try:
            quality = self._ocr_crop_quality(crop)
        except cv2.error as exc:
            print(f"[OCR CROP WARNING] Track #{track.id}: quality check failed: {exc}")
            quality = (crop.shape[1] * crop.shape[0]) / float(192 * 64)
        if quality >= track.best_frame_crop_quality:
            track.best_frame_crop = crop.copy()
            track.best_frame_crop_quality = quality
            track.best_frame_bbox = bbox
            track.best_frame_source = source

    def consume_latest_finished_capture(self):
        if self.finished_captures:
            result = self.finished_captures.popleft()
            self.latest_finished_capture = (
                self.finished_captures[-1] if self.finished_captures else None
            )
            return result

        result = self.latest_finished_capture
        self.latest_finished_capture = None
        return result

    # VALIDASI KELAYAKAN

    def _hasil_layak_ditampilkan(self, hasil, track):
        if hasil is None:
            return False

        text = normalisasi_plat(hasil.get("text", ""))
        if not text:
            return False

        # OCR confidence
        confidence = float(hasil.get("confidence_rata2", 0.0) or 0.0)
        if confidence < self.min_final_confidence:
            print(
                f"[FILTER] Track #{track.id} "
                f"OCR confidence terlalu rendah: {confidence:.1%}"
            )
            return False

        # ✅ Validasi format pakai validator baru
        if not validasi_format_plat(text):
            print(
                f"[FILTER] Track #{track.id} "
                f"format tidak valid: '{text}' "
                f"(kode wilayah tidak dikenal)"
            )
            return False

        # Konsistensi
        jumlah_muncul = int(hasil.get("jumlah_muncul", 0))
        total_bacaan = int(hasil.get("total_bacaan", 0))

        # ✅ Mode normal: minimal N bacaan yang sama
        if jumlah_muncul >= self.min_consistent_reads:
            return True

        # ✅ Mode single strong reading
        if (
            total_bacaan == 1
            and confidence >= self.single_read_ocr_confidence
            and track.detection_confidence >= self.single_read_detection_confidence
        ):
            # ✅ Single read harus valid (sudah dicek di atas, tapi double-check)
            if hasil.get("is_valid"):
                return True

        print(
            f"[FILTER] Track #{track.id} "
            f"tidak konsisten: {jumlah_muncul}/{total_bacaan} bacaan "
            f"| OCR={confidence:.1%} "
            f"| YOLO={track.detection_confidence:.1%}"
        )
        return False

    # FINALIZE

    def _finalize_track(self, track):
        hasil = track.hasil_voting()

        if not self._hasil_layak_ditampilkan(hasil, track):
            return None

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        capture = {
            "track_id": track.id,
            "vehicle_track_id": getattr(track, "vehicle_track_id", None),
            "vehicle_object_group_id": getattr(track, "vehicle_object_group_id", None),
            "vehicle_crop": (
                track.vehicle_crop.copy()
                if getattr(track, "vehicle_crop", None) is not None
                else None
            ),
            "vehicle_generation": getattr(track, "vehicle_generation", 1),
            "text": hasil["text"],
            "formatted": hasil["formatted"],
            "jumlah_muncul": hasil["jumlah_muncul"],
            "total_bacaan": hasil["total_bacaan"],
            "confidence": hasil["confidence_rata2"],
            "confidence_best": hasil["confidence_best"],
            "detection_confidence": track.detection_confidence,
            "bbox": tuple(int(v) for v in track.best_bbox),
            "crop": (
                track.best_crop.copy()
                if track.best_crop is not None
                else None
            ),
            "timestamp": timestamp,
            # ✅ Audit trail
            "raw_texts": hasil.get("raw_texts", []),
            "corrected_texts": hasil.get("corrected_texts", []),
            "is_valid": hasil.get("is_valid", False),
            "is_stable": hasil.get("is_stable", False),
        }

        self.hasil_final.append(capture.copy())
        self.history.insert(0, capture.copy())
        self.history = self.history[:self.max_history]
        self.latest_capture = capture.copy()
        self.latest_finished_capture = capture.copy()
        self.finished_captures.append(capture.copy())

        print(
            f"[TRACKER V2] Track #{track.id} selesai -> "
            f"'{hasil['formatted']}' "
            f"| OCR={hasil['confidence_rata2']:.1%} "
            f"| YOLO={track.detection_confidence:.1%} "
            f"| {hasil['jumlah_muncul']}/{hasil['total_bacaan']} bacaan cocok"
        )

        return capture

    # UPDATE

    def update(self, detections, frame, ocr_reader):
        self.frame_number += 1
        track_id_yang_match = set()

        for deteksi in detections:
            bbox_baru = deteksi["bbox"]
            detection_confidence = float(
                deteksi.get("confidence", deteksi.get("conf", 0.0)) or 0.0
            )

            if detection_confidence < self.min_detection_confidence:
                continue

            # 1. Primary matching: Hubungkan berdasarkan vehicle_track_id / vehicle_object_group_id (ByteTrack)
            v_tid = deteksi.get("vehicle_track_id")
            v_gid = deteksi.get("vehicle_object_group_id")
            try:
                v_tid_int = int(v_tid) if v_tid is not None else -1
            except (TypeError, ValueError):
                v_tid_int = -1
            v_generation = deteksi.get("vehicle_generation")

            track_terbaik = None
            iou_terbaik = 0.0

            if v_gid is not None or v_tid_int >= 0:
                for track in self.tracks.values():
                    if track.finalized:
                        continue
                    track_gid = getattr(track, "vehicle_object_group_id", None)
                    track_generation = getattr(track, "vehicle_generation", None)
                    if (
                        v_generation is not None
                        and track_generation is not None
                        and str(track_generation) != str(v_generation)
                    ):
                        continue
                    if v_gid is not None and track_gid is not None:
                        if str(track_gid) == str(v_gid):
                            track_terbaik = track
                            break
                        continue
                    if v_tid_int >= 0 and getattr(track, "vehicle_track_id", -1) is not None:
                        try:
                            if int(track.vehicle_track_id) == v_tid_int:
                                track_terbaik = track
                                break
                        except (TypeError, ValueError):
                            pass

            # 2. Secondary matching: Fallback ke spatial IoU (untuk deteksi tanpa ID kendaraan aktif)
            if track_terbaik is None:
                for track in self.tracks.values():
                    if track.finalized:
                        continue
                    track_gid = getattr(track, "vehicle_object_group_id", None)
                    track_generation = getattr(track, "vehicle_generation", None)
                    if (
                        v_gid is not None
                        and track_gid is not None
                        and str(track_gid) != str(v_gid)
                    ):
                        continue
                    if (
                        v_generation is not None
                        and track_generation is not None
                        and str(track_generation) != str(v_generation)
                    ):
                        continue
                    iou = hitung_iou(track.bbox, bbox_baru)
                    if iou > iou_terbaik:
                        iou_terbaik = iou
                        track_terbaik = track

                if iou_terbaik < self.iou_threshold:
                    track_terbaik = None

            # Update atau buat Track baru
            if track_terbaik is not None:
                track = track_terbaik
                track.bbox = bbox_baru
                track.last_seen_frame = self.frame_number
                track.matches_since_last_ocr += 1

                if deteksi.get("vehicle_track_id") is not None:
                    track.vehicle_track_id = deteksi.get("vehicle_track_id")
                if deteksi.get("vehicle_object_group_id") is not None:
                    track.vehicle_object_group_id = deteksi.get("vehicle_object_group_id")
                if deteksi.get("vehicle_crop") is not None:
                    track.vehicle_crop = deteksi.get("vehicle_crop")
                if deteksi.get("vehicle_generation") is not None:
                    track.vehicle_generation = deteksi.get("vehicle_generation")

                if detection_confidence > track.detection_confidence:
                    track.detection_confidence = detection_confidence
            else:
                track = Track(
                    track_id=self.next_track_id,
                    bbox=bbox_baru,
                    frame_number=self.frame_number,
                    detection_confidence=detection_confidence,
                    vehicle_track_id=deteksi.get("vehicle_track_id"),
                    vehicle_crop=deteksi.get("vehicle_crop"),
                    vehicle_generation=deteksi.get("vehicle_generation", 1),
                    vehicle_object_group_id=deteksi.get("vehicle_object_group_id"),
                    single_read_ocr_confidence=self.single_read_ocr_confidence,
                    single_read_detection_confidence=self.single_read_detection_confidence,
                )
                self.next_track_id += 1
                self.tracks[track.id] = track
                track.matches_since_last_ocr = min(1, self.ocr_every_n_matches)

            track_id_yang_match.add(track.id)

            x1, y1, x2, y2 = track.bbox
            x1 = max(0, min(int(x1), frame.shape[1]))
            y1 = max(0, min(int(y1), frame.shape[0]))
            x2 = max(0, min(int(x2), frame.shape[1]))
            y2 = max(0, min(int(y2), frame.shape[0]))
            if x2 - x1 >= self.min_ocr_width and y2 - y1 >= self.min_ocr_height:
                frame_crop, frame_source = self._get_ocr_crop(
                    deteksi, frame, (x1, y1, x2, y2)
                )
                if frame_crop is not None:
                    self._remember_best_ocr_crop(
                        track, frame_crop, (x1, y1, x2, y2), frame_source
                    )

            # OCR throttle & anti-redundansi
            # Hindari pengiriman OCR jika sudah stabil atau melebihi kuota percobaan
            if (
                not track.is_stable
                and not track.ocr_pending
                and track.ocr_attempts < track.max_ocr_attempts
                and track.matches_since_last_ocr >= self.ocr_every_n_matches
            ):
                crop = track.best_frame_crop
                if crop is None:
                    continue
                crop_bbox = track.best_frame_bbox
                crop_source = track.best_frame_source
                bx1, by1, bx2, by2 = crop_bbox
                vehicle_id = (
                    track.vehicle_track_id
                    if track.vehicle_track_id is not None
                    else track.vehicle_object_group_id
                )
                ocr_context = f"t{track.id}_v{vehicle_id if vehicle_id is not None else 'unknown'}"
                print(
                    f"[OCR TRACK] track={track.id} vehicle={vehicle_id} "
                    f"bbox=({bx1},{by1},{bx2},{by2}) source={crop_source} "
                    f"crop={crop.shape[1]}x{crop.shape[0]} "
                    f"quality={track.best_frame_crop_quality:.2f} "
                    f"attempt={track.ocr_attempts + 1}/{track.max_ocr_attempts}"
                )

                if hasattr(ocr_reader, "submit") and callable(ocr_reader.submit):
                    track.ocr_pending = True

                    def _make_ocr_callback(tr, c_crop, c_bbox, source):
                        def _on_ocr_done(hasil):
                            tr.ocr_pending = False
                            if hasil is None:
                                print(
                                    f"[OCR TRACK] track={tr.id} returned no result; "
                                    f"retry={'yes' if tr.ocr_attempts < tr.max_ocr_attempts else 'limit reached'}"
                                )
                                return
                            r_text = hasil.get("text", "")
                            n_text = normalisasi_plat(r_text)
                            o_conf = float(hasil.get("confidence", 0.0) or 0.0)
                            print(
                                f"[OCR TRACK] track={tr.id} source={source} "
                                f"result='{n_text}' confidence={o_conf:.3f} "
                                f"queue_wait={float(hasil.get('queue_wait', 0.0) or 0.0):.3f}s "
                                f"elapsed={float(hasil.get('elapsed', 0.0) or 0.0):.3f}s"
                            )
                            if not n_text:
                                print(
                                    f"[OCR TRACK] track={tr.id} empty plate result; "
                                    f"retry={'yes' if tr.ocr_attempts < tr.max_ocr_attempts else 'limit reached'}"
                                )
                                return
                            if o_conf < self.min_ocr_confidence:
                                print(
                                    f"[OCR TRACK] track={tr.id} confidence below threshold; "
                                    f"retry={'yes' if tr.ocr_attempts < tr.max_ocr_attempts else 'limit reached'}"
                                )
                                return
                            c_text, is_val, pen = koreksi_plat(n_text)
                            if not is_val and o_conf < 0.55:
                                return
                            tr.tambah_bacaan_ocr(
                                text=n_text,
                                confidence=o_conf,
                                formatted=hasil.get("formatted", c_text),
                                crop=c_crop,
                                bbox=c_bbox,
                            )
                            print(
                                f"[OCR VOTE] track={tr.id} reads={len(tr.ocr_readings)} "
                                f"stable={tr.is_stable} latest='{c_text}' valid={is_val}"
                            )
                        return _on_ocr_done

                    try:
                        submitted = ocr_reader.submit(
                            crop,
                            _make_ocr_callback(
                                track, crop.copy(), crop_bbox, crop_source
                            ),
                            context=ocr_context,
                        )
                    except Exception as exc:
                        track.ocr_pending = False
                        print(f"[OCR QUEUE ERROR] Track #{track.id}: {exc}")
                        continue
                    if submitted is False:
                        track.ocr_pending = False
                        if not track.ocr_queue_deferred:
                            print(
                                f"[OCR TRACK] track={track.id} queue busy; "
                                "attempt deferred"
                            )
                            track.ocr_queue_deferred = True
                    else:
                        track.matches_since_last_ocr = 0
                        track.ocr_attempts += 1
                        track.ocr_queue_deferred = False
                        track.best_frame_crop = None
                        track.best_frame_crop_quality = -1.0
                    continue

                track.matches_since_last_ocr = 0
                track.ocr_attempts += 1
                track.ocr_queue_deferred = False
                track.best_frame_crop = None
                track.best_frame_crop_quality = -1.0
                try:
                    track.ocr_pending = True
                    hasil_ocr = ocr_reader.read(crop, context=ocr_context)
                except Exception as exc:
                    print(f"[OCR ERROR] Track #{track.id}: {exc}")
                    hasil_ocr = None
                finally:
                    track.ocr_pending = False

                if hasil_ocr is None:
                    continue

                raw_text = hasil_ocr.get("text", "")
                text = normalisasi_plat(raw_text)
                ocr_confidence = float(hasil_ocr.get("confidence", 0.0) or 0.0)
                if not text:
                    print(
                        f"[OCR TRACK] track={track.id} empty plate result; "
                        f"retry={'yes' if track.ocr_attempts < track.max_ocr_attempts else 'limit reached'}"
                    )
                    continue
                print(
                    f"[OCR TRACK] track={track.id} source={crop_source} "
                    f"result='{text}' confidence={ocr_confidence:.3f} "
                    f"elapsed={float(hasil_ocr.get('elapsed', 0.0) or 0.0):.3f}s"
                )

                if ocr_confidence < self.min_ocr_confidence:
                    print(
                        f"[OCR TRACK] track={track.id} confidence below threshold; "
                        f"retry={'yes' if track.ocr_attempts < track.max_ocr_attempts else 'limit reached'}"
                    )
                    continue

                corrected_text, is_valid, penalty = koreksi_plat(text)
                if not is_valid and ocr_confidence < 0.55:
                    continue

                track.tambah_bacaan_ocr(
                    text=text,
                    confidence=ocr_confidence,
                    formatted=hasil_ocr.get("formatted", corrected_text),
                    crop=crop,
                    bbox=crop_bbox,
                )
                print(
                    f"[OCR VOTE] track={track.id} reads={len(track.ocr_readings)} "
                    f"stable={track.is_stable} latest='{corrected_text}' valid={is_valid}"
                )

        # Expired tracks
        track_id_untuk_dihapus = []
        for track_id, track in list(self.tracks.items()):
            gap = self.frame_number - track.last_seen_frame
            if gap > self.max_frame_gap:
                if not track.finalized:
                    track.finalized = True
                    self._finalize_track(track)
                track_id_untuk_dihapus.append(track_id)

        for track_id in track_id_untuk_dihapus:
            if track_id in self.tracks:
                del self.tracks[track_id]

        return [
            self.tracks[tid]
            for tid in track_id_yang_match
            if tid in self.tracks
        ]