import collections
import concurrent.futures
import hashlib
import re
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
    ):
        self.id = track_id
        self.bbox = bbox
        self.last_seen_frame = frame_number
        self.detection_confidence = float(detection_confidence or 0.0)
        self.vehicle_track_id = vehicle_track_id
        self.vehicle_crop = vehicle_crop
        self.vehicle_generation = vehicle_generation
        self.matches_since_last_ocr = 0
        self.ocr_readings = []
        self.best_crop = None
        self.best_ocr_confidence = 0.0
        self.best_bbox = bbox
        self.finalized = False
        self.is_stable = False
        self.ocr_requests = []
        self.ocr_candidates = []
        self.ocr_candidates_submitted = False
        self.ocr_attempts = 0
        self.ocr_sampling_count = 0
        self.ocr_crop_fingerprints = set()
        self.max_ocr_attempts = 3
        self.finalize_requested = False
        self.ocr_retry_limit_logged = False

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
        # Kalau dikoreksi banyak, confidence jadi lebih rendah
        adjusted_confidence = confidence * (1.0 - penalty * 0.5)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Format untuk display
        if is_valid:
            display_text = format_plate_for_display(corrected_text) if HAS_PLATE_VALIDATOR else corrected_text
        else:
            display_text = formatted or corrected_text

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

    # VOTING — PRIORITASKAN YANG VALID

    def hasil_voting(self):
        """
        Voting: pilih teks yang paling sering muncul.

        Kalau ada kandidat yang valid (kode wilayah Indonesia),
        prioritaskan yang valid. Kalau tidak ada, ambil yang paling sering.
        """
        if not self.ocr_readings:
            return None

        # PILIH KANDIDAT TERBAIK

        # Filter: hanya readings yang valid
        valid_readings = [r for r in self.ocr_readings if r.get("is_valid")]

        # Kalau ada yang valid, pakai itu
        if valid_readings:
            # Group by text
            counter = Counter(r["text"] for r in valid_readings)
            teks_terbanyak, jumlah_muncul = counter.most_common(1)[0]

            readings_menang = [r for r in valid_readings if r["text"] == teks_terbanyak]
        else:
            # Kalau tidak ada yang valid, fallback ke semua readings
            counter = Counter(r["text"] for r in self.ocr_readings)
            teks_terbanyak, jumlah_muncul = counter.most_common(1)[0]

            readings_menang = [r for r in self.ocr_readings if r["text"] == teks_terbanyak]

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
            "total_bacaan": len(self.ocr_readings),
            "confidence_rata2": float(confidence_rata2),
            "confidence_best": float(max(r["confidence"] for r in readings_menang)),
            "is_valid": bool(bacaan_terbaik.get("is_valid", False)),
            "has_valid_candidate": bool(valid_readings),
            # Audit trail
            "raw_texts": [r.get("raw_text", "") for r in self.ocr_readings],
            "corrected_texts": [r["text"] for r in self.ocr_readings],
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
        min_consistent_reads=1,
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
        self.ocr_lane_key = f"plate-tracker-{id(self)}"
        self.pipeline_counts = collections.Counter()

    def _count_pipeline(self, stage):
        self.pipeline_counts[stage] += 1

    def consume_finished_captures(self):
        results = list(self.finished_captures)
        self.finished_captures.clear()
        self.latest_finished_capture = None
        return results

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
            self._count_pipeline("final_rejected_no_usable_readings")
            print(
                f"[OCR FINAL REJECT] Track #{track.id} "
                "reason=no_usable_readings"
            )
            return False

        text = normalisasi_plat(hasil.get("text", ""))
        if not text:
            self._count_pipeline("final_rejected_empty_voted_text")
            print(
                f"[OCR FINAL REJECT] Track #{track.id} "
                "reason=empty_voted_text"
            )
            return False

        # OCR confidence
        confidence = float(hasil.get("confidence_rata2", 0.0) or 0.0)
        if confidence < self.min_final_confidence:
            self._count_pipeline("final_rejected_confidence")
            print(
                f"[OCR FINAL REJECT] Track #{track.id} "
                f"reason=confidence_below_final_min confidence={confidence:.3f} "
                f"threshold={self.min_final_confidence:.3f}"
            )
            return False

        # ✅ Validasi format pakai validator baru
        if not validasi_format_plat(text):
            self._count_pipeline("final_rejected_plate_format")
            print(
                f"[OCR FINAL REJECT] Track #{track.id} "
                f"reason=invalid_plate_format text='{text}'"
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

        self._count_pipeline("final_rejected_insufficient_consistency")
        print(
            f"[OCR FINAL REJECT] Track #{track.id} reason=insufficient_consistency "
            f"text='{text}' "
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
        self._count_pipeline("finalized")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        capture = {
            "track_id": track.id,
            "vehicle_track_id": getattr(track, "vehicle_track_id", None),
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

    def _record_ocr_result(self, track, result, crop, bbox, detection_confidence):
        if not result:
            self._count_pipeline("ocr_rejected_empty_result")
            print(
                f"[OCR REJECT] Track #{track.id} reason=empty_worker_result"
            )
            return

        raw_text = result.get("text", "")
        text = normalisasi_plat(raw_text)
        ocr_confidence = float(result.get("confidence", 0.0) or 0.0)
        if not text:
            self._count_pipeline("ocr_empty_text")
            self._count_pipeline(
                f"ocr_rejected_{result.get('rejection_reason', 'no_read')}"
            )
            print(
                f"[OCR REJECT] Track #{track.id} "
                f"reason={result.get('rejection_reason', 'no_read')} "
                f"variant={result.get('variant', 'unknown')} "
                f"elapsed={float(result.get('elapsed', 0.0) or 0.0):.3f}s"
            )
            return
        self._count_pipeline("ocr_nonempty")
        print(
            f"[OCR READ] Track #{track.id} raw='{result.get('raw_text') or raw_text}' "
            f"text='{text or '-'}' confidence={ocr_confidence:.3f} "
            f"variant={result.get('variant', 'unknown')} "
            f"elapsed={float(result.get('elapsed', 0.0) or 0.0):.3f}s"
        )

        if ocr_confidence < self.min_ocr_confidence:
            self._count_pipeline("ocr_rejected_confidence")
            print(
                f"[OCR REJECT] Track #{track.id} "
                f"reason=confidence_below_tracker_min confidence={ocr_confidence:.3f} "
                f"threshold={self.min_ocr_confidence:.3f}"
            )
            return

        corrected_text, is_valid, _ = koreksi_plat(text)
        if not is_valid:
            print(
                f"[OCR CANDIDATE] Track #{track.id} "
                f"reason=invalid_plate_format raw='{text}' "
                f"corrected='{corrected_text}' confidence={ocr_confidence:.3f} "
                f"retained={int(ocr_confidence >= 0.60)}"
            )
            if ocr_confidence < 0.60:
                self._count_pipeline("ocr_rejected_invalid_format")
                return

        track.tambah_bacaan_ocr(
            text=text,
            confidence=ocr_confidence,
            formatted=result.get("formatted", corrected_text),
            crop=crop,
            bbox=bbox,
        )
        self._count_pipeline("ocr_recorded")
        if is_valid:
            self._count_pipeline("ocr_validated")
        else:
            self._count_pipeline("ocr_recorded_unvalidated")

        vote = track.hasil_voting()
        has_consistent_reads = bool(
            vote
            and vote.get("is_valid")
            and vote.get("confidence_rata2", 0.0) >= self.min_final_confidence
            and vote.get("jumlah_muncul", 0) >= self.min_consistent_reads
        )
        has_accepted_single_read = bool(
            vote
            and vote.get("is_valid")
            and vote.get("total_bacaan") == 1
            and vote.get("confidence_rata2", 0.0)
            >= self.single_read_ocr_confidence
            and detection_confidence >= self.single_read_detection_confidence
        )
        if is_valid and (has_consistent_reads or has_accepted_single_read):
            track.is_stable = True

    def _submit_stream_candidate(self, track, ocr_reader):
        if (
            not track.ocr_candidates
            or track.is_stable
            or track.ocr_attempts >= track.max_ocr_attempts
            or any(not item["future"].done() for item in track.ocr_requests)
        ):
            return

        candidate = track.ocr_candidates.pop(0)
        future = ocr_reader.submit(
            candidate["crop"],
            coalesce_key=(self.ocr_lane_key, track.id),
        )
        if future is None:
            print(
                f"[OCR STREAM SAMPLE SKIP] Track #{track.id} "
                "reason=invalid_crop"
            )
            return

        track.ocr_attempts += 1
        self._count_pipeline("ocr_submitted")
        track.ocr_requests.append({
            "future": future,
            "crop": candidate["crop"],
            "bbox": candidate["bbox"],
            "detection_confidence": candidate["detection_confidence"],
        })
        print(
            f"[OCR STREAM SAMPLE] Track #{track.id} "
            f"attempt={track.ocr_attempts}/{track.max_ocr_attempts} "
            f"buffered={len(track.ocr_candidates)}"
        )

    def _drain_ocr_results(self, track, ocr_reader=None):
        pending = []
        for request in track.ocr_requests:
            future = request["future"]
            if not future.done():
                pending.append(request)
                continue
            try:
                result = future.result()
            except concurrent.futures.CancelledError:
                track.ocr_attempts = max(0, track.ocr_attempts - 1)
                print(
                    f"[OCR SUPERSEDED] Track #{track.id} "
                    "reason=queued_request_cancelled"
                )
                continue
            except Exception as exc:
                print(f"[OCR ERROR] Track #{track.id}: {exc}")
                continue
            self._record_ocr_result(
                track,
                result,
                request["crop"],
                request["bbox"],
                request["detection_confidence"],
            )
        if track.is_stable:
            retained = []
            cancelled = 0
            for request in pending:
                if request["future"].cancel():
                    cancelled += 1
                    track.ocr_attempts = max(0, track.ocr_attempts - 1)
                else:
                    retained.append(request)
            if cancelled:
                print(
                    f"[OCR SEGMENT EARLY STOP] Track #{track.id} "
                    f"reason=stable_vote cancelled={cancelled}"
                )
            pending = retained
        track.ocr_requests = pending
        if track.is_stable and track.ocr_candidates:
            self._count_pipeline("ocr_candidate_discarded_stable")
            track.ocr_candidates.clear()
        elif (
            ocr_reader is not None
            and getattr(ocr_reader, "source", None) == "stream"
            and not pending
        ):
            self._submit_stream_candidate(track, ocr_reader)

    @staticmethod
    def _crop_fingerprint(crop):
        shape = tuple(int(value) for value in getattr(crop, "shape", ()))
        digest = hashlib.blake2b(digest_size=12)
        digest.update(repr(shape).encode("ascii"))
        digest.update(crop.tobytes())
        return digest.digest()

    @staticmethod
    def _candidate_quality(crop, detection_confidence):
        shape = getattr(crop, "shape", ())
        if len(shape) < 2:
            return float(detection_confidence)
        area = max(1, int(shape[0]) * int(shape[1]))
        size_score = min(1.0, area / 2500.0)
        sharpness_score = 1.0
        try:
            import cv2

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(shape) == 3 else crop
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_score = min(1.0, max(0.25, sharpness / 180.0))
        except (ImportError, TypeError, ValueError):
            pass
        return (
            float(detection_confidence)
            * (0.55 + 0.45 * size_score)
            * sharpness_score
        )

    def _submit_segment_candidates(self, track, ocr_reader):
        if track.ocr_candidates_submitted:
            return
        track.ocr_candidates_submitted = True
        candidates = sorted(
            track.ocr_candidates,
            key=lambda item: item["quality"],
            reverse=True,
        )
        selected = candidates[:track.max_ocr_attempts]
        for index, candidate in enumerate(selected):
            future = ocr_reader.submit(candidate["crop"], retry=index > 0)
            if future is None:
                print(
                    f"[OCR SEGMENT SAMPLE SKIP] Track #{track.id} "
                    "reason=invalid_crop"
                )
                continue
            track.ocr_attempts += 1
            self._count_pipeline("ocr_submitted")
            track.ocr_requests.append({
                "future": future,
                "crop": candidate["crop"],
                "bbox": candidate["bbox"],
                "detection_confidence": candidate["detection_confidence"],
                "retry": index > 0,
            })
        print(
            f"[OCR SEGMENT SAMPLES] Track #{track.id} "
            f"selected={len(selected)} unique={len(candidates)} "
            f"queued={len(track.ocr_requests)}"
        )

    def update(self, detections, frame, ocr_reader):
        self.frame_number += 1
        track_id_yang_match = set()

        for existing_track in self.tracks.values():
            self._drain_ocr_results(existing_track, ocr_reader)

        for deteksi in detections:
            bbox_baru = deteksi["bbox"]
            detection_confidence = float(
                deteksi.get("confidence", deteksi.get("conf", 0.0)) or 0.0
            )
            try:
                vehicle_track_id = int(deteksi.get("vehicle_track_id"))
            except (TypeError, ValueError):
                vehicle_track_id = None
            if vehicle_track_id is not None and vehicle_track_id < 0:
                vehicle_track_id = None

            if detection_confidence < self.min_detection_confidence:
                continue

            track_terbaik = None
            iou_terbaik = 0.0
            vehicle_track_match = False
            eligible_tracks = [
                track
                for track in self.tracks.values()
                if not track.finalized
                and not track.finalize_requested
                and track.id not in track_id_yang_match
                and self.frame_number - track.last_seen_frame <= self.max_frame_gap
            ]

            if vehicle_track_id is not None:
                matching_tracks = []
                for track in eligible_tracks:
                    try:
                        track_vehicle_id = int(track.vehicle_track_id)
                    except (TypeError, ValueError):
                        continue
                    if track_vehicle_id == vehicle_track_id:
                        matching_tracks.append(track)

                if matching_tracks:
                    track_terbaik = max(
                        matching_tracks,
                        key=lambda track: (
                            track.last_seen_frame,
                            hitung_iou(track.bbox, bbox_baru),
                        ),
                    )
                    iou_terbaik = hitung_iou(track_terbaik.bbox, bbox_baru)
                    vehicle_track_match = True

            if not vehicle_track_match:
                for track in eligible_tracks:
                    iou = hitung_iou(track.bbox, bbox_baru)
                    if iou > iou_terbaik:
                        iou_terbaik = iou
                        track_terbaik = track

            # Match track
            if track_terbaik is not None and (
                vehicle_track_match or iou_terbaik >= self.iou_threshold
            ):
                track = track_terbaik
                track.bbox = bbox_baru
                track.last_seen_frame = self.frame_number
                track.matches_since_last_ocr += 1

                if deteksi.get("vehicle_track_id") is not None:
                    track.vehicle_track_id = deteksi.get("vehicle_track_id")
                if deteksi.get("vehicle_crop") is not None:
                    track.vehicle_crop = deteksi.get("vehicle_crop")
                if deteksi.get("vehicle_generation") is not None:
                    track.vehicle_generation = deteksi.get("vehicle_generation")

                if detection_confidence > track.detection_confidence:
                    track.detection_confidence = detection_confidence
                if track.finalize_requested:
                    track.finalize_requested = False
                if vehicle_track_match:
                    self._count_pipeline("matched_by_vehicle_track_id")
                    print(
                        f"[PLATE TRACK MATCH] Track #{track.id} "
                        f"method=vehicle_track_id vehicle_track_id={vehicle_track_id} "
                        f"iou={iou_terbaik:.3f}"
                    )
            else:
                # Track baru
                track = Track(
                    track_id=self.next_track_id,
                    bbox=bbox_baru,
                    frame_number=self.frame_number,
                    detection_confidence=detection_confidence,
                    vehicle_track_id=deteksi.get("vehicle_track_id"),
                    vehicle_crop=deteksi.get("vehicle_crop"),
                    vehicle_generation=deteksi.get("vehicle_generation", 1),
                )
                self.next_track_id += 1
                self.tracks[track.id] = track
                track_id_yang_match.add(track.id)
                track.matches_since_last_ocr = self.ocr_every_n_matches

            track_id_yang_match.add(track.id)

            # OCR throttle
            if (
                not track.is_stable
                and track.matches_since_last_ocr >= self.ocr_every_n_matches
            ):
                is_segment_lane = (
                    getattr(ocr_reader, "source", None) == "segment"
                )
                has_pending_ocr = False
                if not is_segment_lane:
                    has_pending_ocr = any(
                        not item["future"].done()
                        for item in track.ocr_requests
                    )
                    if track.ocr_attempts >= track.max_ocr_attempts:
                        track.matches_since_last_ocr = 0
                        if not track.ocr_retry_limit_logged:
                            track.ocr_retry_limit_logged = True
                            self._count_pipeline("ocr_retry_limit_reached")
                            print(
                                f"[OCR RETRY LIMIT] Track #{track.id} "
                                f"attempts={track.ocr_attempts}"
                            )
                        continue

                x1, y1, x2, y2 = track.bbox
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame.shape[1], int(x2))
                y2 = min(frame.shape[0], int(y2))
                cw = x2 - x1
                ch = y2 - y1

                if cw < self.min_ocr_width or ch < self.min_ocr_height:
                    continue

                pad_x = max(2, int(cw * 0.08))
                pad_y = max(2, int(ch * 0.08))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(frame.shape[1], x2 + pad_x)
                cy2 = min(frame.shape[0], y2 + pad_y)
                detection_crop = deteksi.get("crop")
                if (
                    detection_crop is not None
                    and getattr(detection_crop, "size", 0) > 0
                    and len(getattr(detection_crop, "shape", ())) >= 2
                ):
                    crop = detection_crop
                    crop_source = "plate_detection"
                else:
                    crop = frame[cy1:cy2, cx1:cx2]
                    crop_source = "frame_fallback"

                if crop is None or getattr(crop, "size", 0) == 0:
                    print(
                        f"[OCR INPUT ERROR] Track #{track.id} crop kosong "
                        f"(bbox={tuple(bbox_baru)})"
                    )
                    continue

                crop_fingerprint = self._crop_fingerprint(crop)
                if crop_fingerprint in track.ocr_crop_fingerprints:
                    self._count_pipeline("ocr_duplicate_crop_skipped")
                    print(
                        f"[OCR SAMPLE SKIP] Track #{track.id} "
                        "reason=duplicate_crop"
                    )
                    track.matches_since_last_ocr = 0
                    continue
                track.ocr_crop_fingerprints.add(crop_fingerprint)

                crop_shape = getattr(crop, "shape", ())
                print(
                    f"[OCR INPUT] Track #{track.id} source={crop_source} "
                    f"bbox={tuple(bbox_baru)} "
                    f"crop={crop_shape[1]}x{crop_shape[0]} "
                    f"yolo_confidence={detection_confidence:.3f}"
                )

                bbox = (x1, y1, x2, y2)
                if is_segment_lane:
                    track.matches_since_last_ocr = 0
                if hasattr(ocr_reader, "submit"):
                    track.ocr_sampling_count += 1
                    candidate = {
                        "crop": crop.copy(),
                        "bbox": bbox,
                        "detection_confidence": detection_confidence,
                        "quality": self._candidate_quality(
                            crop, detection_confidence
                        ),
                        "frame_number": self.frame_number,
                    }
                    if is_segment_lane:
                        track.ocr_candidates.append(candidate)
                        track.ocr_candidates.sort(
                            key=lambda item: item["quality"],
                            reverse=True,
                        )
                        track.ocr_candidates = track.ocr_candidates[
                            :track.max_ocr_attempts
                        ]
                    elif has_pending_ocr:
                        candidate_capacity = max(
                            0, track.max_ocr_attempts - track.ocr_attempts
                        )
                        track.ocr_candidates.append(candidate)
                        track.ocr_candidates.sort(
                            key=lambda item: item["quality"],
                            reverse=True,
                        )
                        track.ocr_candidates = track.ocr_candidates[
                            :candidate_capacity
                        ]
                        if any(
                            item["frame_number"] == candidate["frame_number"]
                            for item in track.ocr_candidates
                        ):
                            self._count_pipeline("ocr_candidate_buffered")
                            print(
                                f"[OCR STREAM BUFFER] Track #{track.id} "
                                f"buffered={len(track.ocr_candidates)} "
                                f"attempts={track.ocr_attempts}/"
                                f"{track.max_ocr_attempts}"
                            )
                        track.matches_since_last_ocr = 0
                    elif (
                        not track.is_stable
                        and not any(
                            not item["future"].done()
                            for item in track.ocr_requests
                        )
                    ):
                        future = ocr_reader.submit(
                            crop,
                            coalesce_key=(self.ocr_lane_key, track.id),
                        )
                        if future is not None:
                            track.ocr_attempts += 1
                            track.matches_since_last_ocr = 0
                            self._count_pipeline("ocr_submitted")
                            track.ocr_requests.append({
                                "future": future,
                                "crop": crop.copy(),
                                "bbox": bbox,
                                "detection_confidence": detection_confidence,
                            })
                        else:
                            track.matches_since_last_ocr = 0
                else:
                    track.matches_since_last_ocr = 0
                    track.ocr_attempts += 1
                    try:
                        result = ocr_reader.read(crop)
                    except Exception as exc:
                        print(f"[OCR ERROR] Track #{track.id}: {exc}")
                        result = None
                    self._record_ocr_result(
                        track,
                        result,
                        crop,
                        bbox,
                        detection_confidence,
                    )

        # Expired tracks
        track_id_untuk_dihapus = []
        for track_id, track in list(self.tracks.items()):
            gap = self.frame_number - track.last_seen_frame
            if gap > self.max_frame_gap:
                if (
                    getattr(ocr_reader, "source", None) == "segment"
                    and not track.ocr_candidates_submitted
                ):
                    self._submit_segment_candidates(track, ocr_reader)
                track.finalize_requested = True
                self._drain_ocr_results(track, ocr_reader)
                if not track.ocr_requests:
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

    def flush_ocr(self, ocr_reader, timeout=30.0):
        """Wait for bounded OCR requests, then finalize the remaining video tracks."""
        if ocr_reader is None:
            return

        for track in self.tracks.values():
            if (
                getattr(ocr_reader, "source", None) == "segment"
                and not track.ocr_candidates_submitted
            ):
                self._submit_segment_candidates(track, ocr_reader)
            track.finalize_requested = True

        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            for track in self.tracks.values():
                self._drain_ocr_results(track, ocr_reader)
            pending = [
                request["future"]
                for track in self.tracks.values()
                for request in track.ocr_requests
                if not request["future"].done()
            ]
            if not pending:
                break
            try:
                pending[0].result(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            except concurrent.futures.TimeoutError:
                continue
            except Exception:
                continue

        for track_id, track in list(self.tracks.items()):
            self._drain_ocr_results(track, ocr_reader)
            if track.ocr_requests:
                print(
                    f"[OCR FLUSH TIMEOUT] Track #{track.id} "
                    f"pending={len(track.ocr_requests)}; finalization deferred"
                )
                continue
            if not track.finalized:
                track.finalized = True
                self._finalize_track(track)
            del self.tracks[track_id]