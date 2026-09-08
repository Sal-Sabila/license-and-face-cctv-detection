import time
from collections import Counter


# ============================================================
# FUNGSI IoU
# ============================================================

def hitung_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1,
    )

    inter_area = (
        inter_width * inter_height
    )

    if inter_area == 0:
        return 0.0

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - inter_area
    )

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ============================================================
# KELAS TRACK
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        bbox,
        frame_number,
        detection_confidence=0.0,
    ):
        self.id = track_id
        self.bbox = bbox

        self.last_seen_frame = frame_number

        self.detection_confidence = (
            float(detection_confidence or 0.0)
        )

        self.matches_since_last_ocr = 0

        # Semua hasil OCR:
        # {
        #   "text": "...",
        #   "confidence": 0.91,
        #   "formatted": "...",
        # }
        self.ocr_readings = []

        # Crop plat terbaik yang pernah diperoleh.
        self.best_crop = None

        # Confidence OCR tertinggi dari bacaan.
        self.best_ocr_confidence = 0.0

        # Bbox ketika crop terbaik diperoleh.
        self.best_bbox = bbox

        self.finalized = False

    def tambah_bacaan_ocr(
        self,
        text,
        confidence,
        formatted=None,
        crop=None,
        bbox=None,
    ):
        if not text:
            return

        confidence = float(
            confidence or 0.0
        )

        self.ocr_readings.append({
            "text": text,
            "confidence": confidence,
            "formatted": formatted or text,
        })

        # Simpan crop dengan confidence tertinggi.
        if (
            crop is not None
            and confidence >= self.best_ocr_confidence
        ):
            try:
                self.best_crop = crop.copy()
            except Exception:
                self.best_crop = crop

            self.best_ocr_confidence = confidence

            if bbox is not None:
                self.best_bbox = tuple(
                    int(v) for v in bbox
                )

    def hasil_voting(self):
        """
        Hasil final berdasarkan teks yang paling sering muncul.

        confidence_rata2:
            rata-rata confidence dari bacaan
            yang memenangkan voting.
        """
        if not self.ocr_readings:
            return None

        semua_teks = [
            item["text"]
            for item in self.ocr_readings
        ]

        hitungan = Counter(
            semua_teks
        )

        teks_terbanyak, jumlah_muncul = (
            hitungan.most_common(1)[0]
        )

        readings_menang = [
            item
            for item in self.ocr_readings
            if item["text"] == teks_terbanyak
        ]

        confidence_rata2 = (
            sum(
                item["confidence"]
                for item in readings_menang
            )
            / len(readings_menang)
        )

        # Gunakan bacaan dengan confidence tertinggi
        # untuk memilih formatted text.
        bacaan_terbaik = max(
            readings_menang,
            key=lambda item: item["confidence"],
        )

        formatted_terbanyak = (
            bacaan_terbaik.get(
                "formatted",
                teks_terbanyak,
            )
        )

        return {
            "text": teks_terbanyak,
            "formatted": formatted_terbanyak,
            "jumlah_muncul": jumlah_muncul,
            "total_bacaan": len(
                self.ocr_readings
            ),
            "confidence_rata2": float(
                confidence_rata2
            ),
            "confidence_best": float(
                max(
                    item["confidence"]
                    for item in readings_menang
                )
            ),
        }


# ============================================================
# PLATE TRACKER
# ============================================================

class PlateTracker:

    def __init__(
        self,
        iou_threshold=0.3,
        max_frame_gap=30,
        ocr_every_n_matches=3,
        min_final_confidence=0.20,
        max_history=5,
    ):
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap
        self.ocr_every_n_matches = (
            ocr_every_n_matches
        )

        # Hasil OCR di bawah threshold ini
        # tidak dimasukkan ke hasil final.
        self.min_final_confidence = (
            min_final_confidence
        )

        # Maksimum history yang dipertahankan.
        self.max_history = max_history

        self.tracks = {}
        self.next_track_id = 1
        self.frame_number = 0

        # Semua hasil final.
        self.hasil_final = []

        # History untuk panel CCTV.
        self.history = []

        # Capture terbaru untuk panel utama.
        self.latest_capture = None

        # Event capture terbaru.
        # Caller dapat membaca ini lalu menampilkannya.
        self.latest_finished_capture = None

    # ========================================================
    # RESET EVENT
    # ========================================================

    def consume_latest_finished_capture(self):
        """
        Ambil event capture terbaru satu kali.

        Setelah dipanggil, event di-reset menjadi None.
        """
        result = self.latest_finished_capture
        self.latest_finished_capture = None
        return result

    # ========================================================
    # VALIDASI HASIL OCR
    # ========================================================

    def _hasil_layak_ditampilkan(
        self,
        hasil,
    ):
        if hasil is None:
            return False

        text = str(
            hasil.get("text", "")
        ).strip()

        if not text:
            return False

        confidence = float(
            hasil.get(
                "confidence_rata2",
                0.0,
            )
            or 0.0
        )

        if confidence < self.min_final_confidence:
            return False

        return True

    # ========================================================
    # FINALIZE TRACK
    # ========================================================

    def _finalize_track(self, track):
        hasil = track.hasil_voting()

        if not self._hasil_layak_ditampilkan(
            hasil
        ):
            return None

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        capture = {
            "track_id": track.id,
            "text": hasil["text"],
            "formatted": hasil["formatted"],
            "jumlah_muncul": hasil[
                "jumlah_muncul"
            ],
            "total_bacaan": hasil[
                "total_bacaan"
            ],
            "confidence": hasil[
                "confidence_rata2"
            ],
            "confidence_best": hasil[
                "confidence_best"
            ],
            "detection_confidence": (
                track.detection_confidence
            ),
            "bbox": tuple(
                int(v)
                for v in track.best_bbox
            ),
            "crop": (
                track.best_crop.copy()
                if track.best_crop is not None
                else None
            ),
            "timestamp": timestamp,
        }

        # Simpan semua hasil final.
        self.hasil_final.append(
            capture.copy()
        )

        # History hanya 5 terakhir.
        self.history.insert(
            0,
            capture.copy()
        )

        self.history = self.history[
            :self.max_history
        ]

        # Panel utama menggunakan hasil terakhir.
        self.latest_capture = (
            capture.copy()
        )

        # Event yang dibaca oleh plate_test.py.
        self.latest_finished_capture = (
            capture.copy()
        )

        print(
            f"[TRACKER] Track #{track.id} selesai -> "
            f"'{hasil['formatted']}' "
            f"| OCR conf={hasil['confidence_rata2']:.1%} "
            f"| best={hasil['confidence_best']:.1%} "
            f"| "
            f"({hasil['jumlah_muncul']}/"
            f"{hasil['total_bacaan']} bacaan cocok)"
        )

        return capture

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        detections,
        frame,
        ocr_reader,
    ):
        """
        detections:
            list dari PlateDetector.detect(frame)

        frame:
            frame CCTV saat ini.

        ocr_reader:
            instance PlateOCR.

        Return:
            list Track yang aktif/match pada frame ini.
        """

        self.frame_number += 1

        track_id_yang_match = set()

        # ====================================================
        # MATCH DETEKSI KE TRACK
        # ====================================================

        for deteksi in detections:

            bbox_baru = deteksi["bbox"]

            detection_confidence = float(
                deteksi.get(
                    "confidence",
                    deteksi.get(
                        "conf",
                        0.0,
                    ),
                )
                or 0.0
            )

            track_terbaik = None
            iou_terbaik = 0.0

            for track in self.tracks.values():

                iou = hitung_iou(
                    track.bbox,
                    bbox_baru,
                )

                if iou > iou_terbaik:
                    iou_terbaik = iou
                    track_terbaik = track

            if (
                iou_terbaik >= self.iou_threshold
                and track_terbaik is not None
            ):

                # MATCH
                track = track_terbaik

                track.bbox = bbox_baru
                track.last_seen_frame = (
                    self.frame_number
                )

                track.matches_since_last_ocr += 1

                # Simpan detection confidence
                # tertinggi untuk informasi panel.
                if (
                    detection_confidence
                    > track.detection_confidence
                ):
                    track.detection_confidence = (
                        detection_confidence
                    )

            else:

                # TRACK BARU
                track = Track(
                    track_id=self.next_track_id,
                    bbox=bbox_baru,
                    frame_number=self.frame_number,
                    detection_confidence=(
                        detection_confidence
                    ),
                )

                self.next_track_id += 1

                self.tracks[
                    track.id
                ] = track

                track_id_yang_match.add(
                    track.id
                )

                # OCR langsung untuk bacaan pertama.
                track.matches_since_last_ocr = (
                    self.ocr_every_n_matches
                )

            track_id_yang_match.add(
                track.id
            )

            # =================================================
            # THROTTLE OCR
            # =================================================

            if (
                track.matches_since_last_ocr
                >= self.ocr_every_n_matches
            ):

                track.matches_since_last_ocr = 0

                x1, y1, x2, y2 = (
                    track.bbox
                )

                x1 = max(
                    0,
                    int(x1),
                )

                y1 = max(
                    0,
                    int(y1),
                )

                x2 = min(
                    frame.shape[1],
                    int(x2),
                )

                y2 = min(
                    frame.shape[0],
                    int(y2),
                )

                if (
                    x2 > x1
                    and y2 > y1
                ):

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    try:
                        hasil_ocr = (
                            ocr_reader.read(
                                crop
                            )
                        )
                    except Exception as exc:
                        print(
                            f"[OCR ERROR] "
                            f"Track #{track.id}: "
                            f"{exc}"
                        )
                        hasil_ocr = None

                    if (
                        hasil_ocr is not None
                        and hasil_ocr.get("text")
                    ):
                        track.tambah_bacaan_ocr(
                            hasil_ocr["text"],
                            hasil_ocr[
                                "confidence"
                            ],
                            formatted=(
                                hasil_ocr.get(
                                    "formatted",
                                    hasil_ocr["text"],
                                )
                            ),
                            crop=crop,
                            bbox=(
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                        )

        # ====================================================
        # CEK TRACK EXPIRED
        # ====================================================

        track_id_untuk_dihapus = []

        for track_id, track in list(
            self.tracks.items()
        ):

            gap = (
                self.frame_number
                - track.last_seen_frame
            )

            if gap > self.max_frame_gap:

                if not track.finalized:
                    track.finalized = True

                    self._finalize_track(
                        track
                    )

                track_id_untuk_dihapus.append(
                    track_id
                )

        for track_id in track_id_untuk_dihapus:
            if track_id in self.tracks:
                del self.tracks[
                    track_id
                ]

        # ====================================================
        # RETURN TRACK AKTIF
        # ====================================================

        return [
            self.tracks[tid]
            for tid in track_id_yang_match
            if tid in self.tracks
        ]