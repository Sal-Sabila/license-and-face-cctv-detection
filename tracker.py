import re
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
        inter_width
        * inter_height
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
# NORMALISASI TEKS
# ============================================================

def normalisasi_plat(text):
    """
    Normalisasi hasil OCR sebelum validasi/voting.
    """

    if text is None:
        return ""

    text = str(text).upper().strip()

    # Ganti karakter umum yang mengganggu
    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\t",
        " "
    )

    # Hanya A-Z dan angka
    text = re.sub(
        r"[^A-Z0-9 ]",
        "",
        text
    )

    # Gabungkan spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# VALIDASI FORMAT PLAT INDONESIA
# ============================================================

def validasi_format_plat(text):
    """
    Validasi format umum plat nomor Indonesia.

    Contoh valid:
        B 1234 ABC
        AB 1934 NY
        A 1234 Z
        BK 1234 XX

    Catatan:
    Validasi format bukan jaminan bahwa objek benar-benar plat.
    Karena itu masih dikombinasikan dengan:
    - YOLO confidence
    - OCR confidence
    - konsistensi antar-frame
    """

    text = normalisasi_plat(text)

    if not text:
        return False

    # Hapus spasi untuk pemeriksaan struktur
    compact = text.replace(
        " ",
        ""
    )

    # Panjang umum
    if len(compact) < 5:
        return False

    if len(compact) > 10:
        return False

    # Harus mempunyai huruf
    if not re.search(
        r"[A-Z]",
        compact
    ):
        return False

    # Harus mempunyai angka
    if not re.search(
        r"\d",
        compact
    ):
        return False

    # Format:
    # 1-2 huruf
    # 1-4 angka
    # 0-3 huruf belakang
    pattern = (
        r"^[A-Z]{1,2}"
        r"\s?"
        r"\d{1,4}"
        r"\s?"
        r"[A-Z]{0,3}$"
    )

    if not re.fullmatch(
        pattern,
        text
    ):
        return False

    return True


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

        self.last_seen_frame = (
            frame_number
        )

        self.detection_confidence = (
            float(
                detection_confidence or 0.0
            )
        )

        self.matches_since_last_ocr = 0

        # Semua hasil OCR yang lolos filter
        self.ocr_readings = []

        # Crop terbaik
        self.best_crop = None

        # OCR confidence terbaik
        self.best_ocr_confidence = 0.0

        # Bbox crop terbaik
        self.best_bbox = bbox

        self.finalized = False
        self.is_stable = False

    # ========================================================
    # TAMBAH OCR
    # ========================================================

    def tambah_bacaan_ocr(
        self,
        text,
        confidence,
        formatted=None,
        crop=None,
        bbox=None,
    ):

        text = normalisasi_plat(
            text
        )

        if not text:
            return

        confidence = float(
            confidence or 0.0
        )

        self.ocr_readings.append({

            "text": text,

            "confidence": confidence,

            "formatted": (
                formatted or text
            ),

        })

        # Simpan crop terbaik
        if (
            crop is not None
            and confidence
            >= self.best_ocr_confidence
        ):

            try:

                self.best_crop = (
                    crop.copy()
                )

            except Exception:

                self.best_crop = crop

            self.best_ocr_confidence = (
                confidence
            )

            if bbox is not None:

                self.best_bbox = tuple(
                    int(v)
                    for v in bbox
                )

    # ========================================================
    # VOTING
    # ========================================================

    def hasil_voting(self):

        if not self.ocr_readings:
            return None

        semua_teks = [
            item["text"]
            for item
            in self.ocr_readings
        ]

        hitungan = Counter(
            semua_teks
        )

        teks_terbanyak, jumlah_muncul = (
            hitungan.most_common(1)[0]
        )

        readings_menang = [

            item

            for item
            in self.ocr_readings

            if item["text"]
            == teks_terbanyak

        ]

        confidence_rata2 = (
            sum(
                item["confidence"]
                for item
                in readings_menang
            )
            / len(
                readings_menang
            )
        )

        bacaan_terbaik = max(
            readings_menang,
            key=lambda item:
            item["confidence"],
        )

        formatted_terbanyak = (
            bacaan_terbaik.get(
                "formatted",
                teks_terbanyak,
            )
        )

        return {

            "text":
                teks_terbanyak,

            "formatted":
                formatted_terbanyak,

            "jumlah_muncul":
                jumlah_muncul,

            "total_bacaan":
                len(
                    self.ocr_readings
                ),

            "confidence_rata2":
                float(
                    confidence_rata2
                ),

            "confidence_best":
                float(
                    max(
                        item["confidence"]
                        for item
                        in readings_menang
                    )
                ),

        }


# ============================================================
# PLATE TRACKER V2
# ============================================================

class PlateTracker:

    def __init__(
        self,
        iou_threshold=0.3,
        max_frame_gap=30,
        ocr_every_n_matches=3,
        # Detection minimum
        min_detection_confidence=0.20,
        # OCR minimum
        min_ocr_confidence=0.45,
        # Final OCR minimum
        min_final_confidence=0.45,
        # Minimal jumlah bacaan yang sama
        min_consistent_reads=1,
        # Satu bacaan boleh lolos jika format valid
        single_read_ocr_confidence=0.55,
        single_read_detection_confidence=0.20,
        max_history=5,
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

        self.tracks = {}

        self.next_track_id = 1

        self.frame_number = 0

        self.hasil_final = []

        self.history = []

        self.latest_capture = None

        self.latest_finished_capture = None

    # ========================================================
    # CONSUME EVENT
    # ========================================================

    def consume_latest_finished_capture(
        self
    ):

        result = (
            self.latest_finished_capture
        )

        self.latest_finished_capture = None

        return result

    # ========================================================
    # VALIDASI OCR
    # ========================================================

    def _hasil_layak_ditampilkan(
        self,
        hasil,
        track,
    ):

        if hasil is None:
            return False

        text = normalisasi_plat(
            hasil.get(
                "text",
                ""
            )
        )

        if not text:
            return False

        # ==============================================
        # OCR CONFIDENCE
        # ==============================================

        confidence = float(
            hasil.get(
                "confidence_rata2",
                0.0,
            )
            or 0.0
        )

        if (
            confidence
            < self.min_final_confidence
        ):
            return False

        # ==============================================
        # FORMAT PLAT
        # ==============================================

        if not validasi_format_plat(
            text
        ):
            print(
                f"[FILTER] Track #{track.id} "
                f"format tidak valid: "
                f"'{text}'"
            )

            return False

        # ==============================================
        # KONSISTENSI
        # ==============================================

        jumlah_muncul = int(
            hasil.get(
                "jumlah_muncul",
                0
            )
        )

        total_bacaan = int(
            hasil.get(
                "total_bacaan",
                0
            )
        )

        # ------------------------------------------------
        # Mode normal:
        # minimal 2 bacaan OCR yang sama
        # ------------------------------------------------

        if (
            jumlah_muncul
            >= self.min_consistent_reads
        ):
            return True

        # ------------------------------------------------
        # Mode single strong reading
        # ------------------------------------------------

        if (
            total_bacaan == 1
            and confidence
            >= self.single_read_ocr_confidence
            and track.detection_confidence
            >= self.single_read_detection_confidence
        ):
            return True

        print(
            f"[FILTER] Track #{track.id} "
            f"tidak konsisten: "
            f"{jumlah_muncul}/"
            f"{total_bacaan} bacaan "
            f"| OCR={confidence:.1%} "
            f"| YOLO="
            f"{track.detection_confidence:.1%}"
        )

        return False

    # ========================================================
    # FINALIZE
    # ========================================================

    def _finalize_track(
        self,
        track
    ):

        hasil = track.hasil_voting()

        if not self._hasil_layak_ditampilkan(
            hasil,
            track
        ):
            return None

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        capture = {

            "track_id":
                track.id,

            "text":
                hasil["text"],

            "formatted":
                hasil["formatted"],

            "jumlah_muncul":
                hasil["jumlah_muncul"],

            "total_bacaan":
                hasil["total_bacaan"],

            "confidence":
                hasil["confidence_rata2"],

            "confidence_best":
                hasil["confidence_best"],

            "detection_confidence":
                track.detection_confidence,

            "bbox":
                tuple(
                    int(v)
                    for v
                    in track.best_bbox
                ),

            "crop":
                (
                    track.best_crop.copy()
                    if track.best_crop
                    is not None
                    else None
                ),

            "timestamp":
                timestamp,

        }

        # Semua hasil final
        self.hasil_final.append(
            capture.copy()
        )

        # History
        self.history.insert(
            0,
            capture.copy()
        )

        self.history = (
            self.history[
                :self.max_history
            ]
        )

        # Capture terbaru
        self.latest_capture = (
            capture.copy()
        )

        self.latest_finished_capture = (
            capture.copy()
        )

        print(
            f"[TRACKER V2] "
            f"Track #{track.id} selesai -> "
            f"'{hasil['formatted']}' "
            f"| OCR="
            f"{hasil['confidence_rata2']:.1%} "
            f"| YOLO="
            f"{track.detection_confidence:.1%} "
            f"| "
            f"{hasil['jumlah_muncul']}/"
            f"{hasil['total_bacaan']} "
            f"bacaan cocok"
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

        self.frame_number += 1

        track_id_yang_match = set()

        # ====================================================
        # MATCH DETEKSI
        # ====================================================

        for deteksi in detections:

            bbox_baru = (
                deteksi["bbox"]
            )

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

            # Safety filter
            if (
                detection_confidence
                < self.min_detection_confidence
            ):
                continue

            track_terbaik = None

            iou_terbaik = 0.0

            # =================================================
            # CARI TRACK TERDEKAT
            # =================================================

            for track in (
                self.tracks.values()
            ):

                if track.finalized:
                    continue

                iou = hitung_iou(
                    track.bbox,
                    bbox_baru
                )

                if iou > iou_terbaik:

                    iou_terbaik = iou

                    track_terbaik = track

            # =================================================
            # MATCH
            # =================================================

            if (
                iou_terbaik
                >= self.iou_threshold
                and track_terbaik
                is not None
            ):

                track = track_terbaik

                track.bbox = (
                    bbox_baru
                )

                track.last_seen_frame = (
                    self.frame_number
                )

                track.matches_since_last_ocr += 1

                # Confidence tertinggi
                if (
                    detection_confidence
                    > track.detection_confidence
                ):

                    track.detection_confidence = (
                        detection_confidence
                    )

            # =================================================
            # TRACK BARU
            # =================================================

            else:

                track = Track(

                    track_id=(
                        self.next_track_id
                    ),

                    bbox=bbox_baru,

                    frame_number=(
                        self.frame_number
                    ),

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

                # OCR pertama langsung
                track.matches_since_last_ocr = (
                    self.ocr_every_n_matches
                )

            track_id_yang_match.add(
                track.id
            )

            # =================================================
            # OCR THROTTLE
            # =================================================

            if (
                not track.is_stable
                and track.matches_since_last_ocr
                >= self.ocr_every_n_matches
            ):

                track.matches_since_last_ocr = 0

                x1, y1, x2, y2 = (
                    track.bbox
                )

                # Batas frame
                x1 = max(
                    0,
                    int(x1)
                )

                y1 = max(
                    0,
                    int(y1)
                )

                x2 = min(
                    frame.shape[1],
                    int(x2)
                )

                y2 = min(
                    frame.shape[0],
                    int(y2)
                )

                cw = x2 - x1
                ch = y2 - y1

                # Abaikan crop yang terlalu kecil untuk menghemat CPU
                if cw < 30 or ch < 10:
                    continue

                # Crop dengan sedikit margin kontekstual
                pad_x = max(2, int(cw * 0.08))
                pad_y = max(2, int(ch * 0.08))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(frame.shape[1], x2 + pad_x)
                cy2 = min(frame.shape[0], y2 + pad_y)

                crop = frame[
                    cy1:cy2,
                    cx1:cx2
                ]

                # =================================================
                # OCR
                # =================================================

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
                    hasil_ocr is None
                ):
                    continue

                text = normalisasi_plat(
                    hasil_ocr.get(
                        "text",
                        ""
                    )
                )

                ocr_confidence = float(
                    hasil_ocr.get(
                        "confidence",
                        0.0
                    )
                    or 0.0
                )

                # =================================================
                # FILTER OCR CONFIDENCE
                # =================================================

                if (
                    ocr_confidence
                    < self.min_ocr_confidence
                ):

                    print(
                        f"[FILTER OCR] "
                        f"Track #{track.id} "
                        f"OCR={ocr_confidence:.1%} "
                        f"< "
                        f"{self.min_ocr_confidence:.1%}"
                    )

                    continue

                # =================================================
                # FILTER FORMAT
                # =================================================

                if not validasi_format_plat(
                    text
                ):

                    print(
                        f"[FILTER FORMAT] "
                        f"Track #{track.id}: "
                        f"'{text}'"
                    )

                    continue

                # =================================================
                # SIMPAN OCR
                # =================================================

                track.tambah_bacaan_ocr(

                    text,

                    ocr_confidence,

                    formatted=(
                        hasil_ocr.get(
                            "formatted",
                            text
                        )
                    ),

                    crop=crop,

                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                )

                # Jika sudah mendapatkan bacaan plat Indonesia yang valid dan jelas,
                # tandai track sebagai stabil agar tidak membebani CPU dengan OCR berulang
                if hasil_ocr.get("is_indonesia_pattern", False) and ocr_confidence >= 0.58:
                    track.is_stable = True

        # ====================================================
        # EXPIRED TRACK
        # ====================================================

        track_id_untuk_dihapus = []

        for (
            track_id,
            track
        ) in list(
            self.tracks.items()
        ):

            gap = (
                self.frame_number
                - track.last_seen_frame
            )

            if (
                gap
                > self.max_frame_gap
            ):

                if not track.finalized:

                    track.finalized = True

                    self._finalize_track(
                        track
                    )

                track_id_untuk_dihapus.append(
                    track_id
                )

        # Hapus track lama
        for track_id in (
            track_id_untuk_dihapus
        ):

            if track_id in self.tracks:

                del self.tracks[
                    track_id
                ]

        # ====================================================
        # RETURN TRACK AKTIF
        # ====================================================

        return [

            self.tracks[tid]

            for tid
            in track_id_yang_match

            if tid
            in self.tracks

        ]