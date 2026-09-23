import time

# ============================================================
# KONFIGURASI COUNTING LINE PER KAMERA (NORMALIZED 0.0 - 1.0)
# ============================================================
# Setiap entri: ((x1, y1), (x2, y2)) - dua titik ujung garis.
#
# Diturunkan dari sisi ATAS polygon "mid" di CAMERA_ZONE_CONFIG
# (stream_ai_service.py) supaya konsisten dengan zona yang sudah
# dikalibrasi per kamera, tanpa perlu kalibrasi ulang.
CAMERA_LINE_CONFIG = {
    1: ((0.10, 0.38), (0.90, 0.38)),   # GSMasukViewDalam
    2: ((0.10, 0.39), (0.90, 0.39)),   # GSMasukViewLuar
    3: ((0.08, 0.37), (0.92, 0.37)),   # GSKeluarViewLuar
    4: ((0.10, 0.37), (0.92, 0.37)),   # GSKeluarViewDalam
}

DEFAULT_LINE = ((0.02, 0.40), (0.98, 0.40))

# Titik referensi "LUAR" (arah menjauhi kamera / bagian atas frame).
# Dipakai untuk menentukan sisi mana dari garis yang dianggap LUAR,
# secara otomatis, tanpa perlu input manual per kamera.
OUTSIDE_REFERENCE_Y_RATIO = 0.02  # titik dekat tepi atas frame

# Berapa detik sebuah track boleh "diam" sebelum generation-nya
# dianggap baru (disamakan dengan TRACK_REUSE_GAP di stream_ai_service).
TRACK_REUSE_GAP = 2.5

# Berapa lama state track yang sudah tidak terlihat disimpan sebelum
# dibersihkan dari memori.
STATE_TTL_SECONDS = 300.0


def _get_line(camera_id):
    try:
        camera_id_int = int(camera_id)
    except (TypeError, ValueError):
        camera_id_int = None

    if camera_id_int in CAMERA_LINE_CONFIG:
        return CAMERA_LINE_CONFIG[camera_id_int]
    return DEFAULT_LINE


def _to_pixels(point, width, height):
    x, y = point
    return (float(x) * float(width), float(y) * float(height))


def _bottom_center(bbox):
    """Titik acuan objek: tengah-bawah bounding box (titik kontak ke tanah)."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return ((x1 + x2) / 2.0, y2)


def _side_of_line(p1, p2, point):
    """
    Tanda hasil cross-product menentukan di sisi mana `point` berada
    relatif terhadap garis berarah p1 -> p2.

    Return:
        > 0  -> sisi A
        < 0  -> sisi B
        == 0 -> tepat di garis (dianggap tidak berubah sisi)
    """
    (x1, y1), (x2, y2) = p1, p2
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class LineCrossingTracker:
    """
    Melacak sisi terakhir setiap (camera_id, track_id, generation) relatif
    terhadap counting line kamera tersebut, dan mendeteksi event crossing.

    Satu event crossing = satu kali objek pindah sisi. Deteksi berulang
    pada track yang sama TIDAK dihitung lagi selama belum crossing lagi
    (mencegah double counting - lihat _save state per generation).
    """

    def __init__(self):
        # key -> {"side": float sign terakhir, "last_seen": ts}
        self._state = {}

    def _key(self, camera_id, track_id, generation):
        return (str(camera_id), int(track_id), int(generation))

    def _outside_side_sign(self, camera_id, frame_width, frame_height):
        p1, p2 = _get_line(camera_id)
        line_p1 = _to_pixels(p1, frame_width, frame_height)
        line_p2 = _to_pixels(p2, frame_width, frame_height)

        # Titik referensi yang PASTI berada di sisi LUAR (dekat tepi atas
        # frame -> menjauhi kamera / menjauhi area dalam).
        ref_point = (frame_width / 2.0, frame_height * OUTSIDE_REFERENCE_Y_RATIO)
        sign = _side_of_line(line_p1, line_p2, ref_point)
        return line_p1, line_p2, sign

    def cleanup(self, now=None):
        now = now or time.time()
        for key, state in list(self._state.items()):
            if now - state.get("last_seen", now) > STATE_TTL_SECONDS:
                self._state.pop(key, None)

    def update(
        self,
        camera_id,
        track_id,
        generation,
        bbox,
        frame_width,
        frame_height,
        camera_direction="unknown",
    ):
        """
        Perbarui posisi track dan kembalikan arah HANYA pada frame saat
        crossing benar-benar terjadi.

        Returns:
            "entry" | "exit" | "unknown"
        """
        if track_id is None or int(track_id) < 0:
            return "unknown"
        if not bbox or len(bbox) != 4 or not frame_width or not frame_height:
            return "unknown"

        now = time.time()
        key = self._key(camera_id, track_id, generation)

        line_p1, line_p2, outside_sign = self._outside_side_sign(
            camera_id, frame_width, frame_height
        )

        point = _bottom_center(bbox)
        current_sign = _side_of_line(line_p1, line_p2, point)

        # Sisi saat ini: True jika berada di sisi yang sama dengan LUAR.
        is_outside_now = (
            (current_sign >= 0) == (outside_sign >= 0)
            if current_sign != 0
            else None
        )

        state = self._state.get(key)

        if state is None:
            # Belum ada riwayat posisi untuk track ini (baru pertama kali
            # terlihat). Simpan sisi awal, belum ada crossing yang bisa
            # dipastikan pada frame ini.
            self._state[key] = {
                "is_outside": is_outside_now,
                "last_seen": now,
            }
            return "unknown"

        previous_is_outside = state.get("is_outside")
        state["last_seen"] = now

        if (
            is_outside_now is None
            or previous_is_outside is None
            or is_outside_now == previous_is_outside
        ):
            # Tidak ada perubahan sisi -> objek belum melewati counting line.
            state["is_outside"] = (
                is_outside_now if is_outside_now is not None else previous_is_outside
            )
            self._state[key] = state
            return "unknown"

        # ------------------------------------------------------------
        # CROSSING TERDETEKSI: sisi berubah dari LUAR<->DALAM
        # ------------------------------------------------------------
        state["is_outside"] = is_outside_now
        self._state[key] = state

        moved_outside_to_inside = (previous_is_outside is True) and (
            is_outside_now is False
        )

        camera_direction = camera_direction if camera_direction in (
            "entry",
            "exit",
        ) else "unknown"

        if camera_direction == "unknown":
            # Kamera belum dikonfigurasi arahnya -> crossing terdeteksi,
            # tapi tidak bisa dipastikan itu Masuk atau Keluar.
            return "unknown"

        if camera_direction == "entry":
            # Untuk kamera ini: LUAR -> DALAM berarti MASUK.
            return "entry" if moved_outside_to_inside else "exit"
        else:
            # camera_direction == "exit": LUAR -> DALAM berarti KELUAR
            # (orientasi kamera ini terbalik dari kamera "entry").
            return "exit" if moved_outside_to_inside else "entry"

    def reset_generation(self, camera_id, track_id, generation):
        """Hapus state satu lifecycle track (dipanggil saat generation berubah)."""
        self._state.pop(self._key(camera_id, track_id, generation), None)


# Satu instance dibagi oleh stream_ai_service.py DAN video_ai_service.py
# supaya konsisten dan tidak duplikasi logika tracking arah.
default_tracker = LineCrossingTracker()
