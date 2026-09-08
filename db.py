import os
import re
import cv2
import pymysql
import pymysql.cursors
from datetime import datetime

# ============================================================
# KONFIGURASI DATABASE (LARAGON MYSQL)
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "real_cctv"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True
}

# Direktori penyimpanan capture lokal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_DIR = os.path.join(CAPTURE_DIR, "plates")
FACE_DIR = os.path.join(CAPTURE_DIR, "faces")

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)


def get_db():
    """Mendapatkan koneksi aktif ke database MySQL."""
    return pymysql.connect(**DB_CONFIG)


# ============================================================
# LOGIKA KLASIFIKASI DETECTION STATUS (0, 1, 2)
# ============================================================

PLATE_REGEX = re.compile(r"^[A-Z]{1,2}\s?[0-9]{1,4}\s?[A-Z]{1,3}$")

def compute_plate_status(plate_number: str, ocr_conf: float) -> int:
    """
    Menghitung status deteksi plat nomor:
    1 = Terdeteksi Jelas (Confidence >= 70% dan pola plat nomor valid)
    2 = Ragu-ragu / Parsial (Confidence 40% - 70% atau teks belum sempurna)
    0 = Tidak Terdeteksi Jelas / Gagal (Confidence < 40%)
    """
    if not plate_number:
        return 0

    clean_text = str(plate_number).strip().upper()
    conf = float(ocr_conf or 0.0)

    is_valid_format = bool(PLATE_REGEX.match(clean_text))

    if conf >= 0.70 and is_valid_format:
        return 1
    elif conf >= 0.40 or is_valid_format:
        return 2
    else:
        return 0


def compute_face_status(face_conf: float) -> int:
    """
    Menghitung status deteksi wajah/orang:
    1 = Terdeteksi Jelas (Confidence >= 65%)
    2 = Ragu-ragu (Confidence 40% - 65%)
    0 = Tidak Jelas (Confidence < 40%)
    """
    conf = float(face_conf or 0.0)
    if conf >= 0.65:
        return 1
    elif conf >= 0.40:
        return 2
    else:
        return 0


# ============================================================
# HELPER PENYIMPANAN FOTO CAPTURE LOKAL
# ============================================================

def save_crop_locally(image, folder_path, prefix="cap", camera_id=1, track_id=None) -> str:
    """
    Menyimpan crop gambar OpenCV secara fisik ke direktori lokal
    dan mengembalikan path relatif web (contoh: static/captures/plates/...).
    """
    if image is None or image.size == 0:
        return None

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    tid_str = f"_{track_id}" if track_id is not None else ""
    filename = f"{prefix}_cam{camera_id}_{now_str}{tid_str}.jpg"
    abs_path = os.path.join(folder_path, filename)

    cv2.imwrite(abs_path, image)

    # Path relatif untuk web (static/captures/...)
    rel_path = os.path.relpath(abs_path, BASE_DIR).replace("\\", "/")
    return rel_path


# ============================================================
# FUNGSI MANAJEMEN KAMERA
# ============================================================

def get_all_cameras():
    """Mengambil semua daftar kamera di database."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT camera_id, location, stream_url, stream_type, status FROM cameras ORDER BY camera_id ASC;")
            return cur.fetchall()


def get_active_cameras():
    """Mengambil daftar kamera yang berstatus aktif (status = 1)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT camera_id, location, stream_url, stream_type, status FROM cameras WHERE status = 1 ORDER BY camera_id ASC;")
            return cur.fetchall()


def update_camera_status(camera_id: int, status: int):
    """Mengubah status kamera (1 = aktif, 0 = nonaktif)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE cameras SET status = %s WHERE camera_id = %s;", (status, camera_id))


def add_camera(location: str, stream_url: str, stream_type: int = 1, status: int = 1):
    """Menambahkan kamera baru ke database."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cameras (location, stream_url, stream_type, status) VALUES (%s, %s, %s, %s);",
                (location, stream_url, stream_type, status)
            )
            return cur.lastrowid


def update_camera(camera_id: int, location: str = None, stream_url: str = None, status: int = None):
    """Memperbarui informasi kamera di database."""
    fields = []
    params = []
    if location is not None:
        fields.append("location = %s")
        params.append(location)
    if stream_url is not None:
        fields.append("stream_url = %s")
        params.append(stream_url)
    if status is not None:
        fields.append("status = %s")
        params.append(status)

    if not fields:
        return False

    params.append(camera_id)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE cameras SET {', '.join(fields)} WHERE camera_id = %s;", params)
            return cur.rowcount > 0


def delete_camera(camera_id: int):
    """Menghapus kamera dari database."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cameras WHERE camera_id = %s;", (camera_id,))
            return cur.rowcount > 0


# ============================================================
# FUNGSI INSERT DETEKSI (PRODUCER / AI WORKER)
# ============================================================

def save_detection_event(
    camera_id: int,
    plate_number: str = None,
    plate_crop = None,
    plate_conf: float = 0.0,
    ocr_conf: float = 0.0,
    face_crop = None,
    face_conf: float = 0.0,
    track_id: int = None
) -> dict:
    """
    Menyimpan hasil deteksi lengkap ke database `real_cctv`:
    - Menyimpan foto crop plat & wajah ke disk lokal
    - Memasukkan ke tabel `plate` jika ada plat
    - Memasukkan ke tabel `full_detection` (plate_id diisi jika ada, atau NULL jika pejalan kaki)
    - Memasukkan log ke tabel `plate_logs`
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            plate_id = None
            plate_rel_path = None
            face_rel_path = None

            # 1. Simpan dan catat Plat Nomor (jika terdeteksi)
            if plate_crop is not None or plate_number:
                if plate_crop is not None:
                    plate_rel_path = save_crop_locally(plate_crop, PLATE_DIR, prefix="plate", camera_id=camera_id, track_id=track_id)

                plate_status = compute_plate_status(plate_number, ocr_conf)
                cur.execute(
                    """
                    INSERT INTO plate (plate_number, detection_status, detection_confidence, ocr_confidence, plate_image_path)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (plate_number, plate_status, plate_conf, ocr_conf, plate_rel_path)
                )
                plate_id = cur.lastrowid

                # Catat ke plate_logs
                log_status = 1 if plate_status == 1 else (2 if plate_status == 2 else 0)
                cur.execute(
                    """
                    INSERT INTO plate_logs (plate_id, camera_id, status)
                    VALUES (%s, %s, %s);
                    """,
                    (plate_id, camera_id, log_status)
                )

            # 2. Simpan dan catat Orang / Wajah (jika terdeteksi)
            detection_id = None
            if face_crop is not None or face_conf > 0:
                if face_crop is not None:
                    face_rel_path = save_crop_locally(face_crop, FACE_DIR, prefix="face", camera_id=camera_id, track_id=track_id)

                face_status = compute_face_status(face_conf)
                cur.execute(
                    """
                    INSERT INTO full_detection (plate_id, camera_id, detection_status, detection_confidence, face_image_path)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (plate_id, camera_id, face_status, face_conf, face_rel_path)
                )
                detection_id = cur.lastrowid

            return {
                "plate_id": plate_id,
                "detection_id": detection_id,
                "plate_image_path": plate_rel_path,
                "face_image_path": face_rel_path,
                "plate_number": plate_number
            }
    finally:
        conn.close()


# ============================================================
# FUNGSI QUERY WEB & DASHBOARD
# ============================================================

def get_recent_detections(limit: int = 50):
    """
    Mengambil data riwayat deteksi terbaru untuk halaman web dashboard & history:
    Menggabungkan tabel full_detection, plate, cameras, dan plate_logs.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    fd.detection_id,
                    fd.plate_id,
                    fd.camera_id,
                    fd.detection_status AS face_status,
                    fd.detection_confidence AS face_confidence,
                    fd.face_image_path,
                    fd.created_at AS detected_at,
                    p.plate_number,
                    p.detection_status AS plate_status,
                    p.detection_confidence AS plate_confidence,
                    p.ocr_confidence,
                    p.plate_image_path,
                    COALESCE(c1.location, c2.location, 'CCTV') AS camera_name
                FROM full_detection fd
                LEFT JOIN plate p ON fd.plate_id = p.plate_id
                LEFT JOIN cameras c1 ON fd.camera_id = c1.camera_id
                LEFT JOIN plate_logs pl ON p.plate_id = pl.plate_id
                LEFT JOIN cameras c2 ON pl.camera_id = c2.camera_id
                ORDER BY fd.created_at DESC
                LIMIT %s;
            """
            cur.execute(query, (limit,))
            return cur.fetchall()


def get_dashboard_stats():
    """Mengambil ringkasan statistik untuk widget dashboard."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total_plates FROM plate;")
            total_plates = cur.fetchone()["total_plates"]

            cur.execute("SELECT COUNT(*) AS total_full FROM full_detection;")
            total_full = cur.fetchone()["total_full"]

            cur.execute("SELECT COUNT(*) AS today_detections FROM full_detection WHERE DATE(created_at) = CURDATE();")
            today_detections = cur.fetchone()["today_detections"]

            cur.execute("SELECT COUNT(*) AS active_cameras FROM cameras WHERE status = 1;")
            active_cameras = cur.fetchone()["active_cameras"]

            return {
                "total_plates": total_plates,
                "total_full_detections": total_full,
                "today_detections": today_detections,
                "active_cameras": active_cameras
            }
