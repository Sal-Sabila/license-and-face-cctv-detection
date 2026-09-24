import os
import re
import time
import cv2
import pymysql
import pymysql.cursors
from datetime import datetime, timedelta
import random
import json

# pHash optional
try:
    import imagehash
    from PIL import Image
    HAS_PHASH = True
except ImportError:
    HAS_PHASH = False
    print("[DB WARNING] imagehash/Pillow tidak terinstall. Dedup visual dinonaktifkan.")
    print("[DB WARNING] Install dengan: pip install imagehash Pillow")

# ============================================================
# KONFIGURASI DATABASE
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_DIR = os.path.join(CAPTURE_DIR, "plates")
FACE_DIR = os.path.join(CAPTURE_DIR, "faces")
VEHICLE_DIR = os.path.join(CAPTURE_DIR, "vehicles")

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(VEHICLE_DIR, exist_ok=True)


def get_db():
    return pymysql.connect(**DB_CONFIG)


# ============================================================
# KONFIGURASI DEDUP
# ============================================================

# Window untuk cek duplikat di DB (30 menit).
HASH_WINDOW_SECONDS = 1800

# Hamming distance maksimum pHash (dari 4 -> 6, sedikit lebih toleran
# untuk crop kecil / cahaya berubah).
HASH_MAX_DISTANCE = 6

# Window dedup untuk query web (30 menit, sebelumnya 5 menit).
DEDUP_WINDOW_SECONDS = 1800


def _compute_phash(crop_bgr):
    if not HAS_PHASH:
        return None
    if crop_bgr is None or not hasattr(crop_bgr, "size") or crop_bgr.size == 0:
        return None
    try:
        h, w = crop_bgr.shape[:2]
        if w < 60 or h < 40:
            return None
        img = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return str(imagehash.phash(pil))
    except Exception as exc:
        print(f"[PHASH ERROR] {exc}")
        return None


def _hash_distance(h1, h2):
    if not h1 or not h2 or not HAS_PHASH:
        return None
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception:
        return None


def _find_visual_duplicate(cur, camera_id, phash, object_type, window_seconds=None):
    """Cari detection_id yang punya hash mirip dalam window."""
    if not phash or not HAS_PHASH:
        return None

    window = int(window_seconds or HASH_WINDOW_SECONDS)
    hash_col = "vehicle_hash" if object_type == "vehicle" else "face_hash"

    cur.execute(f"""
        SELECT detection_id, {hash_col} AS other_hash
        FROM full_detection
        WHERE camera_id = %s
          AND object_type = %s
          AND created_at >= DATE_SUB(NOW(), INTERVAL %s SECOND)
          AND {hash_col} IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 200
    """, (camera_id, object_type, window))

    for row in cur.fetchall():
        dist = _hash_distance(phash, row["other_hash"])
        if dist is not None and dist <= HASH_MAX_DISTANCE:
            return row["detection_id"], dist
    return None


def _find_plate_duplicate_db(cur, camera_id, plate_number, window_seconds=None):
    """Cari detection_id dengan plat nomor sama di kamera sama dalam window."""
    if not plate_number:
        return None
    window = int(window_seconds or HASH_WINDOW_SECONDS)
    try:
        cur.execute("""
            SELECT fd.detection_id, p.plate_id, p.plate_number, fd.created_at
            FROM full_detection fd
            JOIN plate p ON p.plate_id = fd.plate_id
            WHERE fd.camera_id = %s
              AND p.plate_number = %s
              AND fd.created_at >= DATE_SUB(NOW(), INTERVAL %s SECOND)
            ORDER BY fd.created_at DESC
            LIMIT 1
        """, (camera_id, plate_number, window))
        return cur.fetchone()
    except Exception as exc:
        print(f"[DEDUP PLAT DB ERROR] {exc}")
        return None


def ensure_event_schema():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                columns = {
                    "cameras": [("direction", "ENUM('entry','exit','unknown') NOT NULL DEFAULT 'unknown'")],
                    "full_detection": [
                        ("track_id", "BIGINT NULL"), ("object_type", "ENUM('vehicle','person') NOT NULL DEFAULT 'person'"),
                        ("vehicle_type", "ENUM('car','motorcycle','truck','bus','unknown') NOT NULL DEFAULT 'unknown'"),
                        ("has_plate", "TINYINT(1) NOT NULL DEFAULT 0"), ("has_driver", "TINYINT(1) NOT NULL DEFAULT 0"),
                        ("vehicle_confidence", "DECIMAL(6,5) NULL"), ("plate_detection_confidence", "DECIMAL(6,5) NULL"),
                        ("ocr_confidence", "DECIMAL(6,5) NULL"), ("person_confidence", "DECIMAL(6,5) NULL"),
                        ("face_confidence", "DECIMAL(6,5) NULL"), ("direction", "ENUM('entry','exit','unknown') NOT NULL DEFAULT 'unknown'"),
                        ("driver_track_id", "BIGINT NULL"), ("driver_face_path", "VARCHAR(500) NULL"),
                        ("vehicle_image_path", "VARCHAR(500) NULL"), ("event_key", "VARCHAR(160) NULL"),
                        ("vehicle_hash", "VARCHAR(20) NULL"), ("face_hash", "VARCHAR(20) NULL")
                    ],
                    "plate": [("raw_ocr_text", "VARCHAR(100) NULL"), ("normalized_plate_number", "VARCHAR(30) NULL")]
                }
                for table, table_columns in columns.items():
                    for column, definition in table_columns:
                        cur.execute("""SELECT COUNT(*) AS count FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
                                    (DB_CONFIG["database"], table, column))
                        if not cur.fetchone()["count"]:
                            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
                indexes = [
                    ("cameras", "idx_cameras_direction", "(`direction`)"),
                    ("full_detection", "idx_detection_event_type", "(`object_type`, `created_at`)"),
                    ("full_detection", "idx_detection_track", "(`camera_id`, `track_id`, `direction`, `created_at`)"),
                    ("full_detection", "idx_detection_event_key", "(`event_key`)"),
                    ("full_detection", "idx_detection_vehicle_hash", "(`vehicle_hash`)"),
                    ("full_detection", "idx_detection_face_hash", "(`face_hash`)")
                ]
                for table, index, definition in indexes:
                    cur.execute("""SELECT COUNT(*) AS count FROM INFORMATION_SCHEMA.STATISTICS
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s""",
                                (DB_CONFIG["database"], table, index))
                    if not cur.fetchone()["count"]:
                        cur.execute(f"ALTER TABLE `{table}` ADD INDEX `{index}` {definition}")
                cur.execute("""
                    UPDATE cameras SET direction = CASE
                        WHEN LOWER(COALESCE(location, '')) LIKE '%%masuk%%' THEN 'entry'
                        WHEN LOWER(COALESCE(location, '')) LIKE '%%keluar%%' THEN 'exit'
                        ELSE 'unknown' END
                    WHERE direction = 'unknown' OR direction IS NULL
                """)
                cur.execute("""
                    UPDATE full_detection
                    SET object_type = CASE WHEN plate_id IS NOT NULL THEN 'vehicle' ELSE 'person' END,
                        has_plate = CASE WHEN plate_id IS NOT NULL THEN 1 ELSE 0 END,
                        vehicle_confidence = CASE WHEN plate_id IS NOT NULL THEN detection_confidence ELSE NULL END,
                        plate_detection_confidence = CASE WHEN plate_id IS NOT NULL THEN detection_confidence ELSE NULL END,
                        ocr_confidence = CASE WHEN plate_id IS NOT NULL THEN (SELECT p.ocr_confidence FROM plate p WHERE p.plate_id = full_detection.plate_id) ELSE NULL END
                    WHERE object_type = 'person' AND (plate_id IS NOT NULL OR face_image_path IS NULL)
                """)
    except Exception as exc:
        print(f"[DB WARNING] event schema migration unavailable: {exc}")


def get_camera_direction(camera_id: int) -> str:
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT direction, location FROM cameras WHERE camera_id = %s", (camera_id,))
                row = cur.fetchone() or {}
        direction = row.get("direction")
        if direction in ("entry", "exit"):
            return direction
        location = str(row.get("location") or "").lower()
        if "masuk" in location:
            return "entry"
        if "keluar" in location:
            return "exit"
    except Exception:
        pass
    return "unknown"


PLATE_REGEX = re.compile(r"^[A-Z]{1,2}\s?[0-9]{1,4}\s?[A-Z]{1,3}$")

def compute_plate_status(plate_number: str, ocr_conf: float) -> int:
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
    conf = float(face_conf or 0.0)
    if conf >= 0.65:
        return 1
    elif conf >= 0.40:
        return 2
    else:
        return 0


def normalize_plate_number(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().upper())


def save_crop_locally(image, folder_path, prefix="cap", camera_id=1, track_id=None) -> str:
    if image is None or image.size == 0:
        return None
    os.makedirs(folder_path, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    tid_str = f"_{track_id}" if track_id is not None else ""
    filename = f"{prefix}_cam{camera_id}_{now_str}{tid_str}.jpg"
    abs_path = os.path.join(folder_path, filename)
    try:
        written = cv2.imwrite(abs_path, image)
    except Exception as exc:
        print(f"[CAPTURE ERROR] {abs_path}: {exc}")
        return None
    if not written or not os.path.isfile(abs_path):
        print(f"[CAPTURE ERROR] cv2.imwrite failed: {abs_path}")
        return None
    rel_path = os.path.relpath(abs_path, BASE_DIR).replace("\\", "/")
    return rel_path


def get_all_cameras():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT camera_id, location, stream_url, stream_type, status, direction FROM cameras ORDER BY camera_id ASC;")
            return cur.fetchall()


def get_active_cameras():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT camera_id, location, stream_url, stream_type, status, direction FROM cameras WHERE status = 1 ORDER BY camera_id ASC;")
            return cur.fetchall()


def update_camera_status(camera_id: int, status: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE cameras SET status = %s WHERE camera_id = %s;", (status, camera_id))


def add_camera(location: str, stream_url: str, stream_type: int = 1, status: int = 1, direction: str = "unknown"):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cameras (location, stream_url, stream_type, status, direction) VALUES (%s, %s, %s, %s, %s);",
                (location, stream_url, stream_type, status, direction if direction in ("entry", "exit", "unknown") else "unknown")
            )
            return cur.lastrowid


def update_camera(camera_id: int, location: str = None, stream_url: str = None, status: int = None, direction: str = None):
    fields = []; params = []
    if location is not None: fields.append("location = %s"); params.append(location)
    if stream_url is not None: fields.append("stream_url = %s"); params.append(stream_url)
    if status is not None: fields.append("status = %s"); params.append(status)
    if direction in ("entry", "exit", "unknown"): fields.append("direction = %s"); params.append(direction)
    if not fields:
        return False
    params.append(camera_id)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE cameras SET {', '.join(fields)} WHERE camera_id = %s;", params)
            return cur.rowcount > 0


def delete_camera(camera_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM cameras WHERE camera_id = %s;", (camera_id,))
            return cur.rowcount > 0


def _delete_capture_file(path):
    if not path:
        return
    absolute_path = os.path.join(BASE_DIR, str(path).replace("/", os.sep))
    try:
        if os.path.isfile(absolute_path):
            os.remove(absolute_path)
    except OSError:
        pass


def delete_detection(detection_id: int) -> bool:
    with get_db() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT fd.plate_id, fd.face_image_path, fd.vehicle_image_path, p.plate_image_path
                    FROM full_detection fd
                    LEFT JOIN plate p ON p.plate_id = fd.plate_id
                    WHERE fd.detection_id = %s
                """, (detection_id,))
                event = cur.fetchone()
                if not event:
                    return False
                cur.execute("DELETE FROM full_detection WHERE detection_id = %s", (detection_id,))
                if event["plate_id"] is not None:
                    cur.execute("DELETE FROM plate_logs WHERE plate_id = %s", (event["plate_id"],))
                    cur.execute("DELETE FROM suspicious_plates WHERE plate_id = %s", (event["plate_id"],))
                    cur.execute("DELETE FROM plate WHERE plate_id = %s", (event["plate_id"],))
                conn.commit()
        except Exception:
            conn.rollback(); raise
    _delete_capture_file(event.get("face_image_path"))
    _delete_capture_file(event.get("vehicle_image_path"))
    _delete_capture_file(event.get("plate_image_path"))
    return True


def delete_plate(plate_id: int) -> bool:
    with get_db() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT plate_image_path FROM plate WHERE plate_id = %s", (plate_id,))
                plate = cur.fetchone()
                if not plate:
                    return False
                cur.execute("""SELECT face_image_path, vehicle_image_path FROM full_detection WHERE plate_id = %s""", (plate_id,))
                face_paths = cur.fetchall()
                cur.execute("DELETE FROM full_detection WHERE plate_id = %s", (plate_id,))
                cur.execute("DELETE FROM plate_logs WHERE plate_id = %s", (plate_id,))
                cur.execute("DELETE FROM suspicious_plates WHERE plate_id = %s", (plate_id,))
                cur.execute("DELETE FROM plate WHERE plate_id = %s", (plate_id,))
                conn.commit()
        except Exception:
            conn.rollback(); raise
    _delete_capture_file(plate.get("plate_image_path"))
    for face in face_paths:
        _delete_capture_file(face.get("face_image_path"))
        _delete_capture_file(face.get("vehicle_image_path"))
    return True


# ============================================================
# FUNGSI INSERT DETEKSI — DENGAN 3 LAPIS DEDUP
# ============================================================

def save_detection_event(
    camera_id: int,
    plate_number: str = None,
    plate_crop=None,
    plate_conf: float = 0.0,
    ocr_conf: float = 0.0,
    face_crop=None,
    face_conf: float = 0.0,
    track_id: int = None,
    object_type: str = "vehicle",
    vehicle_type: str = "unknown",
    vehicle_confidence: float = 0.0,
    has_driver: bool = False,
    driver_track_id: int = None,
    direction: str = "unknown",
    event_key: str = None,
    raw_ocr_text: str = None,
    vehicle_crop=None
) -> dict:
    """
    Simpan satu event deteksi dengan 3 lapis anti-duplikat:

      Lapis 1: event_key (track_id + generation) — di stream_ai
      Lapis 2: plat nomor sama dalam HASH_WINDOW_SECONDS (30 menit)
      Lapis 3: pHash crop mirip dalam HASH_WINDOW_SECONDS (30 menit)

    Kalau salah satu lapis mendeteksi duplikat, event baru TIDAK di-INSERT.
    """
    object_type = object_type if object_type in ("vehicle", "person") else "vehicle"
    vehicle_type = (
        vehicle_type
        if vehicle_type in ("car", "motorcycle", "truck", "bus", "unknown")
        else "unknown"
    )
    direction = direction if direction in ("entry", "exit", "unknown") else "unknown"

    normalized_plate = normalize_plate_number(plate_number)
    has_plate_input = bool(plate_crop is not None or normalized_plate)

    vehicle_confidence = float(vehicle_confidence or 0.0)
    plate_conf = float(plate_conf or 0.0)
    ocr_conf = float(ocr_conf or 0.0)
    face_conf = float(face_conf or 0.0)

    conn = get_db()

    vehicle_rel_path = None
    plate_rel_path = None
    face_rel_path = None

    try:
        with conn.cursor() as cur:

            # ======================================================
            # 1. CEK EVENT EXISTING (dedup by event_key)
            # ======================================================
            existing = None
            if event_key:
                cur.execute(
                    """
                    SELECT
                        fd.detection_id, fd.plate_id, fd.object_type, fd.vehicle_type,
                        fd.has_plate, fd.has_driver, fd.vehicle_confidence,
                        fd.plate_detection_confidence,
                        fd.ocr_confidence AS event_ocr_confidence,
                        fd.person_confidence, fd.face_confidence, fd.direction,
                        fd.driver_track_id, fd.driver_face_path, fd.face_image_path,
                        fd.vehicle_image_path, fd.detection_status,
                        fd.detection_confidence, fd.created_at,
                        p.plate_number, p.raw_ocr_text, p.normalized_plate_number,
                        p.detection_status AS plate_status,
                        p.detection_confidence AS plate_confidence,
                        p.ocr_confidence AS plate_ocr_confidence,
                        p.plate_image_path
                    FROM full_detection fd
                    LEFT JOIN plate p ON p.plate_id = fd.plate_id
                    WHERE fd.event_key = %s
                    ORDER BY fd.detection_id DESC
                    LIMIT 1
                    """,
                    (event_key,)
                )
                existing = cur.fetchone()

            # ======================================================
            # 2. DUPLICATE / ENRICHMENT (by event_key)
            # ======================================================
            if existing:
                existing_id = existing["detection_id"]
                existing_plate_id = existing.get("plate_id")
                existing_vehicle_path = existing.get("vehicle_image_path")
                existing_face_path = existing.get("face_image_path")

                if object_type == "vehicle":
                    if vehicle_crop is not None and not existing_vehicle_path:
                        vehicle_rel_path = save_crop_locally(
                            vehicle_crop, VEHICLE_DIR,
                            prefix="vehicle", camera_id=camera_id, track_id=track_id
                        )

                    new_plate_id = None
                    if has_plate_input and not existing_plate_id:
                        if plate_crop is not None:
                            plate_rel_path = save_crop_locally(
                                plate_crop, PLATE_DIR,
                                prefix="plate", camera_id=camera_id, track_id=track_id
                            )
                        plate_status = compute_plate_status(normalized_plate, ocr_conf)
                        cur.execute(
                            """INSERT INTO plate (plate_number, raw_ocr_text, normalized_plate_number,
                                detection_status, detection_confidence, ocr_confidence, plate_image_path)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                            (normalized_plate or None, raw_ocr_text or plate_number,
                             normalized_plate or None, plate_status, plate_conf, ocr_conf, plate_rel_path)
                        )
                        new_plate_id = cur.lastrowid
                        cur.execute(
                            """INSERT INTO plate_logs (plate_id, camera_id, status) VALUES (%s, %s, %s)""",
                            (new_plate_id, camera_id, plate_status)
                        )
                    elif has_plate_input and existing_plate_id:
                        old_ocr_conf = float(existing.get("plate_ocr_confidence") or 0.0)
                        should_update_plate = (
                            normalized_plate and (
                                not existing.get("plate_number") or ocr_conf > old_ocr_conf
                            )
                        )
                        if should_update_plate:
                            plate_status = compute_plate_status(normalized_plate, ocr_conf)
                            cur.execute(
                                """UPDATE plate SET plate_number=%s, raw_ocr_text=%s,
                                    normalized_plate_number=%s, detection_status=%s,
                                    detection_confidence=%s, ocr_confidence=%s
                                   WHERE plate_id=%s""",
                                (normalized_plate, raw_ocr_text or plate_number,
                                 normalized_plate, plate_status, plate_conf, ocr_conf, existing_plate_id)
                            )

                    final_plate_id = new_plate_id or existing_plate_id
                    should_update_event = (
                        bool(vehicle_rel_path) or bool(new_plate_id)
                        or (has_driver and not bool(existing.get("has_driver")))
                        or (vehicle_confidence > float(existing.get("vehicle_confidence") or 0.0))
                        or (has_plate_input and ocr_conf > float(existing.get("event_ocr_confidence") or 0.0))
                    )

                    if should_update_event:
                        final_has_plate = bool(final_plate_id)
                        if final_has_plate:
                            final_plate_conf = plate_conf
                            final_ocr_conf = ocr_conf
                        else:
                            final_plate_conf = None
                            final_ocr_conf = None

                        if final_has_plate:
                            status_val = compute_plate_status(
                                normalized_plate if normalized_plate else existing.get("plate_number"),
                                final_ocr_conf or existing.get("event_ocr_confidence") or 0.0
                            )
                            conf_val = final_ocr_conf or existing.get("event_ocr_confidence") or 0.0
                        else:
                            conf_val = max(vehicle_confidence, float(existing.get("vehicle_confidence") or 0.0))
                            status_val = 1 if conf_val >= 0.70 else 2 if conf_val >= 0.40 else 0

                        cur.execute(
                            """UPDATE full_detection SET object_type='vehicle', vehicle_type=%s,
                                has_plate=%s, has_driver=%s, detection_status=%s, detection_confidence=%s,
                                vehicle_confidence=%s, plate_detection_confidence=%s, ocr_confidence=%s,
                                direction=%s, driver_track_id=%s,
                                vehicle_image_path = COALESCE(vehicle_image_path, %s)
                               WHERE detection_id=%s""",
                            (vehicle_type, int(final_has_plate),
                             int(bool(has_driver or existing.get("has_driver"))),
                             status_val, conf_val,
                             max(vehicle_confidence, float(existing.get("vehicle_confidence") or 0.0)),
                             (max(plate_conf, float(existing.get("plate_detection_confidence") or 0.0)) if final_has_plate else None),
                             (max(ocr_conf, float(existing.get("event_ocr_confidence") or 0.0)) if final_has_plate else None),
                             direction, driver_track_id or existing.get("driver_track_id"),
                             vehicle_rel_path, existing_id)
                        )

                    if vehicle_rel_path and existing_vehicle_path:
                        _delete_capture_file(vehicle_rel_path)
                        vehicle_rel_path = None

                    return {
                        "plate_id": final_plate_id, "detection_id": existing_id,
                        "duplicate": True, "enriched": bool(should_update_event),
                        "plate_image_path": plate_rel_path or existing.get("plate_image_path"),
                        "face_image_path": existing_face_path,
                        "vehicle_image_path": existing_vehicle_path or vehicle_rel_path,
                        "plate_number": normalized_plate or existing.get("plate_number") or None,
                        "object_type": "vehicle", "vehicle_type": vehicle_type,
                        "has_plate": bool(final_plate_id),
                        "has_driver": bool(has_driver or existing.get("has_driver"))
                    }

                if object_type == "person":
                    if face_crop is not None and not existing_face_path:
                        face_rel_path = save_crop_locally(
                            face_crop, FACE_DIR,
                            prefix="face", camera_id=camera_id, track_id=track_id
                        )
                    if face_rel_path:
                        cur.execute(
                            """UPDATE full_detection SET object_type='person',
                                person_confidence=%s, face_confidence=%s,
                                detection_confidence=%s, detection_status=%s,
                                direction=%s, face_image_path=%s
                               WHERE detection_id=%s""",
                            (max(face_conf, float(existing.get("person_confidence") or 0.0)),
                             max(face_conf, float(existing.get("face_confidence") or 0.0)),
                             max(face_conf, float(existing.get("detection_confidence") or 0.0)),
                             compute_face_status(max(face_conf, float(existing.get("face_confidence") or 0.0))),
                             direction, face_rel_path, existing_id)
                        )
                    else:
                        face_rel_path = None

                    return {
                        "plate_id": None, "detection_id": existing_id,
                        "duplicate": True, "enriched": bool(face_rel_path),
                        "plate_image_path": None,
                        "face_image_path": face_rel_path or existing_face_path,
                        "vehicle_image_path": None, "plate_number": None,
                        "object_type": "person", "vehicle_type": "unknown",
                        "has_plate": False, "has_driver": False
                    }

            # ======================================================
            # 3. EVENT BARU — DEDUP LAPIS 2: PLAT NOMOR SAMA DI DB
            # ======================================================
            if object_type == "vehicle" and normalized_plate:
                plate_dup = _find_plate_duplicate_db(cur, camera_id, normalized_plate)
                if plate_dup is not None:
                    dup_id = plate_dup["detection_id"]
                    print(f"[DEDUP PLAT DB] Skip: plat {normalized_plate} sudah ada "
                          f"di detection_id={dup_id} (dalam 30 menit)")
                    return {
                        "plate_id": plate_dup.get("plate_id"),
                        "detection_id": dup_id,
                        "duplicate": True, "enriched": False,
                        "reason": "plate_duplicate_db",
                        "plate_number": normalized_plate,
                        "object_type": "vehicle", "vehicle_type": vehicle_type,
                        "has_plate": True, "has_driver": False,
                        "plate_image_path": None, "face_image_path": None,
                        "vehicle_image_path": None,
                    }

            # ======================================================
            # 4. EVENT BARU — DEDUP LAPIS 3: pHASH VISUAL MIRIP
            # ======================================================
            vehicle_hash = _compute_phash(vehicle_crop) if object_type == "vehicle" and vehicle_crop is not None else None
            face_hash = _compute_phash(face_crop) if object_type == "person" and face_crop is not None else None
            candidate_hash = vehicle_hash or face_hash

            if candidate_hash:
                dup = _find_visual_duplicate(cur, camera_id, candidate_hash, object_type)
                if dup is not None:
                    dup_id, distance = dup
                    print(f"[DEDUP VISUAL DB] Skip: mirip detection_id={dup_id} "
                          f"(distance={distance}, type={object_type})")
                    return {
                        "plate_id": None, "detection_id": dup_id,
                        "duplicate": True, "enriched": False,
                        "reason": "visual_duplicate_db", "distance": distance,
                        "plate_image_path": None, "face_image_path": None,
                        "vehicle_image_path": None,
                        "plate_number": normalized_plate or None,
                        "object_type": object_type, "vehicle_type": vehicle_type,
                        "has_plate": bool(normalized_plate), "has_driver": False
                    }

            # ======================================================
            # 5. LANJUT INSERT — tidak ada duplikat
            # ======================================================
            plate_id = None
            plate_rel_path = None
            face_rel_path = None
            vehicle_rel_path = None

            if object_type == "vehicle" and vehicle_crop is not None:
                vehicle_rel_path = save_crop_locally(
                    vehicle_crop, VEHICLE_DIR,
                    prefix="vehicle", camera_id=camera_id, track_id=track_id
                )

            plate_status = 0
            if object_type == "vehicle" and has_plate_input:
                if plate_crop is not None:
                    plate_rel_path = save_crop_locally(
                        plate_crop, PLATE_DIR,
                        prefix="plate", camera_id=camera_id, track_id=track_id
                    )
                plate_status = compute_plate_status(normalized_plate, ocr_conf)
                cur.execute(
                    """INSERT INTO plate (plate_number, raw_ocr_text, normalized_plate_number,
                        detection_status, detection_confidence, ocr_confidence, plate_image_path)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (normalized_plate or None, raw_ocr_text or plate_number,
                     normalized_plate or None, plate_status, plate_conf, ocr_conf, plate_rel_path)
                )
                plate_id = cur.lastrowid
                cur.execute(
                    """INSERT INTO plate_logs (plate_id, camera_id, status) VALUES (%s, %s, %s)""",
                    (plate_id, camera_id, plate_status)
                )

            if object_type == "person" and face_crop is not None:
                face_rel_path = save_crop_locally(
                    face_crop, FACE_DIR,
                    prefix="face", camera_id=camera_id, track_id=track_id
                )

            if object_type == "person":
                status_val = compute_face_status(face_conf)
                conf_val = face_conf
            else:
                if plate_id:
                    status_val = plate_status
                    conf_val = ocr_conf
                else:
                    status_val = 1 if vehicle_confidence >= 0.70 else 2 if vehicle_confidence >= 0.40 else 0
                    conf_val = vehicle_confidence

            cur.execute(
                """INSERT INTO full_detection (
                    plate_id, camera_id, track_id, object_type, vehicle_type,
                    has_plate, has_driver, detection_status, detection_confidence,
                    vehicle_confidence, plate_detection_confidence, ocr_confidence,
                    person_confidence, face_confidence, direction, driver_track_id,
                    face_image_path, vehicle_image_path, event_key,
                    vehicle_hash, face_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (plate_id, camera_id, track_id, object_type, vehicle_type,
                 int(has_plate_input) if object_type == "vehicle" else 0,
                 int(bool(has_driver)) if object_type == "vehicle" else 0,
                 status_val, conf_val,
                 vehicle_confidence if object_type == "vehicle" else None,
                 plate_conf if object_type == "vehicle" and plate_id else None,
                 ocr_conf if object_type == "vehicle" and plate_id else None,
                 face_conf if object_type == "person" else None,
                 face_conf if object_type == "person" else None,
                 direction,
                 driver_track_id if object_type == "vehicle" else None,
                 face_rel_path if object_type == "person" else None,
                 vehicle_rel_path if object_type == "vehicle" else None,
                 event_key,
                 vehicle_hash, face_hash)
            )

            detection_id = cur.lastrowid

            print(
                f"[DB EVENT] detection_id={detection_id} camera={camera_id} type={object_type} "
                f"track_id={track_id} vehicle_image={vehicle_rel_path or '-'} "
                f"plate_image={plate_rel_path or '-'} face_image={face_rel_path or '-'} "
                f"v_hash={vehicle_hash or '-'} f_hash={face_hash or '-'} event_key={event_key or '-'}"
            )

            return {
                "plate_id": plate_id, "detection_id": detection_id,
                "plate_image_path": plate_rel_path, "face_image_path": face_rel_path,
                "vehicle_image_path": vehicle_rel_path,
                "plate_number": normalized_plate or None,
                "object_type": object_type, "vehicle_type": vehicle_type,
                "has_plate": (has_plate_input if object_type == "vehicle" else False),
                "has_driver": (bool(has_driver) if object_type == "vehicle" else False),
                "duplicate": False, "enriched": False
            }

    except Exception as exc:
        if vehicle_rel_path: _delete_capture_file(vehicle_rel_path)
        if plate_rel_path: _delete_capture_file(plate_rel_path)
        if face_rel_path: _delete_capture_file(face_rel_path)
        try: conn.rollback()
        except Exception: pass
        print(f"[DB SAVE ERROR] camera={camera_id} track_id={track_id} type={object_type}: {exc}")
        raise
    finally:
        conn.close()


# ============================================================
# FUNGSI QUERY WEB & DASHBOARD — DEDUP DIPERBAIKI
# ============================================================

def _dedup_event_ids(cur, where_sql, params):
    """
    Kembalikan list detection_id yang sudah di-dedup (urut terbaru).

    Kunci dedup:
      - Kalau ada plat nomor -> (camera_id, plate_number, bucket_waktu)
      - Kalau tanpa plat     -> (camera_id, vehicle_hash/face_hash, bucket_waktu)
      - Fallback             -> (camera_id, track_id, bucket_waktu)

    Ini memperbaiki masalah sebelumnya di mana dedup jatuh ke track_id
    yang berubah-ubah.
    """
    sql = f"""
        SELECT
            fd.detection_id, fd.camera_id, fd.track_id, fd.object_type,
            fd.vehicle_hash, fd.face_hash,
            fd.created_at, fd.detection_confidence,
            p.plate_number
        FROM full_detection fd
        LEFT JOIN plate p ON p.plate_id = fd.plate_id
        LEFT JOIN cameras c ON c.camera_id = fd.camera_id
        WHERE {where_sql}
        ORDER BY fd.created_at DESC
    """
    cur.execute(sql, params)
    rows = cur.fetchall()
    buckets = {}
    for r in rows:
        cam = r.get("camera_id")
        # Prioritas kunci dedup: plat -> hash -> track
        if r.get("plate_number"):
            dedup_key = f"plate:{r['plate_number']}"
        elif r.get("object_type") == "vehicle" and r.get("vehicle_hash"):
            dedup_key = f"vhash:{r['vehicle_hash']}"
        elif r.get("object_type") == "person" and r.get("face_hash"):
            dedup_key = f"fhash:{r['face_hash']}"
        else:
            dedup_key = f"track:{r['track_id']}"

        ts = r.get("created_at")
        epoch = int(ts.timestamp()) if isinstance(ts, datetime) else 0
        bucket = epoch // DEDUP_WINDOW_SECONDS if epoch else 0
        key = (cam, dedup_key, bucket)

        conf = float(r.get("detection_confidence") or 0.0)
        best = buckets.get(key)
        if best is None or conf > best["conf"]:
            buckets[key] = {"detection_id": r["detection_id"], "conf": conf, "created_at": ts}

    selected = sorted(buckets.values(), key=lambda x: (x["created_at"] or datetime.min), reverse=True)
    return [row["detection_id"] for row in selected]


def _dedup_plate_ids(cur, where_sql, params):
    """Kembalikan list plate_id yang sudah di-dedup (urut terbaru)."""
    sql = f"""
        SELECT p.plate_id, p.plate_number, p.created_at,
               p.ocr_confidence, p.detection_confidence, pl.camera_id
        FROM plate p
        LEFT JOIN plate_logs pl ON p.plate_id = pl.plate_id
        LEFT JOIN cameras c ON pl.camera_id = c.camera_id
        WHERE {where_sql}
        ORDER BY p.created_at DESC
    """
    cur.execute(sql, params)
    rows = cur.fetchall()
    buckets = {}
    for r in rows:
        plate_key = r.get("plate_number") or f"none:{r['plate_id']}"
        cam = r.get("camera_id")
        ts = r.get("created_at")
        epoch = int(ts.timestamp()) if isinstance(ts, datetime) else 0
        bucket = epoch // DEDUP_WINDOW_SECONDS if epoch else 0
        key = (cam, plate_key, bucket)
        conf = float(r.get("ocr_confidence") or r.get("detection_confidence") or 0.0)
        best = buckets.get(key)
        if best is None or conf > best["conf"]:
            buckets[key] = {"plate_id": r["plate_id"], "conf": conf, "created_at": ts}
    selected = sorted(buckets.values(), key=lambda x: (x["created_at"] or datetime.min), reverse=True)
    return [row["plate_id"] for row in selected]


def get_recent_detections(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    with get_db() as conn:
        with conn.cursor() as cur:
            where_sql = "1=1"
            params = []
            all_ids = _dedup_event_ids(cur, where_sql, params)
            selected = all_ids[:limit]
            if not selected:
                return []
            placeholders = ",".join(["%s"] * len(selected))
            query = f"""
                SELECT
                    fd.detection_id, fd.plate_id, fd.camera_id, fd.track_id,
                    fd.object_type, fd.vehicle_type, fd.has_plate, fd.has_driver,
                    fd.vehicle_confidence, fd.plate_detection_confidence,
                    fd.ocr_confidence AS event_ocr_confidence,
                    fd.person_confidence,
                    COALESCE(fd.face_confidence, fd.person_confidence) AS face_confidence,
                    fd.direction, fd.detection_status AS face_status,
                    fd.detection_confidence AS legacy_detection_confidence,
                    fd.face_image_path, fd.vehicle_image_path,
                    fd.created_at AS detected_at,
                    p.plate_number, p.detection_status AS plate_status,
                    p.detection_confidence AS plate_confidence,
                    p.ocr_confidence, p.plate_image_path,
                    COALESCE(c1.location, 'CCTV') AS camera_name
                FROM full_detection fd
                LEFT JOIN plate p ON fd.plate_id = p.plate_id
                LEFT JOIN cameras c1 ON fd.camera_id = c1.camera_id
                WHERE fd.detection_id IN ({placeholders})
                ORDER BY fd.created_at DESC
            """
            cur.execute(query, selected)
            return cur.fetchall()


def get_dashboard_stats():
    summary = get_analytics({"period": "today"})["summary"]
    return {
        "total_plates": summary["plates"],
        "unique_plates": summary["unique_plates"],
        "total_full_detections": summary["vehicles"] + summary["people"],
        "today_detections": summary["vehicles"],
        "active_cameras": summary["active_cameras"],
        "total_cameras": summary["total_cameras"],
        "plate_success": summary["plates"],
        "need_check": 0,
        **summary
    }


def _analytics_filters(args=None):
    args = args or {}
    period = str(args.get("period") or "today").lower()
    start_date = args.get("start_date")
    end_date = args.get("end_date")

    has_custom_dates = bool(start_date or end_date)
    if period == "today" and not has_custom_dates:
        start_date = end_date = datetime.now().strftime("%Y-%m-%d")
    elif period == "7d" and not has_custom_dates:
        start_date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
    elif period == "30d" and not has_custom_dates:
        start_date = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
    elif period == "all" and not has_custom_dates:
        start_date = end_date = None
    elif period not in ("today", "7d", "30d", "all"):
        period = "custom"

    clauses = ["1=1"]; params = []
    if start_date: clauses.append("DATE(fd.created_at) >= %s"); params.append(str(start_date))
    if end_date: clauses.append("DATE(fd.created_at) <= %s"); params.append(str(end_date))
    if str(args.get("camera_id") or "").isdigit():
        clauses.append("fd.camera_id = %s"); params.append(int(args["camera_id"]))
    if args.get("region"): clauses.append("c.location LIKE %s"); params.append(f"%{str(args['region']).strip()}%")
    if args.get("gate"): clauses.append("c.location LIKE %s"); params.append(f"%{str(args['gate']).strip()}%")
    if args.get("direction") in ("entry", "exit"):
        clauses.append("COALESCE(fd.direction, 'unknown') = %s"); params.append(args["direction"])
    if args.get("object_type") == "vehicle":
        clauses.append("fd.object_type = 'vehicle'")
    elif args.get("object_type") == "person":
        clauses.append("fd.object_type = 'person'")
    return " AND ".join(clauses), params, period, start_date, end_date


def _get_dedup_subquery(where_sql):
    """
    Subquery dedup berbasis plat -> hash -> track (fallback).
    Konsisten dengan _dedup_event_ids.
    """
    return f"""
        SELECT
            fd.detection_id, fd.camera_id, fd.object_type, fd.has_plate,
            fd.plate_id, fd.detection_confidence, fd.created_at, fd.track_id,
            fd.direction, fd.detection_status,
            fd.vehicle_hash, fd.face_hash,
            CASE
                WHEN p.plate_number IS NOT NULL AND p.plate_number != ''
                    THEN CONCAT('plate:', p.plate_number)
                WHEN fd.object_type = 'vehicle' AND fd.vehicle_hash IS NOT NULL
                    THEN CONCAT('vhash:', fd.vehicle_hash)
                WHEN fd.object_type = 'person' AND fd.face_hash IS NOT NULL
                    THEN CONCAT('fhash:', fd.face_hash)
                ELSE CONCAT('track:', COALESCE(fd.track_id, 0))
            END AS dedup_key,
            ROW_NUMBER() OVER (
                PARTITION BY
                    fd.camera_id,
                    CASE
                        WHEN p.plate_number IS NOT NULL AND p.plate_number != ''
                            THEN CONCAT('plate:', p.plate_number)
                        WHEN fd.object_type = 'vehicle' AND fd.vehicle_hash IS NOT NULL
                            THEN CONCAT('vhash:', fd.vehicle_hash)
                        WHEN fd.object_type = 'person' AND fd.face_hash IS NOT NULL
                            THEN CONCAT('fhash:', fd.face_hash)
                        ELSE CONCAT('track:', COALESCE(fd.track_id, 0))
                    END,
                    FLOOR(UNIX_TIMESTAMP(fd.created_at) / {DEDUP_WINDOW_SECONDS})
                ORDER BY fd.detection_confidence DESC, fd.detection_id DESC
            ) AS rn
        FROM full_detection fd
        LEFT JOIN plate p ON p.plate_id = fd.plate_id
        LEFT JOIN cameras c ON c.camera_id = fd.camera_id
        WHERE {where_sql}
    """


def get_analytics(args=None):
    where_sql, params, period, start_date, end_date = _analytics_filters(args)
    dedup_subquery = _get_dedup_subquery(where_sql)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    SUM(d.object_type = 'vehicle') AS vehicles,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'entry') AS vehicle_entry,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'exit') AS vehicle_exit,
                    SUM(d.object_type = 'person') AS people,
                    SUM(d.object_type = 'person' AND d.direction = 'entry') AS people_entry,
                    SUM(d.object_type = 'person' AND d.direction = 'exit') AS people_exit,
                    SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS plates,
                    COUNT(DISTINCT CASE
                        WHEN d.object_type = 'vehicle'
                            AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)
                            AND p.plate_number IS NOT NULL AND p.plate_number != ''
                        THEN p.plate_number END) AS unique_plates
                FROM ({dedup_subquery}) d
                LEFT JOIN plate p ON p.plate_id = d.plate_id
                WHERE d.rn = 1
            """, params)
            row = cur.fetchone() or {}
            cur.execute("SELECT COUNT(*) AS total, SUM(status = 1) AS active FROM cameras")
            cams = cur.fetchone() or {}
            summary = {
                "vehicles": int(row.get("vehicles") or 0), "vehicle_entry": int(row.get("vehicle_entry") or 0),
                "vehicle_exit": int(row.get("vehicle_exit") or 0), "people": int(row.get("people") or 0),
                "people_entry": int(row.get("people_entry") or 0), "people_exit": int(row.get("people_exit") or 0),
                "plates": int(row.get("plates") or 0), "unique_plates": int(row.get("unique_plates") or 0),
                "total_cameras": int(cams.get("total") or 0), "active_cameras": int(cams.get("active") or 0)
            }

            def query_rows(sql):
                cur.execute(sql, params); return cur.fetchall()

            camera_rows = query_rows(f"""
                SELECT d.camera_id,
                    COALESCE(MAX(c.location), CONCAT('CCTV-', d.camera_id)) AS camera_name,
                    SUM(d.object_type = 'vehicle') AS vehicles,
                    SUM(d.object_type = 'person') AS people,
                    COUNT(DISTINCT CASE WHEN d.object_type = 'vehicle'
                        AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)
                        THEN p.plate_number END) AS unique_plates,
                    CASE WHEN MAX(c.direction) = 'entry' THEN 'Masuk'
                         WHEN MAX(c.direction) = 'exit' THEN 'Keluar'
                         ELSE 'Tidak ditentukan' END AS direction,
                    MAX(c.status) AS status
                FROM ({dedup_subquery}) d
                LEFT JOIN plate p ON p.plate_id = d.plate_id
                LEFT JOIN cameras c ON c.camera_id = d.camera_id
                WHERE d.rn = 1
                GROUP BY d.camera_id ORDER BY vehicles DESC, people DESC
            """)
            cameras = [{"camera_id": r.get("camera_id"), "camera": r.get("camera_name"),
                        "vehicles": int(r.get("vehicles") or 0), "people": int(r.get("people") or 0),
                        "unique_plates": int(r.get("unique_plates") or 0), "direction": r.get("direction"),
                        "status": "Aktif" if r.get("status") == 1 else "Offline"} for r in camera_rows]

            daily_rows = query_rows(f"""
                SELECT DATE(d.created_at) AS day,
                    SUM(d.object_type = 'vehicle') AS vehicles,
                    COUNT(DISTINCT CASE WHEN d.object_type = 'vehicle'
                        AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)
                        THEN p.plate_number END) AS unique_plates,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'entry') AS entry_count,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'exit') AS exit_count,
                    SUM(d.object_type = 'person') AS people,
                    SUM(d.object_type = 'person' AND d.direction = 'entry') AS people_entry_count,
                    SUM(d.object_type = 'person' AND d.direction = 'exit') AS people_exit_count
                FROM ({dedup_subquery}) d
                LEFT JOIN plate p ON p.plate_id = d.plate_id
                WHERE d.rn = 1
                GROUP BY DATE(d.created_at) ORDER BY day ASC
            """)
            daily = [{"date": str(r.get("day")), "vehicles": int(r.get("vehicles") or 0),
                      "unique_plates": int(r.get("unique_plates") or 0),
                      "entry": int(r.get("entry_count") or 0), "exit": int(r.get("exit_count") or 0),
                      "people": int(r.get("people") or 0),
                      "people_entry": int(r.get("people_entry_count") or 0),
                      "people_exit": int(r.get("people_exit_count") or 0)} for r in daily_rows]

            hourly_rows = query_rows(f"""
                SELECT HOUR(d.created_at) AS hour,
                    SUM(d.object_type = 'vehicle') AS vehicles,
                    SUM(d.object_type = 'person') AS people,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'entry') AS vehicle_entry,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'exit') AS vehicle_exit,
                    SUM(d.object_type = 'person' AND d.direction = 'entry') AS people_entry,
                    SUM(d.object_type = 'person' AND d.direction = 'exit') AS people_exit
                FROM ({dedup_subquery}) d
                WHERE d.rn = 1
                GROUP BY HOUR(d.created_at) ORDER BY hour ASC
            """)
            hourly_map = {int(r["hour"]): r for r in hourly_rows}
            hourly = [{"hour": h, "label": f"{h:02d}:00",
                       "vehicles": int(hourly_map.get(h, {}).get("vehicles") or 0),
                       "people": int(hourly_map.get(h, {}).get("people") or 0),
                       "vehicle_entry": int(hourly_map.get(h, {}).get("vehicle_entry") or 0),
                       "vehicle_exit": int(hourly_map.get(h, {}).get("vehicle_exit") or 0),
                       "people_entry": int(hourly_map.get(h, {}).get("people_entry") or 0),
                       "people_exit": int(hourly_map.get(h, {}).get("people_exit") or 0)} for h in range(24)]

            top_rows = query_rows(f"""
                SELECT p.plate_number, COUNT(*) AS total_seen, AVG(p.ocr_confidence) AS avg_confidence,
                    MAX(d.created_at) AS last_seen, COALESCE(MAX(c.location), 'CCTV') AS last_camera,
                    MAX(p.detection_status) AS last_status
                FROM ({dedup_subquery}) d
                JOIN plate p ON p.plate_id = d.plate_id
                LEFT JOIN cameras c ON c.camera_id = d.camera_id
                WHERE d.rn = 1 AND p.plate_number IS NOT NULL AND p.plate_number != ''
                GROUP BY p.plate_number ORDER BY total_seen DESC, last_seen DESC LIMIT 10
            """)
            top_plates = []
            for r in top_rows:
                last_code = r.get("last_status")
                last_text = "Tidak tersedia" if last_code is None else ("Terbaca" if last_code == 1 else ("Perlu cek" if last_code == 2 else "Gagal"))
                top_plates.append({"plate": r["plate_number"], "count": int(r["total_seen"] or 0),
                                   "confidence": round(float(r.get("avg_confidence") or 0) * 100, 1),
                                   "last_seen": str(r.get("last_seen") or ""), "camera": r.get("last_camera") or "CCTV",
                                   "status_code": last_code, "status": last_text})

            recent_rows = query_rows(f"""
                SELECT d.detection_id AS id, p.plate_number AS plate, d.created_at AS detected_at,
                    COALESCE(c.location, 'CCTV') AS camera, d.detection_confidence AS confidence,
                    CASE WHEN d.object_type = 'vehicle' THEN 'Kendaraan' ELSE 'Orang' END AS type,
                    CASE WHEN d.direction = 'entry' THEN 'Masuk'
                        WHEN d.direction = 'exit' THEN 'Keluar' ELSE 'Tidak ditentukan' END AS direction
                FROM ({dedup_subquery}) d
                LEFT JOIN plate p ON p.plate_id = d.plate_id
                LEFT JOIN cameras c ON c.camera_id = d.camera_id
                WHERE d.rn = 1
                ORDER BY d.created_at DESC LIMIT 12
            """)
            recent = [{"id": r["id"], "plate": r.get("plate") or "-", "camera": r.get("camera"),
                       "type": r.get("type"), "direction": r.get("direction"),
                       "timestamp": str(r.get("detected_at") or ""),
                       "confidence": round(float(r.get("confidence") or 0) * 100, 1)} for r in recent_rows]

            peak = max(hourly, key=lambda item: item["vehicles"], default={"label": "-", "vehicles": 0})
            busiest = cameras[0]["camera"] if cameras else "Belum ada data"
            top_plate = top_plates[0]["plate"] if top_plates else "Belum ada data"
            insights = [f"{busiest} memiliki aktivitas kendaraan tertinggi.",
                        f"Periode terpilih mencatat {summary['vehicles']:,} kendaraan.".replace(",", "."),
                        f"{top_plate} merupakan plat yang paling sering terdeteksi.",
                        f"Jam {peak['label']} merupakan periode kendaraan tersibuk ({peak['vehicles']} kendaraan)."]
            return {"period": period, "start_date": start_date, "end_date": end_date, "summary": summary,
                    "cameras": cameras, "daily": daily, "hourly": hourly, "top_plates": top_plates,
                    "recent": recent, "insights": insights}


def ensure_tables_exist():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS `system_settings` (
                        `setting_key` VARCHAR(100) NOT NULL PRIMARY KEY,
                        `setting_value` TEXT NULL,
                        `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
                cur.execute("SELECT COUNT(*) AS c FROM system_settings;")
                if cur.fetchone()["c"] == 0:
                    defaults = [
                        ("operator_name", "Administrator"),
                        ("refresh_interval", "2"),
                        ("stream_url", "http://103.255.15.138:1935/live/GSMasukViewLuar.stream/playlist.m3u8"),
                        ("min_confidence_plate", "70"),
                        ("min_confidence_face", "65"),
                        ("company_name", "PT CCTV Security Solusindo"),
                        ("system_title", "PlateVision Enterprise Monitoring")
                    ]
                    cur.executemany(
                        "INSERT INTO system_settings (setting_key, setting_value) VALUES (%s, %s);",
                        defaults
                    )
    except Exception as e:
        print(f"[DB WARNING] ensure_tables_exist error: {e}")

ensure_tables_exist()
ensure_event_schema()


def get_system_settings() -> dict:
    ensure_tables_exist()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT setting_key, setting_value FROM system_settings;")
            rows = cur.fetchall()
            return {r["setting_key"]: r["setting_value"] for r in rows}


def update_system_settings(settings_dict: dict) -> bool:
    ensure_tables_exist()
    with get_db() as conn:
        with conn.cursor() as cur:
            for k, v in settings_dict.items():
                cur.execute(
                    """INSERT INTO system_settings (setting_key, setting_value)
                       VALUES (%s, %s)
                       ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value);""",
                    (str(k), str(v))
                )
    return True


def get_system_diagnostics() -> dict:
    diag = {
        "db_connected": False, "db_name": DB_CONFIG["database"], "db_host": DB_CONFIG["host"],
        "table_counts": {}, "storage": {"total_files": 0, "total_size_mb": 0.0},
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                diag["db_connected"] = True
                for tbl in ["cameras", "full_detection", "plate", "plate_logs", "suspicious_plates", "system_settings"]:
                    try:
                        cur.execute(f"SELECT COUNT(*) AS cnt FROM {tbl};")
                        diag["table_counts"][tbl] = cur.fetchone()["cnt"]
                    except Exception:
                        diag["table_counts"][tbl] = 0
        total_size = 0; total_files = 0
        if os.path.exists(CAPTURE_DIR):
            for root, _, files in os.walk(CAPTURE_DIR):
                for f in files:
                    total_files += 1
                    fp = os.path.join(root, f)
                    try: total_size += os.path.getsize(fp)
                    except Exception: pass
        diag["storage"]["total_files"] = total_files
        diag["storage"]["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    except Exception as e:
        diag["error"] = str(e)
    return diag


def get_all_detections_paginated(
    page: int = 1, limit: int = 20, type_filter: str = "all",
    status_filter: str = "all", camera_id: int = None,
    search: str = None, start_date: str = None, end_date: str = None
) -> dict:
    page = max(1, int(page)); limit = max(1, min(100, int(limit)))
    offset = (page - 1) * limit

    where_clauses = ["1=1"]; params = []

    if type_filter in ("vehicle", "person"):
        where_clauses.append("fd.object_type = %s"); params.append(type_filter)
    elif type_filter == "plate":
        where_clauses.append("fd.object_type = 'vehicle' AND fd.has_plate = 1")
    elif type_filter == "face":
        where_clauses.append("fd.object_type = 'person'")

    if status_filter in ("0", "1", "2"):
        where_clauses.append("fd.detection_status = %s"); params.append(int(status_filter))
    if camera_id is not None and str(camera_id).isdigit():
        where_clauses.append("fd.camera_id = %s"); params.append(int(camera_id))
    if search:
        s = f"%{search.strip()}%"
        where_clauses.append("(p.plate_number LIKE %s OR c.location LIKE %s)"); params.extend([s, s])
    if start_date:
        where_clauses.append("DATE(fd.created_at) >= %s"); params.append(start_date)
    if end_date:
        where_clauses.append("DATE(fd.created_at) <= %s"); params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    with get_db() as conn:
        with conn.cursor() as cur:
            all_ids = _dedup_event_ids(cur, where_sql, params)
            total = len(all_ids)
            if total == 0:
                return {"items": [], "total": 0, "page": page, "limit": limit, "total_pages": 1}

            page_ids = all_ids[offset: offset + limit]
            placeholders = ",".join(["%s"] * len(page_ids))

            data_sql = f"""
                SELECT fd.detection_id, fd.plate_id, fd.camera_id, fd.track_id,
                    fd.object_type, fd.vehicle_type, fd.has_plate, fd.has_driver,
                    fd.detection_status AS status_code, fd.detection_confidence,
                    fd.vehicle_confidence, fd.plate_detection_confidence,
                    fd.ocr_confidence AS event_ocr_confidence,
                    fd.person_confidence, fd.face_confidence, fd.direction,
                    fd.driver_track_id, fd.driver_face_path, fd.event_key,
                    fd.face_image_path, fd.vehicle_image_path,
                    fd.created_at AS detected_at,
                    p.plate_number, p.detection_status AS plate_status,
                    p.detection_confidence AS plate_confidence,
                    p.ocr_confidence, p.plate_image_path,
                    COALESCE(c.location, 'CCTV') AS camera_name
                FROM full_detection fd
                LEFT JOIN plate p ON fd.plate_id = p.plate_id
                LEFT JOIN cameras c ON fd.camera_id = c.camera_id
                WHERE fd.detection_id IN ({placeholders})
                ORDER BY fd.created_at DESC
            """
            cur.execute(data_sql, page_ids)
            rows = cur.fetchall()

            items = []
            for r in rows:
                p_num = r.get("plate_number")
                object_type = r.get("object_type") or ("vehicle" if r.get("plate_id") else "person")
                has_plate = bool(r.get("has_plate") or p_num or r.get("plate_image_path"))
                dtype = "vehicle_with_plate" if object_type == "vehicle" and has_plate else object_type
                code = r.get("plate_status") if object_type == "vehicle" and has_plate else r.get("status_code", 1)
                stext = "Terbaca" if code == 1 else ("Perlu cek" if code == 2 else "Gagal")
                if object_type == "person":
                    conf = float(r.get("person_confidence") or r.get("face_confidence") or r.get("detection_confidence") or 0.0)
                elif has_plate:
                    conf = float(r.get("plate_detection_confidence") or r.get("plate_confidence") or r.get("event_ocr_confidence") or 0.0)
                else:
                    conf = float(r.get("vehicle_confidence") or r.get("detection_confidence") or 0.0)
                dt_str = r["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get("detected_at"), datetime) else str(r.get("detected_at") or "")

                items.append({
                    "id": r["detection_id"], "type": dtype, "object_type": object_type,
                    "vehicle_type": r.get("vehicle_type") or "unknown",
                    "track_id": r.get("track_id"), "has_plate": has_plate,
                    "has_driver": bool(r.get("has_driver")),
                    "direction": r.get("direction") or "unknown",
                    "event_key": r.get("event_key"), "driver_track_id": r.get("driver_track_id"),
                    "plate": p_num or "-", "camera": r.get("camera_name") or "CCTV",
                    "camera_id": r.get("camera_id"), "confidence": conf,
                    "confidence_percent": round(conf * 100, 1),
                    "vehicle_confidence": float(r.get("vehicle_confidence") or 0),
                    "plate_detection_confidence": float(r.get("plate_confidence") or r.get("plate_detection_confidence") or 0),
                    "ocr_confidence": float(r.get("event_ocr_confidence") or r.get("ocr_confidence") or 0),
                    "person_confidence": float(r.get("person_confidence") or 0),
                    "face_confidence": float(r.get("face_confidence") or 0),
                    "vehicle_confidence_percent": round(float(r.get("vehicle_confidence") or 0) * 100, 1),
                    "plate_confidence_percent": round(float(r.get("plate_confidence") or r.get("plate_detection_confidence") or 0) * 100, 1),
                    "ocr_confidence_percent": round(float(r.get("event_ocr_confidence") or r.get("ocr_confidence") or 0) * 100, 1),
                    "person_confidence_percent": round(float(r.get("person_confidence") or r.get("face_confidence") or 0) * 100, 1),
                    "timestamp": dt_str, "status": stext, "status_code": code,
                    "plate_image_path": r.get("plate_image_path"),
                    "face_image_path": r.get("face_image_path"),
                    "driver_face_path": r.get("driver_face_path"),
                    "vehicle_image_path": r.get("vehicle_image_path")
                })

            total_pages = max(1, (total + limit - 1) // limit)
            return {"items": items, "total": total, "page": page, "limit": limit, "total_pages": total_pages}


def get_plate_history_paginated(
    page: int = 1, limit: int = 20, search: str = None,
    camera_id: int = None, status_filter: str = "all",
    start_date: str = None, end_date: str = None
) -> dict:
    page = max(1, int(page)); limit = max(1, min(100, int(limit)))
    offset = (page - 1) * limit

    where_clauses = ["p.plate_number IS NOT NULL"]; params = []
    if search:
        s = f"%{search.strip()}%"
        where_clauses.append("(p.plate_number LIKE %s OR c.location LIKE %s)"); params.extend([s, s])
    if status_filter in ("0", "1", "2"):
        where_clauses.append("p.detection_status = %s"); params.append(int(status_filter))
    if camera_id is not None and str(camera_id).isdigit():
        where_clauses.append("pl.camera_id = %s"); params.append(int(camera_id))
    if start_date:
        where_clauses.append("DATE(p.created_at) >= %s"); params.append(start_date)
    if end_date:
        where_clauses.append("DATE(p.created_at) <= %s"); params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    with get_db() as conn:
        with conn.cursor() as cur:
            all_ids = _dedup_plate_ids(cur, where_sql, params)
            total = len(all_ids)
            if total == 0:
                return {"items": [], "total": 0, "page": page, "limit": limit, "total_pages": 1}

            page_ids = all_ids[offset: offset + limit]
            placeholders = ",".join(["%s"] * len(page_ids))

            data_sql = f"""
                SELECT p.plate_id, p.plate_number, p.detection_status,
                    p.detection_confidence, p.ocr_confidence, p.plate_image_path,
                    p.created_at AS detected_at,
                    COALESCE(c.location, 'CCTV') AS camera_name, pl.camera_id
                FROM plate p
                LEFT JOIN plate_logs pl ON p.plate_id = pl.plate_id
                LEFT JOIN cameras c ON pl.camera_id = c.camera_id
                WHERE p.plate_id IN ({placeholders})
                ORDER BY p.created_at DESC
            """
            cur.execute(data_sql, page_ids)
            rows = cur.fetchall()

            items = []
            for r in rows:
                code = r.get("detection_status", 1)
                stext = "Terbaca" if code == 1 else ("Perlu cek" if code == 2 else "Gagal")
                conf = float(r.get("ocr_confidence") or r.get("detection_confidence") or 0.0)
                dt_str = r["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get("detected_at"), datetime) else str(r.get("detected_at") or "")
                items.append({
                    "id": r["plate_id"], "plate": r["plate_number"],
                    "camera": r.get("camera_name") or "CCTV", "camera_id": r.get("camera_id"),
                    "confidence": conf, "confidence_percent": round(conf * 100, 1),
                    "timestamp": dt_str, "status": stext, "status_code": code,
                    "image_path": r.get("plate_image_path")
                })

            total_pages = max(1, (total + limit - 1) // limit)
            return {"items": items, "total": total, "page": page, "limit": limit, "total_pages": total_pages}


def get_enterprise_statistics(period: str = "today") -> dict:
    period = period.lower() if period else "today"
    now = datetime.now()

    if period == "today":
        cond_fd = "DATE(fd.created_at) = CURDATE()"
        cond_p = "DATE(p.created_at) = CURDATE()"
        period_label = "Hari Ini"
    elif period == "7d":
        cond_fd = "fd.created_at >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)"
        cond_p = "p.created_at >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)"
        period_label = "7 Hari Terakhir"
    elif period == "30d":
        cond_fd = "fd.created_at >= DATE_SUB(CURDATE(), INTERVAL 29 DAY)"
        cond_p = "p.created_at >= DATE_SUB(CURDATE(), INTERVAL 29 DAY)"
        period_label = "30 Hari Terakhir"
    else:
        cond_fd = "1=1"; cond_p = "1=1"; period_label = "Semua Waktu"

    dedup_subquery = f"""
        SELECT
            fd.detection_id, fd.camera_id, fd.object_type, fd.has_plate,
            fd.plate_id, fd.detection_confidence, fd.created_at, fd.track_id,
            fd.direction, fd.detection_status,
            fd.vehicle_hash, fd.face_hash,
            CASE
                WHEN p.plate_number IS NOT NULL AND p.plate_number != ''
                    THEN CONCAT('plate:', p.plate_number)
                WHEN fd.object_type = 'vehicle' AND fd.vehicle_hash IS NOT NULL
                    THEN CONCAT('vhash:', fd.vehicle_hash)
                WHEN fd.object_type = 'person' AND fd.face_hash IS NOT NULL
                    THEN CONCAT('fhash:', fd.face_hash)
                ELSE CONCAT('track:', COALESCE(fd.track_id, 0))
            END AS dedup_key,
            ROW_NUMBER() OVER (
                PARTITION BY
                    fd.camera_id,
                    CASE
                        WHEN p.plate_number IS NOT NULL AND p.plate_number != ''
                            THEN CONCAT('plate:', p.plate_number)
                        WHEN fd.object_type = 'vehicle' AND fd.vehicle_hash IS NOT NULL
                            THEN CONCAT('vhash:', fd.vehicle_hash)
                        WHEN fd.object_type = 'person' AND fd.face_hash IS NOT NULL
                            THEN CONCAT('fhash:', fd.face_hash)
                        ELSE CONCAT('track:', COALESCE(fd.track_id, 0))
                    END,
                    FLOOR(UNIX_TIMESTAMP(fd.created_at) / {DEDUP_WINDOW_SECONDS})
                ORDER BY fd.detection_confidence DESC, fd.detection_id DESC
            ) AS rn
        FROM full_detection fd
        LEFT JOIN plate p ON p.plate_id = fd.plate_id
        WHERE {cond_fd}
    """

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) AS total_detections,
                    SUM(d.object_type = 'vehicle') AS total_vehicles,
                    SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS total_plates,
                    SUM(d.object_type = 'person') AS total_faces,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'entry') AS vehicle_entry,
                    SUM(d.object_type = 'vehicle' AND d.direction = 'exit') AS vehicle_exit,
                    SUM(d.object_type = 'person' AND d.direction = 'entry') AS people_entry,
                    SUM(d.object_type = 'person' AND d.direction = 'exit') AS people_exit,
                    SUM(d.detection_status = 1) AS status_valid,
                    SUM(d.detection_status = 2) AS status_warning,
                    SUM(d.detection_status = 0) AS status_failed,
                    AVG(d.detection_confidence) AS avg_conf
                FROM ({dedup_subquery}) d
                WHERE d.rn = 1
            """)
            kpi_row = cur.fetchone()

            total_dets = int(kpi_row["total_detections"] or 0)
            total_vehicles = int(kpi_row["total_vehicles"] or 0)
            total_plates = int(kpi_row["total_plates"] or 0)
            total_faces = int(kpi_row["total_faces"] or 0)
            vehicle_entry = int(kpi_row["vehicle_entry"] or 0)
            vehicle_exit = int(kpi_row["vehicle_exit"] or 0)
            people_entry = int(kpi_row["people_entry"] or 0)
            people_exit = int(kpi_row["people_exit"] or 0)
            status_valid = int(kpi_row["status_valid"] or 0)
            status_warning = int(kpi_row["status_warning"] or 0)
            status_failed = int(kpi_row["status_failed"] or 0)
            avg_conf = float(kpi_row["avg_conf"] or 0.85)

            plate_read_rate = round((float(total_plates) / max(total_vehicles, 1)) * 100, 1) if total_vehicles > 0 else 0.0

            cur.execute("SELECT COUNT(*) AS total_cams, SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS active_cams FROM cameras;")
            cam_info = cur.fetchone()
            total_cams = int(cam_info["total_cams"] or 0)
            active_cams = int(cam_info["active_cams"] or 0)
            cam_availability = round((float(active_cams) / max(total_cams, 1)) * 100, 1)

            trend_labels = []; trend_plates = []; trend_faces = []; trend_totals = []

            if period == "today":
                cur.execute(f"""
                    SELECT HOUR(d.created_at) AS hr, COUNT(*) AS total,
                        SUM(d.object_type = 'vehicle') AS vehicles,
                        SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS plates,
                        SUM(d.object_type = 'person') AS faces
                    FROM ({dedup_subquery}) d
                    WHERE d.rn = 1
                    GROUP BY HOUR(d.created_at) ORDER BY hr ASC
                """)
                hour_map = {r["hr"]: r for r in cur.fetchall()}
                for h in range(24):
                    trend_labels.append(f"{h:02d}:00")
                    row = hour_map.get(h, {})
                    trend_plates.append(int(row.get("plates", 0) or 0))
                    trend_faces.append(int(row.get("faces", 0) or 0))
                    trend_totals.append(int(row.get("total", 0) or 0))
            elif period in ("7d", "30d"):
                num_days = 7 if period == "7d" else 30
                cur.execute(f"""
                    SELECT DATE(d.created_at) AS dt, COUNT(*) AS total,
                        SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS plates,
                        SUM(d.object_type = 'person') AS faces
                    FROM ({dedup_subquery}) d
                    WHERE d.rn = 1
                    GROUP BY DATE(d.created_at) ORDER BY dt ASC
                """)
                date_map = {str(r["dt"]): r for r in cur.fetchall()}
                for i in range(num_days - 1, -1, -1):
                    day_d = now - timedelta(days=i)
                    d_key = day_d.strftime("%Y-%m-%d")
                    trend_labels.append(day_d.strftime("%d/%m"))
                    row = date_map.get(d_key, {})
                    trend_plates.append(int(row.get("plates", 0) or 0))
                    trend_faces.append(int(row.get("faces", 0) or 0))
                    trend_totals.append(int(row.get("total", 0) or 0))
            else:
                cur.execute(f"""
                    SELECT DATE(d.created_at) AS dt, COUNT(*) AS total,
                        SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS plates,
                        SUM(d.object_type = 'person') AS faces
                    FROM ({dedup_subquery}) d
                    WHERE d.rn = 1
                    GROUP BY DATE(d.created_at) ORDER BY dt ASC LIMIT 30
                """)
                for r in cur.fetchall():
                    d_obj = r["dt"]
                    trend_labels.append(d_obj.strftime("%d/%m") if isinstance(d_obj, datetime) else str(d_obj))
                    trend_plates.append(int(r.get("plates") or 0))
                    trend_faces.append(int(r.get("faces") or 0))
                    trend_totals.append(int(r.get("total") or 0))

            cur.execute(f"""
                SELECT COALESCE(c.location, 'CCTV') AS camera_name,
                    COUNT(d.detection_id) AS total_count,
                    SUM(d.object_type = 'vehicle' AND (d.has_plate = 1 OR d.plate_id IS NOT NULL)) AS plate_count,
                    SUM(d.object_type = 'person') AS face_count
                FROM ({dedup_subquery}) d
                LEFT JOIN cameras c ON c.camera_id = d.camera_id
                WHERE d.rn = 1
                GROUP BY d.camera_id, c.location
                ORDER BY total_count DESC
            """)
            camera_distribution = []
            for cr in cur.fetchall():
                c_tot = int(cr["total_count"] or 0)
                camera_distribution.append({
                    "camera_name": cr["camera_name"],
                    "total_count": c_tot,
                    "plate_count": int(cr["plate_count"] or 0),
                    "face_count": int(cr["face_count"] or 0),
                    "percentage": round((c_tot / max(total_dets, 1)) * 100, 1)
                })

            cur.execute(f"""
                SELECT HOUR(d.created_at) AS hr, COUNT(*) AS cnt
                FROM ({dedup_subquery}) d
                WHERE d.rn = 1
                GROUP BY HOUR(d.created_at) ORDER BY cnt DESC LIMIT 3
            """)
            peak_hours = []
            for pr in cur.fetchall():
                h = pr["hr"]; c_peak = pr["cnt"]
                peak_hours.append({
                    "time_range": f"{h:02d}:00 - {h+1:02d}:00 WIB",
                    "count": c_peak,
                    "percentage": round((c_peak / max(total_dets, 1)) * 100, 1)
                })

            cur.execute(f"""
                SELECT p.plate_number, COUNT(*) AS total_seen,
                    MAX(p.created_at) AS last_seen,
                    MAX(p.detection_status) AS last_status,
                    AVG(p.detection_confidence) AS avg_conf,
                    COALESCE(MAX(c.location), 'CCTV') AS last_camera
                FROM ({dedup_subquery}) d
                JOIN plate p ON p.plate_id = d.plate_id
                LEFT JOIN cameras c ON c.camera_id = d.camera_id
                WHERE d.rn = 1 AND p.plate_number IS NOT NULL AND p.plate_number != ''
                GROUP BY p.plate_number ORDER BY total_seen DESC LIMIT 10
            """)
            top_plates = []
            for tp in cur.fetchall():
                code = tp.get("last_status", 1)
                stext = "Terbaca" if code == 1 else ("Perlu cek" if code == 2 else "Gagal")
                ls = tp.get("last_seen")
                ls_str = ls.strftime("%Y-%m-%d %H:%M") if isinstance(ls, datetime) else str(ls or "")
                top_plates.append({
                    "plate_number": tp["plate_number"], "total_seen": tp["total_seen"],
                    "last_seen": ls_str, "last_camera": tp.get("last_camera") or "CCTV",
                    "status": stext, "status_code": code,
                    "avg_confidence_percent": round(float(tp.get("avg_conf") or 0.8) * 100, 1)
                })

            return {
                "period": period, "period_label": period_label,
                "kpi": {
                    "total_detections": total_dets, "total_plates": total_plates,
                    "total_faces": total_faces, "plate_read_rate": plate_read_rate,
                    "avg_confidence": round(avg_conf * 100, 1),
                    "need_check_count": status_warning,
                    "status_valid": status_valid, "status_failed": status_failed,
                    "active_cameras": active_cams, "total_cameras": total_cams,
                    "camera_availability": cam_availability
                },
                "trend": {"labels": trend_labels, "plates": trend_plates,
                          "faces": trend_faces, "totals": trend_totals},
                "camera_distribution": camera_distribution,
                "status_breakdown": {"valid": status_valid, "warning": status_warning, "failed": status_failed},
                "peak_hours": peak_hours, "top_plates": top_plates
            }


def seed_demo_data(count: int = 60) -> int:
    sample_plates = [
        "B 1982 UJ", "B 2341 SKO", "D 1089 AB", "B 8821 QW", "B 1204 PF",
        "D 4452 KL", "F 3819 GA", "B 9090 VIP", "L 1422 ZX", "B 5512 TYY",
        "AD 6721 NB", "B 3110 KLA", "DK 8812 BC", "B 4729 MNB", "B 6182 KPR",
        "B 2888 BOS", "D 3311 BB", "F 9012 QQ", "B 1542 WTR", "B 7001 JKT"
    ]
    vehicle_types = ["car", "motorcycle", "car", "car", "motorcycle", "truck"]

    cams = get_all_cameras()
    cam_ids = [c["camera_id"] for c in cams] if cams else [1]
    cam_direction_map = {c["camera_id"]: c.get("direction", "unknown") for c in cams} if cams else {}

    now = datetime.now()
    inserted = 0

    with get_db() as conn:
        with conn.cursor() as cur:
            for _ in range(count):
                days_ago = random.choice([0, 0, 0, 1, 1, 2, 3, 4, 5, 6])
                hour = random.choices(list(range(24)),
                    weights=[1,1,1,1,1,2,5,12,15,8,7,9,11,10,8,9,14,16,12,7,5,4,2,1])[0]
                dt = (now - timedelta(days=days_ago)).replace(hour=hour,
                    minute=random.randint(0, 59), second=random.randint(0, 59))
                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                cam_id = random.choice(cam_ids)
                object_type = "vehicle" if random.random() < 0.75 else "person"
                cam_dir = cam_direction_map.get(cam_id, "unknown")
                direction = cam_dir if cam_dir in ("entry", "exit") else random.choice(["entry", "exit"])

                has_plate = object_type == "vehicle" and random.random() < 0.65
                has_driver = object_type == "vehicle" and random.random() < 0.60
                vehicle_type = random.choice(vehicle_types) if object_type == "vehicle" else "unknown"
                vehicle_conf = round(random.uniform(0.65, 0.97), 4) if object_type == "vehicle" else None

                plate_id = None; p_status = 1; p_conf = round(random.uniform(0.75, 0.98), 4)
                if has_plate:
                    plate_num = random.choice(sample_plates)
                    if random.random() < 0.15:
                        p_status = 2; p_conf = round(random.uniform(0.45, 0.68), 4)
                    elif random.random() < 0.05:
                        p_status = 0; p_conf = round(random.uniform(0.20, 0.38), 4)
                    cur.execute(
                        """INSERT INTO plate (plate_number, normalized_plate_number, raw_ocr_text,
                            detection_status, detection_confidence, ocr_confidence, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s);""",
                        (plate_num, plate_num, plate_num, p_status, p_conf, p_conf, dt_str)
                    )
                    plate_id = cur.lastrowid
                    cur.execute("""INSERT INTO plate_logs (plate_id, camera_id, status, created_at)
                                   VALUES (%s, %s, %s, %s);""", (plate_id, cam_id, p_status, dt_str))

                if object_type == "person":
                    f_conf = round(random.uniform(0.60, 0.96), 4)
                    f_status = compute_face_status(f_conf)
                    person_conf = f_conf; face_conf = f_conf
                    v_conf = None; plate_det_conf = None; ocr_conf_val = None
                elif has_plate:
                    f_conf = p_conf; f_status = p_status
                    person_conf = None; face_conf = None
                    v_conf = vehicle_conf; plate_det_conf = p_conf; ocr_conf_val = p_conf
                else:
                    f_conf = vehicle_conf
                    f_status = 1 if vehicle_conf >= 0.70 else (2 if vehicle_conf >= 0.40 else 0)
                    person_conf = None; face_conf = None
                    v_conf = vehicle_conf; plate_det_conf = None; ocr_conf_val = None

                cur.execute(
                    """INSERT INTO full_detection (plate_id, camera_id, detection_status, detection_confidence,
                        object_type, vehicle_type, has_plate, has_driver,
                        vehicle_confidence, plate_detection_confidence, ocr_confidence,
                        person_confidence, face_confidence, direction, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                    (plate_id, cam_id, f_status, f_conf, object_type, vehicle_type,
                     int(has_plate), int(has_driver), v_conf, plate_det_conf, ocr_conf_val,
                     person_conf, face_conf, direction, dt_str)
                )
                inserted += 1
    return inserted


def get_camera_zone(camera_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT mid_zone, near_zone FROM camera_zones WHERE camera_id = %s", (camera_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {"camera_id": camera_id, "mid": json.loads(row["mid_zone"]), "near": json.loads(row["near_zone"])}
    except Exception as exc:
        print(f"[DB ZONE GET ERROR] camera={camera_id}: {exc}")
        return None


def save_camera_zone(camera_id, mid, near):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO camera_zones (camera_id, mid_zone, near_zone)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE mid_zone = VALUES(mid_zone), near_zone = VALUES(near_zone)
                """, (camera_id, json.dumps(mid), json.dumps(near)))
            conn.commit()
        return True
    except Exception as exc:
        print(f"[DB ZONE SAVE ERROR] camera={camera_id}: {exc}")
        return False


def delete_camera_zone(camera_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM camera_zones WHERE camera_id = %s", (camera_id,))
            conn.commit()
        return True
    except Exception as exc:
        print(f"[DB ZONE DELETE ERROR] camera={camera_id}: {exc}")
        return False


def get_all_camera_zones():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT camera_id, mid_zone, near_zone FROM camera_zones")
                rows = cur.fetchall()
                return [{"camera_id": row["camera_id"], "mid": json.loads(row["mid_zone"]),
                         "near": json.loads(row["near_zone"])} for row in rows]
    except Exception as exc:
        print(f"[DB ZONE LIST ERROR] {exc}")
        return []