import os
import re
import time
import threading
from datetime import datetime

import psutil

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import collections
import uuid

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker
import db

# ✅ IMPORT VALIDATOR PLAT INDONESIA
try:
    from ai.plate.validator import (
        correct_and_validate_plate,
        is_valid_plate,
        normalize_plate_text as _npv_normalize,
        format_plate_for_display,
        VALID_REGION_CODES,
    )
    HAS_PLATE_VALIDATOR = True
except ImportError:
    HAS_PLATE_VALIDATOR = False
    VALID_REGION_CODES = set()
    print("[AI STREAM WARNING] ai.plate.validator tidak tersedia, pakai fallback regex")


# PATH PROJECT

def _find_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    candidates = [current]
    parent = current
    for _ in range(4):
        parent = os.path.dirname(parent)
        candidates.append(parent)
    for candidate in candidates:
        if (
            os.path.isfile(os.path.join(candidate, "yolov8n.pt"))
            or os.path.isdir(os.path.join(candidate, "models"))
        ):
            return candidate
    return current


BASE_DIR = _find_project_root()

PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plate",
    "license-plate-finetune-v3n.pt",
)

CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_CAPTURE_DIR = os.path.join(CAPTURE_DIR, "plates")

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)


# CONFIGURATION

VEHICLE_CLASSES = [0, 2, 3, 5, 7]
VEHICLE_CONFIDENCE = 0.45
VEHICLE_IMGSZ = 480

# DISTANCE / DETECTION ZONES
ENABLE_DISTANCE_ZONE = True

# Ambang batas deteksi zona berbasis persentase body kendaraan (>= 50%)
ZONE_BBOX_OVERLAP_MIN = 0.50       # Minimal 50% body kendaraan berada di dalam zona
ZONE_AREA_COVERAGE_MIN = 0.40      # Proteksi kendaraan besar (truk/bus) yang menutupi >= 40% area zona
ZONE_CENTER_OVERLAP_MIN = 0.35     # Jika titik tengah di dalam zona, butuh minimal 35% body di dalam zona
ZONE_OVERLAP_FALLBACK_RATIO = 0.50  # Fallback kompatibilitas

DEFAULT_MID_ZONE = [
    (0.05, 0.10),
    (0.95, 0.10),
    (0.99, 1.00),
    (0.01, 1.00),
]

DEFAULT_NEAR_ZONE = [
    (0.05, 0.20),
    (0.95, 0.20),
    (0.99, 1.00),
    (0.01, 1.00),
]

_ZONE_MID = list(DEFAULT_MID_ZONE)
_ZONE_NEAR = list(DEFAULT_NEAR_ZONE)

CAMERA_ZONE_CONFIG = {
    1: {"name": "GSMasukViewLuar", "mid": _ZONE_MID, "near": _ZONE_NEAR},
    2: {"name": "GSMasukViewDalam", "mid": _ZONE_MID, "near": _ZONE_NEAR},
    3: {"name": "GSKeluarViewLuar", "mid": _ZONE_MID, "near": _ZONE_NEAR},
    4: {"name": "GSKeluarViewDalam", "mid": _ZONE_MID, "near": _ZONE_NEAR},
    "GSMasukViewLuar": {"mid": _ZONE_MID, "near": _ZONE_NEAR},
    "GSMasukViewDalam": {"mid": _ZONE_MID, "near": _ZONE_NEAR},
    "GSKeluarViewLuar": {"mid": _ZONE_MID, "near": _ZONE_NEAR},
    "GSKeluarViewDalam": {"mid": _ZONE_MID, "near": _ZONE_NEAR},
}

_runtime_zone_override = {}


def _normalize_camera_id(camera_id):
    if isinstance(camera_id, int):
        return camera_id
    if isinstance(camera_id, str):
        stripped = camera_id.strip()
        if stripped.isdigit():
            return int(stripped)
        for key, cfg in CAMERA_ZONE_CONFIG.items():
            if isinstance(cfg, dict) and cfg.get("name") == stripped:
                for k2, c2 in CAMERA_ZONE_CONFIG.items():
                    if isinstance(k2, int) and isinstance(c2, dict) and c2.get("name") == stripped:
                        return k2
                break
    return camera_id


def reload_camera_zone(camera_id, mid, near):
    camera_id = _normalize_camera_id(camera_id)
    _runtime_zone_override[camera_id] = {
        "mid": [(float(x), float(y)) for x, y in mid],
        "near": [(float(x), float(y)) for x, y in near],
    }
    print(f"[ZONES] Runtime override aktif untuk CAM {camera_id}")


def get_default_zone(camera_id):
    camera_id = _normalize_camera_id(camera_id)
    config = CAMERA_ZONE_CONFIG.get(camera_id, {})
    return (
        config.get("mid", DEFAULT_MID_ZONE),
        config.get("near", DEFAULT_NEAR_ZONE),
    )


def load_all_zones_from_db():
    try:
        import db as _db
        if not hasattr(_db, "get_all_camera_zones"):
            return
        rows = _db.get_all_camera_zones()
        for row in rows:
            reload_camera_zone(row["camera_id"], row["mid"], row["near"])
        if rows:
            print(f"[ZONES] {len(rows)} zona custom dimuat dari DB")
    except Exception as exc:
        print(f"[ZONES] Gagal memuat zona dari DB: {exc}")


# THRESHOLD DETEKSI
MIN_PERSON_WIDTH = 15
MIN_PERSON_HEIGHT = 35
MIN_VEHICLE_WIDTH = 30
MIN_VEHICLE_HEIGHT = 20

PLATE_VEHICLE_MIN_WIDTH = 30
PLATE_VEHICLE_MIN_HEIGHT = 20

FAKE_VEHICLE_ASPECT_MIN = 0.9
FAKE_MOTORCYCLE_ASPECT_MIN = 0.20

# AI INTERVAL
BASE_AI_INTERVAL = 0.60
MIN_AI_INTERVAL = 0.40
MAX_AI_INTERVAL = 1.20
TARGET_CPU = 75.0
HIGH_CPU = 85.0
LOW_CPU = 55.0
AI_ADJUST_COOLDOWN = 3.0

VEHICLE_TYPES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# PLATE DETECTION
PLATE_CONFIDENCE = 0.25
PLATE_IMGSZ = 512
PLATE_MAX_DET = 5
PLATE_IOU = 0.45
PLATE_MIN_WIDTH = 12
PLATE_MIN_HEIGHT = 4

OCR_MIN_PLATE_WIDTH = 18
OCR_MIN_PLATE_HEIGHT = 5
OCR_MIN_PLATE_CONFIDENCE = 0.25

MAX_PLATE_ROI = 3

PLATE_DETECT_INTERVAL = 0.60
OCR_INTERVAL = 0.80

PLATE_TRACKER_IOU = 0.35
PLATE_TRACKER_FRAME_GAP = 40
PLATE_TRACKER_OCR_EVERY = 2
PLATE_TRACKER_MIN_CONFIDENCE = 0.35
PLATE_TRACKER_MAX_HISTORY = 5
PLATE_PADDING = 0.08

# ANTI-DUPLIKAT
PLATE_REVIEW_CONFIDENCE = 0.50
PLATE_COOLDOWN = 45.0
PERSON_COOLDOWN = 45.0

TRACK_REUSE_GAP = 8.0

EVENT_SESSION_ID_LENGTH = 12

TRACKER_LOST_BUFFER = 150
TRACKER_MATCH_THRESHOLD = 0.80
TRACKER_ACTIVATION_THRESHOLD = 0.30

TRACK_EVENT_COOLDOWN = 2.0

VISUAL_DEDUP_COOLDOWN = 60.0
VISUAL_HAMMING_THRESHOLD = 3

PERSON_DEDUP_COOLDOWN = 120.0
PERSON_HAMMING_THRESHOLD = 8
PERSON_TRACK_DEDUP_COOLDOWN = 300.0

# Object Grouping Configurations
OBJECT_GROUP_MAX_GAP = 8.0
OBJECT_GROUP_MAX_CENTROID_DISTANCE = 120.0
OBJECT_GROUP_PERSON_MAX_DIST = 130.0
OBJECT_GROUP_AREA_TOLERANCE = 0.50
OBJECT_GROUP_MIN_IOU = 0.15

PERSON_AREA_TOLERANCE = 0.45
PERSON_PROXIMITY_DIST = 60.0

VEHICLE_CENTROID_DEDUP_DISTANCE = 150.0
VEHICLE_CENTROID_DEDUP_AREA_RATIO = 0.40
VEHICLE_CENTROID_DEDUP_COOLDOWN = 30.0
VEHICLE_CENTROID_MAX_HISTORY = 20

PLATE_TEXT_DEDUP_COOLDOWN = 120.0

PERSON_CAPTURE_CONFIDENCE = 0.40
MIN_PERSON_CROP_WIDTH = 25
MIN_PERSON_CROP_HEIGHT = 45

DISPLAY_FPS = 15

FOCUS_TARGET_HOLD_SECONDS = 1.5
RESULT_HOLD_SECONDS = 1.5
FOCUS_PADDING_VEHICLE = 0.12
FOCUS_PADDING_PLATE = 0.55
LIGHT_DARK_THRESHOLD = 72.0
LIGHT_OVER_THRESHOLD = 205.0
LIGHT_GLARE_RATIO = 0.18

DEBUG_PERFORMANCE = True
DEBUG_LOG_SAMPLE_EVERY = 30

# ✅ Confidence improvement threshold untuk "best frame only"
CONFIDENCE_IMPROVEMENT_THRESHOLD = 0.05


# HELPERS

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(text):
    """Normalisasi dasar: uppercase, hapus non-alphanumeric. TIDAK koreksi."""
    if text is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper().strip())


def _correct_plate_text(raw_text):
    """
    ✅ Koreksi OCR mentah menjadi plat valid (kode wilayah Indonesia).
    Return (corrected_text, is_valid, penalty).
    """
    if not raw_text:
        return "", False, 0.0

    if HAS_PLATE_VALIDATOR:
        result = correct_and_validate_plate(raw_text)
        return (
            result["corrected"],
            result["valid"],
            result["confidence_penalty"],
        )

    # Fallback regex longgar
    normalized = _normalize_text(raw_text)
    valid = bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", normalized))
    return normalized, valid, 0.0


def _valid_indonesian_plate(text):
    """✅ Validasi plat dengan kode wilayah resmi."""
    if not text:
        return False

    if HAS_PLATE_VALIDATOR:
        return is_valid_plate(text)

    # Fallback regex
    text = _normalize_text(text)
    if not text or not (3 <= len(text) <= 9):
        return False
    return bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text))


def _format_plate_display(text):
    """Format untuk display: 'AB1234CD' -> 'AB 1234 CD'."""
    if HAS_PLATE_VALIDATOR:
        return format_plate_for_display(text)

    text = _normalize_text(text)
    match = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", text)
    if not match:
        return text
    prefix, number, suffix = match.groups()
    return " ".join(x for x in (prefix, number, suffix) if x)


def _extract_region_code(text):
    """Ambil kode wilayah dari plat (kalau ada)."""
    if not text or not HAS_PLATE_VALIDATOR:
        return None
    text = _normalize_text(text)
    for length in (2, 1):
        if len(text) >= length + 1:
            prefix = text[:length]
            if prefix in VALID_REGION_CODES:
                return prefix
    return None


def _safe_filename(text):
    text = str(text or "unknown")
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:80]


def _save_image(image, directory, filename, quality=94):
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    try:
        ok = cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return path if ok else None
    except Exception as exc:
        print(f"[CAPTURE ERROR] {exc}")
        return None


def _prepare_high_quality_capture(
    frame, bbox, padding=0.18, target_short_side=420,
    max_scale=3.0, max_long_side=1280, sharpen=True,
):
    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * float(padding))
    py = int(bh * float(padding))
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    short_side = min(cw, ch)
    if short_side > 0 and short_side < target_short_side:
        scale = target_short_side / float(short_side)
        scale = min(scale, float(max_scale))
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    ch, cw = crop.shape[:2]
    long_side = max(cw, ch)
    if long_side > max_long_side:
        scale = max_long_side / float(long_side)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if sharpen and crop.shape[1] >= 3 and crop.shape[0] >= 3:
        blur = cv2.GaussianBlur(crop, (0, 0), 0.8)
        crop = cv2.addWeighted(crop, 1.12, blur, -0.12, 0)
    return crop


def save_person_capture(frame, bbox, track_id, camera_id):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 - x1 < MIN_PERSON_CROP_WIDTH or y2 - y1 < MIN_PERSON_CROP_HEIGHT:
        return None
    crop = _prepare_high_quality_capture(
        frame, (x1, y1, x2, y2), padding=0.20, target_short_side=360,
        max_scale=2.5, max_long_side=960, sharpen=True,
    )
    if crop is None:
        return None
    now = datetime.now()
    date_dir = os.path.join(CAPTURE_DIR, now.strftime("%Y%m%d"))
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"person_cam{camera_id}_track{track_id}_{ts}.jpg"
    return _save_image(crop, date_dir, filename, 95)


def save_plate_capture(crop, track_id, camera_id, plate_text, prefix="plate"):
    if crop is None or crop.size == 0:
        return None
    now = datetime.now()
    date_dir = os.path.join(PLATE_CAPTURE_DIR, now.strftime("%Y%m%d"))
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_plate = _safe_filename(plate_text or "unknown")
    filename = f"{prefix}_cam{camera_id}_track{track_id}_{ts}_{safe_plate}.jpg"
    return _save_image(crop, date_dir, filename, 95)


def _prepare_vehicle_capture(frame, bbox):
    return _prepare_high_quality_capture(
        frame, bbox, padding=0.15, target_short_side=480,
        max_scale=2.5, max_long_side=1280, sharpen=True,
    )


def _get_value(data, keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _bbox_from_track(track):
    bbox = getattr(track, "bbox", None)
    if bbox is None:
        return None
    try:
        return [int(v) for v in bbox]
    except Exception:
        return None


def _track_to_result(track):
    """✅ Konversi Track → dict, dengan koreksi & validasi plat."""
    bbox = _bbox_from_track(track)
    if bbox is None:
        return None

    vote = None
    try:
        vote = track.hasil_voting()
    except Exception:
        vote = None

    text = ""
    formatted = ""
    ocr_conf = 0.0
    votes = 0
    total_reads = 0
    raw_text = ""
    correction_applied = False
    penalty = 0.0

    if vote:
        raw_text = vote.get("text", "")
        votes = int(vote.get("jumlah_muncul", 0) or 0)
        total_reads = int(vote.get("total_bacaan", 0) or 0)

        if HAS_PLATE_VALIDATOR:
            result = correct_and_validate_plate(raw_text)
            if result["valid"]:
                text = result["corrected"]
                formatted = _format_plate_display(text)
                penalty = result["confidence_penalty"]
                ocr_conf = _safe_float(vote.get("confidence_rata2", 0.0)) * (1.0 - penalty * 0.5)
                ocr_conf = max(0.0, ocr_conf)
                correction_applied = (text != raw_text)
            else:
                text = _normalize_text(raw_text)
                formatted = vote.get("formatted", text)
                ocr_conf = _safe_float(vote.get("confidence_rata2", 0.0))
        else:
            text = _normalize_text(raw_text)
            formatted = vote.get("formatted", text)
            ocr_conf = _safe_float(vote.get("confidence_rata2", 0.0))

    if not text:
        raw_best = getattr(track, "best_text", "")
        raw_text = raw_text or raw_best
        if HAS_PLATE_VALIDATOR and raw_best:
            result = correct_and_validate_plate(raw_best)
            if result["valid"]:
                text = result["corrected"]
                formatted = _format_plate_display(text)
                penalty = result["confidence_penalty"]
                ocr_conf = _safe_float(getattr(track, "best_ocr_confidence", 0.0)) * (1.0 - penalty * 0.5)
                correction_applied = (text != raw_best)
            else:
                text = _normalize_text(raw_best)
                formatted = text
                ocr_conf = _safe_float(getattr(track, "best_ocr_confidence", 0.0))
        else:
            text = _normalize_text(raw_best)
            ocr_conf = _safe_float(getattr(track, "best_ocr_confidence", 0.0))
            formatted = text

    return {
        "id": int(getattr(track, "id", -1)),
        "track_id": int(getattr(track, "id", -1)),
        "box": bbox, "bbox": bbox,
        "conf": _safe_float(getattr(track, "detection_confidence", 0.0)),
        "detection_confidence": _safe_float(getattr(track, "detection_confidence", 0.0)),
        "text": text,
        "formatted": formatted,
        "raw_text": raw_text,
        "ocr_conf": ocr_conf,
        "confidence": ocr_conf,
        "valid": _valid_indonesian_plate(text),
        "votes": votes,
        "total_reads": total_reads,
        "crop": getattr(track, "best_crop", None),
        "vehicle_track_id": (
            int(getattr(track, "vehicle_track_id"))
            if getattr(track, "vehicle_track_id", None) is not None
            else None
        ),
        "vehicle_generation": getattr(track, "vehicle_generation", 1),
        "distance_zone": "NEAR",
        "correction_applied": correction_applied,
        "correction_penalty": penalty,
        "region_code": _extract_region_code(text),
    }


def _clamp_box(box, w, h, padding=0.0):
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)
    px = int(bw * padding); py = int(bh * padding)
    return [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]


def _lighting_metrics(frame, box):
    h, w = frame.shape[:2]; x1, y1, x2, y2 = _clamp_box(box, w, h, 0.0)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"brightness": 0.0, "contrast": 0.0, "dark_ratio": 1.0, "glare_ratio": 0.0, "score": 0.0, "status": "NO ROI"}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()); contrast = float(gray.std())
    dark_ratio = float(np.mean(gray < 45)); glare_ratio = float(np.mean(gray > 235))
    bscore = max(0.0, 100.0 - abs(brightness - 125.0) * 0.75)
    cscore = min(100.0, contrast * 2.0)
    score = max(0.0, min(100.0, 0.65 * bscore + 0.35 * cscore - glare_ratio * 80.0))
    if glare_ratio >= LIGHT_GLARE_RATIO: status = 'GLARE'
    elif brightness < 45: status = 'TOO DARK'
    elif brightness < LIGHT_DARK_THRESHOLD: status = 'LOW LIGHT'
    elif brightness > LIGHT_OVER_THRESHOLD: status = 'OVEREXPOSED'
    elif score >= 72: status = 'GOOD'
    else: status = 'FAIR'
    return {"brightness": brightness, "contrast": contrast, "dark_ratio": dark_ratio, "glare_ratio": glare_ratio, "score": score, "status": status}


def _enhance_for_lighting(frame, metrics):
    status = (metrics or {}).get('status', 'GOOD')
    if status not in ('TOO DARK', 'LOW LIGHT', 'OVEREXPOSED', 'GLARE'):
        return frame
    img = frame
    if status in ('TOO DARK', 'LOW LIGHT'):
        gamma = 0.58 if status == 'TOO DARK' else 0.78
    else:
        gamma = 1.28
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')
    img = cv2.LUT(img, lut)
    if status in ('TOO DARK', 'LOW LIGHT'):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img


def _center(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def _bottom_center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (int((x1 + x2) / 2.0), int(y2))


def _normalized_polygon_to_pixels(points, width, height):
    return np.array(
        [[
            int(max(0.0, min(1.0, float(x))) * width),
            int(max(0.0, min(1.0, float(y))) * height),
        ] for x, y in points],
        dtype=np.int32,
    )


def _camera_zone_points(camera_id, zone_name):
    camera_id = _normalize_camera_id(camera_id)
    override = _runtime_zone_override.get(camera_id)
    if override:
        if zone_name == "near":
            return override.get("near", DEFAULT_NEAR_ZONE)
        return override.get("mid", DEFAULT_MID_ZONE)
    config = CAMERA_ZONE_CONFIG.get(camera_id, {})
    if zone_name == "near":
        return config.get("near", DEFAULT_NEAR_ZONE)
    return config.get("mid", DEFAULT_MID_ZONE)


def _camera_zone_name(camera_id):
    camera_id = _normalize_camera_id(camera_id)
    config = CAMERA_ZONE_CONFIG.get(camera_id, {})
    return config.get("name", str(camera_id))


def _bbox_polygon_overlap_ratio(box, polygon, width, height):
    if polygon is None:
        return 0.0
    polygon = np.asarray(polygon, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    bw = x2 - x1
    bh = y2 - y1
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] = shifted[:, 0] - x1
    shifted[:, 1] = shifted[:, 1] - y1
    cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 255)
    inside = int(cv2.countNonZero(mask))
    total = int(bw * bh)
    if total <= 0:
        return 0.0
    return inside / float(total)


def _point_in_zone(box, zone_points, width, height):
    if not ENABLE_DISTANCE_ZONE:
        return True
    if not zone_points or len(zone_points) < 3:
        return True
    polygon = _normalized_polygon_to_pixels(zone_points, width, height)
    polygon = np.asarray(polygon, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return True

    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return False

    bw = x2 - x1
    bh = y2 - y1
    box_area = float(bw * bh)
    if box_area <= 0:
        return False

    # Hitung irisan piksel antara bounding box objek dan polygon zona
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] = shifted[:, 0] - x1
    shifted[:, 1] = shifted[:, 1] - y1
    cv2.fillPoly(mask, [shifted.reshape((-1, 1, 2))], 255)
    inside = float(cv2.countNonZero(mask))

    # 1. Rasio irisan terhadap luas bounding box (persentase body di dalam zona)
    box_ratio = inside / box_area

    # 2. Rasio irisan terhadap luas polygon zona (proteksi kendaraan besar yang memenuhi zona)
    poly_area = float(cv2.contourArea(polygon))
    zone_ratio = (inside / poly_area) if poly_area > 0 else 0.0

    # 3. Posisi titik tengah (centroid) objek
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_inside = cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)),
        (float(cx), float(cy)),
        False,
    ) >= 0

    # Evaluasi:
    # A. Minimal 50% body objek berada di dalam zona (menghentikan kendaraan luar yg menyenggol batas)
    if box_ratio >= ZONE_BBOX_OVERLAP_MIN:
        return True

    # B. Proteksi kendaraan besar (truk/bus): objek menutupi >= 40% area zona
    if zone_ratio >= ZONE_AREA_COVERAGE_MIN:
        return True

    # C. Objek berpusat di zona: titik tengah di dalam zona DAN minimal 35% body di dalam zona
    if center_inside and box_ratio >= ZONE_CENTER_OVERLAP_MIN:
        return True

    return False


def _get_distance_zone(box, camera_id, width, height):
    if not ENABLE_DISTANCE_ZONE:
        return "DISABLED"
    near = _point_in_zone(box, _camera_zone_points(camera_id, "near"), width, height)
    if near:
        return "NEAR"
    mid = _point_in_zone(box, _camera_zone_points(camera_id, "mid"), width, height)
    if mid:
        return "MID"
    return "FAR"


def _plate_ready_for_ocr(plate):
    bbox = plate.get("bbox") or plate.get("box") or []
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1
    confidence = _safe_float(plate.get("confidence", plate.get("conf", 0.0)))
    return (
        width >= OCR_MIN_PLATE_WIDTH
        and height >= OCR_MIN_PLATE_HEIGHT
        and confidence >= OCR_MIN_PLATE_CONFIDENCE
    )


def _calculate_iou(boxA, boxB):
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return float(inter) / float(max(1, union))


class AdaptiveAIController:
    def __init__(self, base_interval=BASE_AI_INTERVAL, min_interval=MIN_AI_INTERVAL,
                 max_interval=MAX_AI_INTERVAL, target_cpu=TARGET_CPU,
                 high_cpu=HIGH_CPU, low_cpu=LOW_CPU):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.target_cpu = target_cpu
        self.high_cpu = high_cpu
        self.low_cpu = low_cpu
        self.current_interval = base_interval
        self.last_ai_time = 0.0
        self.last_cpu = 0.0
        self.last_adjustment = 0.0

    def update(self, ai_time):
        self.last_ai_time = float(ai_time or 0.0)
        try:
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            cpu = self.last_cpu
        self.last_cpu = float(cpu or 0.0)
        now = time.time()
        if now - self.last_adjustment < AI_ADJUST_COOLDOWN:
            return self.current_interval
        old_interval = self.current_interval
        if self.last_cpu >= self.high_cpu or self.last_ai_time >= 0.50:
            self.current_interval = min(self.current_interval + 0.05, self.max_interval)
        elif self.last_cpu >= self.target_cpu or self.last_ai_time >= 0.35:
            self.current_interval = min(self.current_interval + 0.02, self.max_interval)
        elif self.last_cpu <= self.low_cpu and self.last_ai_time <= 0.20:
            self.current_interval = max(self.current_interval - 0.02, self.min_interval)
        self.current_interval = max(self.min_interval, min(self.current_interval, self.max_interval))
        self.last_adjustment = now
        if abs(self.current_interval - old_interval) >= 0.01:
            print(f"[ADAPTIVE AI] CPU={self.last_cpu:.1f}% | YOLO={self.last_ai_time * 1000:.0f}ms | interval={self.current_interval:.2f}s")
        return self.current_interval

    def get_interval(self):
        return self.current_interval

    def get_status(self):
        return {"cpu": round(self.last_cpu, 1), "ai_time_ms": round(self.last_ai_time * 1000, 1), "interval": round(self.current_interval, 2)}


# SERVICE

class StreamAIService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("=" * 75)
        print("[AI STREAM] Inisialisasi pipeline Person + Vehicle + Plate + OCR")
        print("=" * 75)

        if not os.path.isfile(PERSON_MODEL_PATH):
            raise FileNotFoundError(f"Model person/vehicle tidak ditemukan: {PERSON_MODEL_PATH}")
        if not os.path.isfile(PLATE_MODEL_PATH):
            raise FileNotFoundError(f"Model plate tidak ditemukan: {PLATE_MODEL_PATH}")

        print(f"[AI STREAM] YOLO    : {PERSON_MODEL_PATH}")
        self.yolo_person = YOLO(PERSON_MODEL_PATH)

        print(f"[AI STREAM] Plate   : {PLATE_MODEL_PATH}")
        self.plate_detector = PlateDetector(
            model_path=PLATE_MODEL_PATH, confidence=PLATE_CONFIDENCE,
            imgsz=PLATE_IMGSZ, device="cpu", max_det=PLATE_MAX_DET, iou=PLATE_IOU,
            min_width=PLATE_MIN_WIDTH, min_height=PLATE_MIN_HEIGHT,
            min_aspect_ratio=1.2, max_aspect_ratio=8.0,
        )

        try:
            self.plate_ocr = PlateOCR(scale=2.5, min_confidence=0.15, verbose=False, fast_mode=True)
            self.plate_ocr.warmup()
            print("[AI STREAM] OCR siap")
        except Exception as exc:
            self.plate_ocr = None
            print(f"[AI STREAM WARNING] OCR tidak tersedia: {exc}")

        self.captured_tracks = {}
        self.saved_plate_events = {}
        self.last_ai_time = 0.0
        self.track_lifecycle = {}

        self.event_session_id = uuid.uuid4().hex[:EVENT_SESSION_ID_LENGTH]

        self.adaptive_ai = AdaptiveAIController()

        self._visual_hashes = {}
        self._plate_text_cache = {}
        self._vehicle_centroids = {}
        self._person_track_cache = {}
        self._person_alias_keys = {}

        self.object_groups = {}
        self._group_counter = 0

        self.frame_queue = collections.deque(maxlen=2)
        self.frame_queue_lock = threading.Lock()
        self.ai_wakeup = threading.Event()
        self.ai_thread = threading.Thread(target=self._ai_loop, name="PlateVision-AI", daemon=True)

        self.latest_results = {"persons": [], "plates": []}
        self.last_results_time = 0.0
        self.last_plate_history = []
        self.last_plate_capture = None

        self.processing_lock = threading.Lock()
        self.ai_lock = threading.Lock()
        self.running = True

        self.display_counter = 0
        self.last_display_fps_calc = time.time()
        self.display_fps = 0.0
        self.ai_cycle_counter = 0
        self.last_perf_log = time.time()

        self.focus_track_id = None
        self.focus_last_seen = 0.0
        self.focus_plate_last_seen = 0.0
        self.focus_info = {"type": "AREA", "box": None, "track_id": None, "lighting": None}
        self.latest_raw_plate_detections = []
        self.camera_states = {}
        self.last_display_time = {}
        self.plate_detection_cache = {}
        self.ocr_cache = {}
        self.saved_event_meta = {}

        self._track_last_event = {}

        load_all_zones_from_db()

        self.ai_thread.start()

        print("[AI STREAM] Vehicle classes : person/car/motorcycle/bus/truck")
        print(f"[AI STREAM] Distance zone  : {'ON' if ENABLE_DISTANCE_ZONE else 'OFF'}")
        print("[AI STREAM] Plate ROI       : NEAR zone vehicle")
        print("[AI STREAM] PlateTracker    : multi-frame voting")
        print("[AI STREAM] Event identity  : object_group_id (stable across track_id switch)")
        print("[AI STREAM] Best frame only : confidence improvement > 5%")
        print(f"[AI STREAM] Plate validator : {'AKTIF (kode wilayah)' if HAS_PLATE_VALIDATOR else 'FALLBACK REGEX'}")
        print("=" * 75)

    def get_performance_status(self):
        status = self.adaptive_ai.get_status()
        status["display_fps"] = DISPLAY_FPS
        status["max_plate_roi"] = MAX_PLATE_ROI
        status["distance_zone_enabled"] = ENABLE_DISTANCE_ZONE
        status["camera_zone_count"] = len(CAMERA_ZONE_CONFIG)
        status["ocr_min_plate_width"] = OCR_MIN_PLATE_WIDTH
        status["ocr_min_plate_height"] = OCR_MIN_PLATE_HEIGHT
        return status

    def _camera_state(self, camera_id):
        camera_id = _normalize_camera_id(camera_id)
        state = self.camera_states.get(camera_id)
        if state is None:
            state = {
                "last_ai_time": 0.0,
                "last_results_time": 0.0,
                "last_results": {"persons": [], "plates": []},
                "person_tracker": sv.ByteTrack(
                    track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD,
                    lost_track_buffer=TRACKER_LOST_BUFFER,
                    minimum_matching_threshold=TRACKER_MATCH_THRESHOLD,
                    frame_rate=25,
                ),
                "plate_tracker": PlateTracker(
                    iou_threshold=PLATE_TRACKER_IOU,
                    max_frame_gap=PLATE_TRACKER_FRAME_GAP,
                    ocr_every_n_matches=PLATE_TRACKER_OCR_EVERY,
                    min_final_confidence=PLATE_TRACKER_MIN_CONFIDENCE,
                    min_consistent_reads=2,
                    single_read_ocr_confidence=0.72,
                    single_read_detection_confidence=0.55,
                    max_history=PLATE_TRACKER_MAX_HISTORY,
                ),
            }
            self.camera_states[camera_id] = state
        return state

    # OBJECT GROUPING
    def _assign_object_groups(self, detections, camera_id, current_time=None, frame=None):
        """Object Grouping: hubungkan track_id temporer dengan object_group_id stabil."""
        camera_id = _normalize_camera_id(camera_id)
        now = float(current_time if current_time is not None else time.time())
        assigned_in_this_frame = set()

        for d in detections:
            bbox = d.get("box", [0, 0, 0, 0])
            if not bbox or len(bbox) != 4:
                continue

            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            area = bw * bh

            track_id = int(d.get("track_id", -1)) if d.get("track_id") is not None else -1
            conf = _safe_float(d.get("conf", d.get("confidence", 0.0)))
            object_type = d.get("object_type", "vehicle")
            vehicle_type = d.get("vehicle_type", "unknown")
            zone = d.get("distance_zone", "UNKNOWN")

            v_hash = None
            if object_type == "person" and frame is not None:
                try:
                    h, w = frame.shape[:2]
                    px1 = max(0, min(w - 1, bbox[0]))
                    py1 = max(0, min(h - 1, bbox[1]))
                    px2 = max(0, min(w, bbox[2]))
                    py2 = max(0, min(h, bbox[3]))
                    if px2 > px1 and py2 > py1:
                        crop = frame[py1:py2, px1:px2]
                        v_hash = self._visual_hash(crop)
                except Exception:
                    v_hash = None

            matched_group = None
            best_score = -1.0
            best_dist = 0.0
            best_iou = 0.0
            is_track_switch = False
            old_track_id = None

            max_dist_allowed = (
                OBJECT_GROUP_PERSON_MAX_DIST
                if object_type == "person"
                else OBJECT_GROUP_MAX_CENTROID_DISTANCE
            )

            for gid, group in self.object_groups.items():
                if gid in assigned_in_this_frame:
                    continue
                if group.get("camera_id") != camera_id:
                    continue
                if group.get("object_type") != object_type:
                    continue
                if now - group.get("last_seen", 0) > OBJECT_GROUP_MAX_GAP:
                    continue

                g_last_track = group.get("last_track_id", -1)
                g_tracks_seen = group.get("tracks_seen", set())
                g_centroid = group.get("last_centroid", (0, 0))
                g_bbox = group.get("last_bbox", [0, 0, 0, 0])
                g_area = max(1, (g_bbox[2] - g_bbox[0]) * (g_bbox[3] - g_bbox[1]))

                dist = ((cx - g_centroid[0]) ** 2 + (cy - g_centroid[1]) ** 2) ** 0.5
                iou_val = _calculate_iou(bbox, g_bbox)
                area_ratio = abs(area - g_area) / max(1, max(area, g_area))

                if track_id >= 0 and (g_last_track == track_id or track_id in g_tracks_seen):
                    if dist <= max_dist_allowed * 1.5 or iou_val > 0.0:
                        score = 1000.0 - dist
                        if score > best_score:
                            best_score = score
                            matched_group = group
                            best_dist = dist
                            best_iou = iou_val
                            is_track_switch = False
                            old_track_id = g_last_track
                        continue

                if object_type == "vehicle":
                    two_wheelers = {"motorcycle", "bicycle"}
                    g_two = group.get("vehicle_type") in two_wheelers
                    v_two = vehicle_type in two_wheelers
                    if group.get("vehicle_type") != "unknown" and vehicle_type != "unknown":
                        if g_two != v_two:
                            continue

                if dist > max_dist_allowed:
                    continue
                if area_ratio > OBJECT_GROUP_AREA_TOLERANCE:
                    continue

                hdist = None
                if object_type == "person" and v_hash and group.get("v_hash"):
                    hdist = self._hamming_distance(v_hash, group["v_hash"])
                    if hdist > PERSON_HAMMING_THRESHOLD * 2:
                        continue

                # Jika visual hash cocok kuat untuk person, toleransi IoU nol akibat jeda frame
                is_visual_strong = (object_type == "person" and hdist is not None and hdist <= PERSON_HAMMING_THRESHOLD)
                if not is_visual_strong:
                    if iou_val < OBJECT_GROUP_MIN_IOU and dist > (max_dist_allowed * 0.65):
                        continue

                score = (iou_val * 100.0) + (100.0 - (dist / max(1.0, max_dist_allowed)) * 50.0)
                if score > best_score:
                    best_score = score
                    matched_group = group
                    best_dist = dist
                    best_iou = iou_val
                    old_track_id = g_last_track
                    is_track_switch = (
                        g_last_track != track_id
                        and g_last_track >= 0
                        and track_id >= 0
                    )

            if matched_group is not None:
                group_id = matched_group["object_group_id"]
                assigned_in_this_frame.add(group_id)

                if is_track_switch and DEBUG_PERFORMANCE:
                    print(
                        f"[TRACK ID SWITCH -> SAME GROUP] "
                        f"camera={camera_id} old_track_id={old_track_id} "
                        f"new_track_id={track_id} object_group_id={group_id} "
                        f"distance={best_dist:.1f} iou={best_iou:.2f}"
                    )

                matched_group["last_seen"] = now
                matched_group["last_bbox"] = bbox
                matched_group["last_centroid"] = (cx, cy)
                matched_group["last_track_id"] = track_id
                matched_group["last_zone"] = zone
                if track_id >= 0:
                    matched_group.setdefault("tracks_seen", set()).add(track_id)
                if conf > matched_group.get("best_confidence", 0.0):
                    matched_group["best_confidence"] = conf
                    matched_group["best_bbox"] = bbox
                    matched_group["best_track_id"] = track_id
                if vehicle_type != "unknown":
                    matched_group["vehicle_type"] = vehicle_type
                if v_hash:
                    matched_group["v_hash"] = v_hash

            else:
                self._group_counter += 1
                group_id = f"{object_type}_group_{self._group_counter:03d}"
                assigned_in_this_frame.add(group_id)

                new_group = {
                    "object_group_id": group_id,
                    "camera_id": camera_id,
                    "object_type": object_type,
                    "vehicle_type": vehicle_type,
                    "last_zone": zone,
                    "last_seen": now,
                    "last_bbox": bbox,
                    "last_centroid": (cx, cy),
                    "last_track_id": track_id,
                    "best_confidence": conf,
                    "best_bbox": bbox,
                    "best_track_id": track_id,
                    "tracks_seen": {track_id} if track_id >= 0 else set(),
                    "v_hash": v_hash,
                }
                self.object_groups[group_id] = new_group

            d["object_group_id"] = group_id
            d["track_id"] = track_id
            d["confidence"] = conf
            d["conf"] = conf
            d["v_hash"] = v_hash

    # DETECTION
    def _detect_vehicles(self, frame, camera_id=1):
        camera_id = _normalize_camera_id(camera_id)
        detections = []
        try:
            h, w = frame.shape[:2]
            result = self.yolo_person(
                frame, classes=VEHICLE_CLASSES, imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONFIDENCE, verbose=False
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                tracked = sv.Detections.empty()
            else:
                keep_indices = []
                for i, box_tensor in enumerate(boxes.xyxy):
                    bbox = box_tensor.cpu().numpy().astype(int).tolist()
                    cls_id = int(boxes.cls[i].item())
                    x1, y1, x2, y2 = bbox
                    bw = max(0, x2 - x1); bh = max(0, y2 - y1)
                    zone = _get_distance_zone(bbox, camera_id, w, h)

                    if ENABLE_DISTANCE_ZONE and zone == "FAR":
                        continue
                    if cls_id == 0:
                        if bw < MIN_PERSON_WIDTH or bh < MIN_PERSON_HEIGHT:
                            continue
                    elif cls_id in VEHICLE_TYPES:
                        if bw < MIN_VEHICLE_WIDTH or bh < MIN_VEHICLE_HEIGHT:
                            continue
                        aspect = bw / max(1, bh)
                        min_aspect = FAKE_MOTORCYCLE_ASPECT_MIN if cls_id == 3 else FAKE_VEHICLE_ASPECT_MIN
                        if aspect < min_aspect:
                            continue
                    else:
                        continue
                    keep_indices.append(i)

                if keep_indices:
                    filtered_result = result[keep_indices]
                    tracked = self._camera_state(camera_id)["person_tracker"].update_with_detections(
                        sv.Detections.from_ultralytics(filtered_result)
                    )
                else:
                    tracked = sv.Detections.empty()

            for i in range(len(tracked)):
                bbox = tracked.xyxy[i].astype(int).tolist()
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                conf = _safe_float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                zone = _get_distance_zone(bbox, camera_id, w, h)
                if ENABLE_DISTANCE_ZONE and zone == "FAR":
                    continue
                detections.append({
                    "box": bbox, "track_id": track_id, "conf": conf, "cls": cls_id,
                    "distance_zone": zone,
                    "object_type": ("vehicle" if cls_id in VEHICLE_TYPES else "person"),
                    "vehicle_type": VEHICLE_TYPES.get(cls_id, "unknown"),
                })

            cam_state = self._camera_state(camera_id)
            prev_objects = cam_state.setdefault("prev_tracked_objects", {})
            current_objects = {}
            now = time.time()

            for d in detections:
                tid = d.get("track_id")
                if tid is None or tid < 0:
                    continue
                current_objects[tid] = {
                    "box": d["box"],
                    "conf": d["conf"],
                    "cls": d["cls"],
                    "time": now
                }

                if tid not in prev_objects:
                    best_old_id = None
                    best_iou = 0.0
                    for old_tid, old_data in prev_objects.items():
                        if old_tid in current_objects:
                            continue
                        same_type = (old_data["cls"] == d["cls"]) or (
                            old_data["cls"] in VEHICLE_TYPES and d["cls"] in VEHICLE_TYPES
                        )
                        if not same_type:
                            continue
                        iou_val = _calculate_iou(old_data["box"], d["box"])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_old_id = old_tid

                    if best_old_id is not None and best_iou >= 0.25 and DEBUG_PERFORMANCE:
                        print(
                            f"[TRACK SWITCH] camera={camera_id} "
                            f"old_track_id={best_old_id} new_track_id={tid} "
                            f"iou={best_iou:.2f}"
                        )

            updated_prev = {
                tid: info for tid, info in prev_objects.items()
                if now - info["time"] < 3.0 and tid not in current_objects
            }
            updated_prev.update(current_objects)
            cam_state["prev_tracked_objects"] = updated_prev

        except Exception as exc:
            print(f"[AI STREAM ERROR] Vehicle detection failed: {exc}")
        return detections

    def _detect_plates_in_vehicles(self, frame, vehicle_dets, camera_id=None, current_time=None):
        h, w = frame.shape[:2]
        candidates = [
            d for d in vehicle_dets
            if d.get("cls") in (2, 3, 5, 7)
            and (not ENABLE_DISTANCE_ZONE or d.get("distance_zone") == "NEAR")
        ]
        candidates.sort(key=lambda d: (
            1 if d.get("distance_zone") == "NEAR" else 0,
            max(1, d["box"][2] - d["box"][0]) * max(1, d["box"][3] - d["box"][1]),
            _safe_float(d.get("conf", 0.0)),
        ), reverse=True)
        all_plates = []
        for vehicle in candidates[:MAX_PLATE_ROI]:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle["box"]]
            vx1 = max(0, min(vx1, w - 1)); vy1 = max(0, min(vy1, h - 1))
            vx2 = max(0, min(vx2, w)); vy2 = max(0, min(vy2, h))
            vw = vx2 - vx1; vh = vy2 - vy1
            if vw < PLATE_VEHICLE_MIN_WIDTH or vh < PLATE_VEHICLE_MIN_HEIGHT:
                continue
            if vx2 <= vx1 or vy2 <= vy1:
                continue
            pad_x = int(vw * 0.06); pad_y = int(vh * 0.10)
            rx1 = max(0, vx1 - pad_x); ry1 = max(0, vy1 - pad_y)
            rx2 = min(w, vx2 + pad_x); ry2 = min(h, vy2 + pad_y)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            try:
                local_plates = self.plate_detector.detect(crop) or []
            except Exception as exc:
                print(f"[AI STREAM ERROR] Plate ROI failed: {exc}")
                continue
            for plate in local_plates:
                lb = plate.get("bbox", [])
                if len(lb) != 4:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in lb]
                bx1 += rx1; by1 += ry1; bx2 += rx1; by2 += ry1
                bx1 = max(0, min(bx1, w - 1)); by1 = max(0, min(by1, h - 1))
                bx2 = max(0, min(bx2, w)); by2 = max(0, min(by2, h))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                plate_width = bx2 - bx1; plate_height = by2 - by1
                if plate_width < PLATE_MIN_WIDTH or plate_height < PLATE_MIN_HEIGHT:
                    continue
                plate_crop = frame[by1:by2, bx1:bx2]
                if plate_crop.size == 0:
                    continue
                item = dict(plate)
                item["bbox"] = [bx1, by1, bx2, by2]
                item["box"] = item["bbox"]
                item["crop"] = plate_crop
                item["vehicle_cls"] = vehicle.get("cls", 0)
                item["vehicle_track_id"] = vehicle.get("track_id", -1)
                item["vehicle_object_group_id"] = vehicle.get("object_group_id")
                item["vehicle_box"] = [vx1, vy1, vx2, vy2]
                item["vehicle_crop"] = crop
                item["distance_zone"] = "NEAR"
                item["ocr_ready"] = _plate_ready_for_ocr(item)
                all_plates.append(item)

        all_plates.sort(
            key=lambda x: _safe_float(x.get("confidence", x.get("conf", 0.0))),
            reverse=True
        )
        final = []
        for candidate in all_plates:
            bx = candidate.get("bbox", [])
            if len(bx) != 4:
                continue
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            duplicate = False
            for existing in final:
                eb = existing.get("bbox", [])
                if len(eb) != 4:
                    continue
                ecx = (eb[0] + eb[2]) / 2.0
                ecy = (eb[1] + eb[3]) / 2.0
                if ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5 < 18:
                    duplicate = True
                    break
            if not duplicate:
                final.append(candidate)
        return final

    def _consume_finished_plate(self, frame, vehicle_dets, camera_id):
        """✅ Ambil plate yang sudah finalized, koreksi, simpan raw_text."""
        camera_id = _normalize_camera_id(camera_id)
        plate_tracker = self._camera_state(camera_id)["plate_tracker"]
        finished_list = plate_tracker.consume_finished_captures()
        if not finished_list:
            return []

        processed = []
        now = time.time()
        for finished in finished_list:
            # ✅ Ambil raw_text dari tracker
            raw_text_from_ocr = finished.get("raw_text") or finished.get("text", "")
            text = _normalize_text(_get_value(finished, ["text", "formatted"], ""))

            # ✅ Koreksi & validasi
            corrected_text, is_valid_from_validator, penalty = _correct_plate_text(text)
            if is_valid_from_validator:
                text = corrected_text

            formatted = finished.get("formatted", text) or _format_plate_display(text)
            ocr_conf = _safe_float(finished.get("confidence", 0.0))
            det_conf = _safe_float(finished.get("detection_confidence", 0.0))
            track_id = _get_value(finished, ["track_id", "id"], "unknown")
            crop = finished.get("crop")
            bbox = finished.get("bbox")

            if frame is not None and bbox and len(bbox) == 4:
                original_plate_crop = _prepare_high_quality_capture(
                    frame, bbox, padding=0.20, target_short_side=400,
                    max_scale=3.0, max_long_side=1000, sharpen=True
                )
                if original_plate_crop is not None:
                    crop = original_plate_crop

            valid = _valid_indonesian_plate(text)
            if not valid and det_conf < PLATE_REVIEW_CONFIDENCE:
                continue

            plate_key = f"plate:{camera_id}:{text}" if valid and text else f"plate-track:{camera_id}:{track_id}"
            last_saved = self.saved_plate_events.get(plate_key)
            if last_saved is not None and now - last_saved < PLATE_COOLDOWN:
                continue
            self.saved_plate_events[plate_key] = now

            prefix = "plate" if valid else "plate_review"
            plate_path = save_plate_capture(
                crop, track_id, camera_id, formatted or text or "unknown", prefix=prefix
            )

            finished = dict(finished)
            vehicle_track_id = finished.get("vehicle_track_id")
            if vehicle_track_id is None and bbox and len(bbox) == 4:
                px1, py1, px2, py2 = [float(v) for v in bbox]
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                best_vehicle = None
                best_score = 0.0
                for vehicle in vehicle_dets or []:
                    if vehicle.get("cls") not in VEHICLE_TYPES:
                        continue
                    vb = vehicle.get("box") or []
                    if len(vb) != 4:
                        continue
                    vx1, vy1, vx2, vy2 = [float(v) for v in vb]
                    if not (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2):
                        continue
                    vw = max(1.0, vx2 - vx1)
                    vh = max(1.0, vy2 - vy1)
                    score = 1.0 / (vw * vh)
                    if score > best_score:
                        best_score = score
                        best_vehicle = vehicle
                if best_vehicle is not None:
                    vehicle_track_id = best_vehicle.get("track_id")
                    finished["vehicle_object_group_id"] = best_vehicle.get("object_group_id")

            try:
                vehicle_track_id = int(vehicle_track_id) if vehicle_track_id is not None else None
            except (TypeError, ValueError):
                vehicle_track_id = None

            vehicle_crop = finished.get("vehicle_crop")
            vehicle_object_group_id = finished.get("vehicle_object_group_id")

            finished.update({
                "text": text,
                "formatted": formatted or text,
                "raw_text": raw_text_from_ocr,  # ✅ simpan raw
                "ocr_conf": ocr_conf,
                "conf": det_conf,
                "valid": valid,
                "plate_image_path": plate_path,
                "vehicle_track_id": vehicle_track_id,
                "vehicle_object_group_id": vehicle_object_group_id,
                "vehicle_crop": vehicle_crop,
                "region_code": _extract_region_code(text),
                "correction_penalty": penalty,
            })
            self.last_plate_capture = finished
            self.last_plate_history = list(plate_tracker.history)
            processed.append(finished)

        return processed

    # SAVE EVENTS (ANTI-DUPLIKAT UTAMA)
    @staticmethod
    def _intersection_ratio(inner_box, outer_box):
        ix1 = max(inner_box[0], outer_box[0])
        iy1 = max(inner_box[1], outer_box[1])
        ix2 = min(inner_box[2], outer_box[2])
        iy2 = min(inner_box[3], outer_box[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
        return intersection / area

    def _save_consistent_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        """Simpan event dengan dedup object_group_id + koreksi plat."""
        camera_id = _normalize_camera_id(camera_id)
        camera_orientation = db.get_camera_direction(camera_id)

        vehicles = [item for item in person_dets if item.get("cls") in VEHICLE_TYPES]
        people = [item for item in person_dets if item.get("cls") == 0]

        associated_people = set()

        # VEHICLE EVENTS
        for vehicle in vehicles:
            vehicle_box = vehicle.get("box", [0, 0, 0, 0])

            try:
                vehicle_id = int(vehicle.get("track_id", -1))
            except (TypeError, ValueError):
                vehicle_id = -1

            object_group_id = vehicle.get("object_group_id")

            vehicle_type = (
                vehicle.get("vehicle_type")
                or VEHICLE_TYPES.get(vehicle.get("cls"), "unknown")
            )

            matching_plates = []
            geometric_plates = []
            for plate in plate_dets:
                plate_vehicle_group = plate.get("vehicle_object_group_id")
                plate_vehicle_id = plate.get("vehicle_track_id")

                if plate_vehicle_group is not None and object_group_id is not None:
                    if str(plate_vehicle_group) == str(object_group_id):
                        matching_plates.append(plate)
                        continue

                if plate_vehicle_id is not None:
                    try:
                        if int(plate_vehicle_id) == vehicle_id:
                            matching_plates.append(plate)
                            continue
                    except (TypeError, ValueError):
                        pass

                plate_box = plate.get("bbox") or plate.get("box") or [0, 0, 0, 0]
                if len(plate_box) != 4:
                    continue

                center = plate.get("center") or [
                    (plate_box[0] + plate_box[2]) / 2,
                    (plate_box[1] + plate_box[3]) / 2,
                ]

                if (
                    vehicle_box[0] <= center[0] <= vehicle_box[2]
                    and vehicle_box[1] <= center[1] <= vehicle_box[3]
                ):
                    geometric_plates.append(plate)

            if not matching_plates:
                matching_plates = geometric_plates

            plate = max(
                matching_plates,
                key=lambda item: float(
                    item.get("conf", item.get("confidence", 0)) or 0
                ),
                default=None,
            )

            bx1, by1, bx2, by2 = vehicle_box
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            aspect = bw / bh

            is_motorcycle = (
                vehicle.get("cls") == 3
                or vehicle.get("vehicle_type") == "motorcycle"
            )
            min_aspect = FAKE_MOTORCYCLE_ASPECT_MIN if is_motorcycle else FAKE_VEHICLE_ASPECT_MIN
            if plate is None and aspect < min_aspect:
                continue

            driver = None
            for index, person in enumerate(people):
                person_box = person.get("box", [0, 0, 0, 0])
                inter_ratio = self._intersection_ratio(person_box, vehicle_box)
                pcx = (person_box[0] + person_box[2]) / 2.0
                pcy = (person_box[1] + person_box[3]) / 2.0
                is_inside = (
                    vehicle_box[0] <= pcx <= vehicle_box[2]
                    and vehicle_box[1] <= pcy <= vehicle_box[3]
                )
                if inter_ratio >= 0.20 or is_inside:
                    if driver is None and index not in associated_people:
                        driver = person
                    associated_people.add(index)

            # ✅ Ambil text terkoreksi DAN raw
            plate_text_corrected = _normalize_text((plate or {}).get("text") or "")
            plate_text_raw = (plate or {}).get("raw_text") or plate_text_corrected

            # ✅ EVENT IDENTITY: pakai object_group_id
            if object_group_id:
                event_key = f"vehicle:{camera_id}:{object_group_id}:sess_{self.event_session_id}"
                generation = 1
                tracked = True
            elif vehicle_id >= 0:
                generation = self._get_track_generation(camera_id, vehicle_id, current_time)
                event_key = self._build_event_key(
                    camera_id=camera_id,
                    object_type="vehicle",
                    track_id=vehicle_id,
                    generation=generation,
                )
                tracked = True
            else:
                center = (
                    int((vehicle_box[0] + vehicle_box[2]) / 2),
                    int((vehicle_box[1] + vehicle_box[3]) / 2),
                )
                generation = int(current_time)
                event_key = (
                    f"vehicle:{camera_id}:"
                    f"untracked_{center[0]}_{center[1]}:"
                    f"gen{generation}:sess_{self.event_session_id}"
                )
                tracked = False

            previous_meta = self.saved_event_meta.get(event_key)
            is_new_event = event_key not in self.captured_tracks

            ocr_conf_now = float((plate or {}).get("ocr_conf", (plate or {}).get("confidence", 0.0)) or 0.0)
            prev_ocr_conf = float((previous_meta or {}).get("ocr_conf", 0.0) or 0.0)
            vehicle_conf_now = float(vehicle.get("conf", 0.0) or 0.0)
            prev_vehicle_conf = float((previous_meta or {}).get("vehicle_confidence", 0.0) or 0.0)

            if not is_new_event and previous_meta is not None:
                better_ocr = plate_text_corrected and (
                    not previous_meta.get("plate_text")
                    or ocr_conf_now > prev_ocr_conf + CONFIDENCE_IMPROVEMENT_THRESHOLD
                    or (not previous_meta.get("plate_valid") and (plate or {}).get("valid"))
                )
                better_vehicle = vehicle_conf_now > prev_vehicle_conf + CONFIDENCE_IMPROVEMENT_THRESHOLD
                needs_driver = driver is not None and not previous_meta.get("has_driver")

                if not (better_ocr or better_vehicle or needs_driver):
                    if DEBUG_PERFORMANCE and self.ai_cycle_counter % DEBUG_LOG_SAMPLE_EVERY == 0:
                        print(
                            f"[SKIP NOT BETTER] camera={camera_id} "
                            f"group={object_group_id} vehicle_conf={vehicle_conf_now:.2f} "
                            f"(prev {prev_vehicle_conf:.2f})"
                        )
                    continue

            if not tracked:
                if plate_text_corrected and _valid_indonesian_plate(plate_text_corrected):
                    if self._is_duplicate_plate(camera_id, plate_text_corrected, current_time):
                        self.captured_tracks[event_key] = current_time
                        continue

                if self._is_duplicate_centroid(camera_id, vehicle_box, current_time):
                    self.captured_tracks[event_key] = current_time
                    continue

            vehicle_crop = _prepare_vehicle_capture(frame, vehicle_box)

            if not tracked:
                v_hash = self._visual_hash(vehicle_crop)
                if self._is_duplicate_visual(camera_id, "vehicle", v_hash, current_time):
                    self.captured_tracks[event_key] = current_time
                    continue

            final_direction = camera_orientation

            try:
                result = db.save_detection_event(
                    camera_id=camera_id,
                    plate_number=plate_text_corrected or None,
                    raw_ocr_text=plate_text_raw or None,  # ✅ raw
                    plate_crop=(plate or {}).get("crop"),
                    plate_conf=float(
                        (plate or {}).get("conf", (plate or {}).get("confidence", 0.0)) or 0
                    ),
                    ocr_conf=float((plate or {}).get("ocr_conf", 0.0) or 0),
                    track_id=(vehicle_id if vehicle_id >= 0 else None),
                    object_type="vehicle",
                    vehicle_type=vehicle_type,
                    vehicle_confidence=vehicle_conf_now,
                    vehicle_crop=vehicle_crop,
                    has_driver=driver is not None,
                    driver_track_id=(driver or {}).get("track_id"),
                    direction=final_direction,
                    event_key=event_key,
                )

                self.captured_tracks[event_key] = current_time

                previous_plate = (previous_meta or {}).get("plate_text", "")
                previous_driver = bool((previous_meta or {}).get("has_driver", False))

                self.saved_event_meta[event_key] = {
                    "plate_text": (plate_text_corrected or previous_plate),
                    "plate_valid": bool((plate or {}).get("valid")) or (previous_meta or {}).get("plate_valid", False),
                    "ocr_conf": max(ocr_conf_now, prev_ocr_conf),
                    "has_driver": (driver is not None or previous_driver),
                    "direction": final_direction,
                    "generation": generation,
                    "track_id": vehicle_id,
                    "object_group_id": object_group_id,
                    "vehicle_confidence": max(vehicle_conf_now, prev_vehicle_conf),
                    "vehicle_type": vehicle_type,
                    "updated_at": current_time,
                }

                if DEBUG_PERFORMANCE:
                    region = _extract_region_code(plate_text_corrected or "")
                    print(
                        f"[DETECTION] camera={camera_id} track_id={vehicle_id} "
                        f"group={object_group_id} gen={generation} "
                        f"type={vehicle_type} conf={vehicle_conf_now:.3f} "
                        f"plate_raw='{plate_text_raw or '-'}' "
                        f"plate_corrected='{plate_text_corrected or '-'}' "
                        f"region={region or '-'} "
                        f"ocr={float((plate or {}).get('ocr_conf', 0) or 0):.3f} "
                        f"driver={driver is not None} "
                        f"event_key={event_key} "
                        f"dup={result.get('duplicate', False)}"
                    )

            except Exception as exc:
                print(f"[AI STREAM ERROR] Save vehicle event failed: {exc}")

        # PERSON EVENTS
        for index, person in enumerate(people):
            if index in associated_people:
                continue

            try:
                track_id = int(person.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1

            object_group_id = person.get("object_group_id")
            person_conf_now = float(person.get("conf", person.get("confidence", 0.0)) or 0.0)
            bbox = person.get("box", [0, 0, 0, 0])
            bx1, by1, bx2, by2 = bbox
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            area = bw * bh

            if object_group_id:
                event_key = f"person:{camera_id}:{object_group_id}:sess_{self.event_session_id}"
                generation = 1
            elif track_id >= 0:
                generation = self._get_track_generation(camera_id, track_id, current_time)
                event_key = self._build_event_key(
                    camera_id=camera_id,
                    object_type="person",
                    track_id=track_id,
                    generation=generation,
                )
            else:
                continue

            # Multi-layer Dedup Evaluation (A -> E)
            is_dup, match_info = self._is_same_person_candidate(
                camera_id, person, current_time, frame=frame
            )

            if is_dup:
                reason = match_info.get("reason", "UNKNOWN")
                old_track_id = match_info.get("old_track_id", -1)
                dist = match_info.get("distance", 0.0)
                area_ratio = match_info.get("area_ratio", 0.0)
                hdist = match_info.get("hamming", -1)
                matched_key = match_info.get("matched_event_key") or event_key
                matched_meta = match_info.get("matched_meta") or self.saved_event_meta.get(matched_key, {})

                print(
                    f"[PERSON DEDUP]\n"
                    f"camera={camera_id}\n"
                    f"track_id={track_id}\n"
                    f"old_track_id={old_track_id}\n"
                    f"group={object_group_id or '-'}\n"
                    f"reason={reason}\n"
                    f"distance={dist:.1f}\n"
                    f"area_ratio={area_ratio:.2f}\n"
                    f"hamming={hdist}\n"
                    f"action=SKIP"
                )

                # Tandai event_key saat ini dan alias ke event_key utama
                self.captured_tracks[event_key] = current_time
                if matched_key != event_key:
                    self.captured_tracks[matched_key] = current_time
                    self._person_alias_keys[event_key] = matched_key

                if track_id >= 0:
                    self._person_track_cache[f"person-track:{camera_id}:{track_id}"] = {
                        "ts": current_time, "text": str(track_id)
                    }

                # Update metadata in-memory agar tracking posisi/waktu terus mutakhir
                if matched_meta:
                    matched_meta["last_seen"] = current_time
                    matched_meta["last_centroid"] = (cx, cy)
                    matched_meta["last_bbox"] = bbox
                    matched_meta["area"] = area
                    if track_id >= 0:
                        matched_meta.setdefault("tracks_seen", set()).add(track_id)
                    if person.get("v_hash"):
                        matched_meta["v_hash"] = person["v_hash"]

                # Peningkatan confidence: UPDATE metadata/crop terbaik di DB jika diperlukan, JANGAN insert baru
                prev_person_conf = float(matched_meta.get("person_confidence", 0.0) or 0.0)
                if person_conf_now > prev_person_conf + CONFIDENCE_IMPROVEMENT_THRESHOLD:
                    crop = _prepare_high_quality_capture(
                        frame, (bx1, by1, bx2, by2),
                        padding=0.20, target_short_side=360,
                        max_scale=2.5, max_long_side=960, sharpen=True,
                    )
                    if crop is not None and crop.size > 0:
                        try:
                            db.save_detection_event(
                                camera_id=camera_id,
                                face_crop=crop,
                                face_conf=person_conf_now,
                                track_id=track_id if track_id >= 0 else None,
                                object_type="person",
                                vehicle_type="unknown",
                                direction=camera_orientation,
                                event_key=matched_key,
                            )
                            matched_meta["person_confidence"] = person_conf_now
                        except Exception as exc:
                            print(f"[AI STREAM WARNING] Update person best crop failed: {exc}")

                continue

            # PERSON BARU -> SAVE
            print(
                f"[PERSON NEW]\n"
                f"camera={camera_id}\n"
                f"track_id={track_id}\n"
                f"group={object_group_id or '-'}\n"
                f"action=SAVE"
            )

            crop = _prepare_high_quality_capture(
                frame, (bx1, by1, bx2, by2),
                padding=0.20, target_short_side=360,
                max_scale=2.5, max_long_side=960, sharpen=True,
            )

            if crop is None or crop.size == 0:
                continue

            final_direction = camera_orientation

            try:
                result = db.save_detection_event(
                    camera_id=camera_id,
                    face_crop=crop,
                    face_conf=person_conf_now,
                    track_id=track_id if track_id >= 0 else None,
                    object_type="person",
                    vehicle_type="unknown",
                    direction=final_direction,
                    event_key=event_key,
                )

                self.captured_tracks[event_key] = current_time

                v_hash = person.get("v_hash") or self._visual_hash(crop)
                if v_hash:
                    self._register_visual_hash(camera_id, "person", v_hash, current_time)

                if track_id >= 0:
                    self._person_track_cache[f"person-track:{camera_id}:{track_id}"] = {
                        "ts": current_time, "text": str(track_id)
                    }

                self.saved_event_meta[event_key] = {
                    "object_type": "person",
                    "camera_id": camera_id,
                    "event_key": event_key,
                    "plate_text": "",
                    "has_driver": False,
                    "direction": final_direction,
                    "generation": generation,
                    "track_id": track_id,
                    "tracks_seen": {track_id} if track_id >= 0 else set(),
                    "object_group_id": object_group_id,
                    "last_centroid": (cx, cy),
                    "last_bbox": bbox,
                    "area": area,
                    "v_hash": v_hash,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "person_confidence": person_conf_now,
                    "face_saved": True,
                    "detection_id": result.get("detection_id"),
                    "updated_at": current_time,
                }

                if DEBUG_PERFORMANCE:
                    print(
                        f"[DETECTION] camera={camera_id} track_id={track_id} "
                        f"group={object_group_id} gen={generation} "
                        f"object_type=person conf={person_conf_now:.3f} "
                        f"event_key={event_key} "
                        f"dup={result.get('duplicate', False)}"
                    )

            except Exception as exc:
                print(f"[AI STREAM ERROR] Save person event failed: {exc}")

    def _save_new_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        return self._save_consistent_events(
            frame, person_dets, plate_dets, camera_id, current_time,
        )

    # PERSON DEDUPLICATION & CANDIDATE EVALUATION
    def _register_visual_hash(self, camera_id, object_type, v_hash, current_time):
        if not v_hash:
            return
        camera_id = _normalize_camera_id(camera_id)
        bucket = "person" if str(object_type).lower() == "person" else "vehicle"
        prefix = f"vhash:{camera_id}:{bucket}:"
        key = f"{prefix}{v_hash}:{int(current_time)}"
        self._visual_hashes[key] = {"ts": current_time, "hash": v_hash}

    def _is_same_person_candidate(self, camera_id, person, current_time, frame=None):
        """
        Evaluasi multi-layer untuk mendeteksi apakah person saat ini merupakan duplikat
        dari person yang sudah pernah disimpan/tercatat sebelumnya.

        Urutan pengecekan (A -> E):
        A. object_group_id sama -> match
        B. track_id sama (atau pernah diasosiasikan) & belum timeout -> match
        C. track_id berbeda tetapi posisi + ukuran bbox sangat mirip -> cek visual hash
        D. visual hash mirip dalam cooldown (PERSON_DEDUP_COOLDOWN)
        E. jika semua indikator berbeda -> False (person baru)
        """
        camera_id = _normalize_camera_id(camera_id)
        now = float(current_time)

        bbox = person.get("box", [0, 0, 0, 0])
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        area = bw * bh

        try:
            track_id = int(person.get("track_id", -1))
        except (TypeError, ValueError):
            track_id = -1

        object_group_id = person.get("object_group_id")

        v_hash = person.get("v_hash")
        if v_hash is None and frame is not None and frame.size > 0:
            try:
                h, w = frame.shape[:2]
                px1 = max(0, min(w - 1, bx1))
                py1 = max(0, min(h - 1, by1))
                px2 = max(0, min(w, bx2))
                py2 = max(0, min(h, by2))
                if px2 > px1 and py2 > py1:
                    crop = frame[py1:py2, px1:px2]
                    v_hash = self._visual_hash(crop)
                    person["v_hash"] = v_hash
            except Exception:
                v_hash = None

        # Resolusi alias event_key
        curr_event_key = (
            f"person:{camera_id}:{object_group_id}:sess_{self.event_session_id}"
            if object_group_id else None
        )
        if curr_event_key and curr_event_key in self._person_alias_keys:
            aliased_key = self._person_alias_keys[curr_event_key]
            aliased_meta = self.saved_event_meta.get(aliased_key)
            if aliased_meta:
                return True, {
                    "reason": "GROUP_ALIAS",
                    "matched_event_key": aliased_key,
                    "old_track_id": aliased_meta.get("track_id", -1),
                    "group_id": aliased_meta.get("object_group_id", object_group_id),
                    "distance": 0.0,
                    "area_ratio": 0.0,
                    "hamming": 0,
                    "matched_meta": aliased_meta,
                }

        # Kumpulkan kandidat event person pada kamera ini
        candidates = []
        max_cooldown = max(PERSON_DEDUP_COOLDOWN, PERSON_TRACK_DEDUP_COOLDOWN)
        for key, meta in self.saved_event_meta.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("object_type") != "person":
                continue
            if meta.get("camera_id") != camera_id:
                continue
            last_ts = float(meta.get("last_seen", meta.get("updated_at", now)))
            if now - last_ts <= max_cooldown:
                candidates.append((key, meta, now - last_ts))

        # A. object_group_id sama -> anggap object yang sama
        if object_group_id:
            for key, meta, age in candidates:
                if meta.get("object_group_id") == object_group_id:
                    if age <= PERSON_DEDUP_COOLDOWN:
                        old_tid = meta.get("track_id", -1)
                        return True, {
                            "reason": "SAME_GROUP",
                            "matched_event_key": key,
                            "old_track_id": old_tid,
                            "group_id": object_group_id,
                            "distance": 0.0,
                            "area_ratio": 0.0,
                            "hamming": 0,
                            "matched_meta": meta,
                        }

        # B. track_id sama dan belum melewati timeout -> anggap object yang sama
        if track_id >= 0:
            for key, meta, age in candidates:
                tracks_seen = meta.get("tracks_seen", set())
                m_track_id = meta.get("track_id", -1)
                if track_id == m_track_id or track_id in tracks_seen:
                    if age <= PERSON_TRACK_DEDUP_COOLDOWN:
                        return True, {
                            "reason": "SAME_TRACK",
                            "matched_event_key": key,
                            "old_track_id": m_track_id,
                            "group_id": meta.get("object_group_id", object_group_id),
                            "distance": 0.0,
                            "area_ratio": 0.0,
                            "hamming": 0,
                            "matched_meta": meta,
                        }

            # Cek juga cache track person
            if self._is_duplicate_person_track(camera_id, track_id, now):
                return True, {
                    "reason": "TRACK_CACHE_MATCH",
                    "matched_event_key": curr_event_key,
                    "old_track_id": track_id,
                    "group_id": object_group_id,
                    "distance": 0.0,
                    "area_ratio": 0.0,
                    "hamming": 0,
                    "matched_meta": None,
                }

        # C. track_id berbeda tetapi posisi + ukuran bbox sangat mirip -> cek visual hash
        best_candidate = None
        best_dist = 9999.0
        best_area_ratio = 1.0
        best_hamming = 999
        best_reason = None

        for key, meta, age in candidates:
            if age > PERSON_DEDUP_COOLDOWN:
                continue

            g_cx, g_cy = meta.get("last_centroid", (0, 0))
            g_bbox = meta.get("last_bbox", [0, 0, 0, 0])
            g_area = meta.get("area", 0)
            if g_area <= 0 and len(g_bbox) == 4:
                g_area = max(1, (g_bbox[2] - g_bbox[0]) * (g_bbox[3] - g_bbox[1]))

            dist = ((cx - g_cx) ** 2 + (cy - g_cy) ** 2) ** 0.5
            area_ratio = abs(area - g_area) / max(1, max(area, g_area))

            hdist = None
            g_v_hash = meta.get("v_hash")
            if v_hash and g_v_hash:
                hdist = self._hamming_distance(v_hash, g_v_hash)

            # C1: Posisi dalam jangkauan wajar + ukuran bbox mirip + visual hash cocok
            if dist <= OBJECT_GROUP_PERSON_MAX_DIST and area_ratio <= PERSON_AREA_TOLERANCE:
                if hdist is not None and hdist <= PERSON_HAMMING_THRESHOLD:
                    if dist < best_dist:
                        best_dist = dist
                        best_area_ratio = area_ratio
                        best_hamming = hdist
                        best_candidate = (key, meta)
                        best_reason = "SPATIAL_VISUAL_MATCH"
                    continue

            # C2: Sangat dekat (proximity / orang berdiri atau berjalan lambat pada titik serupa)
            if dist <= PERSON_PROXIMITY_DIST and area_ratio <= 0.35 and age <= 20.0:
                if hdist is None or hdist <= (PERSON_HAMMING_THRESHOLD + 2):
                    if dist < best_dist:
                        best_dist = dist
                        best_area_ratio = area_ratio
                        best_hamming = hdist if hdist is not None else 0
                        best_candidate = (key, meta)
                        best_reason = "SPATIAL_PROXIMITY"
                    continue

        if best_candidate is not None:
            key, meta = best_candidate
            return True, {
                "reason": best_reason,
                "matched_event_key": key,
                "old_track_id": meta.get("track_id", -1),
                "group_id": meta.get("object_group_id", object_group_id),
                "distance": best_dist,
                "area_ratio": best_area_ratio,
                "hamming": best_hamming,
                "matched_meta": meta,
            }

        # D. visual hash mirip dalam cooldown -> kemungkinan object yang sama
        if v_hash:
            for key, meta, age in candidates:
                if age > PERSON_DEDUP_COOLDOWN:
                    continue
                g_v_hash = meta.get("v_hash")
                if not g_v_hash:
                    continue

                hdist = self._hamming_distance(v_hash, g_v_hash)
                if hdist <= PERSON_HAMMING_THRESHOLD:
                    g_cx, g_cy = meta.get("last_centroid", (0, 0))
                    g_bbox = meta.get("last_bbox", [0, 0, 0, 0])
                    g_area = meta.get("area", 0)
                    if g_area <= 0 and len(g_bbox) == 4:
                        g_area = max(1, (g_bbox[2] - g_bbox[0]) * (g_bbox[3] - g_bbox[1]))
                    dist = ((cx - g_cx) ** 2 + (cy - g_cy) ** 2) ** 0.5
                    area_ratio = abs(area - g_area) / max(1, max(area, g_area))

                    # Pastikan skala tidak bertolak belakang drastis (hindari false positive)
                    if area_ratio <= 0.60:
                        return True, {
                            "reason": "VISUAL_MATCH",
                            "matched_event_key": key,
                            "old_track_id": meta.get("track_id", -1),
                            "group_id": meta.get("object_group_id", object_group_id),
                            "distance": dist,
                            "area_ratio": area_ratio,
                            "hamming": hdist,
                            "matched_meta": meta,
                        }

            # Cek visual hash cache global
            if self._is_duplicate_visual(camera_id, "person", v_hash, now, auto_register=False):
                return True, {
                    "reason": "VISUAL_CACHE_MATCH",
                    "matched_event_key": curr_event_key,
                    "old_track_id": -1,
                    "group_id": object_group_id,
                    "distance": 0.0,
                    "area_ratio": 0.0,
                    "hamming": 0,
                    "matched_meta": None,
                }

        # E. jika semua indikator berbeda -> anggap person baru
        return False, None

    # TRACK GENERATION & EVENT KEY
    def _get_track_generation(self, camera_id, track_id, current_time):
        camera_id = _normalize_camera_id(camera_id)
        try:
            camera_id = int(camera_id)
            track_id = int(track_id)
            current_time = float(current_time)
        except (TypeError, ValueError):
            return 1

        key = (camera_id, track_id)
        state = self.track_lifecycle.get(key)

        if state is None:
            state = {"generation": 1, "last_seen": current_time}
        else:
            last_seen = float(state.get("last_seen", current_time))
            gap = current_time - last_seen
            if gap < 0:
                state["generation"] = int(state.get("generation", 1)) + 1
            elif gap > TRACK_REUSE_GAP:
                state["generation"] = int(state.get("generation", 1)) + 1
            state["last_seen"] = current_time

        self.track_lifecycle[key] = state
        return int(state["generation"])

    def _build_event_key(self, camera_id, object_type, track_id, generation):
        camera_id = _normalize_camera_id(camera_id)
        try:
            camera_id = int(camera_id)
        except (TypeError, ValueError):
            camera_id = str(camera_id)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            track_id = str(track_id)
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 1
        object_type = str(object_type or "unknown").lower().strip()
        return (
            f"{object_type}:{camera_id}:{track_id}:"
            f"gen{generation}:sess_{self.event_session_id}"
        )

    # CLEANUP
    def _cleanup_runtime_state(self, current_time):
        now = float(current_time)
        lifecycle_ttl = max(600.0, TRACK_REUSE_GAP * 4.0)
        event_ttl = 1800.0

        for key, state in list(self.track_lifecycle.items()):
            try:
                last_seen = float(state.get("last_seen", now))
            except (TypeError, ValueError, AttributeError):
                self.track_lifecycle.pop(key, None)
                continue
            if now - last_seen > lifecycle_ttl:
                self.track_lifecycle.pop(key, None)

        for cache_name in ("captured_tracks", "saved_plate_events", "saved_event_meta"):
            cache = getattr(self, cache_name, None)
            if not isinstance(cache, dict):
                continue
            for key, value in list(cache.items()):
                try:
                    if cache_name == "saved_event_meta":
                        stamp = float(value.get("updated_at", now))
                    else:
                        stamp = float(value)
                except (TypeError, ValueError, AttributeError):
                    continue
                if now - stamp > event_ttl:
                    cache.pop(key, None)

        for key, value in list(self._track_last_event.items()):
            try:
                stamp = float(value.get("last_ts", now))
            except (TypeError, ValueError, AttributeError):
                self._track_last_event.pop(key, None)
                continue
            if now - stamp > 7200.0:
                self._track_last_event.pop(key, None)

        for key, value in list(self._visual_hashes.items()):
            try:
                stamp = float(value.get("ts", now))
            except (TypeError, ValueError, AttributeError):
                self._visual_hashes.pop(key, None)
                continue
            if now - stamp > 600.0:
                self._visual_hashes.pop(key, None)

        for key, value in list(self._plate_text_cache.items()):
            try:
                stamp = float(value.get("ts", now))
            except (TypeError, ValueError, AttributeError):
                self._plate_text_cache.pop(key, None)
                continue
            if now - stamp > 600.0:
                self._plate_text_cache.pop(key, None)

        for key, value in list(self._person_track_cache.items()):
            try:
                stamp = float(value.get("ts", now))
            except (TypeError, ValueError, AttributeError):
                self._person_track_cache.pop(key, None)
                continue
            if now - stamp > PERSON_TRACK_DEDUP_COOLDOWN * 2.0:
                self._person_track_cache.pop(key, None)

        if len(self._person_alias_keys) > 500:
            self._person_alias_keys.clear()

        for cam_key, items in list(self._vehicle_centroids.items()):
            self._vehicle_centroids[cam_key] = [
                item for item in items
                if now - item.get("ts", 0) < VEHICLE_CENTROID_DEDUP_COOLDOWN
            ]

        for gid, group in list(self.object_groups.items()):
            try:
                last_seen = float(group.get("last_seen", now))
            except (TypeError, ValueError, AttributeError):
                self.object_groups.pop(gid, None)
                continue
            if now - last_seen > OBJECT_GROUP_MAX_GAP * 30:
                self.object_groups.pop(gid, None)

    # DEDUP VISUAL
    def _visual_hash(self, crop):
        if crop is None or not hasattr(crop, "size") or crop.size == 0:
            return None
        try:
            small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = small.astype(np.float32)
            avg = gray.mean()
            bits = (gray > avg).flatten()
            hex_str = ""
            for i in range(0, len(bits), 4):
                nibble = bits[i:i + 4]
                val = 0
                for b in nibble:
                    val = (val << 1) | int(b)
                hex_str += format(val, "x")
            return hex_str
        except Exception:
            return None

    def _hamming_distance(self, h1, h2):
        if not h1 or not h2 or len(h1) != len(h2):
            return 999
        try:
            b1 = bin(int(h1, 16))[2:].zfill(len(h1) * 4)
            b2 = bin(int(h2, 16))[2:].zfill(len(h2) * 4)
            return sum(c1 != c2 for c1, c2 in zip(b1, b2))
        except Exception:
            return 999

    def _is_duplicate_visual(self, camera_id, object_type, v_hash, current_time, auto_register=True):
        if v_hash is None:
            return False

        object_type = str(object_type or "vehicle").lower().strip()
        if object_type == "person":
            threshold = PERSON_HAMMING_THRESHOLD
            cooldown = PERSON_DEDUP_COOLDOWN
            bucket = "person"
        else:
            threshold = VISUAL_HAMMING_THRESHOLD
            cooldown = VISUAL_DEDUP_COOLDOWN
            bucket = "vehicle"

        camera_id = _normalize_camera_id(camera_id)
        prefix = f"vhash:{camera_id}:{bucket}:"

        for key, val in list(self._visual_hashes.items()):
            if not key.startswith(prefix):
                continue
            stored_hash = val.get("hash")
            age = current_time - val.get("ts", 0)
            if age > cooldown:
                continue
            dist = self._hamming_distance(v_hash, stored_hash)
            if dist <= threshold:
                return True

        if auto_register:
            key = f"{prefix}{v_hash}:{int(current_time)}"
            self._visual_hashes[key] = {"ts": current_time, "hash": v_hash}
        return False

    def _is_duplicate_plate(self, camera_id, plate_text, current_time):
        if not plate_text:
            return False
        camera_id = _normalize_camera_id(camera_id)
        key = f"plate-text:{camera_id}:{plate_text}"
        last = self._plate_text_cache.get(key)
        if last is not None:
            age = current_time - last.get("ts", 0)
            if age < PLATE_TEXT_DEDUP_COOLDOWN:
                return True
        self._plate_text_cache[key] = {"ts": current_time, "text": plate_text}
        return False

    def _is_duplicate_person_track(self, camera_id, track_id, current_time):
        if track_id is None or track_id < 0:
            return False
        camera_id = _normalize_camera_id(camera_id)
        key = f"person-track:{camera_id}:{track_id}"
        last = self._person_track_cache.get(key)
        if last is not None:
            age = current_time - last.get("ts", 0)
            if age < PERSON_TRACK_DEDUP_COOLDOWN:
                return True
        self._person_track_cache[key] = {"ts": current_time, "text": str(track_id)}
        return False

    def _bbox_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _is_duplicate_centroid(self, camera_id, bbox, current_time):
        camera_id = _normalize_camera_id(camera_id)
        cx, cy = self._bbox_centroid(bbox)
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area = bw * bh

        cache = self._vehicle_centroids.get(camera_id, [])
        cache = [
            item for item in cache
            if current_time - item["ts"] < VEHICLE_CENTROID_DEDUP_COOLDOWN
        ]

        for item in cache:
            pcx, pcy = item["centroid"]
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < VEHICLE_CENTROID_DEDUP_DISTANCE:
                prev_area = item["area"]
                max_area = max(area, prev_area)
                ratio = abs(area - prev_area) / max(1, max_area)
                if ratio < VEHICLE_CENTROID_DEDUP_AREA_RATIO:
                    return True

        cache.append({"centroid": (cx, cy), "area": area, "ts": current_time})
        if len(cache) > VEHICLE_CENTROID_MAX_HISTORY:
            cache = cache[-VEHICLE_CENTROID_MAX_HISTORY:]
        self._vehicle_centroids[camera_id] = cache
        return False

    # DYNAMIC FOCUS
    def _select_dynamic_focus(self, frame, vehicle_dets, plate_raw, now):
        h, w = frame.shape[:2]
        vehicles = [
            d for d in vehicle_dets
            if d.get("cls") in (2, 3, 5, 7) and d.get("track_id", -1) >= 0
        ]
        if ENABLE_DISTANCE_ZONE:
            near_vehicles = [d for d in vehicles if d.get("distance_zone") == "NEAR"]
            if near_vehicles:
                vehicles = near_vehicles
        target_type = "VEHICLE"
        if not vehicles:
            people = [
                d for d in vehicle_dets
                if d.get("cls") == 0 and d.get("track_id", -1) >= 0
            ]
            if ENABLE_DISTANCE_ZONE:
                near_people = [d for d in people if d.get("distance_zone") == "NEAR"]
                vehicles = near_people or people
            else:
                vehicles = people
            target_type = "PERSON"
        by_id = {int(d["track_id"]): d for d in vehicles}
        target = by_id.get(self.focus_track_id) if self.focus_track_id is not None else None
        if target is None and self.focus_track_id is not None and now - self.focus_last_seen < FOCUS_TARGET_HOLD_SECONDS:
            old = self.focus_info.get("box")
            if old:
                m = _lighting_metrics(frame, old)
                self.focus_info["lighting"] = m
                return
        if target is None and vehicles:
            target = max(
                vehicles,
                key=lambda d: max(1, d["box"][2] - d["box"][0]) * max(1, d["box"][3] - d["box"][1])
            )
            self.focus_track_id = int(target["track_id"])
        if target is None:
            self.focus_track_id = None
            box = [int(w * .20), int(h * .42), int(w * .88), int(h * .96)]
            self.focus_info = {
                "type": "AREA", "box": box, "track_id": None,
                "lighting": _lighting_metrics(frame, box)
            }
            return
        self.focus_last_seen = now
        tid = int(target["track_id"])
        focus_type = target_type
        box = _clamp_box(target["box"], w, h, FOCUS_PADDING_VEHICLE)
        matches = [
            p for p in plate_raw
            if int(p.get("vehicle_track_id", -999)) == tid and len(p.get("bbox", [])) == 4
        ]
        if matches:
            best = max(matches, key=lambda p: _safe_float(p.get("confidence", p.get("conf", 0))))
            box = _clamp_box(best["bbox"], w, h, FOCUS_PADDING_PLATE)
            focus_type = 'PLATE'
            self.focus_plate_last_seen = now
        elif self.focus_info.get("type") == "PLATE" and now - self.focus_plate_last_seen < FOCUS_TARGET_HOLD_SECONDS:
            box = self.focus_info.get("box") or box
            focus_type = "PLATE"
        self.focus_info = {
            "type": focus_type, "box": box, "track_id": tid,
            "lighting": _lighting_metrics(frame, box)
        }

    def _draw_dynamic_focus(self, frame):
        info = self.focus_info or {}
        box = info.get('box')
        if not box:
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        typ = info.get('type', 'AREA')
        tid = info.get('track_id')
        color = (255, 220, 0) if typ == 'VEHICLE' else ((0, 215, 255) if typ == 'PLATE' else (255, 200, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{typ} FOCUS" + (f" ID:{tid}" if tid is not None else '')
        cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        h, w = frame.shape[:2]
        camera = (w // 2, max(8, int(h * .05)))
        target = _center(box)
        cv2.line(frame, camera, target, color, 2, cv2.LINE_AA)
        cv2.circle(frame, target, 4, color, -1)
        m = info.get('lighting') or {}
        text = f"LIGHT {m.get('status', '-')} | Score {m.get('score', 0):.0f}% | Bright {m.get('brightness', 0):.0f}"
        cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2, cv2.LINE_AA)

    # AI PIPELINE
    def _run_ai_pipeline(self, frame, camera_id):
        camera_id = _normalize_camera_id(camera_id)
        camera_state = self._camera_state(camera_id)
        now = time.time()
        ai_pipeline_start = time.perf_counter()
        person_dets = []
        plate_dets = []
        plate_raw = []
        try:
            h0, w0 = frame.shape[:2]
            base_box = self.focus_info.get("box") or [
                int(w0 * 0.20), int(h0 * 0.42), int(w0 * 0.88), int(h0 * 0.96)
            ]
            base_light = _lighting_metrics(frame, base_box)
            ai_frame = _enhance_for_lighting(frame, base_light)
            person_dets = self._detect_vehicles(ai_frame, camera_id) or []

            # ✅ Panggil object grouping SEBELUM plate detection
            try:
                self._assign_object_groups(
                    person_dets, camera_id, current_time=now, frame=ai_frame
                )
            except Exception as exc:
                print(f"[AI STREAM WARNING] Object grouping failed: {exc}")

            plate_dets = []
            plate_raw = []
            try:
                plate_raw = self._detect_plates_in_vehicles(
                    ai_frame, person_dets, camera_id=camera_id, current_time=now
                ) or []
                self.latest_raw_plate_detections = plate_raw
                self._select_dynamic_focus(frame, person_dets, plate_raw, now)
                plate_tracker = camera_state["plate_tracker"]
                if self.plate_ocr is not None:
                    try:
                        ocr_candidates = [p for p in plate_raw if p.get("ocr_ready", False)]
                        active_tracks = plate_tracker.update(ocr_candidates, ai_frame, self.plate_ocr) or []
                    except Exception as exc:
                        print(f"[AI STREAM WARNING] PlateTracker/OCR failed: {exc}")
                        active_tracks = []
                    for track in active_tracks:
                        result = _track_to_result(track)
                        if result is not None:
                            plate_dets.append(result)
                    finished_items = self._consume_finished_plate(frame, person_dets, camera_id)
                    if finished_items:
                        plate_dets.extend(finished_items)
                else:
                    for item in plate_raw:
                        bbox = item.get("bbox") or item.get("box")
                        if not bbox or len(bbox) != 4:
                            continue
                        conf = _safe_float(item.get("confidence", item.get("conf", 0.0)))
                        plate_dets.append({
                            **item,
                            "id": int(item.get("vehicle_track_id", -1)),
                            "track_id": int(item.get("vehicle_track_id", -1)),
                            "box": bbox, "bbox": bbox, "conf": conf,
                            "detection_confidence": conf,
                            "text": "", "formatted": "", "raw_text": "",
                            "ocr_conf": 0.0, "confidence": 0.0,
                            "valid": False, "votes": 0, "total_reads": 0,
                            "distance_zone": item.get("distance_zone", "NEAR"),
                            "ocr_ready": bool(item.get("ocr_ready", False)),
                        })
            except Exception as exc:
                print(f"[AI STREAM ERROR] Plate pipeline failed: {exc}")

            camera_state["last_results"] = {"persons": person_dets, "plates": plate_dets}
            camera_state["last_results_time"] = now

            self._save_consistent_events(frame, person_dets, plate_dets, camera_id, now)
            self._cleanup_runtime_state(now)

            self.ai_cycle_counter += 1
        except Exception as exc:
            print(f"[AI STREAM ERROR] Background AI cycle failed: {exc}")
        finally:
            elapsed = time.perf_counter() - ai_pipeline_start
            self.last_ai_time = elapsed
            self.adaptive_ai.update(elapsed)
            if DEBUG_PERFORMANCE and self.ai_cycle_counter % DEBUG_LOG_SAMPLE_EVERY == 0:
                print(
                    f"[AI PERF] camera={camera_id} elapsed={elapsed * 1000:.0f}ms "
                    f"groups={len(self.object_groups)} next_interval={self.adaptive_ai.get_interval():.2f}s"
                )

    def _ai_loop(self):
        print("[AI THREAD] Background AI worker started")
        while self.running:
            item = None
            self.ai_wakeup.wait(timeout=0.05)
            self.ai_wakeup.clear()
            if not self.running:
                break
            with self.frame_queue_lock:
                if self.frame_queue:
                    item = self.frame_queue.pop()
                    self.frame_queue.clear()
            if item is None:
                continue
            frame, camera_id = item
            camera_id = _normalize_camera_id(camera_id)
            camera_state = self._camera_state(camera_id)
            now = time.time()
            interval = self.adaptive_ai.get_interval()
            remaining = interval - (now - camera_state["last_ai_time"])
            if remaining > 0:
                time.sleep(min(remaining, 0.05))
                continue
            camera_state["last_ai_time"] = now
            self._run_ai_pipeline(frame, camera_id)
        print("[AI THREAD] Background AI worker stopped")

    # PUBLIC API
    def process_frame(self, frame, draw_bbox=True, camera_id=1):
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return frame
        camera_id = _normalize_camera_id(camera_id)
        try:
            with self.frame_queue_lock:
                self.frame_queue.append((frame.copy(), camera_id))
                while len(self.frame_queue) > 1:
                    self.frame_queue.popleft()
            self.ai_wakeup.set()
        except Exception as exc:
            if DEBUG_PERFORMANCE:
                print(f"[AI QUEUE WARNING] {exc}")
        camera_state = self._camera_state(camera_id)
        if not draw_bbox:
            return self._display_frame(frame, camera_id)
        output = frame.copy()
        self._draw_detection_zones(output, camera_id)
        self._draw_clean_bboxes(output, camera_state["last_results"])
        self._draw_dynamic_focus(output)
        return self._display_frame(output, camera_id)

    def render_frame(self, frame, camera_id=1):
        """Render hasil deteksi terakhir ke frame tanpa memasukkannya ke antrean AI."""
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return frame
        camera_id = _normalize_camera_id(camera_id)
        camera_state = self._camera_state(camera_id)
        output = frame.copy()
        self._draw_detection_zones(output, camera_id)
        self._draw_clean_bboxes(output, camera_state["last_results"])
        self._draw_dynamic_focus(output)
        return self._display_frame(output, camera_id)

    def stop(self):
        self.running = False
        self.ai_wakeup.set()
        thread = getattr(self, "ai_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _display_frame(self, frame, camera_id):
        self.last_display_time[camera_id] = time.time()
        return frame

    # DRAWING
    def _draw_detection_zones(self, frame, camera_id):
        if not ENABLE_DISTANCE_ZONE:
            return
        camera_id = _normalize_camera_id(camera_id)
        h, w = frame.shape[:2]
        mid = _normalized_polygon_to_pixels(_camera_zone_points(camera_id, "mid"), w, h)
        near = _normalized_polygon_to_pixels(_camera_zone_points(camera_id, "near"), w, h)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [mid], (80, 80, 80))
        cv2.fillPoly(overlay, [near], (120, 120, 120))
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
        cv2.polylines(frame, [mid], True, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.polylines(frame, [near], True, (0, 255, 120), 2, cv2.LINE_AA)
        camera_label = f"CAM: {_camera_zone_name(camera_id)}"
        cv2.putText(frame, camera_label, (10, max(24, int(h * 0.07))), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        if len(mid) > 0:
            mx, my = mid[0]
            cv2.putText(frame, "MID: PERSON + VEHICLE", (max(5, mx), max(22, my - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 180, 0), 2, cv2.LINE_AA)
        if len(near) > 0:
            nx, ny = near[0]
            cv2.putText(frame, "NEAR: PLATE + OCR", (max(5, nx), max(22, ny - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2, cv2.LINE_AA)

    def _draw_clean_bboxes(self, frame, results):
        for obj in results.get("persons", []):
            x1, y1, x2, y2 = [int(v) for v in obj["box"]]
            cls_id = int(obj.get("cls", 0))
            conf = _safe_float(obj.get("conf", 0.0))
            track_id = int(obj.get("track_id", -1))
            group_id = obj.get("object_group_id", "-")
            zone = obj.get("distance_zone", "")
            zone_label = f" {zone}" if zone else ""
            if cls_id == 0:
                label = f"Orang ID:{track_id} G:{group_id} {conf:.0%}{zone_label}"; color = (0, 165, 255)
            elif cls_id == 2:
                label = f"Mobil ID:{track_id} G:{group_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 3:
                label = f"Motor ID:{track_id} G:{group_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 5:
                label = f"Bus ID:{track_id} G:{group_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 7:
                label = f"Truk ID:{track_id} G:{group_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            else:
                label = f"Objek {conf:.0%}{zone_label}"; color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            by = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        for plate in results.get("plates", []):
            bbox = plate.get("box", plate.get("bbox"))
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            text = plate.get("text", "")
            ocr_conf = _safe_float(plate.get("ocr_conf", 0.0))
            p_conf = _safe_float(plate.get("conf", 0.0))
            valid = bool(plate.get("valid", False))
            votes = int(plate.get("votes", 0) or 0)
            region = plate.get("region_code") or "-"
            if valid and text:
                label = f"{text} OCR:{ocr_conf:.0%} V:{votes} [{region}]"; color = (0, 230, 118)
            elif text:
                label = f"Review {text} {ocr_conf:.0%}"; color = (0, 215, 255)
            else:
                ocr_ready = bool(plate.get("ocr_ready", False))
                readiness = " OCR-READY" if ocr_ready else " TERLALU KECIL"
                label = f"Plat YOLO:{p_conf:.0%}{readiness}"; color = (0, 215, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            by = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 12, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)