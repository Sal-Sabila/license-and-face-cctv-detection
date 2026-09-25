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

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker
import db


# ============================================================
# PATH PROJECT
# ============================================================

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
    "license-plate-finetune-v2n.pt",
)

CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_CAPTURE_DIR = os.path.join(CAPTURE_DIR, "plates")

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

VEHICLE_CLASSES = [0, 2, 3, 5, 7]
VEHICLE_CONFIDENCE = 0.45
VEHICLE_IMGSZ = 480

# ============================================================
# DISTANCE / DETECTION ZONES
# ============================================================
ENABLE_DISTANCE_ZONE = True

ZONE_OVERLAP_FALLBACK_RATIO = 0.45

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

_ZONE_MID = [(0.05, 0.10), (0.95, 0.10), (0.99, 1.00), (0.01, 1.00)]
_ZONE_NEAR = [(0.05, 0.20), (0.95, 0.20), (0.99, 1.00), (0.01, 1.00)]

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


# ============================================================
# THRESHOLD DETEKSI
# ============================================================
MIN_PERSON_WIDTH = 15
MIN_PERSON_HEIGHT = 35
MIN_VEHICLE_WIDTH = 30
MIN_VEHICLE_HEIGHT = 20

PLATE_VEHICLE_MIN_WIDTH = 30
PLATE_VEHICLE_MIN_HEIGHT = 20

FAKE_VEHICLE_ASPECT_MIN = 0.9

# ============================================================
# AI INTERVAL
# ============================================================
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

# ============================================================
# PLATE DETECTION
# ============================================================
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

# ============================================================
# ANTI-DUPLIKAT
# ============================================================
PLATE_REVIEW_CONFIDENCE = 0.50
PLATE_COOLDOWN = 45.0
PERSON_COOLDOWN = 45.0

TRACK_REUSE_GAP = 15.0

TRACKER_LOST_BUFFER = 150
TRACKER_MATCH_THRESHOLD = 0.75
TRACKER_ACTIVATION_THRESHOLD = 0.30

TRACK_EVENT_COOLDOWN = 600.0

# ✅ FIX: Dedup visual
VISUAL_DEDUP_COOLDOWN = 60.0
VISUAL_HAMMING_THRESHOLD = 3

PERSON_DEDUP_COOLDOWN = 120.0
PERSON_HAMMING_THRESHOLD = 8
PERSON_TRACK_DEDUP_COOLDOWN = 300.0

# ✅ FIX BARU: Dedup berbasis centroid untuk kendaraan
VEHICLE_CENTROID_DEDUP_DISTANCE = 150.0     # px — jarak maksimum centroid (mobil bergerak)
VEHICLE_CENTROID_DEDUP_AREA_RATIO = 0.40    # toleransi ukuran bbox (0.4 = 40%)
VEHICLE_CENTROID_DEDUP_COOLDOWN = 30.0      # detik
VEHICLE_CENTROID_MAX_HISTORY = 20           # max entry per kamera

# ✅ FIX #3: Dedup berbasis plat
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


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper().strip())


def _valid_indonesian_plate(text):
    text = _normalize_text(text)
    if not text or not (3 <= len(text) <= 9):
        return False
    return bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text))


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
    bbox = _bbox_from_track(track)
    if bbox is None:
        return None
    vote = None
    try:
        vote = track.hasil_voting()
    except Exception:
        vote = None
    text = ""; formatted = ""; ocr_conf = 0.0; votes = 0; total_reads = 0
    if vote:
        text = _normalize_text(vote.get("text", ""))
        formatted = vote.get("formatted", text)
        ocr_conf = _safe_float(vote.get("confidence_rata2", 0.0))
        votes = int(vote.get("jumlah_muncul", 0) or 0)
        total_reads = int(vote.get("total_bacaan", 0) or 0)
    if not text:
        text = _normalize_text(getattr(track, "best_text", ""))
        ocr_conf = _safe_float(getattr(track, "best_ocr_confidence", 0.0))
        formatted = text
    return {
        "id": int(getattr(track, "id", -1)),
        "track_id": int(getattr(track, "id", -1)),
        "box": bbox, "bbox": bbox,
        "conf": _safe_float(getattr(track, "detection_confidence", 0.0)),
        "detection_confidence": _safe_float(getattr(track, "detection_confidence", 0.0)),
        "text": text, "formatted": formatted, "raw_text": text,
        "ocr_conf": ocr_conf, "confidence": ocr_conf,
        "valid": _valid_indonesian_plate(text),
        "votes": votes, "total_reads": total_reads,
        "crop": getattr(track, "best_crop", None),
        "vehicle_track_id": (
            int(getattr(track, "vehicle_track_id"))
            if getattr(track, "vehicle_track_id", None) is not None
            else None
        ),
        "distance_zone": "NEAR",
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
    if polygon is None or len(polygon) < 3:
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
    point = _bottom_center(box)
    point_inside = cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0
    if point_inside:
        return True
    overlap = _bbox_polygon_overlap_ratio(box, polygon, width, height)
    if overlap >= ZONE_OVERLAP_FALLBACK_RATIO:
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


# ============================================================
# SERVICE
# ============================================================

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

        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD,
            lost_track_buffer=TRACKER_LOST_BUFFER,
            minimum_matching_threshold=TRACKER_MATCH_THRESHOLD,
            frame_rate=25,
        )

        self.plate_tracker = PlateTracker(
            iou_threshold=PLATE_TRACKER_IOU,
            max_frame_gap=PLATE_TRACKER_FRAME_GAP,
            ocr_every_n_matches=PLATE_TRACKER_OCR_EVERY,
            min_final_confidence=PLATE_TRACKER_MIN_CONFIDENCE,
            min_consistent_reads=2,
            single_read_ocr_confidence=0.72,
            single_read_detection_confidence=0.55,
            max_history=PLATE_TRACKER_MAX_HISTORY,
        )

        self.captured_tracks = {}
        self.saved_plate_events = {}
        self.last_ai_time = 0.0
        self.track_lifecycle = {}
        self.adaptive_ai = AdaptiveAIController()

        # ✅ Cache untuk dedup
        self._visual_hashes = {}
        self._plate_text_cache = {}
        # ✅ FIX BARU: Cache centroid untuk dedup kendaraan
        self._vehicle_centroids = {}

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
        print("[AI STREAM] Arah            : per-kamera (bukan line crossing)")
        print(f"[AI STREAM] Anti-dup        : centroid_dist={VEHICLE_CENTROID_DEDUP_DISTANCE}px, "
              f"centroid_ratio={VEHICLE_CENTROID_DEDUP_AREA_RATIO}, "
              f"centroid_cool={VEHICLE_CENTROID_DEDUP_COOLDOWN}s")
        print("[AI STREAM] OCR variants    : original/clahe/sharpen/threshold")
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

    def _detect_vehicles(self, frame, camera_id=1):
        camera_id = _normalize_camera_id(camera_id)
        detections = []
        try:
            h, w = frame.shape[:2]
            result = self.yolo_person(frame, classes=VEHICLE_CLASSES, imgsz=VEHICLE_IMGSZ, conf=VEHICLE_CONFIDENCE, verbose=False)[0]
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

                    if DEBUG_PERFORMANCE:
                        print(f"[ZONE DEBUG] cam={camera_id} cls={cls_id} bbox={bbox} bw={bw} bh={bh} zone={zone}")

                    if ENABLE_DISTANCE_ZONE and zone == "FAR":
                        if DEBUG_PERFORMANCE:
                            print(f"[SKIP] FAR: {bbox}")
                        continue
                    if cls_id == 0:
                        if bw < MIN_PERSON_WIDTH or bh < MIN_PERSON_HEIGHT:
                            if DEBUG_PERFORMANCE:
                                print(f"[SKIP] PERSON TOO SMALL: bw={bw} bh={bh}")
                            continue
                    elif cls_id in VEHICLE_TYPES:
                        if bw < MIN_VEHICLE_WIDTH or bh < MIN_VEHICLE_HEIGHT:
                            if DEBUG_PERFORMANCE:
                                print(f"[SKIP] VEHICLE TOO SMALL: bw={bw} bh={bh}")
                            continue
                        aspect = bw / max(1, bh)
                        if aspect < FAKE_VEHICLE_ASPECT_MIN:
                            if DEBUG_PERFORMANCE:
                                print(f"[SKIP] FAKE VEHICLE (aspect too small): aspect={aspect:.2f} cls={cls_id}")
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
        except Exception as exc:
            print(f"[AI STREAM ERROR] Vehicle detection failed: {exc}")
        return detections

    def _detect_plates_in_vehicles(self, frame, vehicle_dets):
        h, w = frame.shape[:2]
        candidates = [d for d in vehicle_dets
                      if d.get("cls") in (2, 3, 5, 7)
                      and (not ENABLE_DISTANCE_ZONE or d.get("distance_zone") == "NEAR")]
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
                item["bbox"] = [bx1, by1, bx2, by2]; item["box"] = item["bbox"]
                item["crop"] = plate_crop
                item["vehicle_cls"] = vehicle.get("cls", 0)
                item["vehicle_track_id"] = vehicle.get("track_id", -1)
                item["vehicle_box"] = [vx1, vy1, vx2, vy2]
                item["distance_zone"] = "NEAR"
                item["ocr_ready"] = _plate_ready_for_ocr(item)
                all_plates.append(item)
        all_plates.sort(key=lambda x: _safe_float(x.get("confidence", x.get("conf", 0.0))), reverse=True)
        final = []
        for candidate in all_plates:
            bx = candidate.get("bbox", [])
            if len(bx) != 4:
                continue
            cx = (bx[0] + bx[2]) / 2.0; cy = (bx[1] + bx[3]) / 2.0
            duplicate = False
            for existing in final:
                eb = existing.get("bbox", [])
                if len(eb) != 4:
                    continue
                ecx = (eb[0] + eb[2]) / 2.0; ecy = (eb[1] + eb[3]) / 2.0
                if ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5 < 18:
                    duplicate = True; break
            if not duplicate:
                final.append(candidate)
        return final

    def _consume_finished_plate(self, frame, vehicle_dets, camera_id):
        camera_id = _normalize_camera_id(camera_id)
        plate_tracker = self._camera_state(camera_id)["plate_tracker"]
        finished = plate_tracker.consume_latest_finished_capture()
        if finished is None:
            return None
        text = _normalize_text(_get_value(finished, ["text", "formatted"], ""))
        formatted = finished.get("formatted", text)
        ocr_conf = _safe_float(finished.get("confidence", 0.0))
        det_conf = _safe_float(finished.get("detection_confidence", 0.0))
        track_id = _get_value(finished, ["track_id", "id"], "unknown")
        crop = finished.get("crop")
        bbox = finished.get("bbox")
        if frame is not None and bbox and len(bbox) == 4:
            original_plate_crop = _prepare_high_quality_capture(frame, bbox, padding=0.20, target_short_side=400, max_scale=3.0, max_long_side=1000, sharpen=True)
            if original_plate_crop is not None:
                crop = original_plate_crop
        valid = _valid_indonesian_plate(text)
        if not valid and det_conf < PLATE_REVIEW_CONFIDENCE:
            return None
        plate_key = f"plate:{camera_id}:{text}" if valid and text else f"plate-track:{camera_id}:{track_id}"
        last_saved = self.saved_plate_events.get(plate_key)
        if last_saved is not None and time.time() - last_saved < PLATE_COOLDOWN:
            return None
        self.saved_plate_events[plate_key] = time.time()
        prefix = "plate" if valid else "plate_review"
        plate_path = save_plate_capture(crop, track_id, camera_id, formatted or text or "unknown", prefix=prefix)
        finished = dict(finished)
        vehicle_track_id = finished.get("vehicle_track_id")
        if vehicle_track_id is None and bbox and len(bbox) == 4:
            px1, py1, px2, py2 = [float(v) for v in bbox]
            pcx = (px1 + px2) / 2.0; pcy = (py1 + py2) / 2.0
            best_vehicle = None; best_score = 0.0
            for vehicle in vehicle_dets or []:
                if vehicle.get("cls") not in VEHICLE_TYPES:
                    continue
                vb = vehicle.get("box") or []
                if len(vb) != 4:
                    continue
                vx1, vy1, vx2, vy2 = [float(v) for v in vb]
                if not (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2):
                    continue
                vw = max(1.0, vx2 - vx1); vh = max(1.0, vy2 - vy1)
                score = 1.0 / (vw * vh)
                if score > best_score:
                    best_score = score; best_vehicle = vehicle
            if best_vehicle is not None:
                vehicle_track_id = best_vehicle.get("track_id")
        try:
            vehicle_track_id = int(vehicle_track_id) if vehicle_track_id is not None else None
        except (TypeError, ValueError):
            vehicle_track_id = None
        finished.update({
            "text": text, "formatted": formatted or text, "ocr_conf": ocr_conf,
            "conf": det_conf, "valid": valid, "plate_image_path": plate_path,
            "vehicle_track_id": vehicle_track_id,
        })
        self.last_plate_capture = finished
        self.last_plate_history = list(plate_tracker.history)
        print(f"[PLATE RESULT] Cam={camera_id} Track={track_id} {formatted or text or 'REVIEW'} | OCR={ocr_conf:.1%} | YOLO={det_conf:.1%}")
        return finished

    def _save_person_events(self, frame, person_dets, camera_id, current_time):
        return None

    @staticmethod
    def _intersection_ratio(inner_box, outer_box):
        ix1 = max(inner_box[0], outer_box[0])
        iy1 = max(inner_box[1], outer_box[1])
        ix2 = min(inner_box[2], outer_box[2])
        iy2 = min(inner_box[3], outer_box[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
        return intersection / area

    def _get_track_generation(self, camera_id, track_id, current_time):
        camera_id = _normalize_camera_id(camera_id)
        try:
            camera_id = int(camera_id); track_id = int(track_id); current_time = float(current_time)
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

    def _cleanup_runtime_state(self, current_time):
        now = float(current_time)
        lifecycle_ttl = 300.0
        event_ttl = 1800.0
        for key, state in list(self.track_lifecycle.items()):
            if now - float(state.get("last_seen", now)) > lifecycle_ttl:
                self.track_lifecycle.pop(key, None)
        for cache_name in ("captured_tracks", "saved_plate_events", "saved_event_meta"):
            cache = getattr(self, cache_name, None)
            if not isinstance(cache, dict):
                continue
            for key, value in list(cache.items()):
                try:
                    stamp = float(value if cache_name != "saved_event_meta" else value.get("updated_at", now))
                except (TypeError, ValueError, AttributeError):
                    continue
                if now - stamp > event_ttl:
                    cache.pop(key, None)
        for key, value in list(self._track_last_event.items()):
            try:
                if now - float(value.get("last_ts", now)) > 7200.0:
                    self._track_last_event.pop(key, None)
            except (TypeError, ValueError, AttributeError):
                self._track_last_event.pop(key, None)
        for key, value in list(self._visual_hashes.items()):
            try:
                if now - float(value.get("ts", now)) > 600.0:
                    self._visual_hashes.pop(key, None)
            except (TypeError, ValueError, AttributeError):
                self._visual_hashes.pop(key, None)
        for key, value in list(self._plate_text_cache.items()):
            try:
                if now - float(value.get("ts", now)) > 600.0:
                    self._plate_text_cache.pop(key, None)
            except (TypeError, ValueError, AttributeError):
                self._plate_text_cache.pop(key, None)
        # cleanup centroid cache
        for cam_key, items in list(self._vehicle_centroids.items()):
            self._vehicle_centroids[cam_key] = [
                it for it in items
                if now - it.get("ts", 0) < VEHICLE_CENTROID_DEDUP_COOLDOWN
            ]

    # ============================================================
    # Dedup visual dengan perceptual hash
    # ============================================================
    def _visual_hash(self, crop):
        if crop is None or not hasattr(crop, "size") or crop.size == 0:
            return None
        try:
            small = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = small.astype(np.float32)
            avg = gray.mean()
            bits = (gray > avg).flatten()
            hex_str = ""
            for i in range(0, len(bits), 4):
                nibble = bits[i:i+4]
                val = 0
                for b in nibble:
                    val = (val << 1) | int(b)
                hex_str += format(val, 'x')
            return hex_str[:16]
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

    def _is_duplicate_visual(self, camera_id, object_type, v_hash, current_time):
        if v_hash is None:
            return False
        if object_type == "person":
            threshold = PERSON_HAMMING_THRESHOLD
            cooldown = PERSON_DEDUP_COOLDOWN
        else:
            threshold = VISUAL_HAMMING_THRESHOLD
            cooldown = VISUAL_DEDUP_COOLDOWN

        prefix = f"vhash:{camera_id}:{object_type}:"
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

        key = f"{prefix}{v_hash}:{int(current_time)}"
        self._visual_hashes[key] = {"ts": current_time, "hash": v_hash}
        return False

    def _is_duplicate_plate(self, camera_id, plate_text, current_time):
        if not plate_text:
            return False
        key = f"plate-text:{camera_id}:{plate_text}"
        last = self._plate_text_cache.get(key)
        if last is not None:
            age = current_time - last.get("ts", 0)
            if age < PLATE_TEXT_DEDUP_COOLDOWN:
                return True
        self._plate_text_cache[key] = {"ts": current_time, "text": plate_text}
        return False

    def _is_duplicate_person_track(self, camera_id, track_id, current_time):
        key = f"person-track:{camera_id}:{track_id}"
        last = self._plate_text_cache.get(key)
        if last is not None:
            age = current_time - last.get("ts", 0)
            if age < PERSON_TRACK_DEDUP_COOLDOWN:
                return True
        self._plate_text_cache[key] = {"ts": current_time, "text": str(track_id)}
        return False

    # ============================================================
    # ✅ FIX BARU: Dedup berbasis centroid untuk kendaraan
    # ============================================================
    def _bbox_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _is_duplicate_centroid(self, camera_id, bbox, current_time):
        """
        Cek apakah kendaraan mirip sudah disimpan berdasarkan:
        - Centroid jarak < threshold px
        - Ukuran bbox (area) mirip dalam toleransi ratio
        - Waktu < cooldown
        """
        cx, cy = self._bbox_centroid(bbox)
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area = bw * bh

        # Bersihkan cache lama
        cache = self._vehicle_centroids.get(camera_id, [])
        cache = [it for it in cache if current_time - it["ts"] < VEHICLE_CENTROID_DEDUP_COOLDOWN]

        # Cek duplikat
        for item in cache:
            pcx, pcy = item["centroid"]
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist < VEHICLE_CENTROID_DEDUP_DISTANCE:
                prev_area = item["area"]
                max_area = max(area, prev_area)
                ratio = abs(area - prev_area) / max(1, max_area)
                if ratio < VEHICLE_CENTROID_DEDUP_AREA_RATIO:
                    return True

        # Simpan entry baru
        cache.append({
            "centroid": (cx, cy),
            "area": area,
            "ts": current_time,
        })
        # Batasi panjang
        if len(cache) > VEHICLE_CENTROID_MAX_HISTORY:
            cache = cache[-VEHICLE_CENTROID_MAX_HISTORY:]
        self._vehicle_centroids[camera_id] = cache
        return False

    def _resolve_event_key(self, lookup_key, current_time, plate_text_now=None):
        now_ts = float(current_time)
        last_event = self._track_last_event.get(lookup_key)
        if last_event is not None:
            age = now_ts - last_event["last_ts"]
            if age < TRACK_EVENT_COOLDOWN:
                prev_plate = (last_event.get("plate_text") or "").strip()
                new_plate = (plate_text_now or "").strip()
                if prev_plate and new_plate and prev_plate != new_plate:
                    event_key = f"{lookup_key}:t{int(now_ts)}"
                else:
                    event_key = last_event["event_key"]
            else:
                event_key = f"{lookup_key}:t{int(now_ts)}"
        else:
            event_key = f"{lookup_key}:t{int(now_ts)}"
        self._track_last_event[lookup_key] = {
            "event_key": event_key,
            "last_ts": now_ts,
            "plate_text": plate_text_now or "",
        }
        return event_key

    def _save_consistent_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        camera_id = _normalize_camera_id(camera_id)
        camera_orientation = db.get_camera_direction(camera_id)
        vehicles = [item for item in person_dets if item.get("cls") in VEHICLE_TYPES]
        people = [item for item in person_dets if item.get("cls") == 0]
        associated_people = set()

        for vehicle in vehicles:
            vehicle_box = vehicle.get("box", [0, 0, 0, 0])
            vehicle_id = int(vehicle.get("track_id", -1))
            vehicle_type = vehicle.get("vehicle_type") or VEHICLE_TYPES.get(vehicle.get("cls"), "unknown")

            matching_plates = []
            geometric_plates = []
            for plate in plate_dets:
                plate_vehicle_id = plate.get("vehicle_track_id")
                if plate_vehicle_id is not None:
                    try:
                        if int(plate_vehicle_id) == vehicle_id:
                            matching_plates.append(plate); continue
                    except (TypeError, ValueError):
                        pass
                plate_box = plate.get("bbox") or plate.get("box") or [0, 0, 0, 0]
                if len(plate_box) != 4:
                    continue
                center = plate.get("center") or [(plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2]
                if vehicle_box[0] <= center[0] <= vehicle_box[2] and vehicle_box[1] <= center[1] <= vehicle_box[3]:
                    geometric_plates.append(plate)

            if not matching_plates:
                matching_plates = geometric_plates
            plate = max(matching_plates, key=lambda item: float(item.get("conf", item.get("confidence", 0)) or 0), default=None)

            # ✅ FIX #1: Tolak kendaraan tanpa plat + aspect kecil
            bx1, by1, bx2, by2 = vehicle_box
            bw = max(1, bx2 - bx1); bh = max(1, by2 - by1)
            aspect = bw / bh
            if plate is None and aspect < FAKE_VEHICLE_ASPECT_MIN:
                if DEBUG_PERFORMANCE:
                    print(f"[SKIP FALSE VEHICLE] aspect={aspect:.2f} type={vehicle_type} conf={vehicle.get('conf', 0):.2f}")
                continue

            driver = None
            for index, person in enumerate(people):
                if index in associated_people:
                    continue
                person_box = person.get("box", [0, 0, 0, 0])
                if self._intersection_ratio(person_box, vehicle_box) >= 0.25:
                    driver = person; associated_people.add(index); break

            center = ((vehicle_box[0] + vehicle_box[2]) // 2, (vehicle_box[1] + vehicle_box[3]) // 2)
            stable_id = vehicle_id if vehicle_id >= 0 else f"{center[0]}_{center[1]}"

            if vehicle_id >= 0:
                generation = self._get_track_generation(camera_id, vehicle_id, current_time)
            else:
                generation = int(current_time)

            plate_text_now = _normalize_text((plate or {}).get("text") or "")
            track_lookup_key = f"vehicle:{camera_id}:{stable_id}"
            event_key = self._resolve_event_key(track_lookup_key, current_time, plate_text_now)

            has_driver_now = driver is not None
            previous_meta = self.saved_event_meta.get(event_key)
            is_new_event = event_key not in self.captured_tracks
            needs_enrichment = (
                not is_new_event and previous_meta is not None
                and ((plate_text_now and not previous_meta.get("plate_text"))
                     or (has_driver_now and not previous_meta.get("has_driver")))
            )
            if not is_new_event and not needs_enrichment:
                continue

            # 1. Dedup plat (paling akurat)
            if plate_text_now and _valid_indonesian_plate(plate_text_now):
                if self._is_duplicate_plate(camera_id, plate_text_now, current_time):
                    if DEBUG_PERFORMANCE:
                        print(f"[DEDUP PLAT] Skip: cam={camera_id} plate={plate_text_now}")
                    self.captured_tracks[event_key] = current_time
                    continue

            # ✅ 2. FIX BARU: Dedup centroid (paling cocok untuk kendaraan bergerak)
            if self._is_duplicate_centroid(camera_id, vehicle_box, current_time):
                if DEBUG_PERFORMANCE:
                    print(f"[DEDUP CENTROID] Skip: cam={camera_id} type={vehicle_type} bbox={vehicle_box}")
                self.captured_tracks[event_key] = current_time
                continue

            # 3. Dedup visual
            vehicle_crop = _prepare_vehicle_capture(frame, vehicle_box)
            v_hash = self._visual_hash(vehicle_crop)
            if self._is_duplicate_visual(camera_id, vehicle_type, v_hash, current_time):
                if DEBUG_PERFORMANCE:
                    print(f"[DEDUP VISUAL] Skip: cam={camera_id} type={vehicle_type} hash={v_hash}")
                self.captured_tracks[event_key] = current_time
                continue

            final_direction = camera_orientation
            try:
                result = db.save_detection_event(
                    camera_id=camera_id,
                    plate_number=(plate or {}).get("text") or None,
                    raw_ocr_text=(plate or {}).get("raw_text") or None,
                    plate_crop=(plate or {}).get("crop"),
                    plate_conf=float((plate or {}).get("conf", (plate or {}).get("confidence", 0.0)) or 0),
                    ocr_conf=float((plate or {}).get("ocr_conf", 0.0) or 0),
                    track_id=vehicle_id if vehicle_id >= 0 else None,
                    object_type="vehicle",
                    vehicle_type=vehicle_type,
                    vehicle_confidence=float(vehicle.get("conf", 0.0) or 0),
                    vehicle_crop=vehicle_crop,
                    has_driver=driver is not None,
                    driver_track_id=(driver or {}).get("track_id"),
                    direction=final_direction,
                    event_key=event_key
                )
                self.captured_tracks[event_key] = current_time
                previous_plate = (previous_meta or {}).get("plate_text", "")
                previous_driver = bool((previous_meta or {}).get("has_driver", False))
                self.saved_event_meta[event_key] = {
                    "plate_text": plate_text_now or previous_plate,
                    "has_driver": has_driver_now or previous_driver,
                    "direction": final_direction,
                    "updated_at": current_time,
                }
                print(f"[DETECTION] camera={camera_id} track_id={vehicle_id} object_type=vehicle vehicle_type={vehicle_type} vehicle_confidence={float(vehicle.get('conf', 0) or 0):.3f} plate_detected={bool(plate)} plate={(plate or {}).get('text') or '-'} plate_confidence={float((plate or {}).get('conf', 0) or 0):.3f} ocr_confidence={float((plate or {}).get('ocr_conf', 0) or 0):.3f} driver_detected={driver is not None} direction={final_direction} event_key={event_key} duplicate={result.get('duplicate', False)}")
            except Exception as exc:
                print(f"[AI STREAM ERROR] Save vehicle event failed: {exc}")

        for index, person in enumerate(people):
            if index in associated_people:
                continue
            track_id = int(person.get("track_id", -1))
            if track_id < 0:
                continue
            generation = self._get_track_generation(camera_id, track_id, current_time)

            if self._is_duplicate_person_track(camera_id, track_id, current_time):
                if DEBUG_PERFORMANCE:
                    print(f"[DEDUP PERSON TRACK] Skip: cam={camera_id} track_id={track_id}")
                continue

            track_lookup_key = f"person:{camera_id}:{track_id}"
            event_key = self._resolve_event_key(track_lookup_key, current_time, "")

            if event_key in self.captured_tracks:
                previous_meta = self.saved_event_meta.get(event_key)
                if previous_meta is not None and previous_meta.get("face_saved", False):
                    continue

            bx1, by1, bx2, by2 = person.get("box", [0, 0, 0, 0])
            crop = _prepare_high_quality_capture(frame, (bx1, by1, bx2, by2), padding=0.20, target_short_side=360, max_scale=2.5, max_long_side=960, sharpen=True)
            if crop is None or crop.size == 0:
                continue

            p_hash = self._visual_hash(crop)
            if self._is_duplicate_visual(camera_id, "person", p_hash, current_time):
                if DEBUG_PERFORMANCE:
                    print(f"[DEDUP VISUAL PERSON] Skip: cam={camera_id} hash={p_hash}")
                self.captured_tracks[event_key] = current_time
                continue

            final_direction = camera_orientation
            try:
                result = db.save_detection_event(
                    camera_id=camera_id, face_crop=crop,
                    face_conf=float(person.get("conf", 0.0) or 0),
                    track_id=track_id, object_type="person", vehicle_type="unknown",
                    direction=final_direction, event_key=event_key
                )
                self.captured_tracks[event_key] = current_time
                self.saved_event_meta[event_key] = {
                    "plate_text": "", "has_driver": False,
                    "direction": final_direction, "face_saved": True,
                    "updated_at": current_time,
                }
                print(f"[DETECTION] camera={camera_id} track_id={track_id} object_type=person person_confidence={float(person.get('conf', 0) or 0):.3f} direction={final_direction} duplicate={result.get('duplicate', False)}")
            except Exception as exc:
                print(f"[AI STREAM ERROR] Save person event failed: {exc}")

    def _save_new_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        return self._save_consistent_events(frame, person_dets, plate_dets, camera_id, current_time)

    def _select_dynamic_focus(self, frame, vehicle_dets, plate_raw, now):
        h, w = frame.shape[:2]
        vehicles = [d for d in vehicle_dets if d.get("cls") in (2, 3, 5, 7) and d.get("track_id", -1) >= 0]
        if ENABLE_DISTANCE_ZONE:
            near_vehicles = [d for d in vehicles if d.get("distance_zone") == "NEAR"]
            if near_vehicles:
                vehicles = near_vehicles
        target_type = "VEHICLE"
        if not vehicles:
            people = [d for d in vehicle_dets if d.get("cls") == 0 and d.get("track_id", -1) >= 0]
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
            target = max(vehicles, key=lambda d: max(1, d["box"][2] - d["box"][0]) * max(1, d["box"][3] - d["box"][1]))
            self.focus_track_id = int(target["track_id"])
        if target is None:
            self.focus_track_id = None
            box = [int(w * .20), int(h * .42), int(w * .88), int(h * .96)]
            self.focus_info = {"type": "AREA", "box": box, "track_id": None, "lighting": _lighting_metrics(frame, box)}
            return
        self.focus_last_seen = now
        tid = int(target["track_id"])
        focus_type = target_type
        box = _clamp_box(target["box"], w, h, FOCUS_PADDING_VEHICLE)
        matches = [p for p in plate_raw if int(p.get("vehicle_track_id", -999)) == tid and len(p.get("bbox", [])) == 4]
        if matches:
            best = max(matches, key=lambda p: _safe_float(p.get("confidence", p.get("conf", 0))))
            box = _clamp_box(best["bbox"], w, h, FOCUS_PADDING_PLATE)
            focus_type = 'PLATE'
            self.focus_plate_last_seen = now
        elif self.focus_info.get("type") == "PLATE" and now - self.focus_plate_last_seen < FOCUS_TARGET_HOLD_SECONDS:
            box = self.focus_info.get("box") or box
            focus_type = "PLATE"
        self.focus_info = {"type": focus_type, "box": box, "track_id": tid, "lighting": _lighting_metrics(frame, box)}

    def _draw_dynamic_focus(self, frame):
        info = self.focus_info or {}; box = info.get('box')
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

    def _run_ai_pipeline(self, frame, camera_id):
        camera_id = _normalize_camera_id(camera_id)
        camera_state = self._camera_state(camera_id)
        now = time.time()
        ai_pipeline_start = time.perf_counter()
        person_dets = []; plate_dets = []; plate_raw = []
        try:
            h0, w0 = frame.shape[:2]
            base_box = self.focus_info.get("box") or [int(w0 * 0.20), int(h0 * 0.42), int(w0 * 0.88), int(h0 * 0.96)]
            base_light = _lighting_metrics(frame, base_box)
            ai_frame = _enhance_for_lighting(frame, base_light)
            person_dets = self._detect_vehicles(ai_frame, camera_id) or []
            if DEBUG_PERFORMANCE:
                print(f"[AI RUN] camera={camera_id} objects={len(person_dets)} interval={self.adaptive_ai.get_interval():.2f}s")
            plate_dets = []; plate_raw = []
            try:
                plate_raw = self._detect_plates_in_vehicles(ai_frame, person_dets) or []
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
                    finished = self._consume_finished_plate(frame, person_dets, camera_id)
                    if finished is not None:
                        plate_dets.append(finished)
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
                            "detection_confidence": conf, "text": "", "formatted": "",
                            "raw_text": "", "ocr_conf": 0.0, "confidence": 0.0,
                            "valid": False, "votes": 0, "total_reads": 0,
                            "distance_zone": item.get("distance_zone", "NEAR"),
                            "ocr_ready": bool(item.get("ocr_ready", False)),
                        })
            except Exception as exc:
                print(f"[AI STREAM ERROR] Plate pipeline failed: {exc}")
            camera_state["last_results"] = {"persons": person_dets, "plates": plate_dets}
            camera_state["last_results_time"] = now
            person_dets = person_dets or []
            plate_dets = plate_dets or []
            self._save_consistent_events(frame, person_dets, plate_dets, camera_id, now)
            self._cleanup_runtime_state(now)
        except Exception as exc:
            print(f"[AI STREAM ERROR] Background AI cycle failed: {exc}")
        finally:
            elapsed = time.perf_counter() - ai_pipeline_start
            self.last_ai_time = elapsed
            self.adaptive_ai.update(elapsed)
            if DEBUG_PERFORMANCE:
                print(f"[AI PERF] camera={camera_id} elapsed={elapsed * 1000:.0f}ms next_interval={self.adaptive_ai.get_interval():.2f}s")

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

    def stop(self):
        self.running = False
        self.ai_wakeup.set()
        thread = getattr(self, "ai_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _display_frame(self, frame, camera_id):
        self.last_display_time[camera_id] = time.time()
        return frame

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
            zone = obj.get("distance_zone", "")
            zone_label = f" {zone}" if zone else ""
            if cls_id == 0: label = f"Orang ID:{track_id} {conf:.0%}{zone_label}"; color = (0, 165, 255)
            elif cls_id == 2: label = f"Mobil ID:{track_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 3: label = f"Motor ID:{track_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 5: label = f"Bus ID:{track_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            elif cls_id == 7: label = f"Truk ID:{track_id} {conf:.0%}{zone_label}"; color = (0, 200, 255)
            else: label = f"Objek {conf:.0%}{zone_label}"; color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            by = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
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
            if valid and text:
                label = f"{text} OCR:{ocr_conf:.0%} V:{votes}"; color = (0, 230, 118)
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