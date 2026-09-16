import os
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
import db

# ============================================================
# KONFIGURASI AI STREAM PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plate", "license-plate-finetune-v2n.pt")

<<<<<<< Updated upstream
=======
CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_CAPTURE_DIR = os.path.join(CAPTURE_DIR, "plates")

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# YOLO COCO classes used by this service:
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck.
VEHICLE_CLASSES = [0, 2, 3, 5, 7]
VEHICLE_CONFIDENCE = 0.35
VEHICLE_IMGSZ = 416

# ============================================================
# DISTANCE / DETECTION ZONES
# ============================================================
# The system does not estimate physical meters from a normal CCTV
# image. Instead it uses perspective zones. The bottom-center of an
# object's bounding box is used as its ground/contact position.
#
# Coordinates are NORMALIZED (0.0 - 1.0), so they work with different
# CCTV resolutions.
#
# MID_ZONE:
#   Person/vehicle detection + tracking is allowed here.
#
# NEAR_ZONE:
#   Plate detection + OCR is allowed here.
#
# IMPORTANT:
#   Tune these polygons to the actual road/entrance in each camera.
ENABLE_DISTANCE_ZONE = True

DEFAULT_MID_ZONE = [
    (0.20, 0.40),
    (0.80, 0.40),
    (0.98, 1.00),
    (0.02, 1.00),
]

DEFAULT_NEAR_ZONE = [
    (0.28, 0.56),
    (0.72, 0.56),
    (0.90, 1.00),
    (0.10, 1.00),
]

# ============================================================
# PER-CAMERA DISTANCE ZONES
# ============================================================
# IMPORTANT:
# Coordinates are normalized to the ACTUAL CCTV frame:
#   x=0.0 left, x=1.0 right
#   y=0.0 top,  y=1.0 bottom
#
# Based on the 4 CCTV views supplied:
#   1 = GSMasukViewDalam
#   2 = GSMasukViewLuar
#   3 = GSKeluarViewLuar
#   4 = GSKeluarViewDalam
#
# MID  : person + vehicle detection/tracking
# NEAR : plate detection + OCR
#
# These are perspective zones, NOT physical meters.
# The bottom-center of the bbox must be inside the polygon.
#
# The zones are intentionally different for each camera because the
# camera perspective / road position is different.
CAMERA_ZONE_CONFIG = {
    # ============================================================
    # 1. GSMasukViewDalam
    # ============================================================
    1: {
        "name": "GSMasukViewDalam",
        "mid": [
            (0.10, 0.38),
            (0.90, 0.38),
            (0.99, 1.00),
            (0.01, 1.00),
        ],
        "near": [
            (0.15, 0.53),
            (0.85, 0.53),
            (0.96, 1.00),
            (0.04, 1.00),
        ],
    },

    # ============================================================
    # 2. GSMasukViewLuar
    # ============================================================
    2: {
        "name": "GSMasukViewLuar",
        "mid": [
            (0.10, 0.39),
            (0.90, 0.39),
            (0.99, 1.00),
            (0.01, 1.00),
        ],
        "near": [
            (0.14, 0.54),
            (0.86, 0.54),
            (0.96, 1.00),
            (0.04, 1.00),
        ],
    },

    # ============================================================
    # 3. GSKeluarViewLuar
    # ============================================================
    3: {
        "name": "GSKeluarViewLuar",
        "mid": [
            (0.08, 0.37),
            (0.92, 0.37),
            (0.99, 1.00),
            (0.01, 1.00),
        ],
        "near": [
            (0.14, 0.52),
            (0.86, 0.52),
            (0.96, 1.00),
            (0.04, 1.00),
        ],
    },

    # ============================================================
    # 4. GSKeluarViewDalam
    # ============================================================
    4: {
        "name": "GSKeluarViewDalam",
        "mid": [
            (0.10, 0.37),
            (0.92, 0.37),
            (0.99, 1.00),
            (0.03, 1.00),
        ],
        "near": [
            (0.15, 0.51),
            (0.87, 0.51),
            (0.96, 1.00),
            (0.04, 1.00),
        ],
    },

    # ============================================================
    # ALIAS BERDASARKAN NAMA STREAM CCTV
    # ============================================================
    "GSMasukViewDalam": {
        "mid": [(0.10, 0.38), (0.90, 0.38), (0.99, 1.00), (0.01, 1.00)],
        "near": [(0.15, 0.53), (0.85, 0.53), (0.96, 1.00), (0.04, 1.00)],
    },
    "GSMasukViewLuar": {
        "mid": [(0.10, 0.39), (0.90, 0.39), (0.99, 1.00), (0.01, 1.00)],
        "near": [(0.14, 0.54), (0.86, 0.54), (0.96, 1.00), (0.04, 1.00)],
    },
    "GSKeluarViewLuar": {
        "mid": [(0.08, 0.37), (0.92, 0.37), (0.99, 1.00), (0.01, 1.00)],
        "near": [(0.14, 0.52), (0.86, 0.52), (0.96, 1.00), (0.04, 1.00)],
    },
    "GSKeluarViewDalam": {
        "mid": [(0.10, 0.37), (0.92, 0.37), (0.99, 1.00), (0.03, 1.00)],
        "near": [(0.15, 0.51), (0.87, 0.51), (0.96, 1.00), (0.04, 1.00)],
    },
}

# Minimum object size after it enters the detection zone.
# These are NOT meter measurements; they are image-pixel quality gates.
MIN_PERSON_WIDTH = 30
MIN_PERSON_HEIGHT = 70

MIN_VEHICLE_WIDTH = 120
MIN_VEHICLE_HEIGHT = 60

# Extra minimum size specifically for plate processing.
PLATE_VEHICLE_MIN_WIDTH = 120
PLATE_VEHICLE_MIN_HEIGHT = 60

# Adaptive AI interval for Intel i7-8665U CPU-only
BASE_AI_INTERVAL = 0.40
MIN_AI_INTERVAL = 0.30
MAX_AI_INTERVAL = 0.60
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

# Plate detector
PLATE_CONFIDENCE = 0.55
PLATE_IMGSZ = 512
PLATE_MAX_DET = 3
PLATE_IOU = 0.45
PLATE_MIN_WIDTH = 35
PLATE_MIN_HEIGHT = 10

# OCR only starts when the plate is physically/visually close enough
# in the image to provide useful character pixels.
OCR_MIN_PLATE_WIDTH = 45
OCR_MIN_PLATE_HEIGHT = 12
OCR_MIN_PLATE_CONFIDENCE = 0.55

# Limits for processing plates per AI cycle
# One near vehicle is preferred on a CPU-only laptop.
MAX_PLATE_ROI = 1

# Cooldown intervals (seconds) per vehicle/plate track
PLATE_DETECT_INTERVAL = 0.60
OCR_INTERVAL = 0.80

# PlateTracker settings
PLATE_TRACKER_IOU = 0.40
PLATE_TRACKER_FRAME_GAP = 30
PLATE_TRACKER_OCR_EVERY = 3
PLATE_TRACKER_MIN_CONFIDENCE = 0.42
PLATE_TRACKER_MAX_HISTORY = 5
PLATE_PADDING = 0.08

PLATE_REVIEW_CONFIDENCE = 0.60
PLATE_COOLDOWN = 30.0
PERSON_COOLDOWN = 30.0

# ByteTrack IDs can be reused after an object disappears. Treat a track as a
# new physical event only after it has been absent for this long.
TRACK_REUSE_GAP = 5.0

PERSON_CAPTURE_CONFIDENCE = 0.40
MIN_PERSON_CROP_WIDTH = 25
MIN_PERSON_CROP_HEIGHT = 45
MIN_CAPTURE_WIDTH = 120
MIN_CAPTURE_HEIGHT = 180

DISPLAY_FPS = 15

# Dynamic lighting/focus
FOCUS_TARGET_HOLD_SECONDS = 1.5
RESULT_HOLD_SECONDS = 1.5
FOCUS_PADDING_VEHICLE = 0.12
FOCUS_PADDING_PLATE = 0.55
LIGHT_DARK_THRESHOLD = 72.0
LIGHT_OVER_THRESHOLD = 205.0
LIGHT_GLARE_RATIO = 0.18

DEBUG_PERFORMANCE = False



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
        ok = cv2.imwrite(
            path,
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        return path if ok else None
    except Exception as exc:
        print(f"[CAPTURE ERROR] {exc}")
        return None


def crop_expanded_object(frame, bbox, min_width=MIN_CAPTURE_WIDTH, min_height=MIN_CAPTURE_HEIGHT):
    """
    Crop objek dari frame dengan memastikan ukuran bounding box minimal min_width x min_height.
    Jika bounding box asli lebih kecil dari ukuran minimal, bounding box akan di-expand
    secara simetris dari titik tengah. Jika sudah memenuhi, ukuran asli dipertahankan.
    Pergeseran (shift) dilakukan jika box mendekati tepi frame kamera agar ukuran minimal tetap terjaga.
    """
    if frame is None or getattr(frame, "size", 0) == 0 or bbox is None or len(bbox) != 4:
        return None
    h_frame, w_frame = frame.shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    cur_w = x2 - x1
    cur_h = y2 - y1
    if cur_w <= 0 or cur_h <= 0:
        return None

    # Expand lebar simetris dari center jika kurang dari min_width
    if cur_w < min_width:
        diff_w = min_width - cur_w
        x1 -= diff_w / 2.0
        x2 += diff_w / 2.0

    # Expand tinggi simetris dari center jika kurang dari min_height
    if cur_h < min_height:
        diff_h = min_height - cur_h
        y1 -= diff_h / 2.0
        y2 += diff_h / 2.0

    # Pergeseran batas X
    if x1 < 0:
        x2 += (0.0 - x1)
        x1 = 0.0
    if x2 > w_frame:
        x1 -= (x2 - float(w_frame))
        x2 = float(w_frame)
    x1 = max(0, min(int(round(x1)), w_frame))
    x2 = max(0, min(int(round(x2)), w_frame))

    # Pergeseran batas Y
    if y1 < 0:
        y2 += (0.0 - y1)
        y1 = 0.0
    if y2 > h_frame:
        y1 -= (y2 - float(h_frame))
        y2 = float(h_frame)
    y1 = max(0, min(int(round(y1)), h_frame))
    y2 = max(0, min(int(round(y2)), h_frame))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
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
    if x2 - x1 < MIN_PERSON_CROP_WIDTH:
        return None
    if y2 - y1 < MIN_PERSON_CROP_HEIGHT:
        return None

    crop = crop_expanded_object(frame, bbox, min_width=MIN_CAPTURE_WIDTH, min_height=MIN_CAPTURE_HEIGHT)
    if crop is None or crop.size == 0:
        return None
    now = datetime.now()
    date_dir = os.path.join(CAPTURE_DIR, now.strftime("%Y%m%d"))
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"person_cam{camera_id}_track{track_id}_{ts}.jpg"

    return _save_image(crop, date_dir, filename, 92)


def save_plate_capture(crop, track_id, camera_id, plate_text, prefix="plate"):
    if crop is None or crop.size == 0:
        return None

    now = datetime.now()
    date_dir = os.path.join(PLATE_CAPTURE_DIR, now.strftime("%Y%m%d"))
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_plate = _safe_filename(plate_text or "unknown")
    filename = (
        f"{prefix}_cam{camera_id}_track{track_id}_"
        f"{ts}_{safe_plate}.jpg"
    )

    return _save_image(crop, date_dir, filename, 95)


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

    text = ""
    formatted = ""
    ocr_conf = 0.0
    votes = 0
    total_reads = 0

    if vote:
        text = _normalize_text(vote.get("text", ""))
        formatted = vote.get("formatted", text)
        ocr_conf = _safe_float(vote.get("confidence_rata2", 0.0))
        votes = int(vote.get("jumlah_muncul", 0) or 0)
        total_reads = int(vote.get("total_bacaan", 0) or 0)

    if not text:
        text = _normalize_text(getattr(track, "best_text", ""))
        ocr_conf = _safe_float(
            getattr(track, "best_ocr_confidence", 0.0)
        )
        formatted = text

    return {
        "id": int(getattr(track, "id", -1)),
        "track_id": int(getattr(track, "id", -1)),
        "box": bbox,
        "bbox": bbox,
        "conf": _safe_float(
            getattr(track, "detection_confidence", 0.0)
        ),
        "detection_confidence": _safe_float(
            getattr(track, "detection_confidence", 0.0)
        ),
        "text": text,
        "formatted": formatted,
        "raw_text": text,
        "ocr_conf": ocr_conf,
        "confidence": ocr_conf,
        "valid": _valid_indonesian_plate(text),
        "votes": votes,
        "total_reads": total_reads,
        "crop": getattr(track, "best_crop", None),
        "distance_zone": "NEAR",
    }


def _find_vehicle_for_plate(frame, plate_bbox, vehicle_dets):
    if frame is None or plate_bbox is None:
        return None, 0.0

    px1, py1, px2, py2 = [int(v) for v in plate_bbox]
    best = None
    best_score = 0.0

    for vehicle in vehicle_dets or []:
        if vehicle.get("cls") not in (2, 3, 5):
            continue

        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle["box"]]

        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter = float((ix2 - ix1) * (iy2 - iy1))
        plate_area = float(max(1, (px2 - px1) * (py2 - py1)))
        score = inter / plate_area

        if score > best_score:
            best_score = score
            best = vehicle

    if best is None:
        return None, 0.0

    try:
        crop = crop_expanded_object(frame, best["box"], min_width=MIN_CAPTURE_WIDTH, min_height=MIN_CAPTURE_HEIGHT)
        return crop if (crop is not None and crop.size) else None, _safe_float(best.get("conf", 0.0))
    except Exception:
        return None, 0.0



def _clamp_box(box, w, h, padding=0.0):
    x1,y1,x2,y2=[int(v) for v in box]
    bw=max(1,x2-x1); bh=max(1,y2-y1)
    px=int(bw*padding); py=int(bh*padding)
    return [max(0,x1-px), max(0,y1-py), min(w,x2+px), min(h,y2+py)]

def _lighting_metrics(frame, box):
    h,w=frame.shape[:2]; x1,y1,x2,y2=_clamp_box(box,w,h,0.0)
    roi=frame[y1:y2,x1:x2]
    if roi.size == 0:
        return {"brightness":0.0,"contrast":0.0,"dark_ratio":1.0,"glare_ratio":0.0,"score":0.0,"status":"NO ROI"}
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    brightness=float(gray.mean()); contrast=float(gray.std())
    dark_ratio=float(np.mean(gray < 45)); glare_ratio=float(np.mean(gray > 235))
    bscore=max(0.0,100.0-abs(brightness-125.0)*0.75)
    cscore=min(100.0,contrast*2.0)
    score=max(0.0,min(100.0,0.65*bscore+0.35*cscore-glare_ratio*80.0))
    if glare_ratio >= LIGHT_GLARE_RATIO: status='GLARE'
    elif brightness < 45: status='TOO DARK'
    elif brightness < LIGHT_DARK_THRESHOLD: status='LOW LIGHT'
    elif brightness > LIGHT_OVER_THRESHOLD: status='OVEREXPOSED'
    elif score >= 72: status='GOOD'
    else: status='FAIR'
    return {"brightness":brightness,"contrast":contrast,"dark_ratio":dark_ratio,"glare_ratio":glare_ratio,"score":score,"status":status}

def _enhance_for_lighting(frame, metrics):
    status=(metrics or {}).get('status','GOOD')
    if status not in ('TOO DARK','LOW LIGHT','OVEREXPOSED','GLARE'):
        return frame
    img=frame
    if status in ('TOO DARK','LOW LIGHT'):
        gamma=0.58 if status=='TOO DARK' else 0.78
    else:
        gamma=1.28
    lut=np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype('uint8')
    img=cv2.LUT(img,lut)
    if status in ('TOO DARK','LOW LIGHT'):
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        img=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
    return img

def _center(box):
    return (int((box[0]+box[2])/2), int((box[1]+box[3])/2))


def _bottom_center(box):
    """Return the object's ground/contact point for perspective-zone tests."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return (
        int((x1 + x2) / 2.0),
        int(y2),
    )


def _normalized_polygon_to_pixels(points, width, height):
    """Convert normalized polygon coordinates to image pixels."""
    return np.array(
        [
            [
                int(max(0.0, min(1.0, float(x))) * width),
                int(max(0.0, min(1.0, float(y))) * height),
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )


def _camera_zone_points(camera_id, zone_name):
    """Return normalized zone points for a camera.

    Supports both numeric camera IDs and the CCTV stream names shown
    in the dashboard.
    """
    config = CAMERA_ZONE_CONFIG.get(camera_id, {})

    # Some callers may pass "1" instead of 1.
    if not config and isinstance(camera_id, str):
        stripped = camera_id.strip()
        try:
            config = CAMERA_ZONE_CONFIG.get(int(stripped), {})
        except (TypeError, ValueError):
            pass

    if zone_name == "near":
        return config.get("near", DEFAULT_NEAR_ZONE)
    return config.get("mid", DEFAULT_MID_ZONE)


def _camera_zone_name(camera_id):
    """Return a readable camera name for debug/status output."""
    config = CAMERA_ZONE_CONFIG.get(camera_id, {})

    if not config and isinstance(camera_id, str):
        try:
            config = CAMERA_ZONE_CONFIG.get(int(camera_id.strip()), {})
        except (TypeError, ValueError):
            pass

    return config.get("name", str(camera_id))


def _point_in_zone(box, zone_points, width, height):
    """Test bbox bottom-center against a normalized polygon."""
    if not ENABLE_DISTANCE_ZONE:
        return True

    if not zone_points or len(zone_points) < 3:
        return True

    point = _bottom_center(box)
    polygon = _normalized_polygon_to_pixels(zone_points, width, height)

    return cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0


def _get_distance_zone(box, camera_id, width, height):
    """Return FAR/MID/NEAR based on the bbox bottom-center."""
    if not ENABLE_DISTANCE_ZONE:
        return "DISABLED"

    near = _point_in_zone(
        box,
        _camera_zone_points(camera_id, "near"),
        width,
        height,
    )
    if near:
        return "NEAR"

    mid = _point_in_zone(
        box,
        _camera_zone_points(camera_id, "mid"),
        width,
        height,
    )
    if mid:
        return "MID"

    return "FAR"


def _plate_ready_for_ocr(plate):
    """Gate OCR by plate pixel size and detector confidence."""
    bbox = plate.get("bbox") or plate.get("box") or []
    if len(bbox) != 4:
        return False

    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1
    confidence = _safe_float(
        plate.get("confidence", plate.get("conf", 0.0))
    )

    return (
        width >= OCR_MIN_PLATE_WIDTH
        and height >= OCR_MIN_PLATE_HEIGHT
        and confidence >= OCR_MIN_PLATE_CONFIDENCE
    )


class AdaptiveAIController:
    """Adaptive AI scheduler based on CPU load and YOLO inference time."""

    def __init__(
        self,
        base_interval=BASE_AI_INTERVAL,
        min_interval=MIN_AI_INTERVAL,
        max_interval=MAX_AI_INTERVAL,
        target_cpu=TARGET_CPU,
        high_cpu=HIGH_CPU,
        low_cpu=LOW_CPU,
    ):
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
            self.current_interval = min(
                self.current_interval + 0.05,
                self.max_interval,
            )
        elif self.last_cpu >= self.target_cpu or self.last_ai_time >= 0.35:
            self.current_interval = min(
                self.current_interval + 0.02,
                self.max_interval,
            )
        elif self.last_cpu <= self.low_cpu and self.last_ai_time <= 0.20:
            self.current_interval = max(
                self.current_interval - 0.02,
                self.min_interval,
            )

        self.current_interval = max(
            self.min_interval,
            min(self.current_interval, self.max_interval),
        )
        self.last_adjustment = now

        if abs(self.current_interval - old_interval) >= 0.01:
            print(
                f"[ADAPTIVE AI] CPU={self.last_cpu:.1f}% | "
                f"YOLO={self.last_ai_time * 1000:.0f}ms | "
                f"interval={self.current_interval:.2f}s"
            )

        return self.current_interval

    def get_interval(self):
        return self.current_interval

    def get_status(self):
        return {
            "cpu": round(self.last_cpu, 1),
            "ai_time_ms": round(self.last_ai_time * 1000, 1),
            "interval": round(self.current_interval, 2),
        }


# ============================================================
# SERVICE
# ============================================================
>>>>>>> Stashed changes

import threading

class StreamAIService:
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("[AI STREAM] Inisialisasi model YOLO dan OCR...")
        self.lock = threading.Lock()
        self.yolo_person = YOLO(PERSON_MODEL_PATH)
        self.plate_detector = PlateDetector(PLATE_MODEL_PATH)
        self.plate_ocr = PlateOCR()

        # ByteTrack per kamera: camera_id -> sv.ByteTrack instance
        self.trackers = {}

        # Cache untuk melacak track yang sudah di-capture (mencegah duplicate insert)
        # (camera_id, key) -> timestamp capture terakhir
        self.captured_tracks = {}
        self.last_ai_times = {}
        self.last_results = {}
        print("[AI STREAM] Pipeline AI siap digunakan!")

    def _get_tracker(self, camera_id):
        """Membuat atau mengambil instance ByteTrack terisolasi per kamera."""
        cam_key = int(camera_id) if camera_id else 1
        if cam_key not in self.trackers:
            self.trackers[cam_key] = sv.ByteTrack(
                track_activation_threshold=0.35,
                lost_track_buffer=60,
                minimum_matching_threshold=0.7,
                frame_rate=25
            )
        return self.trackers[cam_key]

<<<<<<< Updated upstream
    def _cleanup_old_tracks(self, current_time):
        """Menghapus cache track yang sudah lewat dari 30 detik."""
        expired = [k for k, ts in list(self.captured_tracks.items()) if current_time - ts > 30.0]
        for k in expired:
            self.captured_tracks.pop(k, None)
=======
        return detections

    def _detect_plates_in_vehicles(self, frame, vehicle_dets):
        """Run plate detection only on NEAR_ZONE vehicles."""
        h, w = frame.shape[:2]

        candidates = [
            d
            for d in vehicle_dets
            if d.get("cls") in (2, 3, 5)
            and (
                not ENABLE_DISTANCE_ZONE
                or d.get("distance_zone") == "NEAR"
            )
        ]

        candidates.sort(
            key=lambda d: (
                1 if d.get("distance_zone") == "NEAR" else 0,
                max(1, d["box"][2] - d["box"][0])
                * max(1, d["box"][3] - d["box"][1]),
                _safe_float(d.get("conf", 0.0)),
            ),
            reverse=True,
        )

        all_plates = []

        # Only the largest near vehicle(s) get the expensive plate model.
        for vehicle in candidates[:MAX_PLATE_ROI]:
            vx1, vy1, vx2, vy2 = [
                int(v) for v in vehicle["box"]
            ]

            vx1 = max(0, min(vx1, w - 1))
            vy1 = max(0, min(vy1, h - 1))
            vx2 = max(0, min(vx2, w))
            vy2 = max(0, min(vy2, h))

            vw = vx2 - vx1
            vh = vy2 - vy1

            # Stronger gate for plate/OCR because distant vehicles
            # may technically be trackable but do not contain enough
            # plate pixels.
            if (
                vw < PLATE_VEHICLE_MIN_WIDTH
                or vh < PLATE_VEHICLE_MIN_HEIGHT
            ):
                continue

            if vx2 <= vx1 or vy2 <= vy1:
                continue

            pad_x = int(vw * 0.06)
            pad_y = int(vh * 0.10)

            rx1 = max(0, vx1 - pad_x)
            ry1 = max(0, vy1 - pad_y)
            rx2 = min(w, vx2 + pad_x)
            ry2 = min(h, vy2 + pad_y)

            crop = frame[ry1:ry2, rx1:rx2]

            if crop.size == 0:
                continue

            try:
                local_plates = (
                    self.plate_detector.detect(crop)
                    or []
                )
            except Exception as exc:
                print(
                    f"[AI STREAM ERROR] Plate ROI failed: {exc}"
                )
                continue

            for plate in local_plates:
                lb = plate.get("bbox", [])

                if len(lb) != 4:
                    continue

                bx1, by1, bx2, by2 = [
                    int(v) for v in lb
                ]

                bx1 += rx1
                by1 += ry1
                bx2 += rx1
                by2 += ry1

                bx1 = max(0, min(bx1, w - 1))
                by1 = max(0, min(by1, h - 1))
                bx2 = max(0, min(bx2, w))
                by2 = max(0, min(by2, h))

                if bx2 <= bx1 or by2 <= by1:
                    continue

                plate_width = bx2 - bx1
                plate_height = by2 - by1

                # Reject tiny plate boxes before tracker/OCR.
                if (
                    plate_width < PLATE_MIN_WIDTH
                    or plate_height < PLATE_MIN_HEIGHT
                ):
                    continue

                plate_crop = frame[
                    by1:by2,
                    bx1:bx2,
                ]

                if plate_crop.size == 0:
                    continue

                item = dict(plate)
                item["bbox"] = [
                    bx1,
                    by1,
                    bx2,
                    by2,
                ]
                item["box"] = item["bbox"]
                item["crop"] = plate_crop
                item["vehicle_cls"] = vehicle.get(
                    "cls",
                    0,
                )
                item["vehicle_track_id"] = vehicle.get(
                    "track_id",
                    -1,
                )
                item["vehicle_box"] = [
                    vx1,
                    vy1,
                    vx2,
                    vy2,
                ]
                item["distance_zone"] = "NEAR"
                item["ocr_ready"] = _plate_ready_for_ocr(
                    item
                )

                all_plates.append(item)

        # Remove duplicates by center distance.
        all_plates.sort(
            key=lambda x: _safe_float(
                x.get(
                    "confidence",
                    x.get("conf", 0.0),
                )
            ),
            reverse=True,
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

                if (
                    (cx - ecx) ** 2
                    + (cy - ecy) ** 2
                ) ** 0.5 < 18:
                    duplicate = True
                    break

            if not duplicate:
                final.append(candidate)

        return final

    # ------------------------------------------------------------
    # PLATE TRACK + FINAL EVENT
    # ------------------------------------------------------------

    def _consume_finished_plate(self, frame, vehicle_dets, camera_id):
        plate_tracker = self._camera_state(camera_id)["plate_tracker"]
        finished = plate_tracker.consume_latest_finished_capture()
        if finished is None:
            return None

        text = _normalize_text(
            _get_value(finished, ["text", "formatted"], "")
        )
        formatted = finished.get("formatted", text)
        ocr_conf = _safe_float(finished.get("confidence", 0.0))
        det_conf = _safe_float(finished.get("detection_confidence", 0.0))
        track_id = _get_value(finished, ["track_id", "id"], "unknown")
        crop = finished.get("crop")
        bbox = finished.get("bbox")

        valid = _valid_indonesian_plate(text)
        if not valid and det_conf < PLATE_REVIEW_CONFIDENCE:
            return None

        plate_key = (
            f"plate:{camera_id}:{text}"
            if valid and text
            else f"plate-track:{camera_id}:{track_id}"
        )
        last_saved = self.saved_plate_events.get(plate_key)
        if last_saved is not None and time.time() - last_saved < PLATE_COOLDOWN:
            return None
        self.saved_plate_events[plate_key] = time.time()

        prefix = "plate" if valid else "plate_review"
        plate_path = save_plate_capture(
            crop,
            track_id,
            camera_id,
            formatted or text or "unknown",
            prefix=prefix,
        )

        finished = dict(finished)
        finished.update({
            "text": text,
            "formatted": formatted or text,
            "ocr_conf": ocr_conf,
            "conf": det_conf,
            "valid": valid,
            "plate_image_path": plate_path,
        })

        self.last_plate_capture = finished
        self.last_plate_history = list(plate_tracker.history)

        print(
            f"[PLATE RESULT] Cam={camera_id} "
            f"Track={track_id} "
            f"{formatted or text or 'REVIEW'} "
            f"| OCR={ocr_conf:.1%} "
            f"| YOLO={det_conf:.1%}"
        )

        return finished

    # ------------------------------------------------------------
    # SAVE PERSON EVENTS
    # ------------------------------------------------------------

    def _save_person_events(self, frame, person_dets, camera_id, current_time):
        # Event writer sudah menggabungkan person-driver dan pedestrian.
        # Jangan menyimpan person melalui jalur kedua karena akan menggandakan event.
        return None

    # ========================================================
    # SAVE EVENTS
    # ========================================================

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
        """Return a stable event generation for a camera/track lifecycle.

        ByteTrack IDs are not globally unique and may be reused after an
        object disappears. The database event_key therefore includes this
        generation so a later vehicle/person with the same tracker ID can
        create a new event.
        """
        key = (int(camera_id), int(track_id))
        state = self.track_lifecycle.get(key)

        if state is None:
            state = {"generation": 1, "last_seen": current_time}
        elif current_time - float(state["last_seen"]) > TRACK_REUSE_GAP:
            state["generation"] = int(state.get("generation", 1)) + 1
            state["last_seen"] = current_time
        else:
            state["last_seen"] = current_time

        self.track_lifecycle[key] = state
        return int(state["generation"])

    def _save_consistent_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        """Persist one explicit event per tracked vehicle or unassociated person."""
        direction = db.get_camera_direction(camera_id)
        vehicles = [item for item in person_dets if item.get("cls") in VEHICLE_TYPES]
        people = [item for item in person_dets if item.get("cls") == 0]
        associated_people = set()

        for vehicle in vehicles:
            vehicle_box = vehicle.get("box", [0, 0, 0, 0])
            vehicle_id = int(vehicle.get("track_id", -1))
            vehicle_type = vehicle.get("vehicle_type") or VEHICLE_TYPES.get(vehicle.get("cls"), "unknown")

            matching_plates = []
            for plate in plate_dets:
                plate_box = plate.get("bbox") or plate.get("box") or [0, 0, 0, 0]
                center = plate.get("center") or [(plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2]
                if vehicle_box[0] <= center[0] <= vehicle_box[2] and vehicle_box[1] <= center[1] <= vehicle_box[3]:
                    matching_plates.append(plate)
            plate = max(matching_plates, key=lambda item: float(item.get("conf", item.get("confidence", 0)) or 0), default=None)

            driver = None
            for index, person in enumerate(people):
                if index in associated_people:
                    continue
                person_box = person.get("box", [0, 0, 0, 0])
                if self._intersection_ratio(person_box, vehicle_box) >= 0.25:
                    driver = person
                    associated_people.add(index)
                    break

            center = ((vehicle_box[0] + vehicle_box[2]) // 2, (vehicle_box[1] + vehicle_box[3]) // 2)
            stable_id = vehicle_id if vehicle_id >= 0 else f"{center[0]}_{center[1]}"

            # IMPORTANT:
            # A ByteTrack ID can be reused. Include a lifecycle generation
            # so a new vehicle using the same track_id is not treated as the
            # old database event forever.
            if vehicle_id >= 0:
                generation = self._get_track_generation(
                    camera_id, vehicle_id, current_time
                )
            else:
                generation = int(current_time)

            event_key = (
                f"vehicle:{camera_id}:{stable_id}:{direction}:g{generation}"
            )

            # One database event per physical track lifecycle.
            # A second DB write is allowed only when a plate/driver becomes
            # available later for the same vehicle. This avoids repeated
            # database writes while still allowing OCR to enrich an event.
            plate_text_now = _normalize_text((plate or {}).get("text") or "")
            has_driver_now = driver is not None
            previous_meta = self.saved_event_meta.get(event_key)
            is_new_event = event_key not in self.captured_tracks
            needs_enrichment = (
                not is_new_event
                and previous_meta is not None
                and (
                    (plate_text_now and not previous_meta.get("plate_text"))
                    or (has_driver_now and not previous_meta.get("has_driver"))
                )
            )
            if not is_new_event and not needs_enrichment:
                continue
            self.captured_tracks[event_key] = current_time

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
                    vehicle_crop=crop_expanded_object(frame, vehicle_box, min_width=MIN_CAPTURE_WIDTH, min_height=MIN_CAPTURE_HEIGHT),
                    has_driver=driver is not None,
                    driver_track_id=(driver or {}).get("track_id"),
                    direction=direction,
                    event_key=event_key
                )
                self.saved_event_meta[event_key] = {
                    "plate_text": plate_text_now,
                    "has_driver": has_driver_now,
                }
                print(
                    f"[DETECTION] camera={camera_id} track_id={vehicle_id} object_type=vehicle "
                    f"vehicle_type={vehicle_type} vehicle_confidence={float(vehicle.get('conf', 0) or 0):.3f} "
                    f"plate_detected={bool(plate)} plate={(plate or {}).get('text') or '-'} "
                    f"plate_confidence={float((plate or {}).get('conf', 0) or 0):.3f} "
                    f"ocr_confidence={float((plate or {}).get('ocr_conf', 0) or 0):.3f} "
                    f"driver_detected={driver is not None} direction={direction} "
                    f"event_key={event_key} duplicate={result.get('duplicate', False)}"
                )
            except Exception as exc:
                print(f"[AI STREAM ERROR] Save vehicle event failed: {exc}")

        for index, person in enumerate(people):
            if index in associated_people:
                continue
            track_id = int(person.get("track_id", -1))
            if track_id < 0:
                continue
            generation = self._get_track_generation(
                camera_id, track_id, current_time
            )
            event_key = (
                f"person:{camera_id}:{track_id}:{direction}:g{generation}"
            )

            # One database event per physical person track lifecycle.
            if event_key in self.captured_tracks:
                continue
            self.captured_tracks[event_key] = current_time
            crop = crop_expanded_object(frame, person.get("box", [0, 0, 0, 0]), min_width=MIN_CAPTURE_WIDTH, min_height=MIN_CAPTURE_HEIGHT)
            if crop is None or crop.size == 0:
                continue
            try:
                db.save_detection_event(
                    camera_id=camera_id,
                    face_crop=crop,
                    face_conf=float(person.get("conf", 0.0) or 0),
                    track_id=track_id,
                    object_type="person",
                    vehicle_type="unknown",
                    direction=direction,
                    event_key=event_key
                )
                print(f"[DETECTION] camera={camera_id} track_id={track_id} object_type=person person_confidence={float(person.get('conf', 0) or 0):.3f} direction={direction}")
            except Exception as exc:
                print(f"[AI STREAM ERROR] Save person event failed: {exc}")

    def _save_new_events(
        self,
        frame,
        person_dets,
        plate_dets,
        camera_id,
        current_time,
    ):
        """Backward-compatible entry point for the explicit event writer."""
        return self._save_consistent_events(
            frame,
            person_dets,
            plate_dets,
            camera_id,
            current_time,
        )

    def _select_dynamic_focus(self, frame, vehicle_dets, plate_raw, now):
        h,w=frame.shape[:2]
        vehicles = [
            d for d in vehicle_dets
            if d.get("cls") in (2, 3, 5, 7)
            and d.get("track_id", -1) >= 0
        ]

        # Prefer the closest perspective zone first, then object size.
        if ENABLE_DISTANCE_ZONE:
            near_vehicles = [
                d for d in vehicles
                if d.get("distance_zone") == "NEAR"
            ]
            if near_vehicles:
                vehicles = near_vehicles

        target_type = "VEHICLE"

        if not vehicles:
            people = [
                d for d in vehicle_dets
                if d.get("cls") == 0
                and d.get("track_id", -1) >= 0
            ]

            if ENABLE_DISTANCE_ZONE:
                near_people = [
                    d for d in people
                    if d.get("distance_zone") == "NEAR"
                ]
                vehicles = near_people or people
            else:
                vehicles = people

            target_type = "PERSON"
        by_id={int(d["track_id"]):d for d in vehicles}
        target=by_id.get(self.focus_track_id) if self.focus_track_id is not None else None
        if target is None and self.focus_track_id is not None and now-self.focus_last_seen < FOCUS_TARGET_HOLD_SECONDS:
            old=self.focus_info.get("box")
            if old:
                m=_lighting_metrics(frame,old); self.focus_info["lighting"]=m
                return
        if target is None and vehicles:
            # Prefer large/near vehicle; stable ByteTrack ID is then locked.
            target=max(vehicles,key=lambda d:max(1,d["box"][2]-d["box"][0])*max(1,d["box"][3]-d["box"][1]))
            self.focus_track_id=int(target["track_id"])
        if target is None:
            self.focus_track_id=None
            box=[int(w*.20),int(h*.42),int(w*.88),int(h*.96)]
            self.focus_info={"type":"AREA","box":box,"track_id":None,"lighting":_lighting_metrics(frame,box)}
            return
        self.focus_last_seen=now
        tid=int(target["track_id"]); focus_type=target_type; box=_clamp_box(target["box"],w,h,FOCUS_PADDING_VEHICLE)
        matches=[p for p in plate_raw if int(p.get("vehicle_track_id",-999))==tid and len(p.get("bbox",[]))==4]
        if matches:
            best=max(matches,key=lambda p:_safe_float(p.get("confidence",p.get("conf",0))))
            box=_clamp_box(best["bbox"],w,h,FOCUS_PADDING_PLATE); focus_type='PLATE'
            self.focus_plate_last_seen = now
        elif self.focus_info.get("type") == "PLATE" and now - self.focus_plate_last_seen < FOCUS_TARGET_HOLD_SECONDS:
            box = self.focus_info.get("box") or box
            focus_type = "PLATE"
        self.focus_info={"type":focus_type,"box":box,"track_id":tid,"lighting":_lighting_metrics(frame,box)}

    def _draw_dynamic_focus(self, frame):
        info=self.focus_info or {}; box=info.get('box')
        if not box: return
        x1,y1,x2,y2=[int(v) for v in box]; typ=info.get('type','AREA'); tid=info.get('track_id')
        color=(255,220,0) if typ=='VEHICLE' else ((0,215,255) if typ=='PLATE' else (255,200,0))
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label=f"{typ} FOCUS" + (f" ID:{tid}" if tid is not None else '')
        cv2.putText(frame,label,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2,cv2.LINE_AA)
        h,w=frame.shape[:2]; camera=(w//2,max(8,int(h*.05))); target=_center(box)
        cv2.line(frame,camera,target,color,2,cv2.LINE_AA); cv2.circle(frame,target,4,color,-1)
        m=info.get('lighting') or {}
        text=f"LIGHT {m.get('status','-')} | Score {m.get('score',0):.0f}% | Bright {m.get('brightness',0):.0f}"
        cv2.putText(frame,text,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2,cv2.LINE_AA)

    # ------------------------------------------------------------
    # PROCESS FRAME
    # ------------------------------------------------------------

    def _run_ai_pipeline(self, frame, camera_id):
        """Run one complete AI cycle in the background thread.

        This function deliberately never runs from Flask's CCTV frame
        generator. That is the key point that keeps the live CCTV responsive:
        the web request only receives/draws frames while this worker performs
        YOLO, plate detection, OCR and DB writes in the background.
        """
        camera_state = self._camera_state(camera_id)
        now = time.time()
        ai_pipeline_start = time.perf_counter()

        try:
            # ==============================================================
            # 1. LIGHTWEIGHT LIGHTING CHECK
            # ============================================================== 
            h0, w0 = frame.shape[:2]
            base_box = (
                self.focus_info.get("box")
                or [
                    int(w0 * 0.20),
                    int(h0 * 0.42),
                    int(w0 * 0.88),
                    int(h0 * 0.96),
                ]
            )
            base_light = _lighting_metrics(frame, base_box)

            # Enhancement is ONLY for AI inference. Never replace the
            # original frame used for CCTV display or captures.
            ai_frame = _enhance_for_lighting(frame, base_light)

            # ==============================================================
            # 2. YOLO + BYTETRACK
            # ============================================================== 
            person_dets = self._detect_vehicles(ai_frame, camera_id) or []

            if DEBUG_PERFORMANCE:
                print(
                    f"[AI RUN] camera={camera_id} "
                    f"objects={len(person_dets)} "
                    f"interval={self.adaptive_ai.get_interval():.2f}s"
                )

            # ==============================================================
            # 3. PLATE DETECTION -> PLATE TRACKER -> OCR
            # ============================================================== 
            plate_dets = []
            plate_raw = []

            try:
                # Plate detection is restricted to vehicle ROIs. This is much
                # cheaper than running the plate model on the full 1080p/1520p
                # CCTV frame.
                plate_raw = self._detect_plates_in_vehicles(
                    ai_frame,
                    person_dets,
                ) or []
                self.latest_raw_plate_detections = plate_raw

                # Focus follows the largest/stable vehicle and then its plate.
                self._select_dynamic_focus(
                    frame,
                    person_dets,
                    plate_raw,
                    now,
                )

                plate_tracker = camera_state["plate_tracker"]

                if self.plate_ocr is not None:
                    try:
                        # OCR/PlateTracker receives only plates that
                        # have enough pixels to read reliably.
                        ocr_candidates = [
                            p for p in plate_raw
                            if p.get("ocr_ready", False)
                        ]

                        active_tracks = plate_tracker.update(
                            ocr_candidates,
                            ai_frame,
                            self.plate_ocr,
                        ) or []
                    except Exception as exc:
                        print(f"[AI STREAM WARNING] PlateTracker/OCR failed: {exc}")
                        active_tracks = []

                    for track in active_tracks:
                        result = _track_to_result(track)
                        if result is not None:
                            plate_dets.append(result)

                    # A finished vote becomes the persisted plate capture.
                    finished = self._consume_finished_plate(
                        frame,
                        person_dets,
                        camera_id,
                    )
                    if finished is not None:
                        plate_dets.append(finished)
                else:
                    # Keep raw plate detections visible even if OCR failed to
                    # initialize. This prevents the entire plate pipeline from
                    # disappearing merely because OCR is unavailable.
                    for item in plate_raw:
                        bbox = item.get("bbox") or item.get("box")
                        if not bbox or len(bbox) != 4:
                            continue
                        conf = _safe_float(
                            item.get("confidence", item.get("conf", 0.0))
                        )
                        plate_dets.append({
                            **item,
                            "id": int(item.get("vehicle_track_id", -1)),
                            "track_id": int(item.get("vehicle_track_id", -1)),
                            "box": bbox,
                            "bbox": bbox,
                            "conf": conf,
                            "detection_confidence": conf,
                            "text": "",
                            "formatted": "",
                            "raw_text": "",
                            "ocr_conf": 0.0,
                            "confidence": 0.0,
                            "valid": False,
                            "votes": 0,
                            "total_reads": 0,
                            "distance_zone": item.get(
                                "distance_zone",
                                "NEAR",
                            ),
                            "ocr_ready": bool(
                                item.get("ocr_ready", False)
                            ),
                        })

            except Exception as exc:
                # Plate failures must not erase valid YOLO vehicle/person
                # detections or interrupt the CCTV.
                print(f"[AI STREAM ERROR] Plate pipeline failed: {exc}")

            # ==============================================================
            # 4. STORE LATEST RESULTS
            # ============================================================== 
            camera_state["last_results"] = {
                "persons": person_dets,
                "plates": plate_dets,
            }
            camera_state["last_results_time"] = now

            # ==============================================================
            # 5. SAVE DATABASE EVENTS
            # ============================================================== 
            self._save_consistent_events(
                frame,
                person_dets,
                plate_dets,
                camera_id,
                now,
            )

        except Exception as exc:
            # Never allow an AI exception to terminate the worker thread.
            print(f"[AI STREAM ERROR] Background AI cycle failed: {exc}")

        finally:
            elapsed = time.perf_counter() - ai_pipeline_start
            self.last_ai_time = elapsed
            self.adaptive_ai.update(elapsed)

            if DEBUG_PERFORMANCE:
                print(
                    f"[AI PERF] camera={camera_id} "
                    f"elapsed={elapsed * 1000:.0f}ms "
                    f"next_interval={self.adaptive_ai.get_interval():.2f}s"
                )

    def _ai_loop(self):
        """Background worker that processes only the newest CCTV frame.

        Important performance rules:
        - Never block the Flask response thread.
        - Never accumulate old frames.
        - Never run AI faster than AdaptiveAIController allows.
        - Prefer a fresh frame over a stale backlog.
        """
        print("[AI THREAD] Background AI worker started")

        while self.running:
            item = None

            # Wait briefly when there is no incoming frame. The event is set
            # by process_frame() whenever a fresh frame enters the queue.
            self.ai_wakeup.wait(timeout=0.05)
            self.ai_wakeup.clear()

            if not self.running:
                break

            with self.frame_queue_lock:
                if self.frame_queue:
                    # Keep the newest frame and discard stale queued frames.
                    item = self.frame_queue.pop()
                    self.frame_queue.clear()

            if item is None:
                continue

            frame, camera_id = item
            camera_state = self._camera_state(camera_id)
            now = time.time()
            interval = self.adaptive_ai.get_interval()

            # Respect the adaptive cadence. We intentionally do NOT sleep for
            # a full interval if another frame has not arrived yet.
            remaining = interval - (now - camera_state["last_ai_time"])
            if remaining > 0:
                # Put the newest frame back only when the queue is empty; this
                # allows the next incoming CCTV frame to replace it naturally.
                time.sleep(min(remaining, 0.05))
                with self.frame_queue_lock:
                    self.frame_queue.append((frame, camera_id))
                self.ai_wakeup.set()
                continue

            camera_state["last_ai_time"] = now
            self._run_ai_pipeline(frame, camera_id)

        print("[AI THREAD] Background AI worker stopped")
>>>>>>> Stashed changes

    def process_frame(self, frame, draw_bbox=True, camera_id=1):
        """
        Memproses 1 frame video:
        1. Menjalankan deteksi AI (YOLO + ByteTrack + OCR) pada interval ~0.2s
        2. Menyimpan hasil deteksi baru secara lokal dan mencatat ke database
        3. Menggambar bounding box bersih (hanya box & label, tanpa FPS) jika draw_bbox=True
        """
        if frame is None or frame.size == 0:
            return frame

        cam_key = int(camera_id) if camera_id else 1
        now = time.time()
        self._cleanup_old_tracks(now)

        h, w = frame.shape[:2]
        last_time = self.last_ai_times.get(cam_key, 0.0)

        # Jalankan AI inference setiap interval ~0.2s per kamera agar performa optimal
        if now - last_time >= 0.2:
            self.last_ai_times[cam_key] = now

            person_dets = []
            plate_dets = []

            # Ambil model lock agar thread-safe saat banyak kamera berjalan paralel
            with self.lock:
                # 1. Deteksi Orang / Kendaraan (Person 0, Car 2, Motorcycle 3)
                try:
                    p_results = self.yolo_person(frame, classes=[0, 2, 3], imgsz=416, verbose=False)[0]
                    if len(p_results.boxes) > 0:
                        sv_dets = sv.Detections.from_ultralytics(p_results)
                        tracker = self._get_tracker(cam_key)
                        tracked = tracker.update_with_detections(sv_dets)
                        for i in range(len(tracked)):
                            box = tracked.xyxy[i].astype(int)
                            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                            person_dets.append({"box": box, "track_id": tid, "conf": conf, "cls": cls_id})
                except Exception as e:
                    pass

                # 2. Deteksi Plat Nomor
                try:
                    plate_boxes = self.plate_detector.detect(frame)
                    for pb in plate_boxes:
                        bx = [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])]
                        p_conf = float(pb[4]) if len(pb) > 4 else 0.5

                        px1, py1, px2, py2 = max(0, bx[0]), max(0, bx[1]), min(w, bx[2]), min(h, bx[3])
                        plate_crop = frame[py1:py2, px1:px2]
                        plate_text = ""
                        ocr_conf = 0.0

                        if plate_crop.size > 0 and (px2 - px1) > 25 and (py2 - py1) > 12:
                            try:
                                plate_text, ocr_conf, _ = self.plate_ocr.read_plate(plate_crop)
                            except Exception:
                                pass

                        plate_dets.append({
                            "box": bx,
                            "conf": p_conf,
                            "text": plate_text,
                            "ocr_conf": ocr_conf,
                            "crop": plate_crop
                        })
                except Exception as e:
                    pass

            self.last_results[cam_key] = {"persons": person_dets, "plates": plate_dets}

            # 3. Simpan Deteksi Baru ke Database & Lokal
            self._save_new_events(frame, person_dets, plate_dets, cam_key, now)

        # 4. Gambar Bounding Box jika draw_bbox diaktifkan
        cam_results = self.last_results.get(cam_key, {"persons": [], "plates": []})
        if draw_bbox:
            output_frame = frame.copy()
            self._draw_clean_bboxes(output_frame, cam_results)
            return output_frame

        return frame

    def _save_new_events(self, frame, person_dets, plate_dets, camera_id, current_time):
        """Menyimpan deteksi ke database MySQL dan file lokal (dengan filter cooldown)."""
        h, w = frame.shape[:2]

        # A. Cek Plat Nomor
        for p in plate_dets:
            p_text = p.get("text")
            p_crop = p.get("crop")
            p_conf = p.get("conf", 0.0)
            ocr_conf = p.get("ocr_conf", 0.0)

            # Jika plat terbaca atau confidence bagus
            if p_text or (p_crop is not None and p_conf >= 0.45):
                cache_key = f"cam{camera_id}_plate_{p_text or id(p_crop)}"
                if cache_key not in self.captured_tracks or (current_time - self.captured_tracks[cache_key] > 5.0):
                    self.captured_tracks[cache_key] = current_time

                    # Cari person/rider yang berdekatan dengan plat ini
                    associated_person_crop = None
                    associated_person_conf = 0.0
                    for per in person_dets:
                        px1, py1, px2, py2 = per["box"]
                        bx1, by1, bx2, by2 = p["box"]
                        # Jika plat berada di area bawah orang/motor
                        if px1 <= bx2 and px2 >= bx1 and py2 >= by1 - 50:
                            p_x1, p_y1 = max(0, px1), max(0, py1)
                            p_x2, p_y2 = min(w, px2), min(h, py2)
                            associated_person_crop = frame[p_y1:p_y2, p_x1:p_x2]
                            associated_person_conf = per["conf"]
                            break

                    try:
                        db.save_detection_event(
                            camera_id=camera_id,
                            plate_number=p_text if p_text else None,
                            plate_crop=p_crop,
                            plate_conf=p_conf,
                            ocr_conf=ocr_conf,
                            face_crop=associated_person_crop,
                            face_conf=associated_person_conf
                        )
                        print(f"[AI STREAM] Saved Plate Event -> {p_text or 'Plate'} (Cam {camera_id})")
                    except Exception as e:
                        print(f"[AI STREAM ERROR] Save plate failed: {e}")

        # B. Cek Pejalan Kaki (Person yang tidak berdekatan dengan plat)
        for per in person_dets:
            tid = per.get("track_id", -1)
            cls_id = per.get("cls", 0)

            # Hanya simpan pejalan kaki (class 0 = person) yang belum dicapture
            if cls_id == 0 and tid != -1:
                cache_key = f"cam{camera_id}_person_{tid}"
                if cache_key not in self.captured_tracks or (current_time - self.captured_tracks[cache_key] > 8.0):
                    self.captured_tracks[cache_key] = current_time

                    bx1, by1, bx2, by2 = per["box"]
                    x1, y1 = max(0, bx1), max(0, by1)
                    x2, y2 = min(w, bx2), min(h, by2)
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size > 0 and (x2 - x1) >= 30 and (y2 - y1) >= 60:
                        try:
                            db.save_detection_event(
                                camera_id=camera_id,
                                plate_number=None,
                                plate_crop=None,
                                face_crop=person_crop,
                                face_conf=per["conf"],
                                track_id=tid
                            )
                            print(f"[AI STREAM] Saved Pedestrian -> Track {tid} (Cam {camera_id})")
                        except Exception as e:
                            print(f"[AI STREAM ERROR] Save person failed: {e}")

    def _draw_clean_bboxes(self, frame, results):
        """Menggambar bounding box modern & minimalis (tanpa info FPS/debug berlebihan)."""
        # 1. Gambar Person / Pengendara (Amber / Vibrant Orange)
        for per in results.get("persons", []):
            x1, y1, x2, y2 = per["box"]
            cls_id = per.get("cls", 0)
            label = "Pengendara" if cls_id in (2, 3) else "Orang"
            color = (0, 165, 255) if cls_id == 0 else (0, 200, 255)

            # Rectangle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Badge Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            bg_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Gambar Plat Nomor (Emerald Green / Neon Cyan)
        for p in results.get("plates", []):
            x1, y1, x2, y2 = p["box"]
            plate_text = p.get("text")
            label = plate_text if plate_text else "Plat Nomor"
            color = (0, 230, 118) if plate_text else (255, 215, 0)

            # Rectangle border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Badge Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            bg_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 12, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
