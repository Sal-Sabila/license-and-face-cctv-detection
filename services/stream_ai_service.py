"""
STREAM AI SERVICE

Pipeline:
    RTMP/RTSP/HTTP frame
        -> YOLO person + car + motorcycle + bus
        -> ByteTrack
        -> PlateDetector di ROI kendaraan
        -> PlateTracker
        -> PlateOCR multi-preprocessing
        -> voting antar-frame melalui PlateTracker
        -> capture
        -> database

API utama tetap kompatibel:
    service = StreamAIService.get_instance()
    frame = service.process_frame(frame, draw_bbox=True, camera_id=1)
"""

import os
import re
import time
import threading
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker
import db


# ============================================================
# PATH PROJECT
# ============================================================

# File service biasanya berada di folder ai/.
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

VEHICLE_CLASSES = [0, 2, 3, 5]  # person, car, motorcycle, bus
VEHICLE_CONFIDENCE = 0.35
VEHICLE_IMGSZ = 416

PLATE_CONFIDENCE = 0.55
PLATE_IMGSZ = 416
PLATE_MAX_DET = 5
PLATE_IOU = 0.45
PLATE_MIN_WIDTH = 30
PLATE_MIN_HEIGHT = 10

# Stream tetap hemat CPU: AI dijalankan berbasis waktu nyata.
AI_INTERVAL = 0.35

# PlateTracker melakukan OCR setiap N match.
PLATE_TRACKER_IOU = 0.40
PLATE_TRACKER_FRAME_GAP = 30
PLATE_TRACKER_OCR_EVERY = 3
PLATE_TRACKER_MIN_CONFIDENCE = 0.42
PLATE_TRACKER_MAX_HISTORY = 5

PLATE_REVIEW_CONFIDENCE = 0.60
PLATE_COOLDOWN = 30.0
PERSON_COOLDOWN = 30.0

PERSON_CAPTURE_CONFIDENCE = 0.40
MIN_PERSON_CROP_WIDTH = 25
MIN_PERSON_CROP_HEIGHT = 45

MAX_PLATE_ROIS_PER_CYCLE = 2
MAX_PLATES_PER_CYCLE = 3

# Dynamic lighting/focus
FOCUS_TARGET_HOLD_SECONDS = 1.5
FOCUS_PADDING_VEHICLE = 0.12
FOCUS_PADDING_PLATE = 0.55
LIGHT_DARK_THRESHOLD = 72.0
LIGHT_OVER_THRESHOLD = 205.0
LIGHT_GLARE_RATIO = 0.18



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

    crop = frame[y1:y2, x1:x2]
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
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in best["box"]]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None, _safe_float(best.get("conf", 0.0))
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
            raise FileNotFoundError(
                f"Model person/vehicle tidak ditemukan: {PERSON_MODEL_PATH}"
            )
        if not os.path.isfile(PLATE_MODEL_PATH):
            raise FileNotFoundError(
                f"Model plate tidak ditemukan: {PLATE_MODEL_PATH}"
            )

        print(f"[AI STREAM] YOLO    : {PERSON_MODEL_PATH}")
        self.yolo_person = YOLO(PERSON_MODEL_PATH)

        print(f"[AI STREAM] Plate   : {PLATE_MODEL_PATH}")
        self.plate_detector = PlateDetector(
            model_path=PLATE_MODEL_PATH,
            confidence=PLATE_CONFIDENCE,
            imgsz=PLATE_IMGSZ,
            device="cpu",
            max_det=PLATE_MAX_DET,
            iou=PLATE_IOU,
            min_width=PLATE_MIN_WIDTH,
            min_height=PLATE_MIN_HEIGHT,
            min_aspect_ratio=1.8,
            max_aspect_ratio=6.0,
        )

        try:
            self.plate_ocr = PlateOCR(
                scale=2.5,
                min_confidence=0.15,
                verbose=False,
                fast_mode=True,
            )
            self.plate_ocr.warmup()
            print("[AI STREAM] OCR siap")
        except Exception as exc:
            self.plate_ocr = None
            print(f"[AI STREAM WARNING] OCR tidak tersedia: {exc}")

        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
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
        self.last_results = {"persons": [], "plates": []}
        self.last_plate_history = []
        self.last_plate_capture = None
        self.processing_lock = threading.Lock()
        self.focus_track_id = None
        self.focus_last_seen = 0.0
        self.focus_info = {"type":"AREA", "box":None, "track_id":None, "lighting":None}
        self.latest_raw_plate_detections = []

        print("[AI STREAM] Vehicle classes : person/car/motorcycle/bus")
        print("[AI STREAM] Plate ROI       : kendaraan")
        print("[AI STREAM] PlateTracker    : multi-frame voting")
        print("[AI STREAM] OCR variants    : original/clahe/sharpen/threshold")
        print("=" * 75)

    # ------------------------------------------------------------
    # DETECTION
    # ------------------------------------------------------------

    def _detect_vehicles(self, frame):
        detections = []

        try:
            result = self.yolo_person(
                frame,
                classes=VEHICLE_CLASSES,
                imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONFIDENCE,
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                tracked = sv.Detections.empty()
            else:
                tracked = self.person_tracker.update_with_detections(
                    sv.Detections.from_ultralytics(result)
                )

            for i in range(len(tracked)):
                bbox = tracked.xyxy[i].astype(int).tolist()
                cls_id = (
                    int(tracked.class_id[i])
                    if tracked.class_id is not None
                    else 0
                )
                conf = (
                    _safe_float(tracked.confidence[i])
                    if tracked.confidence is not None
                    else 0.0
                )
                track_id = (
                    int(tracked.tracker_id[i])
                    if tracked.tracker_id is not None
                    else -1
                )

                detections.append({
                    "box": bbox,
                    "track_id": track_id,
                    "conf": conf,
                    "cls": cls_id,
                })

        except Exception as exc:
            print(f"[AI STREAM ERROR] Vehicle detection failed: {exc}")

        return detections

    def _detect_plates_in_vehicles(self, frame, vehicle_dets):
        h, w = frame.shape[:2]
        candidates = [
            d for d in vehicle_dets
            if d.get("cls") in (2, 3, 5)
        ]
        candidates.sort(
            key=lambda d: max(1, d["box"][2] - d["box"][0])
            * max(1, d["box"][3] - d["box"][1]),
            reverse=True,
        )

        all_plates = []

        for vehicle in candidates[:MAX_PLATE_ROIS_PER_CYCLE]:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle["box"]]
            vx1 = max(0, min(vx1, w - 1))
            vy1 = max(0, min(vy1, h - 1))
            vx2 = max(0, min(vx2, w))
            vy2 = max(0, min(vy2, h))

            if vx2 <= vx1 or vy2 <= vy1:
                continue
            if vx2 - vx1 < 50 or vy2 - vy1 < 40:
                continue

            vw = vx2 - vx1
            vh = vy2 - vy1
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
                local_plates = self.plate_detector.detect(crop) or []
            except Exception as exc:
                print(f"[AI STREAM ERROR] Plate ROI failed: {exc}")
                continue

            for plate in local_plates:
                lb = plate.get("bbox", [])
                if len(lb) != 4:
                    continue

                bx1, by1, bx2, by2 = [int(v) for v in lb]
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

                plate_crop = frame[by1:by2, bx1:bx2]
                if plate_crop.size == 0:
                    continue

                item = dict(plate)
                item["bbox"] = [bx1, by1, bx2, by2]
                item["crop"] = plate_crop
                item["vehicle_cls"] = vehicle.get("cls", 0)
                item["vehicle_track_id"] = vehicle.get("track_id", -1)
                item["vehicle_box"] = [vx1, vy1, vx2, vy2]
                all_plates.append(item)

        # Hilangkan duplikat berdasarkan jarak center.
        all_plates.sort(
            key=lambda x: _safe_float(x.get("confidence", x.get("conf", 0.0))),
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
                ecx = (eb[0] + eb[2]) / 2.0
                ecy = (eb[1] + eb[3]) / 2.0
                if ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5 < 18:
                    duplicate = True
                    break
            if not duplicate:
                final.append(candidate)

        return final

    # ------------------------------------------------------------
    # PLATE TRACK + FINAL EVENT
    # ------------------------------------------------------------

    def _consume_finished_plate(self, frame, vehicle_dets, camera_id):
        finished = self.plate_tracker.consume_latest_finished_capture()
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

        associated_crop, associated_conf = _find_vehicle_for_plate(
            frame,
            bbox,
            vehicle_dets,
        )

        # DB tetap dipanggil seperti pipeline lama.
        try:
            db.save_detection_event(
                camera_id=camera_id,
                plate_number=text if valid else None,
                plate_crop=crop,
                plate_conf=det_conf,
                ocr_conf=ocr_conf,
                face_crop=associated_crop,
                face_conf=associated_conf,
            )
        except Exception as exc:
            print(f"[AI STREAM ERROR] Save plate DB failed: {exc}")

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
        self.last_plate_history = list(self.plate_tracker.history)

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
        h, w = frame.shape[:2]

        for person in person_dets:
            if person.get("cls") != 0:
                continue
            track_id = int(person.get("track_id", -1))
            conf = _safe_float(person.get("conf", 0.0))

            if track_id < 0 or conf < PERSON_CAPTURE_CONFIDENCE:
                continue

            key = f"person_{camera_id}_{track_id}"
            last = self.captured_tracks.get(key)
            if last is not None and current_time - last < PERSON_COOLDOWN:
                continue

            x1, y1, x2, y2 = [int(v) for v in person["box"]]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            path = save_person_capture(
                frame,
                [x1, y1, x2, y2],
                track_id,
                camera_id,
            )
            if not path:
                continue

            self.captured_tracks[key] = current_time

            try:
                db.save_detection_event(
                    camera_id=camera_id,
                    plate_number=None,
                    plate_crop=None,
                    face_crop=crop,
                    face_conf=conf,
                    track_id=track_id,
                )
            except Exception as exc:
                print(f"[AI STREAM ERROR] Save person DB failed: {exc}")

            print(
                f"[PERSON CAPTURE] Cam={camera_id} "
                f"ID={track_id} conf={conf:.1%} -> "
                f"{os.path.basename(path)}"
            )

    def _select_dynamic_focus(self, frame, vehicle_dets, plate_raw, now):
        h,w=frame.shape[:2]
        vehicles=[d for d in vehicle_dets if d.get("cls") in (2,3,5) and d.get("track_id",-1)>=0]
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
        tid=int(target["track_id"]); focus_type='VEHICLE'; box=_clamp_box(target["box"],w,h,FOCUS_PADDING_VEHICLE)
        matches=[p for p in plate_raw if int(p.get("vehicle_track_id",-999))==tid and len(p.get("bbox",[]))==4]
        if matches:
            best=max(matches,key=lambda p:_safe_float(p.get("confidence",p.get("conf",0))))
            box=_clamp_box(best["bbox"],w,h,FOCUS_PADDING_PLATE); focus_type='PLATE'
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

    def process_frame(self, frame, draw_bbox=True, camera_id=1):
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return frame

        now = time.time()

        # Lock mencegah dua request Flask menjalankan model bersamaan.
        with self.processing_lock:
            should_run = (now - self.last_ai_time) >= AI_INTERVAL

            if should_run:
                self.last_ai_time = now

                # Enhancement adaptif untuk inference; output/capture tetap memakai frame asli.
                h0, w0 = frame.shape[:2]
                base_box = self.focus_info.get("box") or [int(w0*.20), int(h0*.42), int(w0*.88), int(h0*.96)]
                base_light = _lighting_metrics(frame, base_box)
                ai_frame = _enhance_for_lighting(frame, base_light)

                person_dets = self._detect_vehicles(ai_frame)
                plate_dets = []

                # PlateTracker melakukan OCR multi-frame.
                if self.plate_ocr is not None:
                    try:
                        plate_detections = self._detect_plates_in_vehicles(
                            ai_frame,
                            person_dets,
                        )
                        self.latest_raw_plate_detections = plate_detections
                        self._select_dynamic_focus(frame, person_dets, plate_detections, now)

                        active_tracks = self.plate_tracker.update(
                            plate_detections,
                            ai_frame,
                            self.plate_ocr,
                        )

                        for track in active_tracks:
                            result = _track_to_result(track)
                            if result is not None:
                                plate_dets.append(result)

                        # Event baru keluar saat tracker memfinalisasi track.
                        self._consume_finished_plate(
                            frame,
                            person_dets,
                            camera_id,
                        )

                    except Exception as exc:
                        print(f"[AI STREAM ERROR] Plate pipeline failed: {exc}")

                self.last_results = {
                    "persons": person_dets,
                    "plates": plate_dets,
                }

                self._save_person_events(
                    frame,
                    person_dets,
                    camera_id,
                    now,
                )

        if not draw_bbox:
            return frame

        output = frame.copy()
        self._draw_clean_bboxes(output, self.last_results)
        self._draw_dynamic_focus(output)
        return output

    # ------------------------------------------------------------
    # DRAW
    # ------------------------------------------------------------

    def _draw_clean_bboxes(self, frame, results):
        for obj in results.get("persons", []):
            x1, y1, x2, y2 = [int(v) for v in obj["box"]]
            cls_id = int(obj.get("cls", 0))
            conf = _safe_float(obj.get("conf", 0.0))
            track_id = int(obj.get("track_id", -1))

            if cls_id == 0:
                label = f"Orang ID:{track_id} {conf:.0%}"
                color = (0, 165, 255)
            elif cls_id == 2:
                label = f"Mobil ID:{track_id} {conf:.0%}"
                color = (0, 200, 255)
            elif cls_id == 3:
                label = f"Motor ID:{track_id} {conf:.0%}"
                color = (0, 200, 255)
            elif cls_id == 5:
                label = f"Bus ID:{track_id} {conf:.0%}"
                color = (0, 200, 255)
            else:
                label = f"Objek {conf:.0%}"
                color = (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                1,
            )
            by = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

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
                label = f"{text} OCR:{ocr_conf:.0%} V:{votes}"
                color = (0, 230, 118)
            elif text:
                label = f"Review {text} {ocr_conf:.0%}"
                color = (0, 215, 255)
            else:
                label = f"Plat YOLO:{p_conf:.0%}"
                color = (0, 215, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                2,
            )
            by = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 12, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )