"""
VIDEO AI SERVICE - FFmpeg + YOLO + PlateTracker + PaddleOCR

Pipeline:
    HEVC/H.265 video file
        -> FFmpeg raw BGR reader
        -> YOLO person/car/motorcycle/bus + ByteTrack
        -> PlateDetector di ROI kendaraan
        -> PlateTracker
        -> PlateOCR multi-preprocessing
        -> multi-frame voting
        -> capture + database
        -> output video + panel history

Default input:
    C:/Users/Lenovo/Downloads/GSMasukViewLuar.stream.mp4
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker
import db
import line_crossing


# PATH

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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VIDEO_PATH = r"C:\Users\Lenovo\Downloads\GSMasukViewLuar.stream.mp4"
OUTPUT_VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "hasil_deteksi.mp4",
)

PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plate",
    "license-plate-finetune-v3n.pt",
)

CAPTURE_DIR = os.path.join(BASE_DIR, "static", "captures")
PLATE_CAPTURE_DIR = os.path.join(CAPTURE_DIR, "plates")

CAMERA_ID = 1
CAMERA_NAME = "Video CCTV"


# AI CONFIG

VEHICLE_CLASSES = [0, 2, 3, 5, 7]
VEHICLE_CONFIDENCE = 0.35
VEHICLE_IMGSZ = 480
VEHICLE_MAX_DET = 500

PLATE_CONFIDENCE = 0.25
PLATE_IMGSZ = 512
PLATE_MAX_DET = 40
PLATE_IOU = 0.45
PLATE_MIN_WIDTH = 12
PLATE_MIN_HEIGHT = 4
PLATE_VEHICLE_MIN_WIDTH = 30
PLATE_VEHICLE_MIN_HEIGHT = 20

# Video offline: AI berbasis frame, bukan wall-clock.
# AI memproses tiap frame ketiga, sekitar 8 AI FPS pada video 25 FPS.
PROCESS_EVERY_N_FRAMES = 3
VIDEO_AI_QUEUE_SIZE = 1
VIDEO_TIMING_LOG_EVERY = 10

# PlateTracker OCR setiap 2 match agar voting mempunyai beberapa sampel.
PLATE_TRACKER_IOU = 0.35
PLATE_TRACKER_FRAME_GAP = 40
PLATE_TRACKER_OCR_EVERY = 2
PLATE_TRACKER_MIN_CONFIDENCE = 0.35
PLATE_TRACKER_MAX_HISTORY = 5

PLATE_REVIEW_CONFIDENCE = 0.65
PERSON_CAPTURE_CONFIDENCE = 0.40
VIDEO_EVENT_TRACK_MATCH_IOU = 0.70
VIDEO_EVENT_TRACK_MATCH_GAP = 1.5
VIDEO_EVENT_TRACK_REUSE_GAP = 8.0

MAX_PLATE_ROIS_PER_CYCLE = 6
MAX_PLATES_PER_CYCLE = 5


# LIGHTING / FOCUS CONFIG
# Semua koordinat ROI memakai rasio 0.0-1.0 terhadap lebar/tinggi video.
# Default diarahkan ke area tengah-bawah, tempat kendaraan/plat biasanya
# menjadi titik fokus. Sesuaikan 4 angka ini jika posisi CCTV berbeda.
FOCUS_ROI_NORMALIZED = (0.20, 0.42, 0.88, 0.96)  # x1,y1,x2,y2

# Fokus dinamis: area awal tetap dipakai sebagai zona pencarian target.
DYNAMIC_FOCUS_ENABLED = True
FOCUS_VEHICLE_PADDING_X = 0.12
FOCUS_VEHICLE_PADDING_Y = 0.16
FOCUS_PLATE_PADDING_X = 1.10
FOCUS_PLATE_PADDING_Y = 1.60
FOCUS_MIN_WIDTH = 120
FOCUS_MIN_HEIGHT = 80
FOCUS_TARGET_HOLD_FRAMES = 8
FOCUS_SWITCH_MARGIN = 0.12
FOCUS_PLATE_PRIORITY = True
FOCUS_PLATE_HOLD_FRAMES = 12
PLATE_RESULT_HOLD_FRAMES = 6
FOCUS_PLATE_HOLD_FRAMES = 12

# Titik asal beam/cahaya pada tampilan: posisi kamera relatif terhadap frame.
CAMERA_POINT_NORMALIZED = (0.50, 0.05)

# Bentuk beam sedikit lebih lebar di area fokus.
BEAM_EXPANSION = 0.16
BEAM_ALPHA = 0.12
DYNAMIC_BEAM_ALPHA = 0.16

# Target kualitas pencahayaan ROI fokus.
LIGHT_DARK_THRESHOLD = 65.0
LIGHT_BRIGHT_THRESHOLD = 205.0
LIGHT_VERY_BRIGHT_THRESHOLD = 242.0
LIGHT_LOW_CONTRAST_THRESHOLD = 24.0

# Adaptive enhancement.
ENABLE_ADAPTIVE_ENHANCEMENT = True
GAMMA_DARK = 0.72
GAMMA_VERY_DARK = 0.58
GAMMA_BRIGHT = 1.18
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)



# OUTPUT DISPLAY

TILE_WIDTH = 960
TILE_HEIGHT = 540
PANEL_WIDTH = 400
PANEL_HEIGHT = TILE_HEIGHT
GRID_WIDTH = TILE_WIDTH + PANEL_WIDTH
GRID_HEIGHT = TILE_HEIGHT
HEADER_HEIGHT = 52


# FFMPEG READER


class FFmpegVideoReader:
    """Decode video dengan FFmpeg ke frame BGR24."""

    def __init__(self, source, width=None, height=None):
        self.source = source
        self.width = width
        self.height = height
        self.fps = 25.0
        self.frame_count = 0
        self.duration = 0.0
        self.process = None
        self.frame_size = None
        self.eof = False

        self._probe()
        self._start()

    def _probe(self):
        if not os.path.isfile(self.source):
            raise FileNotFoundError(
                f"Video tidak ditemukan:\n{self.source}"
            )

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "default=noprint_wrappers=1",
            self.source,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffprobe tidak ditemukan. Pastikan FFmpeg sudah masuk PATH Windows."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ffprobe gagal membaca video:\n"
                + (exc.stderr or "")
            ) from exc

        info = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()

        self.width = int(info.get("width") or self.width or 1920)
        self.height = int(info.get("height") or self.height or 1080)

        fps_text = info.get("r_frame_rate", "25/1")
        try:
            num, den = fps_text.split("/", 1)
            self.fps = float(num) / float(den)
            if self.fps <= 0:
                raise ValueError
        except Exception:
            self.fps = 25.0

        try:
            self.duration = float(info.get("duration") or 0.0)
        except Exception:
            self.duration = 0.0

        try:
            self.frame_count = int(info.get("nb_frames") or 0)
        except Exception:
            self.frame_count = 0

        if self.frame_count <= 0 and self.duration > 0:
            self.frame_count = int(round(self.duration * self.fps))

        self.frame_size = self.width * self.height * 3

    def _start(self):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", self.source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.width}:{self.height}",
            "-an",
            "pipe:1",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size * 2,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg tidak ditemukan. Jalankan 'ffmpeg -version' di terminal."
            ) from exc

    def read(self):
        if self.process is None or self.process.stdout is None:
            return False, None

        raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            self.eof = True
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height,
            self.width,
            3,
        )
        return True, frame.copy()

    def release(self):
        if self.process is None:
            return

        try:
            if self.process.stdout:
                self.process.stdout.close()
        except Exception:
            pass

        try:
            self.process.terminate()
            self.process.wait(timeout=2)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


# HELPERS


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_plate(text):
    if text is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper().strip())


def valid_indonesian_plate(text):
    text = normalize_plate(text)
    return bool(
        text
        and 3 <= len(text) <= 9
        and re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text)
    )


def format_plate(text):
    text = normalize_plate(text)
    m = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", text)
    if not m:
        return text
    return " ".join(x for x in m.groups() if x)


def safe_filename(text):
    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        str(text or "unknown"),
    )[:80]


def save_jpg(image, directory, filename, quality=94):
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


def save_person_capture(frame, bbox, track_id, video_time):
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
    if x2 - x1 < 25 or y2 - y1 < 45:
        return None

    crop = frame[y1:y2, x1:x2]
    date_dir = os.path.join(CAPTURE_DIR, datetime.now().strftime("%Y%m%d"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = (
        f"person_cam{CAMERA_ID}_track{track_id}_"
        f"{stamp}_v{video_time:.2f}s.jpg"
    )
    return save_jpg(crop, date_dir, filename, 92)


def save_plate_capture(crop, track_id, text, video_time, prefix="plate"):
    if crop is None or crop.size == 0:
        return None

    date_dir = os.path.join(PLATE_CAPTURE_DIR, datetime.now().strftime("%Y%m%d"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    clean = safe_filename(text or "unknown")
    filename = (
        f"{prefix}_cam{CAMERA_ID}_track{track_id}_"
        f"{stamp}_v{video_time:.2f}s_{clean}.jpg"
    )
    return save_jpg(crop, date_dir, filename, 95)


def get_vehicle_crop_for_plate(frame, plate_bbox, vehicle_dets):
    if frame is None or plate_bbox is None:
        return None, 0.0, None

    px1, py1, px2, py2 = [int(v) for v in plate_bbox]
    plate_area = max(1, (px2 - px1) * (py2 - py1))

    best_crop = None
    best_score = 0.0
    best_track_id = None
    best_confidence = 0.0

    for vehicle in vehicle_dets or []:
        if vehicle.get("cls") not in (2, 3, 5, 7):
            continue

        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle["box"]]
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter = (ix2 - ix1) * (iy2 - iy1)
        score = inter / float(plate_area)
        if score <= best_score:
            continue

        h, w = frame.shape[:2]
        vx1 = max(0, min(vx1, w - 1))
        vy1 = max(0, min(vy1, h - 1))
        vx2 = max(0, min(vx2, w))
        vy2 = max(0, min(vy2, h))
        crop = frame[vy1:vy2, vx1:vx2]
        if crop.size:
            best_crop = crop
            best_score = score
            best_track_id = vehicle.get("track_id")
            best_confidence = safe_float(vehicle.get("conf", 0.0))

    return best_crop, best_confidence, best_track_id


# VIDEO AI SERVICE


class VideoAIService:
    def __init__(
        self,
        video_path,
        output_path=None,
        camera_id=CAMERA_ID,
        show_window=True,
        progress_callback=None,
        source_name=None,
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.camera_id = camera_id
        self.show_window = show_window
        self.progress_callback = progress_callback
        self.source_name = str(source_name or "Video offline").strip()[:255]

        self.session_id = uuid.uuid4().hex[:12]
        self.track_lifecycle = {}
        self.person_capture_state = {}
        self.last_results = {"persons": [], "plates": []}
        self.last_plate_result_frame = -999999
        self.last_plate_capture = None
        self.last_plate_history = []
        self.person_capture_count = 0
        self.plate_capture_count = 0
        self.plate_roi_cursor = 0

        # Status pencahayaan terbaru untuk overlay/panel.
        self.lighting = {
            "brightness": 0.0,
            "contrast": 0.0,
            "dark_ratio": 0.0,
            "bright_ratio": 0.0,
            "glare_ratio": 0.0,
            "score": 0.0,
            "status": "UNKNOWN",
            "mode": "NORMAL",
            "gamma": 1.0,
            "target_type": "STATIC",
        }

        # Target fokus dinamis.
        self.focus_bbox = None
        self.focus_point = None
        self.focus_target_type = "STATIC"
        self.focus_target_track_id = -1
        self.focus_target_conf = 0.0
        self.focus_target_plate_bbox = None
        self.focus_target_vehicle_bbox = None
        self.focus_last_seen_frame = -999999
        self.focus_plate_last_seen_frame = -999999

        self.yolo_person = None
        self.plate_detector = None
        self.plate_ocr = None
        self.person_tracker = None
        self.plate_tracker = None
        self.results_lock = threading.Lock()
        self.inference_count = 0
        self.inference_timing_totals = {}
        self.zone_classifier = None
        self.zone_points_provider = None
        self.zone_to_pixels = None
        self.video_event_groups = {"vehicle": [], "person": []}
        self.video_track_event_groups = {}

    def _get_track_generation(self, track_id, video_time):
        try:
            track_id = int(track_id)
            video_time = float(video_time)
        except (TypeError, ValueError):
            return 1
        if track_id < 0:
            return 1

        state = self.track_lifecycle.get(track_id)
        if state is None:
            state = {"generation": 1, "last_seen": video_time}
        else:
            last_seen = float(state.get("last_seen", video_time))
            gap = video_time - last_seen
            if gap < 0 or gap > 8.0:
                state["generation"] = int(state.get("generation", 1)) + 1
            state["last_seen"] = video_time

        self.track_lifecycle[track_id] = state
        return int(state["generation"])

    def _build_event_key(self, object_type, track_id, generation=1):
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            track_id = str(track_id)
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 1
        object_type = str(object_type or "unknown").lower().strip()
        return f"{object_type}:{self.camera_id}:{track_id}:gen{generation}:sess_{self.session_id}"

    # ------------------------------------------------------------
    # LIGHTING / FOCUS
    # ------------------------------------------------------------

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def _get_static_focus_bbox(self, frame):
        """Ubah Focus ROI normalized menjadi bbox pixel."""
        h, w = frame.shape[:2]
        nx1, ny1, nx2, ny2 = FOCUS_ROI_NORMALIZED

        x1 = int(self._clip01(nx1) * w)
        y1 = int(self._clip01(ny1) * h)
        x2 = int(self._clip01(nx2) * w)
        y2 = int(self._clip01(ny2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def _get_focus_bbox(self, frame):
        """Return focus dinamis; fallback ke Focus Area statis."""
        if self.focus_bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in self.focus_bbox]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            return x1, y1, x2, y2
        return self._get_static_focus_bbox(frame)

    @staticmethod
    def _bbox_center(bbox):
        if bbox is None or len(bbox) != 4:
            return None
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )

    @staticmethod
    def _bbox_area(bbox):
        if bbox is None or len(bbox) != 4:
            return 0.0
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    def _expand_bbox(self, bbox, frame_shape, pad_x, pad_y, min_width=0, min_height=0):
        """Perbesar bbox dengan batas frame."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        if pad_x < 1.0:
            px = bw * float(pad_x)
        else:
            px = bw * float(pad_x)
        if pad_y < 1.0:
            py = bh * float(pad_y)
        else:
            py = bh * float(pad_y)

        target_w = max(bw + 2.0 * px, float(min_width))
        target_h = max(bh + 2.0 * py, float(min_height))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        x1 = int(round(cx - target_w / 2.0))
        y1 = int(round(cy - target_h / 2.0))
        x2 = int(round(cx + target_w / 2.0))
        y2 = int(round(cy + target_h / 2.0))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return [x1, y1, x2, y2]

    def _focus_candidate_score(self, item, frame_shape):
        """Skor target kendaraan: confidence + ukuran + kedekatan zona fokus."""
        h, w = frame_shape[:2]
        box = item.get("box")
        if not box:
            return -1.0
        area_ratio = self._bbox_area(box) / float(max(1, w * h))
        size_score = min(1.0, area_ratio / 0.12)
        conf_score = safe_float(item.get("conf", 0.0))

        static = self._get_static_focus_bbox(np.zeros((h, w, 3), dtype=np.uint8))
        sc = self._bbox_center(static)
        bc = self._bbox_center(box)
        if sc is None or bc is None:
            proximity = 0.0
        else:
            max_dist = max(1.0, (w * w + h * h) ** 0.5)
            dist = ((bc[0] - sc[0]) ** 2 + (bc[1] - sc[1]) ** 2) ** 0.5
            proximity = max(0.0, 1.0 - dist / max_dist)

        # Kendaraan mendapat bobot lebih tinggi daripada person.
        vehicle_bonus = 0.08 if item.get("cls") in (2, 3, 5) else 0.0
        return conf_score * 0.40 + size_score * 0.35 + proximity * 0.17 + vehicle_bonus

    def _set_static_focus(self):
        self.focus_bbox = None
        self.focus_point = None
        self.focus_target_type = "STATIC"
        self.focus_target_track_id = -1
        self.focus_target_conf = 0.0
        self.focus_target_plate_bbox = None
        self.focus_target_vehicle_bbox = None

    def _update_vehicle_focus(self, frame, vehicle_dets, frame_number):
        """Pilih kendaraan utama sebagai target fokus dinamis dengan hysteresis."""
        if not DYNAMIC_FOCUS_ENABLED:
            self._set_static_focus()
            return

        candidates = [d for d in (vehicle_dets or []) if d.get("cls") in (2, 3, 5)]
        target_kind = "VEHICLE"
        if not candidates:
            # Jika tidak ada kendaraan, orang terdekat area fokus menjadi target
            # agar sistem pencahayaan tetap relevan untuk face/person detection.
            candidates = [d for d in (vehicle_dets or []) if d.get("cls") == 0]
            target_kind = "PERSON"
        if not candidates:
            if frame_number - self.focus_last_seen_frame > FOCUS_TARGET_HOLD_FRAMES:
                self._set_static_focus()
            return

        scored = [(self._focus_candidate_score(d, frame.shape), d) for d in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]

        current = None
        if self.focus_target_track_id >= 0:
            for score, item in scored:
                if int(item.get("track_id", -1)) == int(self.focus_target_track_id):
                    current = (score, item)
                    break

        if current is not None:
            current_score, current_item = current
            if current_score + FOCUS_SWITCH_MARGIN >= best_score:
                best_score, best = current_score, current_item

        box = best.get("box")
        if not box:
            return

        previous_focus_type = self.focus_target_type
        previous_focus_track_id = self.focus_target_track_id
        previous_focus_bbox = self.focus_bbox
        previous_focus_point = self.focus_point
        self.focus_bbox = self._expand_bbox(
            box,
            frame.shape,
            FOCUS_VEHICLE_PADDING_X,
            FOCUS_VEHICLE_PADDING_Y,
            FOCUS_MIN_WIDTH,
            FOCUS_MIN_HEIGHT,
        )
        self.focus_point = self._bbox_center(box)
        self.focus_target_type = target_kind
        self.focus_target_track_id = int(best.get("track_id", -1))
        self.focus_target_conf = safe_float(best.get("conf", 0.0))
        self.focus_target_vehicle_bbox = list(box)
        self.focus_last_seen_frame = frame_number

        if (
            previous_focus_type == "PLATE"
            and previous_focus_track_id == int(best.get("track_id", -1))
            and frame_number - self.focus_plate_last_seen_frame <= FOCUS_PLATE_HOLD_FRAMES
        ):
            if previous_focus_bbox is not None:
                self.focus_bbox = previous_focus_bbox
            self.focus_point = previous_focus_point
            self.focus_target_type = "PLATE"
            return

        self.focus_target_plate_bbox = None

    def _update_plate_focus(self, frame, plate_dets, frame_number):
        """Jika plat ditemukan pada target kendaraan, fokus dipindahkan ke plat."""
        if not DYNAMIC_FOCUS_ENABLED or not FOCUS_PLATE_PRIORITY:
            return
        target_tid = int(self.focus_target_track_id)
        candidates = []
        for item in plate_dets or []:
            if int(item.get("vehicle_track_id", -999)) != target_tid:
                continue
            bbox = item.get("bbox")
            if bbox and len(bbox) == 4:
                candidates.append(item)
        if not candidates:
            return

        candidates.sort(
            key=lambda x: safe_float(x.get("confidence", x.get("conf", 0.0))),
            reverse=True,
        )
        plate = candidates[0]
        bbox = plate.get("bbox")
        self.focus_bbox = self._expand_bbox(
            bbox,
            frame.shape,
            FOCUS_PLATE_PADDING_X,
            FOCUS_PLATE_PADDING_Y,
            max(80, FOCUS_MIN_WIDTH // 2),
            max(45, FOCUS_MIN_HEIGHT // 2),
        )
        self.focus_point = self._bbox_center(bbox)
        self.focus_target_type = "PLATE"
        self.focus_target_plate_bbox = list(bbox)
        self.focus_plate_last_seen_frame = frame_number

    def _analyze_lighting_bbox(self, frame, bbox):
        """Hitung kualitas cahaya pada bbox tertentu."""
        if frame is None or frame.size == 0 or bbox is None:
            return dict(self.lighting)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return dict(self.lighting)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dark_ratio = float(np.mean(gray < 35.0))
        bright_ratio = float(np.mean(gray > LIGHT_VERY_BRIGHT_THRESHOLD))
        glare_ratio = float(np.mean(gray > LIGHT_VERY_BRIGHT_THRESHOLD))

        if brightness < LIGHT_DARK_THRESHOLD:
            brightness_score = 35.0 * brightness / max(1.0, LIGHT_DARK_THRESHOLD)
        elif brightness <= 175.0:
            brightness_score = 100.0
        elif brightness <= LIGHT_BRIGHT_THRESHOLD:
            brightness_score = 100.0 - ((brightness - 175.0) / 30.0) * 35.0
        else:
            brightness_score = max(15.0, 65.0 - ((brightness - LIGHT_BRIGHT_THRESHOLD) / 50.0) * 50.0)

        contrast_score = min(100.0, max(0.0, (contrast / 55.0) * 100.0))
        glare_penalty = min(45.0, glare_ratio * 180.0)
        dark_penalty = min(35.0, dark_ratio * 70.0)
        score = (
            brightness_score * 0.55
            + contrast_score * 0.30
            + (100.0 - glare_penalty) * 0.10
            + (100.0 - dark_penalty) * 0.05
        )
        score = max(0.0, min(100.0, score))

        if brightness < 45.0 or dark_ratio > 0.45:
            status = "TOO DARK"
        elif brightness < LIGHT_DARK_THRESHOLD or contrast < LIGHT_LOW_CONTRAST_THRESHOLD:
            status = "LOW LIGHT"
        elif bright_ratio > 0.10 or brightness > LIGHT_BRIGHT_THRESHOLD:
            status = "OVEREXPOSED"
        elif glare_ratio > 0.05:
            status = "GLARE"
        elif score >= 80.0:
            status = "GOOD"
        elif score >= 60.0:
            status = "FAIR"
        else:
            status = "POOR"

        return {
            "brightness": brightness,
            "contrast": contrast,
            "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio,
            "glare_ratio": glare_ratio,
            "score": score,
            "status": status,
            "mode": "NORMAL",
            "gamma": 1.0,
            "target_type": self.focus_target_type,
        }

    def _analyze_lighting(self, frame):
        """Analisis cahaya pada focus aktif."""
        bbox = self._get_focus_bbox(frame)
        return self._analyze_lighting_bbox(frame, bbox)

    @staticmethod
    def _gamma_correct(frame, gamma):
        gamma = max(0.35, min(2.0, float(gamma)))
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255.0 for i in np.arange(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, table)

    def _adaptive_enhance(self, frame, lighting):
        """
        Enhancement adaptif untuk frame AI.

        - TOO DARK / LOW LIGHT : gamma brighten + CLAHE
        - OVEREXPOSED            : gamma sedikit darken
        - GLARE                  : gamma darken ringan
        - GOOD/FAIR              : frame asli

        Frame hasil enhancement dipakai untuk YOLO, PlateDetector dan OCR.
        Frame asli tetap dipakai untuk tampilan sehingga video tidak terlihat
        terlalu diproses.
        """
        if not ENABLE_ADAPTIVE_ENHANCEMENT:
            return frame

        status = lighting.get("status", "UNKNOWN")
        out = frame

        if status == "TOO DARK":
            gamma = GAMMA_VERY_DARK
            out = self._gamma_correct(out, gamma)
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP,
                tileGridSize=CLAHE_GRID,
            )
            l = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        elif status == "LOW LIGHT":
            gamma = GAMMA_DARK
            out = self._gamma_correct(out, gamma)
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP,
                tileGridSize=CLAHE_GRID,
            )
            l = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        elif status in ("OVEREXPOSED", "GLARE"):
            gamma = GAMMA_BRIGHT
            out = self._gamma_correct(out, gamma)
        else:
            gamma = 1.0

        lighting["mode"] = status
        lighting["gamma"] = gamma
        return out

    def _prepare_ai_frame(self, frame):
        # Analisis awal dari zona statis agar YOLO tetap dapat menemukan kendaraan
        # walaupun target dinamis belum diketahui. Setelah kendaraan ditemukan,
        # lighting akan dihitung ulang pada target tersebut.
        static_bbox = self._get_static_focus_bbox(frame)
        lighting = self._analyze_lighting_bbox(frame, static_bbox)
        lighting["target_type"] = "STATIC"
        enhanced = self._adaptive_enhance(frame, lighting)
        self.lighting = lighting
        return enhanced

    def _draw_focus_and_beam(self, frame):
        """Gambar Focus Area dinamis dan beam dari kamera ke target."""
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._get_focus_bbox(frame)
        cx, cy = self._bbox_center([x1, y1, x2, y2])
        cx, cy = int(cx), int(cy)

        cam_nx, cam_ny = CAMERA_POINT_NORMALIZED
        cam_x = int(self._clip01(cam_nx) * (w - 1))
        cam_y = int(self._clip01(cam_ny) * (h - 1))

        # Beam mengikuti target aktif. Jika target statis, gunakan beam lebar.
        expansion = BEAM_EXPANSION if self.focus_target_type == "STATIC" else 0.08
        alpha = BEAM_ALPHA if self.focus_target_type == "STATIC" else DYNAMIC_BEAM_ALPHA
        bx1 = max(0, int(x1 - (x2 - x1) * expansion))
        bx2 = min(w - 1, int(x2 + (x2 - x1) * expansion))

        polygon = np.array(
            [
                [cam_x - max(8, int(w * 0.012)), cam_y],
                [cam_x + max(8, int(w * 0.012)), cam_y],
                [bx2, y2],
                [bx1, y2],
            ],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (255, 210, 80))
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        cv2.line(frame, (cam_x, cam_y), (cx, cy), (255, 220, 80), 2, cv2.LINE_AA)
        angle = np.arctan2(cy - cam_y, cx - cam_x)
        arrow_len = max(18, int(min(w, h) * 0.035))
        ax = int(cx - np.cos(angle) * arrow_len)
        ay = int(cy - np.sin(angle) * arrow_len)
        cv2.arrowedLine(frame, (ax, ay), (cx, cy), (255, 220, 80), 3, cv2.LINE_AA, tipLength=0.35)

        color = (0, 230, 118) if self.focus_target_type == "PLATE" else (255, 220, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        if self.focus_target_type == "PLATE":
            label = f"PLATE FOCUS | Vehicle ID:{self.focus_target_track_id}"
        elif self.focus_target_type == "VEHICLE":
            label = f"VEHICLE FOCUS | ID:{self.focus_target_track_id}"
        elif self.focus_target_type == "PERSON":
            label = f"PERSON FOCUS | ID:{self.focus_target_track_id}"
        else:
            label = "CCTV FOCUS AREA"

        cv2.putText(
            frame, label, (x1 + 10, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA,
        )
        cv2.circle(frame, (cx, cy), 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cam_x, cam_y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CAMERA", (cam_x + 12, cam_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Bbox asli plat juga ditandai ketika plate focus aktif.
        if self.focus_target_plate_bbox is not None:
            pb = self.focus_target_plate_bbox
            cv2.rectangle(
                frame,
                (int(pb[0]), int(pb[1])),
                (int(pb[2]), int(pb[3])),
                (0, 230, 118),
                2,
            )

    def _draw_lighting_status(self, frame):
        lighting = self.lighting or {}
        score = safe_float(lighting.get("score", 0.0))
        brightness = safe_float(lighting.get("brightness", 0.0))
        contrast = safe_float(lighting.get("contrast", 0.0))
        glare = safe_float(lighting.get("glare_ratio", 0.0))
        status = str(lighting.get("status", "UNKNOWN"))
        mode = str(lighting.get("mode", "NORMAL"))
        gamma = safe_float(lighting.get("gamma", 1.0))

        h, w = frame.shape[:2]
        box_w = min(430, max(330, int(w * 0.26)))
        box_h = 118
        x1 = 16
        y1 = 16
        x2 = min(w - 8, x1 + box_w)
        y2 = min(h - 8, y1 + box_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 80), 2)

        cv2.putText(
            frame,
            f"LIGHTING  {status}",
            (x1 + 12, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Score {score:5.1f}% | Bright {brightness:5.1f}",
            (x1 + 12, y1 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Contrast {contrast:5.1f} | Glare {glare:.1%}",
            (x1 + 12, y1 + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Enhancement: {mode} | Gamma {gamma:.2f}",
            (x1 + 12, y1 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # MODEL LOAD
    # ------------------------------------------------------------

    def _load_models(self):
        print("=" * 75)
        print("[VIDEO AI] Inisialisasi pipeline")
        print("[VIDEO AI] FFmpeg + YOLO + PlateDetector + OCR + PlateTracker")
        print("=" * 75)

        if not os.path.isfile(PERSON_MODEL_PATH):
            raise FileNotFoundError(
                f"Model YOLO tidak ditemukan:\n{PERSON_MODEL_PATH}"
            )
        if not os.path.isfile(PLATE_MODEL_PATH):
            raise FileNotFoundError(
                f"Model plate tidak ditemukan:\n{PLATE_MODEL_PATH}"
            )

        print(f"[PERSON] {PERSON_MODEL_PATH}")
        self.yolo_person = YOLO(PERSON_MODEL_PATH)

        print(f"[PLATE] {PLATE_MODEL_PATH}")
        self.plate_detector = PlateDetector(
            PLATE_MODEL_PATH,
            confidence=PLATE_CONFIDENCE,
            imgsz=PLATE_IMGSZ,
            device="cpu",
            max_det=PLATE_MAX_DET,
            iou=PLATE_IOU,
            min_width=PLATE_MIN_WIDTH,
            min_height=PLATE_MIN_HEIGHT,
            min_aspect_ratio=1.2,
            max_aspect_ratio=8.0,
        )

        try:
            self.plate_ocr = PlateOCR(
                scale=2.5,
                min_confidence=0.15,
                verbose=False,
                fast_mode=True,
            )
            self.plate_ocr.warmup()
            print("[OCR] Siap")
        except Exception as exc:
            self.plate_ocr = None
            print(f"[OCR WARNING] OCR tidak tersedia: {exc}")

        os.makedirs(CAPTURE_DIR, exist_ok=True)
        os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # PERSON
    # ------------------------------------------------------------

    def _detect_person_vehicle(self, frame):
        detections = []

        try:
            result = self.yolo_person(
                frame,
                classes=VEHICLE_CLASSES,
                imgsz=VEHICLE_IMGSZ,
                conf=VEHICLE_CONFIDENCE,
                max_det=VEHICLE_MAX_DET,
                device="cpu",
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                raw = sv.Detections.empty()
            else:
                keep_indices = []
                for index, box in enumerate(boxes.xyxy):
                    bbox = box.cpu().numpy().astype(int).tolist()
                    zone = (
                        self.zone_classifier(
                            bbox, self.camera_id, frame.shape[1], frame.shape[0]
                        )
                        if self.zone_classifier
                        else "MID"
                    )
                    if zone != "FAR":
                        keep_indices.append(index)
                filtered_result = result[keep_indices] if keep_indices else None
                raw = (
                    sv.Detections.from_ultralytics(filtered_result)
                    if filtered_result is not None
                    else sv.Detections.empty()
                )
            tracked = self.person_tracker.update_with_detections(raw)

            for i in range(len(tracked)):
                bbox = tracked.xyxy[i].astype(int).tolist()
                zone = (
                    self.zone_classifier(bbox, self.camera_id, frame.shape[1], frame.shape[0])
                    if self.zone_classifier
                    else "MID"
                )
                if zone == "FAR":
                    continue
                cls_id = (
                    int(tracked.class_id[i])
                    if tracked.class_id is not None
                    else 0
                )
                conf = (
                    safe_float(tracked.confidence[i])
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
                    "distance_zone": zone,
                    "object_type": "vehicle" if cls_id in (2, 3, 5, 7) else "person",
                    "vehicle_type": {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}.get(cls_id, "unknown"),
                })

        except Exception as exc:
            print(f"[PERSON ERROR] {exc}")

        return detections

    def _save_person_events(self, frame, detections, video_time):
        h, w = frame.shape[:2]
        vehicle_boxes = [item["box"] for item in detections if item.get("cls") in (2, 3, 5, 7)]

        for item in detections:
            if item.get("cls") != 0:
                continue

            track_id = int(item.get("track_id", -1))
            event_track_id = int(item.get("_event_track_id", track_id))
            generation = int(item.get("_event_generation", 1))
            conf = safe_float(item.get("conf", 0.0))
            if event_track_id < 0 or conf < PERSON_CAPTURE_CONFIDENCE:
                continue

            event_key = self._build_event_key("person", event_track_id, generation)
            if event_key in self.person_capture_state:
                continue

            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if any(self._intersection_ratio(item["box"], box) >= 0.25 for box in vehicle_boxes):
                continue
            path = save_person_capture(
                frame,
                [x1, y1, x2, y2],
                track_id,
                video_time,
            )
            if not path or crop.size == 0:
                continue

            self.person_capture_count += 1

            crossing_direction = line_crossing.default_tracker.update(
                camera_id=self.camera_id,
                track_id=event_track_id,
                generation=generation,
                bbox=item["box"],
                frame_width=w,
                frame_height=h,
                camera_direction=db.get_camera_direction(self.camera_id),
            )
            previous_meta = self.person_capture_state.get(f"{event_key}:meta")
            direction_final = (
                crossing_direction
                if crossing_direction in ("entry", "exit")
                else (previous_meta or {}).get("direction", "unknown")
            )
            self.person_capture_state[f"{event_key}:meta"] = {
                "direction": direction_final,
            }
            try:
                db.save_detection_event(
                    camera_id=self.camera_id,
                    plate_number=None,
                    plate_crop=None,
                    face_crop=crop,
                    face_conf=conf,
                    track_id=event_track_id,
                    object_type="person",
                    direction=direction_final,
                    event_key=event_key,
                    source_type="video",
                    source_name=self.source_name,
                )
                self.person_capture_state[event_key] = video_time
            except Exception as exc:
                print(f"[DB PERSON ERROR] {exc}")

            print(
                f"[PERSON CAPTURE] ID={track_id} "
                f"conf={conf:.1%} -> {os.path.basename(path)}"
            )

    @staticmethod
    def _intersection_ratio(inner_box, outer_box):
        ix1 = max(inner_box[0], outer_box[0])
        iy1 = max(inner_box[1], outer_box[1])
        ix2 = min(inner_box[2], outer_box[2])
        iy2 = min(inner_box[3], outer_box[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
        return intersection / area

    @staticmethod
    def _bbox_iou(box_a, box_b):
        intersection_width = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        intersection_height = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        intersection = intersection_width * intersection_height
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _update_video_event_identities(self, detections, video_time, frame_number):
        active_ids = {"vehicle": set(), "person": set()}
        for item in detections:
            track_id = int(item.get("track_id", -1))
            object_type = item.get("object_type")
            if track_id >= 0 and object_type in active_ids:
                active_ids[object_type].add(track_id)

        for item in detections:
            track_id = int(item.get("track_id", -1))
            object_type = item.get("object_type")
            if track_id < 0 or object_type not in self.video_event_groups:
                continue

            key = (object_type, track_id)
            groups = self.video_event_groups[object_type]
            group = self.video_track_event_groups.get(key)
            if group and video_time - group["last_seen"] > VIDEO_EVENT_TRACK_REUSE_GAP:
                self.video_track_event_groups.pop(key, None)
                group = None

            if group is None:
                candidates = [
                    candidate for candidate in groups
                    if (
                        video_time - candidate["last_seen"] <= VIDEO_EVENT_TRACK_MATCH_GAP
                        and candidate["last_frame"] != frame_number
                        and candidate["last_track_id"] not in active_ids[object_type]
                        and candidate["class_id"] == int(item.get("cls", -1))
                    )
                ]
                if candidates:
                    best = max(
                        candidates,
                        key=lambda candidate: self._bbox_iou(
                            item["box"], candidate["last_bbox"]
                        ),
                    )
                    if self._bbox_iou(item["box"], best["last_bbox"]) >= VIDEO_EVENT_TRACK_MATCH_IOU:
                        group = best
                        self.video_track_event_groups[key] = group

            if group is None:
                group = {
                    "event_track_id": track_id,
                    "generation": self._get_track_generation(track_id, video_time),
                    "last_track_id": track_id,
                    "last_seen": video_time,
                    "last_frame": frame_number,
                    "last_bbox": list(item["box"]),
                    "class_id": int(item.get("cls", -1)),
                }
                groups.append(group)
                self.video_track_event_groups[key] = group
            else:
                group.update({
                    "last_track_id": track_id,
                    "last_seen": video_time,
                    "last_frame": frame_number,
                    "last_bbox": list(item["box"]),
                })

            item["_event_track_id"] = group["event_track_id"]
            item["_event_generation"] = group["generation"]

        for object_type, groups in self.video_event_groups.items():
            expired = [
                group for group in groups
                if video_time - group["last_seen"] > VIDEO_EVENT_TRACK_REUSE_GAP * 4
            ]
            if expired:
                expired_group_ids = {id(group) for group in expired}
                self.video_event_groups[object_type] = [
                    group for group in groups if id(group) not in expired_group_ids
                ]
                for key, group in list(self.video_track_event_groups.items()):
                    if id(group) in expired_group_ids:
                        self.video_track_event_groups.pop(key, None)

    def _save_vehicle_events(self, frame, detections, video_time):
        camera_orientation = db.get_camera_direction(self.camera_id)
        h_frame, w_frame = frame.shape[:2]

        for item in detections:
            if item.get("cls") not in (2, 3, 5, 7):
                continue
            track_id = int(item.get("track_id", -1))
            event_track_id = int(item.get("_event_track_id", track_id))
            generation = int(item.get("_event_generation", 1))
            if track_id < 0 or event_track_id < 0:
                continue

            crossing_direction = line_crossing.default_tracker.update(
                camera_id=self.camera_id,
                track_id=event_track_id,
                generation=generation,
                bbox=item["box"],
                frame_width=w_frame,
                frame_height=h_frame,
                camera_direction=camera_orientation,
            )

            event_key = self._build_event_key("vehicle", event_track_id, generation)
            meta_key = f"{event_key}:meta"
            previous_meta = self.person_capture_state.get(meta_key) or {}
            previous_direction = previous_meta.get("direction", "unknown")
            direction_final = (
                crossing_direction
                if crossing_direction in ("entry", "exit")
                else previous_direction
            )

            direction_newly_determined = (
                crossing_direction in ("entry", "exit")
                and previous_direction == "unknown"
            )

            if event_key in self.person_capture_state and not direction_newly_determined:
                continue

            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                continue

            self.person_capture_state[event_key] = video_time
            self.person_capture_state[meta_key] = {"direction": direction_final}
            try:
                db.save_detection_event(
                    camera_id=self.camera_id,
                    track_id=event_track_id,
                    object_type="vehicle",
                    vehicle_type=item.get("vehicle_type", "unknown"),
                    vehicle_confidence=safe_float(item.get("conf", 0.0)),
                    vehicle_crop=crop,
                    direction=direction_final,
                    event_key=event_key,
                    source_type="video",
                    source_name=self.source_name,
                )
            except Exception as exc:
                print(f"[DB VEHICLE ERROR] {exc}")

    # ------------------------------------------------------------
    # PLATE ROI
    # ------------------------------------------------------------

    def _detect_plates_in_vehicles(self, frame, vehicle_dets, video_time=0.0):
        h, w = frame.shape[:2]
        vehicles = [
            item for item in vehicle_dets
            if item.get("cls") in (2, 3, 5, 7)
        ]

        def _vehicle_order(item):
            try:
                track_id = int(item.get("_event_track_id", item.get("track_id", -1)))
            except (TypeError, ValueError):
                track_id = -1
            return (
                track_id if track_id >= 0 else 2**31,
                int(item["box"][1]),
                int(item["box"][0]),
            )

        vehicles.sort(key=_vehicle_order)

        results = []

        if vehicles:
            start = self.plate_roi_cursor % len(vehicles)
            roi_count = min(MAX_PLATE_ROIS_PER_CYCLE, len(vehicles))
            self.plate_roi_cursor = (start + roi_count) % len(vehicles)
            vehicles = vehicles[start:] + vehicles[:start]
            vehicles = vehicles[:roi_count]

        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle["box"]]
            vx1 = max(0, min(vx1, w - 1))
            vy1 = max(0, min(vy1, h - 1))
            vx2 = max(0, min(vx2, w))
            vy2 = max(0, min(vy2, h))

            if vx2 <= vx1 or vy2 <= vy1:
                continue
            if (
                vx2 - vx1 < PLATE_VEHICLE_MIN_WIDTH
                or vy2 - vy1 < PLATE_VEHICLE_MIN_HEIGHT
            ):
                continue

            vw = vx2 - vx1
            vh = vy2 - vy1
            pad_x = int(vw * 0.06)
            pad_y = int(vh * 0.10)

            rx1 = max(0, vx1 - pad_x)
            ry1 = max(0, vy1 - pad_y)
            rx2 = min(w, vx2 + pad_x)
            ry2 = min(h, vy2 + pad_y)

            vehicle_crop = frame[ry1:ry2, rx1:rx2]
            if vehicle_crop.size == 0:
                continue

            # Jika kendaraan ini adalah target fokus, lakukan enhancement lokal.
            detector_crop = vehicle_crop
            if (
                DYNAMIC_FOCUS_ENABLED
                and int(vehicle.get("track_id", -1)) == int(self.focus_target_track_id)
            ):
                local_light = self._analyze_lighting_bbox(
                    vehicle_crop,
                    (0, 0, vehicle_crop.shape[1], vehicle_crop.shape[0]),
                )
                detector_crop = self._adaptive_enhance(vehicle_crop.copy(), local_light)

            try:
                local = self.plate_detector.detect(detector_crop) or []
            except Exception as exc:
                print(f"[PLATE DETECTOR ERROR] {exc}")
                continue

            for plate in local:
                local_box = plate.get("bbox", [])
                if len(local_box) != 4:
                    continue

                bx1, by1, bx2, by2 = [int(v) for v in local_box]
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

                # Crop OCR memakai frame yang benar-benar dipakai detector.
                # Untuk target kendaraan, detector_crop sudah mendapat enhancement lokal.
                crop_source = detector_crop if int(vehicle.get("track_id", -1)) == int(self.focus_target_track_id) else frame
                local_bx1 = bx1 - rx1
                local_by1 = by1 - ry1
                local_bx2 = bx2 - rx1
                local_by2 = by2 - ry1
                if crop_source is detector_crop:
                    crop = detector_crop[local_by1:local_by2, local_bx1:local_bx2]
                else:
                    crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue

                item = dict(plate)
                item["bbox"] = [bx1, by1, bx2, by2]
                item["crop"] = crop
                item["vehicle_cls"] = vehicle.get("cls", 0)
                item["vehicle_track_id"] = vehicle.get(
                    "_event_track_id", vehicle.get("track_id", -1)
                )
                item["vehicle_box"] = [vx1, vy1, vx2, vy2]
                item["vehicle_crop"] = vehicle_crop
                vehicle_generation = vehicle.get("_event_generation")
                if vehicle_generation is None:
                    vehicle_generation = self._get_track_generation(
                        vehicle.get("track_id", -1), video_time
                    )
                item["vehicle_generation"] = vehicle_generation
                results.append(item)

        # Deduplikasi center.
        results.sort(
            key=lambda x: safe_float(x.get("confidence", x.get("conf", 0.0))),
            reverse=True,
        )

        final = []
        for candidate in results:
            bx = candidate["bbox"]
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0

            duplicate = False
            for existing in final:
                eb = existing["bbox"]
                ecx = (eb[0] + eb[2]) / 2.0
                ecy = (eb[1] + eb[3]) / 2.0
                if ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5 < 18:
                    duplicate = True
                    break

            if not duplicate:
                final.append(candidate)

        return final

    def _build_plate_display(self, detections, active_tracks):
        """Keep raw YOLO boxes visible while OCR tracking is still collecting reads."""
        plate_display = []
        for track in active_tracks:
            item = self._track_to_display(track)
            if item is not None:
                plate_display.append(item)

        for detection in detections:
            bbox = detection.get("bbox") or detection.get("box")
            if not bbox or len(bbox) != 4:
                continue
            if any(
                self._bbox_iou(bbox, item.get("bbox") or item.get("box")) >= 0.30
                for item in plate_display
                if item.get("bbox") or item.get("box")
            ):
                continue

            confidence = safe_float(
                detection.get("confidence", detection.get("conf", 0.0))
            )
            plate_display.append({
                **detection,
                "box": bbox,
                "bbox": bbox,
                "track_id": int(detection.get("vehicle_track_id", -1)),
                "conf": confidence,
                "detection_confidence": confidence,
                "text": "",
                "formatted": "",
                "raw_text": "",
                "ocr_conf": 0.0,
                "valid": False,
                "votes": 0,
                "total_reads": 0,
                "status": "detected_waiting_for_ocr",
            })
        return plate_display

    # ------------------------------------------------------------
    # PLATE RESULT
    # ------------------------------------------------------------

    def _track_to_display(self, track):
        bbox = getattr(track, "bbox", None)
        if bbox is None:
            return None

        try:
            bbox = [int(v) for v in bbox]
        except Exception:
            return None

        vote = None
        try:
            vote = track.hasil_voting()
        except Exception:
            pass

        text = ""
        formatted = ""
        ocr_conf = 0.0
        votes = 0
        total = 0

        if vote:
            text = normalize_plate(vote.get("text", ""))
            formatted = vote.get("formatted", text)
            ocr_conf = safe_float(vote.get("confidence_rata2", 0.0))
            votes = int(vote.get("jumlah_muncul", 0) or 0)
            total = int(vote.get("total_bacaan", 0) or 0)

        if not text:
            text = normalize_plate(getattr(track, "best_text", ""))
            ocr_conf = safe_float(getattr(track, "best_ocr_confidence", 0.0))
            formatted = format_plate(text)

        return {
            "track_id": int(getattr(track, "id", -1)),
            "id": int(getattr(track, "id", -1)),
            "box": bbox,
            "bbox": bbox,
            "conf": safe_float(getattr(track, "detection_confidence", 0.0)),
            "detection_confidence": safe_float(
                getattr(track, "detection_confidence", 0.0)
            ),
            "text": text,
            "formatted": formatted or format_plate(text),
            "raw_text": text,
            "ocr_conf": ocr_conf,
            "confidence": ocr_conf,
            "valid": valid_indonesian_plate(text),
            "votes": votes,
            "total_reads": total,
            "crop": getattr(track, "best_crop", None),
        }

    def _save_finished_capture(self, finished, frame, vehicle_dets, video_time):
        if finished is None:
            return None

        text = normalize_plate(
            finished.get("text") or finished.get("formatted") or ""
        )
        formatted = finished.get("formatted") or format_plate(text)
        ocr_conf = safe_float(finished.get("confidence", 0.0))
        det_conf = safe_float(finished.get("detection_confidence", 0.0))
        track_id = finished.get("track_id", "unknown")
        crop = finished.get("crop")
        bbox = finished.get("bbox")

        valid = valid_indonesian_plate(text)
        if not valid and det_conf < PLATE_REVIEW_CONFIDENCE:
            return None

        prefix = "plate" if valid else "plate_review"
        path = save_plate_capture(
            crop,
            track_id,
            formatted or text or "unknown",
            video_time,
            prefix=prefix,
        )

        vehicle_track_id = finished.get("vehicle_track_id")
        if vehicle_track_id is None or str(vehicle_track_id).strip() == "" or str(vehicle_track_id) == "-1":
            vehicle_crop, vehicle_score, found_vid = get_vehicle_crop_for_plate(
                frame,
                bbox,
                vehicle_dets,
            )
            found_vehicle = next(
                (
                    vehicle for vehicle in vehicle_dets
                    if vehicle.get("track_id") == found_vid
                ),
                None,
            )
            vehicle_track_id = (
                found_vehicle.get("_event_track_id", found_vid)
                if found_vehicle else (found_vid if found_vid is not None else track_id)
            )
            vehicle_generation = (
                found_vehicle.get("_event_generation", 1)
                if found_vehicle else 1
            )
        else:
            vehicle_score = 0.8
            vehicle_crop = finished.get("vehicle_crop")
            if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
                vehicle_crop, vehicle_score, _ = get_vehicle_crop_for_plate(
                    frame,
                    bbox,
                    vehicle_dets,
                )

        vehicle_generation = finished.get("vehicle_generation")
        if vehicle_generation is None:
            vehicle_generation = next(
                (
                    vehicle.get("_event_generation", 1)
                    for vehicle in vehicle_dets
                    if vehicle.get("_event_track_id", vehicle.get("track_id"))
                    == vehicle_track_id
                ),
                self.track_lifecycle.get(vehicle_track_id, {}).get("generation", 1),
            )

        # Event plat SELALU terhubung ke event kendaraan yang sama
        # (event_key sama dengan yang dipakai _save_vehicle_events),
        # supaya arah yang tersimpan konsisten - bukan dihitung ulang
        # dari kamera secara statis.
        event_key = self._build_event_key("vehicle", vehicle_track_id, vehicle_generation)
        previous_meta = self.person_capture_state.get(f"{event_key}:meta") or {}
        direction = previous_meta.get("direction", "unknown")

        try:
            db.save_detection_event(
                camera_id=self.camera_id,
                plate_number=text or None,
                plate_crop=crop,
                plate_conf=det_conf,
                ocr_conf=ocr_conf,
                ocr_vote_count=int(finished.get("total_bacaan", 0) or 0),
                ocr_consensus=safe_float(finished.get("consensus", 0.0)),
                plate_crop_quality=safe_float(finished.get("crop_quality", 0.0)),
                ocr_stable=bool(finished.get("is_stable", False)),
                track_id=int(vehicle_track_id) if str(vehicle_track_id).lstrip("-").isdigit() else None,
                object_type="vehicle",
                vehicle_type="unknown",
                vehicle_confidence=vehicle_score,
                vehicle_crop=vehicle_crop,
                direction=direction,
                event_key=event_key,
                source_type="video",
                source_name=self.source_name,
            )
        except Exception as exc:
            print(f"[DB PLATE ERROR] {exc}")

        result = dict(finished)
        result.update({
            "text": text,
            "formatted": formatted or format_plate(text),
            "conf": det_conf,
            "ocr_conf": ocr_conf,
            "confidence": ocr_conf,
            "valid": valid,
            "plate_image_path": path,
            "video_time": video_time,
        })

        self.last_plate_capture = result
        self.last_plate_history = list(self.plate_tracker.history)
        self.plate_capture_count += 1

        print(
            f"[PLATE RESULT] Track={track_id} "
            f"{formatted or text or 'REVIEW'} "
            f"| OCR={ocr_conf:.1%} "
            f"| YOLO={det_conf:.1%} "
            f"| capture={os.path.basename(path) if path else '-'}"
        )

        return result

    # ------------------------------------------------------------
    # PROCESS FRAME
    # ------------------------------------------------------------

    def process_frame(
        self,
        frame,
        video_time,
        frame_number,
        *,
        render=True,
        queue_wait_ms=0.0,
    ):
        if frame is None or frame.size == 0:
            return frame
        if frame_number % PROCESS_EVERY_N_FRAMES != 1 and frame_number != 1:
            return self._render(frame) if render else None

        timings = {}
        total_start = time.perf_counter()
        stage_start = total_start

        # Analisis cahaya dilakukan dari frame asli. Enhancement hanya untuk
        # jalur AI agar YOLO/PlateDetector/OCR mendapatkan frame yang lebih
        # mudah dibaca. Tampilan utama tetap frame asli + overlay.
        ai_frame = self._prepare_ai_frame(frame)
        timings["prepare"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        person_dets = self._detect_person_vehicle(ai_frame)
        timings["vehicle_yolo_tracker"] = time.perf_counter() - stage_start

        self._update_video_event_identities(
            person_dets,
            video_time,
            frame_number,
        )

        # Tahap 1: pilih kendaraan utama setelah YOLO + ByteTrack.
        stage_start = time.perf_counter()
        self._update_vehicle_focus(ai_frame, person_dets, frame_number)
        target_bbox = self._get_focus_bbox(ai_frame)
        self.lighting = self._analyze_lighting_bbox(ai_frame, target_bbox)
        self.lighting["target_type"] = self.focus_target_type
        timings["focus_lighting"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        self._save_vehicle_events(ai_frame, person_dets, video_time)
        self._save_person_events(ai_frame, person_dets, video_time)
        timings["event_capture_db"] = time.perf_counter() - stage_start

        plate_display = []

        if self.plate_detector is not None:
            stage_start = time.perf_counter()
            detections = []
            try:
                detections = self._detect_plates_in_vehicles(
                    ai_frame,
                    person_dets,
                    video_time,
                )
                timings["plate_detection"] = time.perf_counter() - stage_start
                plate_display = self._build_plate_display(detections, [])

                # Tahap 2: jika plat milik target kendaraan ditemukan,
                # pindahkan focus + beam ke plat tersebut.
            except Exception as exc:
                print(f"[PLATE DETECTION ERROR] {exc}")

            if self.plate_ocr is not None and self.plate_tracker is not None and detections:
                stage_start = time.perf_counter()
                self._update_plate_focus(ai_frame, detections, frame_number)
                target_bbox = self._get_focus_bbox(ai_frame)
                self.lighting = self._analyze_lighting_bbox(ai_frame, target_bbox)
                self.lighting["target_type"] = self.focus_target_type
                try:
                    active_tracks = self.plate_tracker.update(
                        detections,
                        ai_frame,
                        self.plate_ocr,
                    )
                    plate_display = self._build_plate_display(detections, active_tracks)

                    for finished in self.plate_tracker.consume_finished_captures():
                        self._save_finished_capture(
                            finished,
                            frame,
                            person_dets,
                            video_time,
                        )
                except Exception as exc:
                    print(f"[PLATE OCR/TRACKER ERROR] {exc}")
                timings["plate_tracking_ocr"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        if (
            not plate_display
            and self.last_results.get("plates")
            and frame_number - self.last_plate_result_frame <= PLATE_RESULT_HOLD_FRAMES
        ):
            plate_display = list(self.last_results["plates"])
        elif plate_display:
            self.last_plate_result_frame = frame_number

        with self.results_lock:
            self.last_results = {
                "persons": person_dets,
                "plates": plate_display,
            }
        timings["result_update"] = time.perf_counter() - stage_start
        timings["total"] = time.perf_counter() - total_start

        self.inference_count += 1
        for name, duration in timings.items():
            self.inference_timing_totals[name] = (
                self.inference_timing_totals.get(name, 0.0) + duration
            )
        if self.inference_count % VIDEO_TIMING_LOG_EVERY == 0:
            timings_ms = " ".join(
                f"{name}={duration * 1000:.0f}ms"
                for name, duration in timings.items()
            )
            print(
                f"[VIDEO TIMING] frame={frame_number} "
                f"queue_wait={queue_wait_ms:.0f}ms "
                f"{timings_ms} "
                f"inference_total={(time.perf_counter() - total_start) * 1000:.0f}ms"
            )

        return self._render(frame) if render else None

    # ------------------------------------------------------------
    # FLUSH ACTIVE TRACKS AT EOF
    # ------------------------------------------------------------

    def flush_plate_tracker(self, last_frame, video_time, vehicle_dets):
        if self.plate_tracker is None or last_frame is None:
            return

        # Tracker finalizes track setelah gap > max_frame_gap.
        for _ in range(PLATE_TRACKER_FRAME_GAP + 1):
            try:
                self.plate_tracker.update(
                    [],
                    last_frame,
                    self.plate_ocr,
                )
                for finished in self.plate_tracker.consume_finished_captures():
                    self._save_finished_capture(
                        finished,
                        last_frame,
                        vehicle_dets,
                        video_time,
                    )
            except Exception as exc:
                print(f"[TRACKER FLUSH ERROR] {exc}")
                break

        self.last_plate_history = list(self.plate_tracker.history)

    # ------------------------------------------------------------
    # DRAW
    # ------------------------------------------------------------

    def _draw_person_vehicle(self, frame):
        for obj in self.last_results.get("persons", []):
            x1, y1, x2, y2 = [int(v) for v in obj["box"]]
            cls_id = int(obj.get("cls", 0))
            conf = safe_float(obj.get("conf", 0.0))
            tid = int(obj.get("track_id", -1))

            if cls_id == 0:
                label = f"Orang ID:{tid} {conf:.0%}"
                color = (0, 165, 255)
            elif cls_id == 2:
                label = f"Mobil ID:{tid} {conf:.0%}"
                color = (0, 200, 255)
            elif cls_id == 3:
                label = f"Motor ID:{tid} {conf:.0%}"
                color = (0, 200, 255)
            elif cls_id == 5:
                label = f"Bus ID:{tid} {conf:.0%}"
                color = (0, 200, 255)
            else:
                label = f"Objek {conf:.0%}"
                color = (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1,
            )
            by = max(0, y1 - th - 7)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_plates(self, frame):
        for p in self.last_results.get("plates", []):
            bbox = p.get("box", p.get("bbox"))
            if not bbox:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            text = p.get("text", "")
            conf = safe_float(p.get("ocr_conf", 0.0))
            yolo_conf = safe_float(p.get("conf", 0.0))
            votes = int(p.get("votes", 0) or 0)
            valid = bool(p.get("valid", False))

            if valid and text:
                label = f"{text} OCR:{conf:.0%} V:{votes}"
                color = (0, 230, 118)
            elif text:
                label = f"Review {text} {conf:.0%}"
                color = (0, 215, 255)
            elif p.get("status") == "detected_waiting_for_ocr":
                label = f"Plat ditemukan YOLO:{yolo_conf:.0%} | OCR menunggu"
                color = (0, 215, 255)
            else:
                label = f"Plat YOLO:{yolo_conf:.0%}"
                color = (0, 215, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                2,
            )
            by = max(0, y1 - th - 7)
            cv2.rectangle(frame, (x1, by), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def _resize_tile(self, frame):
        return cv2.resize(
            frame,
            (TILE_WIDTH, TILE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def _draw_panel_title(self, panel, text, y, font_scale=0.58):
        cv2.putText(
            panel,
            text,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_panel(self, panel):
        panel[:] = (22, 22, 28)

        lighting = self.lighting or {}
        score = safe_float(lighting.get("score", 0.0))
        brightness = safe_float(lighting.get("brightness", 0.0))
        status = str(lighting.get("status", "UNKNOWN"))
        gamma = safe_float(lighting.get("gamma", 1.0))
        target_type = str(lighting.get("target_type", self.focus_target_type))

        self._draw_panel_title(panel, "LIGHTING / FOCUS", 32, 0.52)
        cv2.putText(
            panel,
            f"Target : {target_type}",
            (22, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 220, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Target ID : {self.focus_target_track_id if self.focus_target_track_id >= 0 else '-'}",
            (22, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (205, 205, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Status : {status}",
            (22, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 225, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Score  : {score:.1f}% | Bright : {brightness:.0f}",
            (22, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Enhance: {status} | Gamma {gamma:.2f}",
            (22, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 190, 200),
            1,
            cv2.LINE_AA,
        )

        self._draw_panel_title(panel, "CAPTURE PLAT TERBARU", 158, 0.52)

        latest = self.last_plate_capture
        if latest is not None:
            crop = latest.get("crop")
            text = latest.get("formatted") or latest.get("text") or "REVIEW"
            conf = safe_float(latest.get("confidence", latest.get("ocr_conf", 0.0)))
            votes = int(latest.get("jumlah_muncul", 0) or 0)
            total = int(latest.get("total_bacaan", 0) or 0)

            if crop is not None and crop.size:
                ch, cw = crop.shape[:2]
                scale = min(350 / max(1, cw), 125 / max(1, ch))
                preview = cv2.resize(
                    crop,
                    (max(1, int(cw * scale)), max(1, int(ch * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                ph, pw = preview.shape[:2]
                x = (PANEL_WIDTH - pw) // 2
                y = 173
                max_y = min(PANEL_HEIGHT, y + ph)
                max_x = min(PANEL_WIDTH, x + pw)
                panel[y:max_y, x:max_x] = preview[:max_y - y, :max_x - x]

            cv2.putText(
                panel,
                text,
                (22, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90,
                (0, 230, 118),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"OCR confidence : {conf:.1%}",
                (22, 328),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 235),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"Voting          : {votes}/{total}",
                (22, 353),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 235),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                panel,
                "Belum ada hasil final",
                (22, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 180, 190),
                1,
                cv2.LINE_AA,
            )

        history_top = 365
        self._draw_panel_title(
            panel,
            "HISTORY 5 PLAT TERAKHIR",
            history_top + 30,
            0.52,
        )

        history = self.last_plate_history or []
        for idx, item in enumerate(history[:5], start=1):
            text = item.get("formatted") or item.get("text") or "REVIEW"
            conf = safe_float(item.get("confidence", 0.0))
            votes = int(item.get("jumlah_muncul", 0) or 0)
            y = history_top + 55 + (idx - 1) * 28
            cv2.putText(
                panel,
                f"{idx}. {text}",
                (22, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"conf {conf:.0%} | vote {votes}",
                (185, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (175, 175, 185),
                1,
                cv2.LINE_AA,
            )

    def _render(self, frame):
        output = frame.copy()
        if self.zone_points_provider and self.zone_to_pixels:
            overlay = output.copy()
            for zone_name, color in (
                ("mid", (235, 145, 35)),
                ("near", (60, 190, 70)),
            ):
                points = self.zone_points_provider(self.camera_id, zone_name)
                polygon = self.zone_to_pixels(points, output.shape[1], output.shape[0])
                polygon = np.asarray(polygon, dtype=np.int32)
                if polygon.ndim != 2 or polygon.shape[0] < 3:
                    continue
                polygon = polygon.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [polygon], color)
                cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
                label_x = max(5, int(polygon[0][0][0]))
                label_y = max(20, int(polygon[0][0][1]))
                cv2.putText(
                    output,
                    f"ZONE {zone_name.upper()}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.06, output, 0.94, 0, output)
        with self.results_lock:
            self._draw_person_vehicle(output)
            self._draw_plates(output)
        return self._resize_tile(output)

    @staticmethod
    def _log_frame_stage_timings(frame_number, start_time, totals, frame_count):
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        timings_ms = " ".join(
            f"{name}={duration * 1000.0 / frame_count:.1f}ms/frame"
            for name, duration in totals.items()
        )
        print(
            f"[VIDEO FRAME TIMING] frame={frame_number} "
            f"fps={frame_number / elapsed:.1f} "
            f"{timings_ms}"
        )
        for name in totals:
            totals[name] = 0.0

    # ------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------

    def run(self):
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)

        print("=" * 75)
        print("VIDEO AI DETECTION")
        print("Person/Vehicle : YOLO + ByteTrack")
        print("Plate          : PlateDetector + PlateTracker")
        print("OCR            : PaddleOCR multi-preprocessing")
        print("Voting         : multi-frame tracker voting")
        print("Lighting       : Dynamic Vehicle/Plate Focus + adaptive enhancement")
        print("Decoder        : FFmpeg raw BGR (HEVC compatible)")
        print("=" * 75)
        print(f"[VIDEO] Input  : {self.video_path}")
        print(f"[VIDEO] Output : {self.output_path}")

        try:
            grabber = FFmpegVideoReader(self.video_path)
        except Exception as exc:
            print(f"[VIDEO ERROR] {exc}")
            raise RuntimeError(f"Gagal membuka video input: {exc}") from exc

        print(f"[VIDEO] Resolution : {grabber.width}x{grabber.height}")
        print(f"[VIDEO] FPS        : {grabber.fps:.2f}")
        print(f"[VIDEO] Duration   : {grabber.duration:.2f}s")
        print(f"[VIDEO] Frames     : {grabber.frame_count}")

        tracker_fps = max(1, int(round(grabber.fps / PROCESS_EVERY_N_FRAMES)))
        inference_queue = queue.Queue(maxsize=VIDEO_AI_QUEUE_SIZE)
        inference_stop = object()
        inference_error = []
        inference_stats = {"dropped_frames": 0}

        def run_inference():
            model_load_start = time.perf_counter()
            try:
                from services.stream_ai_service import (
                    _camera_zone_points,
                    _get_distance_zone,
                    _normalized_polygon_to_pixels,
                    load_all_zones_from_db,
                )

                load_all_zones_from_db()
                self.zone_classifier = _get_distance_zone
                self.zone_points_provider = _camera_zone_points
                self.zone_to_pixels = _normalized_polygon_to_pixels
                self._load_models()
                self.person_tracker = sv.ByteTrack(
                    track_activation_threshold=0.35,
                    lost_track_buffer=max(30, int(round(tracker_fps * 3))),
                    minimum_matching_threshold=0.7,
                    frame_rate=tracker_fps,
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
                    min_ocr_width=18,
                    min_ocr_height=6,
                )
                print(
                    f"[VIDEO TIMING] model_load="
                    f"{time.perf_counter() - model_load_start:.2f}s"
                )

                while True:
                    item = inference_queue.get()
                    if item is inference_stop:
                        break
                    frame, video_time, frame_number, queued_at = item
                    queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
                    self.process_frame(
                        frame,
                        video_time,
                        frame_number,
                        render=False,
                        queue_wait_ms=queue_wait_ms,
                    )
            except Exception as exc:
                inference_error.append(exc)
                print(f"[VIDEO INFERENCE ERROR] {exc}")

        inference_thread = threading.Thread(
            target=run_inference,
            name=f"VideoAI-{self.session_id}",
            daemon=True,
        )

        writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                grabber.fps,
                (TILE_WIDTH, TILE_HEIGHT),
            )
            if not writer.isOpened():
                grabber.release()
                raise RuntimeError(
                    f"Tidak dapat membuat output video:\n{self.output_path}"
                )

        frame_number = 0
        processed_frames = 0
        start_time = time.perf_counter()
        last_frame = None
        last_vehicle_dets = []
        inference_thread.start()
        frame_period = 1.0 / max(1.0, grabber.fps)
        next_frame_deadline = time.perf_counter()
        frame_stage_totals = {
            "decode": 0.0,
            "render": 0.0,
            "writer": 0.0,
            "callback": 0.0,
            "display": 0.0,
            "pacing": 0.0,
        }
        frames_since_timing_log = 0
        callback_every = max(1, int(round(grabber.fps / 8.0)))

        try:
            while True:
                if inference_error:
                    break

                stage_start = time.perf_counter()
                ret, frame = grabber.read()
                frame_stage_totals["decode"] += time.perf_counter() - stage_start
                if not ret:
                    break

                frame_number += 1
                video_time = max(0.0, (frame_number - 1) / grabber.fps)
                last_frame = frame

                if frame_number % PROCESS_EVERY_N_FRAMES == 1 or frame_number == 1:
                    queued_item = (
                        frame.copy(),
                        video_time,
                        frame_number,
                        time.perf_counter(),
                    )
                    try:
                        inference_queue.put_nowait(queued_item)
                    except queue.Full:
                        try:
                            inference_queue.get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            inference_stats["dropped_frames"] += 1
                        inference_queue.put_nowait(queued_item)

                stage_start = time.perf_counter()
                processed = self._render(frame)
                frame_stage_totals["render"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                if writer is not None:
                    writer.write(processed)
                frame_stage_totals["writer"] += time.perf_counter() - stage_start

                processed_frames += 1
                frames_since_timing_log += 1

                if self.progress_callback and (
                    frame_number % callback_every == 0 or frame_number == 1
                ):
                    stage_start = time.perf_counter()
                    self.progress_callback(
                        frame_number,
                        grabber.frame_count,
                        processed,
                    )
                    frame_stage_totals["callback"] += time.perf_counter() - stage_start

                if self.show_window:
                    preview = processed.copy()
                    progress = (
                        frame_number / grabber.frame_count * 100
                        if grabber.frame_count > 0
                        else 0.0
                    )
                    elapsed = time.perf_counter() - start_time
                    proc_fps = frame_number / elapsed if elapsed > 0 else 0.0

                    cv2.putText(
                        preview,
                        f"Progress: {progress:.1f}% | Proc FPS: {proc_fps:.2f}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    stage_start = time.perf_counter()
                    cv2.imshow("Video AI Detection - Q untuk keluar", preview)
                    key = cv2.waitKey(1) & 0xFF
                    frame_stage_totals["display"] += time.perf_counter() - stage_start
                    if key in (27, ord("q")):
                        print("[VIDEO AI] Dihentikan user.")
                        break

                if frame_number % 50 == 0:
                    progress = (
                        frame_number / grabber.frame_count * 100
                        if grabber.frame_count > 0
                        else 0.0
                    )
                    print(
                        f"[PROGRESS] {progress:.1f}% | "
                        f"frame={frame_number}/{grabber.frame_count} | "
                        f"t={video_time:.1f}s | "
                        f"queue_dropped={inference_stats['dropped_frames']} | "
                        f"person_capture={self.person_capture_count} | "
                        f"plate_capture={self.plate_capture_count} | "
                        f"light={self.lighting.get('score', 0.0):.0f}% "
                        f"({self.lighting.get('status', 'UNKNOWN')})"
                    )

                next_frame_deadline += frame_period
                wait_seconds = next_frame_deadline - time.perf_counter()
                if wait_seconds > 0:
                    stage_start = time.perf_counter()
                    if self.show_window:
                        key = cv2.waitKey(max(1, int(wait_seconds * 1000))) & 0xFF
                        if key in (27, ord("q")):
                            print("[VIDEO AI] Dihentikan user.")
                            break
                    else:
                        time.sleep(wait_seconds)
                    frame_stage_totals["pacing"] += time.perf_counter() - stage_start
                elif wait_seconds < -frame_period:
                    next_frame_deadline = time.perf_counter()

                if frame_number % 50 == 0:
                    self._log_frame_stage_timings(
                        frame_number,
                        start_time,
                        frame_stage_totals,
                        frames_since_timing_log,
                    )
                    frames_since_timing_log = 0

        except KeyboardInterrupt:
            print("[VIDEO AI] Ctrl+C diterima.")

        finally:
            if inference_thread.is_alive():
                while inference_thread.is_alive():
                    try:
                        inference_queue.put(inference_stop, timeout=0.1)
                        break
                    except queue.Full:
                        if inference_error:
                            break
            inference_thread.join()

            if inference_error:
                grabber.release()
                if writer is not None:
                    writer.release()
                if self.show_window:
                    cv2.destroyAllWindows()
                raise RuntimeError(
                    f"Proses inferensi video gagal: {inference_error[0]}"
                ) from inference_error[0]

            with self.results_lock:
                last_vehicle_dets = list(self.last_results.get("persons", []))

            # Finalisasi track aktif agar video terakhir tetap masuk voting/history.
            if last_frame is not None and self.plate_tracker is not None:
                self.flush_plate_tracker(
                    last_frame,
                    max(0.0, (frame_number - 1) / grabber.fps),
                    last_vehicle_dets,
                )
            if last_frame is not None:
                final_render_start = time.perf_counter()
                final_frame = self._render(last_frame)
                if self.progress_callback:
                    self.progress_callback(
                        frame_number,
                        grabber.frame_count,
                        final_frame,
                    )
                print(
                    f"[VIDEO TIMING] final_render_and_callback="
                    f"{time.perf_counter() - final_render_start:.3f}s"
                )

            grabber.release()
            if writer is not None:
                writer.release()
            if self.show_window:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time
        proc_fps = processed_frames / elapsed if elapsed > 0 else 0.0
        if frames_since_timing_log:
            self._log_frame_stage_timings(
                frame_number,
                start_time,
                frame_stage_totals,
                frames_since_timing_log,
            )

        print()
        print("=" * 75)
        print("PEMROSESAN VIDEO SELESAI")
        print("=" * 75)
        print(f"Frame diproses       : {processed_frames}")
        print(f"Waktu proses         : {elapsed:.2f} detik")
        print(f"Processing FPS       : {proc_fps:.2f}")
        print(f"Inference frames     : {self.inference_count}")
        print(f"Dropped AI frames    : {inference_stats['dropped_frames']}")
        if self.inference_count:
            timing_averages = " ".join(
                f"{name}={duration * 1000.0 / self.inference_count:.0f}ms"
                for name, duration in self.inference_timing_totals.items()
            )
            print(f"[VIDEO TIMING] inference_avg ({self.inference_count} frames): {timing_averages}")
        print(f"Person capture       : {self.person_capture_count}")
        print(f"Plate capture        : {self.plate_capture_count}")
        print(f"Capture person       : {CAPTURE_DIR}")
        print(f"Capture plate        : {PLATE_CAPTURE_DIR}")
        print(f"Output video         : {self.output_path}")
        print("=" * 75)


# MAIN

if __name__ == "__main__":
    service = VideoAIService(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        camera_id=CAMERA_ID,
        show_window=True,
    )
    service.run()
