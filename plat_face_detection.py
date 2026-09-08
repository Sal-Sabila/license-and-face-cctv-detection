import os
import sys
import time
import threading
import warnings
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from ffmpeg_stream_reader import FFmpegStreamReader
from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CCTV
# ============================================================

CAMERA_STREAMS = [
    {
        "id": 1,
        "name": "GSMasukViewLuar",
        "url": "rtmp://103.255.15.138:1935/live/GSMasukViewLuar.stream",
    },
]

STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

# Tampilan 1 CCTV besar + panel informasi di kanan
TILE_WIDTH = 960
TILE_HEIGHT = 540

PANEL_WIDTH = 400
PANEL_HEIGHT = TILE_HEIGHT

GRID_WIDTH = TILE_WIDTH + PANEL_WIDTH
GRID_HEIGHT = TILE_HEIGHT
HEADER_HEIGHT = 52


# ============================================================
# PERSON
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PERSON_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt",
)

CAPTURE_DIR = os.path.join(
    BASE_DIR,
    "captures",
)

PLATE_CAPTURE_DIR = os.path.join(
    CAPTURE_DIR,
    "plates",
)

PERSON_CONFIDENCE = 0.40
PERSON_CLASS_ID = 0
PERSON_MIN_BBOX_AREA = 1200
PERSON_INFERENCE_IMGSZ = 416

LOST_TRACK_BUFFER = 120
FRAME_RATE = 25

CAPTURE_ONCE_ONLY = True
CAPTURE_INTERVAL = 5.0
MIN_PERSON_CROP_WIDTH = 25
MIN_PERSON_CROP_HEIGHT = 45

MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 3


# ============================================================
# PLATE
# ============================================================

PLATE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plate",
    "license-plate-finetune-v2n.pt",
)

PLATE_IMGSZ = 640
PLATE_CONFIDENCE = 0.40

# Satu kamera diproses per cycle.
AI_INTERVAL = 0.20

OCR_MIN_CONFIDENCE = 0.20

PLATE_TRACKER_IOU = 0.30
PLATE_TRACKER_FRAME_GAP = 30
PLATE_TRACKER_OCR_EVERY = 3
PLATE_TRACKER_MAX_HISTORY = 5


# ============================================================
# STREAM READER
# ============================================================

class FrameGrabber:
    """Membaca stream di thread dan hanya menyimpan frame terbaru."""

    def __init__(self, source, width=1280, height=720, name="Camera"):
        self.source = source
        self.width = width
        self.height = height
        self.name = name

        self.cap = self._open_stream()
        self.frame = None
        self.running = (
            self.cap is not None
            and self.cap.isOpened()
        )

        self.lock = threading.Lock()
        self.fail_count = 0
        self.is_connected = self.running

        self._thread = threading.Thread(
            target=self._reader,
            daemon=True,
        )

    def _open_stream(self):
        if isinstance(self.source, str) and (
            self.source.startswith("rtmp://")
            or self.source.startswith("http://")
            or self.source.startswith("https://")
            or self.source.endswith(".stream")
            or self.source.endswith(".m3u8")
        ):
            return FFmpegStreamReader(
                self.source,
                width=self.width,
                height=self.height,
            )

        cap = cv2.VideoCapture(self.source)

        if cap.isOpened():
            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

        return cap

    def start(self):
        self._thread.start()
        return self

    def _reconnect(self):
        for attempt in range(
            1,
            MAX_RECONNECT_ATTEMPTS + 1,
        ):
            if self.cap is not None:
                self.cap.release()

            time.sleep(RECONNECT_DELAY)

            self.cap = self._open_stream()

            if (
                self.cap is not None
                and self.cap.isOpened()
            ):
                self.fail_count = 0
                self.is_connected = True

                print(
                    f"[CAMERA] {self.name} "
                    f"reconnect berhasil "
                    f"(attempt {attempt})"
                )

                return True

        self.is_connected = False
        return False

    def _reader(self):
        while self.running:

            if (
                self.cap is None
                or not self.cap.isOpened()
            ):
                if not self._reconnect():
                    self.running = False
                    break

                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:

                self.fail_count += 1

                if self.fail_count > 90:

                    self.is_connected = False

                    if not self._reconnect():
                        self.running = False
                        break

                time.sleep(0.02)
                continue

            self.fail_count = 0
            self.is_connected = True

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False

        if self._thread.is_alive():
            self._thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()


# ============================================================
# PERSON CAPTURE
# ============================================================

def save_person_crop(
    frame,
    bbox,
    track_id,
    timestamp,
    camera_name,
):
    x1, y1, x2, y2 = map(
        int,
        bbox,
    )

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if (
        crop_w < MIN_PERSON_CROP_WIDTH
        or crop_h < MIN_PERSON_CROP_HEIGHT
    ):
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    os.makedirs(
        CAPTURE_DIR,
        exist_ok=True,
    )

    ts = timestamp.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    clean_cam = camera_name.replace(
        " ",
        "_",
    )

    filename = (
        f"person_{clean_cam}_"
        f"{ts}_track{track_id}.jpg"
    )

    filepath = os.path.join(
        CAPTURE_DIR,
        filename,
    )

    cv2.imwrite(
        filepath,
        crop,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            92,
        ],
    )

    return filepath


# ============================================================
# PLATE CAPTURE
# ============================================================

def save_plate_crop(frame, bbox, track_id, timestamp, camera_name, plate_text=""):
    """Simpan crop plat final ke captures/plates/."""
    if frame is None or bbox is None:
        return None

    try:
        x1, y1, x2, y2 = map(int, bbox)
    except (TypeError, ValueError):
        return None

    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    os.makedirs(PLATE_CAPTURE_DIR, exist_ok=True)

    ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    clean_cam = camera_name.replace(" ", "_")
    clean_plate = "".join(
        ch if ch.isalnum() else "_"
        for ch in str(plate_text or "unknown")
    )

    filename = (
        f"plate_{clean_cam}_{ts}_track{track_id}_{clean_plate}.jpg"
    )
    filepath = os.path.join(PLATE_CAPTURE_DIR, filename)

    ok = cv2.imwrite(
        filepath,
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )

    return filepath if ok else None


def _get_capture_value(data, keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _get_finished_plate_bbox(finished_capture, active_tracks):
    """Ambil bbox dari final capture atau fallback ke active track."""
    bbox = _get_capture_value(
        finished_capture,
        ["bbox", "plate_bbox", "box", "xyxy"],
        None,
    )
    if bbox is not None:
        return bbox

    track_id = _get_capture_value(
        finished_capture,
        ["track_id", "tracker_id", "id"],
        None,
    )
    if track_id is None:
        return None

    try:
        track_id = int(track_id)
    except (TypeError, ValueError):
        return None

    for track in active_tracks or []:
        try:
            if int(getattr(track, "id", -999999)) == track_id:
                return getattr(track, "bbox", None)
        except (TypeError, ValueError):
            continue

    return None


# ============================================================
# DRAW
# ============================================================

def scale_bbox_to_tile(
    bbox,
    original_width,
    original_height,
):
    x1, y1, x2, y2 = bbox

    sx = (
        TILE_WIDTH
        / original_width
    )

    sy = (
        TILE_HEIGHT
        / original_height
    )

    return [
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    ]


def draw_persons(
    tile,
    tracked,
    capture_state,
):
    for i in range(len(tracked)):

        x1, y1, x2, y2 = (
            tracked.xyxy[i].astype(int)
        )

        track_id = (
            int(tracked.tracker_id[i])
            if tracked.tracker_id is not None
            else -1
        )

        conf = (
            float(tracked.confidence[i])
            if tracked.confidence is not None
            else 0.0
        )

        state = capture_state.get(
            track_id
        )

        if state:
            if (
                time.time()
                - state["last_capture_time"]
                < 1.5
            ):
                color = (0, 255, 255)
                status = "CAPTURED!"
            else:
                color = (255, 200, 0)
                status = "saved"
        else:
            color = (0, 255, 0)
            status = "tracking"

        cv2.rectangle(
            tile,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"PERSON ID:{track_id} "
            f"{conf:.0%} [{status}]"
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )

        label_y = max(
            0,
            y1 - th - 6,
        )

        cv2.rectangle(
            tile,
            (x1, label_y),
            (x1 + tw + 4, y1),
            color,
            -1,
        )

        cv2.putText(
            tile,
            label,
            (
                x1 + 2,
                max(12, y1 - 3),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def draw_plates(
    tile,
    active_tracks,
    original_width,
    original_height,
):
    for track in active_tracks:

        bbox = getattr(
            track,
            "bbox",
            None,
        )

        if bbox is None:
            continue

        x1, y1, x2, y2 = (
            scale_bbox_to_tile(
                bbox,
                original_width,
                original_height,
            )
        )

        hasil = track.hasil_voting()

        if hasil is not None:

            plate_text = hasil.get(
                "formatted",
                hasil.get(
                    "text",
                    "",
                ),
            )

            confidence = float(
                hasil.get(
                    "confidence_rata2",
                    0.0,
                )
                or 0.0
            )

            label = (
                f"PLATE ID:{track.id} "
                f"{plate_text} "
                f"{confidence:.0%}"
            )

        else:

            label = (
                f"PLATE ID:{track.id} "
                "Membaca..."
            )

        color = (0, 255, 0)

        cv2.rectangle(
            tile,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            2,
        )

        label_y = max(
            th + 5,
            y1,
        )

        cv2.rectangle(
            tile,
            (
                x1,
                label_y - th - 6,
            ),
            (
                x1 + tw + 8,
                label_y + 3,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            tile,
            label,
            (
                x1 + 4,
                label_y - 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_camera_tile(
    frame,
    camera,
):
    tile = cv2.resize(
        frame,
        (
            TILE_WIDTH,
            TILE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    draw_persons(
        tile,
        camera["person_tracked"],
        camera["person_capture_state"],
    )

    draw_plates(
        tile,
        camera["plate_active_tracks"],
        frame.shape[1],
        frame.shape[0],
    )

    banner_h = 30

    overlay = tile.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (TILE_WIDTH, banner_h),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        tile,
        0.22,
        0,
        tile,
    )

    status_color = (
        (0, 255, 0)
        if camera["grabber"].is_connected
        else (0, 0, 255)
    )

    cv2.circle(
        tile,
        (12, 15),
        5,
        status_color,
        -1,
    )

    title = (
        f"CAM {camera['info']['id']}: "
        f"{camera['info']['name']}"
    )

    cv2.putText(
        tile,
        title,
        (24, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    person_count = len(
        camera["person_tracked"]
    )

    plate_count = len(
        camera["plate_active_tracks"]
    )

    stats = (
        f"Orang: {person_count} | "
        f"Plat: {plate_count} | "
        f"Cap: {camera['person_total_captured']}"
    )

    cv2.putText(
        tile,
        stats,
        (
            250,
            20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        tile,
        (0, 0),
        (
            TILE_WIDTH - 1,
            TILE_HEIGHT - 1,
        ),
        (50, 50, 50),
        1,
    )

    return tile


def create_placeholder_tile(
    cam_info,
    message="Menghubungkan ke stream...",
):
    tile = np.zeros(
        (
            TILE_HEIGHT,
            TILE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    tile[:] = (
        25,
        25,
        28,
    )

    cv2.rectangle(
        tile,
        (0, 0),
        (TILE_WIDTH, 30),
        (15, 15, 18),
        -1,
    )

    cv2.circle(
        tile,
        (12, 15),
        5,
        (0, 165, 255),
        -1,
    )

    cv2.putText(
        tile,
        (
            f"CAM {cam_info['id']}: "
            f"{cam_info['name']}"
        ),
        (24, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        message,
        (
            TILE_WIDTH // 2 - 130,
            TILE_HEIGHT // 2,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 180, 255),
        1,
        cv2.LINE_AA,
    )

    return tile



def _first_value(data, keys, default=None):
    """Ambil nilai dari dict dengan beberapa kemungkinan nama key."""
    if not isinstance(data, dict):
        return default
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _plate_text(data):
    value = _first_value(
        data,
        ["formatted", "text", "plate_number", "plate", "hasil"],
        "",
    )
    return str(value) if value is not None else ""


def _plate_confidence(data):
    value = _first_value(
        data,
        [
            "confidence_rata2",
            "ocr_confidence",
            "ocr_conf",
            "confidence",
        ],
        0.0,
    )
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0

    # Jika sumber menyimpan confidence 0-100, ubah ke 0-1.
    if value > 1.0:
        value /= 100.0

    return max(0.0, min(1.0, value))


def _plate_crop_from_capture(capture):
    """Cari crop plat dari hasil tracker jika tersedia."""
    if not isinstance(capture, dict):
        return None

    # Bisa berupa numpy image.
    for key in ["crop", "plate_crop", "image", "plate_image"]:
        value = capture.get(key)
        if isinstance(value, np.ndarray) and value.size > 0:
            return value

    # Bisa berupa path file.
    for key in [
        "image_path",
        "plate_image_path",
        "crop_path",
        "filepath",
        "file_path",
        "path",
    ]:
        value = capture.get(key)
        if isinstance(value, str) and os.path.exists(value):
            image = cv2.imread(value)
            if image is not None:
                return image

    return None


def _draw_panel_title(panel, text, y, font_scale=0.58):
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


def draw_side_panel(camera):
    """
    Panel kanan:
    - Capture plat terbaru
    - Nomor plat
    - OCR confidence
    - YOLO confidence jika tersedia
    - Track ID
    - Waktu
    - History 5 plat terakhir
    """
    panel = np.zeros(
        (PANEL_HEIGHT, PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    panel[:] = (18, 18, 22)

    # Border panel
    cv2.rectangle(
        panel,
        (1, 1),
        (PANEL_WIDTH - 2, PANEL_HEIGHT - 2),
        (55, 55, 65),
        1,
    )

    # ========================================================
    # CAPTURE PLAT TERBARU
    # ========================================================
    cv2.rectangle(
        panel,
        (12, 12),
        (PANEL_WIDTH - 12, 285),
        (35, 35, 42),
        1,
    )

    _draw_panel_title(
        panel,
        "CAPTURE PLAT TERBARU",
        40,
        0.52,
    )

    latest = camera.get("plate_latest_capture")

    # Area gambar plat
    image_x1, image_y1 = 28, 58
    image_x2, image_y2 = PANEL_WIDTH - 28, 172

    cv2.rectangle(
        panel,
        (image_x1, image_y1),
        (image_x2, image_y2),
        (65, 65, 75),
        1,
    )

    crop = _plate_crop_from_capture(latest)

    if crop is not None:
        try:
            crop_h, crop_w = crop.shape[:2]
            box_w = image_x2 - image_x1
            box_h = image_y2 - image_y1

            scale = min(box_w / crop_w, box_h / crop_h)
            new_w = max(1, int(crop_w * scale))
            new_h = max(1, int(crop_h * scale))

            resized = cv2.resize(
                crop,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

            px = image_x1 + (box_w - new_w) // 2
            py = image_y1 + (box_h - new_h) // 2

            panel[py:py + new_h, px:px + new_w] = resized
        except Exception:
            cv2.putText(
                panel,
                "Preview tidak tersedia",
                (85, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            panel,
            "Menunggu hasil plat...",
            (92, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 140, 150),
            1,
            cv2.LINE_AA,
        )

    if latest is not None:
        plate = _plate_text(latest)
        ocr_conf = _plate_confidence(latest)

        yolo_conf = _first_value(
            latest,
            [
                "yolo_confidence",
                "detection_confidence",
                "det_confidence",
                "yolo_conf",
            ],
            None,
        )

        if yolo_conf is not None:
            try:
                yolo_conf = float(yolo_conf)
                if yolo_conf > 1.0:
                    yolo_conf /= 100.0
            except (TypeError, ValueError):
                yolo_conf = None

        track_id = _first_value(
            latest,
            ["track_id", "id", "tracker_id"],
            "-",
        )

        timestamp = _first_value(
            latest,
            ["timestamp", "detected_at", "time", "datetime"],
            None,
        )

        cv2.putText(
            panel,
            f"Plat       : {plate or '-'}",
            (22, 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            panel,
            f"OCR Conf   : {ocr_conf:.1%}",
            (22, 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 80),
            1,
            cv2.LINE_AA,
        )

        if yolo_conf is not None:
            cv2.putText(
                panel,
                f"YOLO Conf  : {yolo_conf:.1%}",
                (205, 226),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            panel,
            f"Track      : #{track_id}",
            (22, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if timestamp is not None:
            timestamp = str(timestamp)
            if len(timestamp) > 25:
                timestamp = timestamp[:25]

            cv2.putText(
                panel,
                timestamp,
                (22, 274),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (160, 160, 170),
                1,
                cv2.LINE_AA,
            )

    # ========================================================
    # HISTORY 5 PLAT TERAKHIR
    # ========================================================
    history_top = 302

    cv2.rectangle(
        panel,
        (12, history_top),
        (PANEL_WIDTH - 12, PANEL_HEIGHT - 12),
        (35, 35, 42),
        1,
    )

    _draw_panel_title(
        panel,
        "HISTORY 5 PLAT TERAKHIR",
        history_top + 30,
        0.52,
    )

    history = camera.get("plate_history") or []

    # Tampilkan 5 terakhir, terbaru di atas.
    history = list(history)[-5:][::-1]

    if not history:
        cv2.putText(
            panel,
            "Belum ada hasil OCR",
            (90, history_top + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (130, 130, 140),
            1,
            cv2.LINE_AA,
        )
    else:
        y = history_top + 72

        for idx, item in enumerate(history, start=1):
            plate = _plate_text(item) or "-"
            conf = _plate_confidence(item)

            track_id = _first_value(
                item,
                ["track_id", "id", "tracker_id"],
                "-",
            )

            label = f"{idx}. #{track_id}  {plate}"

            cv2.putText(
                panel,
                label[:35],
                (22, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                f"OCR {conf:.1%}",
                (42, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 220, 80),
                1,
                cv2.LINE_AA,
            )

            y += 48

            if y > PANEL_HEIGHT - 20:
                break

    return panel


def draw_combined_display(frame, camera):
    """Live CCTV di kiri + panel hasil plat di kanan."""
    live = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    # Bounding box PERSON + PLATE tetap digambar di live CCTV.
    draw_persons(
        live,
        camera["person_tracked"],
        camera["person_capture_state"],
    )

    draw_plates(
        live,
        camera["plate_active_tracks"],
        frame.shape[1],
        frame.shape[0],
    )

    # Banner kamera.
    overlay = live.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (TILE_WIDTH, 34),
        (15, 15, 18),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.78,
        live,
        0.22,
        0,
        live,
    )

    status_color = (
        (0, 255, 0)
        if camera["grabber"].is_connected
        else (0, 0, 255)
    )

    cv2.circle(
        live,
        (14, 17),
        6,
        status_color,
        -1,
    )

    title = (
        f"CAM {camera['info']['id']}: "
        f"{camera['info']['name']}"
    )

    cv2.putText(
        live,
        title,
        (28, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    person_count = len(camera["person_tracked"])
    plate_count = len(camera["plate_active_tracks"])

    cv2.putText(
        live,
        f"Orang: {person_count} | Plat: {plate_count}",
        (TILE_WIDTH - 260, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 235, 235),
        1,
        cv2.LINE_AA,
    )

    panel = draw_side_panel(camera)

    return np.hstack([live, panel])


def draw_master_header(
    total_people,
    total_plates,
    total_captures,
    total_plate_captures,
    fps,
    active_camera,
):
    header = np.zeros(
        (
            HEADER_HEIGHT,
            GRID_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    header[:] = (
        18,
        18,
        22,
    )

    cv2.putText(
        header,
        "CCTV AI MONITORING - PERSON + PLATE",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stats = (
        f"Orang aktif: {total_people} | "
        f"Plat aktif: {total_plates} | "
        f"Orang total: {total_captures} | "
        f"Plat total: {total_plate_captures} | "
        f"AI: {active_camera} | "
        f"FPS: {fps:.1f} | [Q] Keluar"
    )

    cv2.putText(
        header,
        stats,
        (550, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.line(
        header,
        (0, HEADER_HEIGHT - 1),
        (
            GRID_WIDTH,
            HEADER_HEIGHT - 1,
        ),
        (60, 60, 70),
        1,
    )

    return header


# ============================================================
# MAIN
# ============================================================

def main():
    global running

    running = True

    os.makedirs(
        CAPTURE_DIR,
        exist_ok=True,
    )

    os.makedirs(
        PLATE_CAPTURE_DIR,
        exist_ok=True,
    )

    print("=" * 75)
    print("MULTI-CCTV PERSON + PLATE DETECTION")
    print("1 CCTV | Person YOLO + ByteTrack | Plate YOLO + OCR")
    print("=" * 75)

    # --------------------------------------------------------
    # Person model
    # --------------------------------------------------------

    print(
        f"[PERSON] Loading model: "
        f"{PERSON_MODEL_PATH}"
    )

    try:
        person_model = YOLO(
            PERSON_MODEL_PATH
        )
    except Exception as exc:
        print(
            f"[PERSON ERROR] "
            f"Gagal memuat model: {exc}"
        )
        return

    print("[PERSON] Model siap.")

    # --------------------------------------------------------
    # Plate detector + OCR
    # --------------------------------------------------------

    print(
        f"[PLATE] Loading model: "
        f"{PLATE_MODEL_PATH}"
    )

    try:

        plate_detector = PlateDetector(
            model_path=PLATE_MODEL_PATH,
            confidence=PLATE_CONFIDENCE,
            imgsz=PLATE_IMGSZ,
        )

        plate_ocr = PlateOCR(
            min_confidence=OCR_MIN_CONFIDENCE
        )

    except Exception as exc:

        print(
            f"[PLATE ERROR] "
            f"Gagal memuat detector/OCR: {exc}"
        )
        return

    print(
        "[PLATE] Detector + OCR siap."
    )

    # --------------------------------------------------------
    # Cameras
    # --------------------------------------------------------

    cameras = []

    print(
        "[CAMERA] Membuka 1 stream CCTV..."
    )

    for info in CAMERA_STREAMS:

        print(
            f"  -> CAM {info['id']}: "
            f"{info['name']} "
            f"({info['url']})"
        )

        grabber = FrameGrabber(
            info["url"],
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            name=info["name"],
        ).start()

        person_tracker = sv.ByteTrack(
            lost_track_buffer=LOST_TRACK_BUFFER,
            frame_rate=FRAME_RATE,
        )

        plate_tracker = PlateTracker(
            iou_threshold=PLATE_TRACKER_IOU,
            max_frame_gap=PLATE_TRACKER_FRAME_GAP,
            ocr_every_n_matches=PLATE_TRACKER_OCR_EVERY,
            min_final_confidence=OCR_MIN_CONFIDENCE,
            max_history=PLATE_TRACKER_MAX_HISTORY,
        )

        cameras.append({
            "info": info,
            "grabber": grabber,

            "person_tracker": person_tracker,
            "person_tracked": sv.Detections.empty(),
            "person_capture_state": {},
            "person_total_captured": 0,

            "plate_tracker": plate_tracker,
            "plate_active_tracks": [],
            "plate_history": [],
            "plate_latest_capture": None,
            "plate_total_detected": 0,
        })

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    window_name = (
        "CCTV AI Monitoring - Person + Plate "
        "(1 CCTV) - Q untuk keluar"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        GRID_WIDTH,
        GRID_HEIGHT + HEADER_HEIGHT,
    )

    print(
        "\n[INFO] Menunggu frame..."
    )

    time.sleep(1.0)

    total_loop_frames = 0
    start_time = time.time()

    # Satu CCTV saja.
    active_cam_idx = 0

    try:

        while running:

            total_loop_frames += 1
            now = datetime.now()

            # ------------------------------------------------
            # Baca frame terbaru dari semua kamera
            # ------------------------------------------------

            current_frames = []

            for cam in cameras:
                current_frames.append(
                    cam["grabber"].read()
                )

            # ------------------------------------------------
            # AI round-robin
            # Person + plate pada kamera yang sama
            # ------------------------------------------------

            target_cam = cameras[
                active_cam_idx
            ]

            target_frame = current_frames[
                active_cam_idx
            ]

            if target_frame is not None:

                original_h, original_w = (
                    target_frame.shape[:2]
                )

                # ============================================
                # PERSON YOLO
                # ============================================

                try:

                    results = person_model(
                        target_frame,
                        imgsz=PERSON_INFERENCE_IMGSZ,
                        classes=[PERSON_CLASS_ID],
                        conf=PERSON_CONFIDENCE,
                        verbose=False,
                    )[0]

                    boxes = results.boxes

                    if len(boxes) > 0:

                        xyxy = (
                            boxes.xyxy
                            .cpu()
                            .numpy()
                        )

                        confs = (
                            boxes.conf
                            .cpu()
                            .numpy()
                        )

                        class_ids = (
                            boxes.cls
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        areas = (
                            xyxy[:, 2]
                            - xyxy[:, 0]
                        ) * (
                            xyxy[:, 3]
                            - xyxy[:, 1]
                        )

                        valid = (
                            areas
                            >= PERSON_MIN_BBOX_AREA
                        )

                        xyxy = xyxy[valid]
                        confs = confs[valid]
                        class_ids = class_ids[valid]

                        detections = sv.Detections(
                            xyxy=xyxy.astype(
                                np.float32
                            ),
                            confidence=confs.astype(
                                np.float32
                            ),
                            class_id=class_ids,
                        )

                    else:

                        detections = (
                            sv.Detections.empty()
                        )

                    if len(detections) > 0:

                        tracked = (
                            target_cam[
                                "person_tracker"
                            ].update_with_detections(
                                detections
                            )
                        )

                    else:

                        tracked = (
                            sv.Detections.empty()
                        )

                    target_cam[
                        "person_tracked"
                    ] = tracked

                    # ========================================
                    # Person capture
                    # ========================================

                    for i in range(
                        len(tracked)
                    ):

                        track_id = (
                            int(
                                tracked.tracker_id[i]
                            )
                            if tracked.tracker_id
                            is not None
                            else -1
                        )

                        if track_id < 0:
                            continue

                        bbox = (
                            tracked.xyxy[i]
                        )

                        conf = (
                            float(
                                tracked.confidence[i]
                            )
                            if tracked.confidence
                            is not None
                            else 0.0
                        )

                        state = (
                            target_cam[
                                "person_capture_state"
                            ].get(track_id)
                        )

                        now_ts = time.time()

                        if CAPTURE_ONCE_ONLY:

                            should_capture = (
                                state is None
                            )

                        else:

                            should_capture = (
                                state is None
                                or (
                                    now_ts
                                    - state[
                                        "last_capture_time"
                                    ]
                                    >= CAPTURE_INTERVAL
                                )
                            )

                        if should_capture:

                            saved_path = (
                                save_person_crop(
                                    target_frame,
                                    bbox,
                                    track_id,
                                    now,
                                    target_cam[
                                        "info"
                                    ]["name"],
                                )
                            )

                            if saved_path:

                                target_cam[
                                    "person_total_captured"
                                ] += 1

                                count = (
                                    state[
                                        "capture_count"
                                    ] + 1
                                    if state
                                    else 1
                                )

                                target_cam[
                                    "person_capture_state"
                                ][track_id] = {
                                    "last_capture_time":
                                        now_ts,
                                    "capture_count":
                                        count,
                                }

                                print(
                                    f"[PERSON CAPTURE] "
                                    f"[{target_cam['info']['name']}] "
                                    f"ID={track_id} "
                                    f"conf={conf:.0%} -> "
                                    f"{os.path.basename(saved_path)}"
                                )

                except Exception as exc:

                    print(
                        f"[PERSON ERROR] "
                        f"CAM {target_cam['info']['id']}: "
                        f"{exc}"
                    )

                # ============================================
                # PLATE YOLO + OCR
                # ============================================

                try:

                    plate_detections = (
                        plate_detector.detect(
                            target_frame
                        )
                        or []
                    )

                    active_plate_tracks = (
                        target_cam[
                            "plate_tracker"
                        ].update(
                            plate_detections,
                            target_frame,
                            plate_ocr,
                        )
                    )

                    target_cam[
                        "plate_active_tracks"
                    ] = list(
                        active_plate_tracks
                    )

                    finished_capture = (
                        target_cam[
                            "plate_tracker"
                        ].consume_latest_finished_capture()
                    )

                    if finished_capture is not None:

                        target_cam[
                            "plate_latest_capture"
                        ] = finished_capture

                        # Counter kumulatif hasil plat yang lolos
                        # finalisasi tracker/OCR.
                        target_cam[
                            "plate_total_detected"
                        ] += 1

                        target_cam[
                            "plate_history"
                        ] = list(
                            target_cam[
                                "plate_tracker"
                            ].history
                        )

                        plate_text = (
                            finished_capture.get(
                                "formatted",
                                finished_capture.get(
                                    "text",
                                    "",
                                ),
                            )
                        )

                        plate_track_id = _get_capture_value(
                            finished_capture,
                            ["track_id", "tracker_id", "id"],
                            "unknown",
                        )

                        plate_bbox = _get_finished_plate_bbox(
                            finished_capture,
                            active_plate_tracks,
                        )

                        saved_plate_path = save_plate_crop(
                            target_frame,
                            plate_bbox,
                            plate_track_id,
                            now,
                            target_cam["info"]["name"],
                            plate_text,
                        )

                        if saved_plate_path:
                            finished_capture[
                                "plate_image_path"
                            ] = saved_plate_path

                            print(
                                f"[PLATE CAPTURE] "
                                f"[{target_cam['info']['name']}] "
                                f"Track={plate_track_id} -> "
                                f"{os.path.basename(saved_plate_path)}"
                            )
                        else:
                            print(
                                f"[PLATE CAPTURE] "
                                f"[{target_cam['info']['name']}] "
                                f"Track={plate_track_id} "
                                f"gagal menyimpan crop "
                                f"(bbox tidak tersedia)."
                            )

                        print(
                            f"[PLATE RESULT] "
                            f"[{target_cam['info']['name']}] "
                            f"{plate_text}"
                        )

                except Exception as exc:

                    print(
                        f"[PLATE ERROR] "
                        f"CAM {target_cam['info']['id']}: "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # Kamera berikutnya
            # ------------------------------------------------

            active_cam_idx = (
                active_cam_idx + 1
            ) % len(cameras)

            # ------------------------------------------------
            # Render 1 CCTV + panel informasi
            # ------------------------------------------------

            cam = cameras[0]
            frame_i = current_frames[0]

            if frame_i is None:
                live = create_placeholder_tile(
                    cam["info"],
                    "Menunggu frame CCTV..."
                )
                panel = draw_side_panel(cam)
                grid = np.hstack([live, panel])
            else:
                grid = draw_combined_display(
                    frame_i,
                    cam,
                )
            # ------------------------------------------------
            # Statistik
            # ------------------------------------------------

            elapsed = (
                time.time()
                - start_time
            )

            fps = (
                total_loop_frames
                / elapsed
                if elapsed > 0
                else 0
            )

            total_people = sum(
                len(
                    cam["person_tracked"]
                )
                for cam in cameras
            )

            total_plates = sum(
                len(
                    cam["plate_active_tracks"]
                )
                for cam in cameras
            )

            total_captures = sum(
                cam["person_total_captured"]
                for cam in cameras
            )

            # Total plat yang sudah selesai diproses oleh tracker/OCR.
            # Ini kumulatif sejak program dijalankan, bukan jumlah plat
            # yang sedang terlihat pada frame saat ini.
            total_plate_captures = sum(
                cam["plate_total_detected"]
                for cam in cameras
            )

            active_camera_name = (
                cameras[
                    active_cam_idx
                ]["info"]["name"]
            )

            header = draw_master_header(
                total_people,
                total_plates,
                total_captures,
                total_plate_captures,
                fps,
                active_camera_name,
            )

            final_display = np.vstack([
                header,
                grid,
            ])

            cv2.imshow(
                window_name,
                final_display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                print(
                    "\n[INFO] "
                    "Dihentikan pengguna."
                )

                running = False
                break

            try:

                if (
                    cv2.getWindowProperty(
                        window_name,
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
                ):
                    running = False
                    break

            except cv2.error:

                running = False
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Ctrl+C."
        )

    finally:

        running = False

        print(
            "\n[INFO] Menghentikan stream..."
        )

        for cam in cameras:
            cam["grabber"].stop()

        cv2.destroyAllWindows()

        print("=" * 75)
        print("RINGKASAN PERSON CAPTURE")
        print("=" * 75)

        total = 0

        for cam in cameras:

            count = cam[
                "person_total_captured"
            ]

            total += count

            print(
                f"CAM {cam['info']['id']} "
                f"{cam['info']['name']}: "
                f"{count}"
            )

        total_plate = 0

        for cam in cameras:
            total_plate += cam["plate_total_detected"]

        print(
            f"TOTAL PERSON CAPTURE: {total}"
        )
        print(
            f"TOTAL PLATE TERDETEKSI: {total_plate}"
        )
        print("=" * 75)


if __name__ == "__main__":
    main()