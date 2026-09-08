import cv2
import sys
import time
import threading
import queue
import re
from difflib import SequenceMatcher

from ffmpeg_stream_reader import FFmpegStreamReader
from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker


# ============================================================
# KONFIGURASI CCTV
# ============================================================

RTMP_URL = "rtmp://103.255.15.138:1935/live/GSKeluarViewLuar.stream"

STREAM_WIDTH = 2688
STREAM_HEIGHT = 1520


# ============================================================
# KONFIGURASI YOLO
# ============================================================

YOLO_IMGSZ = 960
MODEL_PATH = "models/plate/license-plate-finetune-v2n.pt"

# Confidence minimal deteksi
YOLO_CONFIDENCE = 0.50

# Jalankan YOLO setiap 0.20 detik
AI_INTERVAL = 0.20

# ============================================================
# TRACKING + OCR
# ============================================================

TRACK_MATCH_TIMEOUT = 2.0
TRACK_TIMEOUT = 20.0
IOU_THRESHOLD = 0.50
CENTER_DISTANCE_THRESHOLD = 120
MIN_DETECTION_FRAMES = 2

OCR_FIRST_DELAY = 0.05
OCR_INTERVAL = 0.50
OCR_SCALE = 2.0
OCR_MIN_CONFIDENCE = 0.20

MIN_CROP_WIDTH = 60
MIN_CROP_HEIGHT = 20
MIN_PLATE_ASPECT_RATIO = 2.0
MAX_PLATE_ASPECT_RATIO = 6.0
MIN_CROP_SCORE_FOR_OCR = 0.55
MIN_SHARPNESS = 12.0

MIN_VOTES = 3
MAX_OCR_HISTORY = 8
OCR_SIMILARITY_THRESHOLD = 0.75


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
PANEL_WIDTH = 390
HISTORY_SIZE = 5


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

latest_detections = []
detections_lock = threading.Lock()

latest_active_tracks = []
latest_capture = None
latest_history = []

running = True

tracks = {}
tracks_lock = threading.Lock()
track_counter = 0
track_counter_lock = threading.Lock()

ocr_queue = queue.Queue(maxsize=1)
ocr_busy = False
ocr_busy_lock = threading.Lock()


# ============================================================
# MEMBUKA STREAM CCTV
# ============================================================

print("=" * 70)
print("PLATE AI - TAHAP 3")
print("CCTV RTMP + YOLO + TRACKING + OCR")
print("=" * 70)

print("\nMembuka stream CCTV...")
print(f"URL        : {RTMP_URL}")
print(f"Resolution : {STREAM_WIDTH}x{STREAM_HEIGHT}")


cap = FFmpegStreamReader(
    RTMP_URL,
    width=STREAM_WIDTH,
    height=STREAM_HEIGHT,
)


# ============================================================
# CEK KONEKSI
# ============================================================

if not cap.isOpened():

    print("\n[ERROR] CCTV tidak dapat dibuka.")

    print("Periksa kembali:")
    print("1. URL RTMP benar")
    print("2. CCTV sedang aktif")
    print("3. Jaringan dapat mengakses CCTV")
    print("4. FFmpeg dapat membaca stream RTMP")

    cap.release()
    sys.exit(1)


print("\n[OK] CCTV berhasil dibuka.")


# ============================================================
# INFORMASI RESOLUSI
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# Kalau reader tidak mengembalikan resolusi,
# gunakan konfigurasi.
if width <= 0:
    width = STREAM_WIDTH

if height <= 0:
    height = STREAM_HEIGHT


print(
    f"[CAMERA] Resolution: "
    f"{width}x{height}"
)


# ============================================================
# THREAD PEMBACA FRAME
# ============================================================

def baca_frame_terus():

    global latest_frame
    global running

    print("[CAMERA] Reader thread started")

    reconnect_attempt = 0

    while running:

        ret, frame = cap.read()

        if not ret:
            reconnect_attempt += 1
            print(
                f"\n[WARNING] Frame gagal dibaca; "
                f"reconnect {reconnect_attempt}/5..."
            )

            if reconnect_attempt >= 5:
                print("[CAMERA] Reconnect gagal 5 kali, menghentikan stream.")
                running = False
                break

            time.sleep(min(2.0 * reconnect_attempt, 8.0))
            cap.reconnect()
            continue

        reconnect_attempt = 0

        with frame_lock:

            latest_frame = frame

    print("[CAMERA] Reader thread stopped")


# ============================================================
# TRACKING
# ============================================================

def next_track_id():
    global track_counter
    with track_counter_lock:
        track_counter += 1
        return track_counter


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(box1, box2):
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def create_track(det):
    now = time.time()
    tid = next_track_id()

    track = {
        "id": tid,
        "bbox": det["bbox"],
        "confidence": float(det.get("confidence", 0.0)),
        "first_seen": now,
        "last_seen": now,
        "detection_frames": 1,
        "last_ocr": 0.0,
        "ocr_running": False,
        "ocr_history": [],
        "stable_text": "",
        "stable_votes": 0,
        "best_confidence": 0.0,
        "best_crop": None,
        "best_crop_score": 0.0,
        "ocr_attempts": 0,
    }

    tracks[tid] = track
    print(f"[TRACK] New plate ID={tid}")
    return tid


def match_track(det, used):
    now = time.time()
    best_id = None
    best_score = -999.0

    for tid, track in tracks.items():
        if tid in used:
            continue

        if now - track["last_seen"] > TRACK_MATCH_TIMEOUT:
            continue

        overlap = iou(det["bbox"], track["bbox"])
        distance = center_distance(det["bbox"], track["bbox"])

        if overlap >= IOU_THRESHOLD:
            score = overlap * 2.0 - distance * 0.001
        elif distance <= CENTER_DISTANCE_THRESHOLD:
            score = 0.5 - distance * 0.001
        else:
            continue

        if score > best_score:
            best_score = score
            best_id = tid

    return best_id


def update_tracking(detections):
    now = time.time()
    used = set()
    ids = []

    with tracks_lock:
        for det in detections:
            tid = match_track(det, used)

            if tid is None:
                tid = create_track(det)
            else:
                track = tracks[tid]

                if now - track["last_seen"] <= TRACK_MATCH_TIMEOUT:
                    track["detection_frames"] += 1
                else:
                    track["detection_frames"] = 1

                track["bbox"] = det["bbox"]
                track["confidence"] = float(det.get("confidence", 0.0))
                track["last_seen"] = now

            used.add(tid)
            ids.append(tid)

        expired = [
            tid for tid, track in tracks.items()
            if now - track["last_seen"] > TRACK_TIMEOUT
        ]

        for tid in expired:
            print(f"[TRACK] Remove ID={tid}")
            del tracks[tid]

    return ids


# ============================================================
# CROP + QUALITY
# ============================================================

def crop_plate(frame, bbox):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    aspect = bw / bh if bh else 0

    if not MIN_PLATE_ASPECT_RATIO <= aspect <= MAX_PLATE_ASPECT_RATIO:
        return None

    px = max(2, int(bw * 0.12))
    py = max(2, int(bh * 0.12))

    crop = frame[
        max(0, y1 - py):min(h, y2 + py),
        max(0, x1 - px):min(w, x2 + px)
    ]

    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]

    if cw < MIN_CROP_WIDTH or ch < MIN_CROP_HEIGHT:
        return None

    return crop.copy()


def crop_score(crop):
    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]
    aspect = w / h if h else 0

    aspect_score = max(0.0, 1.0 - abs(aspect - 2.5) / 3.5)
    size_score = min(w / 100.0, 1.0) * 0.65 + min(h / 40.0, 1.0) * 0.35

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_score = min(sharpness / 150.0, 1.0)

    brightness = float(gray.mean())
    bright_score = 1.0 if 30 <= brightness <= 230 else 0.4

    return (
        aspect_score * 0.35
        + size_score * 0.25
        + sharp_score * 0.30
        + bright_score * 0.10
    )


# ============================================================
# OCR VOTING
# ============================================================

def normalize_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def add_vote(track, text, confidence):
    text = normalize_text(text)

    if len(text) < 3 or len(text) > 12:
        return

    track["ocr_history"].append({
        "text": text,
        "confidence": float(confidence),
        "time": time.time(),
    })

    track["ocr_history"] = track["ocr_history"][-MAX_OCR_HISTORY:]

    groups = []

    for item in track["ocr_history"]:
        placed = False

        for group in groups:
            if similarity(item["text"], group[0]["text"]) >= OCR_SIMILARITY_THRESHOLD:
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])

    if not groups:
        return

    best = max(
        groups,
        key=lambda g: (
            len(g),
            sum(x["confidence"] for x in g) / len(g)
        )
    )

    votes = len(best)

    if votes >= MIN_VOTES:
        selected = max(best, key=lambda x: x["confidence"])

        track["stable_text"] = selected["text"]
        track["stable_votes"] = votes
        track["best_confidence"] = selected["confidence"]

        print(
            f"[OCR STABLE] ID={track['id']} "
            f"plate={selected['text']} "
            f"votes={votes} "
            f"conf={selected['confidence']:.3f}"
        )


def submit_ocr(tid, crop):

    now = time.time()

    with tracks_lock:

        track = tracks.get(tid)

        if track is None:
            return

        # Sudah berhasil
        if track["stable_text"]:
            return

        # Validasi minimal
        if (
            track["detection_frames"]
            < MIN_DETECTION_FRAMES
        ):
            return

        # Tunggu sebentar setelah track dibuat
        if (
            now - track["first_seen"]
            < OCR_FIRST_DELAY
        ):
            return

        # Jangan OCR terlalu sering
        if (
            now - track["last_ocr"]
            < OCR_INTERVAL
        ):
            return

        # OCR sedang berjalan
        if track["ocr_running"]:
            return

        # Maksimal 5 percobaan
        if (
            track.get(
                "ocr_attempts",
                0
            ) >= 5
        ):
            return

        # --------------------------------------------------------
        # PILIH CROP TERBAIK
        # --------------------------------------------------------

        best_crop = track.get(
            "best_crop"
        )

        best_score = track.get(
            "best_crop_score",
            0.0
        )

        if (
            best_crop is not None
            and best_score
            >= MIN_CROP_SCORE_FOR_OCR
        ):

            crop_to_queue = (
                best_crop.copy()
            )

            score = best_score

        else:

            crop_to_queue = (
                crop.copy()
            )

            score = crop_score(
                crop_to_queue
            )

        if (
            crop_to_queue is None
            or crop_to_queue.size == 0
        ):
            return

        if (
            score
            < MIN_CROP_SCORE_FOR_OCR
        ):
            return

        # --------------------------------------------------------
        # SHARPNESS
        # --------------------------------------------------------

        gray = cv2.cvtColor(
            crop_to_queue,
            cv2.COLOR_BGR2GRAY
        )

        sharpness = (
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        if sharpness < MIN_SHARPNESS:
            return

        track["last_ocr"] = now
        track["ocr_running"] = True
        track["ocr_attempts"] = (
            track.get(
                "ocr_attempts",
                0
            ) + 1
        )

    # ========================================================
    # QUEUE
    # ========================================================

    try:

        ocr_queue.put_nowait(
            (
                tid,
                crop_to_queue
            )
        )

        print(
            f"[OCR QUEUE] "
            f"ID={tid} "
            f"crop="
            f"{crop_to_queue.shape[1]}x"
            f"{crop_to_queue.shape[0]} "
            f"score={score:.3f}"
        )

    except queue.Full:

        with tracks_lock:

            if tid in tracks:

                tracks[tid][
                    "ocr_running"
                ] = False


def ocr_worker():
    global running, ocr_busy

    print("\n" + "=" * 60)
    print("[OCR] Loading NEW PlateOCR...")
    print(f"[OCR] Scale       : {OCR_SCALE}")
    print(f"[OCR] Confidence  : {OCR_MIN_CONFIDENCE}")
    print("=" * 60)

    try:
        ocr = PlateOCR(
            scale=OCR_SCALE,
            min_confidence=OCR_MIN_CONFIDENCE,
            verbose=False,
            fast_mode=True,
        )

        # Warm-up di awal agar latency inference pertama tidak terjadi
        # saat kendaraan/plat pertama muncul.
        print("[OCR] Warm-up model...")
        ocr.warmup()

    except Exception as exc:
        print(f"[OCR ERROR] {exc}")
        running = False
        return

    print("[OCR] Worker ready")

    while running:
        try:
            tid, crop = ocr_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        with ocr_busy_lock:
            ocr_busy = True

        try:
            started = time.perf_counter()
            result = ocr.read(crop)
            elapsed = time.perf_counter() - started

            if result:
                text = normalize_text(result.get("text", ""))
                confidence = float(result.get("confidence", 0.0))

                if text:
                    with tracks_lock:
                        if tid in tracks:
                            add_vote(
                                tracks[tid],
                                text,
                                confidence
                            )

                    print(
                        f"[OCR RESULT] ID={tid} "
                        f"{text} "
                        f"conf={confidence:.3f} "
                        f"time={elapsed:.2f}s"
                    )
                else:
                    print(f"[OCR RESULT] ID={tid} kosong | {elapsed:.2f}s")
            else:
                print(f"[OCR RESULT] ID={tid} kosong | {elapsed:.2f}s")

        except Exception as exc:
            print(f"[OCR ERROR] ID={tid}: {exc}")

        finally:
            with tracks_lock:
                if tid in tracks:
                    tracks[tid]["ocr_running"] = False

            with ocr_busy_lock:
                ocr_busy = False

            ocr_queue.task_done()


# ============================================================
# THREAD YOLO + TRACKING + OCR
# ============================================================

def yolo_worker():
    global latest_detections
    global latest_active_tracks
    global latest_capture
    global latest_history
    global running

    print("\n" + "=" * 60)
    print("[AI] Loading PlateDetector + PlateOCR...")
    print(f"[AI] Model           : {MODEL_PATH}")
    print(f"[AI] YOLO confidence : {YOLO_CONFIDENCE}")
    print(f"[AI] YOLO imgsz      : {YOLO_IMGSZ}")
    print("=" * 60)

    try:
        detector = PlateDetector(
            model_path=MODEL_PATH,
            confidence=YOLO_CONFIDENCE,
            imgsz=YOLO_IMGSZ,
        )
        ocr_reader = PlateOCR(min_confidence=OCR_MIN_CONFIDENCE)
        tracker = PlateTracker(
            iou_threshold=0.30,
            max_frame_gap=30,
            ocr_every_n_matches=3,
            min_final_confidence=OCR_MIN_CONFIDENCE,
            max_history=5,
        )
    except Exception as exc:
        print(f"[AI ERROR] Gagal menyiapkan detector/OCR: {exc}")
        running = False
        return

    print("[AI] Detector, OCR, dan tracker siap")

    while running:
        cycle_start = time.perf_counter()

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            detections = detector.detect(frame) or []
            active_tracks = tracker.update(
                detections,
                frame,
                ocr_reader,
            )
            finished_capture = (
                tracker.consume_latest_finished_capture()
            )
        except Exception as exc:
            print(f"[AI ERROR] Deteksi/OCR gagal: {exc}")
            detections = []
            active_tracks = []
            finished_capture = None

        with detections_lock:
            latest_detections = detections.copy()

        with frame_lock:
            latest_active_tracks = list(active_tracks)
            if finished_capture is not None:
                latest_capture = finished_capture
            latest_history = list(tracker.history)

        elapsed = time.perf_counter() - cycle_start
        delay = AI_INTERVAL - elapsed

        if delay > 0:
            time.sleep(delay)

    print("[AI] Worker stopped")


# ============================================================
# FUNGSI SCALE BBOX
# ============================================================

def scale_bbox_to_display(
    bbox,
    original_width,
    original_height,
    display_width,
    display_height,
):

    x1, y1, x2, y2 = bbox

    sx = (
        display_width
        /
        original_width
    )

    sy = (
        display_height
        /
        original_height
    )

    return [
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    ]


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    frame,
    detection,
    original_width,
    original_height,
):

    bbox = detection.get(
        "bbox",
        None,
    )

    confidence = float(
        detection.get(
            "confidence",
            0.0,
        )
    )

    if bbox is None:
        return

    # --------------------------------------------------------
    # Scale bbox dari resolusi CCTV
    # ke resolusi display
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        scale_bbox_to_display(
            bbox,
            original_width,
            original_height,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )
    )

    # --------------------------------------------------------
    # Rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"PLATE "
        f"{confidence:.0%}"
    )

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )

    label_y = max(
        y1,
        text_height + baseline + 5,
    )

    # Background label
    cv2.rectangle(
        frame,
        (
            x1,
            label_y
            -
            text_height
            -
            baseline
            -
            5,
        ),
        (
            x1
            +
            text_width
            +
            10,
            label_y
            +
            5,
        ),
        (0, 0, 0),
        -1,
    )

    # Text
    cv2.putText(
        frame,
        label,
        (
            x1 + 5,
            label_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# PANEL HASIL PLAT
# ============================================================

def safe_text(text):
    return str(text or "").encode("ascii", "replace").decode("ascii")


def draw_text(frame, text, position, scale=0.65, thickness=2):
    cv2.putText(
        frame,
        safe_text(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel_box(frame, x, y, w, h, alpha=0.88):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def put_crop_on_panel(frame, crop, x, y, w, h):
    if crop is None:
        draw_text(frame, "Capture tidak tersedia", (x + 15, y + h // 2), 0.55, 1)
        return

    try:
        crop = crop.copy()
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return

        scale = min(w / cw, h / ch)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 10, 10), -1)
        offset_x = x + (w - nw) // 2
        offset_y = y + (h - nh) // 2
        frame[offset_y:offset_y + nh, offset_x:offset_x + nw] = resized
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    except Exception:
        pass


def draw_latest_capture(frame, capture):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_WIDTH, max(280, w - 20))
    panel_x = w - panel_w - 15
    panel_y = 15
    panel_h = 285

    draw_panel_box(frame, panel_x, panel_y, panel_w, panel_h)
    draw_text(frame, "CAPTURE PLAT TERBARU", (panel_x + 15, panel_y + 30), 0.65, 2)

    if capture is None:
        draw_text(frame, "Belum ada plat selesai.", (panel_x + 15, panel_y + 75), 0.55, 1)
        draw_text(frame, "Menunggu kendaraan...", (panel_x + 15, panel_y + 105), 0.55, 1)
        return

    put_crop_on_panel(frame, capture.get("crop"), panel_x + 30, panel_y + 45, panel_w - 60, 95)
    plate_text = capture.get("formatted", capture.get("text", ""))
    confidence = float(capture.get("confidence", 0.0) or 0.0)
    det_conf = float(capture.get("detection_confidence", 0.0) or 0.0)
    track_id = capture.get("track_id", "-")
    timestamp = capture.get("timestamp", "-")

    draw_text(frame, f"Plat       : {plate_text}", (panel_x + 15, panel_y + 165), 0.60, 2)
    draw_text(frame, f"OCR Conf   : {confidence:.1%}", (panel_x + 15, panel_y + 193), 0.52, 1)
    draw_text(frame, f"YOLO Conf  : {det_conf:.1%}", (panel_x + 15, panel_y + 218), 0.52, 1)
    draw_text(frame, f"Track      : #{track_id}", (panel_x + 15, panel_y + 243), 0.52, 1)
    draw_text(frame, timestamp, (panel_x + 15, panel_y + 268), 0.45, 1)


def draw_history(frame, history):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_WIDTH, max(280, w - 20))
    panel_x = w - panel_w - 15
    panel_y = 315
    row_h = 54
    panel_h = 45 + max(1, len(history)) * row_h + 15

    draw_panel_box(frame, panel_x, panel_y, panel_w, panel_h)
    draw_text(frame, "HISTORY 5 PLAT TERAKHIR", (panel_x + 15, panel_y + 30), 0.58, 2)

    if not history:
        draw_text(frame, "Belum ada history.", (panel_x + 15, panel_y + 70), 0.55, 1)
        return

    for index, item in enumerate(history[:HISTORY_SIZE]):
        y = panel_y + 55 + index * row_h
        plate_text = item.get("formatted", item.get("text", ""))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        track_id = item.get("track_id", "-")
        draw_text(frame, f"{index + 1}. #{track_id} {plate_text}", (panel_x + 15, y), 0.52, 1)
        draw_text(frame, f"   OCR {confidence:.1%}", (panel_x + 15, y + 22), 0.45, 1)


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_info(
    frame,
    detection_count,
    fps,
):

    # --------------------------------------------------------
    # Background panel
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 120),
        (0, 0, 0),
        -1,
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Display FPS : {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Plate       : {detection_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Resolution  : {width}x{height}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )



# ============================================================
# DRAW TRACKING / OCR
# ============================================================

def draw_track_overlay(frame):
    with frame_lock:
        items = list(latest_active_tracks)

    for track in items:
        bbox = scale_bbox_to_display(
            track.bbox,
            width,
            height,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 3
        )

        hasil = track.hasil_voting()
        if hasil is not None:
            label = (
                f"ID {track.id} | "
                f"{hasil.get('formatted', hasil['text'])} "
                f"{hasil.get('confidence_rata2', 0.0):.0%}"
            )
        else:
            label = f"ID {track.id} | Membaca plat..."

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

# ============================================================
# START THREAD
# ============================================================

reader_thread = threading.Thread(
    target=baca_frame_terus,
    daemon=True,
)

yolo_thread = threading.Thread(
    target=yolo_worker,
    daemon=True,
)

print("\n[ SYSTEM ] Starting threads...")

reader_thread.start()
time.sleep(0.5)
yolo_thread.start()

# ============================================================
# WINDOW
# ============================================================

window_name = "CCTV - Plate AI"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    window_name,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)


# ============================================================
# DISPLAY LOOP
# ============================================================

frame_count = 0
start_time = time.time()


print("\n" + "=" * 60)
print("[SYSTEM] CCTV + YOLO + TRACKING + OCR aktif")
print("[SYSTEM] Tekan Q untuk keluar")
print("=" * 60)


while running:

    # --------------------------------------------------------
    # Ambil frame terbaru
    # --------------------------------------------------------

    with frame_lock:

        if latest_frame is None:

            frame = None
            capture = None
            history = []

        else:

            frame = latest_frame.copy()
            capture = latest_capture.copy() if latest_capture is not None else None
            history = list(latest_history)

    if frame is None:

        time.sleep(0.01)

        continue

    frame_count += 1

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    elapsed_time = (
        time.time()
        -
        start_time
    )

    if elapsed_time > 0:

        fps = (
            frame_count
            /
            elapsed_time
        )

    else:

        fps = 0


    # --------------------------------------------------------
    # Resize hanya untuk display
    #
    # YOLO tetap bekerja pada frame asli.
    # --------------------------------------------------------

    display = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )


    # --------------------------------------------------------
    # Ambil hasil YOLO terbaru
    # --------------------------------------------------------

    with detections_lock:

        detections = (
            latest_detections.copy()
        )


    # --------------------------------------------------------
    # Gambar tracking + OCR
    # --------------------------------------------------------

    draw_track_overlay(display)

    draw_latest_capture(display, capture)
    draw_history(display, history)

    # --------------------------------------------------------
    # Info
    # --------------------------------------------------------

    draw_info(
        display,
        len(detections),
        fps,
    )


    # --------------------------------------------------------
    # Tampilkan
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        display,
    )


    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )

    if key == ord("q"):

        print(
            "\n[SYSTEM] "
            "Program dihentikan user."
        )

        running = False

        break


# ============================================================
# CLEANUP
# ============================================================

print("\n[SYSTEM] Stopping...")

running = False

# Beri waktu thread berhenti
time.sleep(0.2)

cap.release()

cv2.destroyAllWindows()


print("\n" + "=" * 70)
print("TEST SELESAI")
print(f"Total frame display : {frame_count}")
print("=" * 70)