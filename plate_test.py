import cv2
import sys
import time
import os
import threading

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker


# ============================================================
# KONFIGURASI
# ============================================================

RTMP_URL = (
    "rtmp://103.255.15.222:1935/"
    "atcs-kota/"
    "FMNotoMcDonalds.stream"
)

MODEL_PATH = (
    "models/plate/"
    "license-plate-finetune-v1n.pt"
)

CONFIDENCE = 0.40

# AI memproses kira-kira setiap 3 frame.
AI_FRAME_INTERVAL = 3

# Tracker.
IOU_THRESHOLD = 0.3
MAX_FRAME_GAP = 30
OCR_EVERY_N_MATCHES = 3

# Minimum confidence OCR agar hasil final masuk panel.
MIN_FINAL_OCR_CONFIDENCE = 0.20

# History panel.
HISTORY_SIZE = 5

# Ukuran panel kanan.
PANEL_WIDTH = 390

# Ukuran area crop pada panel.
CROP_WIDTH = 330
CROP_HEIGHT = 110


# ============================================================
# CEK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model tidak ditemukan:")
    print(MODEL_PATH)
    sys.exit(1)


# ============================================================
# LOAD MODEL DETEKSI
# ============================================================

print("=" * 60)
print("PLATE AI - DETECTION + TRACKING + OCR")
print("=" * 60)

print("\nMemuat model deteksi...")

detector = PlateDetector(
    model_path=MODEL_PATH,
    confidence=CONFIDENCE,
)

print("[OK] Model deteksi siap.")


# ============================================================
# LOAD OCR
# ============================================================

print("\nMemuat model OCR...")

ocr_reader = PlateOCR(
    min_confidence=0.20,
)

print("[OK] Model OCR siap.")


# ============================================================
# TRACKER
# ============================================================

tracker = PlateTracker(
    iou_threshold=IOU_THRESHOLD,
    max_frame_gap=MAX_FRAME_GAP,
    ocr_every_n_matches=OCR_EVERY_N_MATCHES,
    min_final_confidence=MIN_FINAL_OCR_CONFIDENCE,
    max_history=HISTORY_SIZE,
)

print("[OK] Tracker siap.")


# ============================================================
# BUKA CCTV
# ============================================================

print("\nMembuka CCTV...")

cap = cv2.VideoCapture(RTMP_URL)

if not cap.isOpened():
    print("[ERROR] CCTV tidak dapat dibuka.")
    cap.release()
    sys.exit(1)

print("[OK] CCTV berhasil dibuka.")


# ============================================================
# INFORMASI STREAM
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print(
    f"Resolusi CCTV : "
    f"{width} x {height}"
)


# ============================================================
# SHARED DATA
# ============================================================

latest_frame = None
latest_active_tracks = []

# Event capture final terbaru.
latest_capture = None

frame_lock = threading.Lock()

running = True


# ============================================================
# HELPER DISPLAY
# ============================================================

def safe_text(text):
    """
    OpenCV default font tidak selalu mampu menampilkan
    karakter non-ASCII. Untuk UI CCTV, ganti karakter
    yang tidak aman dengan '?'.
    """
    if text is None:
        return ""

    try:
        return str(text).encode(
            "ascii",
            "replace",
        ).decode("ascii")
    except Exception:
        return str(text)


def draw_text(
    frame,
    text,
    position,
    scale=0.65,
    thickness=2,
):
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


def draw_panel_box(
    frame,
    x,
    y,
    w,
    h,
    alpha=0.88,
):
    """
    Membuat panel hitam semi-transparan.
    """
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )


def put_crop_on_panel(
    frame,
    crop,
    x,
    y,
    w,
    h,
):
    """
    Menampilkan crop plat di panel.
    """
    if crop is None:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (80, 80, 80),
            1,
        )

        draw_text(
            frame,
            "Capture tidak tersedia",
            (x + 15, y + h // 2),
            0.55,
            1,
        )

        return

    try:
        crop = crop.copy()

        ch, cw = crop.shape[:2]

        if ch <= 0 or cw <= 0:
            return

        # Pertahankan aspect ratio.
        scale = min(
            w / cw,
            h / ch,
        )

        nw = max(
            1,
            int(cw * scale),
        )

        nh = max(
            1,
            int(ch * scale),
        )

        resized = cv2.resize(
            crop,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

        # Background area.
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (10, 10, 10),
            -1,
        )

        offset_x = (
            x + (w - nw) // 2
        )

        offset_y = (
            y + (h - nh) // 2
        )

        frame[
            offset_y:offset_y + nh,
            offset_x:offset_x + nw
        ] = resized

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1,
        )

    except Exception:
        pass


def draw_latest_capture(
    frame,
    capture,
):
    """
    Panel capture terbaru.
    """
    h, w = frame.shape[:2]

    panel_w = min(
        PANEL_WIDTH,
        max(280, w - 20),
    )

    panel_x = (
        w - panel_w - 15
    )

    panel_y = 15

    panel_h = 285

    draw_panel_box(
        frame,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
    )

    draw_text(
        frame,
        "CAPTURE PLAT TERBARU",
        (
            panel_x + 15,
            panel_y + 30,
        ),
        0.65,
        2,
    )

    if capture is None:
        draw_text(
            frame,
            "Belum ada plat selesai.",
            (
                panel_x + 15,
                panel_y + 75,
            ),
            0.55,
            1,
        )

        draw_text(
            frame,
            "Menunggu kendaraan...",
            (
                panel_x + 15,
                panel_y + 105,
            ),
            0.55,
            1,
        )

        return

    crop_x = (
        panel_x + 30
    )

    crop_y = (
        panel_y + 45
    )

    crop_w = panel_w - 60
    crop_h = 95

    put_crop_on_panel(
        frame,
        capture.get("crop"),
        crop_x,
        crop_y,
        crop_w,
        crop_h,
    )

    plate_text = capture.get(
        "formatted",
        capture.get("text", ""),
    )

    confidence = float(
        capture.get(
            "confidence",
            0.0,
        )
        or 0.0
    )

    det_conf = float(
        capture.get(
            "detection_confidence",
            0.0,
        )
        or 0.0
    )

    track_id = capture.get(
        "track_id",
        "-",
    )

    timestamp = capture.get(
        "timestamp",
        "-",
    )

    draw_text(
        frame,
        f"Plat       : {plate_text}",
        (
            panel_x + 15,
            panel_y + 165,
        ),
        0.60,
        2,
    )

    draw_text(
        frame,
        f"OCR Conf   : {confidence:.1%}",
        (
            panel_x + 15,
            panel_y + 193,
        ),
        0.52,
        1,
    )

    draw_text(
        frame,
        f"YOLO Conf  : {det_conf:.1%}",
        (
            panel_x + 15,
            panel_y + 218,
        ),
        0.52,
        1,
    )

    draw_text(
        frame,
        f"Track      : #{track_id}",
        (
            panel_x + 15,
            panel_y + 243,
        ),
        0.52,
        1,
    )

    draw_text(
        frame,
        f"{timestamp}",
        (
            panel_x + 15,
            panel_y + 268,
        ),
        0.45,
        1,
    )


def draw_history(
    frame,
    history,
):
    """
    Panel history 5 hasil terakhir.
    """
    h, w = frame.shape[:2]

    panel_w = min(
        PANEL_WIDTH,
        max(280, w - 20),
    )

    panel_x = (
        w - panel_w - 15
    )

    panel_y = 315

    row_h = 54

    panel_h = (
        45
        + max(1, len(history))
        * row_h
        + 15
    )

    draw_panel_box(
        frame,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
    )

    draw_text(
        frame,
        "HISTORY 5 PLAT TERAKHIR",
        (
            panel_x + 15,
            panel_y + 30,
        ),
        0.58,
        2,
    )

    if not history:
        draw_text(
            frame,
            "Belum ada history.",
            (
                panel_x + 15,
                panel_y + 70,
            ),
            0.55,
            1,
        )

        return

    for index, item in enumerate(
        history[:HISTORY_SIZE]
    ):
        y = (
            panel_y
            + 55
            + index * row_h
        )

        plate_text = item.get(
            "formatted",
            item.get("text", ""),
        )

        confidence = float(
            item.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        track_id = item.get(
            "track_id",
            "-",
        )

        draw_text(
            frame,
            f"{index + 1}. "
            f"#{track_id} "
            f"{plate_text}",
            (
                panel_x + 15,
                y,
            ),
            0.52,
            1,
        )

        draw_text(
            frame,
            f"   OCR {confidence:.1%}",
            (
                panel_x + 15,
                y + 22,
            ),
            0.45,
            1,
        )


def draw_active_tracks(
    frame,
    active_tracks,
):
    """
    Bounding box + OCR sementara untuk track aktif.
    """
    for track in active_tracks:

        x1, y1, x2, y2 = (
            track.bbox
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        hasil_sementara = (
            track.hasil_voting()
        )

        if hasil_sementara is not None:
            display_text = (
                hasil_sementara.get(
                    "formatted",
                    hasil_sementara[
                        "text"
                    ],
                )
            )

            confidence = float(
                hasil_sementara.get(
                    "confidence_rata2",
                    0.0,
                )
                or 0.0
            )

            label = (
                f"#{track.id} "
                f"{display_text} "
                f"{confidence:.0%}"
            )

        else:
            label = (
                f"#{track.id} ..."
            )

        # Background kecil untuk label.
        text_y = max(
            y1 - 10,
            25,
        )

        draw_text(
            frame,
            label,
            (
                x1,
                text_y,
            ),
            0.62,
            2,
        )


# ============================================================
# CCTV CAPTURE THREAD
# ============================================================

def capture_camera():
    global latest_frame
    global running

    while running:

        ret, frame = cap.read()

        if not ret:
            print(
                "[WARNING] "
                "Frame gagal dibaca."
            )

            time.sleep(0.1)
            continue

        with frame_lock:
            latest_frame = frame


# ============================================================
# AI THREAD
# ============================================================

def detect_plate():
    global latest_active_tracks
    global latest_capture
    global running

    ai_frame_counter = 0

    while running:

        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        ai_frame_counter += 1

        # Skip frame sesuai interval.
        if (
            ai_frame_counter
            % AI_FRAME_INTERVAL
            != 0
        ):
            time.sleep(0.01)
            continue

        # ====================================================
        # YOLO
        # ====================================================

        try:
            detections = detector.detect(
                frame
            )
        except Exception as exc:
            print(
                f"[DETECTOR ERROR] {exc}"
            )
            time.sleep(0.05)
            continue

        # ====================================================
        # TRACKER + OCR
        # ====================================================

        active_tracks = tracker.update(
            detections,
            frame,
            ocr_reader,
        )

        # Event capture final.
        finished_capture = (
            tracker.consume_latest_finished_capture()
        )

        with frame_lock:
            latest_active_tracks = (
                list(active_tracks)
            )

            if finished_capture is not None:
                latest_capture = (
                    finished_capture
                )

        time.sleep(0.01)


# ============================================================
# START THREAD
# ============================================================

camera_thread = threading.Thread(
    target=capture_camera,
    daemon=True,
)

ai_thread = threading.Thread(
    target=detect_plate,
    daemon=True,
)

camera_thread.start()
ai_thread.start()


# ============================================================
# WINDOW
# ============================================================

window_name = (
    "Plate Detection - CCTV"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    window_name,
    1280,
    720,
)


# ============================================================
# FPS
# ============================================================

display_frame_count = 0
display_start_time = time.time()


# ============================================================
# MAIN DISPLAY LOOP
# ============================================================

try:

    while True:

        with frame_lock:

            if latest_frame is None:
                frame = None
                active_tracks = []
                capture = None

            else:
                frame = (
                    latest_frame.copy()
                )

                active_tracks = list(
                    latest_active_tracks
                )

                capture = latest_capture

        if frame is None:
            time.sleep(0.01)
            continue

        display_frame_count += 1

        # ====================================================
        # TRACK AKTIF
        # ====================================================

        draw_active_tracks(
            frame,
            active_tracks,
        )

        # ====================================================
        # PANEL CAPTURE TERBARU
        # ====================================================

        draw_latest_capture(
            frame,
            capture,
        )

        # ====================================================
        # HISTORY
        # ====================================================

        draw_history(
            frame,
            tracker.history,
        )

        # ====================================================
        # INFO KIRI ATAS
        # ====================================================

        elapsed = (
            time.time()
            - display_start_time
        )

        if elapsed > 0:
            display_fps = (
                display_frame_count
                / elapsed
            )
        else:
            display_fps = 0.0

        draw_text(
            frame,
            f"Display FPS: "
            f"{display_fps:.2f}",
            (20, 30),
            0.65,
            2,
        )

        draw_text(
            frame,
            f"Active tracks: "
            f"{len(active_tracks)}",
            (20, 60),
            0.65,
            2,
        )

        draw_text(
            frame,
            f"Finished plates: "
            f"{len(tracker.hasil_final)}",
            (20, 90),
            0.65,
            2,
        )

        draw_text(
            frame,
            "Press Q to exit",
            (20, 120),
            0.65,
            2,
        )

        # ====================================================
        # TAMPILKAN CCTV
        # ====================================================

        cv2.imshow(
            window_name,
            frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break


finally:

    running = False

    time.sleep(0.2)

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# HASIL AKHIR
# ============================================================

print("\n" + "=" * 60)
print("TEST SELESAI")
print("=" * 60)

print(
    f"Total display frame : "
    f"{display_frame_count}"
)

print(
    f"Total plat terbaca  : "
    f"{len(tracker.hasil_final)}"
)

for hasil in tracker.hasil_final:

    print(
        f"  Track #{hasil['track_id']}: "
        f"'{hasil.get('formatted', hasil['text'])}' "
        f"| OCR conf={hasil.get('confidence', 0.0):.1%} "
        f"| YOLO conf={hasil.get('detection_confidence', 0.0):.1%} "
        f"| "
        f"({hasil['jumlah_muncul']}/"
        f"{hasil['total_bacaan']} bacaan cocok)"
    )