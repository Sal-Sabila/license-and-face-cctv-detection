import cv2
import sys
import time
import threading
import queue
import re
import os
from difflib import SequenceMatcher

from ffmpeg_stream_reader import FFmpegStreamReader
from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR
from tracker import PlateTracker
from ultralytics import YOLO


# ============================================================
# KONFIGURASI CCTV
# ============================================================

RTMP_URL = "rtmp://103.255.15.222:1935/atcs-kota/JogokariyanUtara.stream"

STREAM_WIDTH = 2688
STREAM_HEIGHT = 1520


# ============================================================
# KONFIGURASI AI (PLAT & PERSON)
# ============================================================

# Plate YOLO
YOLO_IMGSZ = 960
MODEL_PATH = "models/plate/license-plate-finetune-v2n.pt"
YOLO_CONFIDENCE = 0.50

# Person YOLO
PERSON_MODEL_PATH = "yolov8n.pt"
PERSON_CONFIDENCE = 0.35
PERSON_IMGSZ = 640

# AI Worker Interval
AI_INTERVAL = 0.20

# OCR & Tracking
OCR_MIN_CONFIDENCE = 0.20
HISTORY_SIZE = 5

# Buat folder captures jika belum ada
os.makedirs("captures", exist_ok=True)


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
PANEL_WIDTH = 410


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

latest_detections = []
detections_lock = threading.Lock()

latest_active_tracks = []
latest_active_persons = []
latest_capture = None
latest_history = []

running = True


# ============================================================
# MEMBUKA STREAM CCTV
# ============================================================

print("=" * 70)
print("PLATE & PERSON/RIDER AI - CCTV SYSTEM")
print("CCTV RTMP/HLS + YOLO (PLAT & ORANG) + SPATIAL ASSOCIATION + OCR")
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
    print("1. URL RTMP/HLS benar")
    print("2. CCTV sedang aktif")
    print("3. Jaringan dapat mengakses CCTV")
    print("4. FFmpeg dapat membaca stream")

    cap.release()
    sys.exit(1)

print("\n[OK] CCTV berhasil dibuka.")


# ============================================================
# INFORMASI RESOLUSI
# ============================================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0:
    width = STREAM_WIDTH
if height <= 0:
    height = STREAM_HEIGHT

print(f"[CAMERA] Resolution: {width}x{height}")


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
            print(f"\n[WARNING] Frame gagal dibaca; reconnect {reconnect_attempt}/5...")

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
# THREAD YOLO (PLAT + ORANG) + TRACKING + SPATIAL ASSOCIATION
# ============================================================

def yolo_worker():
    global latest_detections
    global latest_active_tracks
    global latest_active_persons
    global latest_capture
    global latest_history
    global running

    print("\n" + "=" * 60)
    print("[AI] Loading PlateDetector, PlateOCR & Person Detector...")
    print(f"[AI] Plate Model     : {MODEL_PATH}")
    print(f"[AI] Person Model    : {PERSON_MODEL_PATH}")
    print(f"[AI] Plate Conf      : {YOLO_CONFIDENCE}")
    print(f"[AI] Person Conf     : {PERSON_CONFIDENCE}")
    print("=" * 60)

    try:
        plate_detector = PlateDetector(
            model_path=MODEL_PATH,
            confidence=YOLO_CONFIDENCE,
            imgsz=YOLO_IMGSZ,
        )
        ocr_reader = PlateOCR(min_confidence=OCR_MIN_CONFIDENCE)
        plate_tracker = PlateTracker(
            iou_threshold=0.30,
            max_frame_gap=30,
            ocr_every_n_matches=3,
            min_final_confidence=OCR_MIN_CONFIDENCE,
            max_history=5,
        )
        person_model = YOLO(PERSON_MODEL_PATH)
    except Exception as exc:
        print(f"[AI ERROR] Gagal inisialisasi AI models: {exc}")
        running = False
        return

    print("[AI] Plate & Person Detectors, OCR, dan Trackers siap")

    plate_to_rider = {}
    person_meta = {}

    while running:
        cycle_start = time.perf_counter()

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        plate_detections = []
        active_plate_tracks = []
        finished_plate_capture = None

        # 1. Plate Detection & Tracking
        try:
            plate_detections = plate_detector.detect(frame) or []
            active_plate_tracks = plate_tracker.update(
                plate_detections,
                frame,
                ocr_reader,
            )
            finished_plate_capture = (
                plate_tracker.consume_latest_finished_capture()
            )
        except Exception as exc:
            print(f"[AI ERROR] Deteksi/OCR plat gagal: {exc}")
            plate_detections = []
            active_plate_tracks = []
            finished_plate_capture = None

        # 2. Person Detection & Tracking
        person_tracks = []
        try:
            person_results = person_model.track(
                frame,
                classes=[0],  # 0: person
                conf=PERSON_CONFIDENCE,
                imgsz=PERSON_IMGSZ,
                persist=True,
                verbose=False,
            )[0]

            if person_results.boxes is not None and len(person_results.boxes) > 0:
                boxes = person_results.boxes
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                track_ids = (
                    boxes.id.int().cpu().numpy()
                    if boxes.id is not None
                    else [i + 1 for i in range(len(coords))]
                )

                for p_box, p_conf, p_id in zip(coords, confs, track_ids):
                    person_tracks.append({
                        "id": int(p_id),
                        "bbox": [float(p_box[0]), float(p_box[1]), float(p_box[2]), float(p_box[3])],
                        "conf": float(p_conf),
                        "is_rider": False,
                        "linked_plate_id": None,
                        "linked_plate_bbox": None,
                    })
        except Exception as exc:
            pass

        # 3. Spatial Association (Pengendara Motor vs Plat Nomor)
        now_time = time.time()
        for p in person_tracks:
            px1, py1, px2, py2 = p["bbox"]
            pw = px2 - px1
            ph = py2 - py1
            pid = p["id"]

            if pid not in person_meta:
                person_meta[pid] = {
                    "first_seen": now_time,
                    "seen_count": 0,
                    "is_rider": False,
                    "captured": False,
                    "best_bbox": p["bbox"],
                    "best_conf": p["conf"],
                }

            meta = person_meta[pid]
            meta["seen_count"] += 1
            meta["last_seen"] = now_time
            if p["conf"] >= meta["best_conf"]:
                meta["best_conf"] = p["conf"]
                meta["best_bbox"] = p["bbox"]

            for pl in active_plate_tracks:
                lx1, ly1, lx2, ly2 = pl.bbox
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0

                # Plat berada dalam rentang horizontal orang dan di bagian bawah/sekitar tubuh
                horiz_match = (px1 - pw * 0.40 <= lcx <= px2 + pw * 0.40)
                vert_match = (py1 + ph * 0.20 <= lcy <= py2 + ph * 0.85)

                if horiz_match and vert_match:
                    p["is_rider"] = True
                    p["linked_plate_id"] = pl.id
                    p["linked_plate_bbox"] = [lx1, ly1, lx2, ly2]
                    meta["is_rider"] = True

                    plate_to_rider[pl.id] = {
                        "rider_id": pid,
                        "rider_bbox": p["bbox"],
                        "plate_bbox": [lx1, ly1, lx2, ly2],
                        "last_seen": now_time,
                    }
                    break

        # 4. Handle Capture Events
        new_capture = None

        # Kasus A: Plat selesai dibaca -> Cek apakah terkait pengendara motor
        if finished_plate_capture is not None:
            plate_id = finished_plate_capture["track_id"]
            rider_info = plate_to_rider.get(plate_id)

            plate_crop = finished_plate_capture.get("crop")
            unit_crop = None
            face_crop = None
            capture_type = "VEHICLE"
            rider_id = None

            fh, fw = frame.shape[:2]

            if rider_info is not None:
                capture_type = "RIDER"
                rider_id = rider_info["rider_id"]
                rb = rider_info["rider_bbox"]
                pb = finished_plate_capture.get("bbox", rider_info["plate_bbox"])

                # Union box Motor + Pengendara
                ux1 = max(0, int(min(rb[0], pb[0]) - 30))
                uy1 = max(0, int(min(rb[1], pb[1]) - 30))
                ux2 = min(fw, int(max(rb[2], pb[2]) + 30))
                uy2 = min(fh, int(max(rb[3], pb[3]) + 30))
                if ux2 > ux1 and uy2 > uy1:
                    unit_crop = frame[uy1:uy2, ux1:ux2].copy()

                # Crop wajah/kepala pengendara
                rx1, ry1, rx2, ry2 = map(int, rb)
                fx1 = max(0, rx1 - 10)
                fy1 = max(0, ry1 - 10)
                fx2 = min(fw, rx2 + 10)
                fy2 = min(fh, ry1 + int((ry2 - ry1) * 0.45))
                if fx2 > fx1 and fy2 > fy1:
                    face_crop = frame[fy1:fy2, fx1:fx2].copy()

                if rider_id in person_meta:
                    person_meta[rider_id]["captured"] = True
            else:
                unit_crop = plate_crop

            # Simpan capture ke folder captures/
            try:
                ts_clean = time.strftime("%Y%m%d_%H%M%S")
                plat_text = finished_plate_capture.get("text", "PLAT")
                if capture_type == "RIDER":
                    if unit_crop is not None and unit_crop.size > 0:
                        cv2.imwrite(f"captures/motor_{ts_clean}_id{plate_id}_{plat_text}.jpg", unit_crop)
                    if face_crop is not None and face_crop.size > 0:
                        cv2.imwrite(f"captures/face_{ts_clean}_rider{rider_id}.jpg", face_crop)
                    if plate_crop is not None and plate_crop.size > 0:
                        cv2.imwrite(f"captures/plate_{ts_clean}_id{plate_id}_{plat_text}.jpg", plate_crop)
                else:
                    if plate_crop is not None and plate_crop.size > 0:
                        cv2.imwrite(f"captures/kendaraan_{ts_clean}_id{plate_id}_{plat_text}.jpg", plate_crop)
            except Exception as save_err:
                print(f"[CAPTURE SAVE ERROR] {save_err}")

            new_capture = {
                "type": capture_type,
                "track_id": finished_plate_capture["track_id"],
                "rider_id": rider_id,
                "text": finished_plate_capture["text"],
                "formatted": finished_plate_capture["formatted"],
                "confidence": finished_plate_capture["confidence"],
                "detection_confidence": finished_plate_capture["detection_confidence"],
                "plate_crop": plate_crop,
                "unit_crop": unit_crop if unit_crop is not None else plate_crop,
                "face_crop": face_crop,
                "timestamp": finished_plate_capture["timestamp"],
            }

        # Kasus B: Pejalan Kaki (Orang tanpa kendaraan/plat)
        if new_capture is None:
            for p in person_tracks:
                pid = p["id"]
                meta = person_meta.get(pid)
                if meta and not meta["is_rider"] and not meta["captured"]:
                    pb = p["bbox"]
                    ph = pb[3] - pb[1]
                    # Syarat capture: terdeteksi stabil minimal 8 frame dan tinggi memadai
                    if meta["seen_count"] >= 8 and ph >= 120:
                        fh, fw = frame.shape[:2]
                        px1, py1, px2, py2 = map(int, pb)
                        cx1 = max(0, px1 - 15)
                        cy1 = max(0, py1 - 15)
                        cx2 = min(fw, px2 + 15)
                        cy2 = min(fh, py2 + 15)
                        p_crop = frame[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else None

                        hx1 = max(0, px1 - 10)
                        hy1 = max(0, py1 - 10)
                        hx2 = min(fw, px2 + 10)
                        hy2 = min(fh, py1 + int(ph * 0.40))
                        head_crop = frame[hy1:hy2, hx1:hx2].copy() if hx2 > hx1 and hy2 > hy1 else None

                        ts_clean = time.strftime("%Y%m%d_%H%M%S")
                        try:
                            if p_crop is not None and p_crop.size > 0:
                                cv2.imwrite(f"captures/pedestrian_{ts_clean}_p{pid}.jpg", p_crop)
                            if head_crop is not None and head_crop.size > 0:
                                cv2.imwrite(f"captures/face_{ts_clean}_pedestrian_p{pid}.jpg", head_crop)
                        except Exception as save_err:
                            print(f"[CAPTURE SAVE ERROR] {save_err}")

                        meta["captured"] = True
                        new_capture = {
                            "type": "PEDESTRIAN",
                            "track_id": f"P-{pid}",
                            "rider_id": None,
                            "text": "PEJALAN KAKI",
                            "formatted": f"ORANG #{pid}",
                            "confidence": float(p["conf"]),
                            "detection_confidence": float(p["conf"]),
                            "plate_crop": None,
                            "unit_crop": p_crop,
                            "face_crop": head_crop,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        break

        # Bersihkan metadata lama
        clean_time = time.time()
        for k in [k for k, v in plate_to_rider.items() if clean_time - v["last_seen"] > 30.0]:
            del plate_to_rider[k]
        for k in [k for k, v in person_meta.items() if clean_time - v["last_seen"] > 30.0]:
            del person_meta[k]

        # Update State Global
        with detections_lock:
            latest_detections = plate_detections.copy()

        with frame_lock:
            latest_active_tracks = list(active_plate_tracks)
            latest_active_persons = list(person_tracks)
            if new_capture is not None:
                latest_capture = new_capture
                latest_history.insert(0, new_capture)
                latest_history = latest_history[:HISTORY_SIZE]

        elapsed = time.perf_counter() - cycle_start
        delay = AI_INTERVAL - elapsed
        if delay > 0:
            time.sleep(delay)

    print("[AI] Worker stopped")


# ============================================================
# FUNGSI HELPER DISPLAY & DRAWING
# ============================================================

def scale_bbox_to_display(bbox, orig_w, orig_h, disp_w, disp_h):
    x1, y1, x2, y2 = bbox
    sx = disp_w / orig_w
    sy = disp_h / orig_h
    return [
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    ]


def safe_text(text):
    return str(text or "").encode("ascii", "replace").decode("ascii")


def draw_text(frame, text, position, scale=0.65, thickness=2, color=(0, 255, 0)):
    cv2.putText(
        frame,
        safe_text(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel_box(frame, x, y, w, h, alpha=0.88):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 55), 1)


def put_crop_on_panel(frame, crop, x, y, w, h, placeholder_text="Capture N/A"):
    if crop is None or getattr(crop, "size", 0) == 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (15, 15, 15), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), 1)
        draw_text(frame, placeholder_text, (x + 10, y + h // 2 + 5), 0.42, 1, color=(120, 120, 120))
        return

    try:
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)
    except Exception:
        pass


# ============================================================
# PANEL KANAN: CAPTURE TERBARU (KENDARAAN/MOTOR/ORANG)
# ============================================================

def draw_latest_capture(frame, capture):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_WIDTH, max(300, w - 20))
    panel_x = w - panel_w - 12
    panel_y = 12
    panel_h = 325

    draw_panel_box(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.90)

    if capture is None:
        draw_text(frame, "CAPTURE TERBARU", (panel_x + 15, panel_y + 28), 0.62, 2, color=(0, 255, 0))
        draw_text(frame, "Belum ada objek selesai dideteksi.", (panel_x + 15, panel_y + 80), 0.52, 1, color=(180, 180, 180))
        draw_text(frame, "Menunggu kendaraan atau orang...", (panel_x + 15, panel_y + 110), 0.50, 1, color=(140, 140, 140))
        return

    ctype = capture.get("type", "VEHICLE")

    if ctype == "RIDER":
        # Header Badge
        cv2.rectangle(frame, (panel_x + 12, panel_y + 10), (panel_x + panel_w - 12, panel_y + 34), (30, 80, 140), -1)
        draw_text(frame, "CAPTURE: MOTOR & PENGENDARA", (panel_x + 20, panel_y + 27), 0.55, 2, color=(0, 220, 255))

        # Crops: Kiri (Motor+Pengendara), Kanan Atas (Plat), Kanan Bawah (Wajah)
        put_crop_on_panel(frame, capture.get("unit_crop"), panel_x + 12, panel_y + 42, 185, 128, "Unit Motor N/A")
        put_crop_on_panel(frame, capture.get("plate_crop"), panel_x + 205, panel_y + 42, 190, 60, "Plat N/A")
        put_crop_on_panel(frame, capture.get("face_crop"), panel_x + 205, panel_y + 108, 190, 62, "Wajah N/A")

        # Labels
        plate_text = capture.get("formatted", capture.get("text", "-"))
        conf = float(capture.get("confidence", 0.0) or 0.0)
        det_conf = float(capture.get("detection_confidence", 0.0) or 0.0)
        track_id = capture.get("track_id", "-")
        rider_id = capture.get("rider_id", "-")
        ts = capture.get("timestamp", "-")

        draw_text(frame, f"Plat    : {plate_text}", (panel_x + 15, panel_y + 195), 0.65, 2, color=(0, 255, 0))
        draw_text(frame, "Status  : Pengendara Motor", (panel_x + 15, panel_y + 222), 0.52, 1, color=(0, 215, 255))
        draw_text(frame, f"OCR     : {conf:.1%} | YOLO: {det_conf:.1%}", (panel_x + 15, panel_y + 248), 0.50, 1, color=(220, 220, 220))
        draw_text(frame, f"Track   : Plat #{track_id} & Rider #{rider_id}", (panel_x + 15, panel_y + 274), 0.50, 1, color=(200, 200, 200))
        draw_text(frame, ts, (panel_x + 15, panel_y + 300), 0.45, 1, color=(160, 160, 160))

    elif ctype == "PEDESTRIAN":
        # Header Badge
        cv2.rectangle(frame, (panel_x + 12, panel_y + 10), (panel_x + panel_w - 12, panel_y + 34), (100, 70, 20), -1)
        draw_text(frame, "CAPTURE: PEJALAN KAKI", (panel_x + 20, panel_y + 27), 0.55, 2, color=(255, 220, 0))

        # Crops: Kiri (Orang Penuh), Kanan (Wajah/Kepala)
        put_crop_on_panel(frame, capture.get("unit_crop"), panel_x + 12, panel_y + 42, 185, 128, "Orang N/A")
        put_crop_on_panel(frame, capture.get("face_crop"), panel_x + 205, panel_y + 42, 190, 128, "Wajah N/A")

        conf = float(capture.get("confidence", 0.0) or 0.0)
        track_id = capture.get("track_id", "-")
        ts = capture.get("timestamp", "-")

        draw_text(frame, "Objek   : Pejalan Kaki (Tanpa Kendaraan)", (panel_x + 15, panel_y + 195), 0.54, 2, color=(255, 220, 0))
        draw_text(frame, "Status  : Terdeteksi & Dicapture", (panel_x + 15, panel_y + 222), 0.52, 1, color=(200, 255, 200))
        draw_text(frame, f"Akurasi : {conf:.1%}", (panel_x + 15, panel_y + 248), 0.50, 1, color=(220, 220, 220))
        draw_text(frame, f"Track   : #{track_id}", (panel_x + 15, panel_y + 274), 0.50, 1, color=(200, 200, 200))
        draw_text(frame, ts, (panel_x + 15, panel_y + 300), 0.45, 1, color=(160, 160, 160))

    else:
        # Standalone Kendaraan / Mobil
        cv2.rectangle(frame, (panel_x + 12, panel_y + 10), (panel_x + panel_w - 12, panel_y + 34), (20, 80, 40), -1)
        draw_text(frame, "CAPTURE: KENDARAAN / PLAT", (panel_x + 20, panel_y + 27), 0.55, 2, color=(0, 255, 150))

        put_crop_on_panel(frame, capture.get("plate_crop"), panel_x + 20, panel_y + 45, panel_w - 40, 122, "Plat N/A")

        plate_text = capture.get("formatted", capture.get("text", "-"))
        conf = float(capture.get("confidence", 0.0) or 0.0)
        det_conf = float(capture.get("detection_confidence", 0.0) or 0.0)
        track_id = capture.get("track_id", "-")
        ts = capture.get("timestamp", "-")

        draw_text(frame, f"Plat    : {plate_text}", (panel_x + 15, panel_y + 195), 0.65, 2, color=(0, 255, 0))
        draw_text(frame, "Status  : Kendaraan (Plat Nomor)", (panel_x + 15, panel_y + 222), 0.52, 1, color=(200, 255, 200))
        draw_text(frame, f"OCR     : {conf:.1%} | YOLO: {det_conf:.1%}", (panel_x + 15, panel_y + 248), 0.50, 1, color=(220, 220, 220))
        draw_text(frame, f"Track   : #{track_id}", (panel_x + 15, panel_y + 274), 0.50, 1, color=(200, 200, 200))
        draw_text(frame, ts, (panel_x + 15, panel_y + 300), 0.45, 1, color=(160, 160, 160))


# ============================================================
# PANEL KANAN: HISTORY 5 DETEKSI TERAKHIR
# ============================================================

def draw_history(frame, history):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_WIDTH, max(300, w - 20))
    panel_x = w - panel_w - 12
    panel_y = 345
    panel_h = 362

    draw_panel_box(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.90)
    draw_text(frame, "HISTORY 5 DETEKSI TERAKHIR", (panel_x + 15, panel_y + 26), 0.58, 2, color=(0, 255, 200))

    if not history:
        draw_text(frame, "Belum ada history deteksi.", (panel_x + 15, panel_y + 65), 0.50, 1, color=(160, 160, 160))
        return

    row_h = 62
    for index, item in enumerate(history[:HISTORY_SIZE]):
        y = panel_y + 40 + index * row_h
        ctype = item.get("type", "VEHICLE")
        conf = float(item.get("confidence", 0.0) or 0.0)
        track_id = item.get("track_id", "-")

        # Garis pembatas tipis
        cv2.line(frame, (panel_x + 15, y), (panel_x + panel_w - 15, y), (40, 40, 40), 1)

        if ctype == "RIDER":
            badge = "[MOTOR]"
            badge_color = (0, 215, 255)
            line1 = f"{index + 1}. #{track_id} {item.get('formatted', item.get('text', ''))}"
            line2 = f"   Pengendara + Plat | OCR {conf:.0%}"
        elif ctype == "PEDESTRIAN":
            badge = "[ORANG]"
            badge_color = (255, 220, 0)
            line1 = f"{index + 1}. #{track_id} Pejalan Kaki"
            line2 = f"   Deteksi Orang/Wajah ({conf:.0%})"
        else:
            badge = "[PLAT ]"
            badge_color = (0, 255, 100)
            line1 = f"{index + 1}. #{track_id} {item.get('formatted', item.get('text', ''))}"
            line2 = f"   Kendaraan | OCR {conf:.0%}"

        draw_text(frame, badge, (panel_x + 15, y + 20), 0.48, 1, color=badge_color)
        draw_text(frame, line1, (panel_x + 85, y + 20), 0.50, 1, color=(240, 240, 240))
        draw_text(frame, line2, (panel_x + 15, y + 42), 0.44, 1, color=(160, 160, 160))


# ============================================================
# PANEL KIRI ATAS: STATUS & FPS
# ============================================================

def draw_info(frame, plate_count, person_count, fps):
    cv2.rectangle(frame, (10, 10), (330, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (330, 130), (60, 60, 60), 1)

    cv2.putText(frame, f"Display FPS : {fps:.1f}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Plat Aktif  : {plate_count}", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Orang/Rider : {person_count}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Resolution  : {width}x{height}", (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# DRAW OVERLAY (TRACKING PLAT & PERSON DENGAN LINKING)
# ============================================================

def draw_track_overlay(frame, tracks, persons):
    # 1. Gambar Bounding Box Plat Nomor (Hijau)
    for track in tracks:
        bbox = scale_bbox_to_display(track.bbox, width, height, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        hasil = track.hasil_voting()
        if hasil is not None:
            label = f"PLAT #{track.id} | {hasil.get('formatted', hasil['text'])}"
        else:
            label = f"PLAT #{track.id} | Membaca..."

        cv2.putText(frame, label, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. Gambar Bounding Box Orang / Pengendara
    for p in persons:
        bbox = scale_bbox_to_display(p["bbox"], width, height, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        px1, py1, px2, py2 = bbox

        if p.get("is_rider"):
            # Pengendara motor (Oranye/Kuning)
            box_color = (0, 200, 255)
            label = f"RIDER #{p['id']} [PLAT #{p.get('linked_plate_id', '-')}]"
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

            # Garis penghubung ke plat
            if p.get("linked_plate_bbox") is not None:
                lb = scale_bbox_to_display(p["linked_plate_bbox"], width, height, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                rcx = (px1 + px2) // 2
                rcy = (py1 + py2) // 2
                lcx = (lb[0] + lb[2]) // 2
                lcy = (lb[1] + lb[3]) // 2
                cv2.line(frame, (rcx, rcy), (lcx, lcy), box_color, 2)
        else:
            # Pejalan kaki (Cyan)
            box_color = (255, 220, 0)
            label = f"PEJALAN KAKI #{p['id']}"
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

        cv2.putText(frame, label, (px1, max(22, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2, cv2.LINE_AA)


# ============================================================
# START BACKGROUND THREADS
# ============================================================

reader_thread = threading.Thread(
    target=baca_frame_terus,
    daemon=True,
)

yolo_thread = threading.Thread(
    target=yolo_worker,
    daemon=True,
)

print("\n[SYSTEM] Starting reader & YOLO threads...")
reader_thread.start()
time.sleep(0.5)
yolo_thread.start()


# ============================================================
# SETUP WINDOW DISPLAY
# ============================================================

window_name = "CCTV Monitoring - Plate & Person/Rider AI"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)


# ============================================================
# DISPLAY LOOP
# ============================================================

frame_count = 0
start_time = time.time()

print("\n" + "=" * 60)
print("[SYSTEM] Pipeline AI aktif: CCTV + YOLO (Plat & Orang) + OCR")
print("[SYSTEM] Tekan 'Q' untuk keluar")
print("=" * 60)

while running:
    with frame_lock:
        if latest_frame is None:
            frame = None
            capture = None
            history = []
            tracks = []
            persons = []
        else:
            frame = latest_frame.copy()
            capture = latest_capture.copy() if latest_capture is not None else None
            history = list(latest_history)
            tracks = list(latest_active_tracks)
            persons = list(latest_active_persons)

    if frame is None:
        time.sleep(0.01)
        continue

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = (frame_count / elapsed_time) if elapsed_time > 0 else 0

    display = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    with detections_lock:
        detections = latest_detections.copy()

    # Render Visuals
    draw_track_overlay(display, tracks, persons)
    draw_latest_capture(display, capture)
    draw_history(display, history)
    draw_info(display, len(tracks), len(persons), fps)

    cv2.imshow(window_name, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        print("\n[SYSTEM] Program dihentikan user.")
        running = False
        break


# ============================================================
# CLEANUP
# ============================================================

print("\n[SYSTEM] Stopping threads and closing camera...")
running = False
time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("TEST SELESAI")
print(f"Total frame display : {frame_count}")
print("=" * 70)