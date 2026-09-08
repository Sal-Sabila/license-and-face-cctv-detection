import cv2
import time
import threading
import queue
import re

from difflib import SequenceMatcher

from ai.plate.detector import PlateDetector
from ai.plate.ocr import PlateOCR


# ============================================================
# CONFIGURATION
# ============================================================

RTMP_URL = (
    "rtmp://103.255.15.222:1935/"
    "atcs-kota/"
    "FMNotoMcDonalds.stream"
)

# ------------------------------------------------------------
# DISPLAY
# Frame asli TIDAK diperkecil sebelum OCR.
# Ukuran ini hanya untuk jendela tampilan.
# ------------------------------------------------------------

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------

YOLO_IMGSZ = 960
YOLO_CONFIDENCE = 0.30

# Jalankan YOLO sekitar 5x/detik
AI_INTERVAL = 0.20

# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

# Beri waktu track mengumpulkan crop yang lebih baik.
OCR_FIRST_DELAY = 0.5

# Satu track di-OCR berkala selama belum STABLE.
OCR_INTERVAL = 1.2

# OCR CPU bisa lama, sehingga track dipertahankan lebih lama.
TRACK_TIMEOUT = 20.0

# ------------------------------------------------------------
# TRACKING
# ------------------------------------------------------------

MAX_TRACKS = 30

IOU_THRESHOLD = 0.50

CENTER_DISTANCE_THRESHOLD = 120

# ------------------------------------------------------------
# OCR VOTING
# ------------------------------------------------------------

MAX_OCR_HISTORY = 8

MIN_VOTES = 3

# Hasil OCR minimal 3 karakter agar hasil sangat pendek seperti "MN"
# tidak langsung dianggap sebagai plat final.
MIN_STABLE_TEXT_LENGTH = 3

# Setelah stable, track tidak akan dikirim OCR lagi.
STOP_OCR_WHEN_STABLE = True

# Kemiripan OCR agar hasil seperti B1234ABC dan B1234A8C
# tetap bisa masuk ke kelompok voting yang sama.
OCR_SIMILARITY_THRESHOLD = 0.75

# ------------------------------------------------------------
# CROP
# ------------------------------------------------------------

CROP_PAD_X = 0.15
CROP_PAD_Y = 0.15

MIN_CROP_WIDTH = 60
MIN_CROP_HEIGHT = 20

MIN_PLATE_ASPECT_RATIO = 2.0
MAX_PLATE_ASPECT_RATIO = 6.0

# Minimal skor agar crop layak dikirim ke OCR.
# Crop yang sangat kecil/aneh tidak diproses.
MIN_CROP_SCORE_FOR_OCR = 0.65

# ------------------------------------------------------------
# HASIL OCR TERAKHIR
# ------------------------------------------------------------

# Hasil OCR yang berhasil tetap ditampilkan walaupun track
# yang bersangkutan sudah tidak terlihat lagi.
LAST_RESULT_DISPLAY_SECONDS = 15.0

# ------------------------------------------------------------
# DEBUG
# ------------------------------------------------------------

SAVE_DEBUG_CROP = True

# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

# Batasi log AI agar console tidak terlalu penuh.
PRINT_AI_LOG = True


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
latest_frame_lock = threading.Lock()

latest_detections = []
detections_lock = threading.Lock()

stop_event = threading.Event()

# Hanya satu OCR berjalan pada satu waktu karena CPU.
ocr_queue = queue.Queue(maxsize=1)

display_tracks = []
display_tracks_lock = threading.Lock()

tracks = {}
tracks_lock = threading.Lock()

track_counter = 0
track_counter_lock = threading.Lock()

ocr_busy = False
ocr_busy_lock = threading.Lock()

# ------------------------------------------------------------
# GLOBAL LAST OCR RESULT
# ------------------------------------------------------------

last_ocr_result = None
last_ocr_result_lock = threading.Lock()


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():
    global latest_frame

    print("[CAMERA] Opening RTMP...")

    cap = cv2.VideoCapture(
        RTMP_URL,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print(
            "[CAMERA ERROR] "
            "Tidak dapat membuka RTMP"
        )

        stop_event.set()
        return

    print("[CAMERA] Connected")

    logged = False

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:

            time.sleep(0.05)
            continue

        if not logged:

            print(
                f"[CAMERA] Resolution: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

            logged = True

        # ----------------------------------------------------
        # SIMPAN FRAME ASLI.
        # Jangan resize di sini.
        # ----------------------------------------------------

        with latest_frame_lock:
            latest_frame = frame

    cap.release()

    print(
        "[CAMERA] Thread stopped"
    )


# ============================================================
# GET FRAME
# ============================================================

def get_latest_frame():

    with latest_frame_lock:

        if latest_frame is None:
            return None

        return latest_frame.copy()


# ============================================================
# BBOX UTILS
# ============================================================

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


def bbox_center(
    bbox
):

    return (
        (bbox[0] + bbox[2]) / 2.0,
        (bbox[1] + bbox[3]) / 2.0
    )


def bbox_distance(
    box1,
    box2
):

    c1 = bbox_center(
        box1
    )

    c2 = bbox_center(
        box2
    )

    return (
        (
            c1[0] - c2[0]
        ) ** 2
        +
        (
            c1[1] - c2[1]
        ) ** 2
    ) ** 0.5


# ============================================================
# TRACK ID
# ============================================================

def get_new_track_id():

    global track_counter

    with track_counter_lock:

        track_counter += 1

        return track_counter


# ============================================================
# CREATE TRACK
# ============================================================

def create_track(
    detection
):

    track_id = get_new_track_id()

    now = time.time()

    track = {

        "id":
            track_id,

        "bbox":
            detection["bbox"],

        "confidence":
            detection["confidence"],

        "first_seen":
            now,

        "last_seen":
            now,

        "last_ocr":
            0.0,

        "ocr_running":
            False,

        "ocr_history":
            [],

        "best_text":
            "",

        "best_confidence":
            0.0,

        "stable_text":
            "",

        "stable_votes":
            0,

        "ocr_stable":
            False,

        "crop":
            None,

        "best_crop":
            None,

        "best_crop_score":
            0.0,
    }

    with tracks_lock:

        tracks[
            track_id
        ] = track

    print(
        f"[TRACK] New plate ID={track_id}"
    )

    return track_id


# ============================================================
# FIND MATCHING TRACK
# ============================================================

def find_matching_track(
    detection,
    used_tracks
):

    current_bbox = detection[
        "bbox"
    ]

    now = time.time()

    with tracks_lock:

        track_items = list(
            tracks.items()
        )

    best_id = None

    best_score = float(
        "-inf"
    )

    for track_id, track in track_items:

        if track_id in used_tracks:
            continue

        if (
            now
            - track["last_seen"]
            > TRACK_TIMEOUT
        ):
            continue

        previous_bbox = track[
            "bbox"
        ]

        iou = calculate_iou(
            current_bbox,
            previous_bbox
        )

        distance = bbox_distance(
            current_bbox,
            previous_bbox
        )

        if iou >= IOU_THRESHOLD:

            score = (
                iou * 2.0
                -
                distance * 0.001
            )

        elif distance <= CENTER_DISTANCE_THRESHOLD:

            score = (
                0.5
                -
                distance * 0.001
            )

        else:

            continue

        if score > best_score:

            best_score = score
            best_id = track_id

    return best_id


# ============================================================
# UPDATE TRACKS
# ============================================================

def update_tracks(
    detections
):

    now = time.time()

    used_tracks = set()

    current_ids = []

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    for detection in detections:

        track_id = find_matching_track(
            detection,
            used_tracks
        )

        if track_id is None:

            track_id = create_track(
                detection
            )

        # Track ini sudah dipakai oleh detection pada frame AI ini.
        used_tracks.add(
            track_id
        )

        with tracks_lock:

            track = tracks.get(
                track_id
            )

            if track is None:
                continue

            track[
                "bbox"
            ] = detection[
                "bbox"
            ]

            track[
                "confidence"
            ] = detection[
                "confidence"
            ]

            track[
                "last_seen"
            ] = now

        current_ids.append(
            track_id
        )

    # --------------------------------------------------------
    # Hapus track yang sudah terlalu lama.
    #
    # TRACK_TIMEOUT dibuat 20 detik karena OCR CPU Anda
    # sebelumnya bisa memakan 7-11 detik.
    # --------------------------------------------------------

    with tracks_lock:

        expired_ids = [

            track_id

            for track_id, track
            in tracks.items()

            if (
                now
                -
                track["last_seen"]
                >
                TRACK_TIMEOUT
            )
        ]

        for track_id in expired_ids:

            print(
                f"[TRACK] Remove ID={track_id}"
            )

            del tracks[
                track_id
            ]

        # ----------------------------------------------------
        # Batas jumlah track
        # ----------------------------------------------------

        if len(tracks) > MAX_TRACKS:

            ordered = sorted(
                tracks.items(),
                key=lambda item:
                    item[1]["last_seen"]
            )

            remove_count = (
                len(tracks)
                -
                MAX_TRACKS
            )

            for track_id, _ in ordered[
                :remove_count
            ]:

                tracks.pop(
                    track_id,
                    None
                )

    return current_ids


# ============================================================
# CROP PLATE
# ============================================================

def crop_plate(
    frame,
    bbox
):

    if frame is None:
        return None

    if frame.size == 0:
        return None

    height, width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(
            int(x1),
            width - 1
        )
    )

    y1 = max(
        0,
        min(
            int(y1),
            height - 1
        )
    )

    x2 = max(
        0,
        min(
            int(x2),
            width
        )
    )

    y2 = max(
        0,
        min(
            int(y2),
            height
        )
    )

    if x2 <= x1:
        return None

    if y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    if box_height <= 0:
        return None

    # --------------------------------------------------------
    # Rasio bbox asli
    # --------------------------------------------------------

    aspect_ratio = (
        box_width
        /
        box_height
    )

    if (
        aspect_ratio
        <
        MIN_PLATE_ASPECT_RATIO

        or

        aspect_ratio
        >
        MAX_PLATE_ASPECT_RATIO
    ):

        return None

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    pad_x = int(
        box_width
        *
        CROP_PAD_X
    )

    pad_y = int(
        box_height
        *
        CROP_PAD_Y
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        width,
        x2 + pad_x
    )

    y2 = min(
        height,
        y2 + pad_y
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop is None:
        return None

    if crop.size == 0:
        return None

    crop_height, crop_width = (
        crop.shape[:2]
    )

    if (
        crop_width
        <
        MIN_CROP_WIDTH

        or

        crop_height
        <
        MIN_CROP_HEIGHT
    ):

        return None

    return crop.copy()


# ============================================================
# CROP QUALITY SCORE
# ============================================================

def calculate_crop_score(
    crop
):

    if crop is None:
        return 0.0

    if crop.size == 0:
        return 0.0

    height, width = (
        crop.shape[:2]
    )

    if height <= 0:
        return 0.0

    aspect = (
        width
        /
        height
    )

    # --------------------------------------------------------
    # Rasio ideal sekitar 2.5
    # --------------------------------------------------------

    aspect_score = max(
        0.0,
        1.0
        -
        abs(
            aspect - 2.5
        )
        /
        3.5
    )

    # --------------------------------------------------------
    # Ukuran
    # --------------------------------------------------------

    width_score = min(
        width / 80.0,
        1.0
    )

    height_score = min(
        height / 30.0,
        1.0
    )

    size_score = (
        width_score * 0.65
        +
        height_score * 0.35
    )

    return (
        aspect_score * 0.55
        +
        size_score * 0.45
    )


# ============================================================
# TEXT NORMALIZE
# ============================================================

def normalize_plate_text(
    text
):

    if not text:

        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


# ============================================================
# TEXT SIMILARITY
# ============================================================

def text_similarity(
    text1,
    text2
):

    if not text1 or not text2:

        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# ============================================================
# OCR VOTING
# ============================================================

def add_ocr_result(
    track,
    text,
    confidence
):

    text = normalize_plate_text(
        text
    )

    if not text:
        return

    if len(text) < MIN_STABLE_TEXT_LENGTH:
        return

    if len(text) > 12:
        return

    try:

        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.0

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    track[
        "ocr_history"
    ].append({

        "text":
            text,

        "confidence":
            confidence,

        "time":
            time.time()
    })

    track[
        "ocr_history"
    ] = track[
        "ocr_history"
    ][
        -MAX_OCR_HISTORY:
    ]

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    if (
        confidence
        >
        track["best_confidence"]
    ):

        track[
            "best_confidence"
        ] = confidence

        track[
            "best_text"
        ] = text

    # --------------------------------------------------------
    # VOTING
    # --------------------------------------------------------

    calculate_voting(
        track
    )


# ============================================================
# CALCULATE VOTING
# ============================================================

def calculate_voting(
    track
):

    history = track[
        "ocr_history"
    ]

    if not history:
        return

    groups = []

    for item in history:

        item_text = item[
            "text"
        ]

        matched = False

        for group in groups:

            if (
                text_similarity(
                    item_text,
                    group[
                        "representative"
                    ]
                )
                >= OCR_SIMILARITY_THRESHOLD
            ):

                group[
                    "items"
                ].append(
                    item
                )

                matched = True

                break

        if not matched:

            groups.append({

                "representative":
                    item_text,

                "items":
                    [item]
            })

    if not groups:
        return

    # --------------------------------------------------------
    # Group dengan voting terbanyak lalu confidence tertinggi
    # --------------------------------------------------------

    best_group = max(
        groups,
        key=lambda group: (
            len(
                group["items"]
            ),
            sum(
                item["confidence"]
                for item
                in group["items"]
            ) / len(group["items"]),
            max(
                item["confidence"]
                for item
                in group["items"]
            )
        )
    )

    items = best_group[
        "items"
    ]

    votes = len(
        items
    )

    best_item = max(
        items,
        key=lambda item:
            item["confidence"]
    )

    candidate_text = best_item[
        "text"
    ]

    candidate_confidence = (
        best_item[
            "confidence"
        ]
    )

    # Jangan jadikan hasil sangat pendek sebagai plat final.
    if len(candidate_text) < MIN_STABLE_TEXT_LENGTH:
        return

    # Stable hanya jika voting benar-benar mencapai minimum.
    # Confidence tinggi satu kali saja TIDAK langsung membuat stable.
    if votes >= MIN_VOTES:

        previous = track[
            "stable_text"
        ]

        track[
            "stable_text"
        ] = candidate_text

        track[
            "stable_votes"
        ] = votes

        track[
            "ocr_stable"
        ] = True

        if previous != candidate_text:

            print(
                "[OCR STABLE] "
                f"ID={track['id']} "
                f"plate={candidate_text} "
                f"votes={votes} "
                f"conf={candidate_confidence:.3f} "
                f"-> STOP OCR"
            )


# ============================================================
# SUBMIT OCR
# ============================================================

def submit_ocr(
    track_id,
    crop
):

    global ocr_busy

    if crop is None:
        return False

    if crop.size == 0:
        return False

    crop_score = calculate_crop_score(
        crop
    )

    # --------------------------------------------------------
    # Jangan kirim crop yang terlalu buruk
    # --------------------------------------------------------

    if (
        crop_score
        <
        MIN_CROP_SCORE_FOR_OCR
    ):

        return False

    now = time.time()

    with tracks_lock:

        track = tracks.get(
            track_id
        )

        if track is None:
            return False

        # ----------------------------------------------------
        # STOP OCR SETELAH HASIL STABIL
        # ----------------------------------------------------

        if (
            STOP_OCR_WHEN_STABLE
            and
            track.get("ocr_stable", False)
        ):
            return False

        # ----------------------------------------------------
        # Track baru -> beri waktu mencari crop lebih baik
        # ----------------------------------------------------

        if (
            now
            -
            track["first_seen"]
            <
            OCR_FIRST_DELAY
        ):

            return False

        # ----------------------------------------------------
        # Interval OCR
        # ----------------------------------------------------

        if (
            now
            -
            track["last_ocr"]
            <
            OCR_INTERVAL
        ):

            return False

        if track[
            "ocr_running"
        ]:

            return False

    # --------------------------------------------------------
    # OCR global sedang sibuk
    # --------------------------------------------------------

    with ocr_busy_lock:

        if ocr_busy:

            return False

    # --------------------------------------------------------
    # Simpan status
    # --------------------------------------------------------

    with tracks_lock:

        track = tracks.get(
            track_id
        )

        if track is None:
            return False

        # Double-check karena state bisa berubah di antara dua lock.
        if (
            STOP_OCR_WHEN_STABLE
            and
            track.get("ocr_stable", False)
        ):
            return False

        track[
            "last_ocr"
        ] = now

        track[
            "ocr_running"
        ] = True

        track[
            "crop"
        ] = crop.copy()

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    if SAVE_DEBUG_CROP:

        try:

            cv2.imwrite(
                f"debug_plate_crop_"
                f"{track_id}.jpg",
                crop
            )

        except Exception as exc:

            print(
                f"[DEBUG CROP ERROR] "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Queue
    # --------------------------------------------------------

    try:

        # Hanya pertahankan job terbaru.
        while True:

            try:

                old_track_id, _ = (
                    ocr_queue.get_nowait()
                )

            except queue.Empty:

                break

            with tracks_lock:

                old_track = tracks.get(
                    old_track_id
                )

                if old_track is not None:

                    old_track[
                        "ocr_running"
                    ] = False

            ocr_queue.task_done()

        ocr_queue.put_nowait(
            (
                track_id,
                crop.copy()
            )
        )

        print(
            f"[OCR QUEUE] "
            f"Track={track_id} "
            f"crop={crop.shape[1]}x"
            f"{crop.shape[0]} "
            f"score={crop_score:.3f}"
        )

        return True

    except queue.Full:

        with tracks_lock:

            if track_id in tracks:

                tracks[
                    track_id
                ][
                    "ocr_running"
                ] = False

        return False


# ============================================================
# SAVE GLOBAL OCR RESULT
# ============================================================

def save_last_ocr_result(
    track_id,
    text,
    confidence,
    variant,
    crop
):

    global last_ocr_result

    with last_ocr_result_lock:

        last_ocr_result = {

            "track_id":
                track_id,

            "text":
                text,

            "confidence":
                confidence,

            "variant":
                variant,

            "crop":
                crop.copy()
                if crop is not None
                else None,

            "time":
                time.time()
        }


# ============================================================
# OCR WORKER
# ============================================================

def ocr_worker():

    global ocr_busy

    print(
        "[OCR WORKER] Starting..."
    )

    # Scale 2x cukup untuk testing dan lebih ringan daripada 3x.
    ocr = PlateOCR(
        scale=3.0,
        min_confidence=0.20,
        verbose=False
    )

    print(
        "[OCR WORKER] Ready"
    )

    while not stop_event.is_set():

        try:

            track_id, crop = (
                ocr_queue.get(
                    timeout=0.2
                )
            )

        except queue.Empty:

            continue

        with ocr_busy_lock:

            ocr_busy = True

        try:

            started = (
                time.perf_counter()
            )

            print("=" * 60)

            print(
                f"[OCR START] "
                f"Track={track_id}"
            )

            result = ocr.read(
                crop
            )

            elapsed = (
                time.perf_counter()
                -
                started
            )

            # ------------------------------------------------
            # Track boleh saja sudah hilang.
            # Hasil OCR tetap disimpan secara global.
            # ------------------------------------------------

            if result:

                text = result.get(
                    "text",
                    ""
                )

                confidence = float(
                    result.get(
                        "confidence",
                        0.0
                    )
                )

                variant = result.get(
                    "variant",
                    "unknown"
                )

                text = normalize_plate_text(
                    text
                )

                if text:

                    with tracks_lock:

                        track = tracks.get(
                            track_id
                        )

                        if track is not None:

                            add_ocr_result(
                                track,
                                text,
                                confidence
                            )

                    save_last_ocr_result(
                        track_id,
                        text,
                        confidence,
                        variant,
                        crop
                    )

                    print(
                        f"[OCR RESULT] "
                        f"{text} | "
                        f"conf={confidence:.3f} | "
                        f"variant={variant} | "
                        f"{elapsed:.2f}s"
                    )

                else:

                    print(
                        f"[OCR RESULT] "
                        f"kosong | "
                        f"{elapsed:.2f}s"
                    )

            else:

                print(
                    f"[OCR RESULT] "
                    f"kosong | "
                    f"{elapsed:.2f}s"
                )

        except Exception as exc:

            print(
                f"[OCR WORKER ERROR] "
                f"{exc}"
            )

        finally:

            with tracks_lock:

                if track_id in tracks:

                    tracks[
                        track_id
                    ][
                        "ocr_running"
                    ] = False

            with ocr_busy_lock:

                ocr_busy = False

            ocr_queue.task_done()

    print(
        "[OCR WORKER] Stopped"
    )


# ============================================================
# AI WORKER
# ============================================================

def ai_worker():

    global latest_detections
    global display_tracks

    detector = PlateDetector(
        confidence=YOLO_CONFIDENCE,
        imgsz=YOLO_IMGSZ
    )

    print(
        "[AI] Worker started"
    )

    while not stop_event.is_set():

        cycle_start = (
            time.perf_counter()
        )

        frame = get_latest_frame()

        if frame is None:

            time.sleep(0.05)

            continue

        # ====================================================
        # YOLO PADA FRAME ASLI
        # ====================================================

        yolo_start = (
            time.perf_counter()
        )

        detections = detector.detect(
            frame
        )

        yolo_time = (
            time.perf_counter()
            -
            yolo_start
        )

        with detections_lock:

            latest_detections = (
                detections.copy()
            )

        # ====================================================
        # TRACK
        # ====================================================

        current_ids = update_tracks(
            detections
        )

        # ====================================================
        # KUMPULKAN CROP TERBAIK
        # ====================================================

        for track_id in current_ids:

            with tracks_lock:

                track = tracks.get(
                    track_id
                )

                if track is None:
                    continue

                bbox = track[
                    "bbox"
                ]

            crop = crop_plate(
                frame,
                bbox
            )

            if crop is None:
                continue

            score = calculate_crop_score(
                crop
            )

            with tracks_lock:

                track = tracks.get(
                    track_id
                )

                if track is None:
                    continue

                if (
                    score
                    >
                    track[
                        "best_crop_score"
                    ]
                ):

                    track[
                        "best_crop_score"
                    ] = score

                    track[
                        "best_crop"
                    ] = crop.copy()

                    track[
                        "crop"
                    ] = crop.copy()

                    print(
                        f"[CROP BEST] "
                        f"Track={track_id} "
                        f"size={crop.shape[1]}x"
                        f"{crop.shape[0]} "
                        f"score={score:.3f}"
                    )

        # ====================================================
        # OCR
        # ====================================================

        for track_id in current_ids:

            with tracks_lock:

                track = tracks.get(
                    track_id
                )

                if track is None:
                    continue

                # OCR voting harus memakai crop dari frame-frame berbeda.
                # best_crop tetap disimpan sebagai crop terbaik untuk display/debug,
                # tetapi bukan satu-satunya crop yang dikirim ke OCR.
                current_crop = track.get(
                    "crop"
                )

                # Jika sudah stabil, jangan proses OCR lagi.
                if (
                    STOP_OCR_WHEN_STABLE
                    and
                    track.get("ocr_stable", False)
                ):
                    continue

                best_crop = track.get(
                    "best_crop"
                )

                current_score = calculate_crop_score(
                    current_crop
                ) if current_crop is not None else 0.0

                best_crop_score = float(
                    track.get(
                        "best_crop_score",
                        0.0
                    )
                )

                # Prioritaskan crop frame saat ini jika kualitasnya cukup.
                # Jika belum cukup, fallback ke best crop.
                if (
                    current_crop is not None
                    and
                    current_score >= MIN_CROP_SCORE_FOR_OCR
                ):
                    ocr_crop = current_crop.copy()
                elif (
                    best_crop is not None
                    and
                    best_crop_score >= MIN_CROP_SCORE_FOR_OCR
                ):
                    ocr_crop = best_crop.copy()
                else:
                    ocr_crop = None

            if ocr_crop is None:
                continue

            submitted = submit_ocr(
                track_id,
                ocr_crop
            )

            if submitted:
                with tracks_lock:
                    current_track = tracks.get(track_id)

                    if current_track is not None:
                        history_count = len(
                            current_track.get(
                                "ocr_history",
                                []
                            )
                        )

                        print(
                            f"[OCR COLLECT] "
                            f"ID={track_id} "
                            f"history={history_count}/{MIN_VOTES}"
                        )

        # ====================================================
        # DISPLAY TRACKS
        # ====================================================

        now = time.time()

        with tracks_lock:

            visible = [

                track.copy()

                for track
                in tracks.values()

                if (
                    now
                    -
                    track["last_seen"]
                    <=
                    TRACK_TIMEOUT
                )
            ]

        with display_tracks_lock:

            display_tracks = visible

        if PRINT_AI_LOG:
            print(
                f"[AI] "
                f"YOLO={yolo_time:.2f}s "
                f"DET={len(detections)} "
                f"DISPLAY={len(visible)}"
            )

        # ====================================================
        # INTERVAL
        # ====================================================

        elapsed = (
            time.perf_counter()
            -
            cycle_start
        )

        delay = (
            AI_INTERVAL
            -
            elapsed
        )

        if delay > 0:

            time.sleep(
                delay
            )

    print(
        "[AI] Worker stopped"
    )


# ============================================================
# DISPLAY BBOX
# ============================================================

def scale_bbox_to_display(
    bbox,
    original_width,
    original_height
):

    sx = (
        DISPLAY_WIDTH
        /
        original_width
    )

    sy = (
        DISPLAY_HEIGHT
        /
        original_height
    )

    return [

        int(
            bbox[0]
            * sx
        ),

        int(
            bbox[1]
            * sy
        ),

        int(
            bbox[2]
            * sx
        ),

        int(
            bbox[3]
            * sy
        )
    ]


# ============================================================
# DRAW TRACK
# ============================================================

def draw_track(
    frame,
    track,
    original_width,
    original_height
):

    bbox = scale_bbox_to_display(
        track["bbox"],
        original_width,
        original_height
    )

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (0, 255, 0),
        2
    )

    track_id = track[
        "id"
    ]

    stable_text = track.get(
        "stable_text",
        ""
    )

    best_text = track.get(
        "best_text",
        ""
    )

    confidence = float(
        track.get(
            "best_confidence",
            0.0
        )
    )

    if stable_text:

        plate_text = stable_text

        label = (
            f"ID {track_id} | "
            f"{plate_text} "
            f"✓ ({confidence:.0%})"
        )

    else:

        history_count = len(
            track.get(
                "ocr_history",
                []
            )
        )

        # Jangan tampilkan best_text sebagai hasil final sebelum voting stabil.
        if history_count > 0:

            label = (
                f"ID {track_id} | "
                f"Mengumpulkan OCR "
                f"({history_count}/{MIN_VOTES})"
            )

        else:

            label = (
                f"ID {track_id} | "
                f"Membaca plat..."
            )

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    label_y = max(
        y1 - 8,
        text_height
        + baseline
        + 5
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_y
            - text_height
            - baseline
        ),
        (
            x1
            + text_width
            + 8,
            label_y
            + 4
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW INFO
# ============================================================

def draw_info(
    frame
):

    with detections_lock:

        detection_count = len(
            latest_detections
        )

    with display_tracks_lock:

        track_count = len(
            display_tracks
        )

    cv2.putText(
        frame,
        (
            f"Plate detected: "
            f"{detection_count}"
        ),
        (
            20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        (
            f"Active tracks: "
            f"{track_count}"
        ),
        (
            20,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (
            20,
            90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# GET LAST OCR RESULT
# ============================================================

def get_last_ocr_result():

    with last_ocr_result_lock:

        if last_ocr_result is None:

            return None

        result = (
            last_ocr_result.copy()
        )

        if result.get(
            "crop"
        ) is not None:

            result[
                "crop"
            ] = result[
                "crop"
            ].copy()

        return result


# ============================================================
# OCR POPUP
# ============================================================

def show_ocr_popup():

    import numpy as np

    popup = np.zeros(
        (
            610,
            820,
            3
        ),
        dtype=np.uint8
    )

    with display_tracks_lock:

        items = [
            track.copy()
            for track
            in display_tracks
        ]

    last_result = (
        get_last_ocr_result()
    )

    now = time.time()

    # ========================================================
    # PRIORITAS:
    # Hasil OCR berhasil yang masih baru.
    # ========================================================

    if (

        last_result is not None

        and

        now
        -
        last_result["time"]
        <=
        LAST_RESULT_DISPLAY_SECONDS

    ):

        crop = last_result.get(
            "crop"
        )

        track_id = last_result[
            "track_id"
        ]

        plate = last_result[
            "text"
        ]

        confidence = float(
            last_result[
                "confidence"
            ]
        )

        title = (
            "HASIL OCR TERAKHIR"
        )

    # ========================================================
    # Kalau belum ada OCR berhasil,
    # gunakan crop terbaik yang sedang tersedia.
    # ========================================================

    elif items:

        candidates = [

            track

            for track
            in items

            if track.get(
                "best_crop"
            ) is not None
        ]

        if candidates:

            selected = max(
                candidates,
                key=lambda track: (
                    track.get(
                        "best_crop_score",
                        0.0
                    ),
                    track.get(
                        "last_seen",
                        0.0
                    )
                )
            )

        else:

            selected = max(
                items,
                key=lambda track:
                    track.get(
                        "last_seen",
                        0.0
                    )
            )

        crop = selected.get(
            "crop"
        )

        if crop is None:

            crop = selected.get(
                "best_crop"
            )

        track_id = selected[
            "id"
        ]

        plate = selected.get(
            "stable_text",
            ""
        )

        # Sebelum voting stabil, jangan tampilkan OCR mentah
        # sebagai hasil final.
        if plate:

            confidence = float(
                selected.get(
                    "best_confidence",
                    0.0
                )
            )

            title = (
                "HASIL DETEKSI PLAT"
            )

        else:

            plate = ""
            confidence = 0.0

            title = (
                "MENGUMPULKAN OCR"
            )

    # ========================================================
    # Tidak ada apa-apa
    # ========================================================

    else:

        cv2.putText(
            popup,
            "Tidak ada plat",
            (
                250,
                300
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Hasil Deteksi Plat",
            popup
        )

        return

    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        popup,
        title,
        (
            220,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # SHOW CROP
    # ========================================================

    if (
        crop is not None
        and
        crop.size > 0
    ):

        crop_display = crop.copy()

        h, w = (
            crop_display.shape[:2]
        )

        if w > 0 and h > 0:

            max_width = 760
            max_height = 360

            scale = min(
                max_width / w,
                max_height / h
            )

            nw = max(
                1,
                int(w * scale)
            )

            nh = max(
                1,
                int(h * scale)
            )

            crop_display = cv2.resize(
                crop_display,
                (
                    nw,
                    nh
                ),
                interpolation=
                    cv2.INTER_LANCZOS4
            )

            x = max(
                0,
                (
                    820
                    -
                    nw
                ) // 2
            )

            y = 80

            end_x = min(
                820,
                x + nw
            )

            end_y = min(
                480,
                y + nh
            )

            actual_width = (
                end_x - x
            )

            actual_height = (
                end_y - y
            )

            if (
                actual_width > 0
                and
                actual_height > 0
            ):

                popup[
                    y:end_y,
                    x:end_x
                ] = crop_display[
                    :actual_height,
                    :actual_width
                ]

    # ========================================================
    # TRACK ID
    # ========================================================

    cv2.putText(
        popup,
        f"Track ID: {track_id}",
        (
            30,
            525
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # RESULT
    # ========================================================

    if plate:

        result_label = (
            f"{plate} "
            f"({confidence:.0%})"
        )

    else:

        result_label = (
            "MEMBACA PLAT..."
        )

    (
        text_width,
        _
    ), _ = cv2.getTextSize(
        result_label,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        3
    )

    text_x = max(
        10,
        (
            820
            -
            text_width
        ) // 2
    )

    cv2.putText(
        popup,
        result_label,
        (
            text_x,
            585
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    cv2.imshow(
        "Hasil Deteksi Plat",
        popup
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "LICENSE PLATE DETECTION + OCR TEST"
    )
    print("=" * 70)

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    ocr_thread = threading.Thread(
        target=ocr_worker,
        daemon=True
    )

    ai_thread = threading.Thread(
        target=ai_worker,
        daemon=True
    )

    # ========================================================
    # START CAMERA
    # ========================================================

    camera_thread.start()

    time.sleep(
        1
    )

    # ========================================================
    # START OCR
    # ========================================================

    ocr_thread.start()

    # ========================================================
    # START AI
    # ========================================================

    ai_thread.start()

    print(
        "[SYSTEM] All workers started"
    )

    print(
        "[SYSTEM] Press Q to quit"
    )

    try:

        while not stop_event.is_set():

            frame = get_latest_frame()

            if frame is None:

                time.sleep(
                    0.05
                )

                continue

            original_height, original_width = (
                frame.shape[:2]
            )

            # =================================================
            # DISPLAY SAJA
            # =================================================

            display = cv2.resize(
                frame,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                ),
                interpolation=cv2.INTER_AREA
            )

            with display_tracks_lock:

                items = [
                    track.copy()
                    for track
                    in display_tracks
                ]

            for track in items:

                draw_track(
                    display,
                    track,
                    original_width,
                    original_height
                )

            draw_info(
                display
            )

            cv2.imshow(
                "CCTV - Plate Detection",
                display
            )

            show_ocr_popup()

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q")
            ):

                print(
                    "[SYSTEM] Stopping..."
                )

                stop_event.set()

                break

    except KeyboardInterrupt:

        print(
            "[SYSTEM] Keyboard interrupt"
        )

        stop_event.set()

    except Exception as exc:

        print(
            f"[SYSTEM ERROR] "
            f"{exc}"
        )

        stop_event.set()

    finally:

        stop_event.set()

        camera_thread.join(
            timeout=3
        )

        ai_thread.join(
            timeout=3
        )

        ocr_thread.join(
            timeout=3
        )

        cv2.destroyAllWindows()

        print()
        print("=" * 70)
        print(
            "TEST SELESAI"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
