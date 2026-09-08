"""
face_detection_pipeline.py (v2)
============================
Prototipe training/eksperimen deteksi wajah untuk 1 kamera CCTV.

PERUBAHAN DI VERSI INI (mengatasi: video slow-motion, bounding box
berkedip, ID berubah-ubah untuk orang yang sama, wajah jelas tidak
terdeteksi):

1. Frame dibaca di THREAD TERPISAH (FrameGrabber) yang selalu menyimpan
   frame TERBARU saja. Loop deteksi mengambil frame terbaru yang
   tersedia, tanpa pernah menumpuk backlog -- ini menghilangkan efek
   slow-motion dan mencegah stream disconnect karena buffer penuh.
2. ByteTrack di-tuning (lost_track_buffer lebih besar) supaya track
   tidak langsung "hilang" hanya karena deteksi gagal di 1-2 frame.
3. Setiap ID wajah menyimpan BEBERAPA embedding (bukan cuma 1), jadi
   pencocokan lebih toleran terhadap variasi pose/pencahayaan --
   mengurangi kasus orang yang sama tiba-tiba dianggap wajah baru.
4. MIN_DET_SCORE diturunkan sedikit dan DET_SIZE dinaikkan supaya
   wajah yang lebih kecil/jauh dari kamera juga bisa terdeteksi.

Instalasi dependency: sama seperti versi sebelumnya (lihat komentar
instalasi di file ini bagian bawah, tidak berubah).
"""

import os
import time
import uuid
import sqlite3
import threading
from datetime import datetime

# beri FFmpeg lebih banyak kesempatan sebelum menyerah pada 1 frame,
# sekadar jaring pengaman tambahan -- perbaikan utama tetap di
# FrameGrabber (poin 1 di atas)
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "50000")

import cv2
import numpy as np

from insightface.app import FaceAnalysis
import supervision as sv

from ffmpeg_reader import FFmpegStreamReader


# =========================================
# PENGATURAN
# =========================================
RTMP_URL = "rtmp://103.255.15.138:1935/live/GSKeluarViewLuar.stream"
CAMERA_NAME = "GSKeluarViewLuar"
REGION = "Pintu Keluar"

DB_PATH = "faces_training.db"
THUMBNAIL_DIR = "face_thumbnails"

PROVIDERS = ["CPUExecutionProvider"]  # ganti ke ["CUDAExecutionProvider", "CPUExecutionProvider"] kalau pakai GPU
DET_SIZE = (800, 800)                 # dinaikkan dari 640 -> bantu deteksi wajah yang lebih kecil/jauh
                                       # (naikkan lagi ke (960,960) kalau CPU masih kuat & masih ada wajah terlewat)

SIMILARITY_THRESHOLD = 0.45           # ambang cosine similarity untuk anggap wajah sama
MIN_DET_SCORE = 0.5                   # diturunkan dari 0.6 -> lebih toleran, tapi tetap buang deteksi sangat lemah
MAX_EMBEDDINGS_PER_FACE = 5           # simpan beberapa "contoh wajah" per ID untuk matching yang lebih stabil
EMBEDDING_UPDATE_INTERVAL = 1.0       # detik, jangan ekstrak embedding tiap frame untuk track_id yang sama


# =========================================
# THREAD PEMBACA FRAME (FIX UTAMA: hilangkan backlog/slow-motion)
# =========================================
class FrameGrabber:
    """
    Baca frame dari stream (RTMP/HLS) di thread terpisah, TERUS-MENERUS.
    Menggunakan FFmpegStreamReader untuk dukungan penuh codec H.265/HEVC.
    """

    def __init__(self, url):
        self.url = url
        if isinstance(url, str) and (url.startswith("rtmp://") or url.startswith("http://") or url.endswith(".stream")):
            self.cap = FFmpegStreamReader(url, width=1920, height=1080)
        else:
            self.cap = cv2.VideoCapture(url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = self.cap.isOpened()
        self.lock = threading.Lock()
        self.fail_count = 0
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.fail_count += 1
                if self.fail_count > 60:
                    print("[GRABBER] Stream terputus terlalu lama, berhenti.")
                    self.running = False
                    break
                time.sleep(0.05)
                continue

            self.fail_count = 0
            with self.lock:
                self.frame = frame  # timpa frame lama -> tidak ada backlog

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# =========================================
# DATABASE
# =========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_ids (
            id TEXT PRIMARY KEY,
            embedding BLOB,
            first_seen TEXT,
            last_seen TEXT,
            times_seen INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            camera TEXT,
            region TEXT,
            timestamp TEXT,
            thumbnail_path TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


def load_known_faces():
    """
    Muat wajah yang sudah pernah tersimpan. Tiap face_id dimulai
    dengan 1 embedding dari database, nanti akan bertambah (sampai
    MAX_EMBEDDINGS_PER_FACE) selama program berjalan.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM face_ids")
    rows = cur.fetchall()
    conn.close()

    known = {}
    for face_id, emb_blob in rows:
        known[face_id] = [np.frombuffer(emb_blob, dtype=np.float32)]
    return known


def upsert_face_id(face_id, embedding, now_str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT times_seen FROM face_ids WHERE id = ?", (face_id,))
    row = cur.fetchone()

    if row is None:
        cur.execute("""
            INSERT INTO face_ids (id, embedding, first_seen, last_seen, times_seen)
            VALUES (?, ?, ?, ?, 1)
        """, (face_id, embedding.astype(np.float32).tobytes(), now_str, now_str))
    else:
        cur.execute("""
            UPDATE face_ids SET last_seen = ?, times_seen = times_seen + 1
            WHERE id = ?
        """, (now_str, face_id))

    conn.commit()
    conn.close()


def save_face_event(face_id, thumbnail_path, confidence, now_str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO face_events (face_id, camera, region, timestamp, thumbnail_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (face_id, CAMERA_NAME, REGION, now_str, thumbnail_path, float(confidence)))
    conn.commit()
    conn.close()


# =========================================
# PENCOCOKAN EMBEDDING -> ID PERMANEN
# =========================================
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_or_create_face_id(embedding, known_faces):
    """
    Bandingkan embedding baru terhadap SEMUA contoh embedding yang
    tersimpan untuk tiap face_id (bukan cuma 1), ambil similarity
    TERTINGGI di antara contoh-contoh itu. Ini membuat pencocokan
    lebih toleran terhadap variasi pose/sudut/pencahayaan orang yang
    sama, dibanding hanya membandingkan ke 1 embedding tunggal.
    """
    best_id = None
    best_score = -1.0

    for face_id, embedding_list in known_faces.items():
        score = max(cosine_similarity(embedding, e) for e in embedding_list)
        if score > best_score:
            best_score = score
            best_id = face_id

    if best_id is not None and best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_score, False  # False = bukan wajah baru

    new_id = uuid.uuid4().hex[:10]
    return new_id, 1.0, True  # True = wajah baru


def add_embedding_sample(known_faces, face_id, embedding):
    """Tambah contoh embedding baru untuk face_id ini, buang yang paling lama kalau sudah penuh."""
    samples = known_faces.setdefault(face_id, [])
    samples.append(embedding)
    if len(samples) > MAX_EMBEDDINGS_PER_FACE:
        samples.pop(0)


# =========================================
# MAIN PIPELINE
# =========================================
def main():
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    init_db()
    known_faces = load_known_faces()
    print(f"Memuat {len(known_faces)} wajah yang sudah pernah tersimpan sebelumnya.")

    print("Memuat model RetinaFace + ArcFace (InsightFace)...")
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print("Model siap.")

    # lost_track_buffer dinaikkan (default 30) -> track tidak langsung
    # hilang cuma karena deteksi gagal di 1-2 frame berturut-turut,
    # mengurangi kasus ID berubah-ubah untuk orang yang sama
    tracker = sv.ByteTrack(lost_track_buffer=90, frame_rate=30)

    print(f"Membuka stream: {RTMP_URL}")
    grabber = FrameGrabber(RTMP_URL)
    if not grabber.running:
        print("GAGAL membuka stream RTMP. Jalankan test_rtmp_connection.py dulu untuk debug.")
        return
    grabber.start()

    window_name = "Face Detection - Tekan 'q' untuk keluar"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # track_id (sementara, dari ByteTrack) -> {face_id, last_embed_time}
    track_state = {}

    print("Menunggu frame pertama dari stream...")
    while grabber.read() is None and grabber.running:
        time.sleep(0.1)

    try:
        while grabber.running:
            frame = grabber.read()
            if frame is None:
                time.sleep(0.02)
                continue

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")

            # ---- 1. Deteksi + embedding sekaligus ----
            faces = app.get(frame)
            faces = [f for f in faces if f.det_score >= MIN_DET_SCORE]

            # ---- 2. Tracking sementara (ByteTrack) ----
            if len(faces) > 0:
                boxes = np.array([f.bbox for f in faces], dtype=np.float32)
                confidences = np.array([f.det_score for f in faces], dtype=np.float32)
                class_ids = np.zeros(len(faces), dtype=int)

                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confidences,
                    class_id=class_ids,
                )
                tracked = tracker.update_with_detections(detections)
            else:
                tracked = sv.Detections.empty()

            # ---- 3. Untuk tiap wajah yang tertrack, cocokkan/buat ID permanen ----
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                matched_face = None
                for f in faces:
                    fb = f.bbox
                    if abs(fb[0] - x1) < 15 and abs(fb[1] - y1) < 15:
                        matched_face = f
                        break

                if matched_face is None:
                    continue

                state = track_state.get(track_id)
                need_new_embedding = (
                    state is None
                    or (time.time() - state["last_embed_time"]) >= EMBEDDING_UPDATE_INTERVAL
                )

                if need_new_embedding:
                    embedding = matched_face.embedding
                    face_id, score, is_new = match_or_create_face_id(embedding, known_faces)
                    add_embedding_sample(known_faces, face_id, embedding)

                    if is_new:
                        print(f"[WAJAH BARU] ID={face_id} (confidence deteksi={confidence:.2f})")
                    else:
                        print(f"[WAJAH DIKENALI] ID={face_id} (similarity={score:.2f})")

                    upsert_face_id(face_id, embedding, now_str)

                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    thumb_filename = f"{face_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                    thumb_path = os.path.join(THUMBNAIL_DIR, thumb_filename)
                    if crop.size > 0:
                        cv2.imwrite(thumb_path, crop)

                    save_face_event(face_id, thumb_filename, confidence, now_str)

                    track_state[track_id] = {
                        "face_id": face_id,
                        "last_embed_time": time.time(),
                    }
                else:
                    face_id = state["face_id"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{face_id[:6]} track:{track_id}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        grabber.stop()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    main()