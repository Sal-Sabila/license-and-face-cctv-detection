"""
test_bytetrack_optimization.py
==============================
Skrip benchmark untuk menguji dan membandingkan performa berbagai konfigurasi
parameter ByteTrack pada video rekaman CCTV (samples/deteksi.mp4).

Tujuan:
1. Menghitung total ID unik yang tercipta (ID switching minim = ID unik lebih sedikit & stabil).
2. Menghitung rata-rata durasi hidup track (panjang frame sebuah objek terlacak terus-menerus).
3. Mengukur efektivitas dalam menekan fragmentasi track pada kendaraan & orang.
"""

import os
import sys
import time
from collections import defaultdict
import cv2
import numpy as np
import torch

torch.set_num_threads(2)
cv2.setNumThreads(2)

from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "deteksi.mp4")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
MAX_FRAMES = 150  # ~6 detik footage CCTV padat untuk benchmark cepat & akurat

VEHICLE_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

CONFIGS = [
    {
        "name": "1. Baseline (Sekarang)",
        "track_activation_threshold": 0.35,
        "lost_track_buffer": 60,
        "minimum_matching_threshold": 0.70,
        "frame_rate": 25,
    },
    {
        "name": "2. Kandidat 1 (Persistensi Sedang)",
        "track_activation_threshold": 0.40,
        "lost_track_buffer": 120,
        "minimum_matching_threshold": 0.75,
        "frame_rate": 25,
    },
    {
        "name": "3. Kandidat 2 (Persistensi Tinggi)",
        "track_activation_threshold": 0.45,
        "lost_track_buffer": 150,
        "minimum_matching_threshold": 0.80,
        "frame_rate": 25,
    },
]


def extract_raw_detections(video_path, model_path, max_frames=MAX_FRAMES):
    """Jalankan YOLO sekali dan simpan hasil deteksi mentah per frame agar semua tracker diuji pada data yang 100% identik."""
    print(f"[1/3] Memuat model YOLO dari: {os.path.basename(model_path)}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_video_frames, max_frames) if total_video_frames > 0 else max_frames

    print(f"[2/3] Mengekstrak deteksi mentah dari {frames_to_process} frame video...")
    raw_detections_per_frame = []
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model(
            frame,
            classes=VEHICLE_CLASSES,
            imgsz=480,
            conf=0.30,
            verbose=False,
        )[0]

        dets = sv.Detections.from_ultralytics(results)
        raw_detections_per_frame.append(dets)
        frame_idx += 1

        if frame_idx % 75 == 0 or frame_idx == frames_to_process:
            print(f"      Frame {frame_idx}/{frames_to_process} ({time.time() - t0:.1f}s)")

    cap.release()
    print(f"      Selesai ekstraksi {len(raw_detections_per_frame)} frame dalam {time.time() - t0:.1f}s.\n")
    return raw_detections_per_frame


def evaluate_tracker(config, raw_detections):
    """Uji satu konfigurasi ByteTrack dengan data deteksi mentah."""
    tracker = sv.ByteTrack(
        track_activation_threshold=config["track_activation_threshold"],
        lost_track_buffer=config["lost_track_buffer"],
        minimum_matching_threshold=config["minimum_matching_threshold"],
        frame_rate=config["frame_rate"],
    )

    all_tracked_ids = set()
    track_frames_count = defaultdict(int)  # ID -> berapa frame objek terlihat
    active_per_frame = []

    for dets in raw_detections:
        tracked = tracker.update_with_detections(dets)
        active_ids = []
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for tid in tracked.tracker_id:
                tid = int(tid)
                all_tracked_ids.add(tid)
                track_frames_count[tid] += 1
                active_ids.append(tid)
        active_per_frame.append(len(active_ids))

    durations = list(track_frames_count.values())
    total_ids = len(all_tracked_ids)
    avg_duration = np.mean(durations) if durations else 0
    median_duration = np.median(durations) if durations else 0
    
    # Track pendek (< 10 frame = kemungkinan noise atau ID switch yang langsung hilang)
    short_tracks = sum(1 for d in durations if d < 10)
    stable_tracks = sum(1 for d in durations if d >= 25)

    return {
        "name": config["name"],
        "total_unique_ids": total_ids,
        "avg_duration_frames": avg_duration,
        "median_duration_frames": median_duration,
        "short_tracks": short_tracks,
        "stable_tracks": stable_tracks,
        "max_active_simultaneous": max(active_per_frame) if active_per_frame else 0,
    }


def main():
    print("=" * 70)
    print(" BENCHMARK OPTIMASI PARAMETER BYTETRACK ")
    print("=" * 70)

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] File video tidak ditemukan: {VIDEO_PATH}")
        sys.exit(1)

    raw_detections = extract_raw_detections(VIDEO_PATH, MODEL_PATH, MAX_FRAMES)
    if not raw_detections:
        print("[ERROR] Tidak ada deteksi yang diekstrak.")
        sys.exit(1)

    print("[3/3] Menjalankan evaluasi tracker pada setiap konfigurasi...")
    results = []
    for cfg in CONFIGS:
        print(f"      Menguji: {cfg['name']}...")
        res = evaluate_tracker(cfg, raw_detections)
        results.append(res)

    print("\n" + "=" * 80)
    print(" HASIL BENCHMARK OPTIMASI BYTETRACK")
    print("=" * 80)
    print(f"{'Konfigurasi':<32} | {'Total ID':<9} | {'Avg Frame':<10} | {'Short (<10)':<12} | {'Stabil (>=25)':<12}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['name']:<32} | "
            f"{r['total_unique_ids']:<9} | "
            f"{r['avg_duration_frames']:<10.1f} | "
            f"{r['short_tracks']:<12} | "
            f"{r['stable_tracks']:<12}"
        )

    print("=" * 80)
    print("\nAnalisis Kesimpulan:")
    baseline_ids = results[0]["total_unique_ids"]
    best_cfg = min(results, key=lambda x: x["total_unique_ids"])
    print(f"- Total ID unik Baseline         : {baseline_ids} ID")
    print(f"- Total ID unik Terbaik           : {best_cfg['total_unique_ids']} ID ({best_cfg['name']})")
    reduction = ((baseline_ids - best_cfg['total_unique_ids']) / baseline_ids) * 100 if baseline_ids > 0 else 0
    print(f"- Reduksi ID Switch / Duplikasi   : {reduction:.1f}%")
    print(f"- Rata-rata durasi objek terlacak : {best_cfg['avg_duration_frames']:.1f} frame")


if __name__ == "__main__":
    main()
