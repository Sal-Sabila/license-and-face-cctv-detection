import os
import sys
import time
import json
import psutil
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager
from services.stream_ai_service import StreamAIService
from app import app
import db

def run_tahap4_5minute_test():
    print("=" * 80)
    print("PENGUJIAN TAHAP 4: INTEGRASI DUAL-PATH REALTIME + HYBRID RECORDER (MINIMAL 5 MENIT)")
    print("=" * 80)

    test_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_tahap4_segments")
    os.makedirs(test_dir, exist_ok=True)

    # Pastikan target kamera adalah Kamera 6 (FM Noto McDonalds) yang aktif dan online
    target_camera_id = 6
    cameras = db.get_all_cameras()
    cam6 = next((c for c in cameras if c["camera_id"] == target_camera_id), None)
    assert cam6 is not None, f"Kamera {target_camera_id} tidak ditemukan di database"

    print(f"\n[INFO TARGET CCTV]")
    print(f"  Camera ID   : {cam6['camera_id']}")
    print(f"  Location    : {cam6['location']}")
    print(f"  Stream URL  : {cam6['stream_url']}")
    print(f"  Test Dir    : {test_dir}")

    # 1. Jalankan Jalur A: Realtime Detection (CameraWorkerManager + StreamAIService)
    print("\n[1] MEMULAI JALUR REALTIME (CameraWorkerManager & StreamAIService)...")
    worker_mgr = CameraWorkerManager.get_instance()
    worker_mgr.start()
    worker_mgr.sync_now()

    # Tunggu sebentar hingga worker kamera 6 terinisialisasi
    time.sleep(2.0)
    cam6_worker = worker_mgr.get_worker(target_camera_id)
    print(f"    Worker kamera {target_camera_id} status: {cam6_worker.get_status() if cam6_worker else 'Belum aktif'}")

    # 2. Jalankan Jalur B: Hybrid Recorder Service (Segmenter 10s + Single Worker)
    print("\n[2] MEMULAI JALUR HYBRID RECORDER (CameraRecorderService)...")
    os.environ["RECORD_CAMERA_ID"] = str(target_camera_id)
    os.environ["HYBRID_MAX_QUEUE_SIZE"] = "15"
    recorder = CameraRecorderService.get_instance()
    recorder.stop()
    time.sleep(1.0)
    recorder.camera_id = None
    recorder.stream_url = None
    recorder.segments_dir = test_dir
    recorder.segment_duration = 10
    recorder.max_queue_size = 15
    recorder.auto_process_enabled = True

    # Bersihkan file CSV index atau state lama dari test_dir agar pengujian 5 menit bersih
    csv_index = os.path.join(test_dir, "segments_index.csv")
    state_file = os.path.join(test_dir, "segment_queue_state.json")
    if os.path.exists(csv_index):
        try: os.remove(csv_index)
        except Exception: pass
    if os.path.exists(state_file):
        try: os.remove(state_file)
        except Exception: pass

    recorder.start(camera_id=target_camera_id)
    print("    Recorder & Single AI Worker dimulai.")

    # 3. Jalankan Pengujian Selama Minimal 5 Menit (300 detik)
    total_test_duration = 310  # 310 detik (5 menit 10 detik untuk memastikan >= 5 menit)
    sample_interval = 10       # Log metrik setiap 10 detik

    print(f"\n[3] MEMULAI PENGUKURAN DUAL-PATH SELAMA {total_test_duration} DETIK ({total_test_duration/60:.1f} MENIT)...")
    print(f"{'Waktu':<8} | {'SysCPU':<7} | {'ProcCPU':<8} | {'RAM(MB)':<8} | {'RT Status':<9} | {'RT Frame':<9} | {'Recorded':<8} | {'Pend':<5} | {'Proc':<5} | {'Comp':<5} | {'Fail':<5} | {'Stored':<6}")
    print("-" * 115)

    metrics_history = []
    start_time = time.time()
    last_frame_hashes = []
    realtime_stalls = 0
    process_obj = psutil.Process()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_test_duration:
            break

        time.sleep(sample_interval)
        current_elapsed = int(time.time() - start_time)

        # Ukur CPU & Memori
        sys_cpu = psutil.cpu_percent(interval=None)
        proc_cpu = process_obj.cpu_percent(interval=None)
        mem_mb = round(process_obj.memory_info().rss / (1024 * 1024), 1)

        # Periksa Kelancaran Jalur Realtime
        rt_status = "UNKNOWN"
        rt_frame_ok = False
        cam6_worker = worker_mgr.get_worker(target_camera_id)
        if cam6_worker:
            st = cam6_worker.get_status()
            rt_status = st.get("status", "UNKNOWN")
            frame = cam6_worker.get_last_frame()
            if frame is not None and getattr(frame, "size", 0) > 0:
                rt_frame_ok = True

        if not rt_frame_ok or rt_status != "ONLINE":
            realtime_stalls += 1

        # Periksa Status & Backlog Jalur Rekaman / Antrean
        rec_status = recorder.get_status()
        b = rec_status.get("backlog", {})
        total_rec = b.get("total_recorded", 0)
        pend = b.get("pending", 0)
        proc = b.get("processing", 0)
        comp = b.get("completed", 0)
        fail = b.get("failed", 0)
        stored = b.get("stored", 0)

        # Simpan metrik
        metrics_history.append({
            "elapsed_seconds": current_elapsed,
            "sys_cpu": sys_cpu,
            "proc_cpu": proc_cpu,
            "mem_mb": mem_mb,
            "realtime_status": rt_status,
            "realtime_frame_ok": rt_frame_ok,
            "total_recorded": total_rec,
            "pending": pend,
            "processing": proc,
            "completed": comp,
            "failed": fail,
            "stored": stored,
            "worker_busy": b.get("worker_is_busy", False),
        })

        # Cetak baris pemantauan
        print(
            f"{current_elapsed:03d}s    | "
            f"{sys_cpu:5.1f}% | "
            f"{proc_cpu:6.1f}% | "
            f"{mem_mb:7.1f} | "
            f"{rt_status:<9} | "
            f"{'OK' if rt_frame_ok else 'NO':<9} | "
            f"{total_rec:<8} | "
            f"{pend:<5} | "
            f"{proc:<5} | "
            f"{comp:<5} | "
            f"{fail:<5} | "
            f"{stored:<6}"
        )

    # 4. Evaluasi Potensi Deteksi Ganda
    print("\n[4] EVALUASI POTENSI DETEKSI GANDA ANTARA KEDUA JALUR...")
    dup_eval = recorder.evaluate_duplicates(time_window_seconds=300)
    print(f"  Sampel Deteksi Realtime : {dup_eval['realtime_detections_sampled']}")
    print(f"  Sampel Deteksi Segmen   : {dup_eval['segment_detections_sampled']}")
    print(f"  Kandidat Duplikasi      : {dup_eval['duplicate_candidates_count']}")
    print(f"  Tingkat Duplikasi       : {dup_eval['duplication_rate_percent']}%")
    if dup_eval["duplicate_candidates"]:
        print("  Contoh Pasangan Duplikasi Terdeteksi:")
        for idx, dc in enumerate(dup_eval["duplicate_candidates"][:3]):
            print(f"    [{idx+1}] Plat: {dc['plate']} | Realtime ID={dc['realtime_id']} ({dc['realtime_time']}) vs Segmen={dc['segment_file']} ({dc['segment_time']})")

    # 5. Hentikan Layanan Uji
    print("\n[5] MENGHENTIKAN LAYANAN UJI...")
    recorder.stop()
    worker_mgr.stop()

    # 6. Kalkulasi Statistik & Ringkasan Laporan
    avg_sys_cpu = round(sum(m["sys_cpu"] for m in metrics_history) / len(metrics_history), 1) if metrics_history else 0
    max_sys_cpu = round(max(m["sys_cpu"] for m in metrics_history), 1) if metrics_history else 0
    avg_proc_cpu = round(sum(m["proc_cpu"] for m in metrics_history) / len(metrics_history), 1) if metrics_history else 0
    final_metric = metrics_history[-1] if metrics_history else {}

    print("\n" + "=" * 80)
    print("RINGKASAN HASIL PENGUJIAN TAHAP 4 (DURASI >= 5 MENIT)")
    print("=" * 80)
    print(f"Total Durasi Pengujian      : {int(elapsed)} detik ({elapsed/60:.2f} menit)")
    print(f"Rata-rata System CPU        : {avg_sys_cpu}% (Puncak: {max_sys_cpu}%)")
    print(f"Rata-rata Process CPU       : {avg_proc_cpu}%")
    print(f"Penggunaan Memori Terakhir  : {final_metric.get('mem_mb', 0)} MB")
    print(f"Kelancaran Realtime         : {'LANCAR (Tanpa Stall)' if realtime_stalls == 0 else f'{realtime_stalls} kali frame drop/offline'}")
    print(f"Total Segmen Direkam        : {final_metric.get('total_recorded', 0)} segmen 10 detik")
    print(f"Segmen Sukses Diproses AI   : {final_metric.get('completed', 0)} segmen")
    print(f"Segmen Gagal (Failed)       : {final_metric.get('failed', 0)} segmen")
    print(f"Antrean Pending             : {final_metric.get('pending', 0)} segmen")
    print(f"Antrean Stored (Aman Disk)  : {final_metric.get('stored', 0)} segmen")
    print(f"Potensi Duplikasi Plat      : {dup_eval['duplicate_candidates_count']} pasang ({dup_eval['duplication_rate_percent']}%)")
    print("=" * 80)

    # Simpan laporan data JSON untuk dokumentasi
    report_file = os.path.join(test_dir, "tahap4_test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": int(elapsed),
            "camera_id": target_camera_id,
            "avg_sys_cpu": avg_sys_cpu,
            "max_sys_cpu": max_sys_cpu,
            "avg_proc_cpu": avg_proc_cpu,
            "final_metrics": final_metric,
            "realtime_stalls": realtime_stalls,
            "duplicate_evaluation": dup_eval,
            "history": metrics_history,
        }, f, indent=2)
    print(f"Laporan metrik lengkap tersimpan di: {report_file}")

    assert elapsed >= 300, "Pengujian harus berjalan minimal 5 menit (300 detik)!"
    assert final_metric.get("total_recorded", 0) >= 15, "Perekaman harus menghasilkan minimal 15 segmen 10 detik dalam 5 menit!"
    print("\n[PENGUJIAN TAHAP 4 SELESAI DENGAN SUKSES]")
    return True

if __name__ == "__main__":
    run_tahap4_5minute_test()
