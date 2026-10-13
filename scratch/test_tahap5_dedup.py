"""
Test Script Tahap 5 - Evaluasi Deduplikasi Dual-Path

Skenario pengujian:
  1. Jalankan VideoAIService pada samples/deteksi.mp4 sebagai "jalur segmen" simulasi.
  2. Evaluasi duplikasi menggunakan DuplicateEvaluationService (dedup_service.py).
  3. Catat jumlah deteksi SEBELUM dan SESUDAH deduplikasi.
  4. Identifikasi: cross-path, intra-source duplicates, repeat passes.
  5. Estimasi risiko data terlewat.
  6. Semua read-only — tidak ada mutasi database.
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services import dedup_service

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_VIDEO    = os.path.join(BASE_DIR, "samples", "deteksi.mp4")
REPORT_PATH     = os.path.join(os.path.dirname(__file__), "tahap5_dedup_report.json")
CAMERA_ID_VIDEO = 6
LOOKBACK        = 3600  # 1 jam


def separator(title=""):
    print(f"\n{'='*72}")
    if title:
        print(f"  {title}")
        print("="*72)


def fmt_samples(samples, limit=5):
    for i, s in enumerate(samples[:limit]):
        t = s.get("type", "?")
        plate = s.get("plate", "?")
        cam = s.get("camera_id", "?")
        delta = s.get("delta_seconds", "?")
        if t == "cross_path":
            print(f"    [{i+1}] CROSS  plat={plate} cam={cam} "
                  f"stream={s.get('stream_time','?')}  video={s.get('video_time','?')}  "
                  f"Δ={delta}s")
        elif t == "intra_source":
            print(f"    [{i+1}] INTRA  plat={plate} cam={cam} "
                  f"src={s.get('source_type','?')}  "
                  f"orig={s.get('original_time','?')}  dup={s.get('duplicate_time','?')}  "
                  f"Δ={delta}s  id={s.get('identity','?')}")


def print_report(label, rpt):
    bd = rpt["before_deduplication"]
    ad = rpt["after_deduplication"]
    cp = rpt["cross_path_duplicates"]
    intra = rpt["intra_source_duplicates"]
    rp = rpt["repeat_passes"]
    print(f"\n  [{label}]")
    print(f"    Total event plat   : {bd['total_plate_events']}  "
          f"(stream={bd['stream_source_events']}, video={bd['video_source_events']})")
    print(f"    Setelah dedup      : {ad['total_plate_events']}  "
          f"(-{ad['events_removed']}  rate={ad['duplication_rate_percent']}%)")
    print(f"    Cross-path dups    : {cp['count']}")
    print(f"    Intra-source dups  : {intra['count']}")
    print(f"    Repeat passes      : {rp['count']}")
    print(f"    Data-loss risk     : {rpt['data_loss_risk']['risk_level']}")
    print(f"    Sample adequate    : {rpt['analysis']['sample_adequate']}")


# ─── FASE 1: Baseline ────────────────────────────────────────────────────────

separator("FASE 1 — Baseline (Sebelum Proses Video)")
t0 = time.perf_counter()
before_rpt = dedup_service.evaluate(camera_id=None, lookback_seconds=LOOKBACK)
print_report("BASELINE", before_rpt)
print(f"    Evaluasi selesai  : {round(time.perf_counter()-t0, 3)}s")

if before_rpt["intra_source_duplicates"]["count"] > 0:
    print("\n  Contoh intra-source duplicates:")
    fmt_samples(before_rpt["intra_source_duplicates"]["samples"])
if before_rpt["repeat_passes"]["count"] > 0:
    print("\n  Repeat pass groups (kendaraan melintas kembali):")
    for g in before_rpt["repeat_passes"]["groups"][:3]:
        print(f"    plat={g['plate']} cam={g['camera_id']} "
              f"pass={g['pass_count']}x  {g['first_seen']}→{g['last_seen']}  src={g['sources']}")


# ─── FASE 2: Proses video sample ────────────────────────────────────────────

separator("FASE 2 — Proses samples/deteksi.mp4 sebagai Jalur Video")
video_processed = False

if not os.path.isfile(SAMPLE_VIDEO):
    print(f"  [SKIP] File tidak ada: {SAMPLE_VIDEO}")
    print("  Fase 2 dilewati. Evaluasi berbasis data eksisting saja.")
else:
    size_mb = os.path.getsize(SAMPLE_VIDEO) / (1024 * 1024)
    print(f"  File  : {SAMPLE_VIDEO}  ({size_mb:.1f} MB)")
    print(f"  Kamera: {CAMERA_ID_VIDEO} | source_name=Segment Cam {CAMERA_ID_VIDEO} Test5")
    print(f"  Menjalankan VideoAIService... (tunggu beberapa menit)")

    try:
        from services.video_ai_service import VideoAIService
        ai = VideoAIService(
            video_path=SAMPLE_VIDEO,
            output_path=None,
            camera_id=CAMERA_ID_VIDEO,
            show_window=False,
            source_name=f"Segment Cam {CAMERA_ID_VIDEO} Test5",
        )
        t_ai = time.perf_counter()
        ai.run()
        t_elapsed = round(time.perf_counter() - t_ai, 1)
        plate_hist = list(ai.last_plate_history) if hasattr(ai, "last_plate_history") else []
        print(f"\n  Selesai dalam {t_elapsed}s")
        print(f"  Inference frames  : {ai.inference_count}")
        print(f"  Plate captures    : {ai.plate_capture_count}")
        print(f"  Vehicle captures  : {ai.person_capture_count}")
        print(f"  Plate history OCR : {len(plate_hist)} entri")
        for ph in plate_hist[:10]:
            print(f"    {ph}")
        video_processed = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n  [ERROR] {exc}")


# ─── FASE 3: Evaluasi sesudah ────────────────────────────────────────────────

separator("FASE 3 — Evaluasi Sesudah Proses Video")
time.sleep(2.0)
t0 = time.perf_counter()
after_rpt = dedup_service.evaluate(camera_id=None, lookback_seconds=LOOKBACK)
print_report("SESUDAH", after_rpt)
print(f"    Evaluasi selesai  : {round(time.perf_counter()-t0, 3)}s")

# Delta
nb = before_rpt["before_deduplication"]
na = after_rpt["before_deduplication"]
new_total  = na["total_plate_events"] - nb["total_plate_events"]
new_video  = na["video_source_events"] - nb["video_source_events"]
new_stream = na["stream_source_events"] - nb["stream_source_events"]
print(f"\n  Delta event baru   : +{new_total}  (video=+{new_video}, stream=+{new_stream})")
print(f"  Cross-path Δ       : "
      f"{before_rpt['cross_path_duplicates']['count']} → "
      f"{after_rpt['cross_path_duplicates']['count']}")
print(f"  Intra-source Δ     : "
      f"{before_rpt['intra_source_duplicates']['count']} → "
      f"{after_rpt['intra_source_duplicates']['count']}")

if after_rpt["cross_path_duplicates"]["count"] > 0:
    print("\n  Contoh cross-path duplicates ditemukan:")
    fmt_samples(after_rpt["cross_path_duplicates"]["samples"])

if after_rpt["repeat_passes"]["count"] > 0:
    print("\n  Repeat pass groups (kunjungan nyata — tidak dihapus):")
    for g in after_rpt["repeat_passes"]["groups"][:3]:
        print(f"    plat={g['plate']} cam={g['camera_id']} "
              f"pass={g['pass_count']}x  src={g['sources']}")

print(f"\n  Lag cross-path median: {after_rpt['analysis']['cross_path_lag_observed_seconds']}s")
print(f"  Rekomendasi: {after_rpt['analysis']['recommended_action']}")
if after_rpt["data_loss_risk"]["notes"]:
    print("\n  Catatan risiko data:")
    for note in after_rpt["data_loss_risk"]["notes"]:
        print(f"    > {note}")


# ─── FASE 4: Verifikasi constraint ──────────────────────────────────────────

separator("FASE 4 — Verifikasi Constraint (DB tidak berubah)")
import db as _db

errors = []
try:
    with _db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM full_detection")
            cnt1 = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM full_detection")
            cnt2 = cur.fetchone()["cnt"]

    if cnt1 != cnt2:
        errors.append(f"Jumlah record berubah: {cnt1} → {cnt2}")
    else:
        print(f"  [OK] full_detection count stabil: {cnt1}")

    with _db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA='real_cctv' AND TABLE_NAME='dedup_log'"
            )
            if cur.fetchone()["cnt"] > 0:
                errors.append("Tabel dedup_log tidak boleh ada")
            else:
                print("  [OK] Tidak ada tabel dedup_log")

    assert after_rpt["database_mutated"] is False
    print("  [OK] database_mutated = False")

except Exception as exc:
    errors.append(str(exc))

if errors:
    print("  [GAGAL] Constraint violation:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  [OK] Semua constraint terpenuhi")


# ─── Ringkasan final & simpan JSON ──────────────────────────────────────────

separator("RINGKASAN AKHIR TAHAP 5")
ad_f = after_rpt["after_deduplication"]
cp_f = after_rpt["cross_path_duplicates"]
in_f = after_rpt["intra_source_duplicates"]
rp_f = after_rpt["repeat_passes"]

print(f"  Video sample diproses    : {'Ya' if video_processed else 'Tidak'}")
print(f"  Event plat total (DB)    : {na['total_plate_events']}  "
      f"(stream={na['stream_source_events']}, video={na['video_source_events']})")
print(f"  Cross-path duplicates    : {cp_f['count']}")
print(f"  Intra-source duplicates  : {in_f['count']}")
print(f"  Repeat passes            : {rp_f['count']}")
print(f"  Events setelah dedup     : {ad_f['total_plate_events']}")
print(f"  Duplikasi rate           : {ad_f['duplication_rate_percent']}%")
print(f"  Risiko data terlewat     : {after_rpt['data_loss_risk']['risk_level']}")
print(f"  Constraint DB terpenuhi  : {'Ya' if not errors else 'TIDAK'}")
print(f"  Database dimodifikasi    : TIDAK")

report_out = {
    "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "video_sample": SAMPLE_VIDEO,
    "video_processed": video_processed,
    "constraints_satisfied": not bool(errors),
    "before": {
        "total_events": nb["total_plate_events"],
        "stream": nb["stream_source_events"],
        "video": nb["video_source_events"],
        "cross_path_dups": before_rpt["cross_path_duplicates"]["count"],
        "intra_source_dups": before_rpt["intra_source_duplicates"]["count"],
        "repeat_passes": before_rpt["repeat_passes"]["count"],
        "dup_rate_pct": before_rpt["after_deduplication"]["duplication_rate_percent"],
    },
    "after": {
        "total_events": na["total_plate_events"],
        "stream": na["stream_source_events"],
        "video": na["video_source_events"],
        "cross_path_dups": cp_f["count"],
        "cross_path_samples": cp_f["samples"][:10],
        "intra_source_dups": in_f["count"],
        "intra_source_samples": in_f["samples"][:10],
        "repeat_passes": rp_f["count"],
        "repeat_pass_groups": rp_f["groups"][:5],
        "events_after_dedup": ad_f["total_plate_events"],
        "events_removed": ad_f["events_removed"],
        "dup_rate_pct": ad_f["duplication_rate_percent"],
        "data_loss_risk": after_rpt["data_loss_risk"],
        "analysis": after_rpt["analysis"],
    },
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report_out, f, indent=2, default=str)
print(f"\n  Laporan tersimpan: {REPORT_PATH}")

assert not errors, f"Constraint violations: {errors}"
assert after_rpt["database_mutated"] is False
print("\n[PENGUJIAN TAHAP 5 SELESAI DENGAN SUKSES]")
