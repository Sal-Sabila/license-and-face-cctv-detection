"""
DEDUP SERVICE — Tahap 5 Sistem Hybrid CCTV Flask

Engine evaluasi dan deteksi duplikasi antarpipeline (realtime stream vs segmen video).

Prinsip:
  1. READ-ONLY terhadap database real_cctv (tidak ada INSERT/UPDATE/DELETE).
  2. Tidak memodifikasi pipeline YOLO, ByteTrack, OCR, capture, atau StreamAIService.
  3. Membedakan tiga jenis deteksi:
       a. Cross-path duplicate  : Plat yang sama muncul di jalur stream DAN jalur video
                                  dalam jendela waktu DEDUP_CROSS_WINDOW_SECONDS.
       b. Intra-source duplicate: Plat yang sama, kamera yang sama, sumber yang sama,
                                  dalam waktu DEDUP_INTRA_WINDOW_SECONDS
                                  (kendaraan bergerak pelan / dua frame tracking berbeda).
       c. Repeat pass           : Plat yang sama, kamera yang sama, jeda > REPEAT_PASS_GAP_SECONDS
                                  → kendaraan benar-benar melintas kembali, BUKAN duplikat.
  4. Menghasilkan laporan sebelum/sesudah deduplikasi tanpa mengubah data.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional

import db


# ─── KONSTANTA ────────────────────────────────────────────────────────────────

# Jendela waktu maksimum delta antara deteksi stream dan segment agar dianggap duplikasi
DEDUP_CROSS_WINDOW_SECONDS = 90

# Jendela intra-source: plat yang sama, sumber yang sama, dalam window ini → duplikat
DEDUP_INTRA_WINDOW_SECONDS = 45

# Jika Δt ≥ nilai ini, dianggap lintasan kembali (repeat pass), bukan duplikat
REPEAT_PASS_GAP_SECONDS = 300

# Maksimum baris yang di-query dari DB per evaluasi
MAX_QUERY_ROWS = 2000

# Jendela waktu ke belakang untuk evaluasi (default 10 menit)
DEFAULT_LOOKBACK_SECONDS = 600


# ─── NORMALISASI ──────────────────────────────────────────────────────────────

def _normalize_plate(value) -> str:
    """Hapus semua karakter non-alfanumerik dan ubah ke uppercase."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _parse_ts(value) -> Optional[datetime]:
    """Parse timestamp ke datetime naive (tanpa timezone)."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


# ─── QUERY DATABASE ───────────────────────────────────────────────────────────

def _fetch_plate_events(camera_id=None, lookback_seconds=DEFAULT_LOOKBACK_SECONDS):
    """
    Query semua deteksi berplat valid dari DB dalam jendela waktu terakhir.
    Mengembalikan list dict dengan field:
      detection_id, camera_id, plate, timestamp, source_type, source_name,
      track_id, event_key, ocr_confidence, vehicle_type
    """
    events = []
    lookback = max(30, int(lookback_seconds))
    try:
        with db.get_db() as conn:
            with conn.cursor() as cur:
                params = [lookback]
                cam_clause = ""
                if camera_id is not None:
                    cam_clause = "AND fd.camera_id = %s"
                    params.append(int(camera_id))

                cur.execute(f"""
                    SELECT
                        fd.detection_id,
                        fd.camera_id,
                        fd.source_type,
                        fd.source_name,
                        fd.track_id,
                        fd.event_key,
                        fd.created_at,
                        fd.vehicle_type,
                        p.plate_number,
                        p.normalized_plate_number,
                        p.ocr_confidence
                    FROM full_detection fd
                    JOIN plate p ON p.plate_id = fd.plate_id
                    WHERE fd.created_at >= DATE_SUB(NOW(), INTERVAL %s SECOND)
                      AND p.plate_number IS NOT NULL
                      AND p.plate_number != '-'
                      {cam_clause}
                    ORDER BY fd.created_at ASC
                    LIMIT {MAX_QUERY_ROWS}
                """, params)

                for r in cur.fetchall():
                    plate_raw = r.get("normalized_plate_number") or r.get("plate_number", "")
                    plate = _normalize_plate(plate_raw)
                    if not plate or len(plate) < 3:
                        continue
                    ts = _parse_ts(r.get("created_at"))
                    events.append({
                        "detection_id": r["detection_id"],
                        "camera_id": str(r["camera_id"]),
                        "plate": plate,
                        "plate_raw": r.get("plate_number", ""),
                        "timestamp": ts,
                        "source_type": r.get("source_type", "stream"),
                        "source_name": r.get("source_name"),
                        "track_id": r.get("track_id"),
                        "event_key": r.get("event_key"),
                        "ocr_confidence": float(r.get("ocr_confidence") or 0.0),
                        "vehicle_type": r.get("vehicle_type", "unknown"),
                    })
    except Exception as exc:
        print(f"[DEDUP SERVICE] DB query error: {exc}")

    return events


# ─── ENGINE DEDUPLIKASI ───────────────────────────────────────────────────────

def _classify_events(events):
    """Pisahkan events berdasarkan source_type."""
    stream_events = [e for e in events if e["source_type"] == "stream"]
    video_events = [e for e in events if e["source_type"] == "video"]
    return stream_events, video_events


def _find_intra_source_duplicates(events, window_seconds):
    """
    Deduplikasi dalam satu jalur — dua tier:

    Tier 1 – Identity-match (event_key/track_id SAMA, dalam window_seconds):
        Deteksi frame lanjutan dari kendaraan yang sama dalam satu sesi tracking.

    Tier 2 – Plate-only match (plat SAMA, beda identity, dalam PLATE_ONLY_WINDOW detik):
        Menangkap kasus di mana satu kendaraan memicu dua tracking group berbeda
        (vehicle_group_NNN) dalam beberapa detik — biasanya karena re-ID tracker.
        Hanya ditandai duplikat jika Δt sangat pendek (≤ PLATE_ONLY_WINDOW detik).

    Repeat-pass (Δt > REPEAT_PASS_GAP_SECONDS) tidak diklasifikasikan sebagai duplikat.
    Mengembalikan (compacted_events, duplicate_records).
    """
    PLATE_ONLY_WINDOW = 10.0   # detik — pendek agar tidak salah hapus kendaraan berbeda

    compacted = []
    duplicates = []

    sorted_events = sorted(events, key=lambda e: (e["timestamp"] or datetime.min))

    # Tier 1: (camera_id, plate, identity) → event terakhir
    last_by_identity = {}
    # Tier 2: (camera_id, plate) → event terakhir (untuk plate-only match)
    last_by_plate = {}

    for event in sorted_events:
        plate = event["plate"]
        camera_id = event["camera_id"]
        ts = event["timestamp"]
        event_key = event.get("event_key")
        track_id = event.get("track_id")

        if ts is None:
            compacted.append(event)
            continue

        # Bangun identity key
        identity = None
        if event_key:
            # Hapus suffix generasi agar sesi yang sama tetap satu identity
            identity = re.sub(r":(?:gen|g)\d+", "", str(event_key)).strip()
        elif track_id is not None:
            identity = f"track:{track_id}"

        # ─── Tier 1: identity-based check ──────────────────────────────────
        if identity is not None:
            id_key = (camera_id, plate, str(identity))
            prev_id = last_by_identity.get(id_key)
            if prev_id is not None and prev_id["timestamp"] is not None:
                delta = (ts - prev_id["timestamp"]).total_seconds()
                if 0 < delta <= window_seconds:
                    duplicates.append({
                        "type": "intra_source",
                        "subtype": "same_identity",
                        "plate": plate,
                        "camera_id": camera_id,
                        "source_type": event["source_type"],
                        "original_id": prev_id["detection_id"],
                        "duplicate_id": event["detection_id"],
                        "original_time": prev_id["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "duplicate_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "delta_seconds": round(delta, 2),
                        "identity": str(identity),
                    })
                    continue
            last_by_identity[id_key] = event

        # ─── Tier 2: plate-only check (tight window ≤ PLATE_ONLY_WINDOW) ──
        plate_key = (camera_id, plate)
        prev_plate = last_by_plate.get(plate_key)
        if prev_plate is not None and prev_plate["timestamp"] is not None:
            delta = (ts - prev_plate["timestamp"]).total_seconds()
            if 0 < delta <= PLATE_ONLY_WINDOW:
                duplicates.append({
                    "type": "intra_source",
                    "subtype": "plate_only_tight_window",
                    "plate": plate,
                    "camera_id": camera_id,
                    "source_type": event["source_type"],
                    "original_id": prev_plate["detection_id"],
                    "duplicate_id": event["detection_id"],
                    "original_time": prev_plate["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "duplicate_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "delta_seconds": round(delta, 2),
                    "identity": (
                        f"{prev_plate.get('event_key','?')} vs "
                        f"{event_key or '?'}"
                    ),
                })
                continue

        compacted.append(event)
        last_by_plate[plate_key] = event

    return compacted, duplicates



def _find_cross_path_duplicates(stream_events, video_events, window_seconds):
    """
    Temukan duplikasi lintas jalur: plat yang sama, kamera yang sama,
    |Δt| ≤ window_seconds, antara stream dan video.

    Strategi: one-to-one matching, menjaga repeat_pass sebagai event terpisah.
    """
    cross_duplicates = []

    # Kelompokkan per (camera_id, plate)
    group_keys = set(
        (e["camera_id"], e["plate"])
        for events_list in (stream_events, video_events)
        for e in events_list
        if e["timestamp"] is not None
    )

    for camera_id, plate in sorted(group_keys):
        stream_group = sorted(
            [e for e in stream_events if e["camera_id"] == camera_id and e["plate"] == plate and e["timestamp"] is not None],
            key=lambda e: e["timestamp"]
        )
        video_group = sorted(
            [e for e in video_events if e["camera_id"] == camera_id and e["plate"] == plate and e["timestamp"] is not None],
            key=lambda e: e["timestamp"]
        )

        if not stream_group or not video_group:
            continue

        si, vi = 0, 0
        while si < len(stream_group) and vi < len(video_group):
            s_ev = stream_group[si]
            v_ev = video_group[vi]
            delta = (v_ev["timestamp"] - s_ev["timestamp"]).total_seconds()

            if delta < -window_seconds:
                # video jauh lebih awal dari stream → naikkan vi
                vi += 1
                continue
            if delta > window_seconds:
                # stream jauh lebih awal dari video → naikkan si
                si += 1
                continue

            # Pasangan duplikasi ditemukan
            cross_duplicates.append({
                "type": "cross_path",
                "plate": plate,
                "camera_id": camera_id,
                "stream_id": s_ev["detection_id"],
                "video_id": v_ev["detection_id"],
                "stream_time": s_ev["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "video_time": v_ev["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "delta_seconds": round(abs(delta), 2),
                "stream_track_id": s_ev.get("track_id"),
                "video_track_id": v_ev.get("track_id"),
                "stream_source_name": s_ev.get("source_name"),
                "video_source_name": v_ev.get("source_name"),
                "ocr_confidence_stream": s_ev.get("ocr_confidence", 0.0),
                "ocr_confidence_video": v_ev.get("ocr_confidence", 0.0),
                "merge_policy": "keep_stream_for_latency_drop_video_as_dup",
            })
            si += 1
            vi += 1  # one-to-one matching

    return cross_duplicates


def _find_repeat_passes(events, gap_seconds):
    """
    Identifikasi kendaraan berplat sama yang melintas kembali (Δt ≥ gap_seconds).
    Ini bukan duplikat — ini kunjungan ulang nyata.
    Mengembalikan list kelompok per (camera_id, plate) dengan timestamps setiap lintasan.
    """
    repeat_groups = []
    group_keys = set(
        (e["camera_id"], e["plate"])
        for e in events
        if e["timestamp"] is not None
    )

    for camera_id, plate in sorted(group_keys):
        group = sorted(
            [e for e in events if e["camera_id"] == camera_id and e["plate"] == plate and e["timestamp"] is not None],
            key=lambda e: e["timestamp"]
        )
        if len(group) < 2:
            continue

        passes = [group[0]]
        for ev in group[1:]:
            delta = (ev["timestamp"] - passes[-1]["timestamp"]).total_seconds()
            if delta >= gap_seconds:
                passes.append(ev)

        if len(passes) >= 2:
            repeat_groups.append({
                "plate": plate,
                "camera_id": camera_id,
                "pass_count": len(passes),
                "first_seen": passes[0]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": passes[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "pass_timestamps": [
                    p["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for p in passes
                ],
                "sources": list(set(p["source_type"] for p in passes)),
            })

    return repeat_groups


# ─── API PUBLIK ───────────────────────────────────────────────────────────────

def evaluate(camera_id=None, lookback_seconds=DEFAULT_LOOKBACK_SECONDS):
    """
    Evaluasi duplikasi antarpipeline untuk periode waktu tertentu.

    Args:
        camera_id: Filter ke satu kamera, atau None untuk semua.
        lookback_seconds: Jendela waktu lookback ke belakang dari sekarang.

    Returns:
        dict laporan lengkap (read-only, tidak ada mutasi DB).
    """
    t_start = time.perf_counter()

    # 1. Ambil semua event berplat dari DB
    all_events = _fetch_plate_events(camera_id=camera_id, lookback_seconds=lookback_seconds)

    stream_events_raw, video_events_raw = _classify_events(all_events)

    # 2. Deduplikasi intra-source (dalam jalur yang sama)
    stream_compacted, stream_intra_dups = _find_intra_source_duplicates(
        stream_events_raw, DEDUP_INTRA_WINDOW_SECONDS
    )
    video_compacted, video_intra_dups = _find_intra_source_duplicates(
        video_events_raw, DEDUP_INTRA_WINDOW_SECONDS
    )
    all_intra_dups = stream_intra_dups + video_intra_dups

    # 3. Deduplikasi lintas jalur (stream vs video)
    cross_dups = _find_cross_path_duplicates(
        stream_compacted, video_compacted, DEDUP_CROSS_WINDOW_SECONDS
    )

    # 4. Identifikasi repeat passes (kunjungan kembali nyata)
    all_compacted = stream_compacted + video_compacted
    repeat_groups = _find_repeat_passes(all_compacted, REPEAT_PASS_GAP_SECONDS)

    # 5. Hitung statistik
    total_raw = len(all_events)
    total_intra_dups = len(all_intra_dups)
    total_cross_dups = len(cross_dups)
    total_dups = total_intra_dups + total_cross_dups
    after_dedup = max(0, total_raw - total_dups)
    dup_rate = round(total_dups * 100.0 / total_raw, 1) if total_raw > 0 else 0.0
    data_loss_risk = _estimate_data_loss_risk(cross_dups, repeat_groups)

    elapsed = round(time.perf_counter() - t_start, 3)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": camera_id,
        "lookback_seconds": lookback_seconds,
        "evaluation_elapsed_seconds": elapsed,
        "database_mutated": False,

        # Statistik sebelum deduplikasi
        "before_deduplication": {
            "total_plate_events": total_raw,
            "stream_source_events": len(stream_events_raw),
            "video_source_events": len(video_events_raw),
        },

        # Statistik setelah deduplikasi
        "after_deduplication": {
            "total_plate_events": after_dedup,
            "events_removed": total_dups,
            "duplication_rate_percent": dup_rate,
        },

        # Rincian intra-source duplicates
        "intra_source_duplicates": {
            "count": total_intra_dups,
            "stream_duplicates": len(stream_intra_dups),
            "video_duplicates": len(video_intra_dups),
            "window_seconds": DEDUP_INTRA_WINDOW_SECONDS,
            "description": (
                "Frame/event lanjutan dari tracking kendaraan yang sama dalam satu jalur. "
                "Umumnya disebabkan kendaraan bergerak pelan atau terpantau pada beberapa frame."
            ),
            "samples": all_intra_dups[:10],
        },

        # Rincian cross-path duplicates
        "cross_path_duplicates": {
            "count": total_cross_dups,
            "window_seconds": DEDUP_CROSS_WINDOW_SECONDS,
            "description": (
                "Plat yang sama terdeteksi oleh jalur realtime (stream) DAN jalur segmen video "
                f"dalam jendela ±{DEDUP_CROSS_WINDOW_SECONDS}s. Jalur realtime dipertahankan "
                "sebagai sumber utama; jalur video dianggap duplikat prospektif."
            ),
            "samples": cross_dups[:10],
        },

        # Repeat passes (kunjungan kembali nyata — bukan duplikat)
        "repeat_passes": {
            "count": len(repeat_groups),
            "gap_seconds": REPEAT_PASS_GAP_SECONDS,
            "description": (
                f"Kendaraan berplat sama yang terdeteksi lagi setelah jeda ≥{REPEAT_PASS_GAP_SECONDS}s. "
                "Ini adalah kunjungan kembali nyata, BUKAN duplikat, dan tidak dihapus."
            ),
            "groups": repeat_groups[:10],
        },

        # Estimasi risiko data terlewat
        "data_loss_risk": data_loss_risk,

        # Analisis dan rekomendasi
        "analysis": {
            "sample_adequate": bool(
                total_raw >= 5
                and (len(video_events_raw) >= 2 or total_cross_dups > 0)
            ),
            "note": (
                "Sampel belum mencukupi untuk estimasi duplikasi cross-path yang valid."
                if len(video_events_raw) < 2
                else "Evaluasi berhasil dengan sampel lintas jalur yang cukup."
            ),
            "cross_path_lag_observed_seconds": _compute_median_lag(cross_dups),
            "recommended_action": _recommend_action(
                total_cross_dups, total_intra_dups, total_raw
            ),
        },
    }


def _estimate_data_loss_risk(cross_dups, repeat_groups):
    """
    Estimasi risiko data terlewat jika deduplikasi diterapkan.
    Data terlewat berarti kendaraan yang benar-benar berbeda ikut terhapus.
    """
    # Cek apakah ada repeat_passes dari plat yang juga muncul di cross_dups
    cross_plates = {(d["camera_id"], d["plate"]) for d in cross_dups}
    repeat_plates = {(g["camera_id"], g["plate"]) for g in repeat_groups}
    overlap = cross_plates & repeat_plates
    risk_level = "LOW"
    risk_notes = []
    if overlap:
        risk_level = "MEDIUM"
        risk_notes.append(
            f"{len(overlap)} plat ditemukan di cross-path duplicates DAN repeat-pass groups. "
            "Deduplikasi lintas jalur untuk plat ini perlu verifikasi manual."
        )
    if not cross_plates:
        risk_notes.append("Belum ada cross-path duplicate — risiko sangat rendah untuk saat ini.")
    return {
        "risk_level": risk_level,
        "overlap_plates_count": len(overlap),
        "overlap_plates": [
            {"camera_id": k[0], "plate": k[1]} for k in sorted(overlap)
        ][:5],
        "notes": risk_notes,
    }


def _compute_median_lag(cross_dups):
    """Hitung median lag antara stream dan video untuk cross-path duplicates."""
    if not cross_dups:
        return None
    deltas = sorted(d["delta_seconds"] for d in cross_dups)
    mid = len(deltas) // 2
    if len(deltas) % 2 == 1:
        return deltas[mid]
    return round((deltas[mid - 1] + deltas[mid]) / 2, 1)


def _recommend_action(cross_dups_count, intra_dups_count, total_raw):
    """Rekomendasi tindakan berdasarkan hasil evaluasi."""
    if total_raw < 5:
        return (
            "Sampel terlalu sedikit. Jalankan evaluasi setelah rekaman segmen mengandung "
            "minimal beberapa deteksi plat terbaca."
        )
    if cross_dups_count == 0 and intra_dups_count == 0:
        return (
            "Tidak ada duplikasi terdeteksi. Kemungkinan segmen belum menghasilkan "
            "deteksi plat yang dapat dicocokkan dengan jalur realtime."
        )
    if cross_dups_count >= 3 and intra_dups_count >= 3:
        return (
            "Duplikasi lintas jalur dan intra-source terkonfirmasi. "
            "Siap untuk validasi mekanisme deduplikasi sebelum penerapan ke pipeline DB."
        )
    return (
        f"Ditemukan {cross_dups_count} cross-path dan {intra_dups_count} intra-source duplicates. "
        "Lanjutkan pengujian dengan lebih banyak segmen sebelum memutuskan penerapan dedup."
    )
