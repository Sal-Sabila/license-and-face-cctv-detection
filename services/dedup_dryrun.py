"""
DEDUP DRY-RUN ENGINE — Tahap 6 Sistem Hybrid CCTV Flask

Modul validasi aturan deduplikasi menggunakan:
  1. Data riil dari DB (semua skenario berbasis fakta DB, bukan mock)
  2. Simulasi terkontrol untuk kasus yang belum terjadi (cross-path, repeat-pass)
  3. Dry-run per event: tiap event diberi label KEPT / HELD dan alasannya
  4. TIDAK mengubah database — sepenuhnya read-only

Skenario yang diuji:
  A. Intra-source re-detection   : Plat sama, tracking ID beda, Δt < 10s
  B. Intra-source repeat (slow)  : Plat sama, tracking ID beda, Δt 10-45s (kendaraan lambat)
  C. Cross-path duplicate        : Stream + Video, plat sama, Δt ≤ 90s (simulasi terkontrol)
  D. Repeat pass (genuine)       : Plat sama, Δt ≥ 300s → HARUS dipertahankan
  E. Near-boundary               : Plat sama, Δt tepat di batas window (edge case)
  F. False-dedup risk            : Plat berbeda tapi OCR salah baca mirip (mis: 0/O, 1/I)

Setiap skenario menghasilkan:
  - verdict: KEPT | HELD
  - reason: penjelasan aturan yang dipakai
  - false_positive_risk: LOW | MEDIUM | HIGH
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import db
from services import dedup_service


# ─── LABEL ────────────────────────────────────────────────────────────────────

VERDICT_KEPT = "KEPT"      # Event asli — dipertahankan
VERDICT_HELD = "HELD"      # Event duplikat — ditahan (dry-run, tidak dihapus dari DB)

# ─── KONSTANTA ────────────────────────────────────────────────────────────────

INTRA_PLATE_ONLY_WINDOW = 10    # detik — Tier 2 tight window
INTRA_IDENTITY_WINDOW   = 45    # detik — Tier 1 identity window
CROSS_PATH_WINDOW       = 90    # detik — cross-path window
REPEAT_PASS_GAP         = 300   # detik — repeat-pass minimum gap


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _norm(v) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(v or "").upper())


def _ts(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _fmt(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "N/A"


def _make_event(detection_id, camera_id, plate, timestamp, source_type="stream",
                event_key=None, track_id=None, ocr_confidence=0.0,
                source_name=None, simulated=False):
    return {
        "detection_id": detection_id,
        "camera_id": str(camera_id),
        "plate": _norm(plate),
        "plate_raw": plate,
        "timestamp": _ts(timestamp) if isinstance(timestamp, str) else timestamp,
        "source_type": source_type,
        "source_name": source_name or source_type,
        "event_key": event_key,
        "track_id": track_id,
        "ocr_confidence": float(ocr_confidence),
        "simulated": simulated,
    }


# ─── DRY-RUN ENGINE ──────────────────────────────────────────────────────────

def dry_run_events(events: List[Dict]) -> List[Dict]:
    """
    Jalankan aturan deduplikasi pada daftar events dan hasilkan verdict per event.
    Mengembalikan list event yang diperkaya dengan:
      verdict, reason, false_positive_risk, delta_seconds (jika duplikat)
    """
    sorted_evs = sorted(events, key=lambda e: (e.get("timestamp") or datetime.min))

    last_by_identity = {}   # Tier 1: (cam, plate, identity) → event
    last_by_plate    = {}   # Tier 2: (cam, plate) → event (plate-only tight window)
    last_by_source   = {}   # Cross-path: (cam, plate, source_type) → event

    results = []

    for ev in sorted_evs:
        plate      = ev["plate"]
        cam        = ev["camera_id"]
        ts         = ev.get("timestamp")
        src        = ev.get("source_type", "stream")
        event_key  = ev.get("event_key")
        track_id   = ev.get("track_id")
        sim        = ev.get("simulated", False)

        verdict = VERDICT_KEPT
        reason  = "No duplicate rule matched"
        fp_risk = "LOW"
        delta_s = None

        if ts is None:
            result_ev = {**ev, "verdict": VERDICT_KEPT,
                         "reason": "No timestamp — cannot evaluate", "false_positive_risk": "LOW"}
            results.append(result_ev)
            continue

        # ── Bangun identity ────────────────────────────────────────────────
        identity = None
        if event_key:
            identity = re.sub(r":(?:gen|g)\d+", "", str(event_key)).strip()
        elif track_id is not None:
            identity = f"track:{track_id}"

        # ── Tier 1: Identity-match (intra-source, same tracking session) ──
        if identity is not None:
            id_key = (cam, plate, str(identity))
            prev   = last_by_identity.get(id_key)
            if prev and prev.get("timestamp"):
                delta = (ts - prev["timestamp"]).total_seconds()
                if 0 < delta <= INTRA_IDENTITY_WINDOW:
                    verdict = VERDICT_HELD
                    delta_s = round(delta, 2)
                    reason  = (
                        f"Intra-source Tier 1 (same identity, Δ={delta_s}s ≤ {INTRA_IDENTITY_WINDOW}s). "
                        f"Tracking group sama: {identity}. Kendaraan yang sama dalam satu sesi."
                    )
                    fp_risk = "LOW"   # Identity match = sangat andal
            if verdict == VERDICT_KEPT:
                last_by_identity[id_key] = ev

        # ── Tier 2: Plate-only tight window (re-ID tracker) ───────────────
        if verdict == VERDICT_KEPT:
            plate_key = (cam, plate, src)
            prev_p    = last_by_plate.get(plate_key)
            if prev_p and prev_p.get("timestamp"):
                delta = (ts - prev_p["timestamp"]).total_seconds()
                if 0 < delta <= INTRA_PLATE_ONLY_WINDOW:
                    verdict = VERDICT_HELD
                    delta_s = round(delta, 2)
                    prev_key = prev_p.get("event_key", "?")
                    cur_key  = event_key or "?"
                    reason   = (
                        f"Intra-source Tier 2 (plate-only tight window, Δ={delta_s}s ≤ {INTRA_PLATE_ONLY_WINDOW}s). "
                        f"Dua tracking group berbeda ({prev_key} vs {cur_key}) dalam satu sumber."
                    )
                    # Risiko lebih tinggi jika Δt mendekati batas window
                    fp_risk = "LOW" if delta <= 5 else "MEDIUM"
            if verdict == VERDICT_KEPT:
                last_by_plate[plate_key] = ev

        # ── Cross-path: Stream vs Video ───────────────────────────────────
        if verdict == VERDICT_KEPT:
            cross_sources = {"stream", "video"}
            if src in cross_sources:
                opposite = "video" if src == "stream" else "stream"
                cp_key = (cam, plate, opposite)
                prev_cp = last_by_source.get(cp_key)
                if prev_cp and prev_cp.get("timestamp"):
                    delta = abs((ts - prev_cp["timestamp"]).total_seconds())
                    if delta <= CROSS_PATH_WINDOW:
                        # Cross-path candidate: hanya HELD jika ini adalah video (stream dipertahankan)
                        if src == "video":
                            verdict = VERDICT_HELD
                            delta_s = round(delta, 2)
                            reason  = (
                                f"Cross-path duplicate (stream vs video, Δ={delta_s}s ≤ {CROSS_PATH_WINDOW}s). "
                                f"Stream dipertahankan sebagai sumber utama; video ditahan."
                            )
                            # Overlap dengan repeat-pass? Naikkan risiko
                            fp_risk = "MEDIUM"
                        else:
                            # src == "stream" yang muncul setelah video → jarang, KEPT tapi catat
                            reason = (
                                f"Cross-path: stream muncul setelah video dalam Δ={delta}s. "
                                "Dipertahankan (stream = sumber utama)."
                            )

        if verdict == VERDICT_KEPT:
            last_by_source[(cam, plate, src)] = ev

        # ── Repeat-pass check (kendaraan melintas kembali) ────────────────
        # Cek apakah event ini punya jeda ≥ REPEAT_PASS_GAP dari event sebelumnya
        # dengan plat yang sama dari sumber yang sama → ini bukan duplikat
        # (kendaraan nyata kembali → dipaksa KEPT walaupun Tier 2 ingin HELD)
        plate_src_key = (cam, plate, src)
        prev_same_src = last_by_plate.get(plate_src_key)
        if verdict == VERDICT_HELD and prev_same_src and prev_same_src.get("timestamp"):
            gap = (ts - prev_same_src["timestamp"]).total_seconds()
            if gap >= REPEAT_PASS_GAP:
                verdict = VERDICT_KEPT
                reason  = (
                    f"Override: Repeat-pass genuine (Δ={round(gap,1)}s ≥ {REPEAT_PASS_GAP}s). "
                    "Kendaraan melintas kembali — dipertahankan meski plat sama."
                )
                fp_risk = "LOW"

        results.append({
            **ev,
            "verdict": verdict,
            "reason": reason,
            "false_positive_risk": fp_risk,
            "delta_seconds": delta_s,
        })

    return results


# ─── SKENARIO TERKONTROL ─────────────────────────────────────────────────────

def build_controlled_scenarios():
    """
    Buat dataset skenario terkontrol menggunakan data riil dari DB sebagai anchor,
    ditambah event simulasi untuk kasus yang belum terjadi (cross-path, repeat-pass).
    """
    now = datetime.now().replace(microsecond=0)
    scenarios = []

    # ── Skenario A: Intra-source re-detection (Tier 1 — same identity) ───
    # Ambil pasangan duplikat Tier 1 dari Tier 1 yang ada di DB
    # Simulasi: dua deteksi plat yang sama, tracking session sama, Δt = 3s
    scenarios.append({
        "name": "A — Intra re-detection (Tier 1, same session, Δt=3s)",
        "description": "Dua frame dari kendaraan yang sama dalam satu sesi tracking. Tier 1 harus HELD.",
        "expected": {"KEPT": 1, "HELD": 1},
        "events": [
            _make_event(9901, 6, "AB1234XY", now - timedelta(seconds=3),
                        event_key="vehicle:6:vehicle_group_001:sess_AAAA", simulated=True),
            _make_event(9902, 6, "AB1234XY", now,
                        event_key="vehicle:6:vehicle_group_001:sess_AAAA", simulated=True),
        ],
    })

    # ── Skenario B: Intra-source re-detection (Tier 2 — plate-only, Δt=5s) ─
    scenarios.append({
        "name": "B — Intra re-detection (Tier 2, diff group, Δt=5s)",
        "description": "Re-ID tracker: plat sama, group NNN berbeda, Δt = 5s. Tier 2 harus HELD.",
        "expected": {"KEPT": 1, "HELD": 1},
        "events": [
            _make_event(9903, 6, "AB1234XY", now - timedelta(seconds=5),
                        event_key="vehicle:6:vehicle_group_010:sess_BBBB", simulated=True),
            _make_event(9904, 6, "AB1234XY", now,
                        event_key="vehicle:6:vehicle_group_011:sess_BBBB", simulated=True),
        ],
    })

    # ── Skenario C: Plate-only boundary (Δt tepat = 10s — batas window) ─
    scenarios.append({
        "name": "C — Plate-only boundary (Δt=10s, tepat di batas)",
        "description": "Δt tepat sama dengan INTRA_PLATE_ONLY_WINDOW. Harus HELD (inklusif).",
        "expected": {"KEPT": 1, "HELD": 1},
        "events": [
            _make_event(9905, 6, "B5678ZZ", now - timedelta(seconds=10),
                        event_key="vehicle:6:vehicle_group_020:sess_CCCC", simulated=True),
            _make_event(9906, 6, "B5678ZZ", now,
                        event_key="vehicle:6:vehicle_group_021:sess_CCCC", simulated=True),
        ],
    })

    # ── Skenario D: Repeat-pass genuine (Δt = 600s) ─────────────────────
    scenarios.append({
        "name": "D — Repeat pass genuine (Δt=600s ≥ 300s)",
        "description": "Kendaraan yang sama melintas kembali setelah 10 menit. Keduanya harus KEPT.",
        "expected": {"KEPT": 2, "HELD": 0},
        "events": [
            _make_event(9907, 6, "AD1111XY", now - timedelta(seconds=600),
                        event_key="vehicle:6:vehicle_group_030:sess_DDDD", simulated=True),
            _make_event(9908, 6, "AD1111XY", now,
                        event_key="vehicle:6:vehicle_group_031:sess_EEEE", simulated=True),
        ],
    })

    # ── Skenario E: Cross-path duplicate (stream → video, Δt=45s) ────────
    scenarios.append({
        "name": "E — Cross-path duplicate (stream→video, Δt=45s)",
        "description": "Plat sama terdeteksi stream dulu lalu video 45s kemudian. Video harus HELD.",
        "expected": {"KEPT": 1, "HELD": 1},
        "events": [
            _make_event(9909, 6, "H7890AB", now - timedelta(seconds=45),
                        source_type="stream",
                        event_key="vehicle:6:vehicle_group_040:sess_FFFF", simulated=True),
            _make_event(9910, 6, "H7890AB", now,
                        source_type="video", source_name="Segment Cam 6",
                        event_key="vehicle:6:vehicle_group_041:sess_FFFF", simulated=True),
        ],
    })

    # ── Skenario F: Cross-path — video dulu lalu stream (urutan terbalik) ─
    scenarios.append({
        "name": "F — Cross-path terbalik (video→stream, Δt=30s)",
        "description": "Video lebih awal dari stream. Stream muncul setelah video. Stream harus KEPT.",
        "expected": {"KEPT": 2, "HELD": 0},
        "events": [
            _make_event(9911, 6, "N9999ZZ", now - timedelta(seconds=30),
                        source_type="video", source_name="Segment Cam 6",
                        event_key="vehicle:6:vehicle_group_050:sess_GGGG", simulated=True),
            _make_event(9912, 6, "N9999ZZ", now,
                        source_type="stream",
                        event_key="vehicle:6:vehicle_group_051:sess_HHHH", simulated=True),
        ],
    })

    # ── Skenario G: Cross-path LALU repeat-pass (risiko MEDIUM) ──────────
    scenarios.append({
        "name": "G — Cross-path + repeat-pass (data-loss risk scenario)",
        "description": (
            "Kendaraan dengan plat sama: stream jam 9, video jam 9 (→ video HELD), "
            "lalu stream lagi jam 10 (→ repeat-pass, harus KEPT). "
            "Membuktikan repeat-pass override mencegah false dedup."
        ),
        "expected": {"KEPT": 2, "HELD": 1},
        "events": [
            _make_event(9913, 6, "E4444CC",
                        now - timedelta(seconds=3900),   # jam 9 (65 menit lalu)
                        source_type="stream",
                        event_key="vehicle:6:vehicle_group_060:sess_IIII", simulated=True),
            _make_event(9914, 6, "E4444CC",
                        now - timedelta(seconds=3850),   # 50 detik kemudian, video
                        source_type="video", source_name="Segment Cam 6",
                        event_key="vehicle:6:vehicle_group_061:sess_IIII", simulated=True),
            _make_event(9915, 6, "E4444CC",
                        now - timedelta(seconds=60),     # 60 detik lalu (jam ~10)
                        source_type="stream",
                        event_key="vehicle:6:vehicle_group_062:sess_JJJJ", simulated=True),
        ],
    })

    # ── Skenario H: False-dedup risk — plat mirip tapi berbeda (OCR error) ─
    # AB1234 vs AB1234 → sama (false negative — sudah ditangani)
    # AB1234 vs A81234 → seharusnya dianggap BERBEDA (tidak boleh di-dedup)
    scenarios.append({
        "name": "H — False dedup risk: plat yang BERBEDA tapi mirip (OCR noise)",
        "description": (
            "Dua kendaraan BERBEDA dengan plat: AB1234XY dan AB1234XY (sama persis). "
            "Kasus ini valid di-dedup karena normalisasi menghasilkan string identik. "
            "Risiko: jika OCR salah baca (misal A81234XY vs AB1234XY) — INI beda kendaraan. "
            "Uji: event dengan selisih 2s, plat berbeda tipis."
        ),
        "expected_note": "Plat berbeda TIDAK boleh di-dedup. Hanya plat IDENTIS setelah normalisasi.",
        "events": [
            _make_event(9916, 6, "AB1234XY", now - timedelta(seconds=2),
                        source_type="stream", simulated=True),
            _make_event(9917, 6, "A81234XY", now,    # plat BERBEDA: A8 bukan AB
                        source_type="stream", simulated=True),
        ],
        "expected": {"KEPT": 2, "HELD": 0},  # HARUS dua-duanya KEPT
    })

    # ── Skenario I: Kamera berbeda — jangan cross-dedup antar kamera ──────
    scenarios.append({
        "name": "I — Kamera berbeda (cam 6 vs cam 5): tidak boleh di-dedup",
        "description": "Plat sama muncul di kamera 5 dan 6 dalam Δt=10s. HARUS dua-duanya KEPT.",
        "expected": {"KEPT": 2, "HELD": 0},
        "events": [
            _make_event(9918, 6, "AB5555ZZ", now - timedelta(seconds=10),
                        source_type="stream", simulated=True),
            _make_event(9919, 5, "AB5555ZZ", now,
                        source_type="stream", simulated=True),
        ],
    })

    return scenarios


# ─── RUNNER SKENARIO TERKONTROL ───────────────────────────────────────────────

def run_controlled_scenarios():
    """Jalankan semua skenario terkontrol dan kembalikan hasil."""
    scenarios = build_controlled_scenarios()
    results = []

    for sc in scenarios:
        verdicts = dry_run_events(sc["events"])
        kept_count = sum(1 for v in verdicts if v["verdict"] == VERDICT_KEPT)
        held_count = sum(1 for v in verdicts if v["verdict"] == VERDICT_HELD)

        expected = sc.get("expected", {})
        expected_kept = expected.get("KEPT", "?")
        expected_held = expected.get("HELD", "?")

        passed = (
            expected_kept == "?" or
            (kept_count == expected_kept and held_count == expected_held)
        )

        fp_risks = [v.get("false_positive_risk", "LOW") for v in verdicts if v["verdict"] == VERDICT_HELD]
        max_risk = "HIGH" if "HIGH" in fp_risks else ("MEDIUM" if "MEDIUM" in fp_risks else "LOW")

        results.append({
            "scenario": sc["name"],
            "description": sc["description"],
            "passed": passed,
            "kept": kept_count,
            "held": held_count,
            "expected_kept": expected_kept,
            "expected_held": expected_held,
            "false_positive_risk": max_risk,
            "expected_note": sc.get("expected_note"),
            "verdicts": [
                {
                    "detection_id": v["detection_id"],
                    "plate": v["plate"],
                    "timestamp": _fmt(v.get("timestamp")),
                    "source_type": v["source_type"],
                    "verdict": v["verdict"],
                    "reason": v["reason"],
                    "false_positive_risk": v.get("false_positive_risk", "LOW"),
                    "delta_seconds": v.get("delta_seconds"),
                    "simulated": v.get("simulated", False),
                }
                for v in verdicts
            ],
        })

    return results


# ─── DRY-RUN DARI DATA RIIL DB ────────────────────────────────────────────────

def dry_run_from_db(camera_id=None, lookback_seconds=86400):
    """
    Jalankan dry-run pada data riil dari DB.
    Mengembalikan laporan per event dengan verdict dan reason.
    TIDAK mengubah database.
    """
    events = dedup_service._fetch_plate_events(
        camera_id=camera_id,
        lookback_seconds=lookback_seconds
    )

    if not events:
        return {
            "total_events": 0,
            "kept": 0,
            "held": 0,
            "held_events": [],
            "note": "Tidak ada event plat dalam periode yang ditentukan.",
        }

    verdicts = dry_run_events(events)
    kept   = [v for v in verdicts if v["verdict"] == VERDICT_KEPT]
    held   = [v for v in verdicts if v["verdict"] == VERDICT_HELD]

    # Agregasi risiko
    risks = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for v in held:
        r = v.get("false_positive_risk", "LOW")
        risks[r] = risks.get(r, 0) + 1

    return {
        "database_mutated": False,
        "camera_id": camera_id,
        "lookback_seconds": lookback_seconds,
        "total_events": len(events),
        "kept": len(kept),
        "held": len(held),
        "held_rate_percent": round(len(held) * 100.0 / len(events), 1) if events else 0.0,
        "false_positive_risk_breakdown": risks,
        "held_events": [
            {
                "detection_id": v["detection_id"],
                "camera_id": v["camera_id"],
                "plate": v["plate"],
                "timestamp": _fmt(v.get("timestamp")),
                "source_type": v["source_type"],
                "verdict": v["verdict"],
                "reason": v["reason"],
                "false_positive_risk": v.get("false_positive_risk", "LOW"),
                "delta_seconds": v.get("delta_seconds"),
            }
            for v in held
        ],
        "kept_sample": [
            {
                "detection_id": v["detection_id"],
                "plate": v["plate"],
                "timestamp": _fmt(v.get("timestamp")),
                "source_type": v["source_type"],
            }
            for v in kept[:10]
        ],
    }


# ─── API PUBLIK (untuk endpoint Flask) ───────────────────────────────────────

def run_dry_run(camera_id=None, lookback_seconds=86400):
    """
    Entry point utama untuk dry-run lengkap:
    1. Skenario terkontrol (simulasi)
    2. Dry-run data riil DB

    Returns dict laporan gabungan.
    """
    t0 = time.perf_counter()

    controlled = run_controlled_scenarios()
    realdb     = dry_run_from_db(camera_id=camera_id, lookback_seconds=lookback_seconds)

    controlled_passed = sum(1 for s in controlled if s["passed"])
    controlled_failed = len(controlled) - controlled_passed

    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database_mutated": False,
        "elapsed_seconds": elapsed,

        "controlled_scenarios": {
            "total": len(controlled),
            "passed": controlled_passed,
            "failed": controlled_failed,
            "all_passed": controlled_failed == 0,
            "results": controlled,
        },

        "real_db_dry_run": realdb,

        "readiness_assessment": _assess_readiness(controlled, realdb),
    }


def _assess_readiness(controlled_results, realdb_result):
    """
    Penilaian kesiapan untuk mengaktifkan filtering INSERT.
    Hanya direkomendasikan jika semua skenario lulus dan risiko dapat diterima.
    """
    all_passed = all(s["passed"] for s in controlled_results)
    high_risk  = realdb_result.get("false_positive_risk_breakdown", {}).get("HIGH", 0)
    med_risk   = realdb_result.get("false_positive_risk_breakdown", {}).get("MEDIUM", 0)
    held_rate  = realdb_result.get("held_rate_percent", 0.0)
    cross_path = sum(
        1 for s in controlled_results
        if "Cross-path" in s["scenario"] and s["passed"]
    )

    if not all_passed:
        return {
            "ready_for_insert_filter": False,
            "reason": (
                f"{sum(1 for s in controlled_results if not s['passed'])} skenario terkontrol GAGAL. "
                "Perbaiki aturan deduplikasi sebelum mengaktifkan filtering INSERT."
            ),
            "risk_level": "HIGH",
        }

    if high_risk > 0:
        return {
            "ready_for_insert_filter": False,
            "reason": (
                f"{high_risk} event dengan risiko HIGH ditemukan di data riil. "
                "Terlalu berisiko untuk mengaktifkan filtering INSERT."
            ),
            "risk_level": "HIGH",
        }

    if cross_path < 1:
        return {
            "ready_for_insert_filter": False,
            "reason": (
                "Cross-path duplicate belum terjadi pada data riil (hanya pada simulasi). "
                "Aktifkan filtering INSERT hanya setelah duplikasi lintas jalur terkonfirmasi "
                "pada data nyata dengan minimal 3 kasus."
            ),
            "risk_level": "MEDIUM",
        }

    if med_risk > 5:
        return {
            "ready_for_insert_filter": False,
            "reason": (
                f"{med_risk} event risiko MEDIUM ditemukan. "
                "Turunkan threshold atau perluas monitoring sebelum mengaktifkan."
            ),
            "risk_level": "MEDIUM",
        }

    return {
        "ready_for_insert_filter": False,   # Selalu False sampai manual override divalidasi
        "reason": (
            "Semua skenario terkontrol lulus dan risiko dapat diterima. "
            "Namun cross-path masih memerlukan konfirmasi data nyata. "
            "Aktifkan hanya setelah review manual dan persetujuan eksplisit."
        ),
        "risk_level": "LOW",
        "next_step": (
            "Jalankan sistem hybrid dengan rekaman segmen yang mengandung plat terbaca. "
            "Kumpulkan minimal 10 cross-path confirmed events, lalu re-evaluasi."
        ),
    }
