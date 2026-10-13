"""
SHADOW DEDUP SERVICE — Tahap 8 Sistem Hybrid CCTV Flask

Service layer deduplikasi dalam mode SHADOW / DRY-RUN.
Bertugas mengevaluasi event deteksi jalur video segmen sebelum disimpan,
mencatat keputusan KEEP atau HOLD beserta alasan, metrik selisih waktu,
dan log audit detail, TANPA membatalkan atau memblokir INSERT aktual.

Prinsip Utama:
  1. Event stream (realtime) TIDAK PERNAH diubah atau ditahan.
  2. Kendaraan tanpa plat (unplated) SELALU KEPT (tidak dibuang).
  3. Kendaraan yang melintas kembali (repeat pass, gap >= 300s) SELALU KEPT.
  4. Pasangan cross-path hanya dievaluasi jika KEDUA sumber memiliki plat terbaca valid.
  5. Setiap keputusan dicatat dalam audit log in-memory dan persisted ke JSON.
  6. ZERO MUTATION pada skema atau data database real_cctv.
"""

import os
import re
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import db
from services import dedup_service

# ─── KONSTANTA ────────────────────────────────────────────────────────────────
VERDICT_KEEP = "KEEP"
VERDICT_HOLD = "HOLD"

INTRA_PLATE_ONLY_WINDOW = 10.0   # detik (Tier 2 tight window)
INTRA_IDENTITY_WINDOW   = 45.0   # detik (Tier 1 session/track ID window)
CROSS_PATH_WINDOW       = 90.0   # detik (Stream vs Segment window)
REPEAT_PASS_GAP         = 300.0  # detik (Kunjungan kembali genuine)

DEFAULT_AUDIT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scratch",
    "shadow_dedup_audit.json"
)


# ─── NORMALISASI & HELPERS ────────────────────────────────────────────────────
def normalize_plate(value: Any) -> str:
    """Bersihkan plat menjadi huruf besar alfanumerik tanpa spasi/simbol."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def parse_timestamp(value: Any) -> Optional[datetime]:
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


class ShadowDedupService:
    """
    Singleton service untuk shadow deduplication.
    Mengevaluasi event sebelum/saat proses video segmen selesai diproses.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, audit_file_path: Optional[str] = None):
        self.audit_file_path = audit_file_path or DEFAULT_AUDIT_LOG_FILE
        self._audit_records: List[Dict[str, Any]] = []
        self._audit_lock = threading.Lock()
        self._load_audit_log()

    @classmethod
    def get_instance(cls, audit_file_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(audit_file_path)
            return cls._instance

    def _load_audit_log(self):
        with self._audit_lock:
            if os.path.isfile(self.audit_file_path):
                try:
                    with open(self.audit_file_path, "r", encoding="utf-8") as f:
                        self._audit_records = json.load(f)
                except Exception as exc:
                    print(f"[SHADOW DEDUP] Gagal memuat audit log: {exc}")
                    self._audit_records = []
            else:
                self._audit_records = []

    def _save_audit_log(self):
        try:
            os.makedirs(os.path.dirname(self.audit_file_path), exist_ok=True)
            with open(self.audit_file_path, "w", encoding="utf-8") as f:
                json.dump(self._audit_records, f, indent=2, default=str)
        except Exception as exc:
            print(f"[SHADOW DEDUP] Gagal menyimpan audit log: {exc}")

    def evaluate_candidate(
        self,
        camera_id: int,
        plate_number: Optional[str],
        timestamp: Optional[datetime],
        source_type: str = "video",
        object_type: str = "vehicle",
        vehicle_type: str = "unknown",
        event_key: Optional[str] = None,
        track_id: Optional[int] = None,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluasi kandidat event deteksi (terutama dari jalur video segmen).
        Mode: SHADOW / DRY-RUN.
        Mengembalikan verdict KEEP atau HOLD beserta alasan dan rincian audit.
        TIDAK melakukan pembatalan INSERT.
        """
        eval_time = datetime.now()
        ts = parse_timestamp(timestamp) or eval_time
        plate_norm = normalize_plate(plate_number)
        has_plate = bool(plate_norm and plate_norm not in ("-", "UNKNOWN") and len(plate_norm) >= 3)

        verdict = VERDICT_KEEP
        rule_triggered = "DEFAULT_KEEP"
        reason = "Event unik atau baru pertama kali terdeteksi"
        false_positive_risk = "LOW"
        matched_event = None
        delta_seconds = None

        # ── ATURAN 1: Event stream selalu dipertahankan ───────────────────────
        if str(source_type).lower().strip() == "stream":
            verdict = VERDICT_KEEP
            rule_triggered = "STREAM_SOURCE_IMMUNE"
            reason = "Event stream realtime selalu dipertahankan tanpa perubahan"
            return self._record_decision(
                camera_id, plate_norm, ts, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds
            )

        # ── ATURAN 2: Kendaraan tanpa plat nomor selalu dipertahankan ─────────
        if not has_plate:
            verdict = VERDICT_KEEP
            rule_triggered = "UNPLATED_VEHICLE_KEPT"
            reason = "Kendaraan tanpa plat terbaca tidak dibuang untuk mencegah false deduplication"
            return self._record_decision(
                camera_id, plate_norm, ts, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds
            )

        # ── ATURAN 3: Periksa apakah plat sama melintas kembali (Repeat Pass) ──
        # Query event sebelumnya untuk plat dan kamera yang sama di DB
        history = self._query_plate_history(camera_id, plate_norm, lookback_seconds=86400)
        
        # Cek repeat pass: jika event terakhir berjarak >= REPEAT_PASS_GAP detik
        if history:
            last_event = history[-1]
            last_ts = last_event["timestamp"]
            gap = abs((ts - last_ts).total_seconds())

            if gap >= REPEAT_PASS_GAP:
                verdict = VERDICT_KEEP
                rule_triggered = "REPEAT_PASS_GENUINE"
                reason = f"Kendaraan melintas kembali (Δt={gap:.1f}s >= {REPEAT_PASS_GAP}s); kunjungan ulang sah"
                delta_seconds = round(gap, 2)
                matched_event = last_event
                return self._record_decision(
                    camera_id, plate_norm, ts, source_type, object_type,
                    verdict, rule_triggered, reason, false_positive_risk,
                    event_key, track_id, source_name, matched_event, delta_seconds
                )

            # ── ATURAN 4: Cross-path deduplication (Stream vs Video) ──────────
            # Hanya berlaku jika event pembanding dari stream memiliki plat terbaca valid
            stream_matches = [
                h for h in history 
                if h["source_type"] == "stream" and abs((ts - h["timestamp"]).total_seconds()) <= CROSS_PATH_WINDOW
            ]
            if stream_matches:
                closest_stream = min(stream_matches, key=lambda x: abs((ts - x["timestamp"]).total_seconds()))
                delta_cross = abs((ts - closest_stream["timestamp"]).total_seconds())
                verdict = VERDICT_HOLD
                rule_triggered = "CROSS_PATH_DUPLICATE_SHADOW"
                reason = (
                    f"Plat sama ({plate_norm}) sudah terdeteksi di jalur stream "
                    f"(ID={closest_stream['detection_id']}) dalam jendela Δt={delta_cross:.1f}s <= {CROSS_PATH_WINDOW}s. "
                    f"Keputusan shadow: HOLD (di jalur video)."
                )
                false_positive_risk = "LOW" if delta_cross <= 45 else "MEDIUM"
                delta_seconds = round(delta_cross, 2)
                matched_event = closest_stream
                return self._record_decision(
                    camera_id, plate_norm, ts, source_type, object_type,
                    verdict, rule_triggered, reason, false_positive_risk,
                    event_key, track_id, source_name, matched_event, delta_seconds
                )

            # ── ATURAN 5: Intra-source video duplication (segmen berturutan) ──
            video_matches = [
                h for h in history 
                if h["source_type"] == "video" and abs((ts - h["timestamp"]).total_seconds()) <= INTRA_IDENTITY_WINDOW
            ]
            if video_matches:
                closest_video = min(video_matches, key=lambda x: abs((ts - x["timestamp"]).total_seconds()))
                delta_intra = abs((ts - closest_video["timestamp"]).total_seconds())
                
                # Tier 1: Identity match (sesi/track sama)
                curr_ident = self._extract_identity(event_key, track_id)
                prev_ident = self._extract_identity(closest_video.get("event_key"), closest_video.get("track_id"))
                
                if curr_ident and prev_ident and curr_ident == prev_ident and delta_intra <= INTRA_IDENTITY_WINDOW:
                    verdict = VERDICT_HOLD
                    rule_triggered = "INTRA_SOURCE_TIER1_SHADOW"
                    reason = f"Duplikat sesi tracking yang sama ({curr_ident}) dalam video (Δt={delta_intra:.1f}s <= {INTRA_IDENTITY_WINDOW}s)"
                    false_positive_risk = "LOW"
                    delta_seconds = round(delta_intra, 2)
                    matched_event = closest_video
                elif delta_intra <= INTRA_PLATE_ONLY_WINDOW:
                    verdict = VERDICT_HOLD
                    rule_triggered = "INTRA_SOURCE_TIER2_SHADOW"
                    reason = f"Duplikat plat sama dalam jendela ketat video (Δt={delta_intra:.1f}s <= {INTRA_PLATE_ONLY_WINDOW}s)"
                    false_positive_risk = "LOW" if delta_intra <= 5 else "MEDIUM"
                    delta_seconds = round(delta_intra, 2)
                    matched_event = closest_video

        return self._record_decision(
            camera_id, plate_norm, ts, source_type, object_type,
            verdict, rule_triggered, reason, false_positive_risk,
            event_key, track_id, source_name, matched_event, delta_seconds
        )

    def _query_plate_history(self, camera_id: int, plate_norm: str, lookback_seconds: int = 86400) -> List[Dict[str, Any]]:
        """Ambil riwayat deteksi plat tertentu pada kamera tertentu dari DB."""
        history = []
        try:
            with db.get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT fd.detection_id, fd.camera_id, fd.source_type, fd.source_name,
                               fd.track_id, fd.event_key, fd.created_at, p.plate_number,
                               p.normalized_plate_number
                        FROM full_detection fd
                        JOIN plate p ON p.plate_id = fd.plate_id
                        WHERE fd.camera_id = %s
                          AND (p.normalized_plate_number = %s OR p.plate_number = %s)
                          AND fd.created_at >= DATE_SUB(NOW(), INTERVAL %s SECOND)
                        ORDER BY fd.created_at ASC
                    """, (camera_id, plate_norm, plate_norm, lookback_seconds))
                    for r in cur.fetchall():
                        history.append({
                            "detection_id": r["detection_id"],
                            "camera_id": r["camera_id"],
                            "source_type": r.get("source_type") or "stream",
                            "source_name": r.get("source_name"),
                            "track_id": r.get("track_id"),
                            "event_key": r.get("event_key"),
                            "timestamp": parse_timestamp(r.get("created_at")),
                            "plate": normalize_plate(r.get("normalized_plate_number") or r.get("plate_number")),
                        })
        except Exception as exc:
            print(f"[SHADOW DEDUP] DB query error: {exc}")
        return history

    @staticmethod
    def _extract_identity(event_key: Optional[str], track_id: Optional[int]) -> Optional[str]:
        if event_key:
            return re.sub(r":(?:gen|g)\d+", "", str(event_key)).strip()
        if track_id is not None:
            return f"track:{track_id}"
        return None

    def _record_decision(
        self,
        camera_id: int,
        plate: str,
        timestamp: datetime,
        source_type: str,
        object_type: str,
        verdict: str,
        rule: str,
        reason: str,
        risk: str,
        event_key: Optional[str],
        track_id: Optional[int],
        source_name: Optional[str],
        matched_event: Optional[Dict[str, Any]],
        delta_seconds: Optional[float],
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": camera_id,
            "plate": plate or None,
            "source_type": source_type,
            "object_type": object_type,
            "verdict": verdict,
            "actual_action": "INSERT_EXECUTED",  # Selalu INSERT dalam mode shadow
            "rule_triggered": rule,
            "reason": reason,
            "false_positive_risk": risk,
            "event_key": event_key,
            "track_id": track_id,
            "source_name": source_name,
            "delta_seconds": delta_seconds,
            "matched_detection_id": matched_event.get("detection_id") if matched_event else None,
            "matched_source_type": matched_event.get("source_type") if matched_event else None,
            "matched_time": matched_event.get("timestamp").strftime("%Y-%m-%d %H:%M:%S") if matched_event and matched_event.get("timestamp") else None,
        }
        with self._audit_lock:
            self._audit_records.append(record)
            # Batasi memori hingga 1000 record terakhir
            if len(self._audit_records) > 1000:
                self._audit_records = self._audit_records[-1000:]
            self._save_audit_log()
        return record

    def get_audit_summary(self) -> Dict[str, Any]:
        """Kembalikan ringkasan audit log shadow deduplication."""
        with self._audit_lock:
            records = list(self._audit_records)

        total = len(records)
        keep_count = sum(1 for r in records if r["verdict"] == VERDICT_KEEP)
        hold_count = sum(1 for r in records if r["verdict"] == VERDICT_HOLD)
        rules_breakdown = {}
        for r in records:
            rule = r["rule_triggered"]
            rules_breakdown[rule] = rules_breakdown.get(rule, 0) + 1

        return {
            "mode": "SHADOW_DRY_RUN",
            "blocking_enabled": False,
            "total_evaluated": total,
            "shadow_verdicts": {
                "KEEP": keep_count,
                "HOLD": hold_count,
                "hold_rate_percent": round(hold_count * 100.0 / total, 1) if total else 0.0,
            },
            "actual_storage": {
                "INSERT_EXECUTED": total,
                "INSERT_BLOCKED": 0,
            },
            "rules_breakdown": rules_breakdown,
            "recent_records": records[-20:],
        }

    def clear_audit_log(self):
        with self._audit_lock:
            self._audit_records = []
            self._save_audit_log()
