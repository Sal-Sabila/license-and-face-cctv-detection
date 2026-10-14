import os
import re
import json
import hashlib
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

import db

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
        self._audit_index: Dict[str, int] = {}
        self._audit_lock = threading.Lock()
        self._last_history_error = None
        self._duplicate_audit_records_removed = 0
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
                        records = json.load(f)
                    if not isinstance(records, list):
                        raise ValueError("Format audit shadow harus berupa list JSON")

                    unique_records = []
                    seen_ids = set()
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        evaluation_id = record.get("evaluation_id")
                        if not evaluation_id:
                            evaluation_id = self._build_evaluation_id(
                                record.get("camera_id"),
                                record.get("source_type"),
                                record.get("object_type"),
                                record.get("event_key"),
                                record.get("track_id"),
                                record.get("event_time"),
                                record.get("plate"),
                                record.get("segment_filename"),
                                record.get("tracking_generation"),
                            )
                            record["evaluation_id"] = evaluation_id
                            record["audit_schema_version"] = 2
                            record.setdefault("event_origin", "legacy_unverified")
                            record.setdefault("event_is_live", False)
                            record.setdefault("database_insert_status", "NOT_OBSERVED")
                        if evaluation_id in seen_ids:
                            self._duplicate_audit_records_removed += 1
                            continue
                        seen_ids.add(evaluation_id)
                        unique_records.append(record)
                    self._audit_records = unique_records
                    if len(unique_records) != len(records):
                        self._save_audit_log()
                except Exception as exc:
                    print(f"[SHADOW DEDUP] Gagal memuat audit log: {exc}")
                    raise RuntimeError(
                        f"Audit shadow tidak dapat dimuat; penulisan audit dihentikan: {exc}"
                    ) from exc
            else:
                self._audit_records = []
            self._audit_index = {
                record["evaluation_id"]: index
                for index, record in enumerate(self._audit_records)
            }

    def _save_audit_log(self):
        parent_dir = os.path.dirname(self.audit_file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        temp_path = f"{self.audit_file_path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._audit_records, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.audit_file_path)
        except Exception as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise OSError(f"Gagal menyimpan audit log shadow: {exc}") from exc

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
        ts = parse_timestamp(timestamp)
        plate_norm = normalize_plate(plate_number)
        has_plate = bool(plate_norm and plate_norm not in ("-", "UNKNOWN") and len(plate_norm) >= 3)
        metadata = dict(metadata or {})
        event_origin = str(metadata.get("event_origin") or "unverified_candidate")
        event_is_live = event_origin in ("stream_pipeline", "segment_pipeline")
        metadata.setdefault("vehicle_type", vehicle_type)

        verdict = VERDICT_KEEP
        rule_triggered = "DEFAULT_KEEP"
        reason = "Event unik atau baru pertama kali terdeteksi"
        false_positive_risk = "LOW"
        matched_event = None
        delta_seconds = None
        history_error = None

        if ts is None:
            rule_triggered = "MISSING_EVENT_TIMESTAMP_KEEP"
            reason = "Timestamp event tidak tersedia/valid; pencocokan dinonaktifkan agar event tidak tertahan keliru"
            false_positive_risk = "UNKNOWN"
            return self._record_decision(
                camera_id, plate_norm, None, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds,
                metadata, event_origin, event_is_live, eval_time,
            )

        # ── ATURAN 1: Event stream selalu dipertahankan ───────────────────────
        if str(source_type).lower().strip() == "stream":
            verdict = VERDICT_KEEP
            rule_triggered = "STREAM_SOURCE_IMMUNE"
            reason = "Event stream realtime selalu dipertahankan tanpa perubahan"
            return self._record_decision(
                camera_id, plate_norm, ts, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds,
                metadata, event_origin, event_is_live, eval_time,
            )

        # ── ATURAN 2: Kendaraan tanpa plat nomor selalu dipertahankan ─────────
        if not has_plate:
            verdict = VERDICT_KEEP
            rule_triggered = "UNPLATED_VEHICLE_KEPT"
            reason = "Kendaraan tanpa plat terbaca tidak dibuang untuk mencegah false deduplication"
            return self._record_decision(
                camera_id, plate_norm, ts, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds,
                metadata, event_origin, event_is_live, eval_time,
            )

        # ── ATURAN 3: Periksa apakah plat sama melintas kembali (Repeat Pass) ──
        # Query event sebelumnya untuk plat dan kamera yang sama di DB
        history, history_error = self._query_plate_history(
            camera_id,
            plate_norm,
            lookback_seconds=86400,
            exclude_event_key=event_key,
        )

        if history_error:
            rule_triggered = "HISTORY_UNAVAILABLE_KEEP"
            reason = "Riwayat MySQL tidak dapat dibaca; kandidat dipertahankan dan evaluasi HOLD tidak dibuat"
            false_positive_risk = "UNKNOWN"
            return self._record_decision(
                camera_id, plate_norm, ts, source_type, object_type,
                verdict, rule_triggered, reason, false_positive_risk,
                event_key, track_id, source_name, matched_event, delta_seconds,
                metadata, event_origin, event_is_live, eval_time,
                history_error=history_error,
            )
        
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
                    event_key, track_id, source_name, matched_event, delta_seconds,
                    metadata, event_origin, event_is_live, eval_time,
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
                false_positive_risk = (
                    "LOW" if delta_cross <= 10
                    else "MEDIUM" if delta_cross <= 45
                    else "HIGH"
                )
                delta_seconds = round(delta_cross, 2)
                matched_event = closest_stream
                return self._record_decision(
                    camera_id, plate_norm, ts, source_type, object_type,
                    verdict, rule_triggered, reason, false_positive_risk,
                    event_key, track_id, source_name, matched_event, delta_seconds,
                    metadata, event_origin, event_is_live, eval_time,
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
            event_key, track_id, source_name, matched_event, delta_seconds,
            metadata, event_origin, event_is_live, eval_time,
        )

    def _query_plate_history(
        self,
        camera_id: int,
        plate_norm: str,
        lookback_seconds: int = 86400,
        exclude_event_key: Optional[str] = None,
    ):
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
                          AND (%s IS NULL OR fd.event_key IS NULL OR fd.event_key <> %s)
                        ORDER BY fd.created_at ASC
                    """, (
                        camera_id, plate_norm, plate_norm, lookback_seconds,
                        exclude_event_key, exclude_event_key,
                    ))
                    for r in cur.fetchall():
                        history_timestamp = parse_timestamp(r.get("created_at"))
                        if history_timestamp is None:
                            continue
                        history.append({
                            "detection_id": r["detection_id"],
                            "camera_id": r["camera_id"],
                            "source_type": r.get("source_type") or "stream",
                            "source_name": r.get("source_name"),
                            "track_id": r.get("track_id"),
                            "event_key": r.get("event_key"),
                            "timestamp": history_timestamp,
                            "plate": normalize_plate(r.get("normalized_plate_number") or r.get("plate_number")),
                        })
            self._last_history_error = None
        except Exception as exc:
            print(f"[SHADOW DEDUP] DB query error: {exc}")
            self._last_history_error = str(exc)
            return [], str(exc)
        return history, None

    @staticmethod
    def _extract_identity(event_key: Optional[str], track_id: Optional[int]) -> Optional[str]:
        if event_key:
            return str(event_key).strip()
        if track_id is not None:
            return f"track:{track_id}"
        return None

    def _record_decision(
        self,
        camera_id: int,
        plate: str,
        timestamp: Optional[datetime],
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
        metadata: Optional[Dict[str, Any]],
        event_origin: str,
        event_is_live: bool,
        evaluated_at: datetime,
        history_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        event_time = timestamp.isoformat(sep=" ", timespec="microseconds") if timestamp else None
        evaluation_id = self._build_evaluation_id(
            camera_id,
            source_type,
            object_type,
            event_key,
            track_id,
            None if (metadata or {}).get("stable_event_identity") else event_time,
            plate,
            (metadata or {}).get("segment_filename"),
            (metadata or {}).get("tracking_generation"),
        )
        event_identity = str(event_key).strip() if event_key else (
            f"camera:{camera_id}:source:{str(source_type).lower().strip()}:"
            f"object:{str(object_type).lower().strip()}:track:{track_id}:"
            f"event_time:{event_time or 'unknown'}"
        )
        record = {
            "evaluation_id": evaluation_id,
            "event_identity": event_identity,
            "audit_schema_version": 2,
            "evaluated_at": evaluated_at.isoformat(sep=" ", timespec="seconds"),
            "event_time": event_time,
            "camera_id": camera_id,
            "plate": plate or None,
            "source_type": source_type,
            "object_type": object_type,
            "vehicle_type": (metadata or {}).get("vehicle_type"),
            "verdict": verdict,
            "actual_action": (metadata or {}).get(
                "actual_action", "SHADOW_ONLY_NO_BLOCK"
            ),
            "database_insert_status": (metadata or {}).get(
                "database_insert_status", "NOT_OBSERVED"
            ),
            "blocking_enabled": bool((metadata or {}).get("blocking_enabled", False)),
            "rule_triggered": rule,
            "reason": reason,
            "false_positive_risk": risk,
            "event_key": event_key,
            "track_id": track_id,
            "source_name": source_name,
            "event_origin": event_origin,
            "event_is_live": event_is_live,
            "event_time_quality": (metadata or {}).get("event_time_quality", "provided"),
            "event_time_basis": (metadata or {}).get("event_time_basis"),
            "segment_filename": (metadata or {}).get("segment_filename"),
            "tracking_generation": (metadata or {}).get("tracking_generation"),
            "plate_track_id": (metadata or {}).get("plate_track_id"),
            "delta_seconds": delta_seconds,
            "matched_detection_id": matched_event.get("detection_id") if matched_event else None,
            "matched_camera_id": matched_event.get("camera_id") if matched_event else None,
            "matched_source_type": matched_event.get("source_type") if matched_event else None,
            "matched_time": matched_event.get("timestamp").isoformat(sep=" ", timespec="seconds") if matched_event and matched_event.get("timestamp") else None,
            "matched_event_key": matched_event.get("event_key") if matched_event else None,
            "matched_track_id": matched_event.get("track_id") if matched_event else None,
            "history_query_error": history_error,
        }
        with self._audit_lock:
            existing_index = self._audit_index.get(evaluation_id)
            if existing_index is not None:
                existing = self._audit_records[existing_index]
                enriched = dict(existing)
                enriched["evaluation_count"] = int(existing.get("evaluation_count", 1)) + 1
                enriched.setdefault("first_evaluated_at", existing.get("evaluated_at"))
                if (
                    existing.get("verdict") != record.get("verdict")
                    or existing.get("rule_triggered") != record.get("rule_triggered")
                ):
                    revisions = list(existing.get("prior_decisions", []))
                    revisions.append({
                        "evaluated_at": existing.get("evaluated_at"),
                        "verdict": existing.get("verdict"),
                        "rule_triggered": existing.get("rule_triggered"),
                        "reason": existing.get("reason"),
                    })
                    enriched["prior_decisions"] = revisions
                    enriched["decision_revision"] = int(existing.get("decision_revision", 0)) + 1
                enriched.update({
                    key: value for key, value in record.items()
                    if value not in (None, "", [], {})
                })
                enriched.setdefault("evaluation_count", 1)
                enriched.setdefault("first_evaluated_at", record["evaluated_at"])
                record = enriched
                self._audit_records[existing_index] = record
            else:
                record["evaluation_count"] = 1
                record["first_evaluated_at"] = record["evaluated_at"]
                record["decision_revision"] = 0
                self._audit_records.append(record)
                self._audit_index[evaluation_id] = len(self._audit_records) - 1
            self._save_audit_log()
        return record

    @staticmethod
    def _is_stable_video_identity(
        camera_id: int,
        track_id: Optional[int],
        event_key: Optional[str],
    ) -> bool:
        if track_id is None or not str(track_id).isdigit() or not event_key:
            return False
        pattern = (
            rf"vehicle:{re.escape(str(camera_id))}:{re.escape(str(track_id))}:"
            r"gen[1-9][0-9]*:sess_[A-Za-z0-9_-]+"
        )
        return re.fullmatch(pattern, str(event_key).strip()) is not None

    def evaluate_segment_insert(
        self,
        camera_id: int,
        plate_number: Optional[str],
        event_key: Optional[str],
        track_id: Optional[int],
        source_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        plate_is_valid: bool = False,
        blocking_enabled: bool = True,
        segment_filename: Optional[str] = None,
        tracking_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate only strong same-event video duplicates before segment storage."""
        evaluated_at = datetime.now()
        event_time = parse_timestamp(timestamp) or evaluated_at
        plate_norm = normalize_plate(plate_number)
        metadata = {
            "event_origin": "segment_pipeline",
            "event_time_quality": "storage_time",
            "event_time_basis": "video_segment_insert_attempt",
            "stable_event_identity": True,
            "blocking_enabled": blocking_enabled,
            "actual_action": "SEGMENT_INSERT_NOT_BLOCKED",
            "segment_filename": segment_filename,
            "tracking_generation": tracking_generation,
        }

        def keep(rule: str, reason: str, risk: str = "LOW") -> Dict[str, Any]:
            return self._record_decision(
                camera_id, plate_norm, event_time, "video", "vehicle",
                VERDICT_KEEP, rule, reason, risk,
                event_key, track_id, source_name, None, None,
                metadata, "segment_pipeline", True, evaluated_at,
            )

        if not blocking_enabled:
            return keep(
                "VIDEO_SEGMENT_DEDUP_DISABLED_KEEP",
                "Konfigurasi deduplikasi segmen nonaktif; penyimpanan diteruskan",
            )
        if not plate_is_valid or len(plate_norm) < 5:
            return keep(
                "INVALID_OR_MISSING_PLATE_KEEP",
                "Plat tidak valid/terbaca; event dipertahankan",
            )
        if not self._is_stable_video_identity(camera_id, track_id, event_key):
            return keep(
                "AMBIGUOUS_EVENT_IDENTITY_KEEP",
                "Identitas event/session tidak lengkap atau tidak konsisten; event dipertahankan",
                "UNKNOWN",
            )

        history, history_error = self._query_plate_history(
            camera_id,
            plate_norm,
            lookback_seconds=86400,
            exclude_event_key=None,
        )
        if history_error:
            metadata["history_query_error"] = history_error
            metadata["false_positive_risk"] = "UNKNOWN"
            metadata["actual_action"] = "DEDUP_ERROR_FAIL_OPEN"
            record = self._record_decision(
                camera_id, plate_norm, event_time, "video", "vehicle",
                VERDICT_KEEP, "HISTORY_UNAVAILABLE_KEEP",
                "Riwayat tidak tersedia; penyimpanan video diteruskan (fail-open)",
                "UNKNOWN", event_key, track_id, source_name, None, None,
                metadata, "segment_pipeline", True, evaluated_at,
                history_error=history_error,
            )
            return record

        for previous in history:
            if (
                str(previous.get("camera_id")) != str(camera_id)
                or str(previous.get("source_type", "")).lower() != "video"
                or str(previous.get("event_key") or "").strip()
                != str(event_key).strip()
                or str(previous.get("track_id")) != str(track_id)
                or normalize_plate(previous.get("plate")) != plate_norm
            ):
                continue
            previous_time = parse_timestamp(previous.get("timestamp"))
            if previous_time is None:
                continue
            gap = abs((event_time - previous_time).total_seconds())
            if gap >= REPEAT_PASS_GAP:
                return self._record_decision(
                    camera_id, plate_norm, event_time, "video", "vehicle",
                    VERDICT_KEEP, "REPEAT_PASS_GENUINE",
                    f"Event baru berjarak {gap:.1f}s; repeat pass tetap disimpan",
                    "LOW", event_key, track_id, source_name, previous,
                    round(gap, 2), metadata, "segment_pipeline", True, evaluated_at,
                )
            if gap <= INTRA_IDENTITY_WINDOW:
                metadata["actual_action"] = "SEGMENT_INSERT_BLOCKED"
                metadata["database_insert_status"] = "NOT_ATTEMPTED_DEDUP_HOLD"
                return self._record_decision(
                    camera_id, plate_norm, event_time, "video", "vehicle",
                    VERDICT_HOLD, "SAME_VIDEO_EVENT_DUPLICATE",
                    (
                        f"Plat {plate_norm}, kamera {camera_id}, track/session "
                        f"{event_key} sama dalam {gap:.1f}s "
                        f"(window <= {INTRA_IDENTITY_WINDOW:.0f}s)"
                    ),
                    "LOW", event_key, track_id, source_name, previous,
                    round(gap, 2), metadata, "segment_pipeline", True, evaluated_at,
                )

        return keep(
            "NO_STRONG_SAME_EVENT_DUPLICATE_KEEP",
            "Tidak ada bukti kuat untuk duplikasi event video; penyimpanan diteruskan",
        )

    def get_audit_summary(
        self,
        camera_id: Optional[int] = None,
        recent_limit: int = 20,
    ) -> Dict[str, Any]:
        """Kembalikan ringkasan audit log shadow deduplication."""
        with self._audit_lock:
            records = list(self._audit_records)
        if camera_id is not None:
            records = [
                record for record in records
                if str(record.get("camera_id")) == str(camera_id)
            ]

        total = len(records)
        keep_count = sum(1 for r in records if r["verdict"] == VERDICT_KEEP)
        hold_count = sum(1 for r in records if r["verdict"] == VERDICT_HOLD)
        live_records = [r for r in records if r.get("event_is_live") is True]
        live_hold_records = [r for r in live_records if r.get("verdict") == VERDICT_HOLD]
        synthetic_records = [r for r in records if r.get("event_is_live") is not True]
        configured_segment_blocking = os.environ.get(
            "VIDEO_SEGMENT_DEDUP_ENABLED", "true"
        ).strip().lower() in ("true", "1", "yes", "on")
        segment_blocking_seen = any(r.get("blocking_enabled") is True for r in records)
        segment_inserts_blocked = sum(
            1 for r in records
            if r.get("actual_action") == "SEGMENT_INSERT_BLOCKED"
        )
        rules_breakdown = {}
        for r in records:
            rule = r["rule_triggered"]
            rules_breakdown[rule] = rules_breakdown.get(rule, 0) + 1
        risk_breakdown = {
            level: sum(1 for record in records if record.get("false_positive_risk") == level)
            for level in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
        }
        live_hold_risk_breakdown = {
            level: sum(
                1 for record in live_hold_records
                if record.get("false_positive_risk") == level
            )
            for level in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
        }
        runtime = self._get_runtime_metrics(records)
        recent_limit = max(0, min(100, int(recent_limit)))

        return {
            "mode": (
                "LIMITED_VIDEO_SEGMENT_BLOCKING"
                if configured_segment_blocking
                else "SHADOW"
            ),
            "blocking_enabled": configured_segment_blocking,
            "segment_blocking": {
                "configured_enabled": configured_segment_blocking,
                "active_for_audited_records": segment_blocking_seen,
                "blocked_insert_count": segment_inserts_blocked,
                "cross_path_blocking_enabled": False,
                "rule": (
                    "valid plate + same camera + exact vehicle event/session + "
                    f"within {INTRA_IDENTITY_WINDOW:.0f}s"
                ),
            },
            "total_evaluated": total,
            "unique_decisions": len({r.get("evaluation_id") for r in records}),
            "duplicate_audit_records_removed_on_load": self._duplicate_audit_records_removed,
            "evidence_counts": {
                "live_pipeline_events": len(live_records),
                "live_keep": len(live_records) - len(live_hold_records),
                "live_hold": len(live_hold_records),
                "non_live_or_unverified": len(synthetic_records),
            },
            "shadow_verdicts": {
                "KEEP": keep_count,
                "HOLD": hold_count,
                "hold_rate_percent": round(hold_count * 100.0 / total, 1) if total else 0.0,
                "live_hold_count": len(live_hold_records),
                "live_hold_rate_percent": round(
                    len(live_hold_records) * 100.0 / len(live_records), 1
                ) if live_records else 0.0,
            },
            "actual_storage": {
                "SHADOW_BLOCKED": 0,
                "INSERT_BLOCKED": segment_inserts_blocked,
                "SEGMENT_INSERTS_BLOCKED": segment_inserts_blocked,
                "DB_INSERTS_OBSERVED": 0,
                "DB_INSERT_STATUS_UNOBSERVED": sum(
                    1 for r in records
                    if r.get("database_insert_status", "NOT_OBSERVED")
                    == "NOT_OBSERVED"
                ),
            },
            "rules_breakdown": rules_breakdown,
            "false_positive_risk": {
                "all_decisions": risk_breakdown,
                "live_hold": live_hold_risk_breakdown,
                "live_hold_unreviewed": sum(
                    1 for r in live_hold_records
                    if r.get("review_status", "UNREVIEWED") != "REVIEWED"
                ),
                "confirmed_false_positives": None,
                "assessment": (
                    "Belum ada label review ground-truth pada HOLD; angka ini menunjukkan "
                    "risiko yang diperkirakan aturan, bukan false-positive terkonfirmasi."
                ),
            },
            "runtime": runtime,
            "activation_readiness": {
                "ready_for_blocking": False,
                "limited_segment_blocking_active": configured_segment_blocking,
                "cross_path_blocking_enabled": False,
                "reason": (
                    "Blocking hanya aktif untuk duplikasi event video beridentitas sama. "
                    "Cross-path dan blocking umum tetap nonaktif sampai HOLD ditinjau "
                    "terhadap video/ground-truth, false-positive terukur, dan operasi disetujui."
                ),
            },
            "recent_records": records[-recent_limit:] if recent_limit else [],
        }

    @staticmethod
    def _build_evaluation_id(
        camera_id,
        source_type,
        object_type,
        event_key,
        track_id,
        event_time,
        plate,
        segment_filename=None,
        tracking_generation=None,
    ):
        segment_event = bool(segment_filename)
        stable_identity = {
            "camera_id": camera_id,
            "source_type": str(source_type or "").lower().strip(),
            "object_type": str(object_type or "").lower().strip(),
            "event_key": (
                None if segment_event
                else str(event_key or "").strip() or None
            ),
            "track_id": track_id,
            "event_time": event_time,
            "plate": normalize_plate(plate) or None,
            "segment_filename": segment_filename,
            "tracking_generation": tracking_generation,
        }
        if segment_event and track_id is not None:
            stable_identity["event_time"] = None
            stable_identity["plate"] = None
        payload = json.dumps(stable_identity, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_runtime_metrics(records):
        metrics = {
            "cpu_system_percent": None,
            "cpu_process_percent": None,
            "memory_process_mb": None,
            "recorder": None,
            "realtime": None,
            "metrics_error": None,
        }
        try:
            import psutil

            metrics["cpu_system_percent"] = psutil.cpu_percent(interval=None)
            process = psutil.Process(os.getpid())
            metrics["cpu_process_percent"] = process.cpu_percent(interval=None)
            metrics["memory_process_mb"] = round(
                process.memory_info().rss / (1024 * 1024), 1
            )
        except Exception as exc:
            metrics["metrics_error"] = f"CPU/RAM unavailable: {exc}"

        camera_ids = sorted({
            record.get("camera_id")
            for record in records
            if record.get("camera_id") is not None
        })
        active_recorder = None
        try:
            from services.camera_recorder_service import CameraRecorderService
            recorder = CameraRecorderService.get_instance()
            recorder_status = recorder.get_status()
            active_recorder = recorder_status
            metrics["recorder"] = {
                "status": recorder_status.get("status"),
                "is_running": recorder_status.get("is_running"),
                "camera_id": recorder_status.get("camera_id"),
                "backlog": recorder_status.get("backlog"),
            }
        except Exception as exc:
            metrics["metrics_error"] = (
                f"{metrics['metrics_error']}; recorder unavailable: {exc}"
                if metrics["metrics_error"] else f"Recorder unavailable: {exc}"
            )

        try:
            from services.camera_worker_manager import CameraWorkerManager
            manager = CameraWorkerManager.get_instance()
            if active_recorder and active_recorder.get("camera_id") is not None:
                camera_ids = sorted(set(camera_ids + [active_recorder["camera_id"]]))
            metrics["realtime"] = {
                str(camera_id): (
                    manager.get_worker(camera_id).get_status()
                    if manager.get_worker(camera_id) is not None
                    else {"status": "NOT_RUNNING"}
                )
                for camera_id in camera_ids
            }
            if not camera_ids:
                metrics["realtime"] = {"status": "NO_AUDITED_CAMERA"}
        except Exception as exc:
            metrics["metrics_error"] = (
                f"{metrics['metrics_error']}; realtime unavailable: {exc}"
                if metrics["metrics_error"] else f"Realtime unavailable: {exc}"
            )
        return metrics

    def clear_audit_log(self):
        with self._audit_lock:
            self._audit_records = []
            self._audit_index = {}
            self._save_audit_log()
