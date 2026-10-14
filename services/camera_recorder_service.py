import os
import re
import csv
import json
import time
import queue
import shutil
import subprocess
import threading
from datetime import datetime, timedelta

from ffmpeg_stream_reader import find_ffmpeg_executable, normalize_stream_url
import db

# Root directory untuk penyimpanan rekaman segmen
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEGMENTS_DIR = os.path.join(BASE_DIR, "videos", "segments")
_OMIT_JSON_VALUE = object()
_IMAGE_STATE_KEYS = {
    "crop",
    "plate_crop",
    "vehicle_crop",
    "face_crop",
    "frame",
    "image",
    "image_data",
}


def _is_numpy_array(value):
    return (
        type(value).__module__.split(".", 1)[0] == "numpy"
        and getattr(value, "ndim", 0) > 0
        and hasattr(value, "shape")
    )


def _json_safe_state_value(value, omitted_arrays):
    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy":
        if getattr(value, "ndim", None) == 0:
            return value.item()
        if hasattr(value, "shape"):
            if getattr(value, "size", 0) <= 256:
                return _json_safe_state_value(value.tolist(), omitted_arrays)
            omitted_arrays[0] += 1
            return _OMIT_JSON_VALUE

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if (
                isinstance(key, str)
                and key.lower() in _IMAGE_STATE_KEYS
                and _is_numpy_array(item)
            ):
                omitted_arrays[0] += 1
                continue
            safe_key = _json_safe_state_value(key, omitted_arrays)
            safe_item = _json_safe_state_value(item, omitted_arrays)
            if safe_key is _OMIT_JSON_VALUE or safe_item is _OMIT_JSON_VALUE:
                continue
            if not isinstance(safe_key, (str, int, float, bool)):
                raise TypeError(
                    f"Segment state contains unsupported JSON key: "
                    f"{type(safe_key).__name__}"
                )
            result[safe_key] = safe_item
        return result

    if isinstance(value, (list, tuple, set)):
        items = value if not isinstance(value, set) else sorted(value, key=repr)
        result = []
        for item in items:
            safe_item = _json_safe_state_value(item, omitted_arrays)
            if safe_item is not _OMIT_JSON_VALUE:
                result.append(safe_item)
        return result

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Segment state contains unsupported JSON value: "
        f"{type(value).__name__}"
    )


class CameraRecorderService:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, segments_dir=None, segment_duration=10):
        self.segments_dir = segments_dir or DEFAULT_SEGMENTS_DIR
        self.segment_duration = max(5, int(segment_duration))
        self.ffmpeg_path = find_ffmpeg_executable()

        self.camera_id = None
        self.camera_name = None
        self.stream_url = None

        self.process = None
        self.process_lock = threading.Lock()

        self.is_running = False
        self.status = "STOPPED"  # STOPPED, CONNECTING, RECORDING, ERROR
        self.last_error = None
        self.last_ffmpeg_stderr = []
        self._stderr_lock = threading.Lock()
        self.started_at = None

        # Threading & sinkronisasi perekaman
        self.supervisor_thread = None
        self.monitor_thread = None
        self.stderr_thread = None
        self.stop_event = threading.Event()
        self.worker_idle_event = threading.Event()
        self.worker_idle_event.set()

        # Pelacakan direktori & file sesi
        self.session_id = None
        self.camera_dir = None
        self.list_file = None
        self.state_file = None

        # Tahap 3: Antrean bounded & Single Worker AI
        self.max_queue_size = int(os.environ.get("HYBRID_MAX_QUEUE_SIZE", "50"))
        self.auto_process_enabled = os.environ.get("ENABLE_AUTO_PROCESS", "true").lower() in ("true", "1", "yes")
        self._processing_queue = queue.Queue(maxsize=self.max_queue_size)
        self._worker_thread = None

        # State storage (filename -> metadata dict)
        self._segments_state = {}
        self._completed_segments = []
        self._completed_segments_lock = threading.RLock()
        self._processed_segment_filenames = set()
        self._queued_segment_filenames = set()
        self._loaded_state_files = set()
        self._unvalidated_segment_rows = {}
        self._callbacks = []

        # Status pemantauan worker
        self.worker_is_busy = False
        self.current_processing_segment = None
        self.latest_completed_segment = None
        self.latest_failed_segment = None
        self.latest_processed_segment = None

    # ------------------------------------------------------------------
    # MANAJEMEN SIKLUS HIDUP (LIFECYCLE)
    # ------------------------------------------------------------------

    def start(self, camera_id=None):
        """
        Memulai perekaman otomatis dan single processing worker.
        Jika camera_id berbeda dari target aktif, recorder di-restart ke kamera baru.
        """
        if camera_id is not None:
            try:
                camera_id = int(camera_id)
            except (TypeError, ValueError):
                self.status = "ERROR"
                self.last_error = f"ID kamera tidak valid: {camera_id!r}"
                return False
            if camera_id <= 0:
                self.status = "ERROR"
                self.last_error = f"ID kamera tidak valid: {camera_id}"
                return False

        if self.is_running:
            if camera_id is not None and self.camera_id != camera_id:
                print(f"[RECORDER] Mengganti target kamera ke {camera_id}, me-restart recorder...")
                self.stop()
                self.camera_id = None
                self.stream_url = None
            else:
                print("[RECORDER] Layanan perekaman sudah berjalan")
                return True

        self.stop_event.clear()
        self.is_running = True
        self.status = "CONNECTING"
        self.last_error = None

        # Restore a known camera's disk backlog before DB/FFmpeg startup. This
        # lets already-recorded segments drain even if the live stream is down.
        if camera_id is not None:
            self.camera_id = camera_id
            self.camera_dir = os.path.join(self.segments_dir, f"cam_{camera_id}")
            self.state_file = os.path.join(
                self.camera_dir,
                "segment_queue_state.json",
            )
            self._load_state()

        # 1. Jalankan supervisor perekam FFmpeg
        self.supervisor_thread = threading.Thread(
            target=self._run_supervisor,
            args=(camera_id,),
            name="CameraRecorderSupervisor",
            daemon=True,
        )
        self.supervisor_thread.start()
        print("[RECORDER] Supervisor perekaman otomatis dimulai")

        # 2. Jalankan single processing worker
        if self.auto_process_enabled and (self._worker_thread is None or not self._worker_thread.is_alive()):
            self._worker_thread = threading.Thread(
                target=self._run_processing_worker,
                name="SegmentAIWorker",
                daemon=True,
            )
            self._worker_thread.start()
            print(
                "[RECORDER] Worker antrean pemrosesan tunggal dimulai "
                f"(alive={self._worker_thread.is_alive()}, "
                f"auto_process={self.auto_process_enabled})"
            )

        return True

    def stop(self):
        """Menghentikan perekaman dan membersihkan worker background."""
        print("[RECORDER] Menghentikan perekaman dan worker...")
        self.is_running = False
        self.status = "STOPPING"
        self.stop_event.set()

        self._terminate_ffmpeg()

        if self.supervisor_thread and self.supervisor_thread.is_alive():
            self.supervisor_thread.join(timeout=3.0)

        if self._worker_thread and self._worker_thread.is_alive():
            if self.worker_is_busy:
                drain_timeout = max(60.0, self.segment_duration * 6.0)
                print(
                    f"[RECORDER] Menunggu segment aktif selesai "
                    f"({self.current_processing_segment}); timeout={drain_timeout:.0f}s"
                )
                self.worker_idle_event.wait(timeout=drain_timeout)
            self._worker_thread.join(timeout=3.0)

        worker_alive = bool(self._worker_thread and self._worker_thread.is_alive())
        supervisor_alive = bool(
            self.supervisor_thread and self.supervisor_thread.is_alive()
        )
        if worker_alive or supervisor_alive:
            self.status = "STOPPING"
            active_segment = self.current_processing_segment or "-"
            print(
                "[RECORDER STOP WARNING] Shutdown belum tuntas: "
                f"worker_alive={worker_alive}, supervisor_alive={supervisor_alive}, "
                f"segment={active_segment}"
            )
        else:
            self.status = "STOPPED"
            print("[RECORDER] Perekaman dan worker berhasil dihentikan")

    def register_completed_callback(self, callback):
        """Mendaftarkan callback fungsi saat ada segmen selesai diproses."""
        if callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def get_completed_segments(self, status=None):
        """
        Mengambil salinan daftar segmen.
        Dapat difilter berdasarkan status ('pending', 'processing', 'completed', 'failed').
        """
        with self._completed_segments_lock:
            all_segs = list(self._segments_state.values())
            if status:
                st = str(status).lower().strip()
                return [s for s in all_segs if str(s.get("status", "")).lower() == st]
            return all_segs

    def pop_next_completed_segment(self, timeout=None):
        """Ambil segmen selesai dari queue lama jika dibutuhkan."""
        try:
            return self._processing_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self):
        """Mengembalikan status komprehensif recorder, worker, dan backlog antrean."""
        with self.process_lock:
            proc_alive = self.process is not None and self.process.poll() is None

        with self._completed_segments_lock:
            total_recorded = len(self._segments_state)
            count_pending = sum(1 for s in self._segments_state.values() if s.get("status") == "pending")
            count_processing = sum(1 for s in self._segments_state.values() if s.get("status") == "processing")
            count_completed = sum(1 for s in self._segments_state.values() if s.get("status") == "completed")
            count_failed = sum(1 for s in self._segments_state.values() if s.get("status") == "failed")
            count_stored = sum(1 for s in self._segments_state.values() if s.get("status") == "stored")
            latest_seg = self._completed_segments[-1] if self._completed_segments else None
            worker_is_busy = self.worker_is_busy
            current_processing_segment = self.current_processing_segment

        uptime_seconds = 0
        if self.started_at and self.is_running:
            uptime_seconds = round(time.time() - self.started_at, 1)

        backlog = {
            "total_recorded": total_recorded,
            "pending": count_pending,
            "processing": count_processing,
            "completed": count_completed,
            "failed": count_failed,
            "stored": count_stored,
            "queue_size": self._processing_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "worker_is_busy": worker_is_busy,
            "worker_thread_alive": bool(
                self._worker_thread and self._worker_thread.is_alive()
            ),
            "queued_segments": self._processing_queue.qsize(),
            "current_processing_segment": current_processing_segment,
        }

        return {
            "status": self.status,
            "is_running": self.is_running,
            "process_active": proc_alive,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "stream_url": self.stream_url,
            "segment_duration": self.segment_duration,
            "auto_process_enabled": self.auto_process_enabled,
            "backlog": backlog,
            "latest_segment": latest_seg,
            "latest_processed_segment": self.latest_processed_segment,
            "latest_completed_segment": self.latest_completed_segment,
            "latest_failed_segment": self.latest_failed_segment,
            "uptime_seconds": uptime_seconds,
            "last_error": self.last_error,
            "ffmpeg_stderr": self._get_ffmpeg_stderr(),
        }

    def _log_segment_progress(self, event, filename=None, elapsed_seconds=None, error=None):
        with self._completed_segments_lock:
            counts = {
                status: sum(
                    1
                    for segment in self._segments_state.values()
                    if segment.get("status") == status
                )
                for status in ("pending", "processing", "completed", "failed", "stored")
            }
            queue_size = self._processing_queue.qsize()

        fields = [
            f"[SEGMENT PROGRESS] event={event}",
            f"segment={filename or '-'}",
            *(f"{status}={count}" for status, count in counts.items()),
            f"queue={queue_size}/{self.max_queue_size}",
        ]
        if elapsed_seconds is not None:
            fields.append(f"elapsed={elapsed_seconds:.2f}s")
        if error:
            fields.append(f"error={error}")
        print(" | ".join(fields))

    def _get_ffmpeg_stderr(self):
        with self._stderr_lock:
            return list(self.last_ffmpeg_stderr)

    # ------------------------------------------------------------------
    # PERSISTENSI STATE (JSON PADA DISK)
    # ------------------------------------------------------------------

    def _load_state(self):
        """
        Memuat riwayat segmen dari file JSON saat aplikasi start / restart.
        Menjamin tidak ada pemrosesan ganda untuk segmen yang sudah 'completed'.
        """
        if not self.state_file or not os.path.isfile(self.state_file):
            return

        state_path = os.path.abspath(self.state_file)
        if state_path in self._loaded_state_files:
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                loaded_count = 0
                requeued_count = 0
                pending_segments = []
                with self._completed_segments_lock:
                    for filename, info in data.items():
                        if not isinstance(info, dict):
                            print(
                                f"[RECORDER STATE WARNING] Metadata segmen '{filename}' "
                                "bukan object; dilewati"
                            )
                            continue
                        fpath = info.get("file_path", "")
                        file_exists = os.path.isfile(fpath)
                        # Completed entries remain authoritative after their
                        # MP4 has been removed, preventing CSV replay on restart.
                        if file_exists or info.get("status") == "completed":
                            self._processed_segment_filenames.add(filename)
                            if (
                                info.get("status") == "processing"
                                and filename != self.current_processing_segment
                            ):
                                info["status"] = "pending"
                                info.pop("error", None)
                                info["recovered_at"] = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                print(
                                    f"[RECORDER STATE RECOVERY] Segmen '{filename}' "
                                    "dipulihkan dari processing menjadi pending"
                                )
                            if info.get("status") == "pending":
                                info.pop("completed_at", None)

                            self._segments_state[filename] = info
                            self._completed_segments.append(info)
                            loaded_count += 1

                            if (
                                info.get("status") in ("pending", "stored")
                                and self.auto_process_enabled
                                and file_exists
                            ):
                                pending_segments.append(info)

                    self._loaded_state_files.add(state_path)
                    self._save_state()

                for info in pending_segments:
                    if self._enqueue_segment(info):
                        requeued_count += 1

                print(
                    f"[RECORDER STATE] Berhasil memuat {loaded_count} segmen dari {os.path.basename(self.state_file)} "
                    f"({requeued_count} pending masuk antrean; "
                    f"backlog tersisa={max(0, len(pending_segments) - requeued_count)})"
                )
        except Exception as exc:
            print(f"[RECORDER STATE ERROR] Gagal memuat state: {exc}")

    def _save_state(self):
        """Menyimpan status antrean segmen ke file JSON secara atomik."""
        if not self.state_file:
            return False

        try:
            tmp_file = f"{self.state_file}.tmp"
            omitted_arrays = [0]
            serializable_state = _json_safe_state_value(
                self._segments_state,
                omitted_arrays,
            )
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(serializable_state, f, indent=2, allow_nan=False)
            os.replace(tmp_file, self.state_file)
            if omitted_arrays[0]:
                print(
                    f"[RECORDER STATE] {omitted_arrays[0]} image array(s) "
                    "excluded from persisted metadata; retained in memory"
                )
            return True
        except Exception as exc:
            print(f"[RECORDER STATE ERROR] Gagal menyimpan state: {exc}")
            return False

    def _cleanup_completed_segment(self, segment, status_persisted):
        """Delete only a persisted-completed MP4 located under the segment root."""
        filename = segment.get("filename")
        file_path = segment.get("file_path")
        if not status_persisted:
            print(
                f"[RECORDER CLEANUP SKIP] Segment '{filename}' dipertahankan: "
                "status completed belum berhasil disimpan"
            )
            return False

        if not filename or not file_path or os.path.basename(file_path) != filename:
            print(
                f"[RECORDER CLEANUP ERROR] Metadata path tidak valid untuk "
                f"segment '{filename}'; file dipertahankan"
            )
            return False

        root_path = os.path.realpath(self.segments_dir)
        resolved_path = os.path.realpath(file_path)
        try:
            inside_segment_root = (
                os.path.commonpath((root_path, resolved_path)) == root_path
            )
        except ValueError:
            inside_segment_root = False

        if (
            not inside_segment_root
            or resolved_path == root_path
            or os.path.splitext(filename)[1].lower() != ".mp4"
        ):
            print(
                f"[RECORDER CLEANUP ERROR] Path di luar cakupan segment MP4: "
                f"'{file_path}'; file dipertahankan"
            )
            return False

        with self._completed_segments_lock:
            current_state = self._segments_state.get(filename)
            if not current_state or current_state.get("status") != "completed":
                print(
                    f"[RECORDER CLEANUP SKIP] Segment '{filename}' berstatus "
                    f"{(current_state or {}).get('status', 'missing')}; file dipertahankan"
                )
                return False

            if not os.path.isfile(resolved_path):
                print(
                    f"[RECORDER CLEANUP SKIP] File segment '{filename}' "
                    "sudah tidak ada"
                )
                return False

            try:
                file_size = os.path.getsize(resolved_path)
                os.remove(resolved_path)
            except OSError as exc:
                print(
                    f"[RECORDER CLEANUP ERROR] Gagal menghapus segment "
                    f"'{filename}': {exc}"
                )
                return False

            current_state["video_file_deleted"] = True
            current_state["video_file_deleted_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cleanup_state_saved = self._save_state()
            if cleanup_state_saved:
                print(
                    f"[RECORDER CLEANUP] Segment '{filename}' dihapus "
                    f"({file_size} bytes); status=completed"
                )
            else:
                print(
                    f"[RECORDER CLEANUP WARNING] Segment '{filename}' sudah "
                    "dihapus setelah status completed tersimpan, tetapi metadata "
                    "penghapusan gagal disimpan"
                )
            return True

    # ------------------------------------------------------------------
    # TAHAP 3: SINGLE WORKER CONSUMER LOOP (INFERENSI SERIAL)
    # ------------------------------------------------------------------

    def _run_processing_worker(self):
        """
        Background worker tunggal yang memproses antrean segmen secara serial.
        Inferensi AI tidak pernah berjalan paralel antar-segmen.
        """
        print("[QUEUE WORKER] Worker pemrosesan segmen tunggal aktif")

        while self.is_running and not self.stop_event.is_set():
            try:
                # Ambil segmen berikutnya dari bounded queue dengan timeout
                segment_info = self._processing_queue.get(timeout=1.0)
            except queue.Empty:
                # Periksa apakah ada pending backlog di _segments_state yang belum masuk queue
                self._repopulate_queue_from_backlog()
                continue

            filename = segment_info.get("filename")
            if not filename:
                print("[QUEUE WORKER WARNING] Item antrean tanpa filename dilewati")
                self._processing_queue.task_done()
                continue

            claimed_state = None
            with self._completed_segments_lock:
                self._queued_segment_filenames.discard(filename)
                current_state = self._segments_state.get(filename)
                if not self.is_running or self.stop_event.is_set():
                    print(
                        f"[QUEUE WORKER] Segmen '{filename}' dikembalikan ke backlog "
                        "karena worker sedang berhenti"
                    )
                elif current_state is None:
                    print(
                        f"[QUEUE WORKER WARNING] Segmen '{filename}' tidak ada di state; "
                        "item antrean dilewati"
                    )
                elif current_state.get("status") not in ("pending", "stored"):
                    print(
                        f"[QUEUE WORKER] Segmen '{filename}' dilewati "
                        f"(status={current_state.get('status')!r}, bukan pending)"
                    )
                else:
                    current_state["status"] = "processing"
                    current_state["started_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    current_state.pop("completed_at", None)
                    try:
                        previous_attempts = int(current_state.get("attempts", 0) or 0)
                    except (TypeError, ValueError):
                        previous_attempts = 0
                        print(
                            f"[QUEUE WORKER WARNING] attempts rusak pada '{filename}'; "
                            "menginisialisasi ulang counter"
                        )
                    current_state["attempts"] = previous_attempts + 1
                    self._segments_state[filename] = current_state
                    self.worker_is_busy = True
                    self.current_processing_segment = filename
                    self.worker_idle_event.clear()
                    self._save_state()
                    claimed_state = current_state

            if claimed_state is None:
                self._processing_queue.task_done()
                if not self.is_running or self.stop_event.is_set():
                    break
                continue

            segment_info = claimed_state
            queued_at = segment_info.get("queued_at")
            queue_wait_seconds = None
            if queued_at:
                try:
                    queue_wait_seconds = max(
                        0.0,
                        (
                            datetime.now()
                            - datetime.fromisoformat(queued_at)
                        ).total_seconds(),
                    )
                except (TypeError, ValueError):
                    print(
                        f"[QUEUE WORKER WARNING] queued_at invalid untuk '{filename}': "
                        f"{queued_at!r}"
                    )
            if queue_wait_seconds is not None:
                segment_info["queue_wait_seconds"] = round(queue_wait_seconds, 2)
            print(
                f"[QUEUE WORKER] Segmen '{filename}' diklaim; "
                f"AI mulai (attempt={claimed_state['attempts']}, "
                f"queue_remaining={self._processing_queue.qsize()})"
            )
            self._log_segment_progress("processing", filename)
            try:
                result = self._execute_segment_ai(segment_info)
                detection_summary = result.get("detections") or {}
                persistence_errors = detection_summary.get("persistence_errors") or []
                elapsed_seconds = result.get("elapsed_seconds")

                with self._completed_segments_lock:
                    claimed_state["detections"] = detection_summary
                    claimed_state["cpu_usage"] = result.get("cpu_usage")
                    claimed_state["elapsed_seconds"] = elapsed_seconds
                    claimed_state["processing_ratio"] = result.get("processing_ratio")
                    if persistence_errors:
                        claimed_state["status"] = "failed"
                        claimed_state["failed_at"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        claimed_state["error"] = (
                            f"{len(persistence_errors)} detection persistence error(s)"
                        )
                        self.latest_failed_segment = dict(claimed_state)
                    else:
                        claimed_state["status"] = "completed"
                        claimed_state["completed_at"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        claimed_state.pop("error", None)
                        self.latest_completed_segment = dict(claimed_state)
                        self.latest_processed_segment = dict(claimed_state)
                    state_persisted = self._save_state()

                if persistence_errors:
                    print(
                        f"[QUEUE WORKER ERROR] Segmen '{filename}' inferensi selesai "
                        f"tetapi penyimpanan {len(persistence_errors)} event gagal; "
                        "status=failed"
                    )
                else:
                    self._notify_completed(claimed_state)
                    self._cleanup_completed_segment(
                        claimed_state,
                        status_persisted=state_persisted,
                    )
                    print(
                        f"[QUEUE WORKER] Segmen '{filename}' selesai "
                        f"({elapsed_seconds}s, "
                        f"vehicles={detection_summary.get('event_groups', {}).get('vehicle', 0)}, "
                        f"plates={detection_summary.get('plate_captures', 0)}, "
                        f"queue_remaining={self._processing_queue.qsize()})"
                    )
                self._log_segment_progress(
                    "failed" if persistence_errors else "completed",
                    filename,
                    elapsed_seconds=elapsed_seconds,
                    error=(
                        f"{len(persistence_errors)} event(s) gagal disimpan"
                        if persistence_errors
                        else None
                    ),
                )

            except Exception as exc:
                # Catat kegagalan tanpa menghentikan worker atau recorder
                with self._completed_segments_lock:
                    claimed_state["status"] = "failed"
                    claimed_state["failed_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    claimed_state["error"] = str(exc)
                    self.latest_failed_segment = dict(claimed_state)
                    self._save_state()

                print(f"[QUEUE WORKER ERROR] Segmen '{filename}' gagal diproses: {exc}")
                self._log_segment_progress("failed", filename, error=str(exc))

            finally:
                with self._completed_segments_lock:
                    self.worker_is_busy = False
                    self.current_processing_segment = None
                    self.worker_idle_event.set()
                self._processing_queue.task_done()
                # Pump persistent backlog whenever a slot opens, even if the
                # queue never becomes empty while live recording continues.
                self._repopulate_queue_from_backlog()
                # Jeda kooperatif singkat agar CPU tidak 100% saturasi dan jalur realtime lancar
                time.sleep(0.3)

        print("[QUEUE WORKER] Worker pemrosesan segmen dihentikan")

    def _repopulate_queue_from_backlog(self):
        """Memasukkan segmen 'pending' atau 'stored' dari backlog ke antrean jika ada kapasitas kosong."""
        if not self.auto_process_enabled or self._processing_queue.full():
            return

        with self._completed_segments_lock:
            pending_segments = [
                info
                for info in self._segments_state.values()
                if info.get("status") in ("pending", "stored")
                and info.get("filename") not in self._queued_segment_filenames
            ]
        for info in pending_segments:
            if not self._enqueue_segment(info):
                if self._processing_queue.full():
                    break

    def _enqueue_segment(self, segment_info):
        """Memasukkan segmen ke bounded queue sesuai kebijakan aman backlog (tanpa menghapus file)."""
        filename = segment_info.get("filename")
        if not filename:
            print("[RECORDER QUEUE ERROR] Segmen tanpa filename tidak dapat diantrekan")
            return False

        with self._completed_segments_lock:
            current_state = self._segments_state.get(filename, segment_info)
            if current_state.get("status") not in ("pending", "stored"):
                print(
                    f"[RECORDER QUEUE] Segmen '{filename}' tidak diantrekan "
                    f"(status={current_state.get('status')!r})"
                )
                return False
            if filename in self._queued_segment_filenames:
                print(f"[RECORDER QUEUE] Segmen '{filename}' sudah ada di antrean")
                return True

            try:
                self._processing_queue.put_nowait(current_state)
            except queue.Full:
                current_state["status"] = "stored"
                current_state["note"] = (
                    "Tersimpan aman di disk (kebijakan batas antrean)"
                )
                self._segments_state[filename] = current_state
                self._save_state()
                print(
                    f"[RECORDER QUEUE BACKLOG] Antrean penuh "
                    f"({self.max_queue_size}); segmen '{filename}' tetap stored"
                )
                self._log_segment_progress("stored", filename)
                return False

            current_state["status"] = "pending"
            current_state["queued_at"] = datetime.now().isoformat(timespec="seconds")
            current_state.pop("note", None)
            self._segments_state[filename] = current_state
            self._queued_segment_filenames.add(filename)
            self._save_state()
            print(
                f"[RECORDER QUEUE] Segmen '{filename}' diantrekan "
                f"({self._processing_queue.qsize()}/{self.max_queue_size})"
            )
            self._log_segment_progress("pending", filename)
            return True

    def _execute_segment_ai(self, segment_info, save_annotated=False):
        """
        Menjalankan VideoAIService pada segmen video tunggal dan mengukur metrik CPU.
        """
        import psutil
        from services.video_ai_service import VideoAIService

        file_path = segment_info["file_path"]
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            raise FileNotFoundError(f"File segmen '{file_path}' tidak ditemukan atau kosong di disk.")
        print(
            f"[SEGMENT AI] Validasi input berhasil: {segment_info.get('filename')} "
            f"({os.path.getsize(file_path)} bytes)"
        )

        output_path = None
        if save_annotated or os.environ.get("HYBRID_SAVE_ANNOTATED", "false").lower() in ("true", "1", "yes"):
            processed_dir = os.path.join(BASE_DIR, "videos", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            output_path = os.path.join(processed_dir, f"annotated_{segment_info['filename']}")

        process_obj = psutil.Process()
        psutil.cpu_percent(interval=None)
        process_obj.cpu_percent(interval=None)
        t_start = time.perf_counter()

        # Jalankan VideoAIService bawaan proyek tanpa memodifikasi pipelinenya
        ai_service = VideoAIService(
            video_path=file_path,
            output_path=output_path,
            camera_id=segment_info.get("camera_id") or self.camera_id or 1,
            show_window=False,
            source_name=f"Segment Cam {segment_info.get('camera_id') or self.camera_id}",
            segment_processing=True,
            segment_filename=segment_info.get("filename"),
        )
        print(
            f"[SEGMENT AI] Menjalankan VideoAIService: "
            f"camera={segment_info.get('camera_id') or self.camera_id}, "
            f"segment={segment_info.get('filename')}"
        )
        ai_service.run()
        print(
            f"[SEGMENT AI] Inferensi selesai: segment={segment_info.get('filename')}, "
            f"frames={ai_service.inference_count}, "
            f"vehicles={len(ai_service.video_event_groups.get('vehicle', []))}, "
            f"plates={ai_service.plate_capture_count}"
        )

        t_elapsed = time.perf_counter() - t_start
        sys_cpu = psutil.cpu_percent(interval=None)
        proc_cpu = process_obj.cpu_percent(interval=None)

        # Evaluasi shadow pasca-inferensi; tidak memblokir alur simpan yang sudah berjalan.
        shadow_evaluations = []
        try:
            from services.shadow_dedup_service import ShadowDedupService
            shadow_svc = ShadowDedupService.get_instance()
            cam_id = segment_info.get("camera_id") or self.camera_id or 1
            segment_end = self._segment_recorded_at(segment_info)
            segment_duration = max(0.0, float(segment_info.get("duration") or 0.0))
            vehicle_groups = ai_service.video_event_groups.get("vehicle", [])
            plate_history = list(ai_service.last_plate_history)
            pre_audited_event_keys = {
                evaluation.get("event_key")
                for evaluation in getattr(ai_service, "segment_dedup_evaluations", [])
                if evaluation.get("event_key")
            }

            def _event_timestamp(video_time):
                if segment_end is None:
                    return None, "unavailable"
                if video_time is None:
                    return segment_end - timedelta(seconds=segment_duration / 2.0), "segment_midpoint_estimate"
                try:
                    seconds_before_end = max(
                        0.0,
                        segment_duration - float(video_time),
                    )
                    return segment_end - timedelta(seconds=seconds_before_end), "video_time_estimate"
                except (TypeError, ValueError):
                    return segment_end - timedelta(seconds=segment_duration / 2.0), "segment_midpoint_estimate"

            plate_vehicle_ids = {
                str(event.get("vehicle_track_id"))
                for event in plate_history
                if event.get("vehicle_track_id") is not None
            }

            # Log hanya event kendaraan yang tidak sudah direpresentasikan oleh OCR pelat.
            for vevent in vehicle_groups:
                event_track_id = vevent.get("event_track_id")
                if event_track_id is None:
                    event_track_id = vevent.get("track_id")
                if event_track_id is not None and str(event_track_id) in plate_vehicle_ids:
                    continue
                generation = vevent.get("generation") or vevent.get("_event_generation") or 1
                event_key = ai_service._build_event_key(
                    "vehicle",
                    event_track_id,
                    generation,
                ) if event_track_id is not None else None
                event_time, time_quality = _event_timestamp(vevent.get("last_seen"))
                s_eval = shadow_svc.evaluate_candidate(
                    camera_id=cam_id,
                    plate_number=vevent.get("plate_number") or vevent.get("plate"),
                    timestamp=event_time,
                    source_type="video",
                    object_type="vehicle",
                    vehicle_type=vevent.get("vehicle_type", "unknown"),
                    event_key=event_key or vevent.get("event_key"),
                    track_id=event_track_id,
                    source_name=ai_service.source_name,
                    metadata={
                        "event_origin": "segment_pipeline",
                        "event_time_quality": time_quality,
                        "event_time_basis": "segment_end_minus_video_relative_time",
                        "segment_filename": segment_info.get("filename"),
                        "tracking_generation": generation,
                    },
                )
                shadow_evaluations.append(s_eval)

            # Satu audit per event OCR, dengan identitas kendaraan dan sesi tracking.
            for pevent in plate_history:
                p_text = pevent.get("plate") or pevent.get("text") or pevent.get("formatted")
                vehicle_track_id = pevent.get("vehicle_track_id")
                plate_track_id = pevent.get("track_id")
                audit_track_id = (
                    vehicle_track_id
                    if vehicle_track_id is not None
                    else plate_track_id
                )
                generation = pevent.get("vehicle_generation") or 1
                event_key = ai_service._build_event_key(
                    "vehicle",
                    vehicle_track_id,
                    generation,
                ) if vehicle_track_id is not None else pevent.get("event_key")
                if event_key in pre_audited_event_keys:
                    continue
                matching_group = next(
                    (
                        group for group in vehicle_groups
                        if str(group.get("event_track_id", group.get("track_id")))
                        == str(vehicle_track_id)
                    ),
                    None,
                ) if vehicle_track_id is not None else None
                video_time = (
                    matching_group.get("last_seen")
                    if matching_group is not None
                    else pevent.get("video_time")
                )
                event_time, time_quality = _event_timestamp(video_time)
                s_eval = shadow_svc.evaluate_candidate(
                    camera_id=cam_id,
                    plate_number=p_text,
                    timestamp=event_time,
                    source_type="video",
                    object_type="vehicle",
                    vehicle_type="unknown",
                    event_key=event_key or pevent.get("event_key"),
                    track_id=audit_track_id,
                    source_name=ai_service.source_name,
                    metadata={
                        "event_origin": "segment_pipeline",
                        "event_time_quality": time_quality,
                        "event_time_basis": "segment_end_minus_vehicle_last_seen",
                        "segment_filename": segment_info.get("filename"),
                        "tracking_generation": generation,
                        "plate_track_id": plate_track_id,
                    },
                )
                shadow_evaluations.append(s_eval)
        except Exception as exc:
            print(f"[SHADOW EVAL ERROR] Evaluasi/audit gagal (jalur simpan tidak diubah): {exc}")

        detections = {
            "inference_frames": ai_service.inference_count,
            "inference_fps": round(
                ai_service.inference_count / t_elapsed, 2
            ) if t_elapsed > 0 else 0.0,
            "model_load_seconds": round(
                ai_service.model_load_seconds, 2
            ),
            "segment_duration_seconds": round(
                max(0.0, float(segment_info.get("duration") or 0.0)), 2
            ),
            "person_and_vehicle_captures": ai_service.person_capture_count,
            "plate_captures": ai_service.plate_capture_count,
            "plate_history": list(ai_service.last_plate_history),
            "event_groups": {
                "vehicle": len(ai_service.video_event_groups.get("vehicle", [])),
                "person": len(ai_service.video_event_groups.get("person", [])),
            },
            "vehicle_events": [
                dict(event) for event in ai_service.video_event_groups.get("vehicle", [])
            ],
            "persistence_errors": list(
                getattr(ai_service, "database_save_errors", [])
            ),
            "shadow_dedup_evaluations": shadow_evaluations,
            "annotated_output": output_path if (output_path and os.path.isfile(output_path)) else None,
        }

        return {
            "detections": detections,
            "elapsed_seconds": round(t_elapsed, 2),
            "processing_ratio": round(
                t_elapsed / max(
                    0.1,
                    float(segment_info.get("duration") or self.segment_duration),
                ),
                2,
            ),
            "cpu_usage": {
                "system_cpu_percent": round(sys_cpu, 1),
                "process_cpu_percent": round(proc_cpu, 1),
            }
        }


    def process_single_segment(self, segment_filename=None, save_annotated=False):
        """
        Metode pemrosesan on-demand untuk 1 segmen (kompatibilitas Tahap 2).
        """
        with self._completed_segments_lock:
            if self.worker_is_busy:
                return {
                    "success": False,
                    "message": (
                        "Pemrosesan AI pada worker tunggal sedang aktif. "
                        "Harap tunggu hingga selesai."
                    ),
                }

            target_segment = None
            if segment_filename:
                target_segment = self._segments_state.get(segment_filename)
                if target_segment is None:
                    candidate_path = (
                        os.path.join(self.camera_dir or "", segment_filename)
                        if self.camera_dir else None
                    )
                    if candidate_path and os.path.exists(candidate_path):
                        raise ValueError(
                            f"File '{segment_filename}' masih dalam proses perekaman oleh FFmpeg "
                            f"atau belum divalidasi. Pemrosesan ditolak."
                        )
                    raise ValueError(
                        f"File '{segment_filename}' tidak terdaftar sebagai segmen selesai."
                    )
            else:
                for seg in self._segments_state.values():
                    if seg.get("status") in ("pending", "ready"):
                        target_segment = seg
                        break

            if target_segment is None:
                return {
                    "success": False,
                    "message": "Tidak ada segmen selesai yang siap diproses.",
                }

            filename = target_segment["filename"]
            status = target_segment.get("status")
            if status in ("processing", "completed"):
                return {
                    "success": False,
                    "message": f"Segmen '{filename}' tidak dapat diproses ulang (status={status}).",
                }

            target_segment["status"] = "processing"
            target_segment["started_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            try:
                previous_attempts = int(target_segment.get("attempts", 0) or 0)
            except (TypeError, ValueError):
                previous_attempts = 0
                print(
                    f"[SEGMENT MANUAL WARNING] attempts rusak pada '{filename}'; "
                    "menginisialisasi ulang counter"
                )
            target_segment["attempts"] = previous_attempts + 1
            target_segment.pop("completed_at", None)
            self.worker_is_busy = True
            self.current_processing_segment = filename
            self.worker_idle_event.clear()
            self._save_state()

        print(
            f"[SEGMENT MANUAL] Segmen '{filename}' diklaim untuk inferensi "
            f"(attempt={target_segment['attempts']})"
        )
        try:
            result = self._execute_segment_ai(target_segment, save_annotated=save_annotated)
            detections = result.get("detections") or {}
            persistence_errors = detections.get("persistence_errors") or []
            with self._completed_segments_lock:
                target_segment["detections"] = detections
                target_segment["cpu_usage"] = result["cpu_usage"]
                target_segment["elapsed_seconds"] = result["elapsed_seconds"]
                target_segment["processing_ratio"] = result.get("processing_ratio")
                if persistence_errors:
                    target_segment["status"] = "failed"
                    target_segment["failed_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    target_segment["error"] = (
                        f"{len(persistence_errors)} detection persistence error(s)"
                    )
                    self.latest_failed_segment = dict(target_segment)
                else:
                    target_segment["status"] = "completed"
                    target_segment["completed_at"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    target_segment.pop("error", None)
                    self.latest_completed_segment = dict(target_segment)
                    self.latest_processed_segment = dict(target_segment)
                state_persisted = self._save_state()

            if persistence_errors:
                return {
                    "success": False,
                    "message": (
                        f"AI selesai, tetapi {len(persistence_errors)} event "
                        "gagal disimpan."
                    ),
                    "data": {
                        "segment": target_segment,
                        "detections": detections,
                    },
                }
            self._cleanup_completed_segment(
                target_segment,
                status_persisted=state_persisted,
            )
            return {
                "success": True,
                "message": f"Segmen '{filename}' berhasil diproses.",
                "data": {
                    "segment": target_segment,
                    "detections": detections,
                    "elapsed_seconds": result["elapsed_seconds"],
                    "processing_ratio": result.get("processing_ratio"),
                    "cpu_usage": result["cpu_usage"],
                }
            }
        except Exception as exc:
            with self._completed_segments_lock:
                target_segment["status"] = "failed"
                target_segment["failed_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                target_segment["error"] = str(exc)
                self.latest_failed_segment = dict(target_segment)
                self._save_state()
            print(f"[SEGMENT MANUAL ERROR] Segmen '{filename}' gagal: {exc}")
            return {
                "success": False,
                "message": f"Gagal memproses segmen: {str(exc)}"
            }
        finally:
            with self._completed_segments_lock:
                self.worker_is_busy = False
                self.current_processing_segment = None
                self.worker_idle_event.set()

    def evaluate_duplicates(self, time_window_seconds=45):
        """
        Evaluasi prospektif duplikasi antarpipeline tanpa mengubah data database.
        """
        window = max(0.0, float(time_window_seconds))
        segment_detections = []
        segment_vehicle_events = 0

        with self._completed_segments_lock:
            for seg in self._segments_state.values():
                if seg.get("status") == "completed" and seg.get("detections"):
                    d = seg["detections"]
                    segment_vehicle_events += len(
                        d.get("vehicle_events", [])
                    ) or int(d.get("event_groups", {}).get("vehicle", 0) or 0)

                    segment_end = self._segment_recorded_at(seg)
                    duration = max(0.0, float(seg.get("duration") or 0.0))
                    for plate_event in d.get("plate_history", []):
                        plate = (
                            plate_event.get("plate")
                            or plate_event.get("text")
                            or plate_event.get("formatted")
                        )
                        normalized = self._normalize_duplicate_plate(plate)
                        if not normalized:
                            continue

                        event_time = segment_end
                        video_time = plate_event.get("video_time")
                        try:
                            if event_time is not None and video_time is not None:
                                event_time = event_time - timedelta(
                                    seconds=max(0.0, duration - float(video_time))
                                )
                            elif event_time is not None:
                                event_time = event_time - timedelta(seconds=duration / 2.0)
                        except (TypeError, ValueError):
                            event_time = None

                        segment_detections.append({
                            "camera_id": seg.get("camera_id"),
                            "plate": normalized,
                            "timestamp": event_time,
                            "track_id": (
                                plate_event.get("vehicle_track_id")
                                if plate_event.get("vehicle_track_id") is not None
                                else plate_event.get("track_id")
                            ),
                            "generation": plate_event.get("vehicle_generation")
                            or plate_event.get("generation"),
                            "segment": seg.get("filename"),
                            "confidence": plate_event.get("confidence"),
                            "source": "segment",
                        })

        realtime_detections = []
        realtime_vehicle_events = 0
        query_error = None
        try:
            with db.get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT fd.detection_id, fd.camera_id, fd.track_id, fd.event_key,
                               fd.created_at, p.plate_number, p.ocr_confidence
                        FROM full_detection fd
                        LEFT JOIN plate p ON fd.plate_id = p.plate_id
                        WHERE (fd.source_type = 'stream' OR fd.source_type IS NULL)
                          AND fd.object_type = 'vehicle'
                          AND fd.created_at >= DATE_SUB(NOW(), INTERVAL %s SECOND)
                        ORDER BY fd.created_at DESC
                        LIMIT 1000
                        """,
                        (max(300, int(window * 2)),),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        realtime_vehicle_events += 1
                        plate = self._normalize_duplicate_plate(r.get("plate_number"))
                        if not plate:
                            continue
                        realtime_detections.append({
                            "detection_id": r.get("detection_id"),
                            "camera_id": r.get("camera_id"),
                            "plate": plate,
                            "timestamp": r.get("created_at"),
                            "track_id": r.get("track_id"),
                            "event_key": r.get("event_key"),
                            "confidence": float(r.get("ocr_confidence") or 0.0),
                            "source": "realtime",
                        })
        except Exception as exc:
            query_error = str(exc)
            print(f"[DUPLICATE EVAL ERROR] Gagal query realtime DB: {exc}")

        result = self._deduplicate_plate_events(
            realtime_detections,
            segment_detections,
            window,
        )
        duplicate_count = result["cross_path_duplicates_count"] + result["within_source_duplicates_count"]
        before_count = result["events_before_deduplication"]

        return {
            "time_window_seconds": window,
            "realtime_detections_sampled": len(realtime_detections),
            "segment_detections_sampled": len(segment_detections),
            "realtime_vehicle_events_sampled": realtime_vehicle_events,
            "segment_vehicle_events_sampled": segment_vehicle_events,
            "duplicate_candidates_count": result["cross_path_duplicates_count"],
            "duplicate_candidates": result["duplicate_candidates"][:20],
            "within_source_duplicate_events": result["within_source_duplicates_count"],
            "events_before_deduplication": before_count,
            "events_after_deduplication": max(0, before_count - duplicate_count),
            "duplicate_events_removed_prospectively": duplicate_count,
            "duplication_rate_percent": (
                round(duplicate_count * 100.0 / before_count, 1) if before_count else 0.0
            ),
            "unmatched_events_without_timestamp": result["unmatched_events_without_timestamp"],
            "repeated_pass_groups": result["repeated_pass_groups"],
            "unplated_vehicle_events_not_deduplicated": max(
                0,
                realtime_vehicle_events + segment_vehicle_events - before_count,
            ),
            "database_mutated": False,
            "realtime_query_error": query_error,
            "sample_adequate": bool(
                result["cross_path_duplicates_count"] >= 2
                and result["repeated_pass_groups"] >= 1
                and result["timestamped_realtime_events"] >= 3
                and result["timestamped_segment_events"] >= 3
            ),
            "evaluation_analysis": {
                "matching": (
                    "Satu-ke-satu berdasarkan kamera, pelat ternormalisasi, dan selisih timestamp. "
                    "Tracking ID dipertahankan sebagai bukti peristiwa dalam jalurnya, tetapi ID dari "
                    "dua tracker berbeda tidak dianggap setara."
                ),
                "repeat_passes": (
                    "Pemasangan satu-ke-satu menjaga lintasan berulang berpelat sama sebagai event "
                    "terpisah; event tidak dipasangkan ulang setelah pasangan pertama."
                ),
                "unplated_vehicles": (
                    "Tidak dihapus karena tidak tersedia identitas lintas-jalur yang cukup kuat; "
                    "mengandalkan waktu/kelas saja berisiko menghapus kendaraan lain."
                ),
                "timestamp_basis": (
                    "Timestamp segmen diperkirakan dari waktu akhir rekaman (recorded_at atau mtime) "
                    "dan titik tengah durasi segmen bila video_time tidak tersedia."
                ),
                "persistence": (
                    "Hasil sesudah deduplikasi bersifat simulasi evaluasi. Tidak ada INSERT, UPDATE, "
                    "atau DELETE MySQL maupun perubahan perilaku penulis event."
                ),
            }
        }

    @staticmethod
    def _normalize_duplicate_plate(value):
        normalized = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
        return normalized if normalized and normalized != "-" else ""

    @staticmethod
    def _parse_duplicate_timestamp(value):
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None

        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed

    @classmethod
    def _segment_recorded_at(cls, segment):
        recorded_at = cls._parse_duplicate_timestamp(segment.get("recorded_at"))
        if recorded_at is not None:
            return recorded_at

        file_path = segment.get("file_path")
        if file_path and os.path.isfile(file_path):
            try:
                return datetime.fromtimestamp(os.path.getmtime(file_path))
            except OSError:
                return None
        return None

    @classmethod
    def _deduplicate_plate_events(cls, realtime_events, segment_events, time_window_seconds):
        def prepare(events):
            prepared = []
            for event in events:
                plate = cls._normalize_duplicate_plate(event.get("plate"))
                timestamp = cls._parse_duplicate_timestamp(
                    event.get("timestamp") or event.get("created_at")
                )
                camera_id = event.get("camera_id")
                if not plate or camera_id is None:
                    continue
                prepared.append({
                    **event,
                    "plate": plate,
                    "timestamp": timestamp,
                    "camera_id": str(camera_id),
                })
            return prepared

        raw_realtime = prepare(realtime_events)
        raw_segments = prepare(segment_events)
        no_timestamp = sum(
            1
            for event in raw_realtime + raw_segments
            if event.get("timestamp") is None
        )
        window = max(0.0, float(time_window_seconds))

        def compact_source(events):
            compacted = []
            last_by_identity = {}
            duplicate_count = 0
            for event in sorted(
                events,
                key=lambda item: (
                    item.get("timestamp") is None,
                    item.get("timestamp") or datetime.max,
                ),
            ):
                identity = event.get("event_key")
                if (
                    not identity
                    and event.get("track_id") is not None
                    and event.get("generation") is not None
                ):
                    identity = (
                        f"track:{event.get('track_id')}:"
                        f"generation:{event.get('generation')}"
                    )
                if identity is None or event.get("timestamp") is None:
                    compacted.append(event)
                    continue

                identity_key = (event["camera_id"], event["plate"], str(identity))
                previous = last_by_identity.get(identity_key)
                if previous is not None:
                    delta = (event["timestamp"] - previous["timestamp"]).total_seconds()
                    if 0 <= delta <= window:
                        duplicate_count += 1
                        continue

                compacted.append(event)
                last_by_identity[identity_key] = event
            return compacted, duplicate_count

        realtime, realtime_same_source_duplicates = compact_source(raw_realtime)
        segments, segment_same_source_duplicates = compact_source(raw_segments)
        duplicate_candidates = []

        group_keys = sorted({
            (event["camera_id"], event["plate"])
            for event in realtime + segments
            if event.get("timestamp") is not None
        })
        for camera_id, plate in group_keys:
            stream_group = sorted(
                (
                    (index, event)
                    for index, event in enumerate(realtime)
                    if event["camera_id"] == camera_id
                    and event["plate"] == plate
                    and event.get("timestamp") is not None
                ),
                key=lambda pair: pair[1]["timestamp"],
            )
            segment_group = sorted(
                (
                    (index, event)
                    for index, event in enumerate(segments)
                    if event["camera_id"] == camera_id
                    and event["plate"] == plate
                    and event.get("timestamp") is not None
                ),
                key=lambda pair: pair[1]["timestamp"],
            )

            stream_index = 0
            segment_index = 0
            while stream_index < len(stream_group) and segment_index < len(segment_group):
                rt_event = stream_group[stream_index][1]
                seg_event = segment_group[segment_index][1]
                delta = (seg_event["timestamp"] - rt_event["timestamp"]).total_seconds()

                if delta < -window:
                    segment_index += 1
                    continue
                if delta > window:
                    stream_index += 1
                    continue

                duplicate_candidates.append({
                    "plate": plate,
                    "camera_id": rt_event["camera_id"],
                    "realtime_id": rt_event.get("detection_id"),
                    "realtime_time": rt_event["timestamp"].isoformat(sep=" "),
                    "realtime_track_id": rt_event.get("track_id"),
                    "segment_file": seg_event.get("segment"),
                    "segment_time": seg_event["timestamp"].isoformat(sep=" "),
                    "segment_track_id": seg_event.get("track_id"),
                    "time_delta_seconds": round(abs(delta), 2),
                    "overlap_type": "same_camera_plate_time_match",
                    "merge_policy": "keep_realtime_in_prospective_result",
                })
                stream_index += 1
                segment_index += 1

        valid_realtime = sum(1 for event in raw_realtime if event.get("timestamp") is not None)
        valid_segments = sum(1 for event in raw_segments if event.get("timestamp") is not None)
        cross_path_duplicates = len(duplicate_candidates)
        pass_counts = {}
        for candidate in duplicate_candidates:
            pass_key = (candidate["camera_id"], candidate["plate"])
            pass_counts[pass_key] = pass_counts.get(pass_key, 0) + 1
        repeated_pass_groups = sum(1 for count in pass_counts.values() if count >= 2)
        same_source_duplicates = (
            realtime_same_source_duplicates + segment_same_source_duplicates
        )
        return {
            "duplicate_candidates": duplicate_candidates,
            "cross_path_duplicates_count": cross_path_duplicates,
            "within_source_duplicates_count": same_source_duplicates,
            "events_before_deduplication": len(raw_realtime) + len(raw_segments),
            "events_after_deduplication": (
                len(raw_realtime) + len(raw_segments)
                - cross_path_duplicates - same_source_duplicates
            ),
            "unmatched_events_without_timestamp": no_timestamp,
            "timestamped_realtime_events": valid_realtime,
            "timestamped_segment_events": valid_segments,
            "repeated_pass_groups": repeated_pass_groups,
        }

    # ------------------------------------------------------------------
    # SUPERVISOR & PENANGANAN PROSES FFMPEG
    # ------------------------------------------------------------------

    def _resolve_target_camera(self, specified_camera_id=None):
        """Resolve an active camera; never silently fall back to an inactive one."""
        cameras = db.get_all_cameras()

        target_id = specified_camera_id
        if target_id is None:
            env_cam = os.environ.get("RECORD_CAMERA_ID")
            if env_cam and env_cam.isdigit():
                target_id = int(env_cam)

        if target_id is not None:
            try:
                target_id = int(target_id)
            except (TypeError, ValueError):
                self.last_error = f"ID kamera tidak valid: {target_id!r}"
                return None
            for cam in cameras:
                try:
                    camera_id = int(cam.get("camera_id"))
                except (TypeError, ValueError):
                    continue
                if camera_id != target_id:
                    continue
                try:
                    is_active = int(cam.get("status") or 0) == 1
                except (TypeError, ValueError):
                    is_active = False
                if not is_active:
                    self.last_error = f"Kamera {target_id} tidak aktif di konfigurasi"
                    return None
                if not str(cam.get("stream_url") or "").strip():
                    self.last_error = f"Kamera {target_id} tidak memiliki stream URL"
                    return None
                return cam
            self.last_error = f"Kamera {target_id} tidak ditemukan di database"
            return None

        for cam in cameras:
            try:
                is_active = int(cam.get("status") or 0) == 1
            except (TypeError, ValueError):
                is_active = False
            if is_active and str(cam.get("stream_url") or "").strip():
                return cam

        self.last_error = "Tidak ada kamera aktif dengan stream URL yang valid"

        return None

    def _run_supervisor(self, target_camera_id):
        """Loop pengawas untuk menjaga proses FFmpeg tetap aktif dan merekam."""
        retry_delay = 3.0

        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.stream_url:
                    cam = self._resolve_target_camera(target_camera_id)
                    if not cam or not cam.get("stream_url"):
                        self.status = "ERROR"
                        if not self.last_error:
                            self.last_error = "Tidak ada kamera aktif untuk direkam"
                        print(f"[RECORDER] {self.last_error}. Menunggu 5 detik...")
                        self.stop_event.wait(5.0)
                        continue

                    self.camera_id = int(cam["camera_id"])
                    self.camera_name = cam.get("location") or f"Camera_{self.camera_id}"
                    self.stream_url = normalize_stream_url(cam["stream_url"])
                else:
                    self.camera_id = target_camera_id or self.camera_id or 1
                    self.camera_name = self.camera_name or f"Camera_{self.camera_id}"

                success = self._start_recording_session()
            except Exception as exc:
                self.status = "ERROR"
                self.last_error = f"Gagal menyiapkan recorder: {exc}"
                print(f"[RECORDER ERROR] {self.last_error}")
                self.stop_event.wait(retry_delay)
                continue

            if not success:
                self.stop_event.wait(retry_delay)
                continue

            # Awasi proses FFmpeg sampai selesai atau terhenti
            while self.is_running and not self.stop_event.is_set():
                with self.process_lock:
                    if self.process is None or self.process.poll() is not None:
                        break
                time.sleep(1.0)

            if self.is_running and not self.stop_event.is_set():
                print(f"[RECORDER] FFmpeg terhenti tak terduga. Reconnect dalam {retry_delay}s...")
                self.status = "CONNECTING"
                self.stop_event.wait(retry_delay)

    def _start_recording_session(self):
        """Inisialisasi direktori, sesi, muat state tersimpan, dan jalankan FFmpeg."""
        try:
            self._terminate_ffmpeg()

            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.camera_dir = os.path.join(self.segments_dir, f"cam_{self.camera_id}")
            os.makedirs(self.camera_dir, exist_ok=True)

            self.state_file = os.path.join(self.camera_dir, "segment_queue_state.json")
            self._load_state()

            self.list_file = os.path.join(self.camera_dir, f"segments_{self.session_id}.csv")
            segment_pattern = os.path.join(
                self.camera_dir,
                f"cam_{self.camera_id}_{self.session_id}_%05d.mp4"
            )

            command = self._build_ffmpeg_command(self.stream_url, segment_pattern, self.list_file)
            with self._stderr_lock:
                self.last_ffmpeg_stderr = []

            print(f"[RECORDER] Menjalankan FFmpeg segmenter untuk Kamera {self.camera_id} ({self.camera_name})")
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            with self.process_lock:
                self.process = proc
                self.started_at = time.time()
                self.status = "RECORDING"
                self.is_running = True
                self.last_error = None

            self.stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(proc,),
                name=f"Recorder-stderr-cam{self.camera_id}",
                daemon=True,
            )
            self.stderr_thread.start()

            self.monitor_thread = threading.Thread(
                target=self._monitor_segments,
                args=(self.list_file, self.camera_dir),
                name=f"Recorder-monitor-cam{self.camera_id}",
                daemon=True,
            )
            self.monitor_thread.start()

            time.sleep(0.5)
            if proc.poll() is not None:
                rc = proc.poll()
                self._wait_for_stderr()
                stderr_text = "\n".join(self._get_ffmpeg_stderr())
                details = f": {stderr_text}" if stderr_text else ""
                self.status = "ERROR"
                self.last_error = (
                    f"FFmpeg berhenti saat startup (returncode {rc}){details}"
                )
                print(f"[RECORDER ERROR] {self.last_error}")
                return False

            return True

        except Exception as exc:
            self.status = "ERROR"
            self.last_error = str(exc)
            print(f"[RECORDER ERROR] Gagal memulai FFmpeg: {exc}")
            return False

    def _build_ffmpeg_command(self, stream_url, output_pattern, list_file):
        """Menyusun parameter FFmpeg segmentasi: stream copy, 10s, CSV index."""
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "+genpts",
        ]

        url_lower = stream_url.lower()
        if url_lower.startswith(("http://", "https://", "rtmp://", "rtsp://")):
            cmd.extend(["-rw_timeout", "10000000"])

        if url_lower.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        elif url_lower.startswith("rtmp://"):
            cmd.extend(["-rtmp_live", "live"])
        elif url_lower.startswith(("http://", "https://")):
            cmd.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "2",
            ])
        else:
            cmd.append("-re")

        cmd.extend([
            "-i", stream_url,
            "-c:v", "copy",
            "-an",
            "-f", "segment",
            "-segment_time", str(self.segment_duration),
            "-reset_timestamps", "1",
            "-segment_list", list_file,
            "-segment_list_type", "csv",
            output_pattern,
        ])

        return cmd

    def _drain_stderr(self, proc):
        """Drain and retain a bounded tail of FFmpeg diagnostics."""
        try:
            if proc and proc.stderr:
                for line in iter(proc.stderr.readline, b""):
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    line = str(line).strip()
                    if not line:
                        continue
                    with self._stderr_lock:
                        self.last_ffmpeg_stderr.append(line[-1000:])
                        self.last_ffmpeg_stderr = self.last_ffmpeg_stderr[-30:]
        except Exception as exc:
            with self._stderr_lock:
                self.last_ffmpeg_stderr.append(f"[stderr reader error] {exc}")
                self.last_ffmpeg_stderr = self.last_ffmpeg_stderr[-30:]

    def _wait_for_stderr(self, timeout=0.5):
        thread = self.stderr_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def _terminate_ffmpeg(self):
        """Mematikan proses FFmpeg yang sedang berjalan."""
        with self.process_lock:
            proc = self.process
            self.process = None

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # MONITOR SEGMEN SELESAI & ENQUEUE
    # ------------------------------------------------------------------

    def _monitor_segments(self, list_file, camera_dir):
        """
        Memantau file CSV segment list FFmpeg.
        Hanya entri yang tercatat di CSV dan tervalidasi yang diberi status 'pending'
        dan dimasukkan ke antrean worker tunggal.
        """
        last_read_line = 0

        while self.is_running and not self.stop_event.is_set():
            if not os.path.isfile(list_file):
                time.sleep(0.5)
                continue

            try:
                with open(list_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                complete_line_count = len(lines)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    complete_line_count -= 1

                if complete_line_count > last_read_line:
                    new_lines = lines[last_read_line:complete_line_count]
                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue

                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            filename = parts[0]
                            if (
                                filename not in self._processed_segment_filenames
                                and filename not in self._unvalidated_segment_rows
                            ):
                                try:
                                    start_time = float(parts[1])
                                    end_time = float(parts[2])
                                except ValueError as exc:
                                    print(
                                        f"[RECORDER MONITOR ERROR] CSV row invalid "
                                        f"('{filename}'): {exc}"
                                    )
                                    continue
                                self._unvalidated_segment_rows[filename] = {
                                    "start_time": start_time,
                                    "end_time": end_time,
                                }
                                print(
                                    f"[RECORDER SCAN] CSV menemukan segmen '{filename}' "
                                    f"(start={start_time:.2f}s, end={end_time:.2f}s)"
                                )

                for filename, row_info in list(self._unvalidated_segment_rows.items()):
                    file_path = os.path.join(camera_dir, filename)
                    if not self._validate_segment_file(file_path):
                        print(
                            f"[RECORDER SCAN] Segmen '{filename}' belum tervalidasi; "
                            "akan diperiksa kembali"
                        )
                        continue

                    segment_info = {
                        "camera_id": self.camera_id,
                        "camera_name": self.camera_name,
                        "filename": filename,
                        "file_path": file_path,
                        "start_time": row_info["start_time"],
                        "end_time": row_info["end_time"],
                        "duration": round(
                            row_info["end_time"] - row_info["start_time"], 2
                        ),
                        "size_bytes": os.path.getsize(file_path),
                        "status": "pending",
                        "recorded_at": datetime.now().isoformat(
                            sep=" ",
                            timespec="seconds",
                        ),
                    }

                    should_enqueue = False
                    with self._completed_segments_lock:
                        existing_state = self._segments_state.get(filename)
                        self._processed_segment_filenames.add(filename)
                        if existing_state is None:
                            self._segments_state[filename] = segment_info
                            self._completed_segments.append(segment_info)
                            self._save_state()
                            should_enqueue = self.auto_process_enabled
                        else:
                            segment_info = existing_state

                    self._unvalidated_segment_rows.pop(filename, None)
                    print(
                        f"[RECORDER SCAN] Segmen tervalidasi: '{filename}' "
                        f"({segment_info.get('duration')}s, "
                        f"{segment_info.get('size_bytes')} bytes, "
                        f"status={segment_info.get('status')})"
                    )
                    if should_enqueue:
                        self._enqueue_segment(segment_info)
                last_read_line = complete_line_count

            except Exception as exc:
                print(f"[RECORDER MONITOR ERROR] {exc}")

            time.sleep(0.5)

    def _validate_segment_file(self, file_path, max_wait_seconds=0.0):
        """Memastikan file ada di disk dan ukuran > 1KB."""
        deadline = time.time() + max_wait_seconds
        while True:
            if os.path.isfile(file_path):
                try:
                    size = os.path.getsize(file_path)
                    if size > 1024:
                        return True
                except OSError:
                    pass
            if time.time() >= deadline:
                return False
            time.sleep(0.2)

    def _notify_completed(self, segment_info):
        """Memanggil listener callback."""
        for cb in self._callbacks:
            try:
                cb(segment_info)
            except Exception as exc:
                print(f"[RECORDER CALLBACK ERROR] {exc}")
