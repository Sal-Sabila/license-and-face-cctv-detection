import json
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta

import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import db
import services.stream_ai_service as stream_module
import services.video_ai_service as video_module
from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager
from services.stream_ai_service import StreamAIService


CAMERA_ID = 6
TEST_DURATION_SECONDS = int(os.environ.get("TAHAP5_TEST_SECONDS", "310"))
DEDUP_WINDOW_SECONDS = 45
REPORT_PATH = os.path.join(os.path.dirname(__file__), "tahap5_live_test_report.json")


def run_live_test():
    cameras = db.get_all_cameras()
    camera = next(
        (item for item in cameras if item.get("camera_id") == CAMERA_ID),
        None,
    )
    if not camera or not camera.get("stream_url"):
        raise RuntimeError(f"Kamera {CAMERA_ID} tidak tersedia atau tidak memiliki stream.")

    run_dir = tempfile.mkdtemp(prefix="tahap5_cam6_", dir=os.path.dirname(__file__))
    event_lock = threading.Lock()
    event_rows = {}
    intercepted_calls = 0
    next_id = 0
    test_started_at = datetime.now()

    def intercept_detection_save(**values):
        nonlocal intercepted_calls, next_id
        source = values.get("source_type") or "unknown"
        event_key = values.get("event_key")
        camera_id = values.get("camera_id")
        plate = values.get("plate_number")
        with event_lock:
            intercepted_calls += 1
            key = (
                (source, str(camera_id), str(event_key))
                if event_key
                else (source, "call", intercepted_calls)
            )
            event = event_rows.get(key)
            is_duplicate = event is not None
            if event is None:
                next_id += 1
                event = {
                    "detection_id": next_id,
                    "camera_id": camera_id,
                    "source": source,
                    "event_key": event_key,
                    "track_id": values.get("track_id"),
                    "timestamp": datetime.now(),
                    "plate": "",
                    "object_type": values.get("object_type"),
                }
                event_rows[key] = event
            if plate and str(plate).strip() not in ("", "-"):
                event["plate"] = str(plate)
            return {
                "plate_id": None,
                "detection_id": event["detection_id"],
                "duplicate": is_duplicate,
                "enriched": bool(plate),
                "plate_image_path": None,
                "vehicle_image_path": None,
                "face_image_path": None,
                "plate_number": event["plate"] or None,
                "object_type": event["object_type"],
                "vehicle_type": values.get("vehicle_type", "unknown"),
                "has_plate": bool(event["plate"]),
                "has_driver": bool(values.get("has_driver")),
            }

    original_saver = db.save_detection_event
    original_stream_plate_capture = stream_module.save_plate_capture
    original_stream_person_capture = stream_module.save_person_capture
    original_video_plate_capture = video_module.save_plate_capture
    original_video_person_capture = video_module.save_person_capture
    db.save_detection_event = intercept_detection_save
    stream_module.save_plate_capture = lambda *args, **kwargs: None
    stream_module.save_person_capture = lambda *args, **kwargs: None
    video_module.save_plate_capture = lambda *args, **kwargs: None
    video_module.save_person_capture = lambda *args, **kwargs: None

    recorder = None
    worker_manager = None
    ai_service = None
    elapsed = 0
    realtime_online_samples = 0
    realtime_stalls = 0
    system_cpu_samples = []
    process_cpu_samples = []
    memory_samples = []
    process = psutil.Process()

    try:
        ai_service = StreamAIService.get_instance()
        worker_manager = CameraWorkerManager.get_instance()
        worker_manager._start_worker(CAMERA_ID, camera["stream_url"], ai_service)

        recorder = CameraRecorderService(
            segments_dir=run_dir,
            segment_duration=10,
        )
        recorder.camera_id = CAMERA_ID
        recorder.camera_name = camera.get("location") or f"Camera_{CAMERA_ID}"
        recorder.stream_url = camera["stream_url"]
        recorder.auto_process_enabled = True
        recorder.max_queue_size = 15
        recorder.start(camera_id=CAMERA_ID)

        process.cpu_percent(interval=None)
        started = time.monotonic()
        while elapsed < TEST_DURATION_SECONDS:
            interval = min(10, TEST_DURATION_SECONDS - elapsed)
            time.sleep(interval)
            elapsed = int(time.monotonic() - started)

            system_cpu_samples.append(psutil.cpu_percent(interval=None))
            process_cpu_samples.append(process.cpu_percent(interval=None))
            memory_samples.append(
                round(process.memory_info().rss / (1024 * 1024), 1)
            )

            worker = worker_manager.get_worker(CAMERA_ID)
            status = worker.get_status() if worker else {"status": "MISSING"}
            frame = worker.get_last_frame() if worker else None
            online = (
                status.get("status") == "ONLINE"
                and frame is not None
                and getattr(frame, "size", 0) > 0
            )
            if online:
                realtime_online_samples += 1
            else:
                realtime_stalls += 1

            backlog = recorder.get_status().get("backlog", {})
            print(
                f"[TAHAP 5] t={elapsed}s realtime={status.get('status')} "
                f"segments={backlog.get('total_recorded', 0)} "
                f"completed={backlog.get('completed', 0)} "
                f"pending={backlog.get('pending', 0)} "
                f"plates={sum(bool(row.get('plate')) for row in event_rows.values())}"
            )

        drain_deadline = time.monotonic() + 60
        while recorder.worker_is_busy and time.monotonic() < drain_deadline:
            time.sleep(1)

        recorder.stop()
        worker_manager._stop_worker(CAMERA_ID)
        ai_service.stop()

        realtime_events = []
        vehicle_event_rows = []
        with event_lock:
            captured_events = [dict(row) for row in event_rows.values()]
        for event in captured_events:
            if event.get("source") == "stream":
                if event.get("object_type") == "vehicle":
                    vehicle_event_rows.append(event)
                if event.get("plate"):
                    realtime_events.append(event)

        segment_events = []
        segment_vehicle_events = 0
        completed_segments = recorder.get_completed_segments(status="completed")
        for segment in completed_segments:
            detections = segment.get("detections") or {}
            vehicle_events = detections.get("vehicle_events", [])
            segment_vehicle_events += len(vehicle_events) or int(
                detections.get("event_groups", {}).get("vehicle", 0) or 0
            )
            segment_end = recorder._segment_recorded_at(segment)
            duration = max(0.0, float(segment.get("duration") or 0.0))
            for plate_event in detections.get("plate_history", []):
                plate = (
                    plate_event.get("plate")
                    or plate_event.get("text")
                    or plate_event.get("formatted")
                )
                if not plate:
                    continue
                event_time = segment_end
                if event_time is not None:
                    video_time = plate_event.get("video_time")
                    if video_time is not None:
                        event_time -= timedelta(
                            seconds=max(0.0, duration - float(video_time))
                        )
                    else:
                        event_time -= timedelta(seconds=duration / 2)
                segment_events.append({
                    "camera_id": segment.get("camera_id"),
                    "plate": plate,
                    "timestamp": event_time,
                    "track_id": (
                        plate_event.get("vehicle_track_id")
                        if plate_event.get("vehicle_track_id") is not None
                        else plate_event.get("track_id")
                    ),
                    "generation": (
                        plate_event.get("vehicle_generation")
                        or plate_event.get("generation")
                    ),
                    "segment": segment.get("filename"),
                })

        evaluation = recorder._deduplicate_plate_events(
            realtime_events,
            segment_events,
            DEDUP_WINDOW_SECONDS,
        )
        total_vehicle_events = len(vehicle_event_rows) + segment_vehicle_events
        duplicate_count = (
            evaluation["cross_path_duplicates_count"]
            + evaluation["within_source_duplicates_count"]
        )
        safe_evaluation = dict(evaluation)
        safe_evaluation["duplicate_candidates"] = [
            {
                **{
                    key: value
                    for key, value in candidate.items()
                    if key != "plate"
                },
                "plate_masked": (
                    f"{candidate['plate'][:2]}***{candidate['plate'][-2:]}"
                    if len(candidate.get("plate", "")) > 4
                    else "***"
                ),
            }
            for candidate in evaluation["duplicate_candidates"]
        ]
        report = {
            "started_at": test_started_at.isoformat(sep=" ", timespec="seconds"),
            "camera_id": CAMERA_ID,
            "duration_seconds": elapsed,
            "deduplication_window_seconds": DEDUP_WINDOW_SECONDS,
            "recording_segments": len(recorder.get_completed_segments()),
            "completed_segments": len(completed_segments),
            "realtime_online_samples": realtime_online_samples,
            "realtime_stalls": realtime_stalls,
            "database_mutated": False,
            "database_save_calls_intercepted": intercepted_calls,
            "vehicle_events_before_deduplication": total_vehicle_events,
            "vehicle_events_after_deduplication": max(
                0, total_vehicle_events - duplicate_count
            ),
            "plate_evaluation": safe_evaluation,
            "sample_adequate": bool(
                evaluation["cross_path_duplicates_count"] >= 2
                and evaluation["repeated_pass_groups"] >= 1
                and evaluation["timestamped_realtime_events"] >= 3
                and evaluation["timestamped_segment_events"] >= 3
            ),
            "cpu": {
                "system_average_percent": round(
                    sum(system_cpu_samples) / len(system_cpu_samples), 1
                ) if system_cpu_samples else 0.0,
                "system_peak_percent": round(max(system_cpu_samples), 1)
                if system_cpu_samples else 0.0,
                "process_average_percent": round(
                    sum(process_cpu_samples) / len(process_cpu_samples), 1
                ) if process_cpu_samples else 0.0,
                "process_peak_percent": round(max(process_cpu_samples), 1)
                if process_cpu_samples else 0.0,
                "max_memory_mb": max(memory_samples) if memory_samples else 0.0,
            },
            "potential_data_loss": (
                "Vehicles without a readable plate are counted but not deduplicated. "
                "Matching tracking IDs across the two independent trackers is not assumed."
            ),
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, default=str)
        print(json.dumps(report, indent=2, default=str))
        print(f"[TAHAP 5] Report tersimpan: {REPORT_PATH}")
        return report
    finally:
        if recorder is not None and recorder.is_running:
            recorder.stop()
        if worker_manager is not None:
            worker_manager._stop_worker(CAMERA_ID)
        if ai_service is not None:
            ai_service.stop()
        db.save_detection_event = original_saver
        stream_module.save_plate_capture = original_stream_plate_capture
        stream_module.save_person_capture = original_stream_person_capture
        video_module.save_plate_capture = original_video_plate_capture
        video_module.save_person_capture = original_video_person_capture
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    run_live_test()
