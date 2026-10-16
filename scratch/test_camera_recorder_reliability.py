import io
import json
import os
import queue
import sys
import threading
import time
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.camera_recorder_service import CameraRecorderService
from services.camera_worker_manager import CameraWorkerManager
from services.stream_ai_service import StreamAIService
from services.video_ai_service import VideoAIService
from tracker import PlateTracker


class _ExitedProcess:
    def __init__(self, *args, **kwargs):
        self.stderr = io.BytesIO(b"connection refused by camera\n")

    def poll(self):
        return 1


class _ImmediateThread:
    def __init__(self, target, args=(), name=None, **kwargs):
        self.target = target
        self.args = args
        self.name = name

    def start(self):
        if self.name and self.name.startswith("Recorder-stderr-"):
            self.target(*self.args)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class CameraRecorderReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.recorder = CameraRecorderService(
            segments_dir=self.temp_dir.name,
            segment_duration=10,
        )

    def tearDown(self):
        self.recorder.stop_event.set()
        self.temp_dir.cleanup()

    @staticmethod
    def camera(camera_id, status, stream_url="rtsp://camera/live"):
        return {
            "camera_id": camera_id,
            "status": status,
            "stream_url": stream_url,
            "location": f"Camera {camera_id}",
        }

    def test_default_selection_never_uses_first_inactive_camera(self):
        cameras = [
            self.camera(1, 0),
            self.camera(2, "1"),
            self.camera(3, 1, ""),
        ]
        with patch(
            "services.camera_recorder_service.db.get_all_cameras",
            return_value=cameras,
        ):
            selected = self.recorder._resolve_target_camera()

        self.assertEqual(selected["camera_id"], 2)

    def test_explicit_id_is_normalized_and_inactive_target_is_rejected(self):
        cameras = [self.camera(1, 1), self.camera(2, 0)]
        with patch(
            "services.camera_recorder_service.db.get_all_cameras",
            return_value=cameras,
        ):
            self.assertEqual(
                self.recorder._resolve_target_camera("1")["camera_id"],
                1,
            )
            self.assertIsNone(self.recorder._resolve_target_camera(2))
            self.assertIn("tidak aktif", self.recorder.last_error)

    def test_no_active_camera_returns_clear_error_not_inactive_fallback(self):
        with patch(
            "services.camera_recorder_service.db.get_all_cameras",
            return_value=[self.camera(1, 0)],
        ):
            self.assertIsNone(self.recorder._resolve_target_camera())
        self.assertIn("Tidak ada kamera aktif", self.recorder.last_error)

    def test_segment_worker_waits_for_realtime_stream_without_claiming_backlog(self):
        segment = {
            "filename": "archived.mp4",
            "file_path": os.path.join(self.temp_dir.name, "archived.mp4"),
            "camera_id": 6,
            "status": "pending",
        }
        with open(segment["file_path"], "wb") as segment_file:
            segment_file.write(b"recorded segment")
        self.recorder.is_running = True
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[segment["filename"]] = segment
        self.recorder._enqueue_segment(segment)
        executed = []
        self.recorder._execute_segment_ai = lambda item: executed.append(item)

        with patch.object(
            self.recorder,
            "_stream_inference_active",
            return_value=True,
        ):
            worker = threading.Thread(
                target=self.recorder._run_processing_worker,
                daemon=True,
            )
            worker.start()
            time.sleep(0.1)
            self.assertFalse(self.recorder.worker_is_busy)
            self.assertEqual(self.recorder._processing_queue.qsize(), 1)
            self.assertEqual(executed, [])
            self.recorder.stop_event.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            self.recorder._segments_state["archived.mp4"]["status"],
            "pending",
        )

    def test_manual_segment_processing_is_deferred_while_stream_is_active(self):
        with patch.object(
            self.recorder,
            "_stream_inference_active",
            return_value=True,
        ):
            result = self.recorder.process_single_segment("archive.mp4")

        self.assertFalse(result["success"])
        self.assertIn("stream realtime aktif", result["message"])
        self.assertFalse(self.recorder.worker_is_busy)

    def test_camera_manager_treats_initial_sync_and_worker_presence_as_active(self):
        manager = CameraWorkerManager()
        manager.running = True
        self.assertTrue(manager.is_realtime_stream_active())

        manager.initial_sync_complete.set()
        self.assertFalse(manager.is_realtime_stream_active())
        manager.workers[1] = object()
        self.assertTrue(manager.is_realtime_stream_active())
        manager.workers.clear()
        manager.ai_service = SimpleNamespace(
            frame_queue_lock=threading.Lock(),
            frame_queue={1: object()},
            ai_processing=False,
        )
        self.assertTrue(manager.is_realtime_stream_active())
        manager.ai_service.frame_queue.clear()
        manager.ai_service.ai_processing = True
        self.assertTrue(manager.is_realtime_stream_active())

    def test_segment_reuses_loaded_stream_models(self):
        stream_yolo = object()
        stream_plate_detector = object()
        stream_service = SimpleNamespace(
            yolo_person=stream_yolo,
            plate_detector=stream_plate_detector,
        )
        segment_ocr = SimpleNamespace(warmup=lambda: None)
        segment_service = VideoAIService(
            video_path="unused.mp4",
            show_window=False,
            segment_processing=True,
        )

        with patch.object(StreamAIService, "_instance", stream_service), patch(
            "services.video_ai_service.os.path.isfile",
            return_value=True,
        ), patch(
            "services.video_ai_service.get_shared_ocr_lane",
            return_value=segment_ocr,
        ):
            segment_service._load_models()

        self.assertIs(segment_service.yolo_person, stream_yolo)
        self.assertIs(segment_service.plate_detector, stream_plate_detector)
        self.assertIs(segment_service.plate_ocr, segment_ocr)

    def test_ffmpeg_startup_error_contains_stderr_diagnostic(self):
        self.recorder.camera_id = 1
        self.recorder.camera_name = "Camera 1"
        self.recorder.stream_url = "rtsp://camera/live"
        with patch(
            "services.camera_recorder_service.subprocess.Popen",
            _ExitedProcess,
        ), patch(
            "services.camera_recorder_service.threading.Thread",
            _ImmediateThread,
        ), patch(
            "services.camera_recorder_service.time.sleep",
        ):
            started = self.recorder._start_recording_session()

        self.assertFalse(started)
        self.assertEqual(self.recorder.status, "ERROR")
        self.assertIn("returncode 1", self.recorder.last_error)
        self.assertIn("connection refused", self.recorder.last_error)
        self.assertIn(
            "connection refused by camera",
            self.recorder.get_status()["ffmpeg_stderr"],
        )

    def test_load_state_recovers_processing_and_bounds_queue_without_duplicates(self):
        camera_dir = os.path.join(self.temp_dir.name, "cam_6")
        os.makedirs(camera_dir)
        self.recorder.camera_id = 6
        self.recorder.camera_dir = camera_dir
        self.recorder.state_file = os.path.join(
            camera_dir,
            "segment_queue_state.json",
        )
        self.recorder.max_queue_size = 1
        self.recorder._processing_queue = queue.Queue(maxsize=1)

        segments = {}
        for filename, status in (
            ("interrupted.mp4", "processing"),
            ("waiting.mp4", "pending"),
            ("already-done.mp4", "completed"),
        ):
            file_path = os.path.join(camera_dir, filename)
            with open(file_path, "wb") as segment_file:
                segment_file.write(b"recorded segment")
            segments[filename] = {
                "filename": filename,
                "file_path": file_path,
                "camera_id": 6,
                "status": status,
            }
        with open(self.recorder.state_file, "w", encoding="utf-8") as state_file:
            json.dump(segments, state_file)
        os.remove(segments["already-done.mp4"]["file_path"])

        self.recorder._load_state()
        self.recorder._load_state()

        self.assertEqual(
            self.recorder._segments_state["interrupted.mp4"]["status"],
            "pending",
        )
        self.assertEqual(
            self.recorder._segments_state["already-done.mp4"]["status"],
            "completed",
        )
        self.assertIn(
            "already-done.mp4",
            self.recorder._processed_segment_filenames,
        )
        self.assertEqual(self.recorder._processing_queue.qsize(), 1)
        self.assertEqual(
            self.recorder._queued_segment_filenames,
            {"interrupted.mp4"},
        )

        queued_item = self.recorder._processing_queue.get_nowait()
        self.recorder._queued_segment_filenames.discard(queued_item["filename"])
        self.recorder._processing_queue.task_done()
        self.recorder._segments_state["interrupted.mp4"]["status"] = "processing"
        self.recorder._repopulate_queue_from_backlog()
        self.assertEqual(self.recorder._processing_queue.qsize(), 1)
        self.assertEqual(
            self.recorder._processing_queue.get_nowait()["filename"],
            "waiting.mp4",
        )

    def test_worker_claims_segment_once_and_persists_completion(self):
        file_path = os.path.join(self.temp_dir.name, "segment.mp4")
        with open(file_path, "wb") as segment_file:
            segment_file.write(b"recorded segment")
        self.recorder.state_file = os.path.join(
            self.temp_dir.name,
            "segment_queue_state.json",
        )

        segment = {
            "filename": "segment.mp4",
            "file_path": file_path,
            "camera_id": 6,
            "status": "pending",
        }
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[segment["filename"]] = segment
        self.recorder.is_running = True
        self.recorder._enqueue_segment(segment)
        self.recorder._enqueue_segment(segment)
        execute_calls = []

        def execute_segment(_segment):
            execute_calls.append(_segment["filename"])
            self.recorder.is_running = False
            return {
                "detections": {
                    "event_groups": {"vehicle": 2, "person": 0},
                    "plate_captures": 1,
                    "persistence_errors": [],
                    "plate_history": [{
                        "text": "B1234CD",
                        "confidence": np.float32(0.94),
                        "bbox": np.array([10, 20, 40, 32]),
                        "crop": np.full((18, 46, 3), 211, dtype=np.uint8),
                    }],
                    "vehicle_events": [{
                        "track_id": np.int64(7),
                        "vehicle_crop": np.full((48, 64, 3), 180, dtype=np.uint8),
                    }],
                },
                "elapsed_seconds": 1.2,
                "processing_ratio": 0.12,
                "cpu_usage": {},
            }

        self.recorder._execute_segment_ai = execute_segment
        self.recorder._run_processing_worker()

        self.assertEqual(execute_calls, ["segment.mp4"])
        self.assertEqual(
            self.recorder._segments_state["segment.mp4"]["status"],
            "completed",
        )
        self.assertEqual(
            self.recorder._segments_state["segment.mp4"]["detections"]["plate_captures"],
            1,
        )
        runtime_plate_history = self.recorder._segments_state[
            "segment.mp4"
        ]["detections"]["plate_history"]
        self.assertTrue(
            np.array_equal(runtime_plate_history[0]["crop"], np.full(
                (18, 46, 3),
                211,
                dtype=np.uint8,
            ))
        )
        self.assertTrue(np.array_equal(
            self.recorder._segments_state["segment.mp4"]["detections"]
            ["vehicle_events"][0]["vehicle_crop"],
            np.full((48, 64, 3), 180, dtype=np.uint8),
        ))
        self.assertFalse(os.path.exists(file_path))
        with open(self.recorder.state_file, encoding="utf-8") as state_file:
            persisted_state = json.load(state_file)
        self.assertEqual(persisted_state["segment.mp4"]["status"], "completed")
        self.assertTrue(persisted_state["segment.mp4"]["video_file_deleted"])
        persisted_detections = persisted_state["segment.mp4"]["detections"]
        self.assertEqual(
            persisted_detections["plate_history"][0]["text"],
            "B1234CD",
        )
        self.assertAlmostEqual(
            persisted_detections["plate_history"][0]["confidence"],
            0.9399999976158142,
        )
        self.assertEqual(
            persisted_detections["plate_history"][0]["bbox"],
            [10, 20, 40, 32],
        )
        self.assertNotIn("crop", persisted_detections["plate_history"][0])
        self.assertEqual(
            persisted_detections["vehicle_events"][0]["track_id"],
            7,
        )
        self.assertNotIn(
            "vehicle_crop",
            persisted_detections["vehicle_events"][0],
        )
        self.assertFalse(self.recorder.worker_is_busy)
        self.assertEqual(self.recorder._processing_queue.unfinished_tasks, 0)

    def test_cleanup_skips_pending_processing_failed_and_unpersisted_segments(self):
        self.recorder.state_file = os.path.join(
            self.temp_dir.name,
            "segment_queue_state.json",
        )
        for status in ("pending", "processing", "failed", "completed"):
            filename = f"{status}.mp4"
            file_path = os.path.join(self.temp_dir.name, filename)
            with open(file_path, "wb") as segment_file:
                segment_file.write(b"keep for retry")
            segment = {
                "filename": filename,
                "file_path": file_path,
                "status": status,
            }
            self.recorder._segments_state[filename] = segment

            removed = self.recorder._cleanup_completed_segment(
                segment,
                status_persisted=(status != "completed"),
            )
            self.assertFalse(removed)
            self.assertTrue(os.path.isfile(file_path))

        capture_path = os.path.join(self.temp_dir.name, "plate_capture.jpg")
        with open(capture_path, "wb") as capture_file:
            capture_file.write(b"capture must be preserved")
        capture = {
            "filename": "plate_capture.jpg",
            "file_path": capture_path,
            "status": "completed",
        }
        self.recorder._segments_state[capture["filename"]] = capture
        self.assertFalse(
            self.recorder._cleanup_completed_segment(capture, status_persisted=True)
        )
        self.assertTrue(os.path.isfile(capture_path))

    def test_segment_inference_queue_waits_while_stream_queue_drops_old_frame(self):
        inference_queue = queue.Queue(maxsize=1)
        inference_queue.put("older")
        inference_stats = {"dropped_frames": 0}
        queued = threading.Event()

        def enqueue_segment_frame():
            VideoAIService._enqueue_inference_frame(
                inference_queue,
                "segment-frame",
                inference_stats,
                block=True,
            )
            queued.set()

        enqueue_thread = threading.Thread(target=enqueue_segment_frame)
        enqueue_thread.start()
        time.sleep(0.05)
        self.assertFalse(queued.is_set())
        self.assertEqual(inference_queue.get_nowait(), "older")
        enqueue_thread.join(timeout=2)
        self.assertTrue(queued.is_set())
        self.assertEqual(inference_queue.get_nowait(), "segment-frame")
        self.assertEqual(inference_stats["dropped_frames"], 0)

        inference_queue.put("old-stream-frame")
        VideoAIService._enqueue_inference_frame(
            inference_queue,
            "new-stream-frame",
            inference_stats,
            block=False,
        )
        self.assertEqual(inference_queue.get_nowait(), "new-stream-frame")
        self.assertEqual(inference_stats["dropped_frames"], 1)

    def test_plate_tracker_keeps_ocr_until_a_low_confidence_box_has_consistent_reads(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        plate_crop = np.full((18, 46, 3), 211, dtype=np.uint8)
        detection = {
            "bbox": [20, 30, 80, 50],
            "crop": plate_crop,
            "confidence": 0.40,
        }

        class FakeOCR:
            def __init__(self):
                self.read_count = 0

            def read(self, _image):
                self.read_count += 1
                return {
                    "text": "B1234CD",
                    "formatted": "B 1234 CD",
                    "confidence": 0.90,
                }

        ocr = FakeOCR()
        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_final_confidence=0.45,
            min_consistent_reads=2,
            single_read_ocr_confidence=0.72,
            single_read_detection_confidence=0.55,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        with redirect_stdout(StringIO()):
            tracker.update([detection], frame, ocr)
            self.assertFalse(tracker.tracks[1].is_stable)
            later_detection = dict(detection)
            later_detection["crop"] = np.full((18, 46, 3), 212, dtype=np.uint8)
            tracker.update([later_detection], frame, ocr)
            self.assertTrue(tracker.tracks[1].is_stable)
            for _ in range(tracker.max_frame_gap + 1):
                tracker.update([], frame, ocr)

        captures = tracker.consume_finished_captures()
        self.assertEqual(ocr.read_count, 2)
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["text"], "B1234CD")

    def test_video_plate_detector_crop_and_bbox_are_forwarded_to_ocr_tracker(self):
        padded_plate_crop = np.full((18, 46, 3), 211, dtype=np.uint8)

        class FakePlateDetector:
            def detect(self, roi):
                self.roi_shape = roi.shape
                return [{
                    "bbox": [3, 4, 23, 13],
                    "confidence": 0.94,
                    "crop": padded_plate_crop,
                }]

        service = object.__new__(VideoAIService)
        service.plate_detector = FakePlateDetector()
        service.plate_roi_cursor = 0
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        vehicles = [{
            "box": [20, 20, 120, 80],
            "cls": 2,
            "track_id": 5,
            "_event_track_id": 5,
            "_event_generation": 1,
        }]

        with patch("services.video_ai_service.DYNAMIC_FOCUS_ENABLED", False):
            detections = service._detect_plates_in_vehicles(
                frame,
                vehicles,
                video_time=1.0,
            )

        self.assertEqual(detections[0]["bbox"], [17, 18, 37, 27])
        self.assertIs(detections[0]["crop"], padded_plate_crop)

        class FakeOCR:
            def __init__(self):
                self.received = None

            def read(self, image):
                self.received = image.copy()
                return {
                    "text": "B1234CD",
                    "raw_text": "B1234CD",
                    "formatted": "B 1234 CD",
                    "confidence": 0.91,
                    "variant": "clahe",
                    "elapsed": 0.12,
                }

        ocr = FakeOCR()
        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_final_confidence=0.0,
            min_consistent_reads=1,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        output = StringIO()
        with redirect_stdout(output):
            tracker.update(detections, frame, ocr)

        self.assertTrue(np.array_equal(ocr.received, padded_plate_crop))
        self.assertIn("[OCR INPUT] Track #1 source=plate_detection", output.getvalue())
        self.assertIn("text='B1234CD' confidence=0.910", output.getvalue())

    def test_worker_refills_stored_backlog_after_each_completed_segment(self):
        self.recorder.max_queue_size = 1
        self.recorder._processing_queue = queue.Queue(maxsize=1)
        first = {
            "filename": "first.mp4",
            "camera_id": 6,
            "status": "pending",
        }
        waiting = {
            "filename": "waiting.mp4",
            "camera_id": 6,
            "status": "stored",
        }
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[first["filename"]] = first
            self.recorder._segments_state[waiting["filename"]] = waiting
        self.recorder.is_running = True
        self.recorder._enqueue_segment(first)
        execute_calls = []

        def execute_segment(segment):
            execute_calls.append(segment["filename"])
            if segment["filename"] == "waiting.mp4":
                self.recorder.is_running = False
            return {
                "detections": {
                    "event_groups": {"vehicle": 1, "person": 0},
                    "plate_captures": 0,
                    "persistence_errors": [],
                },
                "elapsed_seconds": 0.1,
                "processing_ratio": 0.01,
                "cpu_usage": {},
            }

        self.recorder._execute_segment_ai = execute_segment
        self.recorder._run_processing_worker()

        self.assertEqual(execute_calls, ["first.mp4", "waiting.mp4"])
        self.assertEqual(
            self.recorder._segments_state["first.mp4"]["status"],
            "completed",
        )
        self.assertEqual(
            self.recorder._segments_state["waiting.mp4"]["status"],
            "completed",
        )
        self.assertEqual(self.recorder._processing_queue.unfinished_tasks, 0)

    def test_worker_marks_database_persistence_errors_as_failed(self):
        file_path = os.path.join(self.temp_dir.name, "db-error.mp4")
        with open(file_path, "wb") as segment_file:
            segment_file.write(b"recorded segment")

        segment = {
            "filename": "db-error.mp4",
            "file_path": file_path,
            "camera_id": 6,
            "status": "pending",
        }
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[segment["filename"]] = segment
        self.recorder.is_running = True
        self.recorder._enqueue_segment(segment)

        def execute_segment(_segment):
            self.recorder.is_running = False
            return {
                "detections": {
                    "event_groups": {"vehicle": 1, "person": 0},
                    "plate_captures": 1,
                    "persistence_errors": [
                        {"operation": "save_plate_event", "error": "database offline"}
                    ],
                },
                "elapsed_seconds": 1.0,
                "processing_ratio": 0.1,
                "cpu_usage": {},
            }

        self.recorder._execute_segment_ai = execute_segment
        self.recorder._run_processing_worker()

        self.assertEqual(
            self.recorder._segments_state["db-error.mp4"]["status"],
            "failed",
        )
        self.assertIn(
            "persistence error",
            self.recorder._segments_state["db-error.mp4"]["error"],
        )
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(self.recorder._processing_queue.unfinished_tasks, 0)

    def test_monitor_retries_csv_segment_until_file_is_ready(self):
        camera_dir = os.path.join(self.temp_dir.name, "cam_6")
        os.makedirs(camera_dir)
        list_file = os.path.join(camera_dir, "segments.csv")
        segment_path = os.path.join(camera_dir, "late-segment.mp4")
        with open(list_file, "w", encoding="utf-8") as csv_file:
            csv_file.write("late-segment.mp4,0.0,10.0\n")
        self.recorder.camera_id = 6
        self.recorder.camera_name = "FM Noto"
        self.recorder.camera_dir = camera_dir
        self.recorder.state_file = os.path.join(
            camera_dir,
            "segment_queue_state.json",
        )
        self.recorder.is_running = True
        self.recorder.stop_event.clear()
        validation_attempts = 0
        sleep_calls = 0

        def validate_after_file_close(path):
            nonlocal validation_attempts
            validation_attempts += 1
            if validation_attempts == 1:
                return False
            with open(path, "wb") as segment_file:
                segment_file.write(b"complete segment bytes")
            return True

        def stop_after_retry(_delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 2:
                self.recorder.stop_event.set()

        with patch.object(
            self.recorder,
            "_validate_segment_file",
            side_effect=validate_after_file_close,
        ), patch("services.camera_recorder_service.time.sleep", side_effect=stop_after_retry):
            self.recorder._monitor_segments(list_file, camera_dir)

        self.assertEqual(validation_attempts, 2)
        self.assertEqual(
            self.recorder._segments_state["late-segment.mp4"]["status"],
            "pending",
        )
        self.assertEqual(self.recorder._processing_queue.qsize(), 1)

    def test_manual_processing_cannot_race_active_worker(self):
        file_path = os.path.join(self.temp_dir.name, "shared-segment.mp4")
        with open(file_path, "wb") as segment_file:
            segment_file.write(b"recorded segment")
        segment = {
            "filename": "shared-segment.mp4",
            "file_path": file_path,
            "camera_id": 6,
            "status": "pending",
        }
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[segment["filename"]] = segment
        self.recorder.is_running = True
        self.recorder._enqueue_segment(segment)
        inference_started = threading.Event()
        allow_inference_to_finish = threading.Event()
        execute_calls = []

        def execute_segment(_segment):
            execute_calls.append(_segment["filename"])
            inference_started.set()
            self.assertTrue(allow_inference_to_finish.wait(timeout=5))
            self.recorder.is_running = False
            return {
                "detections": {
                    "event_groups": {"vehicle": 1, "person": 0},
                    "plate_captures": 0,
                    "persistence_errors": [],
                },
                "elapsed_seconds": 1.0,
                "processing_ratio": 0.1,
                "cpu_usage": {},
            }

        self.recorder._execute_segment_ai = execute_segment
        worker = threading.Thread(target=self.recorder._run_processing_worker)
        worker.start()
        self.assertTrue(inference_started.wait(timeout=5))
        manual_result = self.recorder.process_single_segment(
            segment_filename="shared-segment.mp4"
        )
        allow_inference_to_finish.set()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertFalse(manual_result["success"])
        self.assertEqual(execute_calls, ["shared-segment.mp4"])
        self.assertEqual(
            self.recorder._segments_state["shared-segment.mp4"]["status"],
            "completed",
        )

    def test_stop_waits_for_active_segment_before_reporting_stopped(self):
        segment = {
            "filename": "in-flight.mp4",
            "camera_id": 6,
            "status": "pending",
        }
        with self.recorder._completed_segments_lock:
            self.recorder._segments_state[segment["filename"]] = segment
        self.recorder.is_running = True
        self.recorder._enqueue_segment(segment)
        inference_started = threading.Event()
        finish_inference = threading.Event()

        def execute_segment(_segment):
            inference_started.set()
            self.assertTrue(finish_inference.wait(timeout=5))
            return {
                "detections": {
                    "event_groups": {"vehicle": 1, "person": 0},
                    "plate_captures": 0,
                    "persistence_errors": [],
                },
                "elapsed_seconds": 0.2,
                "processing_ratio": 0.02,
                "cpu_usage": {},
            }

        self.recorder._execute_segment_ai = execute_segment
        self.recorder._worker_thread = threading.Thread(
            target=self.recorder._run_processing_worker
        )
        self.recorder._worker_thread.start()
        self.assertTrue(inference_started.wait(timeout=5))

        stopper = threading.Thread(target=self.recorder.stop)
        stopper.start()
        time.sleep(0.05)
        self.assertTrue(stopper.is_alive())
        self.assertEqual(self.recorder.status, "STOPPING")
        finish_inference.set()
        stopper.join(timeout=5)

        self.assertFalse(stopper.is_alive())
        self.assertEqual(self.recorder.status, "STOPPED")
        self.assertEqual(
            self.recorder._segments_state["in-flight.mp4"]["status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
