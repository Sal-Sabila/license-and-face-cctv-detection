import concurrent.futures
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from ai.plate.ocr_worker import SharedPlateOCRWorkers
from services.stream_ai_service import StreamAIService
from tracker import PlateTracker


class OCRWorkerLaneTests(unittest.TestCase):
    def test_workers_share_one_model_and_stream_preempts_queued_segment_sample(self):
        first_segment_started = threading.Event()
        release_first_segment = threading.Event()
        calls = []
        active_calls = 0
        max_active_calls = 0

        class FakeOCR:
            def read(self, image):
                nonlocal active_calls, max_active_calls
                marker = int(image[0, 0, 0])
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
                calls.append(marker)
                try:
                    if marker == 1:
                        first_segment_started.set()
                        self.assert_release()
                    time.sleep(0.01)
                    return {"text": f"B{marker:04d}CD", "confidence": 0.9}
                finally:
                    active_calls -= 1

            @staticmethod
            def assert_release():
                if not release_first_segment.wait(timeout=2):
                    raise TimeoutError("test did not release first segment OCR")

        factory_calls = []

        def create_ocr():
            factory_calls.append(True)
            return FakeOCR()

        manager = SharedPlateOCRWorkers(
            ocr_factory=create_ocr,
            stream_queue_size=2,
            segment_queue_size=2,
        )
        segment = manager.lane("segment")
        stream = manager.lane("stream")
        try:
            first = segment.submit(np.full((12, 32, 3), 1, dtype=np.uint8))
            self.assertTrue(first_segment_started.wait(timeout=2))
            second = segment.submit(np.full((12, 32, 3), 2, dtype=np.uint8))
            realtime = stream.submit(np.full((12, 32, 3), 9, dtype=np.uint8))
            release_first_segment.set()

            self.assertEqual(first.result(timeout=2)["text"], "B0001CD")
            self.assertEqual(realtime.result(timeout=2)["text"], "B0009CD")
            self.assertEqual(second.result(timeout=2)["text"], "B0002CD")
            self.assertEqual(calls, [1, 9, 2])
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(max_active_calls, 1)
            metrics = manager.stats()
            self.assertEqual(metrics["stream"]["completed"], 1)
            self.assertEqual(metrics["segment"]["completed"], 2)
            self.assertEqual(metrics["stream"]["backlog"], 0)
            self.assertEqual(metrics["segment"]["backlog"], 0)
        finally:
            release_first_segment.set()
            manager.shutdown()

    def test_segment_crop_is_not_retried_in_place_and_stream_gets_next_turn(self):
        first_segment_started = threading.Event()
        release_first_segment = threading.Event()
        calls = []

        class FakeOCR:
            def read(self, image):
                marker = int(image[0, 0, 0])
                calls.append(marker)
                if marker == 1 and calls.count(1) == 1:
                    first_segment_started.set()
                    if not release_first_segment.wait(timeout=2):
                        raise TimeoutError("test did not release OCR")
                    return {
                        "text": "bad",
                        "confidence": 0.95,
                        "is_indonesia_pattern": False,
                    }
                return {
                    "text": "B1234CD",
                    "confidence": 0.95,
                    "is_indonesia_pattern": True,
                }

        manager = SharedPlateOCRWorkers(
            ocr_factory=FakeOCR,
            stream_queue_size=2,
            segment_queue_size=2,
        )
        segment = manager.lane("segment")
        stream = manager.lane("stream")
        try:
            segment_future = segment.submit(
                np.full((12, 32, 3), 1, dtype=np.uint8)
            )
            self.assertTrue(first_segment_started.wait(timeout=2))
            stream_future = stream.submit(
                np.full((12, 32, 3), 9, dtype=np.uint8)
            )
            release_first_segment.set()

            self.assertEqual(stream_future.result(timeout=2)["text"], "B1234CD")
            self.assertEqual(segment_future.result(timeout=2)["text"], "bad")
            self.assertEqual(calls, [1, 9])
            self.assertEqual(manager.stats("segment")["retries"], 0)
        finally:
            release_first_segment.set()
            manager.shutdown()

    def test_stream_queue_is_served_before_waiting_segment_queue(self):
        active_started = threading.Event()
        release_active = threading.Event()
        calls = []

        class FakeOCR:
            def read(self, image):
                marker = int(image[0, 0, 0])
                calls.append(marker)
                if marker == 1:
                    active_started.set()
                    if not release_active.wait(timeout=2):
                        raise TimeoutError("test did not release OCR")
                return {"text": f"B{marker:04d}CD", "confidence": 0.9}

        manager = SharedPlateOCRWorkers(
            ocr_factory=FakeOCR,
            stream_queue_size=8,
            segment_queue_size=2,
        )
        stream, segment = manager.lane("stream"), manager.lane("segment")
        try:
            first_segment = segment.submit(
                np.full((12, 32, 3), 1, dtype=np.uint8)
            )
            self.assertTrue(active_started.wait(timeout=2))
            waiting_segment = segment.submit(
                np.full((12, 32, 3), 2, dtype=np.uint8)
            )
            stream_futures = [
                stream.submit(np.full((12, 32, 3), marker, dtype=np.uint8))
                for marker in (9, 10, 11, 12)
            ]
            release_active.set()

            for future in [first_segment, waiting_segment, *stream_futures]:
                future.result(timeout=2)
            self.assertEqual(calls, [1, 9, 10, 11, 12, 2])
            self.assertEqual(manager.stats("segment")["dropped"], 0)
        finally:
            release_active.set()
            manager.shutdown()

    def test_segment_submission_waits_for_capacity_without_dropping_work(self):
        active_started = threading.Event()
        release_active = threading.Event()
        submit_finished = threading.Event()
        submitted = []

        class FakeOCR:
            def read(self, image):
                marker = int(image[0, 0, 0])
                if marker == 1:
                    active_started.set()
                    if not release_active.wait(timeout=2):
                        raise TimeoutError("test did not release OCR")
                return {"text": f"B{marker:04d}CD", "confidence": 0.9}

        manager = SharedPlateOCRWorkers(
            ocr_factory=FakeOCR,
            stream_queue_size=1,
            segment_queue_size=1,
        )
        segment = manager.lane("segment")
        try:
            first = segment.submit(np.full((12, 32, 3), 1, dtype=np.uint8))
            self.assertTrue(active_started.wait(timeout=2))
            second = segment.submit(np.full((12, 32, 3), 2, dtype=np.uint8))

            def submit_third():
                submitted.append(
                    segment.submit(np.full((12, 32, 3), 3, dtype=np.uint8))
                )
                submit_finished.set()

            producer = threading.Thread(target=submit_third)
            producer.start()
            self.assertFalse(submit_finished.wait(timeout=0.05))
            release_active.set()
            producer.join(timeout=2)
            self.assertTrue(submit_finished.is_set())

            self.assertEqual(first.result(timeout=2)["text"], "B0001CD")
            self.assertEqual(second.result(timeout=2)["text"], "B0002CD")
            self.assertEqual(submitted[0].result(timeout=2)["text"], "B0003CD")
            stats = manager.stats("segment")
            self.assertEqual(stats["submitted"], 3)
            self.assertEqual(stats["completed"], 3)
            self.assertEqual(stats["dropped"], 0)
        finally:
            release_active.set()
            manager.shutdown()

    def test_ocr_inference_error_is_visible_and_counted_as_failed(self):
        class FailingOCR:
            def read(self, _image):
                raise ValueError("inference failed")

        manager = SharedPlateOCRWorkers(ocr_factory=FailingOCR)
        stream = manager.lane("stream")
        try:
            future = stream.submit(np.full((12, 32, 3), 4, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "inference failed"):
                future.result(timeout=2)
            metrics = stream.stats()
            self.assertEqual(metrics["errors"], 1)
            self.assertEqual(metrics["failed"], 1)
            self.assertEqual(metrics["pending"], 0)
            self.assertEqual(metrics["processing"], 0)
        finally:
            manager.shutdown()

    def test_stream_queue_is_bounded_and_discards_an_older_queued_frame(self):
        active_started = threading.Event()
        release_active = threading.Event()

        class FakeOCR:
            def read(self, image):
                marker = int(image[0, 0, 0])
                if marker == 1:
                    active_started.set()
                    if not release_active.wait(timeout=2):
                        raise TimeoutError("test did not release OCR")
                return {"text": f"B{marker:04d}CD", "confidence": 0.9}

        manager = SharedPlateOCRWorkers(
            ocr_factory=FakeOCR,
            stream_queue_size=1,
            segment_queue_size=1,
        )
        stream = manager.lane("stream")
        try:
            active = stream.submit(np.full((12, 32, 3), 1, dtype=np.uint8))
            self.assertTrue(active_started.wait(timeout=2))
            old_queued = stream.submit(
                np.full((12, 32, 3), 2, dtype=np.uint8),
                coalesce_key="older",
            )
            newest = stream.submit(
                np.full((12, 32, 3), 3, dtype=np.uint8),
                coalesce_key="newest",
            )
            self.assertEqual(manager.stats("stream")["backlog"], 1)
            self.assertEqual(manager.stats("stream")["backlog_peak"], 1)
            self.assertEqual(manager.stats("stream")["dropped"], 1)
            release_active.set()

            self.assertEqual(active.result(timeout=2)["text"], "B0001CD")
            self.assertTrue(old_queued.cancelled())
            self.assertEqual(newest.result(timeout=2)["text"], "B0003CD")
        finally:
            release_active.set()
            manager.shutdown()

    def test_segment_tracker_votes_from_several_best_frame_samples(self):
        class FakeSegmentLane:
            source = "segment"

            def submit(self, _image, retry=False):
                future = concurrent.futures.Future()
                future.set_result({
                    "text": "B1234CD",
                    "raw_text": "B1234CD",
                    "formatted": "B 1234 CD",
                    "confidence": 0.92,
                    "elapsed": 0.01,
                })
                return future

        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_final_confidence=0.45,
            min_consistent_reads=2,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        detection = {
            "bbox": [20, 30, 80, 50],
            "crop": np.full((24, 72, 3), 210, dtype=np.uint8),
            "confidence": 0.9,
        }
        lane = FakeSegmentLane()
        for index in range(5):
            sample = dict(detection)
            sample["crop"] = np.full(
                (24, 72, 3),
                210 + index,
                dtype=np.uint8,
            )
            tracker.update([sample], frame, lane)
        for _ in range(tracker.max_frame_gap + 2):
            tracker.update([], frame, lane)

        captures = tracker.consume_finished_captures()
        self.assertEqual(len(captures), 1)
        self.assertGreaterEqual(captures[0]["total_bacaan"], 2)
        self.assertEqual(captures[0]["text"], "B1234CD")

    def test_segment_tracker_discards_identical_crop_samples(self):
        class FakeSegmentLane:
            source = "segment"

            def submit(self, _image, retry=False):
                future = concurrent.futures.Future()
                future.set_result({"text": "B1234CD", "confidence": 0.92})
                return future

        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_final_confidence=0.45,
            min_consistent_reads=1,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        detection = {
            "bbox": [20, 30, 80, 50],
            "crop": np.full((24, 72, 3), 210, dtype=np.uint8),
            "confidence": 0.9,
        }
        lane = FakeSegmentLane()
        for _ in range(5):
            tracker.update([detection], frame, lane)

        track = next(iter(tracker.tracks.values()))
        self.assertEqual(len(track.ocr_candidates), 1)
        self.assertEqual(len(track.ocr_crop_fingerprints), 1)

    def test_stream_tracker_votes_distinct_crops_until_consistent(self):
        class FakeStreamLane:
            source = "stream"

            def __init__(self):
                self.read_count = 0

            def submit(self, _image, coalesce_key=None):
                self.read_count += 1
                future = concurrent.futures.Future()
                future.set_result({
                    "text": "B1234CD",
                    "raw_text": "B1234CD",
                    "formatted": "B 1234 CD",
                    "confidence": 0.90,
                    "is_indonesia_pattern": True,
                })
                return future

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
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        detections = [{
            "bbox": [20, 30, 80, 50],
            "confidence": 0.40,
        } | {
            "crop": np.full((24, 72, 3), marker, dtype=np.uint8),
        } for marker in (210, 211, 212)]
        lane = FakeStreamLane()

        for detection in detections:
            tracker.update([detection], frame, lane)
        for _ in range(tracker.max_frame_gap + 2):
            tracker.update([], frame, lane)

        captures = tracker.consume_finished_captures()
        self.assertEqual(lane.read_count, 2)
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["text"], "B1234CD")
        self.assertEqual(captures[0]["total_bacaan"], 2)

    def test_stream_tracker_retries_fresh_crop_after_pending_ocr(self):
        class FakeStreamLane:
            source = "stream"

            def __init__(self):
                self.futures = []
                self.images = []

            def submit(self, image, coalesce_key=None):
                future = concurrent.futures.Future()
                self.futures.append(future)
                self.images.append(int(image[0, 0, 0]))
                return future

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
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        lane = FakeStreamLane()

        def detection(marker):
            return {
                "bbox": [20, 30, 80, 50],
                "crop": np.full((24, 72, 3), marker, dtype=np.uint8),
                "confidence": 0.40,
                "vehicle_track_id": 7,
            }

        tracker.update([detection(210)], frame, lane)
        tracker.update([detection(211)], frame, lane)
        self.assertEqual(len(lane.futures), 1)

        lane.futures[0].set_result({
            "text": "B1234CD",
            "raw_text": "B1234CD",
            "formatted": "B 1234 CD",
            "confidence": 0.90,
            "is_indonesia_pattern": True,
        })
        tracker.update([detection(210)], frame, lane)
        self.assertEqual(len(lane.futures), 2)
        self.assertEqual(lane.images, [210, 211])
        self.assertEqual(tracker.pipeline_counts["ocr_duplicate_crop_skipped"], 1)

        lane.futures[1].set_result({
            "text": "B1234CD",
            "raw_text": "B1234CD",
            "formatted": "B 1234 CD",
            "confidence": 0.90,
            "is_indonesia_pattern": True,
        })
        tracker.flush_ocr(lane, timeout=1.0)

        captures = tracker.consume_finished_captures()
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["total_bacaan"], 2)

    def test_stream_tracker_limits_ocr_attempts_per_track(self):
        class FakeStreamLane:
            source = "stream"

            def __init__(self):
                self.read_count = 0

            def submit(self, _image, coalesce_key=None):
                self.read_count += 1
                future = concurrent.futures.Future()
                future.set_result(None)
                return future

        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        lane = FakeStreamLane()
        for marker in range(210, 216):
            tracker.update([{
                "bbox": [20, 30, 80, 50],
                "crop": np.full((24, 72, 3), marker, dtype=np.uint8),
                "confidence": 0.40,
                "vehicle_track_id": 7,
            }], frame, lane)

        self.assertEqual(lane.read_count, 3)
        self.assertEqual(tracker.pipeline_counts["ocr_retry_limit_reached"], 1)

    def test_stream_tracker_matches_jittered_plate_boxes_by_vehicle_track_id(self):
        class FakeStreamLane:
            source = "stream"

            def submit(self, _image, coalesce_key=None):
                future = concurrent.futures.Future()
                future.set_result({
                    "text": "B1234CD",
                    "raw_text": "B1234CD",
                    "formatted": "B 1234 CD",
                    "confidence": 0.90,
                    "is_indonesia_pattern": True,
                })
                return future

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
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        lane = FakeStreamLane()
        first_detection = {
            "bbox": [20, 10, 80, 30],
            "crop": np.full((24, 72, 3), 210, dtype=np.uint8),
            "confidence": 0.40,
            "vehicle_track_id": 7,
        }
        jittered_detection = {
            "bbox": [20, 60, 80, 80],
            "crop": np.full((24, 72, 3), 211, dtype=np.uint8),
            "confidence": 0.40,
            "vehicle_track_id": 7,
        }

        tracker.update([first_detection], frame, lane)
        tracker.update([jittered_detection], frame, lane)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.pipeline_counts["matched_by_vehicle_track_id"], 1)

        for _ in range(tracker.max_frame_gap + 2):
            tracker.update([], frame, lane)

        captures = tracker.consume_finished_captures()
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["text"], "B1234CD")
        self.assertEqual(captures[0]["total_bacaan"], 2)
        self.assertEqual(captures[0]["vehicle_track_id"], 7)

    def test_stream_tracker_does_not_merge_distinct_or_expired_vehicle_tracks(self):
        class FakeStreamLane:
            source = "stream"

            def submit(self, _image, coalesce_key=None):
                future = concurrent.futures.Future()
                future.set_result(None)
                return future

        tracker = PlateTracker(
            ocr_every_n_matches=1,
            min_detection_confidence=0.0,
            min_ocr_confidence=0.0,
            min_ocr_width=1,
            min_ocr_height=1,
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        lane = FakeStreamLane()

        def detection(box, vehicle_track_id):
            return {
                "bbox": box,
                "crop": np.full((24, 72, 3), 210, dtype=np.uint8),
                "confidence": 0.40,
                "vehicle_track_id": vehicle_track_id,
            }

        tracker.update([
            detection([10, 10, 50, 30], 7),
            detection([70, 10, 110, 30], 8),
        ], frame, lane)
        self.assertEqual(len(tracker.tracks), 2)

        tracker.update([detection([10, 60, 50, 80], 7)], frame, lane)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.tracks[1].last_seen_frame, 2)
        self.assertEqual(tracker.tracks[2].last_seen_frame, 1)

        for _ in range(tracker.max_frame_gap + 1):
            tracker.update([], frame, lane)
        tracker.update([detection([10, 60, 50, 80], 7)], frame, lane)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(next(iter(tracker.tracks.values())).id, 3)

    def test_stream_plate_crop_and_ocr_result_are_forwarded_to_database(self):
        service = object.__new__(StreamAIService)
        service.event_session_id = "test-session"
        service.captured_tracks = {}
        service.saved_event_meta = {}
        service.ai_cycle_counter = 0
        service.stream_save_totals = {
            "db_event_attempted": 0,
            "db_event_inserted": 0,
            "db_event_enriched": 0,
            "db_event_duplicate": 0,
            "db_event_errors": 0,
            "plate_event_attempted": 0,
            "plate_event_persisted": 0,
            "plate_association_missed": 0,
            "plate_without_ocr": 0,
        }

        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        plate_crop = np.full((18, 46, 3), 211, dtype=np.uint8)
        vehicle = {
            "cls": 2,
            "track_id": 7,
            "object_group_id": "vehicle-group-7",
            "box": [10, 10, 110, 90],
            "conf": 0.91,
        }
        plate = {
            "vehicle_object_group_id": "vehicle-group-7",
            "vehicle_track_id": 7,
            "bbox": [40, 55, 75, 68],
            "text": "B1234CD",
            "raw_text": "B1234CD",
            "conf": 0.95,
            "ocr_conf": 0.92,
            "valid": True,
            "crop": plate_crop,
        }

        with patch(
            "services.stream_ai_service.db.get_camera_direction",
            return_value="IN",
        ), patch(
            "services.stream_ai_service.db.save_detection_event",
            return_value={
                "duplicate": False,
                "detection_id": 101,
                "plate_id": 55,
                "has_plate": True,
            },
        ) as save_event:
            service._save_consistent_events(
                frame,
                [vehicle],
                [plate],
                camera_id=6,
                current_time=123.0,
            )

        save_event.assert_called_once()
        saved_event = save_event.call_args.kwargs
        self.assertEqual(saved_event["plate_number"], "B1234CD")
        self.assertIs(saved_event["plate_crop"], plate_crop)
        self.assertEqual(saved_event["source_type"], "stream")
        self.assertEqual(service.stream_save_totals["plate_event_attempted"], 1)
        self.assertEqual(service.stream_save_totals["plate_event_persisted"], 1)
        self.assertEqual(service.stream_save_totals["db_event_inserted"], 1)

    def test_stream_plate_detection_reports_roi_yolo_and_ocr_ready_counts(self):
        class FakePlateDetector:
            def detect(self, _roi):
                self.last_metrics = {
                    "yolo_boxes": 1,
                    "class_rejected": 0,
                    "width_rejected": 0,
                    "height_rejected": 0,
                    "aspect_rejected": 0,
                    "invalid_box": 0,
                    "empty_crop": 0,
                    "accepted": 1,
                    "confidence_min": 0.92,
                    "confidence_max": 0.92,
                }
                return [{
                    "bbox": [20, 20, 60, 32],
                    "confidence": 0.92,
                    "crop": np.full((14, 44, 3), 210, dtype=np.uint8),
                }]

        service = object.__new__(StreamAIService)
        service.plate_detector = FakePlateDetector()
        service.plate_roi_cursor = {}
        service.last_plate_detection_metrics = {}
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        vehicles = [{
            "box": [20, 20, 120, 80],
            "cls": 2,
            "track_id": 5,
            "distance_zone": "NEAR",
            "object_group_id": "vehicle-5",
        }]

        detections = service._detect_plates_in_vehicles(
            frame,
            vehicles,
            camera_id=6,
            current_time=123.0,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(service.last_plate_detection_metrics["rois_selected"], 1)
        self.assertEqual(service.last_plate_detection_metrics["yolo_boxes"], 1)
        self.assertEqual(service.last_plate_detection_metrics["detector_accepted"], 1)
        self.assertEqual(service.last_plate_detection_metrics["ocr_ready"], 1)


if __name__ == "__main__":
    unittest.main()
