import importlib
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.shadow_dedup_service import ShadowDedupService


video_module = importlib.import_module("services.video_ai_service")
VideoAIService = video_module.VideoAIService


class VideoSegmentDedupTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = ShadowDedupService(
            os.path.join(self.temp_dir.name, "audit.json")
        )
        self.now = datetime(2026, 10, 13, 14, 0, 0)
        self.event_key = "vehicle:6:42:gen2:sess_abc123"
        self.previous = {
            "detection_id": 735,
            "camera_id": 6,
            "source_type": "video",
            "timestamp": self.now - timedelta(seconds=20),
            "event_key": self.event_key,
            "track_id": 42,
            "plate": "B1234XYZ",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_segment_models_are_loaded_once_and_shared_between_camera_services(self):
        from services.video_ai_service import VideoAIService

        class FakeOCR:
            def warmup(self):
                return None

        VideoAIService._segment_model_cache.clear()
        try:
            with patch.object(video_module, "PERSON_MODEL_PATH", "person-model.pt"), patch.object(
                video_module,
                "PLATE_MODEL_PATH",
                "plate-model.pt",
            ), patch.object(video_module.os.path, "isfile", return_value=True), patch.object(
                video_module,
                "CAPTURE_DIR",
                self.temp_dir.name,
            ), patch.object(
                video_module,
                "PLATE_CAPTURE_DIR",
                self.temp_dir.name,
            ), patch.object(
                video_module,
                "YOLO",
                return_value=object(),
            ) as yolo, patch.object(
                video_module,
                "PlateDetector",
                return_value=object(),
            ) as plate_detector, patch.object(
                video_module,
                "PlateOCR",
                return_value=FakeOCR(),
            ) as plate_ocr:
                segment_services = []
                for camera_id in (1, 6, 7):
                    service = VideoAIService.__new__(VideoAIService)
                    service.segment_processing = True
                    service._load_models()
                    segment_services.append(service)

                self.assertEqual(yolo.call_count, 1)
                self.assertEqual(plate_detector.call_count, 1)
                self.assertEqual(plate_ocr.call_count, 1)
                self.assertIs(segment_services[0].yolo_person, segment_services[2].yolo_person)
                self.assertIs(segment_services[0].plate_detector, segment_services[1].plate_detector)
                self.assertIs(segment_services[0].plate_ocr, segment_services[2].plate_ocr)
        finally:
            VideoAIService._segment_model_cache.clear()

    def evaluate(self, **overrides):
        args = {
            "camera_id": 6,
            "plate_number": "B1234XYZ",
            "event_key": self.event_key,
            "track_id": 42,
            "timestamp": self.now,
            "plate_is_valid": True,
            "blocking_enabled": True,
            "segment_filename": "cam6_segment_001.mp4",
            "tracking_generation": 2,
        }
        args.update(overrides)
        return self.service.evaluate_segment_insert(**args)

    def test_same_valid_plate_camera_and_video_event_blocks_and_audits(self):
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([self.previous], None),
        ):
            decision = self.evaluate()

        self.assertEqual(decision["verdict"], "HOLD")
        self.assertEqual(decision["rule_triggered"], "SAME_VIDEO_EVENT_DUPLICATE")
        self.assertEqual(decision["actual_action"], "SEGMENT_INSERT_BLOCKED")
        self.assertEqual(
            decision["database_insert_status"],
            "NOT_ATTEMPTED_DEDUP_HOLD",
        )
        self.assertEqual(decision["event_key"], self.event_key)
        self.assertIn(self.event_key, decision["reason"])
        self.assertEqual(decision["matched_detection_id"], 735)
        self.assertTrue(os.path.isfile(self.service.audit_file_path))

    def test_repeated_evaluation_updates_one_audit_entry(self):
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([self.previous], None),
        ):
            first = self.evaluate()
            second = self.evaluate(timestamp=self.now + timedelta(seconds=1))

        self.assertEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertEqual(len(self.service._audit_records), 1)
        self.assertEqual(self.service._audit_records[0]["evaluation_count"], 2)

    def test_summary_distinguishes_real_segment_block_from_shadow_hold(self):
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([self.previous], None),
        ), patch.object(
            self.service,
            "_get_runtime_metrics",
            return_value={},
        ):
            self.evaluate()
            summary = self.service.get_audit_summary()

        self.assertEqual(summary["mode"], "LIMITED_VIDEO_SEGMENT_BLOCKING")
        self.assertTrue(summary["blocking_enabled"])
        self.assertEqual(summary["segment_blocking"]["blocked_insert_count"], 1)
        self.assertFalse(summary["segment_blocking"]["cross_path_blocking_enabled"])
        self.assertEqual(summary["actual_storage"]["INSERT_BLOCKED"], 1)
        self.assertEqual(summary["actual_storage"]["DB_INSERTS_OBSERVED"], 0)

    def test_missing_plate_ambiguous_identity_and_cross_path_are_kept(self):
        no_plate = self.evaluate(plate_number=None, plate_is_valid=False)
        ambiguous = self.evaluate(event_key=None)
        stream_event = dict(self.previous, source_type="stream")
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([stream_event], None),
        ):
            cross_path = self.evaluate()

        self.assertEqual(no_plate["verdict"], "KEEP")
        self.assertEqual(ambiguous["verdict"], "KEEP")
        self.assertEqual(cross_path["verdict"], "KEEP")
        self.assertNotEqual(cross_path["rule_triggered"], "SAME_VIDEO_EVENT_DUPLICATE")

    def test_only_exact_event_identity_inside_window_blocks(self):
        different_session = dict(
            self.previous,
            event_key="vehicle:6:42:gen2:sess_another",
        )
        other_camera = dict(self.previous, camera_id=7)
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([different_session, other_camera], None),
        ):
            decision = self.evaluate()
        self.assertEqual(decision["verdict"], "KEEP")

    def test_repeat_pass_and_outside_duplicate_window_are_kept(self):
        repeat_pass = dict(
            self.previous,
            timestamp=self.now - timedelta(seconds=301),
        )
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([repeat_pass], None),
        ):
            repeat_decision = self.evaluate()

        outside_window = dict(
            self.previous,
            timestamp=self.now - timedelta(seconds=60),
        )
        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([outside_window], None),
        ):
            outside_decision = self.evaluate()

        self.assertEqual(repeat_decision["verdict"], "KEEP")
        self.assertEqual(repeat_decision["rule_triggered"], "REPEAT_PASS_GENUINE")
        self.assertEqual(outside_decision["verdict"], "KEEP")

    def test_disabled_or_unavailable_dedup_fails_open(self):
        disabled = self.evaluate(blocking_enabled=False)
        self.assertEqual(disabled["verdict"], "KEEP")
        self.assertEqual(
            disabled["rule_triggered"],
            "VIDEO_SEGMENT_DEDUP_DISABLED_KEEP",
        )

        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([], "database unavailable"),
        ):
            unavailable = self.evaluate()
        self.assertEqual(unavailable["verdict"], "KEEP")
        self.assertEqual(unavailable["actual_action"], "DEDUP_ERROR_FAIL_OPEN")

    @staticmethod
    def make_video_service(segment_processing=True):
        ai = VideoAIService.__new__(VideoAIService)
        ai.camera_id = 6
        ai.session_id = "abc123"
        ai.person_capture_state = {}
        ai.source_name = "Segment Cam 6"
        ai.segment_processing = segment_processing
        ai.segment_filename = "cam6_segment_001.mp4"
        ai.segment_dedup_evaluations = []
        ai.track_lifecycle = {}
        ai.plate_tracker = SimpleNamespace(history=[])
        ai.plate_capture_count = 0
        ai.last_plate_capture = None
        ai.last_plate_history = []
        return ai

    @staticmethod
    def finished_capture():
        return {
            "text": "B1234XYZ",
            "formatted": "B 1234 XYZ",
            "confidence": 0.9,
            "detection_confidence": 0.9,
            "track_id": 7,
            "crop": object(),
            "bbox": (0, 0, 10, 10),
            "vehicle_track_id": 42,
            "vehicle_generation": 2,
            "vehicle_crop": SimpleNamespace(size=1),
        }

    def run_save_finished_capture(self, decision_method, segment_processing=True):
        ai = self.make_video_service(segment_processing)
        with patch.object(video_module, "save_plate_capture", return_value="plate.jpg"), patch.object(
            video_module.db,
            "save_detection_event",
            return_value={"detection_id": 1},
        ) as save_event, patch.object(
            ShadowDedupService,
            "get_instance",
            return_value=self.service,
        ), patch.dict(os.environ, {"VIDEO_SEGMENT_DEDUP_ENABLED": "true"}):
            with patch.object(
                self.service,
                "evaluate_segment_insert",
                side_effect=decision_method,
            ):
                result = ai._save_finished_capture(
                    self.finished_capture(),
                    frame=None,
                    vehicle_dets=[],
                    video_time=9.0,
                )
        return result, save_event, ai

    def test_storage_blocks_hold_but_saves_keep_and_dedup_failure(self):
        hold = {
            "verdict": "HOLD",
            "rule_triggered": "SAME_VIDEO_EVENT_DUPLICATE",
            "actual_action": "SEGMENT_INSERT_BLOCKED",
            "reason": "same event",
            "event_key": self.event_key,
        }
        held_result, held_save, held_ai = self.run_save_finished_capture(
            lambda **kwargs: dict(hold)
        )
        self.assertTrue(held_result["database_save_skipped"])
        held_save.assert_not_called()
        self.assertEqual(len(held_ai.segment_dedup_evaluations), 1)

        keep = {
            "verdict": "KEEP",
            "rule_triggered": "NO_STRONG_SAME_EVENT_DUPLICATE_KEEP",
            "actual_action": "SEGMENT_INSERT_NOT_BLOCKED",
            "reason": "new event",
        }
        kept_result, kept_save, _ = self.run_save_finished_capture(
            lambda **kwargs: dict(keep)
        )
        self.assertFalse(kept_result["database_save_skipped"])
        kept_save.assert_called_once()

        error_result, error_save, _ = self.run_save_finished_capture(
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("audit unavailable"))
        )
        self.assertFalse(error_result["database_save_skipped"])
        error_save.assert_called_once()

    def test_realtime_mode_does_not_call_segment_filter(self):
        keep = {"verdict": "HOLD", "rule_triggered": "SAME_VIDEO_EVENT_DUPLICATE"}
        result, save_event, _ = self.run_save_finished_capture(
            lambda **kwargs: dict(keep),
            segment_processing=False,
        )
        self.assertFalse(result["database_save_skipped"])
        save_event.assert_called_once()

    def test_environment_switch_disables_segment_blocking(self):
        ai = self.make_video_service(segment_processing=True)
        with patch.object(video_module, "save_plate_capture", return_value="plate.jpg"), patch.object(
            video_module.db,
            "save_detection_event",
            return_value={"detection_id": 1},
        ) as save_event, patch.object(
            ShadowDedupService,
            "get_instance",
            return_value=self.service,
        ), patch.object(
            self.service,
            "_query_plate_history",
        ) as query_history, patch.dict(
            os.environ,
            {"VIDEO_SEGMENT_DEDUP_ENABLED": "false"},
        ):
            result = ai._save_finished_capture(
                self.finished_capture(),
                frame=None,
                vehicle_dets=[],
                video_time=9.0,
            )

        self.assertFalse(result["database_save_skipped"])
        self.assertEqual(
            result["segment_dedup_decision"]["rule_triggered"],
            "VIDEO_SEGMENT_DEDUP_DISABLED_KEEP",
        )
        save_event.assert_called_once()
        query_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
