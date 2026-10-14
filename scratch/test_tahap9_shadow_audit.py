import os
import sys
import tempfile
import importlib
from types import ModuleType
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from flask import Flask

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import services.camera_recorder_service as recorder_module
from services.shadow_dedup_service import ShadowDedupService


report_export_stub = ModuleType("report_export")
for export_name in (
    "build_detections_excel",
    "build_detections_pdf",
    "build_plates_excel",
    "build_plates_pdf",
    "build_recap_excel",
    "build_recap_pdf",
    "build_statistics_excel",
    "build_statistics_pdf",
    "build_cameras_excel",
    "build_cameras_pdf",
):
    setattr(report_export_stub, export_name, lambda *args, **kwargs: None)
with patch.dict(sys.modules, {"report_export": report_export_stub}):
    plate_bp = importlib.import_module("routes.plate").plate_bp


class _FakeVideoAIService:
    def __init__(self, **kwargs):
        self.source_name = kwargs["source_name"]
        self.session_id = "session-test"
        self.inference_count = 3
        self.model_load_seconds = 0.0
        self.person_capture_count = 0
        self.plate_capture_count = 1
        self.video_event_groups = {
            "vehicle": [{
                "event_track_id": 42,
                "generation": 2,
                "last_seen": 9.0,
                "vehicle_type": "car",
            }],
            "person": [],
        }
        self.last_plate_history = [{
            "plate": "B1234XYZ",
            "vehicle_track_id": 42,
            "vehicle_generation": 2,
            "track_id": 7,
        }]

    def _build_event_key(self, object_type, track_id, generation):
        return f"{object_type}:6:{track_id}:gen{generation}:sess_{self.session_id}"

    def run(self):
        return None


class ShadowAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = os.path.join(self.temp_dir.name, "audit.json")
        self.service = ShadowDedupService(self.audit_path)
        self.event_time = datetime(2026, 10, 13, 12, 0, 0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_decisions_have_auditable_provenance_and_no_claimed_insert(self):
        with patch.object(self.service, "_query_plate_history", return_value=([], None)):
            result = self.service.evaluate_candidate(
                camera_id=6,
                plate_number="B 1234 XYZ",
                timestamp=self.event_time,
                source_type="video",
                event_key="vehicle:6:42:gen2:sess_abcd",
                track_id=42,
                source_name="Segment Cam 6",
                metadata={
                    "event_origin": "segment_pipeline",
                    "event_time_quality": "video_time_estimate",
                    "event_time_basis": "segment_end_minus_video_relative_time",
                },
            )

        self.assertEqual(result["verdict"], "KEEP")
        self.assertEqual(result["camera_id"], 6)
        self.assertEqual(result["source_type"], "video")
        self.assertEqual(result["event_key"], "vehicle:6:42:gen2:sess_abcd")
        self.assertEqual(result["track_id"], 42)
        self.assertEqual(result["event_origin"], "segment_pipeline")
        self.assertEqual(result["rule_triggered"], "DEFAULT_KEEP")
        self.assertTrue(result["reason"])
        self.assertEqual(result["actual_action"], "SHADOW_ONLY_NO_BLOCK")
        self.assertEqual(result["database_insert_status"], "NOT_OBSERVED")

    def test_hold_audit_persists_across_restart_and_repeated_eval_is_idempotent(self):
        matched = {
            "detection_id": 735,
            "camera_id": 6,
            "source_type": "stream",
            "timestamp": self.event_time - timedelta(seconds=20),
            "event_key": "vehicle:6:17:gen1:sess_live",
            "track_id": 17,
        }
        with patch.object(self.service, "_query_plate_history", return_value=([matched], None)):
            first = self.service.evaluate_candidate(
                camera_id=6,
                plate_number="B1234XYZ",
                timestamp=self.event_time,
                source_type="video",
                event_key="vehicle:6:42:gen2:sess_segment",
                track_id=42,
                source_name="Segment Cam 6",
                metadata={"event_origin": "segment_pipeline"},
            )
            second = self.service.evaluate_candidate(
                camera_id=6,
                plate_number="B1234XYZ",
                timestamp=self.event_time,
                source_type="video",
                event_key="vehicle:6:42:gen2:sess_segment",
                track_id=42,
                source_name="Segment Cam 6",
                metadata={"event_origin": "segment_pipeline"},
            )

        self.assertEqual(first["verdict"], "HOLD")
        self.assertEqual(first["rule_triggered"], "CROSS_PATH_DUPLICATE_SHADOW")
        self.assertEqual(first["matched_detection_id"], 735)
        self.assertEqual(first["matched_camera_id"], 6)
        self.assertEqual(first["matched_source_type"], "stream")
        self.assertEqual(second["evaluation_id"], first["evaluation_id"])
        self.assertEqual(len(self.service._audit_records), 1)

        restored = ShadowDedupService(self.audit_path)
        summary = restored.get_audit_summary()
        self.assertEqual(summary["total_evaluated"], 1)
        self.assertEqual(summary["unique_decisions"], 1)
        self.assertEqual(summary["evidence_counts"]["live_hold"], 1)
        self.assertEqual(summary["shadow_verdicts"]["live_hold_count"], 1)
        self.assertEqual(summary["actual_storage"]["SHADOW_BLOCKED"], 0)
        self.assertEqual(summary["actual_storage"]["DB_INSERTS_OBSERVED"], 0)
        self.assertEqual(summary["false_positive_risk"]["live_hold"]["MEDIUM"], 1)
        self.assertIsNone(summary["false_positive_risk"]["confirmed_false_positives"])

    def test_segment_replay_with_new_tracking_session_keeps_one_audit_row(self):
        shared = {
            "camera_id": 6,
            "timestamp": self.event_time,
            "source_type": "video",
            "track_id": 42,
            "metadata": {
                "event_origin": "segment_pipeline",
                "segment_filename": "cam_6_segment_00001.mp4",
                "tracking_generation": 2,
            },
        }
        with patch.object(self.service, "_query_plate_history", return_value=([], None)):
            first = self.service.evaluate_candidate(
                **shared,
                plate_number="B1234XYZ",
                event_key="vehicle:6:42:gen2:sess_first",
            )
            second = self.service.evaluate_candidate(
                **shared,
                plate_number="B1234XY",
                event_key="vehicle:6:42:gen2:sess_after_restart",
            )

        self.assertEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertEqual(len(self.service._audit_records), 1)
        self.assertEqual(self.service._audit_records[0]["evaluation_count"], 2)
        self.assertEqual(self.service._audit_records[0]["plate"], "B1234XY")

    def test_recorder_audits_plate_linked_vehicle_once_with_segment_time(self):
        event_time = self.event_time + timedelta(seconds=9)
        matched = {
            "detection_id": "fixture-stream-735",
            "camera_id": 6,
            "source_type": "stream",
            "timestamp": event_time - timedelta(seconds=20),
            "track_id": 17,
            "event_key": "fixture:stream:17",
        }
        with tempfile.NamedTemporaryFile(delete=False) as video_file:
            video_file.write(b"test segment bytes")
            segment_path = video_file.name
        try:
            with patch.object(
                self.service,
                "_query_plate_history",
                return_value=([matched], None),
            ), patch.object(
                ShadowDedupService,
                "get_instance",
                return_value=self.service,
            ), patch.dict(
                sys.modules,
                {
                    "services.video_ai_service": ModuleType(
                        "services.video_ai_service"
                    )
                },
            ):
                sys.modules["services.video_ai_service"].VideoAIService = _FakeVideoAIService
                recorder = recorder_module.CameraRecorderService(
                    segments_dir=self.temp_dir.name,
                    segment_duration=10,
                )
                result = recorder._execute_segment_ai({
                    "camera_id": 6,
                    "camera_name": "Test Camera",
                    "filename": "cam_6_fixture_00000.mp4",
                    "file_path": segment_path,
                    "duration": 10,
                    "recorded_at": (self.event_time + timedelta(seconds=10)).isoformat(),
                })
        finally:
            os.remove(segment_path)

        evaluations = result["detections"]["shadow_dedup_evaluations"]
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0]["verdict"], "HOLD")
        self.assertEqual(evaluations[0]["event_origin"], "segment_pipeline")
        self.assertEqual(evaluations[0]["camera_id"], 6)
        self.assertEqual(evaluations[0]["track_id"], 42)
        self.assertEqual(evaluations[0]["plate_track_id"], 7)
        self.assertEqual(evaluations[0]["event_time_quality"], "video_time_estimate")
        self.assertEqual(evaluations[0]["matched_detection_id"], "fixture-stream-735")

    def test_missing_timestamp_and_history_error_are_kept_with_unknown_risk(self):
        missing_timestamp = self.service.evaluate_candidate(
            camera_id=6,
            plate_number="B1234XYZ",
            timestamp=None,
            source_type="video",
            event_key="vehicle:6:7:gen1:sess_missing",
            metadata={"event_origin": "segment_pipeline"},
        )
        self.assertEqual(missing_timestamp["verdict"], "KEEP")
        self.assertEqual(missing_timestamp["rule_triggered"], "MISSING_EVENT_TIMESTAMP_KEEP")
        self.assertEqual(missing_timestamp["false_positive_risk"], "UNKNOWN")
        self.assertIsNone(missing_timestamp["event_time"])

        with patch.object(
            self.service,
            "_query_plate_history",
            return_value=([], "database unavailable"),
        ):
            history_error = self.service.evaluate_candidate(
                camera_id=6,
                plate_number="C1234XYZ",
                timestamp=self.event_time,
                source_type="video",
                event_key="vehicle:6:8:gen1:sess_error",
            )
        self.assertEqual(history_error["verdict"], "KEEP")
        self.assertEqual(history_error["rule_triggered"], "HISTORY_UNAVAILABLE_KEEP")
        self.assertEqual(history_error["false_positive_risk"], "UNKNOWN")
        self.assertEqual(history_error["history_query_error"], "database unavailable")

    def test_summary_endpoint_reports_filtered_shadow_audit_and_runtime(self):
        matched = {
            "detection_id": "fixture-stream-1",
            "camera_id": 6,
            "source_type": "stream",
            "timestamp": self.event_time - timedelta(seconds=20),
            "track_id": 17,
            "event_key": "fixture:stream:17",
        }
        with patch.object(self.service, "_query_plate_history", return_value=([matched], None)):
            self.service.evaluate_candidate(
                camera_id=6,
                plate_number="B1234XYZ",
                timestamp=self.event_time,
                source_type="video",
                event_key="fixture:segment:42",
                track_id=42,
                metadata={"event_origin": "segment_pipeline"},
            )
        runtime = {
            "cpu_system_percent": 21.0,
            "cpu_process_percent": 35.0,
            "memory_process_mb": 512.0,
            "recorder": {"backlog": {"pending": 2}},
            "realtime": {"6": {"status": "ONLINE"}},
            "metrics_error": None,
        }
        with patch.object(
            self.service,
            "_get_runtime_metrics",
            return_value=runtime,
        ), patch.object(
            ShadowDedupService,
            "get_instance",
            return_value=self.service,
        ), patch.dict(os.environ, {"VIDEO_SEGMENT_DEDUP_ENABLED": "false"}):
            app = Flask(__name__)
            app.register_blueprint(plate_bp, url_prefix="/api")
            response = app.test_client().get(
                "/api/dedup/shadow/summary?camera_id=6&limit=1"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()["data"]
        self.assertEqual(payload["mode"], "SHADOW")
        self.assertFalse(payload["blocking_enabled"])
        self.assertEqual(payload["evidence_counts"]["live_pipeline_events"], 1)
        self.assertEqual(payload["evidence_counts"]["live_hold"], 1)
        self.assertEqual(payload["shadow_verdicts"]["live_hold_count"], 1)
        self.assertEqual(payload["false_positive_risk"]["live_hold"]["MEDIUM"], 1)
        self.assertEqual(payload["runtime"]["cpu_system_percent"], 21.0)
        self.assertEqual(payload["runtime"]["recorder"]["backlog"]["pending"], 2)
        self.assertEqual(payload["runtime"]["realtime"]["6"]["status"], "ONLINE")
        self.assertFalse(payload["activation_readiness"]["ready_for_blocking"])
        self.assertEqual(len(payload["recent_records"]), 1)

    def test_manual_candidate_endpoint_is_not_counted_as_live_event(self):
        with patch.object(
            ShadowDedupService,
            "get_instance",
            return_value=self.service,
        ):
            app = Flask(__name__)
            app.register_blueprint(plate_bp, url_prefix="/api")
            response = app.test_client().post(
                "/api/dedup/shadow/evaluate",
                json={
                    "camera_id": 6,
                    "plate_number": "B1234XYZ",
                    "timestamp": self.event_time.isoformat(),
                    "source_type": "stream",
                    "event_key": "manual-test-event",
                },
            )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()["data"]
        self.assertEqual(result["event_origin"], "api_candidate")
        self.assertFalse(result["event_is_live"])
        self.assertEqual(result["actual_action"], "SHADOW_ONLY_NO_BLOCK")


if __name__ == "__main__":
    unittest.main()
