import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.camera_recorder_service import CameraRecorderService


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


if __name__ == "__main__":
    unittest.main()
