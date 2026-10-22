import os
import shutil
import tempfile
import time
import unittest
import numpy as np

import db
from ai.plate.ocr import (
    DEBUG_OCR_DIR,
    save_debug_crop_pair,
    get_valid_debug_ocr,
    get_debug_ocr_entry,
    _DEBUG_OCR_METADATA_CACHE,
)
from ai.plate.validator import is_valid_plate
from services.background_plate_ocr import BackgroundPlateOCR


class DebugOCRFlowTests(unittest.TestCase):
    def setUp(self):
        # Bersihkan cache in-memory sementara untuk isolasi tes
        self._orig_cache = dict(_DEBUG_OCR_METADATA_CACHE)
        _DEBUG_OCR_METADATA_CACHE.clear()

        # Dummy crop gambar (30x90 BGR)
        self.sample_crop = np.full((30, 90, 3), 180, dtype=np.uint8)

    def tearDown(self):
        _DEBUG_OCR_METADATA_CACHE.clear()
        _DEBUG_OCR_METADATA_CACHE.update(self._orig_cache)

    def test_save_debug_crop_creates_valid_metadata_json(self):
        # Simpan sampel plat valid "AB1234CD"
        context = "T101_V201"
        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="AB1234CD",
            conf=0.92,
            valid=True,
            context=context,
            raw_text="AB 1234 CD",
        )

        # Ambil kembali lewat fungsi get_valid_debug_ocr
        result = get_valid_debug_ocr(track_id=101, max_age_seconds=60.0)
        self.assertIsNotNone(result, "Metadata valid harus ditemukan!")
        self.assertEqual(result["plate_number"], "AB1234CD")
        self.assertAlmostEqual(result["confidence"], 0.92, places=2)
        self.assertTrue(result["is_valid"])
        self.assertIsNotNone(result["plate_crop"])
        self.assertIsInstance(result["metadata"], dict)
        self.assertEqual(result["metadata"]["status"], "VALID")
        self.assertEqual(result["metadata"]["track_id"], 101)

    def test_save_debug_crop_rejects_invalid_and_empty_labels(self):
        # Kasus 1: Teks angka saja ("8888") meskipun diberi flag valid=True
        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="8888",
            conf=0.95,
            valid=True,  # Caller salah kirim True
            context="T102_V202",
        )

        res_numeric = get_valid_debug_ocr(track_id=102, max_age_seconds=60.0)
        self.assertIsNone(res_numeric, "Teks angka 8888 tidak boleh lolos sebagai valid!")

        # Kasus 2: Teks EMPTY
        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="",
            conf=0.0,
            valid=False,
            context="T103_V203",
        )

        res_empty = get_valid_debug_ocr(track_id=103, max_age_seconds=60.0)
        self.assertIsNone(res_empty, "Teks EMPTY tidak boleh dijadikan hasil valid!")

        # Kasus 3: Teks bukan plat Indonesia ("MERE")
        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="MERE",
            conf=0.85,
            valid=False,
            context="T104_V204",
        )

        res_mere = get_valid_debug_ocr(track_id=104, max_age_seconds=60.0)
        self.assertIsNone(res_mere, "Teks MERE tidak boleh dijadikan hasil valid!")

    def test_single_capture_stores_all_candidates_in_one_json(self):
        # Simpan satu capture dengan banyak kandidat dari berbagai variant preprocessing
        capture_candidates = [
            {"text": "B1234CD", "raw_text": "B 1234 CD", "confidence": 0.94, "variant": "adaptive", "region_code": "B"},
            {"text": "81234CD", "raw_text": "8 1234 CD", "confidence": 0.72, "variant": "otsu", "region_code": None},
            {"text": "B1234CE", "raw_text": "B 1234 CE", "confidence": 0.65, "variant": "enhanced", "region_code": "B"},
        ]

        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="B1234CD",
            conf=0.94,
            valid=True,
            context="T105_V205",
            raw_text="B 1234 CD",
            candidates=capture_candidates,
        )

        entry = get_debug_ocr_entry(track_id=105, max_age_seconds=60.0)
        self.assertIsNotNone(entry, "Entri capture harus ditemukan!")
        self.assertIn("candidates", entry)
        cands = entry["candidates"]
        self.assertEqual(len(cands), 3, "Semua 3 kandidat harus tersimpan dalam 1 JSON!")

        # Kandidat 1: B1234CD -> valid
        self.assertEqual(cands[0]["text"], "B1234CD")
        self.assertTrue(cands[0]["valid"])
        self.assertEqual(cands[0]["status"], "VALID")

        # Kandidat 2: 81234CD -> invalid
        self.assertEqual(cands[1]["text"], "81234CD")
        self.assertFalse(cands[1]["valid"])
        self.assertEqual(cands[1]["status"], "INVALID")

    def test_recovery_of_valid_candidate_from_multi_candidate_json(self):
        # Skenario: Best variant membaca teks ambigu ("81234CD" - tidak valid),
        # namun variant lain pada capture yang sama menghasilkan bacaan valid ("B1234CD")
        multi_candidates = [
            {"text": "81234CD", "raw_text": "81234CD", "confidence": 0.70, "variant": "raw"},
            {"text": "B1234CD", "raw_text": "B 1234 CD", "confidence": 0.89, "variant": "enhanced"},
        ]

        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="81234CD",
            conf=0.70,
            valid=False,
            context="T106_V206",
            candidates=multi_candidates,
        )

        # get_valid_debug_ocr harus mampu memilih kandidat valid "B1234CD" dari JSON capture
        valid_res = get_valid_debug_ocr(track_id=106, max_age_seconds=60.0)
        self.assertIsNotNone(valid_res, "Harus memulihkan kandidat valid!")
        self.assertEqual(valid_res["plate_number"], "B1234CD")
        self.assertAlmostEqual(valid_res["confidence"], 0.89, places=2)
        self.assertTrue(valid_res["is_valid"])

    def test_database_status_sync_and_review(self):
        # Plat valid dengan confidence dan kualitas crop memadai -> Terbaca.
        status_valid = db.compute_plate_status("AB1234CD", 0.85, 0.90)
        self.assertEqual(status_valid, 1)

        # Plat valid tetapi confidence sedang -> Review / Perlu Cek.
        status_review_conf = db.compute_plate_status("AB1234CD", 0.55, 0.90)
        self.assertEqual(status_review_conf, 2)

        # Hasil OCR angka saja merupakan INVALID, bukan nomor plat untuk riwayat.
        status_numeric = db.compute_plate_status("8888", 0.95)
        self.assertEqual(status_numeric, 0)

        # Teks acak tanpa kandidat format plat juga INVALID.
        status_invalid_fmt = db.compute_plate_status("MERE", 0.90)
        self.assertEqual(status_invalid_fmt, 0)

    def test_background_ocr_reuses_valid_metadata_without_rerun(self):
        # Simpan plat valid untuk track_id = 999
        save_debug_crop_pair(
            raw_crop=self.sample_crop,
            prep_crop=self.sample_crop,
            text="B1234XYZ",
            conf=0.91,
            valid=True,
            context="T999_V888",
            raw_text="B1234XYZ",
        )

        bg_service = BackgroundPlateOCR()
        candidate = {
            "detection_id": 55555,
            "camera_id": 1,
            "track_id": 999,
            "vehicle_type": "car",
            "vehicle_image_path": "static/test_dummy.jpg",
            "event_key": "vehicle:1:group_888",
            "plate_detection_confidence": 0.88,
        }

        # Mock db.save_detection_event untuk memverifikasi pembaruan data
        saved_calls = []

        def mock_save_detection_event(**kwargs):
            saved_calls.append(kwargs)
            return {"status": "success", "event_id": 55555, "action": "updated"}

        original_save = db.save_detection_event
        db.save_detection_event = mock_save_detection_event

        try:
            # Panggil pemrosesan kendaraan
            bg_service._process_single_vehicle(candidate)

            self.assertEqual(len(saved_calls), 1, "Harus menyimpan/memperbarui satu event!")
            saved = saved_calls[0]
            # Teks plat diambil langsung dari metadata valid, bukan menebak nama file
            self.assertEqual(saved["plate_number"], "B1234XYZ")
            self.assertAlmostEqual(saved["ocr_conf"], 0.91, places=2)
            self.assertEqual(saved["event_key"], "vehicle:1:group_888")
            self.assertEqual(saved["object_type"], "vehicle")
            self.assertIsNotNone(saved["plate_crop"])
        finally:
            db.save_detection_event = original_save

    def test_background_ocr_falls_back_when_metadata_not_available(self):
        # Candidate dengan track_id yang TIDAK memiliki metadata valid
        candidate = {
            "detection_id": 66666,
            "camera_id": 1,
            "track_id": 77777,
            "vehicle_type": "motorcycle",
            "vehicle_image_path": "",  # Sengaja kosong agar jatuh ke increment_plate_retry
            "event_key": "vehicle:1:group_777",
        }

        retry_calls = []

        def mock_increment_retry(det_id):
            retry_calls.append(det_id)

        original_retry = db.increment_plate_retry
        db.increment_plate_retry = mock_increment_retry

        bg_service = BackgroundPlateOCR()
        try:
            bg_service._process_single_vehicle(candidate)
            self.assertEqual(retry_calls, [66666], "Harus fallback dan merekam retry jika tidak ada metadata & gambar kosong")
        finally:
            db.increment_plate_retry = original_retry


if __name__ == "__main__":
    unittest.main()
