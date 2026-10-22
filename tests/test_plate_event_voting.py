import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from tracker import PlateTracker, Track
from services.stream_ai_service import StreamAIService
import db
from ai.plate.validator import is_valid_plate, correct_and_validate_plate


class PlateEventVotingTests(unittest.TestCase):
    def test_votes_per_character_using_confidence_and_crop_quality(self):
        track = Track(1, (0, 0, 80, 24), 1)
        low_quality_crop = np.full((24, 80, 3), 10, dtype=np.uint8)
        high_quality_crop = np.full((24, 80, 3), 240, dtype=np.uint8)

        track.tambah_bacaan_ocr(
            "B1234CD", 0.75, crop=low_quality_crop, crop_quality=0.35
        )
        track.tambah_bacaan_ocr(
            "B1234CE", 0.95, crop=low_quality_crop, crop_quality=0.35
        )
        track.tambah_bacaan_ocr(
            "B1234CD", 0.70, crop=high_quality_crop, crop_quality=0.95
        )

        result = track.hasil_voting()

        self.assertEqual(result["text"], "B1234CD")
        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_stable"])
        self.assertEqual(result["total_bacaan"], 3)
        self.assertGreater(result["consensus"], 0.8)
        self.assertLess(result["confidence_rata2"], 0.95)
        self.assertEqual(result["crop_quality"], 0.95)
        self.assertTrue(np.array_equal(track.best_crop, high_quality_crop))

    def test_single_high_confidence_read_remains_review(self):
        track = Track(2, (0, 0, 80, 24), 1)
        track.tambah_bacaan_ocr(
            "B1234CD",
            0.99,
            crop=np.full((24, 80, 3), 200, dtype=np.uint8),
            crop_quality=1.0,
        )
        tracker = PlateTracker(min_final_confidence=0.4)

        result = track.hasil_voting()

        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_stable"])
        self.assertFalse(tracker._hasil_layak_ditampilkan(result, track))

    def test_different_length_reads_do_not_create_false_stability(self):
        track = Track(3, (0, 0, 80, 24), 1)
        track.tambah_bacaan_ocr("B1234CD", 0.99, crop_quality=0.9)
        track.tambah_bacaan_ocr("B1234CDX", 0.80, crop_quality=0.9)

        result = track.hasil_voting()

        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_stable"])

    def test_confidence_alone_does_not_replace_better_event_result(self):
        current = {
            "text": "B1234CD",
            "ocr_conf": 0.85,
            "consensus": 0.95,
            "crop_quality": 0.90,
        }
        candidate = {
            "text": "B1234CE",
            "ocr_conf": 0.99,
            "consensus": 0.45,
            "crop_quality": 0.20,
        }

        self.assertFalse(
            StreamAIService._prefer_plate_event_result(current, candidate)
        )

    def test_same_plate_can_replace_crop_with_better_combined_score(self):
        current = {
            "text": "B1234CD",
            "ocr_conf": 0.70,
            "consensus": 0.80,
            "crop_quality": 0.45,
        }
        candidate = {
            "text": "B1234CD",
            "ocr_conf": 0.72,
            "consensus": 0.82,
            "crop_quality": 0.95,
        }

        self.assertTrue(
            StreamAIService._prefer_plate_event_result(current, candidate)
        )

    def test_indonesian_plate_validation_and_review_status(self):
        # Format plat valid Indonesia
        self.assertTrue(is_valid_plate("AB1234CD"))
        self.assertTrue(is_valid_plate("B123A"))
        self.assertTrue(is_valid_plate("D1987XYZ"))
        
        # Format plat tidak valid / meragukan
        self.assertFalse(is_valid_plate("18102513"))
        self.assertFalse(is_valid_plate("MERE"))
        self.assertFalse(is_valid_plate("URA"))
        self.assertFalse(is_valid_plate("1531"))

        # Status 1 (Terbaca) hanya jika format valid dan conf >= 0.70
        self.assertEqual(db.compute_plate_status("AB1234CD", 0.85, 0.90), 1)
        # Status 2 (Perlu Cek / Review) jika format valid tapi conf < 0.70
        self.assertEqual(db.compute_plate_status("AB1234CD", 0.55, 0.90), 2)
        # Format yang jelas tidak mungkin menjadi plat disimpan sebagai INVALID.
        self.assertEqual(db.compute_plate_status("MERE", 0.75), 0)
        self.assertEqual(db.compute_plate_status("18102513", 0.80), 0)

    def test_dedup_merges_same_plate_within_window_and_preserves_distinct_events(self):
        class MockCursor:
            def __init__(self, sample_rows):
                self._rows = sample_rows

            def execute(self, sql, params=None):
                pass

            def fetchall(self):
                return list(self._rows)

        base_time = datetime(2026, 10, 22, 10, 0, 0)
        sample_rows = [
            # Kejadian 1 - bacaan terbaik (conf 0.95)
            {
                "plate_id": 101,
                "plate_number": "AB1234CD",
                "ocr_confidence": 0.95,
                "detection_confidence": 0.88,
                "camera_id": 1,
                "event_key": "vehicle:1:group_1",
                "created_at": base_time + timedelta(seconds=2),
            },
            # Kejadian 1 - bacaan lain frame sebelumnya (conf 0.70)
            {
                "plate_id": 102,
                "plate_number": "AB1234CD",
                "ocr_confidence": 0.70,
                "detection_confidence": 0.80,
                "camera_id": 1,
                "event_key": "vehicle:1:group_2",
                "created_at": base_time,
            },
            # Kejadian 2 - kendaraan yang sama lewat lagi 15 menit kemudian (conf 0.92)
            {
                "plate_id": 103,
                "plate_number": "AB1234CD",
                "ocr_confidence": 0.92,
                "detection_confidence": 0.85,
                "camera_id": 1,
                "event_key": "vehicle:1:group_10",
                "created_at": base_time + timedelta(minutes=15),
            },
            # Kejadian 3 - kendaraan lain dengan plat berbeda (conf 0.88)
            {
                "plate_id": 104,
                "plate_number": "B5678EF",
                "ocr_confidence": 0.88,
                "detection_confidence": 0.82,
                "camera_id": 1,
                "event_key": "vehicle:1:group_3",
                "created_at": base_time + timedelta(seconds=4),
            },
        ]

        cur = MockCursor(sample_rows)
        deduped_ids = db._dedup_plate_ids(cur, "1=1", [])

        self.assertEqual(len(deduped_ids), 3)
        self.assertIn(101, deduped_ids)
        self.assertNotIn(102, deduped_ids)
        self.assertIn(103, deduped_ids)
        self.assertIn(104, deduped_ids)

    def test_vehicle_temporary_candidates_aggregation_and_review_retention(self):
        sharp_crop = np.full((30, 90, 3), 200, dtype=np.uint8)
        blur_crop = np.full((30, 90, 3), 50, dtype=np.uint8)

        # Skenario 1: Satu kendaraan menghasilkan beberapa bacaan plat berbeda
        candidates = [
            {"text": "B1234CD", "confidence": 0.75, "crop_quality": 0.35, "crop": blur_crop},
            {"text": "B1234CE", "confidence": 0.90, "crop_quality": 0.40, "crop": blur_crop},
            {"text": "B1234CD", "confidence": 0.85, "crop_quality": 0.95, "crop": sharp_crop},
        ]

        aggregated = StreamAIService._aggregate_vehicle_candidates(candidates)

        # Karakter 'D' menang voting atas 'E', crop paling tajam terpilih, bukti kuat -> stable = True
        self.assertEqual(aggregated["text"], "B1234CD")
        self.assertTrue(aggregated["stable"])
        self.assertFalse(aggregated["review_required"])
        self.assertTrue(np.array_equal(aggregated["crop"], sharp_crop))
        self.assertEqual(aggregated["crop_quality"], 0.95)

        # Skenario 2: Bukti tidak cukup (hanya 1 bacaan) -> pertahankan status review
        single_candidate = [
            {"text": "B1234CD", "confidence": 0.99, "crop_quality": 0.90, "crop": sharp_crop}
        ]
        agg_single = StreamAIService._aggregate_vehicle_candidates(single_candidate)
        self.assertEqual(agg_single["text"], "B1234CD")
        self.assertFalse(agg_single["stable"])
        self.assertTrue(agg_single["review_required"])

        # Skenario 3: Bacaan berkonflik tanpa konsensus yang cukup -> tetap review
        conflicting = [
            {"text": "B1234CD", "confidence": 0.70, "crop_quality": 0.50, "crop": blur_crop},
            {"text": "B9999ZZ", "confidence": 0.70, "crop_quality": 0.50, "crop": blur_crop},
        ]
        agg_conflicting = StreamAIService._aggregate_vehicle_candidates(conflicting)
        self.assertFalse(agg_conflicting["stable"])
        self.assertTrue(agg_conflicting["review_required"])


    def test_valid_indonesian_plates_are_accepted(self):
        valid_samples = [
            "B1", "D1", "RI1", "D123", "AB1234CD",
            "H99Z", "DK8888XX", "KT1000A", "G1234AB",
            "B 1", "AB 1234 CD", "L2"
        ]
        for plate in valid_samples:
            res = correct_and_validate_plate(plate)
            self.assertTrue(res["valid"], f"Plat valid {plate} tertolak!")
            self.assertTrue(is_valid_plate(plate), f"is_valid_plate gagal untuk {plate}")
            self.assertTrue(db.is_valid_vehicle_plate(plate), f"is_valid_vehicle_plate gagal untuk {plate}")
            # Status harus 1 jika confidence dan kualitas crop memadai.
            self.assertEqual(db.compute_plate_status(plate, 0.85, 0.90), 1)

    def test_purely_numeric_texts_are_rejected_and_status_review(self):
        numeric_samples = ["1531", "18102513", "8888", "1234", "0123", "99999"]
        for num_text in numeric_samples:
            res = correct_and_validate_plate(num_text)
            self.assertFalse(res["valid"], f"Teks angka {num_text} tidak boleh valid!")
            self.assertFalse(is_valid_plate(num_text), f"is_valid_plate harus False untuk {num_text}")
            self.assertFalse(db.is_valid_vehicle_plate(num_text), f"is_valid_vehicle_plate harus False untuk {num_text}")
            # Teks angka tidak boleh diubah otomatis menjadi plat (seperti BB88)
            self.assertEqual(res["candidates"], [])
            # Angka murni adalah INVALID pada confidence berapa pun.
            self.assertEqual(db.compute_plate_status(num_text, 0.95), 0)
            self.assertEqual(db.compute_plate_status(num_text, 0.50), 0)

    def test_invalid_format_ocr_results_are_rejected_and_status_review(self):
        invalid_samples = [
            "MERE", "URA", "18IE", "XYZ999", "123AB",
            "A", "BB", "B0", "POLISI"
        ]
        for inv_text in invalid_samples:
            res = correct_and_validate_plate(inv_text)
            self.assertFalse(res["valid"], f"Teks invalid {inv_text} tidak boleh valid!")
            self.assertFalse(is_valid_plate(inv_text), f"is_valid_plate harus False untuk {inv_text}")
            self.assertFalse(db.is_valid_vehicle_plate(inv_text), f"is_valid_vehicle_plate harus False untuk {inv_text}")
            self.assertEqual(db.compute_plate_status(inv_text, 0.90), 0)
            self.assertEqual(db.compute_plate_status(inv_text, 0.45), 0)

    def test_ambiguous_candidates_remain_review_and_quality_gates_readable(self):
        ambiguous = correct_and_validate_plate("ZZ1234ZZ")
        self.assertFalse(ambiguous["valid"])
        self.assertTrue(ambiguous["candidates"])
        self.assertEqual(db.compute_plate_status("ZZ1234ZZ", 0.95, 0.95), 2)

        # Secara pola L4EDC mungkin merupakan plat, tetapi crop buruk tidak boleh
        # dinaikkan menjadi TERBACA hanya karena confidence OCR tinggi.
        self.assertTrue(is_valid_plate("L4EDC"))
        self.assertEqual(db.compute_plate_status("L4EDC", 0.95, 0.30), 2)

        self.assertEqual(db.compute_plate_status("AB1234CD", 0.95, 0.30), 2)
        self.assertEqual(db.compute_plate_status("AB1234CD", 0.95, 0.90), 1)

    def test_wiko_vision_false_positives_are_invalid(self):
        for candidate in ("19172219", "OA"):
            self.assertFalse(is_valid_plate(candidate))
            self.assertEqual(db.compute_plate_status(candidate, 0.99, 0.99), 0)
        self.assertFalse(is_valid_plate("MERE"))
        self.assertEqual(db.compute_plate_status("MERE", 0.99, 0.99), 0)

    def test_invalid_legacy_history_hides_plate_text_but_keeps_raw_ocr(self):
        class MockCursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, *_args):
                pass

            def fetchall(self):
                return [
                    {
                        "plate_id": 1,
                        "plate_number": "19172219",
                        "detection_status": 2,
                        "ocr_confidence": 0.99,
                        "plate_crop_quality": 0.95,
                        "raw_ocr_text": "19172219",
                        "detected_at": datetime(2026, 10, 22, 10, 0),
                    },
                    {
                        "plate_id": 2,
                        "plate_number": "ZZ1234ZZ",
                        "detection_status": 2,
                        "ocr_confidence": 0.80,
                        "plate_crop_quality": 0.80,
                        "raw_ocr_text": "ZZ1234ZZ",
                        "detected_at": datetime(2026, 10, 22, 10, 1),
                    },
                ]

        class MockConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return MockCursor()

        with (
            patch.object(db, "get_db", return_value=MockConnection()),
            patch.object(db, "_dedup_plate_ids", return_value=[1, 2]),
        ):
            result = db.get_plate_history_paginated()

        invalid, ambiguous = result["items"]
        self.assertEqual(invalid["plate"], "-")
        self.assertEqual(invalid["status_code"], 0)
        self.assertEqual(invalid["raw_ocr_text"], "19172219")
        self.assertEqual(ambiguous["plate"], "ZZ1234ZZ")
        self.assertEqual(ambiguous["status_code"], 2)

    def test_all_three_plate_statuses_are_persisted_without_overrides(self):
        class MockCursor:
            def __init__(self):
                self.statements = []
                self.lastrowid = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, sql, params=None):
                self.statements.append((sql, params))
                if "INSERT INTO plate (" in sql:
                    self.lastrowid = 10
                elif "INSERT INTO full_detection (" in sql:
                    self.lastrowid = 20

            def fetchone(self):
                return None

        class MockConnection:
            def __init__(self):
                self.mock_cursor = MockCursor()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return self.mock_cursor

            def close(self):
                pass

        connection = MockConnection()
        with (
            patch.object(db, "get_db", return_value=connection),
            patch.object(db, "save_crop_locally", return_value="test/plate.jpg"),
        ):
            invalid_result = db.save_detection_event(
                camera_id=1,
                plate_number="OA",
                raw_ocr_text="OA",
                ocr_conf=0.99,
                plate_crop_quality=0.95,
                ocr_stable=False,
                event_key="invalid-ocr-test",
            )
            readable_result = db.save_detection_event(
                camera_id=1,
                plate_number="AB1234CD",
                raw_ocr_text="AB1234CD",
                ocr_conf=0.99,
                plate_crop_quality=0.95,
                ocr_stable=False,
                event_key="readable-ocr-test",
            )
            review_result = db.save_detection_event(
                camera_id=1,
                plate_number="ZZ1234ZZ",
                raw_ocr_text="ZZ1234ZZ",
                ocr_conf=0.99,
                plate_crop_quality=0.95,
                ocr_stable=False,
                event_key="ambiguous-ocr-test",
            )
            failed_result = db.save_detection_event(
                camera_id=1,
                plate_crop=object(),
                ocr_conf=0.0,
                plate_crop_quality=0.0,
                event_key="empty-ocr-test",
            )

        plate_inserts = [
            params for sql, params in connection.mock_cursor.statements
            if "INSERT INTO plate (" in sql
        ]
        invalid_insert, readable_insert, ambiguous_insert, failed_insert = plate_inserts
        self.assertIsNone(invalid_insert[0])
        self.assertEqual(invalid_insert[1], "OA")
        self.assertEqual(invalid_insert[2], "OA")
        self.assertEqual(invalid_insert[4], 0)
        self.assertIsNone(invalid_result["plate_number"])
        self.assertEqual(readable_insert[0], "AB1234CD")
        self.assertEqual(readable_insert[4], 1)
        self.assertEqual(readable_result["plate_number"], "AB1234CD")
        self.assertEqual(ambiguous_insert[0], "ZZ1234ZZ")
        self.assertEqual(ambiguous_insert[4], 2)
        self.assertEqual(review_result["plate_number"], "ZZ1234ZZ")
        self.assertIsNone(failed_insert[0])
        self.assertIsNone(failed_insert[1])
        self.assertEqual(failed_insert[4], 0)
        self.assertEqual(failed_insert[7], "test/plate.jpg")
        self.assertIsNone(failed_result["plate_number"])

    def test_validation_before_saving_to_history(self):
        # OCR angka murni adalah INVALID walau confidence tinggi.
        status_numeric = db.compute_plate_status("8888", 0.92)
        self.assertEqual(status_numeric, 0)

        # Skenario 2: Hasil OCR plat valid ("AB1234CD") dengan confidence 0.88
        status_valid = db.compute_plate_status("AB1234CD", 0.88, 0.90)
        self.assertEqual(status_valid, 1)

        # Skenario 3: Plat valid 1 digit ("B1") dengan confidence 0.75
        status_legal_vip = db.compute_plate_status("B1", 0.75, 0.90)
        self.assertEqual(status_legal_vip, 1)

        # Skenario 4: Plat kosong atau "-"
        self.assertEqual(db.compute_plate_status("", 0.95), 0)
        self.assertEqual(db.compute_plate_status("-", 0.95), 0)


if __name__ == "__main__":
    unittest.main()
