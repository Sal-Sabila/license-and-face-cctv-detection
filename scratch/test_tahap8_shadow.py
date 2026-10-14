"""
Test Script Tahap 8 — Skenario kontrol shadow deduplication.

Kandidat KEEP/HOLD dalam skrip ini adalah fixture terkontrol, bukan event CCTV live.
Skrip ini tidak menyimpulkan akurasi pada data CCTV dan tidak mengklaim INSERT sukses.
"""

import atexit
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import db
from services.shadow_dedup_service import ShadowDedupService

REPORT_PATH = os.path.join(PROJECT_ROOT, "scratch", "tahap8_shadow_dedup_report.json")
_TEST_AUDIT_DIR = tempfile.TemporaryDirectory(prefix="tahap8-shadow-")
atexit.register(_TEST_AUDIT_DIR.cleanup)
AUDIT_LOG_TEST_PATH = os.path.join(_TEST_AUDIT_DIR.name, "audit.json")


def run_tahap8_tests():
    print("=" * 75)
    print("  TAHAP 8 — PENGUJIAN SHADOW DEDUPLICATION SERVICE LAYER")
    print("=" * 75)

    svc = ShadowDedupService(AUDIT_LOG_TEST_PATH)

    now = datetime.now()
    test_results = []
    with db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM full_detection")
            row_count_before = int(cur.fetchone()["cnt"])
    
    # ── Test 1: Controlled Stream Source Immunity ─────────────────────────────
    print("\n[TEST 1] Controlled stream-source immunity (bukan event CCTV)...")
    r1 = svc.evaluate_candidate(
        camera_id=6,
        plate_number="AB1234XY",
        timestamp=now,
        source_type="stream",
        object_type="vehicle",
        vehicle_type="car",
        metadata={"event_origin": "controlled_test"},
    )
    passed1 = (
        r1["verdict"] == "KEEP"
        and r1["actual_action"] == "SHADOW_ONLY_NO_BLOCK"
        and r1["database_insert_status"] == "NOT_OBSERVED"
        and not r1["event_is_live"]
        and r1["rule_triggered"] == "STREAM_SOURCE_IMMUNE"
    )
    print(f"  Verdict: {r1['verdict']} | Actual: {r1['actual_action']} | Rule: {r1['rule_triggered']} -> {'PASS' if passed1 else 'FAIL'}")
    test_results.append({"test": "Stream Immunity", "passed": passed1, "details": r1})

    # ── Test 2: Unplated Vehicle Kept ─────────────────────────────────────────
    print("\n[TEST 2] Unplated Vehicle Kept...")
    r2 = svc.evaluate_candidate(
        camera_id=6,
        plate_number=None,
        timestamp=now,
        source_type="video",
        object_type="vehicle",
        vehicle_type="motorcycle",
        metadata={"event_origin": "controlled_test"},
    )
    passed2 = (
        r2["verdict"] == "KEEP"
        and r2["actual_action"] == "SHADOW_ONLY_NO_BLOCK"
        and not r2["event_is_live"]
        and r2["rule_triggered"] == "UNPLATED_VEHICLE_KEPT"
    )
    print(f"  Verdict: {r2['verdict']} | Actual: {r2['actual_action']} | Rule: {r2['rule_triggered']} -> {'PASS' if passed2 else 'FAIL'}")
    test_results.append({"test": "Unplated Vehicle Kept", "passed": passed2, "details": r2})

    # ── Test 3: Cross-Path Controlled Fixture ─────────────────────────────────
    print("\n[TEST 3] Cross-path fixture (simulasi eksplisit, bukan event CCTV)...")
    stream_time = now - timedelta(seconds=20)
    controlled_history = ([{
        "detection_id": "CONTROLLED-STREAM-1",
        "camera_id": 6,
        "source_type": "stream",
        "timestamp": stream_time,
        "track_id": 101,
        "event_key": "controlled:stream:101",
    }], None)
    with patch.object(svc, "_query_plate_history", return_value=controlled_history):
        r3 = svc.evaluate_candidate(
            camera_id=6,
            plate_number="TEST123",
            timestamp=stream_time + timedelta(seconds=10),
            source_type="video",
            object_type="vehicle",
            vehicle_type="car",
            event_key="controlled:video:202",
            track_id=202,
            source_name="Controlled Test Fixture",
            metadata={"event_origin": "controlled_test"},
        )
    passed3 = (
        r3["verdict"] == "HOLD"
        and r3["actual_action"] == "SHADOW_ONLY_NO_BLOCK"
        and not r3["event_is_live"]
        and "CROSS_PATH" in r3["rule_triggered"]
    )
    print(f"  Verdict: {r3['verdict']} | Actual: {r3['actual_action']} | Rule: {r3['rule_triggered']} | Matched ID: {r3.get('matched_detection_id')} -> {'PASS' if passed3 else 'FAIL'}")
    test_results.append({"test": "Cross-Path Shadow Hold", "passed": passed3, "details": r3})

    # ── Test 4: Controlled Repeat-Pass Rule ───────────────────────────────────
    print("\n[TEST 4] Repeat-pass rule fixture (simulasi eksplisit)...")
    repeat_time = stream_time + timedelta(seconds=900)
    with patch.object(svc, "_query_plate_history", return_value=controlled_history):
        r4 = svc.evaluate_candidate(
            camera_id=6,
            plate_number="TEST123",
            timestamp=repeat_time,
            source_type="video",
            object_type="vehicle",
            vehicle_type="car",
            event_key="controlled:video:303",
            track_id=303,
            source_name="Controlled Test Fixture",
            metadata={"event_origin": "controlled_test"},
        )
    passed4 = (
        r4["verdict"] == "KEEP"
        and r4["actual_action"] == "SHADOW_ONLY_NO_BLOCK"
        and not r4["event_is_live"]
        and r4["rule_triggered"] == "REPEAT_PASS_GENUINE"
    )
    print(f"  Verdict: {r4['verdict']} | Actual: {r4['actual_action']} | Rule: {r4['rule_triggered']} | Gap: {r4.get('delta_seconds')}s -> {'PASS' if passed4 else 'FAIL'}")
    test_results.append({"test": "Repeat Pass Genuine", "passed": passed4, "details": r4})

    # ── Test 5: Controlled Candidate Is Not Reported as Live ──────────────────
    print("\n[TEST 5] Kandidat kontrol tidak dihitung sebagai event pipeline...")
    with patch.object(svc, "_query_plate_history", return_value=([], None)):
        r5 = svc.evaluate_candidate(
            camera_id=6,
            plate_number="TEST123",
            timestamp=repeat_time + timedelta(seconds=10),
            source_type="video",
            object_type="vehicle",
            event_key="controlled:video:404",
            track_id=404,
            source_name="Controlled Test Fixture",
            metadata={"event_origin": "controlled_test"},
        )
    passed5 = (
        r5["event_origin"] == "controlled_test"
        and not r5["event_is_live"]
        and r5["actual_action"] == "SHADOW_ONLY_NO_BLOCK"
    )
    print(f"  Verdict: {r5['verdict']} | Actual: {r5['actual_action']} | Rule: {r5['rule_triggered']} | Delta: {r5.get('delta_seconds')}s -> {'PASS' if passed5 else 'FAIL'}")
    test_results.append({"test": "Controlled Candidate Live Classification", "passed": passed5, "details": r5})

    # ── Test 6: Audit Log Summary ─────────────────────────────────────────────
    print("\n[TEST 6] Verifikasi Audit Log Summary...")
    summary = svc.get_audit_summary()
    print(f"  Total dievaluasi : {summary['total_evaluated']}")
    print(f"  Shadow KEEP      : {summary['shadow_verdicts']['KEEP']}")
    print(f"  Shadow HOLD      : {summary['shadow_verdicts']['HOLD']} ({summary['shadow_verdicts']['hold_rate_percent']}%)")
    print(f"  Pipeline event   : {summary['evidence_counts']['live_pipeline_events']}")
    print(f"  Live HOLD        : {summary['shadow_verdicts']['live_hold_count']}")
    print(
        f"  Shadow blockers  : {summary['actual_storage']['INSERT_BLOCKED']} "
        "(INSERT success is not observed by this evaluator)"
    )
    passed6 = (
        summary["total_evaluated"] >= 5
        and summary["evidence_counts"]["live_pipeline_events"] == 0
        and summary["actual_storage"]["INSERT_BLOCKED"] == 0
    )
    test_results.append({"test": "Audit Log Summary Integrity", "passed": passed6, "summary": summary})

    # ── Test 7: Verifikasi Zero Database Mutation ─────────────────────────────
    print("\n[TEST 7] Verifikasi Zero Mutation pada Database real_cctv...")
    with db.get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM full_detection")
            current_count = cur.fetchone()["cnt"]
            # Memastikan tidak ada tabel dedup liar yang dibuat
            cur.execute("SELECT COUNT(*) as cnt FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='real_cctv' AND TABLE_NAME='shadow_dedup_log'")
            table_created = cur.fetchone()["cnt"]
    
    passed7 = (table_created == 0 and current_count == row_count_before)
    print(f"  Row count full_detection : {row_count_before} -> {current_count}")
    print(f"  Tabel liar dibuat         : {table_created} (Harus 0)")
    print(f"  Status                    : {'PASS' if passed7 else 'FAIL'}")
    test_results.append({"test": "Zero Database Mutation", "passed": passed7, "row_count": current_count})

    # Simpan Laporan JSON
    all_passed = all(t["passed"] for t in test_results)
    final_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tahap": 8,
        "mode": "SHADOW_DRY_RUN",
        "blocking_enabled": False,
        "evidence_type": "controlled_test_fixtures_not_live_cctv_events",
        "all_tests_passed": all_passed,
        "test_results": test_results,
        "audit_summary": summary,
        "database_mutated": False,
        "row_count_verified": current_count,
        "database_row_count_before": row_count_before,
        "recommendation": (
            "Skenario kontrol hanya memverifikasi aturan dan tidak membuktikan tingkat akurasi "
            "pada event CCTV live atau keberhasilan INSERT per event. Pertahankan mode shadow "
            "sampai HOLD pipeline nyata ditinjau terhadap ground-truth."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)

    print("\n" + "=" * 75)
    print(f"  HASIL AKHIR TAHAP 8: {'SEMUA TEST LULUS (ALL PASSED)' if all_passed else 'ADA TEST GAGAL'}")
    print(f"  Laporan tersimpan di: {REPORT_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    run_tahap8_tests()
