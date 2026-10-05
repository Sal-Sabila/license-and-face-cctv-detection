import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from services.stream_ai_service import (
    StreamAIService,
    PERSON_DEDUP_COOLDOWN,
    PERSON_TRACK_DEDUP_COOLDOWN,
    PERSON_HAMMING_THRESHOLD,
    OBJECT_GROUP_MAX_GAP,
    OBJECT_GROUP_PERSON_MAX_DIST,
    PERSON_AREA_TOLERANCE,
    PERSON_PROXIMITY_DIST,
)

def run_tests():
    print("=== TESTING PERSON ANTI-DUPLICATION SYSTEM ===")
    print(f"PERSON_DEDUP_COOLDOWN: {PERSON_DEDUP_COOLDOWN}s")
    print(f"PERSON_TRACK_DEDUP_COOLDOWN: {PERSON_TRACK_DEDUP_COOLDOWN}s")
    print(f"PERSON_HAMMING_THRESHOLD: {PERSON_HAMMING_THRESHOLD}")
    print(f"OBJECT_GROUP_MAX_GAP: {OBJECT_GROUP_MAX_GAP}s")
    print(f"OBJECT_GROUP_PERSON_MAX_DIST: {OBJECT_GROUP_PERSON_MAX_DIST}px")
    print(f"PERSON_AREA_TOLERANCE: {PERSON_AREA_TOLERANCE}")
    print(f"PERSON_PROXIMITY_DIST: {PERSON_PROXIMITY_DIST}px")

    # Create dummy instance without starting background thread
    class DummyService(StreamAIService):
        def __init__(self):
            # initialize required structures only
            self.captured_tracks = {}
            self.saved_event_meta = {}
            self.object_groups = {}
            self._visual_hashes = {}
            self._person_track_cache = {}
            self._person_alias_keys = {}
            self.event_session_id = "test_session_123"
            self.track_lifecycle = {}
            self.camera_states = {}

    service = DummyService()
    now = time.time()
    camera_id = 1

    # Fake visual hash 64-bit
    # Hash A (64 bits)
    hash_a = "0000ffff0000ffff"
    # Hash A' with 2 bits difference (Hamming = 2)
    hash_a_near = "0000ffff0000fffc"
    # Hash B completely different (Hamming = 32)
    hash_b = "ffff0000ffff0000"

    print(f"\nHamming test:")
    print(f"Dist(A, A): {service._hamming_distance(hash_a, hash_a)}")
    print(f"Dist(A, A_near): {service._hamming_distance(hash_a, hash_a_near)}")
    print(f"Dist(A, B): {service._hamming_distance(hash_a, hash_b)}")
    assert service._hamming_distance(hash_a, hash_a_near) == 2
    assert service._hamming_distance(hash_a, hash_b) >= 20

    # 1. Simulasikan Person A pertama kali masuk
    # Box: [100, 100, 160, 260], centroid=(130, 180), area=60*160=9600
    person_1 = {
        "box": [100, 100, 160, 260],
        "track_id": 5,
        "object_group_id": "person_group_001",
        "conf": 0.60,
        "v_hash": hash_a,
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_1, now)
    print(f"\nTest 1 (New Person A): is_dup={is_dup} (Expected: False)")
    assert not is_dup, "Person 1 should be treated as new!"

    # Daftarkan Person A ke saved_event_meta
    event_key_1 = f"person:{camera_id}:person_group_001:sess_{service.event_session_id}"
    service.captured_tracks[event_key_1] = now
    service.saved_event_meta[event_key_1] = {
        "object_type": "person",
        "camera_id": camera_id,
        "event_key": event_key_1,
        "object_group_id": "person_group_001",
        "track_id": 5,
        "tracks_seen": {5},
        "last_centroid": (130.0, 180.0),
        "last_bbox": [100, 100, 160, 260],
        "area": 9600,
        "v_hash": hash_a,
        "first_seen": now,
        "last_seen": now,
        "person_confidence": 0.60,
        "detection_id": 101,
        "updated_at": now,
    }
    service._register_visual_hash(camera_id, "person", hash_a, now)
    service._person_track_cache[f"person-track:{camera_id}:5"] = {"ts": now, "text": "5"}

    # 2. Test 2: Frame berikutnya (0.8s kemudian), group sama
    now_2 = now + 0.8
    person_1_next = {
        "box": [105, 102, 165, 262],
        "track_id": 5,
        "object_group_id": "person_group_001",
        "conf": 0.65,
        "v_hash": hash_a,
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_1_next, now_2)
    print(f"Test 2 (Same group & track): is_dup={is_dup}, reason={info['reason']} (Expected: True, SAME_GROUP)")
    assert is_dup and info["reason"] == "SAME_GROUP"

    # 3. Test 3: ByteTrack track switch (track_id berubah dari 5 ke 12, group_id berubah karena tracking hilang sejenak)
    # Posisi geser 25px: centroid (150, 190), bbox [120, 110, 180, 270], v_hash mirip (hamming=2)
    now_3 = now + 2.5
    person_switch = {
        "box": [120, 110, 180, 270],
        "track_id": 12,
        "object_group_id": "person_group_002",
        "conf": 0.68,
        "v_hash": hash_a_near,
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_switch, now_3)
    print(f"Test 3 (Track switch & new group, spatial+visual match): is_dup={is_dup}, reason={info['reason']}, dist={info['distance']:.1f}, hamming={info['hamming']} (Expected: True, SPATIAL_VISUAL_MATCH)")
    assert is_dup and info["reason"] == "SPATIAL_VISUAL_MATCH"
    assert info["matched_event_key"] == event_key_1

    # 4. Test 4: Orang berdiri di tempat yang sama (Proximity match)
    person_standing = {
        "box": [102, 101, 162, 261],
        "track_id": 18,
        "object_group_id": "person_group_003",
        "conf": 0.58,
        "v_hash": None, # misal visual hash gagal karena gelap
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_standing, now + 5.0)
    print(f"Test 4 (Spatial proximity, standing person): is_dup={is_dup}, reason={info['reason']}, dist={info['distance']:.1f} (Expected: True, SPATIAL_PROXIMITY)")
    assert is_dup and info["reason"] == "SPATIAL_PROXIMITY"

    # 5. Test 5: Orang kedua (Person B) benar-benar berbeda
    # Posisi jauh (400, 200), track 25, v_hash berbeda jauh
    person_b = {
        "box": [400, 200, 470, 380],
        "track_id": 25,
        "object_group_id": "person_group_004",
        "conf": 0.72,
        "v_hash": hash_b,
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_b, now + 6.0)
    print(f"Test 5 (Different Person B): is_dup={is_dup} (Expected: False)")
    assert not is_dup, "Person B should be recognized as a completely new person!"

    # 6. Test 6: Person A kembali setelah cooldown habis (> 120s)
    now_future = now + 150.0 # 2.5 menit kemudian
    person_a_returned = {
        "box": [100, 100, 160, 260],
        "track_id": 40,
        "object_group_id": "person_group_010",
        "conf": 0.65,
        "v_hash": hash_a,
    }
    is_dup, info = service._is_same_person_candidate(camera_id, person_a_returned, now_future)
    print(f"Test 6 (Person A returns after cooldown): is_dup={is_dup} (Expected: False)")
    assert not is_dup, "Person returning after cooldown should trigger a new event!"

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
