import sys
sys.path.insert(0, '.')
import db

try:
    with db.get_db() as conn:
        with conn.cursor() as cur:
            # Check video-type detections WITHOUT plate join to see raw data
            cur.execute("""
                SELECT fd.detection_id, fd.camera_id, fd.plate_id, fd.source_type,
                       fd.source_name, fd.object_type, fd.vehicle_type,
                       fd.created_at, fd.track_id, fd.event_key
                FROM full_detection fd
                WHERE fd.source_type = 'video'
                  AND fd.source_name LIKE 'Segment%'
                ORDER BY fd.created_at DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
            print(f"Segment-sourced detections (video-type, Segment*) - {len(rows)} rows:")
            for r in rows:
                print(
                    f"  id={r['detection_id']} cam={r['camera_id']} "
                    f"plate_id={r['plate_id']} type={r['object_type']} "
                    f"src_name={r['source_name']} time={r['created_at']}"
                )

            # Check the plate table for these plate_ids
            if rows:
                plate_ids = [r['plate_id'] for r in rows if r['plate_id']]
                if plate_ids:
                    placeholders = ','.join(['%s'] * len(plate_ids))
                    cur.execute(f"""
                        SELECT plate_id, plate_number, normalized_plate_number,
                               raw_ocr_text, ocr_confidence, detection_status
                        FROM plate
                        WHERE plate_id IN ({placeholders})
                        ORDER BY plate_id DESC
                        LIMIT 20
                    """, plate_ids)
                    plates = cur.fetchall()
                    print(f"\nPlate records for these detections ({len(plates)}):")
                    for p in plates:
                        print(
                            f"  id={p['plate_id']} plate={p['plate_number']} "
                            f"norm={p['normalized_plate_number']} "
                            f"raw={p['raw_ocr_text']} conf={p['ocr_confidence']} "
                            f"status={p['detection_status']}"
                        )
                else:
                    print("  (no plate_ids found)")

            # Also check plate table for recent entries from Segment
            cur.execute("""
                SELECT p.plate_id, p.plate_number, p.normalized_plate_number,
                       p.raw_ocr_text, p.ocr_confidence, p.detection_status,
                       p.created_at
                FROM plate p
                ORDER BY p.plate_id DESC
                LIMIT 10
            """)
            recent_plates = cur.fetchall()
            print(f"\nMost recent plates in DB ({len(recent_plates)}):")
            for p in recent_plates:
                print(
                    f"  id={p['plate_id']} plate={p['plate_number']} "
                    f"norm={p['normalized_plate_number']} "
                    f"raw={p['raw_ocr_text']} conf={p['ocr_confidence']} "
                    f"status={p['detection_status']} time={p['created_at']}"
                )

except Exception as e:
    import traceback
    traceback.print_exc()
