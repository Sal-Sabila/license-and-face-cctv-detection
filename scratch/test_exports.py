import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = app.test_client()

export_routes = [
    "/api/export/cameras",
    "/api/export/cameras/pdf",
    "/api/export/detections",
    "/api/export/detections/pdf",
    "/api/export/plates",
    "/api/export/plates/pdf",
    "/api/export/recap",
    "/api/export/recap/pdf",
    "/api/export/statistics",
    "/api/export/statistics/pdf",
]

failed = 0
for url in export_routes:
    try:
        res = client.get(url)
        if res.status_code != 200:
            print(f"[FAIL] {url} -> Status {res.status_code}")
            if res.is_json:
                print("       JSON response:", res.get_json())
            else:
                print("       Data prefix:", res.get_data(as_text=True)[:200])
            failed += 1
        else:
            print(f"[PASS] {url} -> Status 200, Content-Type: {res.headers.get('Content-Type')}, Size: {len(res.data)} bytes")
    except Exception as e:
        print(f"[ERROR] {url} -> Exception: {e}")
        failed += 1

print(f"\nTotal tested: {len(export_routes)}, Failed: {failed}")
