import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

client = app.test_client()

routes_to_test = [
    ("/", 200),
    ("/dashboard", 200),
    ("/monitoring", 200),
    ("/detections", 200),
    ("/history", 200),
    ("/recap", 200),
    ("/statistics", 200),
    ("/settings", 200),
    ("/health", 200),
    ("/api/cameras", 200),
    ("/api/detections", 200),
    ("/api/plate/history", 200),
    ("/api/analytics", 200),
    ("/api/analytics?period=today", 200),
    ("/api/analytics?period=7d", 200),
    ("/api/statistics/enterprise", 200),
    ("/api/statistics/enterprise?period=today", 200),
    ("/api/settings", 200),
    ("/api/stats/summary", 200),
]

failed = 0
for url, expected_code in routes_to_test:
    try:
        res = client.get(url)
        if res.status_code != expected_code:
            print(f"[FAIL] {url} -> Status {res.status_code} (Expected {expected_code})")
            if res.is_json:
                print("       JSON response:", res.get_json())
            else:
                print("       Data prefix:", res.get_data(as_text=True)[:200])
            failed += 1
        else:
            print(f"[PASS] {url} -> Status {res.status_code}")
    except Exception as e:
        print(f"[ERROR] {url} -> Exception: {e}")
        failed += 1

print(f"\nTotal tested: {len(routes_to_test)}, Failed: {failed}")
