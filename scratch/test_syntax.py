import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from services import dedup_dryrun

sc = dedup_dryrun.run_controlled_scenarios()
passed = sum(1 for s in sc if s['passed'])
print(f"Controlled scenarios: {passed}/{len(sc)} passed")
for s in sc:
    status = "PASS" if s["passed"] else "FAIL"
    name = s['scenario'].encode('ascii', 'replace').decode('ascii')[:60]
    print(f"  [{status}] {name}  kept={s['kept']} held={s['held']}")

print()
r = dedup_dryrun.dry_run_from_db(camera_id=None, lookback_seconds=86400)
print(f"DB dry-run (24h): total={r['total_events']} kept={r['kept']} held={r['held']} rate={r['held_rate_percent']}%")
print(f"DB mutated      : {r['database_mutated']}")

full = dedup_dryrun.run_dry_run(camera_id=None, lookback_seconds=86400)
assess = full["readiness_assessment"]
print(f"INSERT filter   : {assess['ready_for_insert_filter']}")
print(f"Risk level      : {assess['risk_level']}")

assert not r["database_mutated"], "DB should not be mutated"
assert passed == len(sc), f"{len(sc)-passed} scenarios failed"
print("\nAll assertions OK")
