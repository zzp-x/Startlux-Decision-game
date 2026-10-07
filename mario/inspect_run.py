import json, sys
from collections import Counter
from pathlib import Path

log = Path("jev-mario-main/runs/1-1-branch-startlux-0.8b-20261006-163620.log.jsonl")
rows = [json.loads(l) for l in log.read_text().splitlines()]
print(f"decisions: {len(rows)}   x: {rows[0]['x']} -> {rows[-1]['x']}")
print("choices:", Counter(r["choice"] for r in rows).most_common())
print()
for i in (0, 1, 14, len(rows) - 1):
    r = rows[i]
    print("=" * 76)
    print(f"decision {i+1}: frame={r['frame']} x={r['x']}  ->  {r['choice']}  p={r.get('p')}")
    if r.get("ride"):
        print("  rode the continuation:", r["ride"])
    outs = r.get("outcomes", {})
    for k, v in list(outs.items())[:11]:
        mark = " <== CHOSEN" if k == r["choice"] else ""
        print(f"    {k:34s} {v}{mark}")
