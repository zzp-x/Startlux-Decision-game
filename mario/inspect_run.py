"""决策日志分析：从 harness 写出的 .log.jsonl 里读出每步做了什么、模型给了多少概率。

    python inspect_run.py

默认读那盘完整的 1-1 branch 录像（26 次决策，x 到 2370），打印决策数、动作直方图，
以及首/次/中/末几次决策的完整选项文本。想换日志就改下面的 log 常量。
"""
import json, sys
from collections import Counter
from pathlib import Path

log = Path("jev-mario-main/runs/1-1-branch-startlux-0.8b-20261006-163620.log.jsonl")
rows = [json.loads(l) for l in log.read_text().splitlines()]
print(f"decisions: {len(rows)}   x: {rows[0]['x']} -> {rows[-1]['x']}")
print("choices:", Counter(r["choice"] for r in rows).most_common())
print()
# 抽头、第二次、中间一次和最后一次决策，看选项文本与最终选择的对照
for i in (0, 1, 14, len(rows) - 1):
    r = rows[i]
    print("=" * 76)
    print(f"decision {i+1}: frame={r['frame']} x={r['x']}  ->  {r['choice']}  p={r.get('p')}")
    if r.get("ride"):
        print("  rode the continuation:", r["ride"])
    outs = r.get("outcomes", {})
    for k, v in list(outs.items())[:11]:
        mark = " <== CHOSEN" if k == r["choice"] else ""   # 标出模型实际选的那一项
        print(f"    {k:34s} {v}{mark}")
