"""单个真实请求在决策链路上的延迟，可指向任意 llama-server 地址。

  python bench_local.py --llama http://127.0.0.1:8081 --label "GTX 1050 (Vulkan)"
  python bench_local.py --llama http://127.0.0.1:8082 --label "CPU only"
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from startlux_local import LocalDecision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

STATE = {
    "model": "GPT-5.2 preview",
    "bench_name": "JevBench",
    "eval_set": "held-out-hard",
    "config": {"temperature": 0.0, "num_options": 4, "answers_given": "free text, scored by exact match"},
    "metrics": {"acc_multi_turn": [0.81, 0.79, 0.76, 0.74], "cost_per_1k_calls_usd": 2.4, "p50_latency_ms": 1840},
    "contradiction_flagged": [False, False, True, True],
    "note": "accuracy falls off after the third turn; the last two turns contain a statement that contradicts turn one",
}

QUESTIONS = {
    "verdict": {"type": "choice", "instructions": "Should this model be shipped to production?",
                "criteria": {"ship": "quality holds up", "hold": "needs more work", "reject": "not fit for use"}},
    "contradiction": {"type": "bool", "instructions": "Does the evidence contain an internal contradiction?",
                      "criteria": {"true": "at least one pair of statements cannot both be true",
                                   "false": "the evidence is self-consistent"}},
    "confidence": {"type": "score", "instructions": "How well does this evidence support the verdict?",
                   "criteria": ["none", "weak", "moderate", "strong", "conclusive"]},
    "urgency": {"type": "choice", "instructions": "How urgent is the decision?",
                "criteria": {"now": "act this week", "soon": "act this quarter", "later": "no deadline pressure"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default="http://127.0.0.1:8081")
    ap.add_argument("--label", default="")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "models", "StartLux-Decision-0.8B-Q8_0-GGUF"))
    ap.add_argument("--repo", default=os.path.join(HERE, "..", "..", "Startlux-Decision"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    eng = LocalDecision(a.model_dir, a.llama, repo=os.path.abspath(a.repo), cache_prompt=not a.no_cache)
    if "error" in eng.health():
        print("server not reachable:", eng.health())
        return
    eng.decide(STATE, QUESTIONS)                      # 预热一次，不计时
    times = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        ans, usage = eng.decide(STATE, QUESTIONS)
        times.append((time.perf_counter() - t0) * 1000)
    per_q = [t / len(QUESTIONS) for t in times]
    print(f"{a.label or a.llama}  cache_prompt={not a.no_cache}")
    print(f"  4 questions / request: median {statistics.median(times):.0f} ms   "
          f"min {min(times):.0f} ms   max {max(times):.0f} ms   "
          f"-> {statistics.median(per_q):.0f} ms per question   ({1000/statistics.median(per_q):.1f} decisions/s)")
    print("  answers:", {k: (v.get("choice") or round(v.get("noul", v.get("score")), 3)) for k, v in ans.items()})


if __name__ == "__main__":
    main()
