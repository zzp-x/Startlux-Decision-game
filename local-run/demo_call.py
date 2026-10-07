"""Minimal client for a local StartLux-Decision server (POST /v1/systemone).

Zero dependencies (standard library only).  The server must already be running:

    local-run/1-start-llama-0.8b-gpu.cmd      -> 127.0.0.1:8081  (inference)
    local-run/2-start-decision-server.cmd     -> 127.0.0.1:8090  (this API)

Usage:
    python demo_call.py                                   # built-in sample request
    python demo_call.py --file demo-request.json          # your own request
    python demo_call.py --repeat 3                        # show cold vs cached latency
    python demo_call.py --url http://127.0.0.1:8090/v1/systemone
"""

import argparse
import json
import time
import urllib.request

# state = the evidence the model reads; questions = one entry per decision it has to make.
# Question types (see startlux_decision/jevfmt.py):
#   "choice"  criteria = {option_id: criterion text}  -> answer: the chosen option id + probabilities
#   "noul"    criteria = {"true": .., "false": ..}    -> answer: probability that "true" holds
#   "score"   criteria = [level text, ...] (ordered)  -> answer: expected level index + legend
SAMPLE = {
    "state": {
        "ticket_id": "TCK-4471",
        "message": "I was charged twice for the same order. Please refund the duplicate charge.",
        "amount": 49.99,
        "currency": "USD",
        "previous_refunds": 0,
    },
    "questions": {
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is this ticket?",
            "criteria": {
                "low": "no deadline pressure",
                "normal": "an answer is expected within a day",
                "high": "the customer is losing money right now",
            },
        },
        "refund_now": {
            "type": "noul",
            "instructions": "Should the duplicate charge be refunded without further review?",
            "criteria": {
                "true": "the evidence already proves the duplicate charge",
                "false": "more checking is needed before refunding",
            },
        },
        "sentiment": {
            "type": "score",
            "instructions": "Rate the customer's current sentiment.",
            "criteria": ["furious", "upset", "neutral", "satisfied"],
        },
    },
}


def call(url, payload, timeout=900):
    """One decision request. Returns (response dict, client-side round-trip ms)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, (time.perf_counter() - t0) * 1000


def show(raw):
    for name, ans in raw.get("answers", {}).items():
        kind = ans.get("type")
        if kind == "choice":
            ranked = sorted(ans.get("probabilities", {}).items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{k}={v:.3f}" for k, v in ranked)
            print(f"  {name:14s} choice -> {ans['choice']:10s} ({top})")
        elif kind == "noul":
            print(f"  {name:14s} noul   -> P(true) = {ans['noul']:.4f}")
        elif kind == "score":
            legend = ans.get("legend", {})
            label = legend.get(str(round(ans["score"])), "?")
            print(f"  {name:14s} score  -> {ans['score']:.3f}  ({label})")
    u = raw.get("usage", {})
    # Per request: input_tokens = prompt length, evaluated_tokens = tokens the server really computed,
    # cached_tokens = the part it reused from the KV cache (input - cached ~= evaluated).
    print(f"  [usage] input_tokens={u.get('input_tokens')} evaluated_tokens={u.get('evaluated_tokens')} "
          f"cached_tokens={u.get('cached_tokens')} output_tokens={u.get('output_tokens')} "
          f"server_wall_ms={u.get('wall_ms')}")


def main():
    ap = argparse.ArgumentParser(description="Call a local StartLux-Decision /v1/systemone endpoint")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--file", default=None, help="JSON request file (default: built-in sample)")
    ap.add_argument("--repeat", type=int, default=1, help="send the same request N times")
    a = ap.parse_args()

    payload = json.load(open(a.file, encoding="utf-8")) if a.file else SAMPLE
    for i in range(1, a.repeat + 1):
        raw, ms = call(a.url, payload)
        print(f"[{i}/{a.repeat}] client_round_trip={ms:.0f} ms  model={raw.get('model')}")
        show(raw)


if __name__ == "__main__":
    main()
