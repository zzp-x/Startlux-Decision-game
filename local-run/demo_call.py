"""本地 StartLux-Decision 服务的最小客户端（POST /v1/systemone）。

零依赖，只用标准库。服务必须已经起着：

    local-run/1-start-llama-0.8b-gpu.cmd      -> 127.0.0.1:8081  (推理引擎)
    local-run/2-start-decision-server.cmd     -> 127.0.0.1:8090  (本接口)

用法：
    python demo_call.py                                   # 内置示例请求
    python demo_call.py --file demo-request.json          # 换成你自己的请求
    python demo_call.py --repeat 3                        # 跑 3 次，看冷启动与命中缓存的差别
    python demo_call.py --url http://127.0.0.1:8090/v1/systemone
"""

import argparse
import json
import time
import urllib.request

# state = 模型要读的证据；questions = 每一项是一个要它做的决策。
# 问题类型（见 startlux_decision/jevfmt.py）：
#   "choice"  criteria = {选项id: 判定标准}   -> 回答：选中的选项 id + 全部选项的概率
#   "noul"    criteria = {"true": .., "false": ..}  -> 回答："true" 成立的概率
#   "score"   criteria = [等级文本, ...]（有序）  -> 回答：等级的期望下标 + 图例
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
    """发一次决策请求。返回 (响应字典, 客户端侧往返毫秒数)。"""
    # timeout 刻意开到 900 s：冷启动 + 首次全量前向，或马里奥那种 11 选项排序题，单次会很慢
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
            # score 是期望下标，四舍五入回查图例拿到标签
            legend = ans.get("legend", {})
            label = legend.get(str(round(ans["score"])), "?")
            print(f"  {name:14s} score  -> {ans['score']:.3f}  ({label})")
    u = raw.get("usage", {})
    # 按请求统计：input_tokens = 提示词长度，evaluated_tokens = 服务端真正算过的 token 数，
    # cached_tokens = 从 KV cache 里复用的部分（input - cached ≈ evaluated）。
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
