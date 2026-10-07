"""本地 GGUF 决策引擎的端到端自检：每种问题类型各发一次请求，外加一次 30 选项的宽列表。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from startlux_local import LocalDecision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "models", "StartLux-Decision-0.8B-Q8_0-GGUF")


def show(tag, ans, usage, dt):
    print(f"\n=== {tag}  ({dt:.0f} ms, {usage['input_tokens']} prompt tokens, "
          f"{usage['output_tokens']} generated) ===")
    for k, v in ans.items():
        if v["type"] == "choice":
            top = sorted(v["probabilities"].items(), key=lambda x: -x[1])[:3]
            print(f"  {k}: choice={v['choice']}  conf={v['confidence']:.3f}  "
                  f"top={[(a, round(b, 3)) for a, b in top]}")
        elif v["type"] == "noul":
            print(f"  {k}: yes/no probability of true = {v['noul']:.4f}  -> {'YES' if v['noul'] > 0.5 else 'NO'}")
        else:
            print(f"  {k}: score={v['score']:.3f}  conf={v['confidence']:.3f}  "
                  f"p={[(a, round(b, 3)) for a, b in v['probabilities'].items()]}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL)
    ap.add_argument("--llama", default="http://127.0.0.1:8081")
    a = ap.parse_args()
    eng = LocalDecision(a.model_dir, llama=a.llama,
                        repo=os.path.join(HERE, "..", "..", "Startlux-Decision"), verbose=True)
    print("llama-server health:", eng.health())
    print("model:", os.path.basename(os.path.abspath(a.model_dir)))

    state = {
        "pair": "BTC/USDT",
        "price_now": 68420.0,
        "change_24h_pct": -3.4,
        "volume_24h_usd": 2.1e9,
        "funding_rate_8h": 0.00021,
        "open_interest_change_24h_pct": 6.8,
        "rsi_14_h1": 38.2,
        "above_ema_200_h4": True,
    }

    q = {
        "direction": {"type": "choice", "instructions": "Which direction is the next 4h candle most likely to take?",
                      "criteria": {"long": "price closes above the current level",
                                   "short": "price closes below the current level",
                                   "flat": "price stays within 0.2% of the current level"}},
        "news_driven": {"type": "bool", "instructions": "Given this evidence, is the move driven by a news event?",
                        "criteria": {"true": "an identifiable news event explains the move",
                                     "false": "the move is ordinary flow with no single event behind it"}},
        "conviction": {"type": "score", "instructions": "How strong is the evidence for a mean-reversion bounce?",
                       "criteria": ["very weak", "weak", "neutral", "strong", "very strong"]},
    }
    t0 = time.perf_counter()
    ans, usage = eng.decide(state, q)
    show("mixed request (choice + noul + score)", ans, usage, (time.perf_counter() - t0) * 1000)

    wide = {"type": "choice", "instructions": "Which single asset should the new allocation go into?",
            "criteria": {f"{a}": f"{a} listed equity, sector {s}" for a, s in
                         zip(["AAPL", "MSFT", "NVDA", "AMD", "INTC", "TSM", "ASML", "AMZN", "GOOGL", "META",
                              "TSLA", "NFLX", "ADBE", "CRM", "ORCL", "IBM", "CSCO", "QCOM", "TXN", "AVGO",
                              "MU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "NOW", "PANW", "SNOW", "PLTR"],
                             ["hardware", "software", "semi", "semi", "semi", "foundry", "litho", "retail", "ads",
                              "ads", "auto", "media", "software", "software", "software", "services", "network",
                              "semi", "semi", "semi", "memory", "equip", "equip", "equip", "eda", "eda",
                              "software", "security", "data", "analytics"])}}
    t0 = time.perf_counter()
    ans, usage = eng.decide(state, {"pick": wide})
    show("wide Choice (30 options -> grouped rounds)", ans, usage, (time.perf_counter() - t0) * 1000)
    print("\ntotal llama /completion calls:", eng.calls, "prompt tokens evaluated:", eng.tokens)


if __name__ == "__main__":
    main()
