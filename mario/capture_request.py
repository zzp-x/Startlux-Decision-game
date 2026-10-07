"""抓取一次真实的 jev-mario branch 决策，落成可以直接发送的 /v1/systemone 请求。

用上游自己的模拟器跑完上游自己的 Sim.outcome()（11 个选项全跑）——与 branch.py 完全同一条
代码路径——把 branch.py 会 POST 的那个 JSON body 原样写出。产物可直接喂给
../local-run/demo_call.py。

    python capture_request.py                       # 默认 --at 1，1-1 关卡
    python capture_request.py --at 1 --out x.json

注意 `--at N` 是"先按住 run right 走 N 次决策再抓"。run right 会径直撞上第一只板栗仔
（约 x=315、frame 106 处死亡），所以 N 给大了模拟器已经死了，脚本会明确报错让你调小。
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mario_par  # noqa: E402  （路径在上面已插好）

RUN_RIGHT = 3  # SIMPLE_MOVEMENT 里的下标，对应 ACTIONS["run right"]


def build(ns: dict, level: str, at: int) -> tuple[dict, str, int]:
    sim = ns["Sim"](level)
    horizon = ns["HORIZON"]
    # 前进阶段：连按 run right，每步一次完整决策的时长
    for _ in range(at):
        sim.execute("run right", horizon)
        if sim.over():
            break
    if sim.done:
        sim.env.close()
        raise SystemExit(
            f"died during the {at}-decision forward run (x={sim.x()}); lower --at. "
            "'run right' walks straight into the first goomba."
        )
    # situation 是上游 features() 从网格总结出的那段自然语言
    summary = ns["features"](ns["grid"](sim.ram)[0], ns["play"].speed(sim.ram), airborne=False)["summary"]
    # criteria 是 11 个选项各自的仿真结果句，由上游 describe() 生成
    criteria = {name: ns["describe"](sim.outcome(name)) for name in ns["OPTIONS"]}
    x = sim.x()
    sim.env.close()

    body = {
        "state": {"situation": summary, "outcomes": criteria},
        "model": "StartLux-Decision-0.8B-Q8_0-GGUF",
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "You control Mario. Each option below says what actually happens if Mario does it for "
                    "the next second and then keeps going, measured in the game itself. Pick the option that "
                    "makes the most progress toward the flag (to the right) without dying. A dead end is an "
                    "option after which every continuation dies; never pick one while an alternative survives."
                ),
                "criteria": criteria,
            }
        },
    }
    return body, summary, x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--at", type=int, default=1, help="run 'run right' this many decisions first")
    ap.add_argument("--out", default=str(HERE / "mario-demo-request.json"))
    a = ap.parse_args()

    mario_par.apply_shim()
    ns = mario_par.load_branch()
    body, summary, x = build(ns, a.level, a.at)
    # ensure_ascii=False：中文/特殊字符按原样写，方便人直接读这份请求
    Path(a.out).write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    raw = json.dumps(body, ensure_ascii=False)
    print(f"written: {a.out}")
    print(f"  level {a.level}, {a.at} forward decisions in, Mario at x={x}")
    print(f"  {len(body['questions']['action']['criteria'])} options, "
          f"{len(raw)} chars, {len(raw.encode())} bytes")
    print()
    print("=== state.situation ===")
    print(summary)
    print()
    print("=== criteria（模型真正读到的选项文本）===")
    for k, v in body["questions"]["action"]["criteria"].items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
