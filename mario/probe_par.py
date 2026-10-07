"""A/B 探针：从同一个真实游戏状态出发，比较串行与并行的选项评估。

它从 0.8B 那盘录像的决策日志里重放出真实选择直到第 `--at` 次决策，然后从完全相同的状态把
该次决策的 11 个选项评估两遍：一遍走上游的串行 Sim.outcome()（本机实测 137.9 s/次决策），
一遍走并行分发器。

检查两件事，都不是假设而是实测：
  1. 跨进程保真度——每个 worker 重放父进程的输入历史后回报自己 RAM 与画面的哈希，
     必须与父进程的相等；
  2. 结果等价性——11 份 outcome 字典（dead/flag/dx/frames/alive_paths/best_gain/best_path/…）
     在两条路径上必须完全相同。

    python probe_par.py --at 4 --workers 6,11,12
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mario_par

HERE = Path(__file__).resolve().parent
RUNS = HERE / "jev-mario-main" / "runs"


def fingerprint(core) -> dict:
    """把模拟器状态压成两个哈希（RAM 与画面），用于跨进程比对。"""
    return {
        "ram": hashlib.blake2b(bytes(core.ram), digest_size=16).hexdigest(),
        "screen": hashlib.blake2b(core.screen.tobytes(), digest_size=16).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=int, default=4, help="重放到第几次决策后评估")
    ap.add_argument("--workers", default="6,11,12")
    ap.add_argument("--log", default="1-1-branch-startlux-0.8b-*.log.jsonl")
    a = ap.parse_args()

    mario_par.apply_shim()
    ns = mario_par.load_branch()
    Sim, options, horizon = ns["Sim"], list(ns["OPTIONS"]), ns["HORIZON"]
    import play

    pool_sizes = [int(x) for x in a.workers.split(",") if x.strip()]

    # 必须在建环境之前打补丁：分发器和历史记录器都来自 patch()
    holder = mario_par.patch(ns, 1, quiet=True)
    sim = Sim("1-1")

    log_path = sorted(RUNS.glob(a.log))[-1]
    log = [json.loads(line) for line in log_path.read_text().splitlines()]
    # 按录像里的真实选择重放：每次先等落地，再执行当时选的动作（以及 ride 的续步）
    for entry in log[: a.at]:
        while play.airborne(sim.ram) and not sim.done:
            sim.step(3, [])  # 传空列表 = 标记为"真实步"，历史记录器据此区分模拟分支
        if entry["choice"].startswith("escape:"):
            raise SystemExit("该状态走了 escape 分支，本探针不覆盖（escape 仍是串行路径）")
        real = []  # run() 对真实步传帧列表；探针只需要那个"标记"作用
        sim.execute(entry["choice"], horizon, real)
        real.clear()
        if entry.get("ride") and not sim.over():
            sim.execute(entry["ride"], horizon, real)
            real.clear()
    while play.airborne(sim.ram) and not sim.done:
        sim.step(3, [])

    hist = list(sim.real_hist)
    mine = fingerprint(sim.core)
    info = dict(sim.info)
    print(f"[state] 重放 {a.at} 次决策后：x={sim.x()}  done={sim.done}  flag={info.get('flag_get')}  "
          f"真实按键历史 {len(hist)} 帧  日志={log_path.name}", flush=True)

    # ---------------------------------------------------------------- 串行（上游路径）
    t0 = time.perf_counter()
    serial, per_option = {}, {}
    for n in options:
        t = time.perf_counter()
        serial[n] = Sim._serial_outcome(sim, n)
        per_option[n] = time.perf_counter() - t
    t_serial = time.perf_counter() - t0
    print(f"[serial] {t_serial:6.1f}s  主环境状态是否被扰动: {fingerprint(sim.core) != mine}", flush=True)
    print("[serial] 各选项耗时（降序）:", flush=True)
    for n, t in sorted(per_option.items(), key=lambda kv: -kv[1]):
        o = serial[n]
        print(f"           {t:6.1f}s  {n:34s} dead={int(o['dead'])} dx={o['dx']:+d} "
              f"alive_paths={o['alive_paths']}", flush=True)

    # ---------------------------------------------------------------- 并行
    for w in pool_sizes:
        holder.terminate()
        holder.join()
        holder = mario_par.patch(ns, w, quiet=True)

        # 让 w 个 worker 各自重放同一段历史，比对哈希
        checks = holder.map(mario_par._worker_check_state, [("1-1", hist)] * w)
        restore_ok = all(c["ram"] == mine["ram"] and c["screen"] == mine["screen"] for c in checks)
        pids = sorted({c["pid"] for c in checks})

        t0 = time.perf_counter()
        parallel = {n: sim.outcome(n) for n in options}
        t_par = time.perf_counter() - t0

        diffs = [n for n in options
                 if json.dumps(serial[n], sort_keys=True) != json.dumps(parallel[n], sort_keys=True)]
        print(f"[parallel w={w:2d}] {t_par:6.1f}s  加速 {t_serial / t_par:4.2f}x  "
              f"跨进程状态一致={restore_ok}（{len(pids)} 个 worker 进程）  "
              f"11 个选项结果一致={not diffs}", flush=True)
        for n in diffs:
            print(f"           DIFF {n}:\n             串行 {serial[n]}\n             并行 {parallel[n]}",
                  flush=True)

    holder.terminate()
    holder.join()


if __name__ == "__main__":
    main()
