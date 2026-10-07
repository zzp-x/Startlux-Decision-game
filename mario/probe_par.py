"""A/B probe: serial vs parallel option evaluation from the same real game state.

Replays the real choices out of the 0.8B run's decision log up to decision `--at`, then evaluates all
11 options of that decision twice from the exact same state: once with upstream's serial
Sim.outcome() (137.9 s/decision measured on this machine) and once with the parallel dispatcher.

Two things are checked, not assumed:
  1. cross-process fidelity -- every worker replays the parent's input history and reports a hash of
     its RAM and screen, which must equal the parent's;
  2. result equivalence -- the 11 outcome dicts (dead/flag/dx/frames/alive_paths/best_gain/best_path/...)
     must be identical between the two paths.

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

    # patch before building the env: the dispatcher and the history recorder come from patch()
    holder = mario_par.patch(ns, 1, quiet=True)
    sim = Sim("1-1")

    log_path = sorted(RUNS.glob(a.log))[-1]
    log = [json.loads(line) for line in log_path.read_text().splitlines()]
    for entry in log[: a.at]:
        while play.airborne(sim.ram) and not sim.done:
            sim.step(3, [])  # the empty list marks a real step for the history recorder
        if entry["choice"].startswith("escape:"):
            raise SystemExit("该状态走了 escape 分支，本探针不覆盖（escape 仍是串行路径）")
        real = []  # run() passes its frame list for real steps; the probe only needs the marking
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

    # ---------------------------------------------------------------- serial (upstream path)
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

    # ---------------------------------------------------------------- parallel
    for w in pool_sizes:
        holder.terminate()
        holder.join()
        holder = mario_par.patch(ns, w, quiet=True)

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
