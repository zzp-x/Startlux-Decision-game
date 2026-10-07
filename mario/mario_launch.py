"""原样运行官方 4esv/jev-mario harness，指向本地的 StartLux-Decision 服务。

    python mario_launch.py branch.py --bot jev --level 1-1 \
        --url http://127.0.0.1:8090/v1/systemone --label startlux           # 原样串行
    python mario_launch.py branch.py --bot jev --level 1-1 --parallel 6 \
        --url http://127.0.0.1:8090/v1/systemone --label startlux-par       # 并行评估 11 个选项

这里做两件事，都不碰 harness：

1. 一个兼容垫片。gym-super-mario-bros 9.x 建在 gymnasium 上，后者删掉了 harness 传给 make() 的
   `apply_api_compatibility`。去掉它对这里是无副作用的：环境已经就是 harness 期待的现代 API
   ——(obs, info) 返回、五元组 step——已实测：reset -> (obs, info)，step -> (obs, rew, term,
   trunc, info)，且 nes-py 的 _backup/_restore 是逐帧精确的。
2. `--parallel N` 把每次决策的 11 个选项扇出到 N 个进程上，而不是逐个模拟（见 mario_par.py）。
   不带这个参数时，harness 完全按上游写法运行。

jev-mario-main/ 里的每个文件都与上游逐字节一致。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "jev-mario-main"


def take_flag(argv: list[str], name: str) -> tuple:
    """从 argv 里摘出 `--name N` / `--name=N`。返回 (取到的值或 None, 剩下的参数)。"""
    rest, value, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == name and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
            continue
        if a.startswith(name + "="):
            value = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return value, rest


def main() -> int:
    import os
    import time

    import mario_par

    mario_par.apply_shim()
    sys.path.insert(0, str(HARNESS))

    argv = sys.argv[1:]
    par_raw, argv = take_flag(argv, "--parallel")
    workers = 0
    if par_raw is not None:
        workers = int(par_raw)
        if workers < 1:
            sys.exit("--parallel needs a process count >= 1")

    script = argv[0] if argv else "branch.py"
    src = HARNESS / script
    if not src.exists():
        sys.exit(f"no such harness script: {src}")

    local_url = os.environ.get("JEVMARIO_URL", "http://127.0.0.1:8090/v1/systemone")

    # branch.py 从命令行接收 --url（所以不动它自己的逻辑，自托管运行仍会被记成 cost_usd = 0）。
    # play.py 和 live.py 把 play.JEV_URL 写死在代码里、并且需要 TYPESAFE_API_KEY 环境变量，
    # 所以这里就地改指向本地服务，而不是去编辑 harness。
    if script in ("play.py", "live.py"):
        os.environ.setdefault("TYPESAFE_API_KEY", "local-self-hosted")
        import play as _play

        _play.JEV_URL = local_url

    # 只读的进度探针：每次决策刚本地服务返回就立刻打一行。
    # 它包裹的是 httpx.Client.post，所以哪个 harness 脚本在跑都生效，且完全不影响 harness 的行为。
    import httpx as _httpx

    raw_post = _httpx.Client.post
    prog = {"n": 0, "t0": time.time()}

    def logged_post(self, url, *args, **kwargs):
        r = raw_post(self, url, *args, **kwargs)
        try:
            if str(url).rstrip("/").endswith("/v1/systemone"):
                d = r.json()
                a = d["answers"]["action"]
                prog["n"] += 1
                el = time.time() - prog["t0"]
                print(f"[decision {prog['n']:02d}] +{el:6.0f}s  choice={a['choice']!r} "
                      f"p={a['probabilities'].get(a['choice'], 0):.2f}  "
                      f"wall={d.get('usage', {}).get('wall_ms')} ms", flush=True)
        except Exception as exc:  # 探针出错也绝不干扰正式运行
            print("[probe]", repr(exc), flush=True)
        return r

    _httpx.Client.post = logged_post

    sys.argv = [script] + argv[1:]
    print(f"[launch] {script} -> {local_url}"
          + (f"（并行评估：{workers} 进程）" if workers else ""), flush=True)

    if workers and script == "branch.py":
        ns = mario_par.load_branch()
        pool = mario_par.patch(ns, workers)
        try:
            mario_par.run_main_block(ns)
        finally:
            pool.terminate()
            pool.join()
        return 0

    if workers:
        print("[launch] --parallel 只对 branch.py 生效，本次忽略", flush=True)

    # 其余脚本按普通方式跑，等价于 python <script>
    runpy = __import__("runpy")
    runpy.run_path(str(src), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
