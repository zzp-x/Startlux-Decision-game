"""Run the official 4esv/jev-mario harness unmodified, against the local StartLux-Decision server.

    python mario_launch.py branch.py --bot jev --level 1-1 \
        --url http://127.0.0.1:8090/v1/systemone --label startlux           # 原样串行
    python mario_launch.py branch.py --bot jev --level 1-1 --parallel 6 \
        --url http://127.0.0.1:8090/v1/systemone --label startlux-par       # 并行评估 11 个选项

Two things happen here, neither of which touches the harness:

1. One compatibility shim.  gym-super-mario-bros 9.x is built on gymnasium, which dropped the
   `apply_api_compatibility` kwarg that the harness passes to make().  Dropping it is a no-op here,
   because the env already speaks the current (obs, info) / 5-tuple step API that the harness expects --
   verified: reset -> (obs, info), step -> (obs, rew, term, trunc, info), and nes-py _backup/_restore are
   frame-exact.
2. `--parallel N` fans the 11 options of each decision out over N processes instead of simulating them
   one after another (see mario_par.py).  Without the flag the harness runs exactly as upstream wrote it.

Every file in jev-mario-main/ stays byte-identical to upstream.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "jev-mario-main"


def take_flag(argv: list[str], name: str) -> tuple:
    """Pull `--name N` / `--name=N` out of argv. Returns (value_or_None, rest)."""
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

    # branch.py takes --url on the command line (leave its own logic alone, so a self-hosted run is
    # still recorded as cost_usd = 0).  play.py and live.py hardcode play.JEV_URL and need the
    # TYPESAFE_API_KEY env var, so point those at the local server here instead of editing the harness.
    if script in ("play.py", "live.py"):
        os.environ.setdefault("TYPESAFE_API_KEY", "local-self-hosted")
        import play as _play

        _play.JEV_URL = local_url

    # Read-only progress probe: log every decision the moment it comes back from the local server.
    # It wraps httpx.Client.post, so it works no matter which harness script runs, and it changes
    # nothing about the harness.
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
        except Exception as exc:  # never interfere with the run
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

    runpy = __import__("runpy")
    runpy.run_path(str(src), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
