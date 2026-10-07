"""StartLux-Decision - one-click local launcher (Windows, standard library only).

One command brings the whole stack up:

    llama-server (llama.cpp Vulkan, GPU)  ->  127.0.0.1:8081   inference engine
    startlux_local.py  (/v1/systemone)    ->  127.0.0.1:8090   decision API

Usage
    python run_local.py                      boot everything, then run the demo call
    python run_local.py --mario              ... and then play level 1-1 with jev-mario
    python run_local.py --mario --parallel 6 ... with the 11 options evaluated in parallel
    python run_local.py --follow             keep a live health line; Ctrl+C stops everything
    python run_local.py --status             what is up right now, and on which port
    python run_local.py --stop               shut the stack down
    python run_local.py --restart            --stop then boot
    python run_local.py --model 2b           use the 2B Q4_K_M engine instead of 0.8B Q8_0
    python run_local.py --model cpu          CPU-only fallback (0.8B, port 8082)
    python run_local.py --selftest           boot, then exercise all four question types
    python run_local.py --bench              boot, then run the latency benchmark

By default the two services are detached: this command returns, they keep running, and
stop.cmd shuts them down.  With --follow the window owns them and Ctrl+C stops them.

The settings below are the ones measured on this machine (GTX 1050 / 3 GB); see README.md
for why each one is what it is.  Nothing here touches the upstream Startlux-Decision
checkout -- local-run/gguf-local/startlux_local.py only reads startlux_decision/jevfmt.py
from it (pure standard library).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
LOCAL_RUN = ROOT / "local-run"
LLAMA_DIR = LOCAL_RUN / "llama"
MODEL_ROOT = LOCAL_RUN / "models"
ENGINE = LOCAL_RUN / "gguf-local" / "startlux_local.py"
LOG_DIR = LOCAL_RUN / "logs"
REPO = ROOT / "Startlux-Decision"
MARIO_DIR = ROOT / "mario"

# The managed CPython this project has always used.  STARTLUX_PY overrides it.
PYTHON = os.environ.get(
    "STARTLUX_PY",
    r"C:\Users\Libai\.workbuddy\binaries\python\versions\3.13.12\python.exe",
)
# Separate interpreter for the jev-mario harness: it needs gymnasium + nes-py 9.0.1, which
# must not leak into the decision stack (see mario/README.md).
MARIO_PYTHON = os.environ.get(
    "STARTLUX_MARIO_PY",
    r"C:\Users\Libai\.workbuddy\binaries\python\envs\mario\Scripts\python.exe",
)

# --------------------------------------------------------------------------------------
# engine profiles -- one row per (model, device) combination that was actually validated
# --------------------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    # GPU 1286 MiB total (weights + KV).  Recommended: the KV cache fits in VRAM, so the
    # 4 slots stay warm and a repeated request costs ~16 tokens instead of 727.
    "0.8b": {
        "model": "StartLux-Decision-0.8B-Q8_0-GGUF",
        "port": 8081,
        "ctx": 32768,
        "np": 4,
        "gpu": True,
        "extra": [],
        "note": "0.8B Q8_0 on GPU -- recommended",
    },
    # 1215 MB weights fit, but the GPU KV cache does not: --no-kv-offload is mandatory or the
    # Vulkan device is lost ~13 s into loading.  KV therefore lives in system RAM.
    "2b": {
        "model": "StartLux-Decision-2B-Q4_K_M-GGUF",
        "port": 8081,
        "ctx": 32768,
        "np": 4,
        "gpu": True,
        "extra": ["--no-kv-offload"],
        "note": "2B Q4_K_M on GPU, KV in system RAM",
    },
    # No GPU at all.  ~684 ms per decision instead of ~413 ms, but it leaves the GPU free.
    "cpu": {
        "model": "StartLux-Decision-0.8B-Q8_0-GGUF",
        "port": 8082,
        "ctx": 32768,
        "np": 1,
        "gpu": False,
        "extra": [],
        "note": "0.8B Q8_0 on CPU (no GPU)",
    },
}
DEFAULT_PROFILE = "0.8b"

DECISION_PORT = 8090  # shared: the client always talks to this port


# --------------------------------------------------------------------------------------
# console helpers -- every Chinese message is printed from Python, never from the .cmd
# wrapper, so the batch files can stay pure ASCII and cannot be mangled by the code page.
# --------------------------------------------------------------------------------------


def _init_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def _head(tag: str, msg: str) -> None:
    print(f"\n==> {msg}" if tag == "step" else f"    [{tag}]  {msg}", flush=True)


def step(msg: str) -> None:
    _head("step", msg)


def ok(msg: str) -> None:
    _head("ok", msg)


def info(msg: str) -> None:
    _head("..", msg)


def warn(msg: str) -> None:
    _head("!!", msg)


def fail(msg: str) -> None:
    _head("XX", msg)


# --------------------------------------------------------------------------------------
# process / network plumbing
# --------------------------------------------------------------------------------------


def clean_env() -> dict:
    """A sane environment: exactly one PATH key, plus the OpenMP workaround.

    This shell hands out a duplicate Path/PATH pair, which breaks anything that resolves the
    environment case-insensitively (that is why ../_spawn.py exists).
    """
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        if k.upper() == "PATH":
            if "PATH" not in env:
                env["PATH"] = v
            continue
        env.setdefault(k, v)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUTF8"] = "1"
    return env


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def port_pid(port: int) -> int | None:
    """The PID listening on `port`, via netstat.  None when nothing listens."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True, text=True, errors="replace", timeout=20,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].upper() != "TCP":
            continue
        if parts[1].rsplit(":", 1)[-1] != str(port):
            continue
        if parts[-1].isdigit() and int(parts[-1]) > 0:
            return int(parts[-1])
    return None


def proc_name(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=20,
        ).stdout.strip()
        if out.startswith('"'):
            return out.split('","')[0].strip('"')
    except Exception:
        pass
    return ""


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    return proc_name(pid).lower().endswith(".exe")


def kill_tree(pid: int, expect: tuple[str, ...]) -> bool:
    """taskkill a PID, but only after confirming the image name looks right."""
    name = proc_name(pid)
    if not name:
        return False
    if expect and name.lower() not in [e.lower() for e in expect]:
        warn(f"PID {pid} 是 {name}，不是预期的 {expect}，跳过")
        return False
    r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True, errors="replace")
    return r.returncode == 0


def http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, repr(e)


def tail_lines(path: Path, n: int = 1) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    # skip the "$ <argv>" header we wrote at the top of the log
    lines = [ln for ln in raw if ln.strip() and not ln.startswith("$ ")]
    return lines[-n:]


# --------------------------------------------------------------------------------------
# service descriptor
# --------------------------------------------------------------------------------------


class Service:
    """A detached child process plus a JSON state file recording how it was started."""

    def __init__(self, name: str, port: int, health_url: str, ready,
                 statefile: Path, expect: tuple[str, ...]):
        self.name = name
        self.port = port
        self.health_url = health_url
        self.ready = ready            # (status, body) -> bool
        self.statefile = statefile
        self.expect = expect

    # -- state -------------------------------------------------------------------------
    def state(self) -> dict:
        try:
            return json.loads(self.statefile.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def listening(self) -> bool:
        return port_open(self.port)

    def healthy(self) -> bool:
        if not self.listening():
            return False
        status, body = http_get(self.health_url)
        try:
            return self.ready(status, body)
        except Exception:
            return False

    def recorded_pid(self) -> int | None:
        pid = self.state().get("pid")
        return int(pid) if isinstance(pid, int) else None

    def recorded_profile(self) -> str | None:
        """The profile we started, but only while that very process is still alive.

        A stale state file must not be trusted: an unrelated server (e.g. one started by
        local-run\\1-start-llama-0.8b-gpu.cmd) may be the one holding the port now, and
        judging it by our old record would refuse a perfectly good boot.
        """
        pid = self.recorded_pid()
        if pid and _alive(pid):
            return self.state().get("profile")
        return None

    def running_pid(self) -> int | None:
        """The PID of the live listener: our own child if it is alive, else whoever holds the port."""
        pid = self.recorded_pid()
        if pid and _alive(pid):
            return pid
        return port_pid(self.port)

    # -- lifecycle ---------------------------------------------------------------------
    def spawn(self, argv: list[str], cwd: Path, log: Path, **state_extra) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log, "w", encoding="utf-8", buffering=1)
        handle.write(f"$ {' '.join(argv)}\n\n")
        flags = 0
        if os.name == "nt":
            flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
        p = subprocess.Popen(argv, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags, env=clean_env())
        self.statefile.parent.mkdir(parents=True, exist_ok=True)
        self.statefile.write_text(json.dumps(
            {"pid": p.pid, "port": self.port, "argv": argv, "log": str(log), **state_extra},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return p.pid

    def wait_ready(self, log: Path, timeout: float, label: str) -> bool:
        t0 = time.time()
        shown: set[str] = set()
        while time.time() - t0 < timeout:
            if self.healthy():
                return True
            pid = self.recorded_pid()
            if pid and not _alive(pid):
                fail(f"{label} 进程已退出，日志末尾：")
                for ln in tail_lines(log, 8):
                    print(f"           {ln}", flush=True)
                return False
            for ln in tail_lines(log, 1):
                if ln not in shown:
                    shown.add(ln)
                    info(f"{int(time.time() - t0):>3d}s  {ln[:150]}")
            time.sleep(0.6)
        fail(f"{label} 在 {timeout:.0f}s 内没有就绪；日志：{log}")
        return False

    def stop(self) -> bool:
        pid = self.running_pid()
        if pid is None:
            ok(f"{self.name} 未在运行")
            self._forget()
            return True
        if not kill_tree(pid, self.expect):
            fail(f"{self.name} 停止失败（PID {pid}，{proc_name(pid) or '?'}）")
            return False
        ok(f"{self.name} 已停止（PID {pid}）")
        self._forget()
        for _ in range(20):
            if not self.listening():
                return True
            time.sleep(0.25)
        return not self.listening()

    def _forget(self) -> None:
        try:
            self.statefile.unlink()
        except Exception:
            pass


def llama_service(port: int) -> Service:
    def ready(status: int, body: str) -> bool:
        if status != 200:
            return False
        try:
            return json.loads(body).get("status") == "ok"
        except Exception:
            return '"ok"' in body

    return Service("llama-server", port, f"http://127.0.0.1:{port}/health", ready,
                   LOG_DIR / f"llama-server.{port}.json", ("llama-server.exe",))


def decision_service() -> Service:
    def ready(status: int, body: str) -> bool:
        if status != 200:
            return False
        try:
            return bool(json.loads(body).get("models"))
        except Exception:
            return False

    return Service("决策服务", DECISION_PORT, f"http://127.0.0.1:{DECISION_PORT}/v1/models",
                   ready, LOG_DIR / "decision-server.json",
                   ("python.exe", "python3.13.exe", "python3.exe"))


LLAMA_PORTS = sorted({p["port"] for p in PROFILES.values()})


# --------------------------------------------------------------------------------------
# hardware / model discovery
# --------------------------------------------------------------------------------------


def list_devices() -> dict[str, str]:
    """{'Vulkan0': 'Intel(R) UHD Graphics 630 (8255 MiB, 7465 MiB free)', ...}"""
    exe = LLAMA_DIR / "llama-server.exe"
    if not exe.exists():
        return {}
    try:
        out = subprocess.run([str(exe), "--list-devices"], cwd=str(LLAMA_DIR),
                             capture_output=True, text=True, errors="replace",
                             timeout=60).stdout
    except Exception:
        return {}
    found = {}
    for ln in out.splitlines():
        m = re.match(r"\s*([A-Za-z]+\d+):\s*(.+?)\s*$", ln)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def pick_gpu(devices: dict[str, str]) -> str | None:
    """The discrete GPU: NVIDIA by name, else Vulkan1, else the only device, else None."""
    for idx, desc in devices.items():
        if re.search(r"nvidia|geforce|rtx|gtx|quadro|tesla", desc, re.I):
            return idx
    if "Vulkan1" in devices:
        return "Vulkan1"
    if len(devices) == 1:
        return next(iter(devices))
    return None


def gguf_in(model_dir_name: str) -> Path | None:
    hits = sorted(glob.glob(str(MODEL_ROOT / model_dir_name / "*.gguf")))
    return Path(hits[0]) if hits else None


def human_mib(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1048576:.0f} MiB"
    except Exception:
        return "?"


# --------------------------------------------------------------------------------------
# the API client (same shape as local-run/demo_call.py, inlined so this file stands alone)
# --------------------------------------------------------------------------------------

SMOKE = {
    "state": {"probe": "cold-start sanity check", "note": "run_local.py step 3"},
    "questions": {
        "sanity": {
            "type": "choice",
            "instructions": "Which of these is a colour?",
            "criteria": {"red": "the colour of blood", "loud": "a measure of volume"},
        }
    },
}


def client_call(port: int, payload: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    body = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except Exception as e:
        fail(f"请求失败：{e!r}")
    return body, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------------------


def boot(profile_key: str, device_arg: str | None, use_gpu: bool,
         timeout_llama: float, timeout_decision: float, restart: bool) -> bool:
    prof = PROFILES[profile_key]
    llama_port = prof["port"]
    llama = llama_service(llama_port)
    decision = decision_service()

    model = gguf_in(prof["model"])
    if model is None:
        fail(f"找不到模型：{MODEL_ROOT / prof['model']}/*.gguf")
        info("用 python local-run/fetch_model.py <repo> <file> 下载")
        return False
    if not ENGINE.exists():
        fail(f"找不到决策引擎：{ENGINE}")
        return False
    engine_repo = REPO if (REPO / "startlux_decision" / "jevfmt.py").exists() else None
    if engine_repo is None:
        fail(f"找不到 Startlux-Decision 仓库：{REPO}")
        return False

    if restart:
        step("先停掉现有服务")
        for svc in (decision,) + tuple(llama_service(p) for p in LLAMA_PORTS):
            svc.stop()

    # -- 1. llama-server ----------------------------------------------------------------
    step(f"1/3  推理引擎  llama.cpp Vulkan  ->  127.0.0.1:{llama_port}")
    if llama.healthy():
        running = llama.recorded_profile()
        if running and running != profile_key:
            fail(f"端口 {llama_port} 上运行的是 {running} 档，不是你要的 {profile_key}")
            info("用 --restart 切换档位")
            return False
        ok(f"已在运行，直接复用（端口 {llama_port}"
           + (f"，档位 {running}）" if running else "，外部启动，档位未知）"))
    else:
        if llama.listening():
            fail(f"端口 {llama_port} 被 PID {port_pid(llama_port)} 占用但不健康")
            info("用 --stop 或 --restart 处理")
            return False

        gpu = use_gpu and prof["gpu"]
        device = None
        if gpu:
            devices = list_devices()
            for idx, desc in devices.items():
                info(f"{idx}: {desc}")
            device = device_arg or pick_gpu(devices)
            if device is None:
                warn("没有发现可用的 Vulkan 设备，退回 CPU")
                gpu = False

        argv = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(model)]
        argv += ["--device", device, "-ngl", "99"] if device else ["-ngl", "0"]
        argv += ["-c", str(prof["ctx"]), "-np", str(prof["np"]), *prof["extra"],
                 "--jinja", "--host", "127.0.0.1", "--port", str(llama_port)]

        log = LOG_DIR / f"llama-server.{profile_key}.log"
        info(f"model: {model.name}  ({human_mib(model)})")
        info(f"args : {' '.join(argv[1:])}")
        pid = llama.spawn(argv, LLAMA_DIR, log, profile=profile_key, model=model.name,
                          device=device or "cpu")
        info(f"pid  : {pid}    日志: {log}")
        if not llama.wait_ready(log, timeout_llama, "推理引擎"):
            llama.stop()
            return False
        ok(f"就绪（{device or 'CPU'}）")

    # -- 2. decision service ------------------------------------------------------------
    step(f"2/3  决策服务  POST /v1/systemone  ->  127.0.0.1:{DECISION_PORT}")
    use_model_dir = MODEL_ROOT / prof["model"]
    if decision.healthy():
        running = decision.recorded_profile()
        if running and running != profile_key:
            fail(f"决策服务指向的是 {running} 档，不是你要的 {profile_key}")
            info("用 --restart 切换档位")
            return False
        ok(f"已在运行，直接复用（" + (f"档位 {running}）" if running
                                    else "外部启动，档位未知）"))
    else:
        if decision.listening():
            fail(f"端口 {DECISION_PORT} 被 PID {port_pid(DECISION_PORT)} 占用但不健康")
            info("用 --stop 或 --restart 处理")
            return False
        # --concurrency must be <= the server's -np, or the questions queue up serially instead
        # of batching into one forward pass (measured: 3 x 467 ms vs one 322 ms pass).
        per_slot = prof["ctx"] // max(prof["np"], 1)
        argv = [PYTHON, "-X", "utf8", str(ENGINE),
                "--model-dir", str(use_model_dir),
                "--llama", f"http://127.0.0.1:{llama_port}",
                "--repo", str(engine_repo),
                "--port", str(DECISION_PORT),
                "--max-length", str(per_slot),
                "--concurrency", str(prof["np"])]
        log = LOG_DIR / f"decision-server.{profile_key}.log"
        info(f"args : {' '.join(argv[3:])}")
        pid = decision.spawn(argv, LOCAL_RUN, log, profile=profile_key, engine=llama_port)
        info(f"pid  : {pid}    日志: {log}")
        if not decision.wait_ready(log, timeout_decision, "决策服务"):
            decision.stop()
            return False
        ok(f"就绪（每槽 {per_slot} token）")

    # -- 3. smoke test ------------------------------------------------------------------
    step("3/3  冒烟测试")
    raw, ms = client_call(DECISION_PORT, SMOKE, timeout=300)
    if raw is None:
        fail("决策请求失败")
        return False
    ans = raw["answers"]["sanity"]
    ok(f"往返 {ms:.0f} ms，回答 = {ans.get('choice')} "
       f"(p={ans.get('probabilities', {}).get(ans.get('choice'), 0):.2f})")
    return True


# --------------------------------------------------------------------------------------
# subsystems: demo / selftest / bench / mario
# --------------------------------------------------------------------------------------


def run_script(label: str, argv: list[str], cwd: Path) -> int:
    step(label)
    say(f"    $ {' '.join(argv)}")
    try:
        return subprocess.run(argv, cwd=str(cwd), env=clean_env()).returncode
    except KeyboardInterrupt:
        warn("已中断")
        return 130
    except Exception as e:
        fail(repr(e))
        return 1


def run_demo(profile_key: str, repeat: int) -> int:
    return run_script("调用示例（POST /v1/systemone，4 个问题）",
                      [PYTHON, "-X", "utf8", "demo_call.py", "--file", "demo-request.json",
                       "--repeat", str(repeat)], LOCAL_RUN)


def run_selftest(profile_key: str) -> int:
    prof = PROFILES[profile_key]
    return run_script("自测：四类问题全跑一遍",
                      [PYTHON, "-X", "utf8", "selftest.py",
                       "--model-dir", str(MODEL_ROOT / prof["model"]),
                       "--llama", f"http://127.0.0.1:{prof['port']}"],
                      LOCAL_RUN / "gguf-local")


def run_bench(profile_key: str) -> int:
    prof = PROFILES[profile_key]
    return run_script("延迟基准",
                      [PYTHON, "-X", "utf8", "bench_local.py",
                       "--model-dir", str(MODEL_ROOT / prof["model"]),
                       "--llama", f"http://127.0.0.1:{prof['port']}",
                       "--label", profile_key], LOCAL_RUN / "gguf-local")


def run_mario(level: str, parallel: int, label: str | None) -> int:
    if not MARIO_DIR.exists():
        fail(f"找不到 mario 目录：{MARIO_DIR}")
        return 1
    if not Path(MARIO_PYTHON).exists():
        fail(f"找不到 jev-mario 的 Python：{MARIO_PYTHON}")
        info("见 mario/README.md 里的 venv 创建命令")
        return 1
    tag = label or ("startlux-0.8b-par" if parallel > 1 else "startlux-0.8b")
    argv = [MARIO_PYTHON, "-u", "mario_launch.py", "branch.py", "--bot", "jev",
            "--level", level,
            "--url", f"http://127.0.0.1:{DECISION_PORT}/v1/systemone", "--label", tag]
    if parallel > 1:
        argv += ["--parallel", str(parallel)]
        if parallel > 6:
            warn("并行进程超过 6 时 llama-server 可能报 device lost；实测 6 是安全上限")
    return run_script(f"马里奥官方 harness（关卡 {level}）", argv, MARIO_DIR)


# --------------------------------------------------------------------------------------
# status / stop / follow
# --------------------------------------------------------------------------------------


def _rows() -> list[tuple[str, Service]]:
    rows = [(f"llama-server  :{p}", llama_service(p)) for p in LLAMA_PORTS]
    rows.append((f"决策服务      :{DECISION_PORT}", decision_service()))
    return rows


def status(quiet: bool = False) -> int:
    if not quiet:
        step("当前状态")
    for name, svc in _rows():
        if not port_open(svc.port):
            say(f"    [--]   {name:24s}未运行")
            continue
        pid = svc.running_pid()
        st, _ = http_get(svc.health_url)
        prof = svc.recorded_profile()
        say(f"    [ok]   {name:24s}监听中  PID {pid} "
            f"({proc_name(pid) or '?'})  {'健康' if st == 200 else f'HTTP {st}'}"
            + (f"  档位 {prof}" if prof else ""))
    if not quiet:
        say()
        if not port_open(DECISION_PORT):
            info("决策服务未运行：双击 run.cmd 启动")
        else:
            info(f"调用地址 http://127.0.0.1:{DECISION_PORT}/v1/systemone")
        info(f"日志目录 {LOG_DIR}")
    return 0


def stop_all(quiet: bool = False) -> int:
    if not quiet:
        step("停止服务")
    rc = 0
    for _, svc in _rows():
        if not svc.stop():
            rc = 1
    if not quiet:
        say()
        if rc == 0:
            ok("全部停止")
        else:
            fail("有服务没能停掉，见上方消息")
    return rc


def follow(profile_key: str) -> int:
    """Keep a live health line in this window; Ctrl+C stops everything."""
    step("监视模式：本窗口持有服务，按 Ctrl+C 停止全部")
    t0 = time.time()
    try:
        while True:
            bits, up = [], 0
            for name, svc in _rows():
                short = name.split()[0]
                if not port_open(svc.port):
                    bits.append(f"{short}:{svc.port}=停止")
                    continue
                st, _ = http_get(svc.health_url, timeout=2.0)
                if st == 200:
                    up += 1
                bits.append(f"{short}:{svc.port}=" + ("ok" if st == 200 else f"HTTP{st}"))
            print(f"    [{time.strftime('%H:%M:%S')}] 运行 {time.time() - t0:6.0f}s  "
                  + "  ".join(bits), flush=True)
            if up == 0 and time.time() - t0 > 10:
                say()
                warn("服务已全部停止（可能是另一个窗口执行了 stop.cmd），退出监视")
                return 0
            time.sleep(5)
    except KeyboardInterrupt:
        say()
        stop_all()
        return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _init_console()

    ap = argparse.ArgumentParser(
        prog="run_local.py",
        description="StartLux-Decision 本地一键运行（llama.cpp Vulkan + /v1/systemone）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", choices=sorted(PROFILES), default=DEFAULT_PROFILE,
                    help="引擎档位（默认 0.8b）")
    ap.add_argument("--device", default=None,
                    help="Vulkan 设备名，如 Vulkan1（默认自动挑 NVIDIA 独显）")
    ap.add_argument("--no-gpu", action="store_true", help="强制 CPU 推理（等价于 --model cpu）")
    ap.add_argument("--status", action="store_true", help="只看状态，不做任何改动")
    ap.add_argument("--stop", action="store_true", help="停止全部服务后退出")
    ap.add_argument("--restart", action="store_true", help="先停再起")
    ap.add_argument("--follow", action="store_true", help="前台监视；Ctrl+C 停止全部")
    ap.add_argument("--selftest", action="store_true", help="启动后跑四类问题自测")
    ap.add_argument("--bench", action="store_true", help="启动后跑延迟基准")
    ap.add_argument("--mario", action="store_true", help="启动后跑官方马里奥 harness")
    ap.add_argument("--level", default="1-1", help="马里奥关卡（默认 1-1）")
    ap.add_argument("--parallel", type=int, default=1,
                    help="马里奥：11 个选项的并行进程数，实测安全上限 6")
    ap.add_argument("--label", default=None, help="马里奥录像标签")
    ap.add_argument("--repeat", type=int, default=1, help="demo 重复次数，看冷/热延迟")
    ap.add_argument("--no-demo", action="store_true", help="不跑 demo 调用")
    ap.add_argument("--timeout", type=float, default=300.0, help="单步就绪等待上限（秒）")
    ap.add_argument("--no-pause", action="store_true", help="结束时不等待回车")
    a = ap.parse_args(argv)

    say("=" * 74)
    say("  StartLux-Decision  本地一键运行")
    say(f"  工作目录 {ROOT}")
    say("=" * 74)

    if a.status:
        rc = status()
        _pause(a)
        return rc
    if a.stop:
        rc = stop_all()
        _pause(a)
        return rc

    if a.no_gpu:
        a.model = "cpu"

    for path, what, hint in (
        (LLAMA_DIR / "llama-server.exe", "llama.cpp 引擎",
         "从 https://github.com/ggml-org/llama.cpp/releases 下载 "
         "llama-b11417-bin-win-vulkan-x64.zip，把内容解压到 local-run/llama/"),
        (Path(PYTHON), "Python 解释器",
         "设置环境变量 STARTLUX_PY 指向你的 python.exe"),
    ):
        if not path.exists():
            fail(f"缺少{what}：{path}")
            info(hint)
            _pause(a)
            return 1

    t0 = time.time()
    if not boot(a.model, a.device, not a.no_gpu, a.timeout, 120.0, a.restart):
        say()
        fail(f"启动失败。日志在 {LOG_DIR}")
        _pause(a)
        return 1

    prof = PROFILES[a.model]
    say()
    say("-" * 74)
    ok(f"全部就绪，用时 {time.time() - t0:.1f}s")
    say(f"    引擎档位          {prof['note']}   [--model {a.model}]")
    say(f"    推理引擎          http://127.0.0.1:{prof['port']}   (llama.cpp, 仅本机)")
    say(f"    决策 API          http://127.0.0.1:{DECISION_PORT}/v1/systemone")
    say(f"    接第三方 harness   --url http://127.0.0.1:{DECISION_PORT}/v1/systemone")
    if a.follow:
        say("    停止              Ctrl+C（本窗口持有服务）")
    else:
        say("    停止              stop.cmd  或  python run_local.py --stop")
    say("-" * 74)

    rc = 0
    if not a.no_demo:
        rc |= run_demo(a.model, a.repeat)
    if a.selftest:
        rc |= run_selftest(a.model)
    if a.bench:
        rc |= run_bench(a.model)
    if a.mario:
        rc |= run_mario(a.level, a.parallel, a.label)

    if a.follow:
        follow(a.model)
        return rc

    say()
    if rc == 0:
        ok("完成。服务仍在后台运行，退出时请用 stop.cmd。")
    else:
        warn(f"有子任务返回非零退出码（{rc}），详见上方输出。")
    _pause(a)
    return 0


def _pause(a) -> None:
    if a.no_pause or not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\n按回车关闭本窗口（后台服务继续运行）...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
