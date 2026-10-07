"""为未经改动的 4esv/jev-mario harness 做选项并行评估。

为什么有东西可以并行
--------------------
`branch.py` 从同一个模拟器快照出发评估一次决策：对 11 个选项中的每一个，先把它跑一秒
（HORIZON 帧，跳起后一直按到落地），再从落点跑完 6x6 共 36 条两步后续路径。每个选项都从
同一个快照起步，所以这 11 次推演彼此独立。串行跑它们，就是本机上一次决策要花约 138 秒的原因；
本模块把这一批推演扇出到一个进程池上。

worker 怎么被摆到正确的位置
--------------------------
最自然的传输方式本该是快照：`nes-py 9.0.1` 提供了 `NESEnv.dump_state()`，但它返回的是不透明的
Cython `NativeStateSnapshot`，既不支持 pickle 也不支持 buffer 协议，所以出不了进程。
（这是实测结论不是猜测：`pickle.dumps` 报 "no default __reduce__ due to non-trivial __cinit__"。）

于是改成让 worker 重放。模拟器是确定性的，而父进程会记录自己施加在 env 上的精确输入历史——
`Sim.step(action, record)` 在真实推进时带 `record`，在模拟分支时传 `record=None`，
因此"真实路径"恰好就是所有带 record 的那些步。worker 从 `reset()` 重放这段历史，
就能逐位落到同一个状态；又因为决策只会往历史上追加，已经重放过前缀的 worker 只需接着放新增的帧。
摆好位置之后，worker 跑的是**上游自己的** `Sim.outcome()`，所以选项语义在构造上就没有改变——
本模块没有添加任何自己的模拟逻辑。

jev-mario-main/ 里没有任何东西被修改：branch.py 被 exec 进一个私有命名空间，只有它的
`Sim` 类是从外部扩展的。

关于被跳过的那些状态，正确性说明
--------------------------------
`gym_super_mario_bros` 保留了一些 Python 侧的缓存，但全都是奖励增量
（`_x_position_max`、`_time_last`、`_score_last`、`_coins_last`、`_status_last`），
而 branch.py 从不读奖励。它真正读的东西——`_is_dying`、`_is_dead`、`_flag_get`、`_time`、
x 坐标、是否在空中——全都是直接读 RAM。`probe_par.py` 验证了重放后 RAM 与画面完全一致，
且 11 份 outcome 字典与串行路径逐字段相同。
"""
import ast
import hashlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "jev-mario-main"
BRANCH = HARNESS / "branch.py"

# branch.py 和 play.py 互相以顶层模块的方式 import，所以每个加载它们的进程（父进程和每个
# 池 worker）里 harness 目录都必须可导入。
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


def apply_shim():
    """容忍 branch.py 传给 make() 的 `apply_api_compatibility`。

    gym-super-mario-bros 9.x 建在 gymnasium 上，后者已经删掉了这个 kwarg；而上游 pyproject.toml
    钉的 7.4.0（gym 0.26.2）接受它。所以这里按上游原样调用，只有当本栈拒绝时才去掉 kwarg 重试——
    这样同一个启动器对两种栈都能用（本机 venv，以及容器里的官方 pin）。
    """
    import gym_super_mario_bros as gsmb

    if getattr(gsmb, "_jev_api_shim", False):
        return
    raw_make = gsmb.make

    def make(env_id, *args, **kwargs):  # noqa: A002 - 与上游签名保持一致
        try:
            return raw_make(env_id, *args, **kwargs)
        except TypeError as exc:
            if "apply_api_compatibility" not in str(exc):
                raise
            kwargs.pop("apply_api_compatibility", None)
            return raw_make(env_id, *args, **kwargs)

    gsmb.make = make
    gsmb._jev_api_shim = True


def _branch_source() -> str:
    return BRANCH.read_text(encoding="utf-8")


def load_branch() -> dict:
    """把上游 branch.py 执行进一个私有命名空间；它的 __main__ 守卫不会触发。"""
    ns = {"__name__": "jev_branch", "__file__": str(BRANCH)}
    exec(compile(_branch_source(), str(BRANCH), "exec"), ns)  # noqa: S102 - 上游文件
    return ns


def run_main_block(ns: dict) -> None:
    """执行上游 `if __name__ == "__main__":` 里的那段，从而它的 CLI 保持不变。"""
    for node in ast.parse(_branch_source()).body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            ns["__name__"] = "__main__"
            body = ast.Module(body=node.body, type_ignores=[])
            exec(compile(body, str(BRANCH), "exec"), ns)  # noqa: S102 - 上游文件
            return
    raise SystemExit("branch.py: no __main__ guard found")


# --------------------------------------------------------------------------- worker 侧
_W: dict = {}


def _init_worker(harness: str) -> None:
    if harness not in sys.path:
        sys.path.insert(0, harness)
    apply_shim()
    _W["ns"] = load_branch()
    _W["sim"] = None
    _W["level"] = None
    _W["hist"] = []


def _ensure_sim(level: str):
    ns = _W["ns"]
    sim = _W["sim"]
    if sim is None or _W["level"] != level:
        if sim is not None:
            sim.env.close()
        sim = _W["sim"] = ns["Sim"](level)
        _W["level"] = level
        _W["hist"] = []
    return sim


def _position(sim, level: str, hist: list) -> None:
    """重放父进程的真实输入历史，让本 worker 的模拟器与父进程状态一致。"""
    cur = _W["hist"]
    if len(hist) >= len(cur) and hist[: len(cur)] == cur:
        new = hist[len(cur):]  # 常见情况：上一次决策之后只是追加了若干帧
    else:  # 历史不是本 worker 已放过的延续：从头干净重置
        sim.core._has_backup = False  # 否则 NESEnv.reset() 会去还原我们那个备份槽
        sim.env.reset()
        _W["hist"] = []
        new = hist
    for a in new:
        sim.step(a, None)
    _W["hist"] = list(hist)


def _worker_outcome(task: tuple) -> tuple:
    """把本 worker 摆到父进程的状态，然后跑上游的 Sim.outcome()。"""
    level, hist, info, name = task
    sim = _ensure_sim(level)
    _position(sim, level, hist)
    sim.core.done = False
    sim.done = False
    sim.info = dict(info)
    return name, _W["ns"]["Sim"].outcome(sim, name)


def _worker_check_state(task: tuple) -> dict:
    """重放父进程历史并回报得到的 RAM 与画面，供父进程与自己的对比（这是跨进程的一致性检查，
    不是同进程内的）。"""
    level, hist = task
    sim = _ensure_sim(level)
    _position(sim, level, hist)
    return {
        "ram": hashlib.blake2b(bytes(sim.core.ram), digest_size=16).hexdigest(),
        "screen": hashlib.blake2b(sim.core.screen.tobytes(), digest_size=16).hexdigest(),
        "x": sim.x(),
        "frames": len(_W["hist"]),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- 父进程侧
def patch(ns: dict, workers: int, quiet: bool = False) -> "mp.pool.Pool":
    """给上游 Sim 扩展出并行选项评估，并返回进程池。

    可以反复调用（探针会建好几个池）：上游自己的 outcome() 和 step() 只在第一次调用时记下来，
    后续调用只是围绕它们重建分发层。
    """
    Sim = ns["Sim"]
    options = list(ns["OPTIONS"])
    if not getattr(Sim, "_par_patched", False):
        Sim._serial_outcome = Sim.outcome
        Sim._serial_step = Sim.step
        upstream_init = Sim.__init__

        def __init__(self, level):  # noqa: N807 - 这是给上游类打补丁
            upstream_init(self, level)
            self.level = level
            self.real_hist = []  # harness 真正走过的步，用于和模拟分支区分

        def step(self, a, record):  # noqa: N807 - 这是给上游方法打补丁
            done = Sim._serial_step(self, a, record)
            if record is not None:  # run() 只对真实步传它的帧列表
                self.real_hist.append(a)
            return done

        Sim.__init__ = __init__
        Sim.step = step
        Sim._par_patched = True

    pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker,
                                        initargs=(str(HARNESS),))

    batch: dict = {"key": None, "res": None, "t": 0.0, "n": 0}

    def outcome(self, name):  # noqa: N807 - 这是给上游方法打补丁
        info = dict(self.info)
        # 批次键 = （完整真实输入历史, 各字段取值）。同一个状态下的 11 个选项共享一次派发。
        key = (tuple(self.real_hist), tuple(sorted(info.items())))
        if batch["key"] != key:
            # 一次决策会在同一状态下要全部 11 个选项：一起派出去。
            t0 = time.perf_counter()
            tasks = [(self.level, list(key[0]), info, n) for n in options]
            batch["res"] = dict(pool.map(_worker_outcome, tasks))
            batch["t"] = time.perf_counter() - t0
            batch["key"] = key
            batch["n"] += 1
            if not quiet:
                print(f"[parallel] 决策 {batch['n']:02d}：模拟 {len(options)} 个选项用时 "
                      f"{batch['t']:6.1f}s（{workers} 进程，主环境已走 {len(key[0])} 帧）", flush=True)
        return batch["res"][name]

    Sim.outcome = outcome
    return pool
