"""Parallel option evaluation for the unmodified 4esv/jev-mario harness.

Why there is something to parallelise
-------------------------------------
`branch.py` evaluates a decision from one emulator snapshot: for each of the 11 options it plays the
option for one second (HORIZON frames, held until landing) and then all 6x6 two-move continuations
from where that ended.  Every option starts from the same snapshot, so the 11 rolls are independent
of each other.  Running them one after another is what makes a decision cost ~138 s on this machine;
the rolls are what this module fans out over a process pool.

How a worker is put in the right place
--------------------------------------
The obvious transport would be a snapshot: `nes-py 9.0.1` exposes `NESEnv.dump_state()`, but it returns
an opaque Cython `NativeStateSnapshot` that supports neither pickling nor the buffer protocol, so it
cannot leave the process.  (Measured, not assumed: `pickle.dumps` -> "no default __reduce__ due to
non-trivial __cinit__".)

So the workers replay instead.  The emulator is deterministic, and the parent records the exact input
history it applies to its own env -- `Sim.step(action, record)` is called with `record` set for real
steps and `record=None` for simulated ones, which makes the real path exactly the steps taken with a
record.  A worker replays that history from `reset()` and lands on the same state bit for bit; because
decisions only ever append to the history, a worker that already replayed a prefix just plays the new
frames.  After a worker is positioned, it runs *upstream's own* `Sim.outcome()`, so the option
semantics are unchanged by construction -- this module adds no simulation logic of its own.

Nothing in jev-mario-main/ is modified: branch.py is executed into a private namespace and only its
`Sim` class is extended from the outside.

Correctness note on the skipped state
-------------------------------------
`gym_super_mario_bros` keeps Python-side caches, but they are all reward deltas (`_x_position_max`,
`_time_last`, `_score_last`, `_coins_last`, `_status_last`), and branch.py never reads the reward.
Everything it does read -- `_is_dying`, `_is_dead`, `_flag_get`, `_time`, x, airborne -- is a direct
RAM read.  `probe_par.py` verifies the replay lands on an identical RAM and screen, and that the 11
outcome dicts match the serial path exactly.
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

# branch.py and play.py import each other as top-level modules, so the harness directory has to be
# importable in every process that loads it -- the parent and every pool worker.
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))


def apply_shim():
    """Tolerate `apply_api_compatibility`, which branch.py passes to make().

    gym-super-mario-bros 9.x is built on gymnasium, which dropped that kwarg; the 7.4.0 pin from
    upstream's pyproject.toml (gym 0.26.2) accepts it.  So the call is made as upstream wrote it and
    only retried without the kwarg if this stack is the one that rejects it -- which keeps the same
    launcher usable against either stack (host venv here, official pins in the container).
    """
    import gym_super_mario_bros as gsmb

    if getattr(gsmb, "_jev_api_shim", False):
        return
    raw_make = gsmb.make

    def make(env_id, *args, **kwargs):  # noqa: A002 - mirror upstream signature
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
    """Execute upstream branch.py into a private namespace; its __main__ guard does not fire."""
    ns = {"__name__": "jev_branch", "__file__": str(BRANCH)}
    exec(compile(_branch_source(), str(BRANCH), "exec"), ns)  # noqa: S102 - upstream file
    return ns


def run_main_block(ns: dict) -> None:
    """Run upstream's `if __name__ == "__main__":` body, so its CLI is unchanged."""
    for node in ast.parse(_branch_source()).body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            ns["__name__"] = "__main__"
            body = ast.Module(body=node.body, type_ignores=[])
            exec(compile(body, str(BRANCH), "exec"), ns)  # noqa: S102 - upstream file
            return
    raise SystemExit("branch.py: no __main__ guard found")


# --------------------------------------------------------------------------- worker side
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
    """Replay the parent's real input history so this worker's emulator matches its state."""
    cur = _W["hist"]
    if len(hist) >= len(cur) and hist[: len(cur)] == cur:
        new = hist[len(cur):]  # the common case: the decision only appended frames
    else:  # history is not a continuation of what this worker played: start from a clean reset
        sim.core._has_backup = False  # otherwise NESEnv.reset() restores our backup slot instead
        sim.env.reset()
        _W["hist"] = []
        new = hist
    for a in new:
        sim.step(a, None)
    _W["hist"] = list(hist)


def _worker_outcome(task: tuple) -> tuple:
    """Position this worker at the parent's state and run upstream Sim.outcome() on it."""
    level, hist, info, name = task
    sim = _ensure_sim(level)
    _position(sim, level, hist)
    sim.core.done = False
    sim.done = False
    sim.info = dict(info)
    return name, _W["ns"]["Sim"].outcome(sim, name)


def _worker_check_state(task: tuple) -> dict:
    """Replay the parent's history and report the resulting RAM and screen, so the parent can compare
    them with its own (this is the cross-process fidelity check, not a same-process one)."""
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


# --------------------------------------------------------------------------- parent side
def patch(ns: dict, workers: int, quiet: bool = False) -> "mp.pool.Pool":
    """Extend upstream Sim with parallel option evaluation and return the pool.

    Safe to call repeatedly (the probe builds several pools): upstream's own outcome() and step() are
    remembered on the first call, later calls only rebuild the dispatcher around them.
    """
    Sim = ns["Sim"]
    options = list(ns["OPTIONS"])
    if not getattr(Sim, "_par_patched", False):
        Sim._serial_outcome = Sim.outcome
        Sim._serial_step = Sim.step
        upstream_init = Sim.__init__

        def __init__(self, level):  # noqa: N807 - patching upstream's class
            upstream_init(self, level)
            self.level = level
            self.real_hist = []  # the steps the harness really took, as opposed to simulated branches

        def step(self, a, record):  # noqa: N807 - patching upstream's method
            done = Sim._serial_step(self, a, record)
            if record is not None:  # run() passes its frame list only for real steps
                self.real_hist.append(a)
            return done

        Sim.__init__ = __init__
        Sim.step = step
        Sim._par_patched = True

    pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker,
                                        initargs=(str(HARNESS),))

    batch: dict = {"key": None, "res": None, "t": 0.0, "n": 0}

    def outcome(self, name):  # noqa: N807 - patching upstream's method
        info = dict(self.info)
        key = (tuple(self.real_hist), tuple(sorted(info.items())))
        if batch["key"] != key:
            # One decision asks for all 11 options at the same state: dispatch them together.
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
