"""Two checks before the real run.

1) Contract: post exactly what branch.ask_jev posts, to the local server on 8090.
2) Cost: time one Sim.outcome() so the branch search's wall time can be estimated.
"""
import sys, time, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
HARNESS = HERE / "jev-mario-main"

import gym_super_mario_bros as gsmb
_make = gsmb.make
gsmb.make = lambda id, *a, **kw: (kw.pop("apply_api_compatibility", None), _make(id, *a, **kw))[1]

sys.path.insert(0, str(HARNESS))

import httpx
import branch
import play

URL = "http://127.0.0.1:8090/v1/systemone"

print("=" * 70)
print("1) CONTRACT TEST -- same body branch.ask_jev builds, 11 options")
print("=" * 70)
outcomes = {
    "run right": {"dead": False, "flag": False, "dx": 6, "frames": 60, "on_ground": True,
                  "alive_paths": 12, "best_gain": 14, "best_path": ["run right", "jump right"]},
    "run and jump right": {"dead": False, "flag": False, "dx": 9, "frames": 60, "on_ground": True,
                           "alive_paths": 16, "best_gain": 11, "best_path": ["run right", "run and jump right"]},
    "run then jump": {"dead": True, "flag": False, "dx": 3, "frames": 24, "on_ground": False,
                      "alive_paths": 0, "best_gain": None, "best_path": None},
    "short run then jump": {"dead": False, "flag": False, "dx": 5, "frames": 60, "on_ground": True,
                            "alive_paths": 8, "best_gain": 6, "best_path": ["run right", "jump right"]},
    "jump right": {"dead": False, "flag": False, "dx": 4, "frames": 60, "on_ground": True,
                   "alive_paths": 10, "best_gain": 9, "best_path": ["run right", "run right"]},
    "hop right": {"dead": False, "flag": False, "dx": 2, "frames": 60, "on_ground": True,
                  "alive_paths": 14, "best_gain": 5, "best_path": ["run right", "jump right"]},
    "jump in place": {"dead": False, "flag": False, "dx": 0, "frames": 60, "on_ground": True,
                      "alive_paths": 11, "best_gain": 7, "best_path": ["run right", "jump right"]},
    "stand": {"dead": False, "flag": False, "dx": 0, "frames": 60, "on_ground": True,
              "alive_paths": 9, "best_gain": 8, "best_path": ["run right", "jump right"]},
    "walk left": {"dead": False, "flag": False, "dx": -3, "frames": 60, "on_ground": True,
                  "alive_paths": 13, "best_gain": 12, "best_path": ["run right", "run and jump right"]},
    "bounce on the spring behind": {"dead": True, "flag": False, "dx": -1, "frames": 30, "on_ground": False,
                                    "alive_paths": 0, "best_gain": None, "best_path": None},
    "hop back onto the ledge behind": {"dead": False, "flag": False, "dx": -2, "frames": 48, "on_ground": True,
                                       "alive_paths": 6, "best_gain": 4, "best_path": ["run right", "jump right"]},
}
summary = ("No wall ahead. There are 4 tiles of clear ground behind Mario for a run-up. Solid ground ahead. "
           "Enemies: a goomba 8 tiles ahead at ground level. Mario is on the ground, moving right slowly. "
           "A jump from this speed clears 4 tiles high and 5 tiles far; at full running speed it clears 5 high and 9 far.")
try:
    name, probs, tok, lat = branch.ask_jev(httpx.Client(timeout=120), outcomes, summary, URL, "startlux")
    print(f"  OK  choice = {name!r}   latency = {lat*1000:.0f} ms   input_tokens = {tok}")
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
    for k, v in top:
        print(f"      {v*100:.1f}%  {k}")
    print(f"  sum(probabilities) = {sum(probs.values()):.4f} over {len(probs)} options")
except Exception as e:
    import traceback; traceback.print_exc()

print()
print("=" * 70)
print("2) EMULATOR COST -- time the branching search for one decision")
print("=" * 70)
t0 = time.perf_counter()
sim = branch.Sim("1-1")
print(f"  env build+reset: {time.perf_counter()-t0:.2f} s")
for i in range(3):
    t = time.perf_counter()
    o = sim.outcome("run right")
    print(f"  outcome('run right') #{i+1}: {time.perf_counter()-t:.2f} s -> {branch.describe(o)[:90]}")
t = time.perf_counter()
outs = {n: sim.outcome(n) for n in branch.OPTIONS}
dt = time.perf_counter() - t
print(f"  full 11-option sweep: {dt:.2f} s  -> a ~30-decision run is roughly {dt*30/60:.1f} min of emulation")
sim.env.close()
