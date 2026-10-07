"""Branching play: simulate every action before choosing, describe the outcomes, let Jev (or search) pick.

    uv run python branch.py --bot jev --level 1-1      # Jev picks among simulated outcomes
    uv run python branch.py --bot search --level 1-1   # picks by (alive, continuations alive, distance); no API

At each decision the emulator is snapshotted (nes-py _backup/_restore) and each option is played for one
second, held until Mario lands. Then every two-step continuation over four basic moves (16 paths) is played
from there, one second each. The option text Jev sees is the measured result, not a physics description:
    "alive after 1 s, +6 tiles, on the ground; 12 of 16 continuations survive, best +14 more via 'run right' then 'jump right'"
    "dies in 0.4 s"
The chosen option is executed exactly as simulated. If its value came only from its continuation, the first
move of that continuation is executed too. If nothing scores, a deeper escape search runs.
Writes runs/<level>-branch-<bot>-<stamp>.gif and a log next to it, plus a line in runs/results.jsonl.
"""

import argparse
import json
import time

import httpx

import play
from play import ACTIONS, HOP_FRAMES, JUMPS, RELEASE, RUNS, USD_PER_TOKEN, airborne, features, grid, nes

HORIZON = 60  # frames each move is played, then held until landing
SETTLE_FRAMES = 8  # extra frames after landing so a death on contact is flagged inside the outcome
OPTIONS = ["run right", "run and jump right", "run then jump", "short run then jump", "jump right", "hop right",
           "jump in place", "stand", "walk left", "bounce on the spring behind", "hop back onto the ledge behind"]
FOLLOW_UPS = ["run right", "jump right", "run and jump right", "run then jump", "short run then jump", "walk left"]  # 36 paths
FOLLOW = {"stand": 0, "walk left": 6}  # what to hold after the action itself; default is run right
COMPOSITE = {
    "run then jump": [("run right", 24), ("run and jump right", 60)],  # build full speed, then jump: 9 tiles
    "short run then jump": [("run right", 12), ("run and jump right", 60)],  # walking speed from a 2-tile ledge: 5 tiles
}
# Raw joypad sequences (SIMPLE_MOVEMENT index, frames) for frame-exact moves. Found by probe at the 2-1
# tower: step back onto the springboard, jump, land on it, press A as it releases. One frame off and it fails.
RAW = {
    "bounce on the spring behind": [(6, 2), (5, 40), (0, 6), (2, 90)],
    # The action set has no left+A: tap A, then steer left in the air. Lands on a ledge up to 2 tiles high behind.
    "hop back onto the ledge behind": [(5, 8), (6, 30), (0, 10)],
}
ESCAPE_MOVES = ["walk left", "jump in place", "jump right", "run then jump", "hop back onto the ledge behind"]


class Sim:
    """Drive one nes-py env; the same execute() plays real steps and simulated branches."""

    def __init__(self, level: str):
        self.env = play.JoypadSpace(
            play.gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0", apply_api_compatibility=True), play.SIMPLE_MOVEMENT)
        self.core = nes(self.env)
        self.ram = self.core.ram
        self.obs, _ = self.env.reset()
        self.info = {"x_pos": 40, "flag_get": False}
        self.done = False

    def x(self) -> int:
        return int(self.ram[0x6D]) * 256 + int(self.ram[0x86])

    def over(self) -> bool:
        return self.done or bool(self.info["flag_get"])

    def step(self, a: int, record: list | None) -> bool:
        self.obs, _, term, trunc, self.info = self.env.step(a)
        if record is not None:
            record.append(self.obs.copy())
        self.done = bool(term or trunc)
        return self.done

    def execute(self, name: str, budget: int, record: list | None = None) -> int:
        """Play a move as the harness would, hold its follow-through for budget frames, then until landing."""
        if name in RAW:
            used = 0
            for a, n in RAW[name]:
                for _ in range(n):
                    used += 1
                    if self.step(a, record):
                        return used
            return used
        if name in COMPOSITE:
            used = 0
            for part, part_budget in COMPOSITE[name]:
                used += self.execute(part, part_budget, record)
                if self.over():
                    break
            return used
        a = ACTIONS[name]
        used = 0
        if a in JUMPS:
            if self.step(0, record):  # the NES needs A released before a press; no direction, or a pit edge kills
                return 1
            used += 1
            cap = HOP_FRAMES if name == "hop right" else play.FULL_JUMP_FRAMES
            for i in range(cap):
                used += 1
                if self.step(a, record) or (i > 4 and not airborne(self.ram)):
                    break
            follow = 0  # a jump ends where it lands; walking on afterwards is a separate decision
        else:
            follow = FOLLOW.get(name, 3 if name == "run right" else a)
        while used < budget and not self.over():
            self.step(follow, record)
            used += 1
        # Never end mid-air: what happens on landing must be part of the same simulated outcome.
        while airborne(self.ram) and used < budget + 90 and not self.over():
            self.step(follow, record)
            used += 1
        for _ in range(SETTLE_FRAMES):
            if self.over():
                break
            self.step(follow, record)
            used += 1
        return used

    def snapshot(self):
        self.core._backup()
        return self.info

    def rewind(self, info0):
        self.core._restore()
        self.core.done = False
        self.done = False
        self.info = info0

    def outcome(self, name: str) -> dict:
        """Play name for one second, then all 16 two-move continuations. Exact simulation, no model."""
        x0 = self.x()
        info0 = self.snapshot()
        used = self.execute(name, HORIZON)
        falling = airborne(self.ram) and not self.done  # still in the air after the cap: a pit
        out = {"dead": self.done or falling, "flag": bool(self.info["flag_get"]), "dx": (self.x() - x0) // 16,
               "frames": used, "on_ground": not airborne(self.ram), "alive_paths": 0, "best_gain": None, "best_path": None}
        if not out["dead"] and not out["flag"]:
            fe = features(grid(self.ram)[0], play.speed(self.ram), airborne=airborne(self.ram))
            out["enemies_ahead"] = [e["tiles"] for e in fe["enemies"] if abs(e["up"]) <= 1][:2]
            out["wall"] = fe["wall_ahead"]
            out["gap"] = fe["gap_ahead"]
            for f in FOLLOW_UPS:
                for g in FOLLOW_UPS:
                    self.rewind(info0)
                    self.execute(name, HORIZON)  # single snapshot slot: replay the option each time
                    if self.over():
                        break
                    x1 = self.x()
                    self.execute(f, HORIZON)
                    if self.done or airborne(self.ram):
                        continue
                    if not self.info["flag_get"]:
                        self.execute(g, HORIZON)
                    if self.done or (airborne(self.ram) and not self.info["flag_get"]):
                        continue
                    out["alive_paths"] += 1
                    gain = (self.x() - x1) // 16 + (100 if self.info["flag_get"] else 0)
                    if out["best_gain"] is None or gain > out["best_gain"]:
                        out["best_gain"], out["best_path"] = gain, [f, g]
        self.rewind(info0)
        return out


def describe(o: dict) -> str:
    if o["flag"]:
        return "reaches the flag"
    if o["dead"]:
        return f"dies in {o['frames'] / 60:.1f} s"
    s = f"alive after {o['frames'] / 60:.1f} s, {o['dx']:+d} tiles, {'on the ground' if o['on_ground'] else 'falling'}"
    if o.get("enemies_ahead"):
        s += f", enemy {o['enemies_ahead'][0]} tiles ahead"
    if o.get("wall"):
        s += f", wall {o['wall']['height']} tall {o['wall']['tiles']} tiles ahead"
    if o.get("gap"):
        s += f", gap {o['gap']['width']} wide {o['gap']['tiles']} tiles ahead"
    n = len(FOLLOW_UPS) ** 2
    if out_dead_end(o):
        return s + f"; every one of {n} continuations dies (dead end)"
    s += f"; {o['alive_paths']} of {n} continuations survive"
    if o["best_gain"] is not None:
        gain = o["best_gain"]
        s += ", one reaches the flag" if gain >= 100 else f", best {gain:+d} more via '{o['best_path'][0]}' then '{o['best_path'][1]}'"
    return s


def out_dead_end(o: dict) -> bool:
    return not o["dead"] and not o["flag"] and o["alive_paths"] == 0


def score(o: dict) -> tuple:
    # Alive, not a dead end, then distance over three seconds; survivors only break ties.
    return (o["flag"], not o["dead"], not out_dead_end(o), o["dx"] + (o["best_gain"] or 0), o["dx"], o["alive_paths"])


def escape(sim: Sim) -> list[str] | None:
    """When no option makes progress, search 3 moves deep over a few basic moves and return the best
    sequence by distance gained while alive."""
    x0 = sim.x()
    best, best_seq = None, None
    for a in ESCAPE_MOVES:
        for b in ESCAPE_MOVES:
            for c in ESCAPE_MOVES:
                info0 = sim.snapshot()
                dead = False
                for name in (a, b, c):
                    sim.execute(name, HORIZON)
                    if sim.done or airborne(sim.ram):
                        dead = True
                        break
                gain = sim.x() - x0
                sim.rewind(info0)
                if not dead and (best is None or gain > best):
                    best, best_seq = gain, [a, b, c]
    return best_seq if best and best > 0 else None


def ask_jev(client: httpx.Client, outcomes: dict[str, dict], summary: str,
            url: str = play.JEV_URL, model: str = "jev-latest") -> tuple[str, dict, int, float]:
    criteria = {k: describe(o) for k, o in outcomes.items()}
    body = {
        "state": {"situation": summary, "outcomes": criteria},
        "model": model,
        "questions": {"action": {
            "type": "choice",
            "instructions": "You control Mario. Each option below says what actually happens if Mario does it for "
            "the next second and then keeps going, measured in the game itself. Pick the option that makes the "
            "most progress toward the flag (to the right) without dying. A dead end is an option after which "
            "every continuation dies; never pick one while an alternative survives.",
            "criteria": criteria,
        }},
    }
    key = play.os.environ.get("TYPESAFE_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key and url == play.JEV_URL else {}
    t0 = time.perf_counter()
    r = client.post(url, headers=headers, json=body)
    lat = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    a = d["answers"]["action"]
    return a["choice"], a["probabilities"], d.get("usage", {}).get("input_tokens", 0), lat


def run(bot: str, level: str, url: str = play.JEV_URL, model: str = "jev-latest", label: str = None) -> dict:
    play.warnings.simplefilter("ignore")
    sim = Sim(level)
    frames, log, lats, tokens = [sim.obs.copy()], [], [], 0
    client = httpx.Client(timeout=30)
    frame, best, last_best, last_gain = 0, 0, 0, 0
    while frame < play.MAX_FRAMES and not sim.over():
        while airborne(sim.ram) and not sim.done:
            sim.step(3, frames)
            frame += 1
        if sim.done:
            break
        outcomes = {name: sim.outcome(name) for name in OPTIONS}
        summary = features(grid(sim.ram)[0], play.speed(sim.ram), airborne=False)["summary"]
        if all(o["dead"] or out_dead_end(o) or o["dx"] + (o["best_gain"] or 0) <= 0 for o in outcomes.values()):
            seq = escape(sim)
            if seq:
                log.append({"frame": frame, "x": sim.x(), "choice": "escape:" + ",".join(seq), "p": 1.0,
                            "outcomes": {k: describe(o) for k, o in outcomes.items()}})
                for name in seq:
                    frame += sim.execute(name, HORIZON, frames)
                    if sim.over():
                        break
                best = max(best, sim.x())
                if best > last_best:
                    last_best, last_gain = best, frame
                continue
        if bot == "jev":
            name, probs, tok, lat = ask_jev(client, outcomes, summary, url, model)
            tokens += tok
            lats.append(lat)
        else:
            name = max(OPTIONS, key=lambda k: score(outcomes[k]))
            probs = {}
        log.append({"frame": frame, "x": sim.x(), "choice": name, "p": round(probs.get(name, 1.0), 2),
                    "outcomes": {k: describe(o) for k, o in outcomes.items()}})
        # Execute exactly what was simulated. If the option's value came only from its continuation,
        # play the continuation's first move too, so the branch that was scored is the branch ridden.
        frame += sim.execute(name, HORIZON, frames)
        chosen = outcomes[name]
        if not sim.over() and chosen["dx"] <= 0 and chosen["best_path"] and (chosen["best_gain"] or 0) > 0:
            frame += sim.execute(chosen["best_path"][0], HORIZON, frames)
            log[-1]["ride"] = chosen["best_path"][0]
        best = max(best, sim.x())
        if best > last_best:
            last_best, last_gain = best, frame
        if frame - last_gain > play.STALL_FRAMES * 2:
            break
    sim.env.close()
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = label or bot
    tag = f"{level}-branch-{name}"
    cost = tokens * USD_PER_TOKEN if url == play.JEV_URL else 0.0  # self-hosted servers are free to run
    result = {"level": level, "bot": f"branch-{name}", "stamp": stamp, "best_x": best, "flag": bool(sim.info["flag_get"]),
              "frames": frame, "api_calls": len(lats), "input_tokens": tokens, "cost_usd": round(cost, 5),
              "latency_p50": round(sorted(lats)[len(lats) // 2], 3) if lats else None, "gif": f"{tag}-{stamp}.gif"}
    play.imageio.mimsave(RUNS / result["gif"], frames[::2], duration=1 / 30, loop=0)
    with (RUNS / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    (RUNS / f"{tag}-{stamp}.log.jsonl").write_text("\n".join(json.dumps(l) for l in log) + "\n")
    return result


def session(bot: str, levels: list[str], attempts: int, **kw) -> list[dict]:
    """Play levels in order in one process. A level is retried up to `attempts` times; every attempt is logged."""
    results = []
    for level in levels:
        for i in range(1, attempts + 1):
            r = run(bot, level, **kw)
            r["attempt"] = i
            results.append(r)
            print(json.dumps(r), flush=True)
            if r["flag"]:
                break
        else:
            break  # a level that never cleared ends the session
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="jev", choices=["jev", "search"])
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--levels", help="comma-separated levels to play in order in one session, e.g. 1-1,2-1,3-1")
    ap.add_argument("--attempts", type=int, default=3, help="attempts per level in a session")
    ap.add_argument("--url", default=play.JEV_URL, help="any server speaking the System One contract")
    ap.add_argument("--model", default="jev-latest", help="model id sent in the request body")
    ap.add_argument("--label", help="name for the run in results.jsonl and the gif; defaults to --bot")
    a = ap.parse_args()
    play.load_env()
    kw = {"url": a.url, "model": a.model, "label": a.label}
    if a.levels:
        session(a.bot, a.levels.split(","), a.attempts, **kw)
    else:
        print(json.dumps(run(a.bot, a.level, **kw)))
