"""Jev plays Super Mario Bros from a text description of the emulator RAM.

    uv run python play.py --bot jev --level 1-1                # Jev picks every action
    uv run python play.py --bot "run and jump right"           # scripted baseline, no API calls
    uv run python play.py --dump --level 2-1                   # print what Jev would see, no API calls
    uv run python play.py --inspect runs/<log>.jsonl [-n 3]    # last decisions: state, grid, probabilities
    uv run python play.py --bot "replay:runs/<log>.jsonl@18:jump right,run right"
                                                               # replay the first 18 logged choices, then hold the tail

Each run writes runs/<level>-<bot>-<stamp>.gif, a per-decision log next to it, and one line in runs/results.jsonl.

To improve play, edit the three blocks marked EDIT HERE: the actions Jev can choose, the rules it is given,
and features(), which turns the tile grid into the summary. Reproduce a death with --inspect, test a fix with
replay (no API calls), then run Jev again.
"""

import argparse
import json
import os
import time
from pathlib import Path

import contextlib
import io
import warnings

import httpx
import imageio.v2 as imageio

warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    # gym 0.26 prints a deprecation banner at import; nes-py needs this gym version.
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

HOLD = 6  # frames per decision on the ground; 60 fps
MAX_FRAMES = 6000
STALL_FRAMES = 360  # give up after 6 game-seconds without gaining distance
HOP_FRAMES = 8  # A held this long gives a ~2.4 tile hop
FULL_JUMP_FRAMES = 64  # a full-speed jump lasts about 50 frames
RUNS = Path(__file__).resolve().parent / "runs"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
USD_PER_TOKEN = 0.042 / 1e6

# ----------------------------------------------------------------------------- EDIT HERE: actions
# Joypad index in SIMPLE_MOVEMENT, or BACK_OFF for the harness-executed retreat.
BACK_OFF = -2
ACTIONS = {
    "stand": 0,
    "walk right": 1,
    "jump right": 2,
    "run right": 3,
    "run and jump right": 4,
    "jump in place": 5,
    "walk left": 6,
    "hop right": 2,  # same button as "jump right", released after HOP_FRAMES
    "back off for a run-up": BACK_OFF,
}
ACTION_HELP = {
    "stand": "wait in place; use when an enemy on a wall or pipe ahead has to move away first",
    "walk right": "move right slowly",
    "jump right": "full jump while moving right at walking speed; about 4 tiles high, 3 to 5 tiles far",
    "run right": "move right fast; does not jump; builds speed for a longer jump",
    "run and jump right": "full jump at running speed; up to 5 tiles high and 9 tiles far when already at full speed",
    "jump in place": "full jump straight up",
    "walk left": "back off to the left",
    "hop right": "short low jump to the right, about 2 tiles high; lands 1 to 4 tiles ahead depending on speed; "
    "use to land on top of an enemy 2 tiles ahead",
    "back off for a run-up": "walk left until the wall or gap ahead is 6 tiles away, so that 'run right' can "
    "build full speed before jumping; do not use with an enemy behind Mario",
}
JUMPS = {2, 4, 5}
RELEASE = {2: 1, 4: 3, 5: 0}  # same action without A

# ----------------------------------------------------------------------------- EDIT HERE: rules
RULES = (
    "You control Mario in Super Mario Bros. Pick the joypad action that moves right as far as possible "
    "without dying. A jump action is held until Mario lands, so one decision is one full jump. "
    "Mario must jump over walls, gaps and enemies. A jump clears a wall only if its reach in tiles high "
    "is at least the wall height, and a gap only if its reach in tiles far exceeds the gap width plus 1. "
    "Reach grows with speed: 'run right' reaches full speed after about 3 decisions on clear ground. "
    "When the current reach is not enough and the obstacle is closer than 6 tiles, choose "
    "'back off for a run-up', then 'run right' until full speed, then 'run and jump right' 2 to 3 tiles out. "
    "A jump arc peaks halfway, so start a jump over a wall when the wall is about half the far reach "
    "ahead (4 tiles at full speed, 2 at walking speed); jumping from closer hits the wall and drops. "
    "Start a jump over a gap 1 tile before its edge. Against an enemy ahead, jump when it is 2 to 3 tiles "
    "away; a hop is enough to clear or stomp one enemy and lands sooner, so prefer it when nothing further needs a full jump. "
    "Enemies appear at the right edge of the screen as Mario advances, so a jump that lands past what is visible "
    "lands blind. An enemy 1 or 2 tiles behind Mario will hit him within a second: jump immediately. "
    "The 'summary' field describes what is ahead; the grid is the same information drawn out. "
)
GRID_LEGEND = (
    "Text grid, 13 rows x 20 columns, each cell one 16px tile. Row 7 is Mario's row. "
    "M = Mario (column 5). # = solid ground, brick, block or pipe. . = empty air. Letters are enemies: "
    "G goomba, K koopa, S koopa shell, F flying koopa, P piranha plant, B buzzy beetle, H hammer brother. "
    "Mario walks right (toward higher columns). Falling into a column with no # below Mario is death. "
    "Touching an enemy from the side is death; landing on a goomba or koopa from above kills it. "
    "A stomped koopa leaves a shell that kills on touch and slides when kicked. A piranha plant cannot be "
    "stomped. A flying koopa bounces along the ground."
)

# Measured in this emulator: a full jump (A held until landing) at a given horizontal speed.
# (min speed byte, tiles high, tiles far). Height comes from the hold; distance from speed.
JUMP_TABLE = [(30, 5, 9), (15, 4, 5), (0, 4, 3)]
HOP_TABLE = [(30, 5), (15, 2), (0, 1)]  # (min speed byte, tiles far) for a hop; measured at 40, 28, 0


def jump_reach(v: int) -> tuple[int, int]:
    for min_v, high, far in JUMP_TABLE:
        if abs(v) >= min_v:
            return high, far
    return JUMP_TABLE[-1][1:]


def speed_word(v: int) -> str:
    # SMB horizontal speed byte 0x57, signed. Walking tops out near 28, running near 48.
    if v >= 32:
        return "running at full speed"
    if v >= 8:
        return "moving right slowly"
    if v <= -8:
        return "moving left"
    return "standing still"


# ----------------------------------------------------------------------------- EDIT HERE: state description
def features(g: str, v: int = 0, airborne: bool | None = None, visible: int = 15) -> dict:
    """Turn the grid into fields and a summary sentence. Mario is row 6, column 4; right is +column."""
    rows = g.splitlines()
    look = range(1, 9)
    # The grid is Mario-relative, so mid-jump his own row is air. Anchor on the ground under him.
    ground = next((r for r in range(7, 13) if rows[r][4] == "#"), 7)
    feet = ground - 1
    on_ground = (not airborne) if airborne is not None else ground == 7
    f = {"on_ground": on_ground, "wall_ahead": None, "gap_ahead": None, "enemy_ahead": None}
    for dx in look:
        if rows[feet][4 + dx] == "#":
            height = sum(1 for r in range(feet, -1, -1) if rows[r][4 + dx] == "#")
            f["wall_ahead"] = {"tiles": dx, "height": height}
            break

    def bottomless(col: int) -> bool:
        # No solid tile from the ground row to the bottom of the grid; a step down is not a gap.
        return all(rows[r][col] == "." for r in range(ground, 13))

    for dx in look:
        if bottomless(4 + dx):
            width = 0
            while 4 + dx + width < 20 and bottomless(4 + dx + width):
                width += 1
            f["gap_ahead"] = {"tiles": dx, "width": width}
            break
    # Enemies anywhere ahead, with height above Mario's feet (0 = same level).
    seen = []
    for dx in range(1, 16):
        for r in range(13):
            if rows[r][4 + dx] in ENEMY_LETTERS:
                seen.append({"tiles": dx, "up": feet - r, "kind": ENEMY_NAME[rows[r][4 + dx]]})
    f["enemies"] = seen
    enemies = [e["tiles"] for e in seen if -1 <= e["up"] <= 1]
    f["enemies_ahead"] = enemies
    f["enemy_ahead"] = {"tiles": enemies[0]} if enemies else None
    f["enemies_behind"] = [dx for dx in range(1, 5)
                           if any(rows[r][4 - dx] in ENEMY_LETTERS for r in (feet - 1, feet, ground))]
    behind = 0
    while behind < 4 and rows[feet][3 - behind] == "." and rows[ground][3 - behind] == "#":
        behind += 1
    f["clear_behind"] = behind

    parts, warns = [], []
    w, gp = f["wall_ahead"], f["gap_ahead"]
    parts.append(f"A solid wall {w['height']} tiles tall is {w['tiles']} tile(s) ahead." if w else "No wall ahead.")
    parts.append(f"There are {behind} tiles of clear ground behind Mario for a run-up.")
    parts.append(f"A gap {gp['width']} tiles wide is {gp['tiles']} tile(s) ahead." if gp else "Solid ground ahead.")
    if seen:
        def where(en: dict) -> str:
            if en["up"] > 1:
                return f"a {en['kind']} {en['tiles']} tiles ahead and {en['up']} tiles up (on top of something)"
            if en["up"] < -1:
                return f"a {en['kind']} {en['tiles']} tiles ahead and {-en['up']} tiles below"
            return f"a {en['kind']} {en['tiles']} tiles ahead at ground level"
        parts.append("Enemies: " + "; ".join(where(en) for en in seen[:4]) + ".")
    else:
        parts.append("No enemy ahead.")
    if f["enemies_behind"]:
        parts.append(f"An enemy is {f['enemies_behind'][0]} tile(s) behind Mario; walking left into it is death.")
    f["speed"] = speed_word(v)
    high, far = jump_reach(v)
    hop_far = HOP_TABLE[0][1] if abs(v) >= HOP_TABLE[0][0] else (HOP_TABLE[1][1] if abs(v) >= HOP_TABLE[1][0] else HOP_TABLE[2][1])
    f["hop_lands_tiles_ahead"] = hop_far
    f["visible_tiles_ahead"] = visible
    f["jump_reach_now"] = {"tiles_high": high, "tiles_far": far}
    f["jump_reach_at_full_speed"] = {"tiles_high": JUMP_TABLE[0][1], "tiles_far": JUMP_TABLE[0][2]}
    parts.append(("Mario is on the ground, " if f["on_ground"] else "Mario is in the air, ") + f["speed"] + ".")
    parts.append(f"A jump from this speed clears {high} tiles high and {far} tiles far; "
                 f"at full running speed it clears {JUMP_TABLE[0][1]} high and {JUMP_TABLE[0][2]} far.")
    if w:
        # The arc peaks halfway, so a wall must be about half the far reach away when the jump starts.
        if high >= w["height"]:  # verified by replay: a 4-tile standing jump lands on a 4-tall ledge
            parts.append(f"Start the jump when the wall is about {max(1, far // 2)} tiles ahead.")
        else:
            warns.append("A jump from this speed is not high enough for the wall ahead; more speed is needed.")
    # Headroom: blocks above the arc cut a jump short.
    headroom = 13
    for dx in range(0, min(far, 15) + 1):
        for r in range(feet - 1, -1, -1):
            if rows[r][4 + dx] == "#":
                headroom = min(headroom, feet - r - 1)
                break
    if headroom < high:
        eff_far = max(1, round(far * headroom / high))
        f["headroom"] = headroom
        parts.append(f"Blocks overhead {headroom + 1} tiles up cap the jump: it would land about {eff_far} tiles ahead instead of {far}.")
    else:
        eff_far = far
    # Landing zone: enemies walk toward Mario about 2 tiles during a one-second jump.
    land = [en for en in seen if abs(en["up"]) <= 1 and eff_far - 4 <= en["tiles"] <= eff_far + 3]
    f["enemies_near_landing_spot"] = [en["tiles"] for en in land]
    parts.append(f"A full jump right now would land about {eff_far} tiles ahead; a hop about {hop_far}. "
                 f"The screen shows {visible} tiles ahead; enemies beyond that are unknown until Mario moves closer.")
    if land:
        warns.append(f"Do not take a full jump now: it lands about {eff_far} tiles ahead, where an enemy will be by then.")
    if eff_far >= visible:
        warns.append("A full jump now lands past what is visible, on unknown ground.")
    on_wall = [en for en in seen if w and en["tiles"] in (w["tiles"], w["tiles"] + 1) and en["up"] >= 1]
    if on_wall:
        clear_over = JUMP_TABLE[0][1] >= w["height"] + 2
        warns.append("An enemy is on top of the wall ahead: jumping onto it is death. "
                     + ("A full-speed jump started 4 tiles before the wall clears the wall and the enemy together."
                        if clear_over else "Back off, wait for it to move, then jump when the top is clear."))
    if enemies and enemies[0] <= 1:
        warns.append("An enemy is right in front of Mario: moving toward it is death. Jump in place if standing still; "
                     "momentum carries into a jump, so at speed a full jump forward is the only way over it.")
    if f["enemies_behind"] and f["enemies_behind"][0] <= 2:
        warns.append("The enemy behind reaches Mario in about a second: jump over it or away from it, but not into another enemy.")
    f["warnings"] = warns
    f["summary"] = (("WARNINGS: " + " ".join(warns) + " ") if warns else "") + " ".join(parts)
    return f


# ----------------------------------------------------------------------------- scripted twin
def policy(f: dict) -> str:
    """RULES as a deterministic if-chain over the same fields Jev gets. The 'rules' bot; no API calls."""
    v_full = f["speed"] == "running at full speed"
    high, far = f["jump_reach_now"]["tiles_high"], f["jump_reach_now"]["tiles_far"]
    w, gp = f["wall_ahead"], f["gap_ahead"]
    ahead = f["enemies_ahead"]
    landing_bad = bool(f["enemies_near_landing_spot"])
    forward_jump = "run and jump right" if v_full else "jump right"
    if ahead and ahead[0] <= 1:
        return "jump in place" if f["speed"] == "standing still" else forward_jump
    if f["enemies_behind"] and f["enemies_behind"][0] <= 2:
        return "jump in place" if landing_bad else forward_jump
    on_wall = [e for e in f["enemies"] if w and e["tiles"] in (w["tiles"], w["tiles"] + 1) and e["up"] >= 1]
    if on_wall:
        if JUMP_TABLE[0][1] >= w["height"] + 2:
            if v_full and 3 <= w["tiles"] <= 5:
                return "run and jump right"
            return "run right" if w["tiles"] >= 6 else "back off for a run-up"
        return "stand"
    if w and w["tiles"] <= 8:
        start = max(1, far // 2)
        if high < w["height"]:
            return "run right" if w["tiles"] >= 6 else "back off for a run-up"
        if start - 1 <= w["tiles"] <= start + 1:
            return "jump in place" if landing_bad else forward_jump
        if w["tiles"] > start + 1:
            return "run right"
        return "jump right"
    if gp and gp["tiles"] <= 8:
        if far > gp["width"] + 1:
            return forward_jump if gp["tiles"] <= 1 else "run right"
        return "run right" if gp["tiles"] >= 6 else "back off for a run-up"
    if ahead and 2 <= ahead[0] <= 3:
        blind = f["jump_reach_now"]["tiles_far"] >= f["visible_tiles_ahead"]
        return "hop right" if (landing_bad or blind) else forward_jump
    return "run right"


# ----------------------------------------------------------------------------- emulator and RAM
def load_env() -> None:
    p = Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def nes(env):
    """Walk the wrapper chain to the nes_py env that owns the RAM."""
    e = env
    while not hasattr(e, "ram"):
        e = e.env
    return e


def tile(ram, x: int, y: int) -> int:
    # SMB keeps two 16x13 tile pages at 0x500, 208 bytes each; y offset 32 is the HUD.
    page = (x // 256) % 2
    sx, sy = (x % 256) // 16, (y - 32) // 16
    if sy < 0 or sy > 12:
        return 0
    return int(ram[0x500 + page * 208 + sy * 16 + sx])


# Enemy type byte 0x16+slot. Verified in-game: 6 goomba, 0 green koopa, 13 piranha plant, 14 paratroopa.
# The rest follow the standard SMB RAM map and are UNVERIFIED here.
ENEMY_LETTER = {0: "K", 1: "K", 2: "B", 3: "K", 4: "K", 5: "H", 6: "G", 13: "P", 14: "F", 15: "F", 16: "F", 9: "F"}
ENEMY_NAME = {"G": "goomba", "K": "koopa", "P": "piranha plant", "F": "flying koopa", "B": "buzzy beetle",
              "H": "hammer brother", "S": "koopa shell", "E": "enemy"}
ENEMY_LETTERS = set(ENEMY_NAME)


def grid(ram) -> tuple[str, int, int]:
    """13 rows x 20 columns around Mario: 6 rows above, 4 columns behind, 15 ahead. Letters mark enemy types."""
    mx = int(ram[0x6D]) * 256 + int(ram[0x86])
    my = int(ram[0x03B8]) + 16
    enemies = []
    for i in range(5):
        if ram[0x0F + i]:
            letter = ENEMY_LETTER.get(int(ram[0x16 + i]), "E")
            if letter == "K" and int(ram[0x1E + i]) in (2, 3):  # UNVERIFIED: stomped koopa state
                letter = "S"
            # +8 puts a ground enemy in Mario's row; verified against the first goomba.
            enemies.append((int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]), int(ram[0xCF + i]) + 8, letter))
    rows = []
    for dy in range(-6, 7):
        row = []
        for dx in range(-4, 16):
            x, y = mx + dx * 16, my + dy * 16
            ch = "#" if tile(ram, x, y) else "."
            for ex, ey, letter in enemies:
                if abs(ex - x) <= 8 and abs(ey - y) <= 8:
                    ch = letter
            if dx == 0 and dy == 0:
                ch = "M"
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows), mx, my


def speed(ram) -> int:
    v = int(ram[0x57])
    return v - 256 if v > 127 else v


def airborne(ram) -> bool:
    return bool(ram[0x1D])  # 0 on the ground, 1 for the whole jump; verified against the y trace


# ----------------------------------------------------------------------------- Jev
def ask_jev(client: httpx.Client, state: dict) -> tuple[str, dict, int, float]:
    body = {
        "state": state,
        "model": "jev-latest",
        "questions": {"action": {"type": "choice", "instructions": RULES + GRID_LEGEND, "criteria": ACTION_HELP}},
    }
    t0 = time.perf_counter()
    r = client.post(JEV_URL, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}, json=body)
    lat = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    a = d["answers"]["action"]
    return a["choice"], a["probabilities"], d["usage"]["input_tokens"], lat


# ----------------------------------------------------------------------------- run loop
class Replay:
    """replay:<log.jsonl>[@n]:<action,...> — replay the first n logged choices, then cycle through the tail."""

    def __init__(self, spec: str):
        _, path, tail = spec.split(":", 2)
        n = None
        if "@" in path:
            path, n = path.rsplit("@", 1)
        self.script = [json.loads(l)["choice"] for l in Path(path).read_text().splitlines()][: int(n) if n else None]
        self.tail = tail.split(",")
        self.i = 0

    def next(self) -> str:
        if self.script:
            return self.script.pop(0)
        name = self.tail[min(self.i, len(self.tail) - 1)]
        self.i += 1
        return name


def run(bot: str, level: str = "1-1", dump: bool = False) -> dict:
    warnings.simplefilter("ignore")  # gym's env checker re-enables a numpy deprecation warning
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0", apply_api_compatibility=True), SIMPLE_MOVEMENT)
    ram = nes(env).ram
    obs, _ = env.reset()
    # nes_py reuses one screen buffer, so every stored frame must be a copy.
    frames, tokens, lats, log = [obs.copy()], 0, [], []
    action, prev, hold_cap = 0, 0, FULL_JUMP_FRAMES
    frame, best, last_best, last_gain = 0, 0, 0, 0
    info = {"x_pos": 0, "flag_get": False}
    term = trunc = False
    deciding = bot in ("jev", "rules") or bot.startswith("replay:") or dump
    replay = Replay(bot) if bot.startswith("replay:") else None
    client = httpx.Client(timeout=30)

    def step(a: int) -> bool:
        """Advance one frame; record every other frame; return True when the episode is over."""
        nonlocal obs, term, trunc, info, frame, best
        obs, _, term, trunc, info = env.step(a)
        frame += 1
        best = max(best, int(info["x_pos"]))
        if frame % 2 == 0:
            frames.append(obs.copy())
        return bool(term or trunc)

    while frame < MAX_FRAMES:
        if frame % HOLD == 0 and deciding:
            # Decide only on the ground: keep the current direction (without A) while airborne.
            fall = 0
            while airborne(ram) and fall < 120 and not step(RELEASE.get(action, action)):
                fall += 1
            if term or trunc:
                break
            frame += (-frame) % HOLD
        if frame % HOLD == 0:
            g, _, _ = grid(ram)
            feats = features(g, speed(ram), airborne=airborne(ram), visible=(256 - int(ram[0x03AD])) // 16)
            state = {"summary": feats.pop("summary"), **feats, "grid": g, "action_before": action}
            if dump:
                print(f"frame {frame} x={info['x_pos']}\n{state['summary']}\n{g}\n")
                name = "run and jump right" if (frame // 12) % 2 == 0 else "run right"
            elif bot == "jev":
                name, probs, tok, lat = ask_jev(client, state)
                tokens += tok
                lats.append(lat)
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "p": round(probs[name], 2),
                            "probs": {k: round(p, 2) for k, p in probs.items()}, "summary": state["summary"], "grid": g})
            elif replay:
                name = replay.next()
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "summary": state["summary"], "grid": g})
            elif bot == "rules":
                name = policy({**feats, "summary": state["summary"]})
                log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "summary": state["summary"], "grid": g})
            elif bot == "alternate":
                name = "run and jump right" if (frame // 12) % 2 == 0 else "run right"
            else:
                name = bot
            action = ACTIONS[name]
            hold_cap = HOP_FRAMES if name == "hop right" else FULL_JUMP_FRAMES

            if action == BACK_OFF:
                # Atomic: walk left until the obstacle ahead is 6 tiles away, at most 60 frames.
                for _ in range(60):
                    if step(6):
                        break
                    fe = features(grid(ram)[0], 0)
                    if (fe["wall_ahead"] or fe["gap_ahead"] or {}).get("tiles", 99) >= 6:
                        break
                if term or trunc:
                    break
                frame += (-frame) % HOLD
                action = 0
            # The NES only jumps on an A press, not a hold: release A for one frame between two jumps.
            if action in JUMPS and prev in JUMPS and step(RELEASE[action]):
                break
            prev = action
            if action in JUMPS and bot != "alternate":
                # One decision is one jump: hold A until Mario lands, or hold_cap frames for a hop.
                for i in range(FULL_JUMP_FRAMES):
                    if step(action) or i >= hold_cap or (i > 4 and not airborne(ram)):
                        break
                if term or trunc:
                    break
                frame += (-frame) % HOLD
                action = RELEASE[action]
        if step(action):
            break
        if best > last_best:
            last_best, last_gain = best, frame
        if info["flag_get"] or frame - last_gain > STALL_FRAMES:
            break
    env.close()

    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{level}-{'replay' if replay else bot.replace(' ', '-')}"
    result = {
        "level": level, "bot": bot, "stamp": stamp, "best_x": best, "flag": bool(info["flag_get"]),
        "frames": frame, "api_calls": len(lats), "input_tokens": tokens,
        "cost_usd": round(tokens * USD_PER_TOKEN, 5),
        "latency_p50": round(sorted(lats)[len(lats) // 2], 3) if lats else None,
        "gif": None if dump else f"{tag}-{stamp}.gif",
    }
    if not dump:
        imageio.mimsave(RUNS / result["gif"], frames, duration=1 / 30, loop=0)
        with (RUNS / "results.jsonl").open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        if log:
            (RUNS / f"{tag}-{stamp}.log.jsonl").write_text("\n".join(json.dumps(l) for l in log) + "\n")
    return result


def inspect(path: str, n: int) -> None:
    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    print(f"{len(lines)} decisions:", [(l["x"], l["choice"]) for l in lines])
    for l in lines[-n:]:
        top = sorted(l.get("probs", {}).items(), key=lambda kv: -kv[1])[:4]
        print(f"\nframe {l['frame']} x={l['x']} -> {l['choice']} {l.get('p', '')}  {top}")
        print(l["summary"])
        print(l["grid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="jev", help="jev | rules | alternate | <action name> | replay:<log.jsonl>[@n]:<action,...>")
    ap.add_argument("--level", default="1-1", help="world-stage, e.g. 2-1")
    ap.add_argument("--dump", action="store_true", help="print the state Jev would see, no API calls")
    ap.add_argument("--inspect", metavar="LOG", help="print the last decisions of a run log and exit")
    ap.add_argument("-n", type=int, default=3, help="decisions to show with --inspect")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.inspect, a.n)
    else:
        load_env()
        print(json.dumps(run(a.bot, a.level, a.dump)))
