"""Live-speed play: the emulator never pauses for Jev.

    uv run python live.py --bot jev --level 1-1        # Jev, real latency
    uv run python live.py --bot rules --level 1-1      # deterministic twin with the same fixed delay

Each decision sends a JSON state (player, terrain, hazard, jump, timing, warnings) and asks three questions:
a Choice over controller actions, a Noul "should a jump start or be held now", and a Score for danger.
While the request is in flight the emulator advances by the measured latency holding the previous input,
so the answer lands in the world it will actually act on. Code combines the answers: a strong Noul turns a
non-jump Choice into a jump. Jumps are held only until the next decision, as on a real controller.
Writes runs/<level>-live-<bot>-<stamp>.gif, a log next to it, and a line in runs/results.jsonl.
"""

import argparse
import json
import time

import httpx

import play
from play import ACTIONS, JUMPS, RELEASE, RUNS, USD_PER_TOKEN, airborne, features, grid, nes, policy

MIN_HOLD = 2  # frames an answer is applied for at least; the rest of the cycle is the request latency
RULES_DELAY = 0.2  # seconds of pretend latency for the rules twin, near Jev's median
LIVE_ACTIONS = ["run right", "run and jump right", "jump right", "walk right", "jump in place", "stand", "walk left"]
LIVE_HELP = {
    "run right": "hold right and B; fastest forward speed, needed for long jumps",
    "run and jump right": "hold right, B and A; start a running jump, or keep it rising if already airborne",
    "jump right": "hold right and A; a jump at walking speed, or keep a jump rising",
    "walk right": "hold right; slow forward movement",
    "jump in place": "hold A only; jump straight up, or keep rising",
    "stand": "release everything; only when moving forward would be death and no jump helps",
    "walk left": "hold left; back off",
}
INSTRUCTIONS = {
    "goal": "Reach the flag to the right without dying. Progress is everything: when nothing within 6 tiles "
            "needs handling, run right. Standing still or walking left when the way is clear is the worst choice.",
    "timing": "The chosen action is held until the next answer arrives, about timing.expected_delay_frames "
              "frames later; the world keeps moving meanwhile. Judge from projected positions, not current ones. "
              "timing.tiles_per_decision_at_current_speed says how far Mario travels blind between decisions; "
              "enemies appear only about 9 tiles ahead, so at full speed there are two decisions to react.",
    "jumps": "A jump rises only while A is held: choose a jump action again on the next decision to go higher. "
             "Height at full hold is 4 tiles from rest and 5 at full speed; distance 3, 5 and 9 tiles at rest, "
             "walking and running speed. Start a jump over a wall about half the far reach before it.",
    "enemies": "Touching an enemy from the side is death; landing on a goomba or koopa kills it. A piranha "
               "plant on a pipe cannot be stomped. Enemies walk toward Mario about 2 tiles per second.",
    "warnings": "The warnings list states what will happen if nothing changes; act on it first.",
}


def state_json(ram, delay_frames: int, last: dict, stalled: int = 0, best_x: int = 0) -> dict:
    g, _, _ = grid(ram)
    v = play.speed(ram)
    fe = features(g, v, airborne=airborne(ram), visible=(256 - int(ram[0x03AD])) // 16)
    yv = int(ram[0x9F])
    yv = yv - 256 if yv > 127 else yv
    phase = "on ground" if not airborne(ram) else ("rising" if yv < 0 else "falling")
    # Projected contact with the nearest ground-level enemy ahead, in frames, given current speeds.
    ground = [e for e in fe["enemies"] if abs(e["up"]) <= 1]
    contact = None
    if ground:
        closing = max(0.3, v / 20.0) + 0.6  # Mario px/frame from the speed byte, plus a walking enemy
        contact = int(ground[0]["tiles"] * 16 / closing)
    warnings = list(fe["warnings"])
    reaction = delay_frames + MIN_HOLD + 10  # answer arrives, is held, and a jump needs ~10 frames to rise
    must_jump = contact is not None and contact <= reaction and not airborne(ram)
    if must_jump:
        warnings.insert(0, f"A {ground[0]['kind']} reaches Mario in about {contact} frames, before another decision "
                           f"can act (reaction takes {reaction}). A jump must start on this decision.")
    return {
        "player": {"speed": fe["speed"], "speed_byte": v, "on_ground": fe["on_ground"], "jump_phase": phase},
        "terrain": {"wall_ahead": fe["wall_ahead"], "gap_ahead": fe["gap_ahead"], "clear_behind": fe["clear_behind"],
                    "headroom": fe.get("headroom"), "visible_tiles_ahead": fe["visible_tiles_ahead"]},
        "hazard": {"enemies_ahead": fe["enemies"][:4], "enemies_behind": fe["enemies_behind"],
                   "enemies_near_landing_spot": fe["enemies_near_landing_spot"],
                   "frames_to_contact": contact, "jump_must_start_this_decision": must_jump},
        "jump": {"reach_now": fe["jump_reach_now"], "reach_at_full_speed": fe["jump_reach_at_full_speed"],
                 "hop_lands_tiles_ahead": fe["hop_lands_tiles_ahead"]},
        "timing": {"expected_delay_frames": delay_frames, "min_hold_frames": MIN_HOLD,
                   "tiles_per_decision_at_current_speed": round((delay_frames + MIN_HOLD) * max(0.3, v / 20.0) / 16, 1)},
        "recent": last,
        "episode": {"stalled_frames": stalled, "best_x": best_x},
        "warnings": warnings,
    }


def ask_jev(client: httpx.Client, state: dict) -> tuple[dict, int, float]:
    body = {
        "state": state,
        "model": "jev-latest",
        "questions": {
            "action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": LIVE_HELP},
            "jump_now": {"type": "noul", "instructions": "Should a jump start now, or be kept rising if Mario is "
                         "already in a jump? Yes if a wall, gap or enemy ahead needs it within the next second.",
                         "criteria": {"true": "press or keep holding A now", "false": "no jump needed now"}},
            "danger": {"type": "score", "instructions": "How dangerous is the next second?",
                       "criteria": ["clear", "something to handle soon", "immediate death risk"]},
        },
    }
    t0 = time.perf_counter()
    r = client.post(play.JEV_URL, headers={"Authorization": f"Bearer {play.os.environ['TYPESAFE_API_KEY']}"}, json=body)
    lat = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    return d["answers"], d["usage"]["input_tokens"], lat


def combine(answers: dict, speed_byte: int) -> tuple[str, str | None]:
    """Choice drives; a confident Noul turns a non-jump into a jump. Returns (action, override note)."""
    action = answers["action"]["choice"]
    jump_p = answers["jump_now"]["noul"]
    if jump_p >= 0.75 and ACTIONS[action] not in JUMPS:
        return ("run and jump right" if speed_byte >= 32 else "jump right"), f"noul jump_now={jump_p:.2f}"
    return action, None


def run(bot: str, level: str) -> dict:
    play.warnings.simplefilter("ignore")
    env = play.JoypadSpace(play.gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0", apply_api_compatibility=True), play.SIMPLE_MOVEMENT)
    ram = nes(env).ram
    obs, _ = env.reset()
    frames, log, lats, tokens = [obs.copy()], [], [], 0
    client = httpx.Client(timeout=30)
    info = {"x_pos": 40, "flag_get": False}
    frame, best, last_best, last_gain = 0, 0, 0, 0
    action, prev_name = 0, "stand"
    delay_frames = round(RULES_DELAY * 60)
    last = {"action": "stand", "held_frames": 0, "progress_tiles": 0}
    done = False

    def step(a: int) -> bool:
        nonlocal obs, info, frame, best, done
        obs, _, term, trunc, info = env.step(a)
        frame += 1
        best = max(best, int(info["x_pos"]))
        if frame % 2 == 0:
            frames.append(obs.copy())
        done = bool(term or trunc)
        return done

    while frame < play.MAX_FRAMES and not done and not info["flag_get"]:
        x0 = int(info["x_pos"])
        state = state_json(ram, delay_frames, last, frame - last_gain, best)
        if bot == "jev":
            answers, tok, lat = ask_jev(client, state)
            tokens += tok
            lats.append(lat)
            name, note = combine(answers, state["player"]["speed_byte"])
            probs = {k: round(p, 2) for k, p in answers["action"]["probabilities"].items()}
            extra = {"jump_now": round(answers["jump_now"]["noul"], 2), "danger": round(answers["danger"]["score"], 2)}
        else:
            lat = RULES_DELAY
            fe = features(grid(ram)[0], play.speed(ram), airborne=airborne(ram), visible=state["terrain"]["visible_tiles_ahead"])
            name = policy({**fe, "summary": ""})
            name = name if name in LIVE_ACTIONS else ("jump right" if "jump" in name or "hop" in name else "run right")
            if state["player"]["jump_phase"] == "rising" and ACTIONS[prev_name] in JUMPS:
                name = prev_name  # keep A held while rising, as the instructions tell Jev to
            note, probs, extra = None, {}, {}
        # The world moved while the answer was in flight: advance by the latency holding the previous input.
        delay_frames = max(1, round(lat * 60))
        for _ in range(delay_frames):
            if step(action) or info["flag_get"]:
                break
        if done or info["flag_get"]:
            break
        new = ACTIONS[name]
        # A new jump needs A released for a frame; a held jump (same action while airborne) must not release.
        if new in JUMPS and action in JUMPS and not airborne(ram):
            step(0)  # release A with no direction: a step at a pit edge would walk off
        action = new
        held = 0
        for _ in range(MIN_HOLD):
            held += 1
            if step(action) or info["flag_get"]:
                break
        last = {"action": name, "held_frames": held + delay_frames, "progress_tiles": (int(info["x_pos"]) - x0) // 16}
        log.append({"frame": frame, "x": int(info["x_pos"]), "choice": name, "override": note, "latency_s": round(lat, 3),
                    "probs": probs, **extra, "warnings": state["warnings"],
                    "contact": state["hazard"]["frames_to_contact"], "must_jump": state["hazard"]["jump_must_start_this_decision"],
                    "enemies": state["hazard"]["enemies_ahead"][:2]})
        prev_name = name
        if best > last_best:
            last_best, last_gain = best, frame
        if frame - last_gain > play.STALL_FRAMES:
            break
    env.close()
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{level}-live-{bot}"
    result = {"level": level, "bot": f"live-{bot}", "stamp": stamp, "best_x": best, "flag": bool(info["flag_get"]),
              "frames": frame, "api_calls": len(lats), "input_tokens": tokens, "cost_usd": round(tokens * USD_PER_TOKEN, 5),
              "latency_p50": round(sorted(lats)[len(lats) // 2], 3) if lats else None, "gif": f"{tag}-{stamp}.gif"}
    play.imageio.mimsave(RUNS / result["gif"], frames, duration=1 / 30, loop=0)
    with (RUNS / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    (RUNS / f"{tag}-{stamp}.log.jsonl").write_text("\n".join(json.dumps(l) for l in log) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="jev", choices=["jev", "rules"])
    ap.add_argument("--level", default="1-1")
    a = ap.parse_args()
    play.load_env()
    print(json.dumps(run(a.bot, a.level)))
