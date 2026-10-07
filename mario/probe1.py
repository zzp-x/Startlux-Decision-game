import contextlib, io, os, traceback, warnings

print("=== 1. imports (as play.py does) ===")
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    print("OK imports; SIMPLE_MOVEMENT has", len(SIMPLE_MOVEMENT), "actions")
except Exception:
    traceback.print_exc()

print("\n=== 2. package contents / bundled ROMs ===")
try:
    import gym_super_mario_bros as g
    d = os.path.dirname(g.__file__)
    print("pkg dir:", d)
    for root, dirs, files in os.walk(d):
        if any(f.lower().endswith((".rom", ".nes")) for f in files):
            print("  ROMS in", root, "->", [f for f in files if f.lower().endswith((".rom", ".nes"))])
except Exception:
    traceback.print_exc()

print("\n=== 3. nes_py ROM search ===")
try:
    import nes_py, inspect
    from nes_py.nes_env import NESEnv
    src = inspect.getsource(NESEnv.__init__)
    print(src[:1500])
except Exception:
    traceback.print_exc()

print("\n=== 4. make() signature ===")
try:
    import inspect
    from gym_super_mario_bros import make
    print(inspect.signature(make))
    print(inspect.getsource(make)[:1200])
except Exception:
    traceback.print_exc()
