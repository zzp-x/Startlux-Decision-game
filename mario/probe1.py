"""依赖与 ROM 探测（一）：确认 nes-py / gym-super-mario-bros 装得上、ROM 在哪、make() 长什么样。

排查 Windows 上装不了 nes-py 8.2.1 时写的一次性探针，保留下来备查。
"""
import contextlib, io, os, traceback, warnings

print("=== 1. imports (as play.py does) ===")
try:
    # harness 是在"吞掉 stdout/stderr"的上下文里导入的，这里照做，免得 gym 的横幅刷屏
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    print("OK imports; SIMPLE_MOVEMENT has", len(SIMPLE_MOVEMENT), "actions")
except Exception:
    traceback.print_exc()

print("\n=== 2. package contents / bundled ROMs ===")
try:
    # ROM 是随包发布的，找出来确认不用另外下载
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
