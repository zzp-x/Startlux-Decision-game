"""依赖与模拟器探测（二）：动作集、backup/restore 是否可用、能否建环境、备份还原是否精确。

第 E 段最关键——mario_par.py 的并行重放法依赖"备份/还原后位置分毫不差"，这里就是验证它。
"""
import contextlib, io, inspect, os, traceback, warnings

warnings.filterwarnings("ignore")

print("=== A. JoypadSpace / actions ===")
try:
    from gym_super_mario_bros import SuperMarioBrosEnv
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    print("JoypadSpace OK; SIMPLE_MOVEMENT =", SIMPLE_MOVEMENT)
    print("SuperMarioBrosEnv.__init__:", inspect.signature(SuperMarioBrosEnv.__init__))
except Exception:
    traceback.print_exc()

print("\n=== B. NESEnv backup/restore ===")
try:
    from nes_py.nes_env import NESEnv
    print("has _backup:", hasattr(NESEnv, "_backup"), "| has _restore:", hasattr(NESEnv, "_restore"))
    print("has ram:", hasattr(NESEnv, "ram"), "| has close:", hasattr(NESEnv, "close"))
except Exception:
    traceback.print_exc()

print("\n=== C. build env with apply_api_compatibility=True (play.py's call) ===")
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import gym_super_mario_bros
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    print("make() OK ->", type(env))
except Exception as e:
    print("FAILED:", repr(e))

print("\n=== D. build env WITHOUT that kwarg ===")
env = None
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import gym_super_mario_bros
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    print("make() OK ->", type(env))
except Exception as e:
    print("FAILED:", repr(e))
    traceback.print_exc()

if env is not None:
    print("\n=== E. wrap + reset + step ===")
    try:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # 顺着包装链走到真正持有 RAM 的那一层（nes_py 的 NESEnv）
        e = env
        while not hasattr(e, "ram"):
            e = e.env
        print("core env:", type(e).__name__, "ram type:", type(e.ram), "ram len:", len(e.ram))
        obs, info = env.reset()
        print("reset OK; obs", obs.shape, obs.dtype, "info keys:", sorted(info.keys()))
        for i in range(5):
            obs, rew, term, trunc, info = env.step(3)  # run right
        print("step OK; x_pos =", info.get("x_pos"), "flag_get =", info.get("flag_get"))
        print("ram[0x6D],ram[0x86] =", int(e.ram[0x6D]), int(e.ram[0x86]))
        # x 坐标是两个字节拼出来的：0x6D 高位、0x86 低位
        e._backup(); 
        x0 = int(e.ram[0x6D])*256+int(e.ram[0x86])
        for i in range(60):
            env.step(4)
        x1 = int(e.ram[0x6D])*256+int(e.ram[0x86])
        e._restore(); e.done = False       # _restore 之后必须手动把 done 清掉
        x2 = int(e.ram[0x6D])*256+int(e.ram[0x86])
        print(f"backup/restore: x0={x0} after60={x1} restored={x2} -> {'EXACT' if x0==x2 else 'MISMATCH'}")
        env.close()
        print("close OK")
    except Exception:
        traceback.print_exc()
