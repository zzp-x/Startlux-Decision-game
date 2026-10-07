"""拉起一个完全脱离当前 shell 的后台进程。

为什么要自己写：本机这个 shell 里 PowerShell 的 Start-Process 是坏的
（环境里同时存在 Path 和 PATH 两个键，会报重复键错），所以改用 subprocess 直接起。
"""
import os
import subprocess
import sys

exe = sys.argv[1]
cwd = sys.argv[2]
log = sys.argv[3]
args = sys.argv[4:]

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# 修掉 PATH / Path 同时存在导致子进程找不到 DLL 的问题：统一成一个 PATH，再把 Path 删掉
env = dict(os.environ)
env["PATH"] = env.get("Path", env.get("PATH", ""))
env.pop("Path", None)
env["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 允许 OpenMP 多份 runtime 共存，避免误报冲突退出

f = open(log, "w", encoding="utf-8", buffering=1)   # 行缓冲，日志可以实时看
p = subprocess.Popen(
    [exe] + args,
    cwd=cwd,
    stdout=f,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    env=env,
)
print("pid", p.pid)
