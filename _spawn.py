"""Spawn a detached process (Start-Process is broken in this shell: duplicate Path/PATH keys)."""
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

env = dict(os.environ)
env["PATH"] = env.get("Path", env.get("PATH", ""))
env.pop("Path", None)
env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

f = open(log, "w", encoding="utf-8", buffering=1)
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
