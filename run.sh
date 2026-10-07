#!/usr/bin/env bash
# StartLux-Decision 一键启动器，给 bash / git-bash 用。
#   ./run.sh                    起服务栈，然后跑一次示例调用
#   ./run.sh --mario            再顺便用 jev-mario 玩一遍 1-1
#   ./run.sh --status | --stop | --restart | --selftest | --bench
# 需要换解释器时用 STARTLUX_PY=... 覆盖。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${STARTLUX_PY:-C:/Users/Libai/.workbuddy/binaries/python/versions/3.13.12/python.exe}"
if [ ! -f "$PY" ]; then
  echo "[XX] Python interpreter not found: $PY" >&2
  echo "     export STARTLUX_PY=/path/to/python.exe and retry." >&2
  exit 1
fi
exec "$PY" -X utf8 "$HERE/run_local.py" "$@"
