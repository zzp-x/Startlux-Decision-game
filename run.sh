#!/usr/bin/env bash
# StartLux-Decision one-click launcher, for bash / git-bash.
#   ./run.sh                    boot the stack, then run the demo call
#   ./run.sh --mario            ... and play level 1-1 with jev-mario
#   ./run.sh --status | --stop | --restart | --selftest | --bench
# Override the interpreter with STARTLUX_PY=... if needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${STARTLUX_PY:-C:/Users/Libai/.workbuddy/binaries/python/versions/3.13.12/python.exe}"
if [ ! -f "$PY" ]; then
  echo "[XX] Python interpreter not found: $PY" >&2
  echo "     export STARTLUX_PY=/path/to/python.exe and retry." >&2
  exit 1
fi
exec "$PY" -X utf8 "$HERE/run_local.py" "$@"
