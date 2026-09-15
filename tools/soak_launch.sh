#!/usr/bin/env bash
# Launch a Tumble headless soak DETACHED from this shell, and return at once.
#
# This is the documented "walk away" path: the soak survives the death of the
# terminal, the SSH connection, or the worker session that started it, and it
# bounds itself, so nobody ever has to sit and wait on one. Node 18 lost two
# worker sessions to exactly that mistake.
#
#   tools/soak_launch.sh soak_runs/long --max-seconds 3600 --destroy-every 120
#   tools/soak_driver.py --status soak_runs/long     # poll it from anywhere
#   tools/soak_driver.py --stop   soak_runs/long     # ask it to stop cleanly
#
# The driver's own --detach flag does the same thing (setsid + PID file + log)
# without needing this script; this exists so the plain-nohup form is
# written down somewhere and stays tested.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
  echo "usage: tools/soak_launch.sh <results-dir> [driver flags...]" >&2
  exit 2
fi

DIR="$1"; shift
mkdir -p "$DIR"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

# nohup: immune to SIGHUP when the parent shell goes away.
# setsid : its own session, so it is not in the terminal's process group.
# </dev/null and a log file: nothing can block on a pipe nobody reads.
setsid nohup "$PY" tools/soak_driver.py --results-dir "$DIR" "$@" \
  </dev/null >>"$DIR/soak.log" 2>&1 &

CHILD=$!
# The driver writes its own PID file; this is the fallback until it does.
echo "$CHILD" > "$DIR/soak.pid"
echo "[soak_launch] detached soak started"
echo "  pid     : $CHILD"
echo "  results : $DIR"
echo "  log     : $DIR/soak.log"
echo "  status  : $PY tools/soak_driver.py --status $DIR"
echo "  stop    : $PY tools/soak_driver.py --stop   $DIR"
