#!/usr/bin/env bash
# Tumble - one command from a fresh clone: creates the venv if needed,
# installs deps, launches the game.
#
#   ./run.sh              windowed playtest
#   ./run.sh --headless   no-window smoke run (servers, CI)
#   ./run.sh --test       run the pytest suite headlessly
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"
STAMP="$VENV/.deps-ok"

if ! command -v python3 >/dev/null 2>&1; then
  echo "run.sh: python3 not found - install Python 3.8+ and retry" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[run.sh] creating venv in $VENV ..."
  python3 -m venv "$VENV" || {
    echo "run.sh: venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  }
fi

PY="$VENV/bin/python"

if [ ! -f "$STAMP" ]; then
  echo "[run.sh] installing dependencies (first run only) ..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  touch "$STAMP"
fi

if [ "${1:-}" = "--test" ]; then
  shift
  exec "$PY" -m pytest -q "$@"
fi

exec "$PY" main.py "$@"
