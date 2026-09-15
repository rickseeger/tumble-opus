#!/usr/bin/env bash
# Tumble - one command from a fresh clone: creates the venv if needed,
# installs deps, launches the game.
#
#   ./run.sh              windowed playtest
#   ./run.sh --headless   no-window smoke run (servers, CI)
#   ./run.sh --test       run the pytest suite headlessly
#   ./run.sh --soak       sustained-demolition soak (load + leak harness)
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

if [ "${1:-}" = "--soak" ]; then
  shift
  # The sustained-demolition soak. Defaults to the short CI shape (~8 s);
  # pass --structures N / --settle-seconds S for a longer run.
  if [ "$#" -eq 0 ]; then
    exec "$PY" tools_soak.py --ci
  fi
  exec "$PY" tools_soak.py "$@"
fi

if [ "${1:-}" = "--test" ]; then
  shift
  # NOTE: pytest.ini already sets -q. Passing -q again here makes pytest
  # doubly-quiet, which suppresses the "N passed" summary line entirely and
  # leaves anyone validating this project with no pass/fail count at all.
  # Report verbosely instead: the summary line is the whole point.
  exec "$PY" -m pytest -r a "$@"
fi

exec "$PY" main.py "$@"
