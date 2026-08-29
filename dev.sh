#!/usr/bin/env bash
# macOS / Linux entry point. Creates the virtual environment if it is missing,
# installs anything absent, then starts the server.
#
#   ./dev.sh           start with the classical-CV stand-in
#   ./dev.sh --real    use the trained model in models/
#   ./dev.sh --test    run the test suite instead
#   ./dev.sh --sim     drive the simulator against a running server
#
# The environment lives in .venv-mac, deliberately not .venv: a virtual
# environment holds binaries compiled for one operating system, so a folder
# shared between machines must not have them collide. Windows uses .venv-win
# via dev.ps1.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-mac"
PORT="${PORT:-8000}"

# Pick the newest interpreter available. 3.10 is the floor: FastAPI resolves
# route annotations at runtime and they use `float | None`.
pick_python() {
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
      echo "$c"; return
    fi
  done
  echo "No Python 3.10+ found." >&2
  echo "macOS ships 3.9 as python3; install a newer one:" >&2
  echo "  https://www.python.org/downloads/macos/   (or: brew install python@3.12)" >&2
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  PY=$(pick_python)
  echo "Creating $VENV with $PY ..."
  "$PY" -m venv "$VENV"
fi
PY="$VENV/bin/python"
echo "Python $("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])') at $PY"

# Stub mode, the simulator and the tests need none of PyTorch, so only
# --real pays for the model stack.
if [ "${1:-}" = "--real" ]; then
  REQS="requirements-model.txt"; PROBE="import fastapi, cv2, ultralytics"
else
  REQS="requirements-dev.txt"
  PROBE="import fastapi, cv2, reportlab, requests, cryptography, pytest"
fi
if ! "$PY" -c "$PROBE" 2>/dev/null; then
  echo "Installing dependencies from $REQS (once) ..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r "$REQS"
fi

case "${1:-}" in
  --test) exec "$PY" -m pytest tests/ -q ;;
  --sim)  exec "$PY" scripts/simulate_drive.py --frames 220 --potholes 10 \
                     --oracle --server "https://127.0.0.1:$PORT" ;;
  --real) exec "$PY" run.py --https --port "$PORT" ;;
  *)      exec "$PY" run.py --https --port "$PORT" --stub ;;
esac
