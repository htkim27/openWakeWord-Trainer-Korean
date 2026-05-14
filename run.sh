#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${REC_VENV_DIR:-$ROOT_DIR/.recorder-venv}"
PYTHON_BIN="${REC_PYTHON_BIN:-python3}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8791}"

PY="$VENV_DIR/bin/python"
PIN_FILE="$VENV_DIR/.ui_deps_installed"

echo "openWakeWord Trainer UI"
echo "ROOT: $ROOT_DIR"
echo "VENV: $VENV_DIR"

if [[ ! -x "$PY" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ ! -f "$PIN_FILE" ]]; then
  "$PY" -m pip install -U pip setuptools wheel
  "$PY" -m pip install -r requirements-ui.txt
  touch "$PIN_FILE"
else
  echo "Reusing existing UI venv"
fi

UVICORN="$VENV_DIR/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  "$PY" -m pip install -r requirements-ui.txt
fi

echo "Launching http://127.0.0.1:$PORT"
exec "$UVICORN" trainer_server:app --host "$HOST" --port "$PORT"
