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

install_ui_deps() {
  "$PY" -m pip install -r requirements-ui.txt
}

if [[ ! -f "$PIN_FILE" ]]; then
  "$PY" -m pip install -U pip setuptools wheel
  install_ui_deps
  touch "$PIN_FILE"
else
  echo "Reusing existing UI venv"
  if ! "$PY" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md

def version_tuple(value):
    parts = []
    for token in str(value).replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
            break
    return tuple(parts)

exact = {
    "fastapi": "0.115.6",
    "uvicorn": "0.30.6",
    "python-multipart": "0.0.9",
}
minimum = {
    "PyYAML": "6.0.1",
    "silero-vad": "5.0.0",
    "numpy": "1.24.0",
}
present = ("torch",)

for package, expected in exact.items():
    if md.version(package) != expected:
        raise SystemExit(1)
for package, minimum_version in minimum.items():
    if version_tuple(md.version(package)) < version_tuple(minimum_version):
        raise SystemExit(1)
for package in present:
    md.version(package)
PY
  then
    echo "UI dependencies missing or stale; installing recorder dependencies"
    install_ui_deps
  fi
fi

UVICORN="$VENV_DIR/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  install_ui_deps
fi

echo "Launching http://127.0.0.1:$PORT"
exec "$UVICORN" trainer_server:app --host "$HOST" --port "$PORT"
