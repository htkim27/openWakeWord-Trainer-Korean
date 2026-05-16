#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${OWW_DATA_DIR:-$ROOT_DIR}"
VENV_DIR="${REC_VENV_DIR:-$DATA_DIR/.recorder-venv}"
PYTHON_BIN="${REC_PYTHON_BIN:-python3}"
HOST="${REC_HOST:-0.0.0.0}"
PORT="${REC_PORT:-8791}"
TRAIN_VENV_DIR="${OWW_VENV_DIR:-$DATA_DIR/.venv}"
OUTPUT_ROOT="${OWW_OUTPUT_ROOT:-$DATA_DIR/output}"
EXPORT_DIR="${OWW_EXPORT_DIR:-$DATA_DIR/trained_wake_words}"
OPENWAKEWORD_DIR="${OWW_OPENWAKEWORD_DIR:-$DATA_DIR/vendor/openwakeword}"
PIPER_DIR="${OWW_PIPER_DIR:-$DATA_DIR/vendor/piper-sample-generator}"

PY="$VENV_DIR/bin/python"
PIN_FILE="$VENV_DIR/.ui_deps_installed"

echo "openWakeWord Trainer UI"
echo "ROOT: $ROOT_DIR"
echo "DATA: $DATA_DIR"
echo "VENV: $VENV_DIR"

mkdir -p \
  "$DATA_DIR" \
  "$DATA_DIR/personal_samples" \
  "$DATA_DIR/negative_samples" \
  "$DATA_DIR/captured_audio" \
  "$DATA_DIR/trim_history" \
  "$DATA_DIR/logs" \
  "$EXPORT_DIR" \
  "$OUTPUT_ROOT" \
  "$(dirname "$OPENWAKEWORD_DIR")" \
  "$(dirname "$PIPER_DIR")"

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

export OWW_DATA_DIR="$DATA_DIR"
export OWW_OUTPUT_ROOT="$OUTPUT_ROOT"
export OWW_EXPORT_DIR="$EXPORT_DIR"
export OWW_OPENWAKEWORD_DIR="$OPENWAKEWORD_DIR"
export OWW_PIPER_DIR="$PIPER_DIR"
export OWW_VENV_DIR="$TRAIN_VENV_DIR"
export OWW_PERSONAL_DIR="${OWW_PERSONAL_DIR:-$DATA_DIR/personal_samples}"
export OWW_NEGATIVE_DIR="${OWW_NEGATIVE_DIR:-$DATA_DIR/negative_samples}"
export OWW_CAPTURED_DIR="${OWW_CAPTURED_DIR:-$DATA_DIR/captured_audio}"
export OWW_TRIM_HISTORY_DIR="${OWW_TRIM_HISTORY_DIR:-$DATA_DIR/trim_history}"
export OWW_LOG_DIR="${OWW_LOG_DIR:-$DATA_DIR/logs}"
export OWW_TRAINED_DIR="${OWW_TRAINED_DIR:-$EXPORT_DIR}"
export STATIC_DIR="${STATIC_DIR:-$ROOT_DIR/static}"

echo "Launching http://127.0.0.1:$PORT"
exec "$UVICORN" trainer_server:app --host "$HOST" --port "$PORT"
