#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./train_openwakeword.sh "hey tater" [options]

Common options are passed through to scripts/train_openwakeword.py:
  --samples 20000
  --validation-samples 2000
  --steps 50000
  --custom-negative-phrase "phrase"
  --train-verifier

Environment:
  OWW_NEGATIVE_FEATURES=full|skip   default: full
  OWW_DOWNLOAD_BACKGROUND=1|0       default: 1
  OWW_DOWNLOAD_RIRS=1|0             default: 1
  OWW_FORCE_CPU=1                   disable CUDA/MPS visibility
EOF
  exit 1
fi

VENV_DIR="${OWW_VENV_DIR:-$ROOT_DIR/.venv}"
PIPER_REPO_URL="${OWW_PIPER_REPO_URL:-https://github.com/TaterTotterson/piper-sample-generator.git}"
if [[ -n "${OWW_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$OWW_PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.10)"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.11"
elif [[ -x /opt/homebrew/bin/python3.10 ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3.10"
else
  PYTHON_BIN=""
fi
PY="$VENV_DIR/bin/python"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated as an API:UserWarning}"

mkdir -p vendor data trained_wake_words logs "$MPLCONFIGDIR"
export MPLCONFIGDIR PYTHONWARNINGS

if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ Training requires Python 3.10+ because openWakeWord 0.6.0 declares python_requires >=3.10."
  echo "   Install one, then rerun. On Apple Silicon:"
  echo "     brew install python@3.11"
  echo "   Or set:"
  echo "     OWW_PYTHON_BIN=/path/to/python3.11 ./train_openwakeword.sh ..."
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PYTHON_VERSION" in
  3.10|3.11|3.12) : ;;
  *)
    echo "❌ Training requires Python 3.10+; found $PYTHON_VERSION at $PYTHON_BIN."
    echo "   Install Homebrew Python 3.11 and rerun:"
    echo "     brew install python@3.11"
    echo "     rm -rf .venv"
    echo "     ./train_openwakeword.sh \"hey tater\" ..."
    exit 1
    ;;
esac

if [[ ! -x "$PY" ]]; then
  echo "Creating training venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

VENV_PYTHON_VERSION="$("$PY" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "$VENV_PYTHON_VERSION" in
  3.10|3.11|3.12) : ;;
  *)
    echo "❌ Existing training venv uses Python $VENV_PYTHON_VERSION, but openWakeWord needs Python 3.10+."
    echo "   Remove the old venv after installing Python 3.11:"
    echo "     rm -rf .venv"
    echo "     ./train_openwakeword.sh \"hey tater\" ..."
    exit 1
    ;;
esac

if [[ ! -f "$VENV_DIR/.train_deps_installed" ]]; then
  "$PY" -m pip install -U pip wheel
  "$PY" -m pip install -r requirements-train.txt

  touch "$VENV_DIR/.train_deps_installed"
else
  echo "Reusing existing training venv"
fi

if ! "$PY" - <<'PY' >/dev/null 2>&1
import pkg_resources
PY
then
  echo "Repairing training venv: installing setuptools with pkg_resources support"
  "$PY" -m pip install "setuptools==80.9.0"
fi

if [[ ! -d vendor/openwakeword/.git ]]; then
  git clone https://github.com/dscripka/openWakeWord.git vendor/openwakeword
else
  git -C vendor/openwakeword pull --ff-only origin main || true
fi

"$PY" scripts/patch_openwakeword_device.py vendor/openwakeword

if [[ ! -d vendor/piper-sample-generator/.git ]]; then
  git clone "$PIPER_REPO_URL" vendor/piper-sample-generator
else
  current_piper_origin="$(git -C vendor/piper-sample-generator remote get-url origin 2>/dev/null || true)"
  if [[ "$current_piper_origin" != "$PIPER_REPO_URL" ]]; then
    echo "Updating piper-sample-generator origin: $PIPER_REPO_URL"
    git -C vendor/piper-sample-generator remote set-url origin "$PIPER_REPO_URL"
  fi
  git -C vendor/piper-sample-generator pull --ff-only origin master || true
fi

"$PY" scripts/patch_piper_generator.py vendor/piper-sample-generator
"$PY" -m pip install -e vendor/openwakeword -e vendor/piper-sample-generator

DOWNLOAD_ARGS=()
if [[ "${OWW_NEGATIVE_FEATURES:-full}" == "full" ]]; then
  DOWNLOAD_ARGS+=(--negative-features full)
else
  DOWNLOAD_ARGS+=(--negative-features skip)
fi

if [[ "${OWW_DOWNLOAD_BACKGROUND:-1}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--background-hours "${OWW_BACKGROUND_HOURS:-1}")
else
  DOWNLOAD_ARGS+=(--skip-background)
fi

if [[ "${OWW_DOWNLOAD_RIRS:-1}" == "1" ]]; then
  :
else
  DOWNLOAD_ARGS+=(--skip-rirs)
fi

"$PY" scripts/download_assets.py "${DOWNLOAD_ARGS[@]}"

TRAIN_ARGS=("$@")
if [[ "${OWW_FORCE_CPU:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--force-cpu)
fi

"$PY" scripts/train_openwakeword.py "${TRAIN_ARGS[@]}"
