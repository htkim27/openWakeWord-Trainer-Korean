#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo 'Usage: ./train_korean_wakeword.sh "오둥아" "오동아,우동아,오징어" [generator options]' >&2
  exit 2
fi
TARGET_WORD="$1"
NEGATIVE_WORDS="$2"
shift 2

if [[ "$TARGET_WORD" == *"/"* || "$TARGET_WORD" == *"\\"* ]]; then
  echo "The target word cannot contain a path separator." >&2
  exit 2
fi

command -v uv >/dev/null 2>&1 || {
  echo 'uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  exit 1
}

echo "[1/5] Creating the uv environment and installing dependencies"
uv venv --python "${PYTHON_VERSION:-3.11}" --allow-existing .venv
uv sync --locked --python .venv/bin/python
touch .venv/.train_deps_installed
if [[ -n "${OWW_TORCH_CUDA:-}" && "${OWW_FORCE_CPU:-0}" != 1 ]]; then
  printf '%s\n' "${OWW_TORCH_VERSION:-2.13.0}+${OWW_TORCH_CUDA}" > .venv/.train_deps_key
else
  printf '%s\n' cpu > .venv/.train_deps_key
fi

echo "[2/5] Generating 44,000 Korean clips with OmniVoice"
.venv/bin/python generate_korean_dataset.py "$TARGET_WORD" --negatives "$NEGATIVE_WORDS" "$@"

echo "[3/5] Running upstream openWakeWord augmentation and training"
export OWW_PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
export OWW_OUTPUT_ROOT="$ROOT_DIR/output"
export OWW_EXPORT_DIR="$ROOT_DIR/trained_wake_words"
./train_openwakeword.sh "$TARGET_WORD" --model-name wakeword \
  --samples 20000 --validation-samples 2000 --skip-generate --skip-calibration

echo "[4/5] Renaming and copying model artifacts"
mkdir -p models
for source in output/wakeword/wakeword.{onnx,onnx.data,tflite,yaml} \
              trained_wake_words/wakeword.{onnx,onnx.data,tflite,json}; do
  [[ -f "$source" ]] || continue
  suffix="${source##*/wakeword}"
  cp -f "$source" "models/${TARGET_WORD}${suffix}"
done
[[ -f "models/${TARGET_WORD}.onnx" ]] || {
  echo "Expected ONNX artifact was not produced." >&2
  exit 1
}
echo "[5/5] Complete: models/${TARGET_WORD}.onnx"
