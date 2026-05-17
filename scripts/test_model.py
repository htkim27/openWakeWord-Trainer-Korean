#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_model import prediction_score_series


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an openWakeWord model against a WAV file.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--framework", choices=("onnx", "tflite"), default="onnx")
    args = parser.parse_args()

    model_path = Path(args.model)
    wav_path = Path(args.wav)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not wav_path.exists():
        raise SystemExit(f"WAV not found: {wav_path}")

    from openwakeword.model import Model

    model = Model(wakeword_models=[str(model_path)], inference_framework=args.framework)
    result = model.predict_clip(str(wav_path))

    serializable = {}
    for key, values in prediction_score_series(result).items():
        serializable[key] = {
            "max": float(max(values)) if values else 0.0,
            "frames": values,
        }

    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
