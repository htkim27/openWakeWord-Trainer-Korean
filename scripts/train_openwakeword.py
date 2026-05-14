#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(message, flush=True)


def safe_name(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "wakeword"


def run(cmd: list[str], env: dict[str, str] | None = None, allow_onnx_failure: Path | None = None) -> None:
    log("")
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
    if result.returncode == 0:
        return
    if allow_onnx_failure and allow_onnx_failure.exists():
        log(f"WARNING: command exited {result.returncode}, but ONNX exists at {allow_onnx_failure}")
        return
    raise SystemExit(result.returncode)


def existing_wav_dir(path: Path) -> bool:
    return path.exists() and any(path.glob("*.wav"))


def make_config(args: argparse.Namespace, model_name: str, output_dir: Path) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    piper_dir = Path(args.piper_dir).resolve()
    features = data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation = data_dir / "validation_set_features.npy"

    background_paths = []
    for candidate in (data_dir / "audioset_16k", data_dir / "fma", data_dir / "background_clips"):
        if existing_wav_dir(candidate):
            background_paths.append(str(candidate))

    if not background_paths:
        fallback = data_dir / "background_clips"
        fallback.mkdir(parents=True, exist_ok=True)
        background_paths.append(str(fallback))
        log(f"WARNING: no background WAVs found; using empty {fallback}")

    feature_data_files: dict[str, str] = {}
    batch_n_per_class: dict[str, int] = {
        "positive": args.positive_batch,
        "adversarial_negative": args.adversarial_batch,
    }
    if features.exists():
        feature_data_files["ACAV100M_sample"] = str(features)
        batch_n_per_class["ACAV100M_sample"] = args.negative_batch
    else:
        log("WARNING: full negative feature file is missing; false-positive performance will be weaker")

    if not validation.exists():
        log("WARNING: validation_set_features.npy is missing; run scripts/download_assets.py")

    return {
        "model_name": model_name,
        "target_phrase": [args.phrase],
        "custom_negative_phrases": args.custom_negative_phrase,
        "n_samples": args.samples,
        "n_samples_val": args.validation_samples,
        "tts_batch_size": args.tts_batch_size,
        "negative_tts_batch_divisor": args.negative_tts_batch_divisor,
        "augmentation_batch_size": args.augmentation_batch_size,
        "piper_sample_generator_path": str(piper_dir),
        "output_dir": str(output_dir),
        "rir_paths": [str(data_dir / "mit_rirs")],
        "background_paths": background_paths,
        "background_paths_duplication_rate": [1 for _ in background_paths],
        "false_positive_validation_data_path": str(validation),
        "augmentation_rounds": args.augmentation_rounds,
        "feature_data_files": feature_data_files,
        "batch_n_per_class": batch_n_per_class,
        "model_type": args.model_type,
        "layer_size": args.layer_size,
        "steps": args.steps,
        "target_accuracy": args.target_accuracy,
        "target_recall": args.target_recall,
        "max_negative_weight": args.max_negative_weight,
        "target_false_positives_per_hour": args.target_fp_per_hour,
    }


def sync_artifacts(output_dir: Path, export_dir: Path, model_name: str, metadata: dict[str, Any]) -> Path | None:
    export_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for path in sorted(output_dir.glob("*")):
        if path.suffix.lower() not in {".onnx", ".data", ".tflite"}:
            continue
        dest = export_dir / path.name
        shutil.copy2(path, dest)
        copied.append(dest.name)

    onnx_path = export_dir / f"{model_name}.onnx"
    metadata["artifacts"] = copied
    metadata["synced_at"] = datetime.now(timezone.utc).isoformat()
    (export_dir / f"{model_name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if copied:
        log("Synced artifacts:")
        for name in copied:
            log(f"  {export_dir / name}")
    else:
        log(f"WARNING: no model artifacts found in {output_dir}")

    return onnx_path if onnx_path.exists() else None


def train_verifier(args: argparse.Namespace, base_model: Path, model_name: str) -> None:
    positive_dir = Path(args.positive_dir).resolve()
    negative_dir = Path(args.negative_dir).resolve()
    positives = list(positive_dir.glob("*.wav"))
    negatives = list(negative_dir.glob("*.wav"))

    if not positives or not negatives:
        log("Skipping verifier: personal and negative WAV clips are both required")
        return

    output = Path(args.export_dir).resolve() / f"{model_name}_verifier.pkl"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train_verifier.py"),
        "--base-model",
        str(base_model),
        "--positive-dir",
        str(positive_dir),
        "--negative-dir",
        str(negative_dir),
        "--output",
        str(output),
    ]
    run(cmd)


def log_torch_devices() -> None:
    try:
        import torch
    except Exception as exc:
        log(f"WARNING: could not import torch for device check: {exc}")
        return

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    selected = "cuda:0" if cuda_available else ("mps" if os.environ.get("OWW_ENABLE_MPS", "1") == "1" and mps_available else "cpu")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device availability: cuda={cuda_available}, mps={mps_available}, selected={selected}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the openWakeWord automatic training pipeline.")
    parser.add_argument("phrase", help="Wake phrase to train, for example 'hey tater'")
    parser.add_argument("--model-name")
    parser.add_argument("--samples", type=int, default=int(os.environ.get("OWW_DEFAULT_SAMPLES", "20000")))
    parser.add_argument("--validation-samples", type=int, default=int(os.environ.get("OWW_DEFAULT_VALIDATION_SAMPLES", "2000")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("OWW_DEFAULT_STEPS", "50000")))
    parser.add_argument("--tts-batch-size", type=int, default=int(os.environ.get("OWW_DEFAULT_TTS_BATCH", "50")))
    parser.add_argument(
        "--negative-tts-batch-divisor",
        type=int,
        default=int(os.environ.get("OWW_NEGATIVE_TTS_DIVISOR", "7")),
        help="Divide TTS batch size by this for adversarial negative clip generation; lower is faster but uses more memory.",
    )
    parser.add_argument("--augmentation-batch-size", type=int, default=int(os.environ.get("OWW_DEFAULT_AUG_BATCH", "16")))
    parser.add_argument("--augmentation-rounds", type=int, default=1)
    parser.add_argument("--positive-batch", type=int, default=50)
    parser.add_argument("--adversarial-batch", type=int, default=50)
    parser.add_argument("--negative-batch", type=int, default=1024)
    parser.add_argument("--model-type", choices=("dnn", "rnn"), default="dnn")
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--target-accuracy", type=float, default=0.7)
    parser.add_argument("--target-recall", type=float, default=0.5)
    parser.add_argument("--target-fp-per-hour", type=float, default=float(os.environ.get("OWW_DEFAULT_TARGET_FP", "0.2")))
    parser.add_argument("--max-negative-weight", type=int, default=int(os.environ.get("OWW_DEFAULT_MAX_NEGATIVE_WEIGHT", "1500")))
    parser.add_argument("--custom-negative-phrase", action="append", default=[])
    parser.add_argument("--data-dir", default=os.environ.get("OWW_DATA_DIR", str(ROOT_DIR / "data")))
    parser.add_argument("--output-root", default=str(ROOT_DIR / "output"))
    parser.add_argument("--export-dir", default=str(ROOT_DIR / "trained_wake_words"))
    parser.add_argument("--openwakeword-dir", default=str(ROOT_DIR / "vendor" / "openwakeword"))
    parser.add_argument("--piper-dir", default=str(ROOT_DIR / "vendor" / "piper-sample-generator"))
    parser.add_argument("--positive-dir", default=str(ROOT_DIR / "personal_samples"))
    parser.add_argument("--negative-dir", default=str(ROOT_DIR / "negative_samples"))
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--train-verifier", action="store_true")
    args = parser.parse_args()
    if args.negative_tts_batch_divisor < 1:
        raise SystemExit("--negative-tts-batch-divisor must be 1 or greater")

    model_name = safe_name(args.model_name or args.phrase)
    output_dir = Path(args.output_root).resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{model_name}.yaml"

    openwakeword_dir = Path(args.openwakeword_dir).resolve()
    train_py = openwakeword_dir / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found at {train_py}. Run ./train_openwakeword.sh first.")

    config = make_config(args, model_name, output_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log(f"Wrote config: {config_path}")

    metadata = {
        "model_name": model_name,
        "phrase": args.phrase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config": config,
    }

    if args.config_only:
        sync_artifacts(output_dir, Path(args.export_dir).resolve(), model_name, metadata)
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # deep-phonemizer's checkpoint predates PyTorch 2.6's weights_only default.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env["OWW_ENABLE_MPS"] = "0"
        env["OWW_PIPER_DEVICE"] = "cpu"

    os.environ.update({key: value for key, value in env.items() if key in {"OWW_ENABLE_MPS", "CUDA_VISIBLE_DEVICES"}})
    log_torch_devices()
    log(f"Piper generator device policy: {env.get('OWW_PIPER_DEVICE', 'auto')}")
    negative_tts_batch = max(1, args.tts_batch_size // args.negative_tts_batch_divisor)
    log(
        "Piper negative TTS batch: "
        f"{negative_tts_batch} (tts_batch_size={args.tts_batch_size}, "
        f"divisor={args.negative_tts_batch_divisor})"
    )

    if not args.skip_generate:
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--generate_clips"], env=env)
    if not args.skip_augment:
        run([sys.executable, str(train_py), "--training_config", str(config_path), "--augment_clips"], env=env)
    if not args.skip_train:
        expected_onnx = output_dir / f"{model_name}.onnx"
        run(
            [sys.executable, str(train_py), "--training_config", str(config_path), "--train_model"],
            env=env,
            allow_onnx_failure=expected_onnx,
        )

    onnx_path = sync_artifacts(output_dir, Path(args.export_dir).resolve(), model_name, metadata)
    if args.train_verifier and onnx_path:
        train_verifier(args, onnx_path, model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
