#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OPENWAKEWORD_DIR = Path(
    os.environ.get("OWW_OPENWAKEWORD_DIR", str(ROOT_DIR / "vendor" / "openwakeword"))
).resolve()
PIPER_DIR = Path(
    os.environ.get("OWW_PIPER_DIR", str(ROOT_DIR / "vendor" / "piper-sample-generator"))
).resolve()

FEATURES_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
PIPER_GENERATOR_URL = "https://github.com/TaterTotterson/piper-sample-generator/releases/download/models/en_US-libritts_r-medium.pt"
PIPER_GENERATOR_CONFIG_URL = PIPER_GENERATOR_URL + ".json"
OPENWAKEWORD_RELEASE_BASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
OPENWAKEWORD_RESOURCE_MODELS = [
    "melspectrogram.onnx",
    "embedding_model.onnx",
    "melspectrogram.tflite",
    "embedding_model.tflite",
    "silero_vad.onnx",
]


def log(message: str) -> None:
    print(message, flush=True)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"Reusing {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"Downloading {url}")
    log(f"       -> {dest}")

    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("content-length") or 0)
        seen = 0
        last_pct = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            seen += len(chunk)
            if total:
                pct = int(seen * 100 / total)
                if pct >= last_pct + 5:
                    log(f"  {pct}%")
                    last_pct = pct
    tmp.replace(dest)


def download_mit_rirs(data_dir: Path) -> None:
    out_dir = data_dir / "mit_rirs"
    if list(out_dir.glob("*.wav")):
        log(f"Reusing MIT RIRs in {out_dir}")
        return

    log("Downloading MIT room impulse responses")
    import numpy as np
    import scipy.io.wavfile
    from datasets import load_dataset
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for row in tqdm(dataset, desc="MIT RIRs"):
        name = Path(row["audio"]["path"]).name
        audio = row["audio"]["array"]
        scipy.io.wavfile.write(out_dir / name, 16000, (audio * 32767).astype(np.int16))


def download_background(data_dir: Path, hours: float) -> None:
    out_dir = data_dir / "fma"
    if list(out_dir.glob("*.wav")):
        log(f"Reusing background clips in {out_dir}")
        return

    log(f"Downloading about {hours:g} hour(s) of FMA background audio")
    import numpy as np
    import scipy.io.wavfile
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = max(1, int(hours * 3600 // 30))
    trust_remote_code = os.environ.get("OWW_TRUST_REMOTE_DATASET_CODE", "1") == "1"
    allow_full_download = os.environ.get("OWW_BACKGROUND_ALLOW_FULL_FMA", "0") == "1"

    try:
        dataset = load_dataset(
            "rudraml/fma",
            name="small",
            split="train",
            streaming=not allow_full_download,
            trust_remote_code=trust_remote_code,
        )
        dataset = iter(dataset.cast_column("audio", Audio(sampling_rate=16000)))

        for _ in tqdm(range(n_clips), desc="FMA"):
            row = next(dataset)
            name = Path(row["audio"]["path"]).name
            if not name.endswith(".wav"):
                name = Path(name).with_suffix(".wav").name
            audio = row["audio"]["array"]
            scipy.io.wavfile.write(out_dir / name, 16000, (audio * 32767).astype(np.int16))
        return
    except Exception as exc:
        log(f"WARNING: FMA background download failed: {exc}")
        if not allow_full_download:
            log("         Set OWW_BACKGROUND_ALLOW_FULL_FMA=1 to download the full 7.2 GiB FMA small dataset.")

    if not list(out_dir.glob("*.wav")):
        generate_synthetic_background(out_dir, n_clips)


def generate_synthetic_background(out_dir: Path, n_clips: int) -> None:
    log(f"Generating {n_clips} fallback synthetic background clip(s)")
    import numpy as np
    import scipy.io.wavfile
    from tqdm import tqdm

    sample_rate = 16000
    duration = 30
    n_samples = sample_rate * duration
    rng = np.random.default_rng(1409)

    for idx in tqdm(range(n_clips), desc="Synthetic background"):
        white = rng.normal(0, 0.018, n_samples)
        low = np.cumsum(rng.normal(0, 0.0015, n_samples))
        low = low / max(np.max(np.abs(low)), 1e-6) * rng.uniform(0.006, 0.018)
        t = np.arange(n_samples) / sample_rate
        hum_freq = rng.choice([50.0, 60.0, 120.0])
        hum = np.sin(2 * np.pi * hum_freq * t + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.001, 0.006)
        audio = np.clip(white + low + hum, -1.0, 1.0)
        scipy.io.wavfile.write(out_dir / f"synthetic_background_{idx:04d}.wav", sample_rate, (audio * 32767).astype(np.int16))


def download_openwakeword_resources() -> None:
    models_dir = OPENWAKEWORD_DIR / "openwakeword" / "resources" / "models"
    for name in OPENWAKEWORD_RESOURCE_MODELS:
        download_file(f"{OPENWAKEWORD_RELEASE_BASE}/{name}", models_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download reusable openWakeWord training assets.")
    parser.add_argument("--data-dir", default=os.environ.get("OWW_DATA_DIR", str(ROOT_DIR / "data")))
    parser.add_argument("--negative-features", choices=("full", "skip"), default="full")
    parser.add_argument("--skip-rirs", action="store_true")
    parser.add_argument("--skip-background", action="store_true")
    parser.add_argument("--background-hours", type=float, default=1.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    download_file(VALIDATION_URL, data_dir / "validation_set_features.npy")

    if args.negative_features == "full":
        download_file(FEATURES_URL, data_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    else:
        log("Skipping full negative feature download")

    piper_model = PIPER_DIR / "models" / "en_US-libritts_r-medium.pt"
    download_file(PIPER_GENERATOR_URL, piper_model)
    download_file(PIPER_GENERATOR_CONFIG_URL, Path(str(piper_model) + ".json"))
    download_openwakeword_resources()

    if not args.skip_rirs:
        download_mit_rirs(data_dir)

    if not args.skip_background:
        download_background(data_dir, args.background_hours)

    log("Assets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
