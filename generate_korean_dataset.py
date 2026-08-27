#!/usr/bin/env python3
"""Generate a Korean openWakeWord dataset with k2-fsa/OmniVoice."""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf

MODEL_ID = "k2-fsa/OmniVoice"
DATASET_PATH = Path("output/wakeword/wakeword")
_model = None


class Task(NamedTuple):
    text: str
    output_path: str
    instruct: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 16 kHz Korean wake-word clips with OmniVoice.")
    parser.add_argument("target", help='Target wake word, for example "오둥아"')
    parser.add_argument(
        "--positive-variation",
        action="append",
        default=[],
        help='Additional positive prosody text; repeat it, for example "오둥아!" and "오둥아."',
    )
    parser.add_argument("--negatives", required=True, help="Comma-separated hard negative phrases")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--devices", default="auto", help='For example "cuda:0,cuda:1", "mps", or "cpu"')
    parser.add_argument("--positive-train", type=int, default=20_000)
    parser.add_argument("--positive-test", type=int, default=2_000)
    parser.add_argument("--negative-train", type=int, default=20_000)
    parser.add_argument("--negative-test", type=int, default=2_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent, help=argparse.SUPPRESS)
    args = parser.parse_args()
    counts = (args.positive_train, args.positive_test, args.negative_train, args.negative_test)
    if args.workers < 1 or any(count < 1 for count in counts):
        parser.error("workers and all split sizes must be at least 1")
    return args


def spacing_variants(text: str) -> list[str]:
    """Generate all short-phrase spacing rhythms: 오둥아, 오 둥아, etc."""
    compact = "".join(text.split())
    if not compact:
        raise ValueError("Text must not be empty")
    match = re.fullmatch(r"(.+?)([^\w가-힣]*)", compact)
    assert match is not None
    spoken, punctuation = match.groups()
    return [
        "".join(char + (" " if i < len(spoken) - 1 and mask[i] else "") for i, char in enumerate(spoken))
        + punctuation
        for mask in itertools.product((False, True), repeat=max(0, len(spoken) - 1))
    ]


def random_instruct() -> str | None:
    parts: list[str] = []
    if random.random() < 0.8:
        parts.append(random.choice(("male", "female")))
    if random.random() < 0.8:
        parts.append(random.choice(("child", "teenager", "young adult", "middle-aged", "elderly")))
    if random.random() < 0.7:
        parts.append(random.choice(("very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch")))
    if random.random() < 0.4:
        parts.append(random.choice(("korean accent", "american accent", "whisper")))
    return ", ".join(parts) or None


def resolve_devices(raw: str) -> list[str]:
    if raw != "auto":
        devices = [item.strip() for item in raw.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")
        return devices
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def worker_init(devices: list[str]) -> None:
    global _model
    import torch
    from omnivoice import OmniVoice
    identity = mp.current_process()._identity
    device = devices[((identity[0] - 1) if identity else 0) % len(devices)]
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    _model = OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)


def generate(text: str, instruct: str | None) -> np.ndarray:
    kwargs = {"text": text, "language": "ko"}
    if instruct:
        kwargs["instruct"] = instruct
    audio = _model.generate(**kwargs)[0]
    if audio is None or len(audio) == 0:
        raise ValueError("OmniVoice returned empty audio")
    return np.asarray(audio, dtype=np.float32)


def worker_task(task: Task) -> tuple[bool, str]:
    try:
        try:
            audio = generate(task.text, task.instruct)
        except Exception:
            # Voice-design occasionally returns length-zero audio; auto voice is the fallback.
            audio = generate(task.text, None)
        audio = librosa.resample(audio, orig_sr=24_000, target_sr=16_000)
        if audio.size == 0:
            raise ValueError("Resampling produced empty audio")
        sf.write(task.output_path, audio, 16_000, subtype="PCM_16")
        return True, task.output_path
    except Exception as exc:
        Path(task.output_path).unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def missing_tasks(directory: Path, wanted: int, texts: list[str]) -> list[Task]:
    existing = sum(1 for _ in directory.glob("*.wav"))
    return [Task(random.choice(texts), str(directory / f"{uuid.uuid4().hex}.wav"), random_instruct())
            for _ in range(max(0, wanted - existing))]


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    negatives = [item.strip() for item in args.negatives.split(",") if item.strip()]
    if not negatives:
        raise SystemExit("--negatives must contain at least one phrase")
    positive_phrases = [args.target, *args.positive_variation]
    positive_texts = sorted({variant for phrase in positive_phrases for variant in spacing_variants(phrase)})
    negative_texts = sorted({variant for word in negatives for variant in spacing_variants(word)})
    base = args.project_root.resolve() / DATASET_PATH  # Deliberately fixed: see README.
    specs = {
        "positive_train": (args.positive_train, positive_texts),
        "positive_test": (args.positive_test, positive_texts),
        "negative_train": (args.negative_train, negative_texts),
        "negative_test": (args.negative_test, negative_texts),
    }
    for split in specs:
        (base / split).mkdir(parents=True, exist_ok=True)
    tasks_by_split = {
        split: missing_tasks(base / split, wanted, texts)
        for split, (wanted, texts) in specs.items()
    }
    for split, tasks in tasks_by_split.items():
        wanted = specs[split][0]
        print(f"{split}: {wanted - len(tasks)}/{wanted} present; generating {len(tasks)}", flush=True)
    if not any(tasks_by_split.values()):
        print(f"Dataset already complete: {base}")
        return 0

    devices = resolve_devices(args.devices)
    print(f"Fixed dataset path: {base}\nWorkers: {args.workers}; devices: {', '.join(devices)}", flush=True)
    started, completed = time.monotonic(), 0
    with mp.get_context("spawn").Pool(args.workers, worker_init, (devices,)) as pool:
        for split, (wanted, texts) in specs.items():
            directory = base / split
            tasks = tasks_by_split[split]
            for ok, detail in pool.imap_unordered(worker_task, tasks):
                if not ok:
                    print(f"WARNING: {detail}", file=sys.stderr, flush=True)
                completed += int(ok)
                if completed and completed % 100 == 0:
                    elapsed = time.monotonic() - started
                    print(f"Generated {completed} ({elapsed / completed:.2f}s/clip average)", flush=True)
            actual = sum(1 for _ in directory.glob("*.wav"))
            if actual != wanted:
                raise SystemExit(f"{split} incomplete ({actual}/{wanted}). Rerun to resume missing clips.")
    print(f"Dataset complete in {time.monotonic() - started:.1f}s: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
