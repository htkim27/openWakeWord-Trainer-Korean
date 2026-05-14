#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SHIM = '''"""Compatibility entrypoint for openWakeWord's Colab-style trainer.

The upstream openWakeWord training script imports ``generate_samples`` from the
root of piper-sample-generator. Newer piper-sample-generator releases expose
the function from ``piper_sample_generator.__main__`` and require the model path
explicitly. This shim keeps the old import working and uses the bundled
LibriTTS-R generator model by default.
"""
from __future__ import annotations

import os
from pathlib import Path

from piper_sample_generator.__main__ import generate_samples as _generate_samples


def _default_model() -> str:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "models" / "en_US-libritts_r-medium.pt",
        root / "models" / "en-us-libritts-high.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "No Piper generator model found. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def generate_samples(*args, model=None, **kwargs):
    kwargs.pop("auto_reduce_batch_size", None)
    if os.environ.get("OWW_PIPER_ENABLE_MPS", "0") == "1":
        return _generate_samples(*args, model=model or _default_model(), **kwargs)

    import torch

    original_is_available = None
    if getattr(torch.backends, "mps", None) is not None:
        original_is_available = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False

    try:
        return _generate_samples(*args, model=model or _default_model(), **kwargs)
    finally:
        if original_is_available is not None:
            torch.backends.mps.is_available = original_is_available
'''


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_piper_generator.py /path/to/piper-sample-generator", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"piper-sample-generator not found: {root}", file=sys.stderr)
        return 1

    target = root / "generate_samples.py"
    if target.exists() and target.read_text(encoding="utf-8") == SHIM:
        print(f"Piper generate_samples shim already present: {target}")
        return 0

    target.write_text(SHIM, encoding="utf-8")
    print(f"Patched Piper generate_samples shim: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
