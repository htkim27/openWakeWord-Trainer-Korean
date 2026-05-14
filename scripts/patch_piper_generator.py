#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
from pathlib import Path


SHIM = '''"""Compatibility entrypoint for openWakeWord's Colab-style trainer.

The upstream openWakeWord training script imports ``generate_samples`` from the
root of piper-sample-generator. Newer piper-sample-generator releases expose
the function from ``piper_sample_generator.__main__`` and require the model path
explicitly. This shim keeps the old import working and uses the bundled
LibriTTS-R generator model by default. It also lets Piper use CUDA/MPS by
default while keeping a CPU fallback for Apple Silicon MPS edge cases. Piper
voices commonly emit 22.05 kHz WAVs, so generated clips are resampled to 16 kHz
for openWakeWord augmentation.
"""
from __future__ import annotations

import os
import sys
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


def _device_policy() -> str:
    explicit = os.environ.get("OWW_PIPER_DEVICE")
    if explicit:
        return explicit.strip().lower()

    legacy_mps = os.environ.get("OWW_PIPER_ENABLE_MPS")
    if legacy_mps is not None:
        return "mps" if legacy_mps == "1" else "cpu"

    if os.environ.get("OWW_ENABLE_MPS") == "0":
        return "cpu"

    return "auto"


def _torch_devices():
    import torch

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    return torch, cuda_available, mps_available


def _patch_mps_graph_fuser() -> None:
    """Avoid TorchScript graph fuser issues seen by Piper VITS on some MPS stacks."""
    try:
        import torch
        from piper_train.vits import commons, modules
    except Exception:
        return

    def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
        channel_value = n_channels[0]
        n_channels_int = int(
            channel_value.item() if hasattr(channel_value, "item") else channel_value
        )
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        return t_act * s_act

    commons.fused_add_tanh_sigmoid_multiply = fused_add_tanh_sigmoid_multiply
    modules.fused_add_tanh_sigmoid_multiply = fused_add_tanh_sigmoid_multiply


def _looks_like_mps_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "mps" in message
        or "graph fuser" in message
        or "placeholder storage" in message
        or "notimplementederror" in exc.__class__.__name__.lower()
    )


def _output_dir_from_call(args, kwargs) -> Path | None:
    output_dir = kwargs.get("output_dir")
    if output_dir is None and len(args) >= 2:
        output_dir = args[1]
    return Path(output_dir) if output_dir is not None else None


def _resample_output_wavs(output_dir: Path | None) -> None:
    if output_dir is None:
        return

    target_sample_rate = int(os.environ.get("OWW_PIPER_OUTPUT_SAMPLE_RATE", "16000"))
    if target_sample_rate <= 0:
        return

    import math
    import numpy as np
    import scipy.io.wavfile
    import scipy.signal

    changed = 0
    for wav_path in Path(output_dir).glob("*.wav"):
        sample_rate, audio = scipy.io.wavfile.read(wav_path)
        if sample_rate == target_sample_rate:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        gcd = math.gcd(int(sample_rate), target_sample_rate)
        resampled = scipy.signal.resample_poly(
            audio.astype(np.float32),
            target_sample_rate // gcd,
            int(sample_rate) // gcd,
        )
        resampled = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(wav_path, target_sample_rate, resampled)
        changed += 1

    if changed:
        print(
            f"Resampled {changed} Piper clip(s) to {target_sample_rate} Hz for openWakeWord.",
            file=sys.stderr,
            flush=True,
        )


def _call_with_device_visibility(*args, model=None, disable_cuda=False, disable_mps=False, **kwargs):
    import torch

    original_cuda_is_available = None
    original_mps_is_available = None
    output_dir = _output_dir_from_call(args, kwargs)

    if disable_cuda:
        original_cuda_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

    if disable_mps and getattr(torch.backends, "mps", None) is not None:
        original_mps_is_available = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False

    try:
        result = _generate_samples(*args, model=model or _default_model(), **kwargs)
        _resample_output_wavs(output_dir)
        return result
    finally:
        if original_cuda_is_available is not None:
            torch.cuda.is_available = original_cuda_is_available
        if original_mps_is_available is not None:
            torch.backends.mps.is_available = original_mps_is_available


def generate_samples(*args, model=None, **kwargs):
    kwargs.pop("auto_reduce_batch_size", None)
    policy = _device_policy()
    _, cuda_available, mps_available = _torch_devices()

    if policy in {"cpu", "off"}:
        return _call_with_device_visibility(
            *args, model=model, disable_cuda=True, disable_mps=True, **kwargs
        )

    if policy == "cuda":
        if not cuda_available:
            raise RuntimeError("OWW_PIPER_DEVICE=cuda requested, but CUDA is not available")
        return _call_with_device_visibility(*args, model=model, **kwargs)

    if policy == "mps":
        if not mps_available:
            raise RuntimeError("OWW_PIPER_DEVICE=mps requested, but MPS is not available")
        _patch_mps_graph_fuser()
        return _call_with_device_visibility(
            *args, model=model, disable_cuda=True, disable_mps=False, **kwargs
        )

    if policy != "auto":
        raise RuntimeError(
            "Unsupported OWW_PIPER_DEVICE value "
            f"{policy!r}; expected auto, cpu, cuda, or mps"
        )

    if cuda_available:
        return _call_with_device_visibility(*args, model=model, **kwargs)

    if mps_available:
        _patch_mps_graph_fuser()
        try:
            return _call_with_device_visibility(
                *args, model=model, disable_cuda=True, disable_mps=False, **kwargs
            )
        except (NotImplementedError, RuntimeError) as exc:
            if not _looks_like_mps_error(exc):
                raise
            print(
                "WARNING: Piper MPS generation failed; retrying on CPU. "
                "Set OWW_PIPER_DEVICE=mps to fail fast instead.",
                file=sys.stderr,
                flush=True,
            )

    return _call_with_device_visibility(
        *args, model=model, disable_cuda=True, disable_mps=True, **kwargs
    )
'''


def patch_pyproject(root: Path) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = pyproject.read_text(encoding="utf-8")
    if re.search(r"""["']piper_train\*?["']""", text):
        print(f"Piper package metadata already includes piper_train: {pyproject}")
        return False

    pattern = re.compile(r"(?P<prefix>include\s*=\s*\[)(?P<body>.*?)(?P<suffix>\])", re.DOTALL)
    changed = False

    def add_piper_train(match: re.Match[str]) -> str:
        nonlocal changed
        body = match.group("body")
        if "piper_sample_generator" not in body:
            return match.group(0)
        if "piper_train" in body:
            return match.group(0)

        if "\n" in body:
            body = body.rstrip()
            if body.strip() and not body.rstrip().endswith(","):
                body += ","
            body += '\n    "piper_train*",\n'
        else:
            body = body.strip()
            if body and not body.endswith(","):
                body += ","
            body += ' "piper_train*"'

        changed = True
        return f"{match.group('prefix')}{body}{match.group('suffix')}"

    patched = pattern.sub(add_piper_train, text)
    if not changed:
        raise SystemExit(
            f"Could not find Piper package include list in {pyproject}; "
            "expected an include list containing piper_sample_generator"
        )

    pyproject.write_text(patched, encoding="utf-8")
    print(f"Patched Piper package metadata: {pyproject}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_piper_generator.py /path/to/piper-sample-generator", file=sys.stderr)
        return 2

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"piper-sample-generator not found: {root}", file=sys.stderr)
        return 1

    patch_pyproject(root)

    target = root / "generate_samples.py"
    if target.exists() and target.read_text(encoding="utf-8") == SHIM:
        print(f"Piper generate_samples shim already present: {target}")
        return 0

    target.write_text(SHIM, encoding="utf-8")
    print(f"Patched Piper generate_samples shim: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
