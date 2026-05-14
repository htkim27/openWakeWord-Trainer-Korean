#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ORIGINAL = "self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
PATCHED = """if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif os.environ.get("OWW_ENABLE_MPS", "1") == "1" and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch vendored openWakeWord training to use Apple MPS when available.")
    parser.add_argument("openwakeword_dir")
    args = parser.parse_args()

    train_py = Path(args.openwakeword_dir).resolve() / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found: {train_py}")

    text = train_py.read_text(encoding="utf-8")
    if "OWW_ENABLE_MPS" in text:
        print(f"MPS device patch already present: {train_py}", flush=True)
        return 0

    if ORIGINAL not in text:
        raise SystemExit(f"Could not find upstream device-selection line in {train_py}")

    train_py.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print(f"Patched openWakeWord device selection for MPS: {train_py}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
