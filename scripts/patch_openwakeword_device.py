#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEVICE_ORIGINAL = "self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
DEVICE_PATCHED = """if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif os.environ.get("OWW_ENABLE_MPS", "1") == "1" and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')"""

NEGATIVE_BATCH_ORIGINAL = 'batch_size=config["tts_batch_size"]//7,'
NEGATIVE_BATCH_PATCHED = (
    'batch_size=max(1, int(config["tts_batch_size"])//'
    'max(1, int(config.get("negative_tts_batch_divisor", 7)))),'
)

FEATURE_SKIP_ORIGINAL = (
    'if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) '
    'or args.overwrite is True:'
)
FEATURE_SKIP_PATCHED = (
    'if args.overwrite is True or not all(os.path.exists(os.path.join(feature_save_dir, name)) '
    'for name in ["positive_features_train.npy", "negative_features_train.npy", '
    '"positive_features_test.npy", "negative_features_test.npy"]):'
)

RESAMPLE_ORIGINAL = '''            if clip_sr != sr:
                raise ValueError("Error! Clip does not have the correct sample rate!")'''
RESAMPLE_PATCHED = '''            if clip_sr != sr:
                clip_data = torchaudio.functional.resample(clip_data, orig_freq=clip_sr, new_freq=sr)
                clip_sr = sr'''

OUTPUT_TYPE_REPLACEMENTS = (
    (
        '''                mode="per_batch"
            ),''',
        '''                mode="per_batch",
                output_type="tensor"
            ),''',
    ),
    (
        '''            torch_audiomentations.BandStopFilter(p=augmentation_probabilities["BandStopFilter"], mode="per_batch"),''',
        '''            torch_audiomentations.BandStopFilter(
                p=augmentation_probabilities["BandStopFilter"],
                mode="per_batch",
                output_type="tensor"
            ),''',
    ),
    (
        '''            torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"]),''',
        '''            torch_audiomentations.Gain(
                max_gain_in_db=0,
                p=augmentation_probabilities["Gain"],
                output_type="tensor"
            ),''',
    ),
    (
        '''        ])
    else:
        augment2 = torch_audiomentations.Compose([''',
        '''        ], output_type="tensor")
    else:
        augment2 = torch_audiomentations.Compose([''',
    ),
    (
        '''        ])

    # Iterate through all clips and augment them''',
        '''        ], output_type="tensor")

    # Iterate through all clips and augment them''',
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch vendored openWakeWord training for Apple MPS and local trainer settings."
    )
    parser.add_argument("openwakeword_dir")
    args = parser.parse_args()

    train_py = Path(args.openwakeword_dir).resolve() / "openwakeword" / "train.py"
    if not train_py.exists():
        raise SystemExit(f"openWakeWord train.py not found: {train_py}")

    text = train_py.read_text(encoding="utf-8")
    changed = False

    if "OWW_ENABLE_MPS" in text:
        print(f"MPS device patch already present: {train_py}", flush=True)
    else:
        if DEVICE_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream device-selection line in {train_py}")
        text = text.replace(DEVICE_ORIGINAL, DEVICE_PATCHED)
        changed = True
        print(f"Patched openWakeWord device selection for MPS: {train_py}", flush=True)

    if "negative_tts_batch_divisor" in text:
        print(f"Negative TTS batch divisor patch already present: {train_py}", flush=True)
    else:
        if NEGATIVE_BATCH_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream negative batch-size line in {train_py}")
        text = text.replace(NEGATIVE_BATCH_ORIGINAL, NEGATIVE_BATCH_PATCHED)
        changed = True
        print(f"Patched openWakeWord negative TTS batch divisor: {train_py}", flush=True)

    if "positive_features_test.npy\", \"negative_features_test.npy" in text:
        print(f"Feature completeness patch already present: {train_py}", flush=True)
    else:
        if FEATURE_SKIP_ORIGINAL not in text:
            raise SystemExit(f"Could not find upstream feature-skip line in {train_py}")
        text = text.replace(FEATURE_SKIP_ORIGINAL, FEATURE_SKIP_PATCHED)
        changed = True
        print(f"Patched openWakeWord feature completeness check: {train_py}", flush=True)

    if changed:
        train_py.write_text(text, encoding="utf-8")

    data_py = Path(args.openwakeword_dir).resolve() / "openwakeword" / "data.py"
    if not data_py.exists():
        raise SystemExit(f"openWakeWord data.py not found: {data_py}")

    data_text = data_py.read_text(encoding="utf-8")
    data_changed = False
    if "torchaudio.functional.resample(clip_data" in data_text:
        print(f"Augmentation resample patch already present: {data_py}", flush=True)
    else:
        if RESAMPLE_ORIGINAL not in data_text:
            raise SystemExit(f"Could not find upstream sample-rate check in {data_py}")
        data_text = data_text.replace(RESAMPLE_ORIGINAL, RESAMPLE_PATCHED)
        data_changed = True
        print(f"Patched openWakeWord augmentation sample-rate resampling: {data_py}", flush=True)

    if data_text.count('output_type="tensor"') >= 11:
        print(f"Torch audiomentations output type patch already present: {data_py}", flush=True)
    else:
        if 'output_type="tensor"' in data_text:
            raise SystemExit(f"Partial torch audiomentations output type patch found in {data_py}")
        for original, patched in OUTPUT_TYPE_REPLACEMENTS:
            if original not in data_text:
                raise SystemExit(f"Could not find upstream torch audiomentations pattern in {data_py}")
            data_text = data_text.replace(original, patched)
        data_changed = True
        print(f"Patched torch audiomentations output_type compatibility: {data_py}", flush=True)

    if data_changed:
        data_py.write_text(data_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
