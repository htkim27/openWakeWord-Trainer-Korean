# Korean openWakeWord Trainer

[openWakeWord](https://github.com/dscripka/openWakeWord)의 학습 파이프라인과
[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)를 결합해 한국어 웨이크워드
모델을 만드는 프로젝트입니다.

openWakeWord의 기본 합성 파이프라인은 Piper TTS를 사용하지만 한국어 음성이 제공되지
않습니다. 이 프로젝트는 OmniVoice로 한국어 positive/hard-negative 음성 44,000개를 먼저
병렬 생성하고, 완성된 데이터를 openWakeWord의 augmentation 및 classifier 학습 단계에
전달해 이 제약을 우회합니다.

## 핵심 동작

- 타겟 단어와 쉼표로 구분한 유사 발음 단어를 CLI에서 입력합니다.
- `오둥아`, `오 둥아`, `오둥 아`, `오 둥 아`처럼 음절 경계의 띄어쓰기 운율을 자동 조합합니다.
- 성별, 연령, 음높이, 억양/속삭임을 무작위로 조합해 다양한 음색을 생성합니다.
- voice-design 조합이 빈 오디오를 반환하거나 실패하면 OmniVoice auto-voice로 재시도합니다.
- openWakeWord가 한글 모델명을 `wakeword`로 sanitize하는 동작에 맞춰, WAV는 반드시
  `output/wakeword/wakeword/{positive,negative}_{train,test}` 아래에 생성됩니다.
- 중단 후 같은 명령을 다시 실행하면 부족한 WAV만 이어서 생성합니다.

## 요구 사항

- Linux 권장 (Ubuntu 22.04 이상)
- Python 3.10–3.12 (`3.11` 권장)
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPU와 CUDA 권장
- 저장 공간 30 GB 이상, RAM 16 GB 이상 권장
- GPU worker마다 OmniVoice 모델 하나가 로드됩니다. 24 GB VRAM급 단일 GPU에서는 먼저
  `--workers 1`로 확인하고, 여러 GPU는 `--devices cuda:0,cuda:1`처럼 분산하세요.

CPU에서도 실행할 수 있지만 44,000개 합성에는 매우 오랜 시간이 걸립니다. VRAM이
부족하면 worker 수를 줄이십시오.

## 빠른 시작

```bash
git clone https://github.com/htkim27/openWakeWord-Trainer-Korean.git
cd openWakeWord-Trainer-Korean
./train_korean_wakeword.sh "오둥아" "오동아,우동아,오징어"
```

이 한 명령이 lockfile 기반 `.venv` 구성, 패키지 설치, OmniVoice 모델 다운로드, 44,000개 클립 생성,
openWakeWord 데이터 증강/학습, artifact export를 차례로 수행합니다. 기본 데이터 구성은
positive train 20,000개, positive test 2,000개, negative train 20,000개, negative test
2,000개입니다. 최종 모델은 `models/오둥아.onnx`에 생성됩니다.

여러 GPU를 사용하는 예:

```bash
./train_korean_wakeword.sh "오둥아" "오동아,우동아,오징어" \
  --workers 2 --devices cuda:0,cuda:1
```

상승형/하강형 운율을 함께 학습하는 예:

```bash
./train_korean_wakeword.sh "오둥아" "오동아,우동아,오징어" \
  --positive-variation "오둥아!" \
  --positive-variation "오둥아."
```

생성기만 작은 수량으로 확인할 수도 있습니다.

```bash
uv run python generate_korean_dataset.py "오둥아" \
  --negatives "오동아,우동아" --workers 1 \
  --positive-train 2 --positive-test 2 \
  --negative-train 2 --negative-test 2
```

> 작은 smoke test 뒤 기본 명령을 실행하면 각 split을 44,000개 기본 구성까지 채웁니다.

## 프로젝트 구조

```text
.
├── generate_korean_dataset.py     # OmniVoice 병렬 데이터 생성기
├── train_korean_wakeword.sh       # uv 설정 → 생성 → 학습 → export
├── train_openwakeword.sh          # 원본 trainer 환경/자산 준비 래퍼
├── scripts/
│   └── train_openwakeword.py      # config 생성 및 upstream train.py 호출
├── pyproject.toml
├── requirements-train.txt
├── output/
│   └── wakeword/
│       └── wakeword/
│           ├── positive_train/
│           ├── positive_test/
│           ├── negative_train/
│           └── negative_test/
└── models/
    ├── 오둥아.onnx
    ├── 오둥아.onnx.data           # 생성되는 모델 형식일 때
    ├── 오둥아.tflite              # 변환 성공 시
    └── 오둥아.json
```

`scripts/train_openwakeword.py`는 `vendor/openwakeword/openwakeword/train.py`를 직접 호출합니다.
사전 생성한 네 폴더가 목표 수량을 이미 만족하므로 영어 Piper TTS 생성은 `--skip-generate`로
건너뛰고, augmentation과 feature 생성부터 실행합니다.

## 환경 변수

- `PYTHON_VERSION=3.11`: `uv`가 사용할 Python 버전
- `OWW_TORCH_CUDA=cu124`: CUDA PyTorch wheel 선택
- `OWW_NEGATIVE_FEATURES=skip`: 대용량 negative feature 다운로드 생략(빠른 시험용)
- `OWW_DOWNLOAD_BACKGROUND=0`: background dataset 다운로드 생략
- `HF_TOKEN`: Hugging Face 인증이 필요한 환경의 토큰

모델과 데이터셋의 라이선스 및 이용 조건은 각 upstream 프로젝트와 체크포인트의 조건을
따릅니다. 합성 음성을 사칭, 사기 또는 동의 없는 음성 복제에 사용하지 마십시오.
