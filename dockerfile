FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV REC_HOST=0.0.0.0
ENV REC_PORT=8791
ENV OWW_DATA_DIR=/data
ENV REC_VENV_DIR=/data/.recorder-venv
ENV OWW_VENV_DIR=/data/.venv
ENV OWW_OUTPUT_ROOT=/data/output
ENV OWW_EXPORT_DIR=/data/trained_wake_words
ENV OWW_OPENWAKEWORD_DIR=/data/vendor/openwakeword
ENV OWW_PIPER_DIR=/data/vendor/piper-sample-generator
ENV OWW_TORCH_VERSION=2.6.0
ENV OWW_TORCH_CUDA=cu124
ENV OWW_TRAIN_NUM_WORKERS=2
ENV OWW_TRAIN_PREFETCH_FACTOR=2
ENV PIP_CACHE_DIR=/data/.cache/pip
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    gcc \
    git \
    libc6-dev \
    libgomp1 \
    libsndfile1 \
    wget \
  && rm -rf /var/lib/apt/lists/* \
  && mkdir -p /data

WORKDIR /opt/openwakeword-trainer

COPY --chmod=0755 run.sh train_openwakeword.sh ./
COPY requirements-ui.txt requirements-train.txt trainer_server.py README.md ./
COPY scripts/ scripts/
COPY static/ static/

EXPOSE 8791

CMD ["/bin/bash", "-lc", "/opt/openwakeword-trainer/run.sh"]
