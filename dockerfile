FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV REC_HOST=0.0.0.0
ENV REC_PORT=8791
ENV OWW_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    python3 \
    python3-pip \
    python3-venv \
    wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-ui.txt requirements-train.txt ./
ARG TORCH_VERSION=2.6.0
ARG TORCH_CUDA=cu124
RUN python3 -m venv /opt/openwakeword-trainer \
  && /opt/openwakeword-trainer/bin/python -m pip install -U pip setuptools wheel \
  && /opt/openwakeword-trainer/bin/python -m pip install -r requirements-ui.txt -r requirements-train.txt \
  && /opt/openwakeword-trainer/bin/python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
      "torch==${TORCH_VERSION}+${TORCH_CUDA}" "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}" \
  && /opt/openwakeword-trainer/bin/python -m pip install "tensorflow-cpu>=2.15,<2.17"

ENV REC_VENV_DIR=/opt/openwakeword-trainer
ENV OWW_VENV_DIR=/opt/openwakeword-trainer
ENV PATH=/opt/openwakeword-trainer/bin:$PATH

COPY . .

EXPOSE 8791

CMD ["./run.sh"]
