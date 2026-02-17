# ── RTX 4070 SUPER (Ada Lovelace, sm_89) ─────────────────────────────────────
# CUDA 12.4 + cuDNN 9  |  PyTorch 2.5.1
# nvidia-container-toolkit 설치 후 사용 가능
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data outputs

# cuDNN v8 API 활성화 (Ada Lovelace 최적화)
ENV TORCH_CUDNN_V8_API_ENABLED=1

CMD ["python", "train.py", "--model", "vgg"]
