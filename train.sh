#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-vgg}"          # 첫 번째 인자로 모델 지정 (기본: vgg)

python train.py \
    --model       "$MODEL" \
    --data_dir    ./data \
    --output_dir  ./outputs \
    --batch_size  64 \
    --epochs      50 \
    --lr          1e-3 \
    --dropout     0.3 \
    --seed        42 \
    --amp \
    --no-compile
