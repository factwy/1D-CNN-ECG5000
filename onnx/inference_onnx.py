#!/usr/bin/env python3
"""ECG5000 – ONNX 기반 추론 스크립트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def parse_args():
    p = argparse.ArgumentParser(description="ECG5000 ONNX 추론")
    p.add_argument("--onnx_path",   required=True,
                   help="변환된 .onnx 파일 경로")
    p.add_argument("--data_path",   default="/data/ECG5000/ECG5000_TEST.txt",
                   help="입력 데이터 .txt 파일 경로")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"],
                   help="추론 장치 (기본: cuda)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── OnnxRuntime 세션 ───────────────────────────────────────────────────────
    if args.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.onnx_path, providers=providers)
    active_provider = sess.get_providers()[0]

    print(f"ONNX model   : {args.onnx_path}")
    print(f"Provider     : {active_provider}")

    # ── Data ──────────────────────────────────────────────────────────────────
    arr = pd.read_csv(args.data_path, header=None, sep=r"\s+").values.astype(np.float32)
    X   = arr[:, 1:]                                        # (N, seq_len)
    y   = LabelEncoder().fit_transform(arr[:, 0].astype(int))

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # ── Inference (배치 단위) ──────────────────────────────────────────────────
    all_preds, all_labels = [], []
    for start in range(0, len(X), args.batch_size):
        X_batch = X[start : start + args.batch_size]
        y_batch = y[start : start + args.batch_size]

        # ORT 입력: (B, 1, seq_len) float32
        X_feed  = X_batch[:, np.newaxis, :]                 # channel 차원 추가
        logits  = sess.run([output_name], {input_name: X_feed})[0]
        preds   = logits.argmax(axis=1)

        all_preds.extend(preds)
        all_labels.extend(y_batch)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print(f"\nAccuracy : {(all_preds == all_labels).mean():.4f}")
    print("─" * 50)
    print(classification_report(
        all_labels, all_preds,
        target_names=[f"Class {c}" for c in range(args.num_classes)],
    ))


if __name__ == "__main__":
    main()
