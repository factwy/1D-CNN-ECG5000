#!/usr/bin/env python3
"""ECG5000 – TFLite 기반 추론 스크립트"""

import os
os.environ['TFLITE_DISABLE_XNNPACK'] = '1'  # TF import 전에 설정해야 적용됨

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def parse_args():
    p = argparse.ArgumentParser(description="ECG5000 TFLite 추론")
    p.add_argument("--tflite_path", required=True,
                   help="변환된 .tflite 파일 경로")
    p.add_argument("--data_path",   default="/data/ECG5000/ECG5000_TEST.txt",
                   help="입력 데이터 .txt 파일 경로")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--batch_size",  type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # ── TFLite Interpreter 초기화 ──────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=args.tflite_path)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx      = input_details[0]["index"]
    output_idx     = output_details[0]["index"]
    input_dtype    = input_details[0]["dtype"]

    print(f"TFLite model  : {args.tflite_path}")
    print(f"Input dtype   : {input_dtype.__name__}")
    print(f"Input shape   : {input_details[0]['shape']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    arr = pd.read_csv(args.data_path, header=None, sep=r"\s+").values.astype(np.float32)
    X   = arr[:, 1:]                                    # (N, seq_len)
    y   = LabelEncoder().fit_transform(arr[:, 0].astype(int))

    # ── Inference (배치 단위) ──────────────────────────────────────────────────
    all_preds, all_labels = [], []

    for start in range(0, len(X), args.batch_size):
        X_batch = X[start : start + args.batch_size]   # (B, seq_len)
        y_batch = y[start : start + args.batch_size]
        B       = len(X_batch)

        # onnx2tf 변환 시 channels-last로 전치됨: (B, seq_len, 1)
        X_feed = X_batch[:, :, np.newaxis]              # (B, seq_len, 1)

        # 배치 크기에 맞게 입력 텐서 재설정
        interpreter.resize_tensor_input(input_idx, X_feed.shape)
        interpreter.allocate_tensors()

        # INT8 양자화 모델 처리
        if input_dtype == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            X_feed = (X_feed / scale + zero_point).clip(-128, 127).astype(np.int8)

        interpreter.set_tensor(input_idx, X_feed)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_idx)     # (B, num_classes)

        # INT8 출력 역양자화
        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            logits = (logits.astype(np.float32) - zero_point) * scale

        all_preds.extend(logits.argmax(axis=1))
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
