#!/usr/bin/env python3
"""ONNX 모델 → TFLite 변환 스크립트

변환 파이프라인:
  ONNX  ──(onnx2tf)──►  TF SavedModel  ──(TFLiteConverter)──►  .tflite
"""

import os
import sys
import shutil
import argparse

import onnx2tf
import tensorflow as tf


def parse_args():
    p = argparse.ArgumentParser(description="ONNX → TFLite 변환")
    p.add_argument("--onnx_path",  required=True,
                   help="입력 .onnx 파일 경로 (export_onnx.py 결과물)")
    p.add_argument("--output_dir", default="/workspace/outputs/tflite",
                   help="TFLite 파일 저장 디렉터리")
    p.add_argument("--quantize",   default="none",
                   choices=["none", "fp16", "int8"],
                   help="양자화 옵션\n"
                        "  none : FP32 그대로 변환\n"
                        "  fp16 : 가중치 FP16 (크기 약 1/2, 정밀도 소폭 감소)\n"
                        "  int8 : 동적 범위 양자화 (크기 약 1/4, 추론 속도 향상)")
    return p.parse_args()


def export(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_name      = os.path.splitext(os.path.basename(args.onnx_path))[0]
    suffix          = f"_{args.quantize}" if args.quantize != "none" else ""
    tflite_path     = os.path.join(args.output_dir, f"{model_name}{suffix}.tflite")
    saved_model_dir = os.path.join(args.output_dir, f"_tmp_{model_name}_savedmodel")

    # ── Step 1 : ONNX → TF SavedModel ────────────────────────────────────────
    print(f"[1/3] ONNX → SavedModel")
    print(f"      input  : {args.onnx_path}")
    print(f"      output : {saved_model_dir}")
    onnx2tf.convert(
        input_onnx_file_path=args.onnx_path,
        output_folder_path=saved_model_dir,
        non_verbose=True,
    )

    # ── Step 2 : SavedModel → TFLite ─────────────────────────────────────────
    print(f"[2/3] SavedModel → TFLite  (quantize={args.quantize})")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if args.quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif args.quantize == "int8":
        # representative dataset 없이 가중치만 INT8로 변환하는
        # dynamic range quantization (보정 데이터 불필요)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # ── Step 3 : 임시 SavedModel 제거 & 요약 ─────────────────────────────────
    print(f"[3/3] 임시 SavedModel 제거")
    shutil.rmtree(saved_model_dir)

    size_mb = os.path.getsize(tflite_path) / 1024 ** 2
    print("-" * 50)
    print(f"Model     : {model_name}")
    print(f"Quantize  : {args.quantize}")
    print(f"File size : {size_mb:.2f} MB")
    print(f"Saved     : {tflite_path}")


def main():
    args = parse_args()
    export(args)


if __name__ == "__main__":
    main()
