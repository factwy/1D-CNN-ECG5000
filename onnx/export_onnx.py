#!/usr/bin/env python3
"""PyTorch 모델 → ONNX 변환 및 검증"""

import sys
import os
# onnx/ 폴더 안에서 실행되므로 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from models import MODEL_REGISTRY, build_model


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch 모델 → ONNX 변환")
    p.add_argument("--model",       required=True, choices=list(MODEL_REGISTRY),
                   help="변환할 모델 이름")
    p.add_argument("--model_path",  required=True,
                   help="학습된 .pth 파일 경로")
    p.add_argument("--output_dir",  default="/workspace/outputs/onnx",
                   help="ONNX 파일 저장 디렉터리")
    p.add_argument("--seq_len",     type=int, default=140,
                   help="입력 시계열 길이 (ECG5000 기본값: 140)")
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--opset",       type=int, default=17,
                   help="ONNX opset 버전 (기본: 17)")
    return p.parse_args()


def export(args):
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, f"{args.model}.onnx")

    # ── 모델 로드 (export는 항상 CPU에서) ────────────────────────────────────
    model = build_model(args.model, num_classes=args.num_classes,
                        dropout=args.dropout)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, args.seq_len)  # (batch=1, channel=1, seq_len)

    # ── ONNX export ───────────────────────────────────────────────────────────
    print(f"Exporting '{args.model}' → {onnx_path} (opset {args.opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"Exported  → {onnx_path}")

    # ── ONNX 구조 유효성 검사 ─────────────────────────────────────────────────
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX graph check   : PASSED")

    # ── OnnxRuntime 수치 검증 (PyTorch 출력 vs ORT 출력) ─────────────────────
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    ort_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    passed   = np.allclose(pt_out, ort_out, atol=1e-4)
    status   = "PASSED" if passed else "WARNING"
    print(f"Numerical check    : {status}  (max diff = {max_diff:.2e}, atol=1e-4)")

    # ── 요약 ─────────────────────────────────────────────────────────────────
    size_mb = os.path.getsize(onnx_path) / 1024 ** 2
    print("-" * 50)
    print(f"Model         : {args.model}")
    print(f"Opset         : {args.opset}")
    print(f"Input shape   : (batch, 1, {args.seq_len})")
    print(f"Output shape  : (batch, {args.num_classes})")
    print(f"File size     : {size_mb:.2f} MB")
    print(f"Saved         : {onnx_path}")


def main():
    args = parse_args()
    export(args)


if __name__ == "__main__":
    main()
