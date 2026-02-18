#!/usr/bin/env python3
"""모델 아키텍처 요약 출력"""

import argparse
import torch
from models import MODEL_REGISTRY, build_model


def parse_args():
    p = argparse.ArgumentParser(description="모델 아키텍처 요약")
    p.add_argument("--model",       required=True, choices=list(MODEL_REGISTRY),
                   help="요약할 모델 이름")
    p.add_argument("--num_classes", type=int,   default=5)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--seq_len",     type=int,   default=140,
                   help="입력 시계열 길이 (ECG5000 기본값: 140)")
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="torchinfo summary용 배치 크기")
    return p.parse_args()


def print_model_summary(model, input_size, device="cpu"):
    try:
        from torchinfo import summary
        print("=" * 80)
        print("모델 아키텍처 상세 정보 (torchinfo)")
        print("=" * 80)
        summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"],
            verbose=1,
            device=device,
        )
    except ImportError:
        print("torchinfo가 설치되지 않았습니다.")
        print("설치: pip install torchinfo")
        print("\n기본 모델 정보:")
        print(model)


def print_layer_parameters(model):
    print("\n" + "=" * 80)
    print("레이어별 파라미터 수")
    print("=" * 80)
    print(f"{'Layer Name':<50} {'Parameters':>15} {'Trainable':>12}")
    print("-" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        trainable_str = "✓" if param.requires_grad else "✗"
        print(f"{name:<50} {num_params:>15,} {trainable_str:>12}")

    print("-" * 80)
    print(f"{'Total Parameters':<50} {total_params:>15,}")
    print(f"{'Trainable Parameters':<50} {trainable_params:>15,}")
    print(f"{'Non-trainable Parameters':<50} {total_params - trainable_params:>15,}")

    param_size  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    print(f"\nModel size: {total_size_mb:.2f} MB")
    print(f"  - Parameters: {param_size  / (1024**2):.2f} MB")
    print(f"  - Buffers:    {buffer_size / (1024**2):.2f} MB")

    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "size_mb":          total_size_mb,
    }


def main():
    args = parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    model = build_model(args.model, num_classes=args.num_classes,
                        dropout=args.dropout).to(device)
    model.eval()

    print(f"\nModel      : {args.model}")
    print(f"Device     : {device}")
    print(f"Seq len    : {args.seq_len}")
    print(f"Classes    : {args.num_classes}")

    input_size = (args.batch_size, 1, args.seq_len)
    print_model_summary(model, input_size, device=str(device))
    print_layer_parameters(model)


if __name__ == "__main__":
    main()
