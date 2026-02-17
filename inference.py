#!/usr/bin/env python3
"""ECG5000 – 추론 스크립트"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from models import MODEL_REGISTRY, build_model
from train import TimeSeriesDataset


def parse_args():
    p = argparse.ArgumentParser(description="ECG5000 1D-CNN 추론")
    p.add_argument("--model",
                   required=True, choices=list(MODEL_REGISTRY),
                   help="학습 시 사용한 모델 이름")
    p.add_argument("--model_path",  required=True, help="학습된 .pth 파일 경로")
    p.add_argument("--data_path",   required=True, help="입력 데이터 .txt 파일 경로")
    p.add_argument("--num_classes", type=int,   default=5)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--batch_size",  type=int,   default=64)
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device : {device}")
    print(f"Model        : {args.model}")

    # ── Data ──────────────────────────────────────────────────────────────────
    arr = pd.read_csv(args.data_path, header=None, sep=r"\s+").values.astype(np.float32)
    X   = arr[:, 1:]
    y   = arr[:, 0].astype(int)

    # 레이블 0-based 정규화
    unique    = np.unique(y)
    label_map = {v: i for i, v in enumerate(sorted(unique))}
    y = np.array([label_map[v] for v in y])

    loader = DataLoader(
        TimeSeriesDataset(X, y),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=args.num_classes,
                        dropout=args.dropout)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded weights from {args.model_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nAccuracy : {(all_preds == all_labels).mean():.4f}")
    print("─" * 50)
    print(classification_report(
        all_labels, all_preds,
        target_names=[f"Class {c}" for c in range(args.num_classes)],
    ))


if __name__ == "__main__":
    main()
