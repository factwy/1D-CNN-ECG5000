#!/usr/bin/env python3
"""ECG5000 – 학습 스크립트"""

import os
import argparse
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from models import MODEL_REGISTRY, build_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ECG5000 1D-CNN 학습")
    p.add_argument("--model",
                   default="vgg", choices=list(MODEL_REGISTRY),)
    p.add_argument("--data_dir",   default="./data",    help="데이터 저장 경로")
    p.add_argument("--output_dir", default="./outputs", help="모델/플롯 저장 경로")
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--seed",       type=int,   default=42)
    # GPU 가속 옵션
    p.add_argument("--amp",     action=argparse.BooleanOptionalAction, default=True,
                   help="Automatic Mixed Precision (CUDA 전용, 기본 ON)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="torch.compile() 적용 (PyTorch 2.0+, 기본 OFF)")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def download_ecg5000(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "ECG5000_TRAIN.txt")
    test_path  = os.path.join(save_dir, "ECG5000_TEST.txt")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Downloading ECG5000 ...")
        url      = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
        zip_path = os.path.join(save_dir, "ECG5000.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(save_dir)
        for root, _, files in os.walk(save_dir):
            for f in files:
                src = os.path.join(root, f)
                if "_TRAIN" in f and f.endswith(".txt"):
                    os.replace(src, train_path)
                elif "_TEST" in f and f.endswith(".txt"):
                    os.replace(src, test_path)
        os.remove(zip_path)
        print("Download complete.")
    else:
        print("Using cached data.")

    def _load(path):
        arr = pd.read_csv(path, header=None, sep=r"\s+").values.astype(np.float32)
        return arr[:, 1:], arr[:, 0].astype(int)

    X_tr, y_tr = _load(train_path)
    X_te, y_te = _load(test_path)

    le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    return X_tr, le.transform(y_tr), X_te, le.transform(y_te)


class TimeSeriesDataset(Dataset):
    """(N, L) numpy → (N, 1, L) tensor"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Train / Eval ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device,
              optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    amp_ctx = autocast(device_type="cuda", enabled=use_amp)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with amp_ctx:
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_training_curves(history, best_epoch, output_dir, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"tr_{key}"], label="Train")
        ax.plot(history[f"va_{key}"], label="Val", linestyle="--")
        ax.axvline(best_epoch - 1, color="r", linewidth=0.8,
                   linestyle=":", label=f"Best (e{best_epoch})")
        ax.set_title(f"{model_name} – {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def save_confusion_matrix(all_labels, all_preds, num_classes, output_dir, model_name):
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cls_lbl = [f"C{c}" for c in range(num_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cls_lbl, yticklabels=cls_lbl,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title(f"{model_name} – Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=cls_lbl, yticklabels=cls_lbl,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"{model_name} – Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    out = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    # RTX 4070 SUPER (Ada Lovelace) GPU 최적화
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 활성화

    use_amp = args.amp and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None

    print(f"PyTorch version : {torch.__version__}")
    print(f"Using device    : {device}")
    print(f"Model           : {args.model}")
    print(f"AMP             : {use_amp}")
    print(f"torch.compile   : {args.compile}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = download_ecg5000(args.data_dir)
    NUM_CLASSES = len(np.unique(y_train))

    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print(f"Classes : {NUM_CLASSES}")
    print(f"Label dist (train): { {c: int((y_train==c).sum()) for c in range(NUM_CLASSES)} }")

    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=NUM_CLASSES, dropout=args.dropout)
    model = model.to(device)

    if args.compile and device.type == "cuda":
        model = torch.compile(model)
        print("torch.compile applied.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}
    best_val_acc, best_epoch = 0.0, 0
    best_model_path = os.path.join(args.output_dir, f"{args.model}_best.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
        )
        va_loss, va_acc = run_epoch(
            model, test_loader, criterion, device, use_amp=use_amp,
        )
        scheduler.step()

        history["tr_loss"].append(tr_loss)
        history["va_loss"].append(va_loss)
        history["tr_acc"].append(tr_acc)
        history["va_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            # torch.compile 적용 시 _orig_mod 언랩
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state, best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val   loss={va_loss:.4f} acc={va_acc:.4f}")

    print(f"\nBest Val Accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    save_training_curves(history, best_epoch, args.output_dir, args.model)

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_model = build_model(args.model, num_classes=NUM_CLASSES, dropout=args.dropout)
    eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
    eval_model = eval_model.to(device)
    eval_model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = eval_model(X_batch).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nTest Accuracy : {(all_preds == all_labels).mean():.4f}")
    print("─" * 50)
    print(classification_report(
        all_labels, all_preds,
        target_names=[f"Class {c}" for c in range(NUM_CLASSES)],
    ))

    save_confusion_matrix(all_labels, all_preds, NUM_CLASSES,
                          args.output_dir, args.model)
    print(f"\nModel saved → {best_model_path}")


if __name__ == "__main__":
    main()
