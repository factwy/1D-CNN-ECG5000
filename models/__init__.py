import torch.nn as nn

from .basecnn import BaseCNN
from .vgg import VGG
from .inception import Inception
from .tcn import TCN

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "base-cnn":    BaseCNN,
    "vgg":         VGG,
    "inception":   Inception,
    "tcn":         TCN,
}


def build_model(name: str, num_classes: int, dropout: float = 0.3) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, dropout=dropout)
