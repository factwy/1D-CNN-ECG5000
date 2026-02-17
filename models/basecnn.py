import torch.nn as nn


class Conv1dBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → Dropout → (MaxPool)"""
    def __init__(self, in_ch, out_ch, kernel, pool=True, dropout=0.3):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BaseCNN(nn.Module):
    """
    Input  : (B, 1, L)
    Block1 : Conv(64,  k=7) → BN → ReLU → Dropout → MaxPool
    Block2 : Conv(128, k=5) → BN → ReLU → Dropout → MaxPool
    Block3 : Conv(256, k=3) → BN → ReLU → Dropout
    GAP    : AdaptiveAvgPool → (B, 256, 1)
    Head   : FC(128) → ReLU → Dropout → FC(num_classes)
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            Conv1dBlock(1,   64,  kernel=7, pool=True,  dropout=dropout),
            Conv1dBlock(64,  128, kernel=5, pool=True,  dropout=dropout),
            Conv1dBlock(128, 256, kernel=3, pool=False, dropout=dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)           # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)   # (B, 256, L')
        x = self.gap(x)        # (B, 256, 1)
        return self.classifier(x)