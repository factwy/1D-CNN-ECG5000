import torch.nn as nn


class VGG1dBlockl1(nn.Module):
    """Conv1d → BatchNorm → ReLU → Dropout → (MaxPool)"""
    def __init__(self, in_ch, out_ch, pool=True, dropout=0.3):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG1dBlockl2(nn.Module):
    """Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → ReLU → Dropout → (MaxPool)"""
    def __init__(self, in_ch, out_ch, pool=True, dropout=0.3):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    """
    Based on VGG11
    Input  : (B, 1, L)
    Block1 : Conv(64, k=3) → BN → ReLU → Dropout → MaxPool
    Block2 : Conv(128, k=3) → BN → ReLU → Dropout → MaxPool
    Block3 : Conv(256, k=3) → BN → ReLU → Conv(256, k=3) → BN → ReLU → Dropout → MaxPool
    Block4 : Conv(512, k=3) → BN → ReLU → Conv(512, k=3) → BN → ReLU → Dropout → MaxPool
    Block5 : Conv(512, k=3) → BN → ReLU → Conv(512, k=3) → BN → ReLU → Dropout → MaxPool
    GAP    : AdaptiveAvgPool → (B, 512, 1)
    Head   : FC(512) → ReLU → FC(512) → ReLU → Dropout → FC(num_classes)
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            VGG1dBlockl1(1,   64,  dropout=dropout),
            VGG1dBlockl1(64,  128, dropout=dropout),
            VGG1dBlockl2(128, 256, dropout=dropout),
            VGG1dBlockl2(256, 512, dropout=dropout),
            VGG1dBlockl2(512, 512, dropout=dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)           # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)   # (B, 512, L')
        x = self.gap(x)        # (B, 512, 1)
        return self.classifier(x)