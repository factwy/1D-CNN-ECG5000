import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TemporalModule(nn.Module):
  """
  Input       : (B, in_ch, L)
  Layer 1     : Dilated Conv(out_ch, k=3, dilation=d, padding=calc) → WeightNorm → ReLU → Dropout
  Layer 2     : Dilated Conv(out_ch, k=3, dilation=d, padding=calc) → WeightNorm → ReLU → Dropout
  Residual    : 1x1 Conv(in_ch to out_ch) (채널 수가 다를 경우만 적용)
  Merge       : (Layer 2 Output + Residual) → ReLU
  """
  def __init__(self, in_ch, out_ch, d, dropout):
    super().__init__()
    self.dconv1 = weight_norm(nn.Conv1d(in_ch,  out_ch, kernel_size=3, dilation=d, padding=d))
    self.dconv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size=3, dilation=d, padding=d))
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(dropout)

    self.layer1 = nn.Sequential(self.dconv1, self.relu, self.drop)
    self.layer2 = nn.Sequential(self.dconv2, self.relu, self.drop)

    self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

  def forward(self, x):
    res = self.shortcut(x)
    x = self.layer1(x)
    x = self.layer2(x)
    return self.relu(x + res)


class TCN(nn.Module):
  """
  Based on Temporal Convolutional Network
  Block 1     : Temporal Block (d=1, C=64) → (B, 64, L)
  Block 2     : Temporal Block (d=2, C=64) → (B, 64, L)
  Block 3     : Temporal Block (d=4, C=64) → (B, 64, L)
  Block 4     : Temporal Block (d=8, C=64) → (B, 64, L)
  GAP         : AdaptiveAvgPool1d(1) → (B, 64, 1)
  Head        : Flatten → FC(64) → ReLU → Dropout → FC(num_classes)
  """
  def __init__(self, num_classes: int, dropout: float = 0.3):
    super().__init__()
    self.block1 = TemporalModule(1,  64, 1, dropout)
    self.block2 = TemporalModule(64, 64, 2, dropout)
    self.block3 = TemporalModule(64, 64, 4, dropout)
    self.block4 = TemporalModule(64, 64, 8, dropout)
    self.gap = nn.AdaptiveAvgPool1d(1)           # Global Average Pooling
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, num_classes),
    )

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.gap(x)
    return self.classifier(x)
