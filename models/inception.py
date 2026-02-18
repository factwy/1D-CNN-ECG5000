import torch
import torch.nn as nn


class InceptionModule(nn.Module):
  """
  Input       : (B, 1, L)
  Bottleneck  : Conv(32, 1)
  Parallel Branches
    Branch 1  : Conv(32, 10)
    Branch 2  : Conv(32, 20)
    Branch 3  : Conv(32, 40)
  Pool Branch : MaxPool(3, 1) → Conv(32, 1)
  Merge       : Concat(Branch 1, 2, 3, Pool) → BN -> ReLU
  """
  def __init__(self, in_ch):
    super().__init__()
    self.bottleneck = nn.Sequential(nn.Conv1d(in_ch,  32, kernel_size=1, bias=False))
    self.branch1 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=9, padding=9//2, bias=False))
    self.branch2 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=19, padding=19//2, bias=False))
    self.branch3 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=39, padding=39//2, bias=False))
    self.pool_branch = nn.Sequential(
        nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        nn.Conv1d(in_ch, 32, kernel_size=1, bias=False)
    )
    self.bh = nn.BatchNorm1d(128)
    self.relu = nn.ReLU()
  def forward(self, x):
    out4 = self.pool_branch(x)
    x = self.bottleneck(x)
    out1 = self.branch1(x)
    out2 = self.branch2(x)
    out3 = self.branch3(x)
    outs = torch.cat([out1, out2, out3, out4], dim=1)

    return self.relu(self.bh(outs))


class Inception(nn.Module):
  """
  Based on InceptionTime
  Block1 : Inception Module → (B, 128, L)
  Block2 : Inception Module → (B, 128, L)
  Block3 : Inception Module → Residual(Block1 Input) → ReLU → (B, 128, L)
  Block4 : Inception Module → (B, 128, L)
  Block5 : Inception Module → (B, 128, L)
  Block6 : Inception Module → Residual(Block4 Input) → ReLU → (B, 128, L)
  GAP    : AdaptiveAvgPool → (B, 128, 1)
  Head   : FC(128) → ReLU → Dropout → FC(num_classes)
  """
  def __init__(self, num_classes: int, dropout: float = 0.3):
    super().__init__()
    self.block1 = InceptionModule(1)
    self.block2 = InceptionModule(128)
    self.block3 = InceptionModule(128)
    self.block4 = InceptionModule(128)
    self.block5 = InceptionModule(128)
    self.block6 = InceptionModule(128)
    self.shortcut = nn.Sequential(
        nn.Conv1d(1, 128, kernel_size=1, bias=False),
        nn.BatchNorm1d(128)
    )
    self.bh = nn.BatchNorm1d(128)
    self.relu = nn.ReLU()
    self.gap = nn.AdaptiveAvgPool1d(1)           # Global Average Pooling
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes),
    )

  def forward(self, x):
    res1 = self.shortcut(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x + res1
    x = self.relu(x)

    res2 = self.bh(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = x + res2
    x = self.relu(x)

    x = self.gap(x)
    return self.classifier(x)