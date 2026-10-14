"""ResNet-18 classifier/embedding for federated FR (unchanged backbone)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceNet(nn.Module):
    def __init__(self, n_classes: int, embed_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))
