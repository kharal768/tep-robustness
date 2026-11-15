"""
model.py

Fault-diagnosis classifier f_theta: R^(D-1) -> {0, ..., K}
operating on ILR-transformed compositional observations.
"""

import torch
import torch.nn as nn


class FaultClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 64,
                 depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
