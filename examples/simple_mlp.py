"""Smallest end-to-end example: a 2-layer MLP -> linalg-on-tensors MLIR.

    model2mlir convert examples/simple_mlp.py --out /tmp/mlp.mlir
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, d_in: int = 128, d_hidden: int = 256, d_out: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    model = SimpleMLP().eval()
    example = torch.randn(4, 128)
    return model, (example,)
