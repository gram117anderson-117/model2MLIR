"""Plain-PyTorch 3-layer MLP (fc1 -> relu -> fc2 -> relu -> fc3). No Triton.

Reference baseline for `mlp_basic_triton`: same architecture, same seed, same weights."""
from __future__ import annotations
import torch
from torch import nn

IN, HID, OUT, BATCH = 16, 32, 8, 4


class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act1 = nn.Linear(IN, HID), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(HID, HID), nn.ReLU()
        self.fc3 = nn.Linear(HID, OUT)

    def forward(self, x):
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


def get_model_and_inputs():
    torch.manual_seed(0)
    m = MLP3().eval()
    x = torch.randn(BATCH, IN)
    return m, (x,)
