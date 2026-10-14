"""Same 3-layer MLP as `mlp_basic`, but fc1+relu+fc2+relu run as ONE Triton kernel
(`torch.ops.triton_exp.fused_mlp2`, see kernels.py); fc3 stays plain PyTorch.

Same seed, weights and input as `mlp_basic`. The model stays on CPU: capture never executes the
kernel (torch.export traces it with fake tensors), and m2m emits it as a black-box placeholder
call (`m2m/ir/kernel_registry.py`). The real kernel IR is swapped in later in the stack, for
whatever backend is targeted -- Triton here is only how the kernel is authored."""
from __future__ import annotations
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels  # noqa: E402,F401  (registers torch.ops.triton_exp.*)

IN, HID, OUT, BATCH = 16, 32, 8, 4


class TritonMLP3(nn.Module):
    def __init__(self):
        super().__init__()
        # Same submodule layout/creation order as mlp_basic.MLP3 so seeded init is identical.
        self.fc1, self.act1 = nn.Linear(IN, HID), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(HID, HID), nn.ReLU()
        self.fc3 = nn.Linear(HID, OUT)

    def forward(self, x):
        h = torch.ops.triton_exp.fused_mlp2(x, self.fc1.weight, self.fc1.bias,
                                            self.fc2.weight, self.fc2.bias)
        return self.fc3(h)


def get_model_and_inputs():
    torch.manual_seed(0)
    m = TritonMLP3().eval()
    x = torch.randn(BATCH, IN)
    return m, (x,)
