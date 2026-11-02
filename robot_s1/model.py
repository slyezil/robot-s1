"""Small fixed-schema multi-head decision model.

One shared MLP trunk reads the encoded state; each question gets its own
linear head. A single forward pass answers all questions, in well under a
millisecond, compared with one 421M-parameter sequence per question for Laya.
"""

import torch
from torch import nn

from .schema import HEADS, N_FEATURES


class DecisionNet(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 3):
        super().__init__()
        blocks, d = [], N_FEATURES
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*blocks)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, len(v)) for k, v in HEADS.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns logits per head."""
        z = self.trunk(x)
        return {k: h(z) for k, h in self.heads.items()}


class ProbsWrapper(nn.Module):
    """Export view: features -> (behavior_probs, hazard_probs, needs_human_probs)."""

    def __init__(self, net: DecisionNet):
        super().__init__()
        self.net = net

    def forward(self, features: torch.Tensor):
        out = self.net(features)
        return tuple(out[k].softmax(-1) for k in HEADS)
